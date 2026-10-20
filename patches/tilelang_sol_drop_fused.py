"""SM75 Sol dropping with fused per-token INT8 preparation.

Preserve legacy Sol centering, nearest-even quantization and routes. Omit V8,
value centroids and LSE correction, which dropping-only callers do not consume.
Routes are constructed internally with mandatory prefix/diagonal leaves, so
nonempty rows are guaranteed without a host bool on a device reduction.
"""
import math
import torch
import torch.nn.functional as F
from patches.tilelang_grouped_fused import quantizer
from patches.tilelang_sparse_attention import Prepared,make_routes,get_kernel


def prepare_drop_fused(q,k,v):
    if q.ndim!=4 or q.shape!=k.shape or q.shape!=v.shape or q.shape[-1] not in (64,128) or q.shape[2]<1:
        raise ValueError('Matching nonempty BHND D64/128 required')
    if q.dtype!=torch.float16 or q.dtype!=k.dtype or q.dtype!=v.dtype:
        raise ValueError('FP16 operands required')
    if not q.is_cuda or q.device!=k.device or q.device!=v.device or torch.cuda.get_device_capability(q.device)!=(7,5):
        raise ValueError('Physical SM75 shared device required')
    b,h,n,d=q.shape;nb=math.ceil(n/64);pad=nb*64-n
    counts=(n-torch.arange(nb,device=q.device)*64).clamp(0,64).int()
    raw_kc=F.pad(k.float(),(0,0,0,pad)).reshape(b,h,nb,64,d).sum(-2)/counts[None,None,:,None]
    mean=raw_kc.mean(-2,keepdim=True)
    qi,qs=quantizer(b,h,n,d,d,False)(q.contiguous(),mean)
    ki,ks=quantizer(b,h,n,d,d,True)(k.contiguous(),mean)
    kc=(raw_kc-mean).half().contiguous()
    # Shape-compatible placeholders are never read by the dropping/FP16-PV kernel.
    # Reuse existing allocations rather than compute unused values.
    return Prepared(F.pad(q,(0,0,0,pad)).contiguous(),F.pad(v,(0,0,0,pad)).contiguous(),
        qi,ki,qs.squeeze(-1),ks.squeeze(-1),kc,kc,counts,qi,
        torch.empty((b,h,nb,d),device=q.device,dtype=torch.float32),None,n)


def bound_sol_routes(p,prefix_tokens,tau=0.):
    if not isinstance(prefix_tokens,int) or not 0<prefix_tokens<p.tokens:
        raise ValueError('A nonempty packed prefix and video suffix are required')
    route=make_routes(p,'sol',tau=tau)
    blocks=math.ceil(prefix_tokens/64)
    route[...,:blocks]=1;route[...,:blocks,:]=1
    return route


def _execute_bound_drop(p,routes):
    """Internal: routes must come from bound_sol_routes, never arbitrary masks."""
    b,h,_,d=p.q.shape
    return get_kernel(b,h,p.tokens,d,False,False,sm=75,causal=False)(
        p.q,p.qi,p.ki,p.v,p.qs,p.ks,p.kc,p.vc,p.counts,routes,p.vi,p.vs)[0][:,:,:p.tokens]


def sol_drop_attention(q,k,v,prefix_tokens,tau=0.):
    p=prepare_drop_fused(q,k,v)
    return _execute_bound_drop(p,bound_sol_routes(p,prefix_tokens,tau))
