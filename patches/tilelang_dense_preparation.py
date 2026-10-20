"""Dense-only preparation that preserves the existing INT8 numerical contract.

Nsight Systems on a real H3 TE head showed 76 cudaLaunchKernel calls/request
and a 0.074 ms attention kernel. Avoid unused routing, pooled V, V8 and LSE
preparation. This internal path generates its own all-dense routes, so it can
omit repeated GPU-to-CPU route validation without accepting arbitrary routes.
"""
from functools import lru_cache
import math
import torch
import torch.nn.functional as F
from patches.tilelang_sparse_attention import Prepared, LEAF, _quant, get_kernel


@lru_cache(maxsize=32)
def geometry(b,h,n,d,device):
    nb=(n+LEAF-1)//LEAF
    counts=(n-torch.arange(nb,device=device)*LEAF).clamp(0,LEAF).int()
    routes=torch.ones((b,h,nb,nb),device=device,dtype=torch.int32)
    # Dense kernels do not read these optional pooled/PV8 parameters.
    kc=torch.empty((b,h,nb,d),device=device,dtype=torch.float16)
    vc=torch.empty_like(kc)
    vi=torch.empty((b,h,nb*LEAF,d),device=device,dtype=torch.int8)
    vs=torch.empty((b,h,nb,d),device=device,dtype=torch.float32)
    return counts,routes,kc,vc,vi,vs


def prepare_dense(q,k,v,int8_pv=False,want_lse=False,enable_gqa=False):
    if q.dtype!=torch.float16 or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise ValueError('FP16 inputs required')
    if q.ndim==4 and k.ndim==4 and q.shape[1]!=k.shape[1] and enable_gqa:
        if q.shape[1]%k.shape[1]:raise ValueError('Divisible GQA head counts required')
        k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1);v=v.repeat_interleave(q.shape[1]//v.shape[1],dim=1)
    if q.ndim!=4 or q.shape!=k.shape or k.shape!=v.shape or q.shape[-1] not in (64,128) or q.shape[-2]<1:
        raise ValueError('Matching BHND tensors with D64/128 required; expand GQA before calling')
    if q.device!=k.device or q.device!=v.device:raise ValueError('Shared device required')
    b,h,n,d=q.shape;nb=(n+LEAF-1)//LEAF;pad=nb*LEAF-n
    counts,routes,kc,vc,vi,vs=geometry(b,h,n,d,str(q.device))
    def padded(x):return F.pad(x,(0,0,0,pad)).contiguous()
    qf,kf=q.float(),k.float()
    raw_kc=padded(kf).reshape(b,h,nb,LEAF,d).sum(-2)/counts[None,None,:,None]
    mean=raw_kc.mean(-2,keepdim=True)
    qi,qs=_quant(qf);ki,ks=_quant(kf-mean)
    if int8_pv:
        vp=padded(v.float()).reshape(b,h,nb,LEAF,d)
        vs=(vp.abs().amax(-2)/127).clamp_min(1e-8)
        vi=(vp/vs[...,None,:]).round().clamp(-127,127).to(torch.int8).reshape(b,h,nb*LEAF,d).contiguous()
    shift=(qf*mean).sum(-1)/math.sqrt(d) if want_lse else None
    p=Prepared(padded(q),padded(v),padded(qi),padded(ki),F.pad(qs,(0,pad)).contiguous(),
               F.pad(ks,(0,pad)).contiguous(),kc,vc,counts,vi,vs,shift,n)
    return p,routes


def run_prevalidated_dense(p,routes,int8_pv=False,causal=False,want_lse=False):
    if not p.q.is_cuda or torch.cuda.get_device_capability(p.q.device)!=(7,5):
        raise ValueError('This experiment requires an actual SM75 GPU')
    b,h,_,d=p.q.shape
    expected=geometry(b,h,p.tokens,d,str(p.q.device))[1]
    if routes is not expected:
        raise ValueError('Only internally generated immutable all-dense routes supported')
    kernel=get_kernel(b,h,p.tokens,d,False,int8_pv,sm=75,causal=causal)
    out,lse=kernel(p.q,p.qi,p.ki,p.v,p.qs,p.ks,p.kc,p.vc,p.counts,routes,p.vi,p.vs)
    if want_lse:
        if p.shift is None:raise ValueError('LSE shift must be prepared')
        return out[:,:,:p.tokens],lse[:,:,:p.tokens]+p.shift
    return out[:,:,:p.tokens]


def dense_attention(q,k,v,int8_pv=False,causal=False,enable_gqa=False):
    p,routes=prepare_dense(q,k,v,int8_pv=int8_pv,enable_gqa=enable_gqa)
    return run_prevalidated_dense(p,routes,int8_pv=int8_pv,causal=causal)
