"""Opt-in Kitchen 0.2.37 CUDA experiments for SM75, without TileLang.

No compilation on import, global patch, SDPA fallback, or production dispatch.
``is_causal`` masks scores at key <= query + query_offset. Kitchen's existing
quantizer uses shared scales/anchor selection; it is not prefix-invariant under
changes to future floating-point inputs. This is not a streaming KV-cache API.
"""
import ctypes as C
from dataclasses import dataclass
from functools import lru_cache
import importlib.metadata
import math
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BlockPlan:
    """Sorted leaf64 IDs, counts and their dense block mask (not a token mask)."""
    ids: torch.Tensor
    counts: torch.Tensor
    routes: torch.Tensor
    queries: int
    keys: int


def routing(q, k, prefix_tokens=0):
    """Port of the existing noncausal tau=0/local/global/prefix route policy.

    FP16 input/centroid boundaries and ragged-leaf counts match the old policy.
    This function uses PyTorch tensor operations, never the TileLang compiler.
    """
    if q.ndim != 4 or k.shape != q.shape or q.shape[-1] not in (64,128):
        raise ValueError('Content routing requires matching BHND self-attention, D64/128')
    if q.dtype not in (torch.float16,torch.float32) or k.dtype != q.dtype or k.device != q.device:
        raise ValueError('Matching FP16/FP32 Q/K on one device required')
    n=q.shape[2]
    if not n or not 0 <= prefix_tokens <= n:
        raise ValueError('Invalid sequence/prefix length')
    b,h,_,d=q.shape; nb=(n+63)//64; pad=nb*64-n
    counts=(n-torch.arange(nb,device=q.device)*64).clamp(0,64).int()
    raw=F.pad(k.half().float(),(0,0,0,pad)).reshape(b,h,nb,64,d).sum(-2)/counts[None,None,:,None]
    kc=(raw-raw.mean(-2,keepdim=True)).half().float()
    qc=F.pad(q.half().float(),(0,0,0,pad)).reshape(b,h,nb,64,d).sum(-2)/counts[None,None,:,None]
    scores=qc@kc.transpose(-1,-2)/math.sqrt(d)
    threshold=(qc*kc.mean(-2)[...,None,:]).sum(-1)/math.sqrt(d)
    ids=torch.arange(nb,device=q.device)
    mandatory=((ids[:,None]-ids[None,:]).abs()<=1)|(ids[None,:]<1)
    result=((scores>threshold[...,None])|mandatory[None,None]).int().contiguous()
    prefix=(prefix_tokens+63)//64
    result[:,:,:prefix,:]=1; result[:,:,:,:prefix]=1
    return result


def check_routes(routes, queries, keys):
    if routes.ndim != 4 or routes.shape[2:] != ((queries+63)//64,(keys+63)//64):
        raise ValueError('Routes must be [B,H,ceil(Q/64),ceil(K/64)]')
    if queries<1 or keys<1 or routes.dtype not in (torch.bool,torch.int32):
        raise ValueError('Nonempty lengths and bool/int32 routes required')


def compact_reference(routes, queries, keys):
    """Independent CPU planner/reference. GPU execution uses a warp ballot kernel."""
    check_routes(routes,queries,keys)
    if routes.device.type != 'cpu':
        raise ValueError('CPU planner only; use Backend.plan on CUDA')
    nb=routes.shape[-1]
    mask=(routes!=0).int().contiguous()
    leaves=torch.arange(nb,dtype=torch.int32).expand_as(mask)
    ids=torch.where(mask.bool(),leaves,nb).sort(-1).values.contiguous()
    return BlockPlan(ids,mask.sum(-1,dtype=torch.int32),mask,queries,keys)


def token_mask(plan, *, is_causal=False, query_offset=0):
    """Small-fixture reference only. Native execution never expands this mask."""
    if query_offset<0:
        raise ValueError('query_offset must be nonnegative')
    mask=plan.routes.bool().repeat_interleave(64,2).repeat_interleave(64,3)
    mask=mask[:,:,:plan.queries,:plan.keys]
    if is_causal:
        device=mask.device
        mask=mask & (torch.arange(plan.keys,device=device)[None,:] <=
            torch.arange(plan.queries,device=device)[:,None]+query_offset)
    return mask


def pooled_reference(q, k, v, plan, scale=None):
    """FP32 mathematical oracle on ALREADY dequantized Kitchen operands.

    Retained tokens plus count-weighted mean K/V for each omitted leaf.
    It intentionally does not reproduce P quantization, FP16 coarse rounding,
    Tensor Core reductions, or final output rounding.
    """
    b,h,n,d=q.shape; nk=k.shape[2]; hk=k.shape[1]
    if plan.queries!=n or plan.keys!=nk or h%hk or v.shape!=k.shape:
        raise ValueError('Inconsistent pooled reference geometry')
    ids=torch.arange(h,device=q.device)//(h//hk)
    k=k[:,ids].float(); v=v[:,ids].float(); q=q.float()
    factor=d**-.5 if scale is None else scale
    nb=(nk+63)//64
    counts=(nk-torch.arange(nb,device=q.device)*64).clamp(0,64)
    kc=F.pad(k,(0,0,0,nb*64-nk)).reshape(b,h,nb,64,d).sum(-2)/counts[None,None,:,None]
    vc=F.pad(v,(0,0,0,nb*64-nk)).reshape(b,h,nb,64,d).sum(-2)/counts[None,None,:,None]
    dense=q@k.transpose(-1,-2)*factor
    dense=dense.masked_fill(~token_mask(plan),float('-inf'))
    coarse=q@kc.transpose(-1,-2)*factor+counts.float().log()
    selected=plan.routes.bool().repeat_interleave(64,2)[:,:,:n]
    coarse=coarse.masked_fill(selected,float('-inf'))
    logits=torch.cat((dense,coarse),-1)
    return logits.softmax(-1)@torch.cat((v,vc),-2),logits.logsumexp(-1)


def value_storage_indices(tokens,device='cpu'):
    """Pinned producer's logical-token -> physical V index permutation."""
    index=torch.arange(tokens,device=device)
    return ((index & ~15) | (index & 1) | (((index>>3)&1)<<1) |
            (((index>>1)&1)<<2) | (((index>>2)&1)<<3))


def dequantized_operands(p):
    """Small-fixture oracle for pinned per-thread scale packing, D64/128 only."""
    b,h,n,d=p.q.shape; hk=p.k.shape[1]; nk=p.k.shape[2]; ck=p.cta_k
    if d not in (64,128) or ck not in (64,128):
        raise ValueError('Unsupported Kitchen packing')
    qs=p.q_scale.reshape(b,h,-1)
    qi=torch.arange(n,device=p.q.device)
    q=(p.q.float()*qs[:,:,qi//32*8+qi%8,None])
    ks=p.k_scale.reshape(b,hk,-1); ki=torch.arange(nk,device=p.q.device)
    k=p.k.float()*ks[:,:,ki//ck*4+(ki%8)//2,None]
    kp=((nk+ck-1)//ck)*ck
    locations=value_storage_indices(nk,p.q.device)
    v=p.v.reshape(b,hk,d,kp).index_select(-1,locations).transpose(-1,-2).float()*p.v_scale.reshape(b,hk,1,d)
    return q,k,v


class AttentionArgs(C.Structure):
    _fields_ = [(name,C.c_void_p) for name in ('q','k','v','o','qs','ks','vs','lse','ids','counts')]+[
        (name,C.c_int) for name in ('b','h','hk','n','nk','dim','cta_k','variant','causal','query_offset','dtype','capacity')]+[
        ('scale',C.c_float),('stream',C.c_void_p)]


class PoolArgs(C.Structure):
    _fields_ = [(name,C.c_void_p) for name in ('q','k','v','qs','ks','vs','qh','kc','vc','routes','out','lse')]+[
        (name,C.c_int) for name in ('b','h','hk','n','nk','dim','dtype')]+[('scale',C.c_float),('stream',C.c_void_p)]


def runtime_contract(require_extension=True):
    if importlib.metadata.version('comfy-kitchen') != '0.2.37':
        raise RuntimeError('Exact comfy-kitchen 0.2.37 required')
    from comfy_kitchen import sage_attention as kitchen
    required=('prequantize_int8_attention','int8_attention_from_prequantized','PrequantizedInt8Attention')
    if not all(hasattr(kitchen,name) for name in required):
        raise RuntimeError('Pinned Kitchen split-attention API missing')
    if require_extension and not hasattr(kitchen._cuda_backend._C,'sage_sdpa_quantize'):
        raise RuntimeError('Pinned Kitchen CUDA quantization entry missing')
    return kitchen


def _sm75(device):
    if torch.device(device).type!='cuda' or torch.cuda.get_device_capability(device)!=(7,5):
        raise ValueError('Physical SM75 required; local SM120 is not a T4 substitute')


@lru_cache(maxsize=4)
def _library(path):
    lib=C.CDLL(str(Path(path).resolve()))
    lib.h3_launch.argtypes=[C.POINTER(AttentionArgs)]; lib.h3_launch.restype=C.c_int
    lib.h3_pool.argtypes=[C.POINTER(PoolArgs)]; lib.h3_pool.restype=C.c_int
    lib.h3_compact.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p]; lib.h3_compact.restype=C.c_int
    lib.h3_error.argtypes=[C.c_int]; lib.h3_error.restype=C.c_char_p
    return lib


class Backend:
    def __init__(self, library):
        self.library=str(Path(library).resolve())
        self.kitchen=runtime_contract()  # CPU import/signature check, no module load.

    @property
    def lib(self):
        return _library(self.library)

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.h3_error(code).decode())

    def plan(self,routes,queries,keys):
        check_routes(routes,queries,keys); _sm75(routes.device)
        mask=(routes!=0).int().contiguous()
        ids=torch.empty_like(mask); counts=torch.empty(mask.shape[:-1],device=mask.device,dtype=torch.int32)
        with torch.cuda.device(mask.device):
            stream=torch.cuda.current_stream(mask.device).cuda_stream
            self._check(self.lib.h3_compact(mask.data_ptr(),ids.data_ptr(),counts.data_ptr(),
                counts.numel(),mask.shape[-1],stream))
        return BlockPlan(ids,counts,mask,queries,keys)

    def quantize(self,q,k,v,*,cta_k=None,scale=None):
        _sm75(q.device)
        if any(x.requires_grad for x in (q,k,v)):
            raise ValueError('Inference only; autograd is not implemented')
        self.kitchen._validate_inputs(q,k,v,None)
        b,h,n,d=q.shape; hk=k.shape[1]; nk=k.shape[2]
        if d not in (64,128) or q.dtype not in (torch.float16,torch.float32,torch.bfloat16):
            raise ValueError('This experiment supports D64/128 only')
        ck=cta_k if cta_k is not None else (128 if d==128 and nk>1024 else 64)
        if ck not in (64,128) or (d==64 and ck!=64):
            raise ValueError('Invalid native key tile')
        factor=d**-.5 if scale is None else float(scale)
        if not math.isfinite(factor) or factor<=0:
            raise ValueError('Finite positive scale required for this experiment')
        # Allocate exactly the pinned Kitchen packing, with an explicit key tile.
        qi=torch.empty(q.shape,device=q.device,dtype=torch.int8)
        ki=torch.empty(k.shape,device=q.device,dtype=torch.int8)
        qs=torch.empty((b,h,(n+127)//128*32),device=q.device,dtype=torch.float32)
        ks=torch.empty((b,hk,(nk+ck-1)//ck*4),device=q.device,dtype=torch.float32)
        vi=torch.empty((b*hk*d,(nk+ck-1)//ck*ck),device=q.device,dtype=torch.int8)
        vs=torch.empty((b*hk*d,),device=q.device,dtype=torch.float32)
        anchors=torch.empty((b,hk),device=q.device,dtype=torch.int32)
        backend=self.kitchen._cuda_backend
        with torch.cuda.device(q.device):
            backend._C.sage_sdpa_quantize(*[backend._wrap_for_dlpack(t) for t in (q,k,v,qi,qs,ki,ks,vi,vs)],
                ck,{torch.float32:0,torch.float16:1,torch.bfloat16:2}[q.dtype],
                torch.cuda.current_stream(q.device).cuda_stream,anchors.data_ptr())
        return self.kitchen.PrequantizedInt8Attention(qi,ki,vi,qs,ks,vs,d,q.dtype,factor,ck,None)

    def execute(self,p,*,plan=None,is_causal=False,query_offset=0,variant='baseline',pooled=False):
        _sm75(p.q.device)
        if query_offset<0 or query_offset+p.q.shape[2]>=2**31:
            raise ValueError('query_offset is nonnegative and must fit the native index range')
        if not is_causal and query_offset:
            raise ValueError('query_offset requires is_causal=True')
        if p.attn_mask is not None:
            raise ValueError('Use explicit block routes/causality; arbitrary masks are not supported here')
        if pooled and (plan is None or is_causal):
            raise ValueError('Pooled mode needs a noncausal block plan; partial causal centroids are unsupported')
        b,h,n,d=p.q.shape; hk=p.k.shape[1]; nk=p.k.shape[2]
        if d not in (64,128):
            raise ValueError('D64/128 required')
        packed=(p.q,p.k,p.v,p.q_scale,p.k_scale,p.v_scale)
        if any(t.device!=p.q.device or not t.is_contiguous() for t in packed):
            raise ValueError('Native packed tensors must be contiguous on one device')
        if min(b,h,hk,n,nk)<1 or h%hk or p.k.shape!=(b,hk,nk,d):
            raise ValueError('Invalid packed geometry')
        ck=p.cta_k
        if ck not in (64,128) or (d==64 and ck!=64):
            raise ValueError('Invalid key tile')
        expected=((b,h,(n+127)//128*32),(b,hk,(nk+ck-1)//ck*4))
        if p.q_scale.shape!=expected[0] or p.k_scale.shape!=expected[1]:
            raise ValueError('Invalid packed scale geometry')
        if any(t.dtype!=torch.int8 for t in (p.q,p.k,p.v)) or any(
                t.dtype!=torch.float32 for t in (p.q_scale,p.k_scale,p.v_scale)):
            raise ValueError('Invalid packed operand dtype')
        if p.v.numel()!=b*hk*d*((nk+ck-1)//ck*ck) or p.v_scale.numel()!=b*hk*d:
            raise ValueError('Invalid packed V geometry')
        variants={'baseline':0,'q64':1,'cache_q':2,'pv_n8':4,'q64_pv_n8':5,
                  'narrow':6,'pv_n8_narrow':7,'q64_pv_n8_narrow':8}
        if variant not in variants:
            raise ValueError('Unknown dense candidate')
        routed=plan is not None or is_causal
        if routed and (p.cta_k!=64 or variant!='baseline'):
            raise ValueError('Routed/causal execution uses key64 and its own query64 schedule')
        if variant=='cache_q' and (d!=128 or p.cta_k!=128):
            raise ValueError('Cached-Q candidate requires D128/key128')
        if 'pv_n8' in variant and (d!=128 or ck!=128):
            raise ValueError('n8 PV scheduling requires D128/key128')
        if 'narrow' in variant and b*h*max(n,((nk+ck-1)//ck)*ck)*d>=2**32:
            raise ValueError('32-bit addressing would overflow; retain the 64-bit baseline')
        if plan is not None:
            check_routes(plan.routes,n,nk)
            if (plan.queries,plan.keys)!=(n,nk) or plan.routes.shape[:2]!=(b,h):
                raise ValueError('Block plan geometry mismatch')
            if plan.ids.shape!=plan.routes.shape or plan.counts.shape!=plan.routes.shape[:-1]:
                raise ValueError('Malformed compact metadata')
            if any(t.device!=p.q.device or t.dtype!=torch.int32 or not t.is_contiguous()
                   for t in (plan.ids,plan.counts,plan.routes)):
                raise ValueError('Int32 contiguous plan tensors on the packed device required')
        out_dtype=torch.bfloat16 if p.input_dtype==torch.float32 else p.input_dtype
        output=torch.empty((b,h,n,d),device=p.q.device,dtype=out_dtype)
        lse=torch.empty((b,h,n),device=p.q.device,dtype=torch.float32) if pooled else None
        with torch.cuda.device(p.q.device):
            stream=torch.cuda.current_stream(p.q.device).cuda_stream
            a=AttentionArgs(*[t.data_ptr() for t in (p.q,p.k,p.v,output,p.q_scale,p.k_scale,p.v_scale)],
                lse.data_ptr() if lse is not None else 0,plan.ids.data_ptr() if plan else 0,
                plan.counts.data_ptr() if plan else 0,b,h,hk,n,nk,d,p.cta_k,
                3 if routed else variants[variant],
                int(is_causal),query_offset,1 if out_dtype==torch.float16 else 2,
                plan.ids.shape[-1] if plan else 0,p.attention_scale,stream)
            self._check(self.lib.h3_launch(C.byref(a)))
            if pooled:
                kc=torch.empty((b,hk,(nk+63)//64,d),device=p.q.device,dtype=torch.float16)
                vc=torch.empty_like(kc)
                pool=PoolArgs(*[t.data_ptr() for t in (p.q,p.k,p.v,p.q_scale,p.k_scale,p.v_scale)],
                    0,kc.data_ptr(),vc.data_ptr(),plan.routes.data_ptr(),output.data_ptr(),lse.data_ptr(),
                    b,h,hk,n,nk,d,1 if out_dtype==torch.float16 else 2,p.attention_scale,stream)
                self._check(self.lib.h3_pool(C.byref(pool)))
        return output.float() if p.input_dtype==torch.float32 else output

    def attention(self,q,k,v,*,mode='dense',prefix_tokens=0,routes=None,is_causal=False,
                  query_offset=0,variant='baseline',scale=None):
        if mode not in ('dense','drop','pooled'):
            raise ValueError('Unknown attention mode')
        if mode=='dense' and routes is not None:
            raise ValueError('Use drop/pooled mode with routes')
        if mode=='pooled' and is_causal:
            raise ValueError('Causal pooling is unsupported')
        if mode!='dense' and routes is None:
            if is_causal:
                raise ValueError('Causal sparse mode requires explicit routes; content routing sees full query leaves')
            routes=routing(q,k,prefix_tokens)
        plan=self.plan(routes,q.shape[2],k.shape[2]) if routes is not None else None
        p=self.quantize(q,k,v,cta_k=64 if (plan is not None or is_causal) else None,scale=scale)
        return self.execute(p,plan=plan,is_causal=is_causal,query_offset=query_offset,
                            variant=variant,pooled=mode=='pooled')
