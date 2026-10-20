"""SM75 implementation of Sage-inspired smoothed, token-block INT8 QK.

This is not the upstream Sage CUDA kernel (its dispatcher lacks SM75).
FP32 token-mean subtraction and natural-log scales are explicit choices.
The upstream log2(e)/sqrt(D) Q-scale factor is applied in our attention kernel,
which is algebraically equivalent before rounding. No clipping heuristics.
"""
import math
import torch
import torch.nn.functional as F
from patches.tilelang_sparse_attention import Prepared, LEAF
from patches.tilelang_dense_preparation import geometry, run_prevalidated_dense


def quant_blocks(x, block,rounding='half_away'):
    """One absmax over tokens AND channels; half-away-from-zero rounding."""
    b,h,n,d=x.shape
    padded=F.pad(x,(0,0,0,(-n)%block)).reshape(b,h,-1,block,d)
    scale=(padded.abs().amax((-1,-2))/127).clamp_min(1e-8)
    if rounding=='cuda_rn':
        # Upstream fused CUDA uses reciprocal absmax multiplication and cvt.rni.
        maximum=padded.abs().amax((-1,-2)).clamp_min(1e-7)
        scale=maximum/127
        normalized=padded*(127/maximum)[...,None,None]
        rounded=normalized.round()
    elif rounding=='half_away':
        normalized=padded/scale[...,None,None]
        rounded=normalized.sign()*(normalized.abs()+.5).floor()
    else:raise ValueError(rounding)
    integer=rounded.clamp(-127,127).to(torch.int8)
    return integer.reshape(b,h,-1,d)[:,:,:n].contiguous(), scale.repeat_interleave(block,-1)[:,:,:n].contiguous()


def prepare_sage(q,k,v,q_block=32,k_block=64,smoothing='token',enable_gqa=False,rounding='half_away'):
    if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape or q.shape[0]!=k.shape[0] or q.shape[2:]!=k.shape[2:]:
        raise ValueError('BHND self attention required')
    if q.shape[-1] not in (64,128) or q.shape[-2]<1 or q_block not in (1,16,32,64,128) or k_block not in (1,64):
        raise ValueError('Unsupported dimensions or granularity')
    if q.device!=k.device or q.device!=v.device:raise ValueError('Shared device required')
    if q.shape[1]!=k.shape[1]:
        if not enable_gqa or q.shape[1]%k.shape[1]:raise ValueError('Invalid GQA')
        group=q.shape[1]//k.shape[1]
        k=k.repeat_interleave(group,1);v=v.repeat_interleave(group,1)
    b,h,n,d=q.shape;pad=(-n)%LEAF
    counts,routes,kc,vc,vi,vs=geometry(b,h,n,d,str(q.device))
    qf,kf=q.float(),k.float()
    if smoothing=='token':mean=kf.mean(-2,keepdim=True)
    elif smoothing=='token_native_dtype':mean=k.mean(-2,keepdim=True).float()
    elif smoothing=='none':mean=torch.zeros_like(kf[:,:,:1])
    elif smoothing=='sol':
        centers=F.pad(kf,(0,0,0,pad)).reshape(b,h,-1,LEAF,d).sum(-2)/counts[None,None,:,None]
        mean=centers.mean(-2,keepdim=True)
    else:raise ValueError(smoothing)
    qi,qs=quant_blocks(qf,q_block,rounding);ki,ks=quant_blocks(kf-mean,k_block,rounding)
    padded=lambda x:F.pad(x,(0,0,0,pad)).contiguous()
    p=Prepared(padded(q.half()),padded(v.half()),padded(qi),padded(ki),
               F.pad(qs,(0,pad)),F.pad(ks,(0,pad)),kc,vc,counts,vi,vs,
               (qf*mean).sum(-1)/math.sqrt(d),n)
    return p,routes


def sage_attention(q,k,v,causal=False,enable_gqa=False,**options):
    p,routes=prepare_sage(q,k,v,enable_gqa=enable_gqa,**options)
    return run_prevalidated_dense(p,routes,causal=causal)
