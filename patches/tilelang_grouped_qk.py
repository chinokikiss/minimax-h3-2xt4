"""Dense INT8 QK with 32-channel scale groups, for the SM75 accuracy ablation.

The real TE head gate attributes the dominant error to K quantization. Narrow
scale groups prevent one large channel setting the scale for all 128 channels.
FP16 PV remains unchanged; this is an accuracy candidate, not a tuned kernel.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import torch
import torch.nn.functional as F
from patches.tilelang_dense_preparation import geometry


@dataclass
class Grouped:
    qi:torch.Tensor
    ki:torch.Tensor
    v:torch.Tensor
    qs:torch.Tensor
    ks:torch.Tensor
    shift:torch.Tensor
    tokens:int


def prepare_grouped(q,k,v,enable_gqa=False):
    if q.dtype!=torch.float16 or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise ValueError('FP16 BHND required')
    if q.ndim==4 and k.ndim==4 and q.shape[1]!=k.shape[1] and enable_gqa:
        if q.shape[1]%k.shape[1]:raise ValueError('Divisible GQA head counts required')
        k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1);v=v.repeat_interleave(q.shape[1]//v.shape[1],dim=1)
    if q.ndim!=4 or q.shape!=k.shape or k.shape!=v.shape or q.shape[-1] not in (64,128):
        raise ValueError('Matching expanded BHND inputs, D64/128 required')
    b,h,n,d=q.shape;g=d//32;nb=(n+63)//64;pad=nb*64-n
    counts=geometry(b,h,n,d,str(q.device))[0]
    def padded(x):return F.pad(x,(0,0,0,pad)).contiguous()
    qf,kf=q.float(),k.float()
    raw_kc=padded(kf).reshape(b,h,nb,64,d).sum(-2)/counts[None,None,:,None]
    mean=raw_kc.mean(-2,keepdim=True)
    def quant(x):
        grouped=x.reshape(b,h,n,g,32)
        scale=(grouped.abs().amax(-1)/127).clamp_min(1e-8)
        integer=(grouped/scale[...,None]).round().clamp(-127,127).to(torch.int8).reshape(b,h,n,d)
        return padded(integer),F.pad(scale,(0,0,0,pad)).contiguous()
    qi,qs=quant(qf);ki,ks=quant(kf-mean)
    shift=(qf*mean).sum(-1)/math.sqrt(d)
    return Grouped(qi,ki,padded(v),qs,ks,shift,n)


@lru_cache(maxsize=32)
def grouped_kernel(b,h,n,d,causal=False):
    import tilelang
    import tilelang.language as T
    np=((n+63)//64)*64;nb=np//64;groups=d//32
    scale=d**-.5*math.log2(math.e)
    @T.prim_func
    def main(QI:T.Tensor((b,h,np,d),'int8'),KI:T.Tensor((b,h,np,d),'int8'),
             V:T.Tensor((b,h,np,d),'float16'),QS:T.Tensor((b,h,np,groups),'float32'),
             KS:T.Tensor((b,h,np,groups),'float32'),O:T.Tensor((b,h,np,d),'float16'),
             LSE:T.Tensor((b,h,np),'float32')):
        with T.Kernel(np//32,h,b,threads=128) as (bx,hy,bz):
            q8=T.alloc_shared((32,32),'int8');k8=T.alloc_shared((64,32),'int8')
            v16=T.alloc_shared((64,d),'float16');p16=T.alloc_shared((32,64),'float16')
            dot=T.alloc_fragment((32,64),'int32');scores=T.alloc_fragment((32,64),'float32')
            pv=T.alloc_fragment((32,d),'float32');out=T.alloc_fragment((32,d),'float32')
            m=T.alloc_fragment((32,),'float32');hm=T.alloc_fragment((32,),'float32')
            old_m=T.alloc_fragment((32,),'float32');alpha=T.alloc_fragment((32,),'float32')
            denom=T.alloc_fragment((32,),'float32');psum=T.alloc_fragment((32,),'float32')
            T.clear(out);T.clear(denom);T.fill(m,-T.infinity('float32'))
            loop=T.min(nb,T.ceildiv((bx+1)*32,64)) if causal else nb
            for kb in T.serial(loop):
                T.clear(scores)
                for group in T.serial(groups):
                    T.copy(QI[bz,hy,bx*32:(bx+1)*32,group*32:(group+1)*32],q8)
                    T.copy(KI[bz,hy,kb*64:(kb+1)*64,group*32:(group+1)*32],k8)
                    T.clear(dot);T.gemm(q8,k8,dot,transpose_B=True)
                    for i,j in T.Parallel(32,64):
                        scores[i,j]+=T.cast(dot[i,j],'float32')*QS[bz,hy,bx*32+i,group]*KS[bz,hy,kb*64+j,group]*scale
                for i,j in T.Parallel(32,64):
                    scores[i,j]=T.if_then_else((kb*64+j<n)&((kb*64+j<=bx*32+i) if causal else True),scores[i,j],-T.infinity('float32'))
                T.copy(m,old_m);T.reduce_max(scores,hm,dim=1)
                for i in T.Parallel(32):
                    m[i]=T.max(old_m[i],hm[i]);alpha[i]=T.exp2(old_m[i]-m[i])
                for i,j in T.Parallel(32,64):
                    scores[i,j]=T.exp2(scores[i,j]-m[i]);p16[i,j]=T.cast(scores[i,j],'float16')
                T.reduce_sum(scores,psum,dim=1)
                T.copy(V[bz,hy,kb*64:(kb+1)*64,:],v16);T.clear(pv);T.gemm(p16,v16,pv)
                for i,x in T.Parallel(32,d):out[i,x]=out[i,x]*alpha[i]+pv[i,x]
                for i in T.Parallel(32):denom[i]=denom[i]*alpha[i]+psum[i]
            for i,x in T.Parallel(32,d):O[bz,hy,bx*32+i,x]=T.cast(out[i,x]/denom[i],'float16')
            for i in T.Parallel(32):LSE[bz,hy,bx*32+i]=(m[i]+T.log2(denom[i]))*math.log(2)
    return tilelang.compile(main,out_idx=[5,6],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


def execute_grouped(p,causal=False):
    if not p.qi.is_cuda or torch.cuda.get_device_capability(p.qi.device)!=(7,5):
        raise ValueError('Actual SM75 execution required')
    b,h,_,d=p.qi.shape
    out,lse=grouped_kernel(b,h,p.tokens,d,causal)(p.qi,p.ki,p.v,p.qs,p.ks)
    return out[:,:,:p.tokens],lse[:,:,:p.tokens]+p.shift


def grouped_reference(p,causal=False,query_chunk=None):
    """Independent dense FP32 reference, grouped dequantization before dot products."""
    n=p.tokens;b,h,_,d=p.qi.shape;groups=d//32
    q=(p.qi[:,:,:n].float().reshape(b,h,n,groups,32)*p.qs[:,:,:n,:,None]).reshape(b,h,n,d)
    k=(p.ki[:,:,:n].float().reshape(b,h,n,groups,32)*p.ks[:,:,:n,:,None]).reshape(b,h,n,d)
    query_chunk=n if query_chunk is None else query_chunk
    outputs=[]
    for start in range(0,n,query_chunk):
        end=min(start+query_chunk,n)
        shape=(b,h,end-start)
        m=torch.full(shape,-float('inf'),device=q.device);denom=torch.zeros(shape,device=q.device)
        u=torch.zeros((*shape,d),device=q.device)
        for kb in range(0,n,64):
            ke=min(kb+64,n)
            z=q[:,:,start:end]@k[:,:,kb:ke].transpose(-1,-2)/math.sqrt(d)
            if causal:
                z=z.masked_fill(torch.arange(kb,ke,device=q.device)[None,:]>torch.arange(start,end,device=q.device)[:,None],-float('inf'))
            nm=torch.maximum(m,z.amax(-1));alpha=(m-nm).exp()
            prob=(z-nm[...,None]).exp()
            u=u*alpha[...,None]+prob.half().float()@p.v[:,:,kb:ke].float()
            denom=denom*alpha+prob.sum(-1);m=nm
        outputs.append(u/denom[...,None])
    return torch.cat(outputs,-2)
