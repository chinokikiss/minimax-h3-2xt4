"""SM75 channel-group QK, preserving the audited legacy Group32 contract.

This is per-token/channel-group quantization, not Sage Q32/K64. Integer
group dot-products are scaled and accumulated inside the attention kernel.
No partial N×N GEMM tensors are materialized in global memory.
"""
from functools import lru_cache
import math
import torch
import torch.nn.functional as F
from patches.tilelang_grouped_qk import Grouped


@lru_cache(maxsize=64)
def quantizer(b,h,n,d,group,center):
    import tilelang
    import tilelang.language as T
    np=math.ceil(n/64)*64;ng=d//group
    @T.prim_func
    def main(X:T.Tensor((b,h,n,d),'float16'),Mean:T.Tensor((b,h,1,d),'float32'),
             XI:T.Tensor((b,h,np,d),'int8'),Scale:T.Tensor((b,h,np,ng),'float32')):
        with T.Kernel(T.ceildiv(np,32),h,b,threads=128) as (bx,hy,bz):
            value=T.alloc_fragment((32,group),'float32');absolute=T.alloc_fragment((32,group),'float32')
            maximum=T.alloc_fragment((32,),'float32');scales=T.alloc_shared((32,),'float32')
            for g in T.serial(ng):
                for i,j in T.Parallel(32,group):
                    value[i,j]=T.if_then_else(bx*32+i<n,T.cast(X[bz,hy,bx*32+i,g*group+j],'float32')-(Mean[bz,hy,0,g*group+j] if center else 0.),0.)
                    absolute[i,j]=T.abs(value[i,j])
                T.reduce_max(absolute,maximum,dim=1)
                for i in T.Parallel(32):scales[i]=T.max(maximum[i]*T.float32(1./127.),1e-8)
                for i,j in T.Parallel(32,group):
                    normal=T.call_pure_extern('float32','__fdiv_rn',value[i,j],scales[i])
                    integer=T.call_pure_extern('int32','__float2int_rn',normal)
                    XI[bz,hy,bx*32+i,g*group+j]=T.cast(T.max(-127,T.min(127,integer)),'int8')
                for i in T.Parallel(32):Scale[bz,hy,bx*32+i,g]=scales[i]
    return tilelang.compile(main,out_idx=[2,3],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


def prepare_grouped_fused(q,k,v,group=32,want_lse=False):
    if q.ndim!=4 or q.shape!=k.shape or k.shape!=v.shape or q.shape[-1] not in (64,128):raise ValueError('Matching BHND D64/128 required')
    if group not in (16,32,64,128) or q.shape[-1]%group:raise ValueError('Divisible channel group required')
    if q.dtype!=torch.float16 or k.dtype!=q.dtype or v.dtype!=q.dtype:raise ValueError('FP16 operands required')
    if not q.is_cuda or q.device!=k.device or q.device!=v.device or torch.cuda.get_device_capability(q.device)!=(7,5):raise ValueError('Actual SM75 shared device required')
    b,h,n,d=q.shape;nb=math.ceil(n/64);pad=nb*64-n
    counts=(n-torch.arange(nb,device=k.device)*64).clamp(0,64).int()
    # Preserve the legacy unweighted mean of leaf centroids, including the
    # ragged leaf. Changing this to token mean would change the contract.
    mean=(F.pad(k.float(),(0,0,0,pad)).reshape(b,h,nb,64,d).sum(-2)/counts[None,None,:,None]).mean(-2,keepdim=True)
    qi,qs=quantizer(b,h,n,d,group,False)(q.contiguous(),mean)
    ki,ks=quantizer(b,h,n,d,group,True)(k.contiguous(),mean)
    shift=(q.float()*mean).sum(-1)/math.sqrt(d) if want_lse else None
    return Grouped(qi,ki,F.pad(v,(0,0,0,pad)).contiguous(),qs,ks,shift,n)


@lru_cache(maxsize=64)
def attention_kernel(b,h,n,d,group=32,causal=False):
    import tilelang
    import tilelang.language as T
    np=math.ceil(n/64)*64;nb=np//64;ng=d//group;scale=d**-.5*math.log2(math.e)
    @T.prim_func
    def main(QI:T.Tensor((b,h,np,d),'int8'),KI:T.Tensor((b,h,np,d),'int8'),
             V:T.Tensor((b,h,np,d),'float16'),QS:T.Tensor((b,h,np,ng),'float32'),
             KS:T.Tensor((b,h,np,ng),'float32'),O:T.Tensor((b,h,np,d),'float16'),
             LSE:T.Tensor((b,h,np),'float32')):
        with T.Kernel(np//32,h,b,threads=128) as (bx,hy,bz):
            q8=T.alloc_shared((32,d),'int8');k8=T.alloc_shared((64,d),'int8')
            qs=T.alloc_shared((32,ng),'float32');ks=T.alloc_shared((64,ng),'float32')
            v16=T.alloc_shared((64,d),'float16');p16=T.alloc_shared((32,64),'float16')
            dot=T.alloc_fragment((32,64),'int32');scores=T.alloc_fragment((32,64),'float32')
            pv=T.alloc_fragment((32,d),'float32');out=T.alloc_fragment((32,d),'float32')
            m=T.alloc_fragment((32,),'float32');hm=T.alloc_fragment((32,),'float32')
            old=T.alloc_fragment((32,),'float32');alpha=T.alloc_fragment((32,),'float32')
            denom=T.alloc_fragment((32,),'float32');psum=T.alloc_fragment((32,),'float32')
            T.copy(QI[bz,hy,bx*32:(bx+1)*32,:],q8);T.copy(QS[bz,hy,bx*32:(bx+1)*32,:],qs)
            T.clear(out);T.clear(denom);T.fill(m,-T.infinity('float32'))
            loop=T.min(nb,T.ceildiv((bx+1)*32,64)) if causal else nb
            for kb in T.serial(loop):
                T.copy(KI[bz,hy,kb*64:(kb+1)*64,:],k8);T.copy(KS[bz,hy,kb*64:(kb+1)*64,:],ks)
                T.clear(scores)
                for g in T.serial(ng):
                    T.clear(dot)
                    T.gemm(q8[0:32,g*group:(g+1)*group],k8[0:64,g*group:(g+1)*group],dot,transpose_B=True)
                    for i,j in T.Parallel(32,64):scores[i,j]+=T.cast(dot[i,j],'float32')*qs[i,g]*ks[j,g]*scale
                for i,j in T.Parallel(32,64):
                    scores[i,j]=T.if_then_else((kb*64+j<n)&((kb*64+j<=bx*32+i) if causal else True),scores[i,j],-T.infinity('float32'))
                T.copy(m,old);T.reduce_max(scores,hm,dim=1)
                for i in T.Parallel(32):
                    m[i]=T.max(old[i],hm[i]);alpha[i]=T.exp2(old[i]-m[i])
                for i,j in T.Parallel(32,64):
                    scores[i,j]=T.exp2(scores[i,j]-m[i]);p16[i,j]=T.cast(scores[i,j],'float16')
                T.reduce_sum(scores,psum,dim=1)
                T.copy(V[bz,hy,kb*64:(kb+1)*64,:],v16);T.clear(pv);T.gemm(p16,v16,pv)
                for i,x in T.Parallel(32,d):out[i,x]=out[i,x]*alpha[i]+pv[i,x]
                for i in T.Parallel(32):denom[i]=denom[i]*alpha[i]+psum[i]
            for i,x in T.Parallel(32,d):O[bz,hy,bx*32+i,x]=T.cast(out[i,x]/denom[i],'float16')
            for i in T.Parallel(32):LSE[bz,hy,bx*32+i]=(m[i]+T.log2(denom[i]))*math.log(2)
    return tilelang.compile(main,out_idx=[5,6],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


def execute_grouped_fused(p,group=32,causal=False):
    if torch.cuda.get_device_capability(p.qi.device)!=(7,5):raise ValueError('Actual SM75 required')
    b,h,_,d=p.qi.shape
    return attention_kernel(b,h,p.tokens,d,group,causal)(p.qi,p.ki,p.v,p.qs,p.ks)[0][:,:,:p.tokens]


def dequantized_reference(p,group=32,causal=False):
    """Independent streaming FP32 dot/softmax with leaf-wise FP16 P operand."""
    b,h,_,d=p.qi.shape;n=p.tokens;ng=d//group
    q=(p.qi[:,:,:n].float().reshape(b,h,n,ng,group)*p.qs[:,:,:n,:,None]).reshape(b,h,n,d)
    k=(p.ki[:,:,:n].float().reshape(b,h,n,ng,group)*p.ks[:,:,:n,:,None]).reshape(b,h,n,d)
    m=torch.full((b,h,n),-float('inf'),device=q.device);denom=torch.zeros_like(m);out=torch.zeros_like(q)
    for start in range(0,n,64):
        z=q@k[:,:,start:start+64].transpose(-1,-2)/math.sqrt(d)
        if causal:
            z=z.masked_fill(torch.arange(start,min(start+64,n),device=q.device)[None,:]>torch.arange(n,device=q.device)[:,None],-float('inf'))
        hm=z.amax(-1);nm=torch.maximum(m,hm);alpha=(m-nm).exp();prob=(z-nm[...,None]).exp()
        out=out*alpha[...,None]+prob.half().float()@p.v[:,:,start:min(start+64,n)].float()
        denom=denom*alpha+prob.sum(-1);m=nm
    return out/denom[...,None]
