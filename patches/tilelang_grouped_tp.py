"""Opt-in Group32 SM75 attention for the current TP2 boundaries.

Native GQA (no KV replication), rectangular query partitions, and DiT drop or
scalar pooling. Leaf order, Sol centering and FP16 probability rounding match
the audited Group32 contract. This module does not patch production dispatch.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import torch
import torch.nn.functional as F


@dataclass
class TPGrouped:
    q: torch.Tensor
    qi: torch.Tensor
    ki: torch.Tensor
    v: torch.Tensor
    qs: torch.Tensor
    ks: torch.Tensor
    kc: torch.Tensor
    vc: torch.Tensor
    counts: torch.Tensor
    queries: int
    keys: int


def prepare(q, k, v, fused=False, sparse=False):
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError('BHND Q and matching K/V required')
    b,h,n,d=q.shape; bk,hk,nk,dk=k.shape
    if b!=bk or d!=dk or d not in (64,128) or h%hk or min(n,nk)<1:
        raise ValueError('D64/128, positive lengths and divisible GQA heads required')
    if any(x.dtype!=torch.float16 or x.device!=q.device for x in (q,k,v)):
        raise ValueError('FP16 operands on one device required')
    if sparse and (q.shape!=k.shape):
        raise ValueError('Sparse modes currently require DiT self-attention')
    nqpad=math.ceil(n/64)*64; nkpad=math.ceil(nk/64)*64; nb=nkpad//64
    counts=(nk-torch.arange(nb,device=k.device)*64).clamp(0,64).int()
    raw=F.pad(k.float(),(0,0,0,nkpad-nk)).reshape(b,hk,nb,64,d).sum(-2)/counts[None,None,:,None]
    mean=raw.mean(-2,keepdim=True)  # unweighted leaf mean, including ragged leaf
    if fused:
        if not q.is_cuda or torch.cuda.get_device_capability(q.device)!=(7,5):
            raise ValueError('Actual SM75 required for fused preparation')
        from patches.tilelang_grouped_fused import quantizer
        dummy=torch.empty((b,h,1,d),device=q.device,dtype=torch.float32)
        qi,qs=quantizer(b,h,n,d,32,False)(q.contiguous(),dummy)
        ki,ks=quantizer(b,hk,nk,d,32,True)(k.contiguous(),mean)
    else:
        def quant(x,heads,length,padded):
            a=x.reshape(b,heads,length,d//32,32)
            s=(a.abs().amax(-1)/127).clamp_min(1e-8)
            i=(a/s[...,None]).round().clamp(-127,127).to(torch.int8).reshape(b,heads,length,d)
            return F.pad(i,(0,0,0,padded-length)).contiguous(),F.pad(s,(0,0,0,padded-length)).contiguous()
        qi,qs=quant(q.float(),h,n,nqpad)
        ki,ks=quant(k.float()-mean,hk,nk,nkpad)
    # Dense kernels never access raw Q or the summaries. Do not compute them.
    qp=F.pad(q,(0,0,0,nqpad-n)).contiguous() if sparse else torch.empty((b,h,nqpad,d),device=q.device,dtype=q.dtype)
    kc=(raw-mean).half().contiguous() if sparse else torch.empty((b,hk,nb,d),device=q.device,dtype=q.dtype)
    vc=(F.pad(v.float(),(0,0,0,nkpad-nk)).reshape(b,hk,nb,64,d).sum(-2)/counts[None,None,:,None]).half().contiguous() if sparse else torch.empty_like(kc)
    return TPGrouped(qp,qi,ki,F.pad(v,(0,0,0,nkpad-nk)).contiguous(),qs,ks,kc,vc,counts,n,nk)


def routes(p,prefix_tokens=874):
    from patches.tilelang_sparse_attention import make_routes
    r=make_routes(p,'sol',tau=0.)
    prefix=math.ceil(prefix_tokens/64)
    r[:,:,:prefix,:]=1; r[:,:,:,:prefix]=1
    return r


@lru_cache(maxsize=64)
def kernel(b,h,hk,n,nk,d,mode='dense',causal=False,query_offset=0,cached=False):
    import tilelang
    import tilelang.language as T
    if mode not in ('dense','drop','scalar') or (mode!='dense' and (h!=hk or n!=nk or causal)):
        raise ValueError('Unsupported attention contract')
    np=math.ceil(n/64)*64; kp=math.ceil(nk/64)*64; nb=kp//64; ng=d//32
    scale=d**-.5*math.log2(math.e)
    @T.prim_func
    def main(Q:T.Tensor((b,h,np,d),'float16'), QI:T.Tensor((b,h,np,d),'int8'),
             KI:T.Tensor((b,hk,kp,d),'int8'), V:T.Tensor((b,hk,kp,d),'float16'),
             QS:T.Tensor((b,h,np,ng),'float32'),KS:T.Tensor((b,hk,kp,ng),'float32'),
             KC:T.Tensor((b,hk,nb,d),'float16'),VC:T.Tensor((b,hk,nb,d),'float16'),
             Counts:T.Tensor((nb,),'int32'),Routes:T.Tensor((b,h,math.ceil(n/64),nb),'int32'),
             O:T.Tensor((b,h,np,d),'float16')):
        with T.Kernel(np//32,h,b,threads=128) as (bx,hy,bz):
            kh=hy//(h//hk)
            q8=T.alloc_shared((32,d if cached else 32),'int8')
            k8=T.alloc_shared((64,d if cached else 32),'int8')
            q16=T.alloc_shared((32,d),'float16')
            qscale=T.alloc_shared((32,ng),'float32');kscale=T.alloc_shared((64,ng),'float32')
            v16=T.alloc_shared((64,d),'float16');p16=T.alloc_shared((32,64),'float16')
            dot=T.alloc_fragment((32,64),'int32');scores=T.alloc_fragment((32,64),'float32')
            pv=T.alloc_fragment((32,d),'float32');out=T.alloc_fragment((32,d),'float32')
            m=T.alloc_fragment((32,),'float32');hm=T.alloc_fragment((32,),'float32')
            old=T.alloc_fragment((32,),'float32');alpha=T.alloc_fragment((32,),'float32')
            denom=T.alloc_fragment((32,),'float32');psum=T.alloc_fragment((32,),'float32')
            products=T.alloc_fragment((32,d),'float32');summary=T.alloc_fragment((32,),'float32')
            summary_shared=T.alloc_shared((32,),'float32');beta=T.alloc_fragment((32,),'float32')
            pooled_p=T.alloc_fragment((32,),'float16')
            if cached:
                T.copy(QI[bz,hy,bx*32:(bx+1)*32,:],q8)
                T.copy(QS[bz,hy,bx*32:(bx+1)*32,:],qscale)
            if mode=='scalar':T.copy(Q[bz,hy,bx*32:(bx+1)*32,:],q16)
            T.clear(out);T.clear(denom);T.fill(m,-T.infinity('float32'))
            loop=T.min(nb,T.ceildiv(query_offset+(bx+1)*32,64)) if causal else nb
            for kb in T.serial(loop):
                if mode=='dense' or Routes[bz,hy,bx//2,kb]!=0:
                    T.clear(scores)
                    if cached:
                        T.copy(KI[bz,kh,kb*64:(kb+1)*64,:],k8)
                        T.copy(KS[bz,kh,kb*64:(kb+1)*64,:],kscale)
                    for g in T.serial(ng):
                        T.clear(dot)
                        if cached:
                            T.gemm(q8[0:32,g*32:(g+1)*32],k8[0:64,g*32:(g+1)*32],dot,transpose_B=True)
                            for i,j in T.Parallel(32,64):scores[i,j]+=T.cast(dot[i,j],'float32')*qscale[i,g]*kscale[j,g]*scale
                        else:
                            T.copy(QI[bz,hy,bx*32:(bx+1)*32,g*32:(g+1)*32],q8)
                            T.copy(KI[bz,kh,kb*64:(kb+1)*64,g*32:(g+1)*32],k8)
                            T.gemm(q8,k8,dot,transpose_B=True)
                            for i,j in T.Parallel(32,64):scores[i,j]+=T.cast(dot[i,j],'float32')*QS[bz,hy,bx*32+i,g]*KS[bz,kh,kb*64+j,g]*scale
                    for i,j in T.Parallel(32,64):
                        scores[i,j]=T.if_then_else((kb*64+j<nk)&((kb*64+j<=query_offset+bx*32+i) if causal else True),scores[i,j],-T.infinity('float32'))
                    T.copy(m,old);T.reduce_max(scores,hm,dim=1)
                    for i in T.Parallel(32):
                        m[i]=T.max(old[i],hm[i]);alpha[i]=T.exp2(old[i]-m[i])
                    for i,j in T.Parallel(32,64):
                        scores[i,j]=T.exp2(scores[i,j]-m[i]);p16[i,j]=T.cast(scores[i,j],'float16')
                    T.reduce_sum(scores,psum,dim=1)
                    T.copy(V[bz,kh,kb*64:(kb+1)*64,:],v16);T.clear(pv);T.gemm(p16,v16,pv)
                    for i,x in T.Parallel(32,d):out[i,x]=out[i,x]*alpha[i]+pv[i,x]
                    for i in T.Parallel(32):denom[i]=denom[i]*alpha[i]+psum[i]
                elif mode=='scalar':
                    for i,x in T.Parallel(32,d):products[i,x]=T.cast(q16[i,x],'float32')*T.cast(KC[bz,kh,kb,x],'float32')
                    T.reduce_sum(products,summary,dim=1);T.copy(summary,summary_shared)
                    for i in T.Parallel(32):
                        old[i]=m[i];hm[i]=summary_shared[i]*scale+T.log2(T.cast(Counts[kb],'float32'))
                        m[i]=T.max(old[i],hm[i]);alpha[i]=T.exp2(old[i]-m[i]);beta[i]=T.exp2(hm[i]-m[i])
                        pooled_p[i]=T.cast(beta[i],'float16');denom[i]=denom[i]*alpha[i]+beta[i]
                    for i,x in T.Parallel(32,d):out[i,x]=out[i,x]*alpha[i]+T.cast(pooled_p[i],'float32')*T.cast(VC[bz,kh,kb,x],'float32')
            for i,x in T.Parallel(32,d):O[bz,hy,bx*32+i,x]=T.cast(out[i,x]/denom[i],'float16')
    return tilelang.compile(main,out_idx=[10],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


def execute(p,r=None,mode='dense',causal=False,query_offset=0,cached=False):
    if not p.qi.is_cuda or torch.cuda.get_device_capability(p.qi.device)!=(7,5):
        raise ValueError('Actual SM75 execution required')
    b,h,_,d=p.qi.shape;hk=p.ki.shape[1]
    if r is None:
        if mode!='dense':raise ValueError('Sparse routes required')
        r=torch.empty((b,h,math.ceil(p.queries/64),math.ceil(p.keys/64)),device=p.qi.device,dtype=torch.int32)
    out=kernel(b,h,hk,p.queries,p.keys,d,mode,causal,query_offset,cached)(p.q,p.qi,p.ki,p.v,p.qs,p.ks,p.kc,p.vc,p.counts,r)
    return out[:,:,:p.queries]


def reference(p,r=None,mode='dense',causal=False,query_offset=0,query_chunk=128):
    """Independent FP32 dequantized-dot, streaming-softmax implementation."""
    b,h,_,d=p.qi.shape;hk=p.ki.shape[1];n=p.queries;nk=p.keys
    def dequant(i,s,length,heads):
        return (i[:,:,:length].float().reshape(b,heads,length,d//32,32)*s[:,:,:length,:,None]).reshape(b,heads,length,d)
    q=dequant(p.qi,p.qs,n,h);k=dequant(p.ki,p.ks,nk,hk)
    ids=torch.arange(h,device=q.device)//(h//hk)
    k=k[:,ids];v=p.v[:,ids];outputs=[]
    for a in range(0,n,query_chunk):
        zq=q[:,:,a:a+query_chunk];end=a+zq.shape[2]
        m=torch.full(zq.shape[:-1],-float('inf'),device=q.device);den=torch.zeros_like(m);u=torch.zeros_like(zq)
        qids=torch.arange(a,end,device=q.device)
        for kb in range(math.ceil(nk/64)):
            start=kb*64;ke=min(start+64,nk)
            z=zq@k[:,:,start:ke].transpose(-1,-2)/math.sqrt(d)
            if causal:z=z.masked_fill(torch.arange(start,ke,device=q.device)[None,:]>query_offset+qids[:,None],-float('inf'))
            keep=torch.ones_like(m,dtype=torch.bool) if mode=='dense' else r[:,:,qids//64,kb].bool()
            if mode=='drop':z=z.masked_fill(~keep[...,None],-float('inf'))
            pooled=(p.q[:,:,a:end].float()*p.kc[:,:,kb,None,:].float()).sum(-1)/math.sqrt(d)+math.log(ke-start) if mode=='scalar' else torch.full_like(m,-float('inf'))
            pooled=pooled.masked_fill(keep,-float('inf'))
            if mode=='scalar':z=z.masked_fill(~keep[...,None],-float('inf'))
            nm=torch.maximum(m,torch.maximum(z.amax(-1),pooled))
            # Future causal leaves or skipped early leaves may have no finite mass.
            safe=torch.where(torch.isfinite(nm),nm,torch.zeros_like(nm))
            alpha=(m-safe).exp();prob=(z-safe[...,None]).exp();beta=(pooled-safe).exp()
            u=u*alpha[...,None]+prob.half().float()@v[:,:,start:ke].float()
            if mode=='scalar':u+=beta.half().float()[...,None]*p.vc[:,:,kb,None,:].float()
            den=den*alpha+prob.sum(-1)+beta;m=nm
        outputs.append(u/den[...,None])
    return torch.cat(outputs,-2)
