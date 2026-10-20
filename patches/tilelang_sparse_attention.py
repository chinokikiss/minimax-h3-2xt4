"""Correctness-first SM75 INT8 sparse self-attention, expressed in TileLang.

FP16 BHND inputs with causal prefill and explicit GQA expansion. No arbitrary
mask/dropout yet. Routing and quantization preparation
use PyTorch; the tiled QK/softmax/PV operator runs exclusively in TileLang.
Unselected blocks are either omitted or represented by one weighted summary.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import torch

LEAF = 64


@dataclass
class Prepared:
    q: torch.Tensor
    v: torch.Tensor
    qi: torch.Tensor
    ki: torch.Tensor
    qs: torch.Tensor
    ks: torch.Tensor
    kc: torch.Tensor
    vc: torch.Tensor
    counts: torch.Tensor
    vi: torch.Tensor
    vs: torch.Tensor
    shift: torch.Tensor
    tokens: int


def _quant(x):
    s = (x.abs().amax(-1) / 127).clamp_min(1e-8)
    return (x / s[..., None]).round().clamp(-127,127).to(torch.int8), s


def prepare(q, k, v, enable_gqa=False):
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('FP16 Q/K/V required')
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('BHND self-attention tensors required')
    if q.shape[0]!=k.shape[0] or q.shape[2:]!=k.shape[2:]:
        raise ValueError('matching batch, sequence and feature dimensions required')
    if q.shape[-1] not in (64,128) or q.shape[-2] < 1:
        raise ValueError('D=64/128 and nonempty sequence required')
    if q.device != k.device or q.device != v.device:
        raise ValueError('Q/K/V must share a device')
    if q.shape[1]!=k.shape[1]:
        if not enable_gqa or q.shape[1]%k.shape[1]:
            raise ValueError('GQA requires enable_gqa=True and divisible head counts')
        # Correctness scaffold: preserve the native contiguous group mapping.
        groups=q.shape[1]//k.shape[1]
        k=k.repeat_interleave(groups,dim=1)
        v=v.repeat_interleave(groups,dim=1)
    b,h,n,d = q.shape
    nb = (n+LEAF-1)//LEAF
    pad = nb*LEAF-n
    qf,kf,vf = q.float(),k.float(),v.float()
    counts = (n-torch.arange(nb,device=q.device)*LEAF).clamp(0,LEAF).int()
    def padded(x):
        return torch.nn.functional.pad(x,(0,0,0,pad)).contiguous()
    # Sol's unweighted centroid mean, including a ragged final leaf.
    raw_kc = padded(kf).reshape(b,h,nb,LEAF,d).sum(-2)/counts[None,None,:,None]
    mean = raw_kc.mean(-2,keepdim=True)
    qi,qs = _quant(qf)
    ki,ks = _quant(kf-mean)
    kc = (raw_kc-mean).half().contiguous()
    vc = (padded(vf).reshape(b,h,nb,LEAF,d).sum(-2)/counts[None,None,:,None]).half().contiguous()
    # Block/channel V scales factor outside each PV reduction.
    vp = padded(vf).reshape(b,h,nb,LEAF,d)
    vs = (vp.abs().amax(-2)/127).clamp_min(1e-8)
    vi = (vp/vs[...,None,:]).round().clamp(-127,127).to(torch.int8).reshape(b,h,nb*LEAF,d)
    qs = torch.nn.functional.pad(qs,(0,pad)).contiguous()
    ks = torch.nn.functional.pad(ks,(0,pad)).contiguous()
    shift = (qf*mean).sum(-1)/math.sqrt(d)
    return Prepared(padded(q),padded(v),padded(qi),padded(ki),qs,ks,kc,vc,
                    counts.contiguous(),vi.contiguous(),vs.contiguous(),shift,n)


def make_routes(p, kind='sol', tau=1.0, density=0.125, global_blocks=1, local_radius=1):
    b,h,np,d=p.q.shape
    nb=np//LEAF
    qc=p.q.float().reshape(b,h,nb,LEAF,d).sum(-2)/p.counts[None,None,:,None]
    kc=p.kc.float()
    scores=torch.matmul(qc,kc.transpose(-1,-2))/math.sqrt(d)
    ids=torch.arange(nb,device=p.q.device)
    mandatory=(ids[:,None]-ids[None,:]).abs()<=local_radius
    mandatory=mandatory | (ids[None,:]<global_blocks)
    mandatory=mandatory[None,None].expand(b,h,nb,nb)
    if kind=='dense':
        routes=torch.ones_like(mandatory)
    elif kind=='sol':
        mean=kc.mean(-2)
        variance=((kc-mean[...,None,:])**2).mean(-2)
        threshold=(qc*mean[...,None,:]).sum(-1)/math.sqrt(d)
        threshold+=tau*torch.sqrt((qc.square()*variance[...,None,:]).sum(-1)/d+1e-6)
        routes=(scores>threshold[...,None]) | mandatory
    elif kind=='topk':
        budget=max(1,math.ceil(nb*density))
        # Mandatory leaves remain retained even when they exceed the budget.
        rank_scores=scores.masked_fill(mandatory,float('inf'))
        idx=rank_scores.topk(min(nb,budget),dim=-1).indices
        routes=torch.zeros_like(mandatory).scatter_(-1,idx,True) | mandatory
    elif kind=='local_global':
        # Token-order locality control; geometric video packing is a later step.
        routes=mandatory.clone()
    else:
        raise ValueError(kind)
    return routes.to(torch.int32).contiguous()


@lru_cache(maxsize=64)
def get_kernel(b,h,n,d,pooled=True,int8_pv=False,sm=75,causal=False):
    import tilelang
    import tilelang.language as T
    padded_n=((n+63)//64)*64
    nb=padded_n//64
    bm,bn=32,64
    scale=d**-0.5*math.log2(math.e)
    if causal and pooled:
        raise ValueError('Causal attention currently supports dense/selected tokens only')
    @T.prim_func
    def main(
        Q:T.Tensor((b,h,padded_n,d),'float16'),
        QI:T.Tensor((b,h,padded_n,d),'int8'),
        KI:T.Tensor((b,h,padded_n,d),'int8'),
        V:T.Tensor((b,h,padded_n,d),'float16'),
        QS:T.Tensor((b,h,padded_n),'float32'),
        KS:T.Tensor((b,h,padded_n),'float32'),
        KC:T.Tensor((b,h,nb,d),'float16'),
        VC:T.Tensor((b,h,nb,d),'float16'),
        Counts:T.Tensor((nb,),'int32'),
        Routes:T.Tensor((b,h,nb,nb),'int32'),
        VI:T.Tensor((b,h,padded_n,d),'int8'),
        VS:T.Tensor((b,h,nb,d),'float32'),
        O:T.Tensor((b,h,padded_n,d),'float16'),
        LSE:T.Tensor((b,h,padded_n),'float32'),
    ):
        with T.Kernel(padded_n//bm,h,b,threads=128) as (bx,hy,bz):
            q8=T.alloc_shared((bm,d),'int8')
            q16=T.alloc_shared((bm,d),'float16')
            k8=T.alloc_shared((bn,d),'int8')
            # A pooled leaf contains one summary, not 64 independent keys.
            # A 32-column summary tile keeps D128 dual-INT8 below 64 KiB.
            k16=T.alloc_shared((32,d),'float16')
            v16=T.alloc_shared((bn,d),'float16')
            p16=T.alloc_shared((bm,bn),'float16')
            qk8=T.alloc_fragment((bm,bn),'int32')
            qkp=T.alloc_fragment((bm,32),'float32')
            pooled_score=T.alloc_shared((bm,32),'float32')
            scores=T.alloc_fragment((bm,bn),'float32')
            out=T.alloc_fragment((bm,d),'float32')
            pv16=T.alloc_fragment((bm,d),'float32')
            m=T.alloc_fragment((bm,),'float32')
            old_m=T.alloc_fragment((bm,),'float32')
            tile_m=T.alloc_fragment((bm,),'float32')
            alpha=T.alloc_fragment((bm,),'float32')
            beta=T.alloc_fragment((bm,),'float32')
            denom=T.alloc_fragment((bm,),'float32')
            tile_sum=T.alloc_fragment((bm,),'float32')
            if int8_pv:
                p8=T.alloc_shared((bm,bn),'int8')
                v8=T.alloc_shared((bn,d),'int8')
                pv8=T.alloc_fragment((bm,d),'int32')
            T.copy(QI[bz,hy,bx*bm:(bx+1)*bm,:],q8)
            T.copy(Q[bz,hy,bx*bm:(bx+1)*bm,:],q16)
            T.clear(out); T.clear(denom)
            T.fill(m,-T.infinity('float32'))
            loop_count=T.min(nb,T.ceildiv((bx+1)*bm,bn)) if causal else nb
            for kb in T.serial(loop_count):
                if Routes[bz,hy,bx//2,kb]!=0 or pooled:
                    if Routes[bz,hy,bx//2,kb]!=0:
                        T.copy(KI[bz,hy,kb*bn:(kb+1)*bn,:],k8)
                        T.clear(qk8)
                        T.gemm(q8,k8,qk8,transpose_B=True)
                        for i,j in T.Parallel(bm,bn):
                            scores[i,j]=T.if_then_else((kb*bn+j<n) & ((kb*bn+j<=bx*bm+i) if causal else True),
                                T.cast(qk8[i,j],'float32')*QS[bz,hy,bx*bm+i]*KS[bz,hy,kb*bn+j]*scale,
                                -T.infinity('float32'))
                    else:
                        for j,x in T.Parallel(32,d):
                            k16[j,x]=T.if_then_else(j==0,KC[bz,hy,kb,x],T.float16(0))
                        T.clear(qkp)
                        T.gemm(q16,k16,qkp,transpose_B=True)
                        # Stage through shared memory: this MMA
                        # fragment layout differs from the 64-column score tile.
                        T.copy(qkp,pooled_score)
                        for i,j in T.Parallel(bm,bn):
                            scores[i,j]=T.if_then_else(j==0,pooled_score[i,0]*scale+T.log2(T.cast(Counts[kb],'float32')),
                                                      -T.infinity('float32'))
                    T.copy(m,old_m)
                    T.reduce_max(scores,tile_m,dim=1)
                    for i in T.Parallel(bm):
                        m[i]=T.max(old_m[i],tile_m[i])
                        alpha[i]=T.exp2(old_m[i]-m[i])
                        beta[i]=T.exp2(tile_m[i]-m[i])
                    if int8_pv:
                        if Routes[bz,hy,bx//2,kb]!=0:
                            for i,j in T.Parallel(bm,bn):
                                scores[i,j]=T.exp2(scores[i,j]-tile_m[i])
                                p8[i,j]=T.cast(T.min(T.floor(scores[i,j]*127+0.5),127),'int8')
                            T.reduce_sum(scores,tile_sum,dim=1)
                            T.copy(VI[bz,hy,kb*bn:(kb+1)*bn,:],v8)
                            T.clear(pv8)
                            T.gemm(p8,v8,pv8)
                            for i,x in T.Parallel(bm,d):
                                out[i,x]=out[i,x]*alpha[i]+T.cast(pv8[i,x],'float32')*beta[i]*VS[bz,hy,kb,x]/127
                            for i in T.Parallel(bm):
                                denom[i]=denom[i]*alpha[i]+tile_sum[i]*beta[i]
                        else:
                            for i,j in T.Parallel(bm,bn):
                                scores[i,j]=T.exp2(scores[i,j]-m[i])
                                p16[i,j]=T.cast(scores[i,j],'float16')
                            T.reduce_sum(scores,tile_sum,dim=1)
                            for j,x in T.Parallel(bn,d):
                                v16[j,x]=T.if_then_else(j==0,VC[bz,hy,kb,x],T.float16(0))
                            T.clear(pv16)
                            T.gemm(p16,v16,pv16)
                            for i,x in T.Parallel(bm,d):
                                out[i,x]=out[i,x]*alpha[i]+pv16[i,x]
                            for i in T.Parallel(bm):
                                denom[i]=denom[i]*alpha[i]+tile_sum[i]
                    else:
                        for i,j in T.Parallel(bm,bn):
                            scores[i,j]=T.exp2(scores[i,j]-m[i])
                            p16[i,j]=T.cast(scores[i,j],'float16')
                        T.reduce_sum(scores,tile_sum,dim=1)
                        if Routes[bz,hy,bx//2,kb]!=0:
                            T.copy(V[bz,hy,kb*bn:(kb+1)*bn,:],v16)
                        else:
                            for j,x in T.Parallel(bn,d):
                                v16[j,x]=T.if_then_else(j==0,VC[bz,hy,kb,x],T.float16(0))
                        T.clear(pv16)
                        T.gemm(p16,v16,pv16)
                        for i,x in T.Parallel(bm,d):
                            out[i,x]=out[i,x]*alpha[i]+pv16[i,x]
                        for i in T.Parallel(bm):
                            denom[i]=denom[i]*alpha[i]+tile_sum[i]
            for i,x in T.Parallel(bm,d):
                O[bz,hy,bx*bm+i,x]=T.cast(out[i,x]/denom[i],'float16')
            for i in T.Parallel(bm):
                LSE[bz,hy,bx*bm+i]=(m[i]+T.log2(denom[i]))*math.log(2)
    return tilelang.compile(main,out_idx=[12,13],target={'kind':'cuda','arch':f'sm_{sm}'},
                            pass_configs={'tl.enable_async_copy':False})


def execute(p,routes,pooled=True,int8_pv=False,causal=False):
    if not p.q.is_cuda:
        raise ValueError('TileLang execution requires CUDA')
    major,minor=torch.cuda.get_device_capability(p.q.device)
    if (major,minor) not in ((7,5),(12,0)):
        raise ValueError('This experiment supports SM75 and local SM120')
    b,h,_,d=p.q.shape
    expected=(b,h,p.kc.shape[2],p.kc.shape[2])
    if routes.shape!=expected or routes.dtype!=torch.int32 or routes.device!=p.q.device:
        raise ValueError('routes must be int32 BHLL on the input device')
    # Public safety check; benchmarks time raw kernel and this API separately.
    if not pooled and not bool(routes.bool().any(-1).all()):
        raise ValueError('every query leaf needs at least one retained block')
    if causal:
        if pooled:
            raise ValueError('Causal pooling is not supported')
        if not bool(routes.diagonal(dim1=-2,dim2=-1).bool().all()):
            raise ValueError('causal query leaves must retain their diagonal key block')
    kernel=get_kernel(b,h,p.tokens,d,pooled,int8_pv,sm=major*10+minor,causal=causal)
    o,lse=kernel(p.q,p.qi,p.ki,p.v,p.qs,p.ks,p.kc,p.vc,p.counts,routes.contiguous(),p.vi,p.vs)
    return o[:,:,:p.tokens],lse[:,:,:p.tokens]+p.shift


def reference(p,routes,pooled=True,int8_pv=False,quantized=True,round_p=True,causal=False):
    """Independent streaming PyTorch reference, never allocates NxN scores.

    Mirrors the chosen mathematical contract and optionally the P operand cast.
    TF32 must be disabled by the experiment caller for accurate integer dots.
    """
    b,h,_,d=p.q.shape
    n=p.tokens
    if causal and pooled:
        raise ValueError('Causal pooling is not supported')
    q=p.q[:,:,:n].float()
    qhat=p.qi[:,:,:n].float()
    shape=(b,h,n)
    m=torch.full(shape,float('-inf'),device=q.device)
    l=torch.zeros(shape,device=q.device)
    u=torch.zeros((*shape,d),device=q.device)
    route=routes.repeat_interleave(LEAF,dim=2)[:,:,:n].bool()
    for kb in range(p.kc.shape[2]):
        selected=route[...,kb]
        if not pooled and not bool(selected.any()):
            continue
        count=int(p.counts[kb])
        kslice=slice(kb*LEAF,kb*LEAF+count)
        if quantized:
            z=torch.matmul(qhat,p.ki[:,:,kslice].float().transpose(-1,-2))
            z=z*p.qs[:,:,:n,None]*p.ks[:,:,None,kslice]/math.sqrt(d)
        else:
            # Reconstruct original centered K from a separate optional capture.
            raise ValueError('use dense_reference for unquantized logits')
        if pooled:
            zp=(q*p.kc[:,:,kb,None,:].float()).sum(-1)/math.sqrt(d)+math.log(count)
            z=torch.where(selected[...,None],z,torch.full_like(z,float('-inf')))
            z[...,0]=torch.where(selected,z[...,0],zp)
        else:
            z=z.masked_fill(~selected[...,None],float('-inf'))
        if causal:
            qi=torch.arange(n,device=q.device)[:,None]
            kj=torch.arange(kb*LEAF,kb*LEAF+count,device=q.device)[None,:]
            z=z.masked_fill(kj>qi,float('-inf'))
        hm=z.amax(-1)
        new_m=torch.maximum(m,hm)
        active=torch.isfinite(hm)
        alpha=torch.where(torch.isfinite(m),torch.exp(m-new_m),0.)
        alpha=torch.where(active,alpha,1.)
        prob=torch.where(active[...,None],torch.exp(z-new_m[...,None]),0.)
        v=p.v[:,:,kslice].float()
        if int8_pv:
            pe=torch.where(active[...,None],torch.exp(z-hm[...,None]),0.)
            beta=torch.where(active,torch.exp(hm-new_m),0.)
            pi=(pe*127+0.5).floor().clamp(0,127)
            pv=torch.matmul(pi,p.vi[:,:,kslice].float())*p.vs[:,:,kb,None,:]/127*beta[...,None]
            if pooled:
                pooled_prob=prob[...,0].half().float() if round_p else prob[...,0]
                pooled_pv=pooled_prob[...,None]*p.vc[:,:,kb,None,:].float()
                pv=torch.where(selected[...,None],pv,pooled_pv)
        else:
            operand=prob.half().float() if round_p else prob
            pv=torch.matmul(operand,v)
            if pooled:
                pv=torch.where(selected[...,None],pv,operand[...,0,None]*p.vc[:,:,kb,None,:].float())
        u=u*alpha[...,None]+pv
        l=l*alpha+prob.sum(-1)
        m=new_m
    return (u/l[...,None]).half(),m+torch.log(l)+p.shift
