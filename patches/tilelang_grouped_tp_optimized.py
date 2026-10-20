"""Offline optimization candidates; frozen T4 baseline remains in grouped_tp.

No route policy/precision changes. All candidates require T4 execution gates
before deployment. CPU equivalence/static compilation are not performance data.
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
    mode: str
    query_tile: int


@lru_cache(maxsize=32)
def _geometry_cached(b,h,hk,n,nk,d,device,stream_id):
    counts=(nk-torch.arange(math.ceil(nk/64),device=device)*64).clamp(0,64).int()
    return (counts,torch.empty((1,),device=device,dtype=torch.float16),
            torch.empty((1,),device=device,dtype=torch.int32),
            torch.empty((b,h,1,d),device=device,dtype=torch.float32))


def geometry(b,h,hk,n,nk,d,device):
    # Counts are initialized asynchronously on CUDA. Do not share a newly
    # initialized tensor with an unrelated concurrent request stream.
    stream_id=torch.cuda.current_stream(device).cuda_stream if torch.device(device).type=='cuda' else None
    return _geometry_cached(b,h,hk,n,nk,d,device,stream_id)


def prepare(q, k, v, fused=False, mode='dense',query_tile=32):
    if mode not in ('dense','drop','scalar'):raise ValueError('Unknown attention mode')
    if query_tile not in (32,64,128) or (mode!='dense' and query_tile==128):
        raise ValueError('Sparse query tiles must fit within one routing leaf64')
    sparse=mode!='dense'
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError('BHND Q and matching K/V required')
    b,h,n,d=q.shape; bk,hk,nk,dk=k.shape
    if b!=bk or d!=dk or d not in (64,128) or h%hk or min(n,nk)<1:
        raise ValueError('D64/128, positive lengths and divisible GQA heads required')
    if any(x.dtype!=torch.float16 or x.device!=q.device for x in (q,k,v)):
        raise ValueError('FP16 operands on one device required')
    if sparse and (q.shape!=k.shape):
        raise ValueError('Sparse modes currently require DiT self-attention')
    alignment=max(64,query_tile)
    nqpad=math.ceil(n/alignment)*alignment; nkpad=math.ceil(nk/64)*64; nb=nkpad//64
    counts,dummy_half,_,dummy_mean=geometry(b,h,hk,n,nk,d,str(q.device))
    raw=F.pad(k.float(),(0,0,0,nkpad-nk)).reshape(b,hk,nb,64,d).sum(-2)/counts[None,None,:,None]
    mean=raw.mean(-2,keepdim=True)  # unweighted leaf mean, including ragged leaf
    if fused:
        if not q.is_cuda or torch.cuda.get_device_capability(q.device)!=(7,5):
            raise ValueError('Actual SM75 required for fused preparation')
        from patches.tilelang_grouped_fused import quantizer
        qi,qs=quantizer(b,h,n,d,32,False)(q.contiguous(),dummy_mean)
        ki,ks=quantizer(b,hk,nk,d,32,True)(k.contiguous(),mean)
        extra=nqpad-qi.shape[2]
        if extra:
            qi=F.pad(qi,(0,0,0,extra));qs=F.pad(qs,(0,0,0,extra))
    else:
        def quant(x,heads,length,padded):
            a=x.reshape(b,heads,length,d//32,32)
            s=(a.abs().amax(-1)/127).clamp_min(1e-8)
            i=(a/s[...,None]).round().clamp(-127,127).to(torch.int8).reshape(b,heads,length,d)
            return F.pad(i,(0,0,0,padded-length)).contiguous(),F.pad(s,(0,0,0,padded-length)).contiguous()
        qi,qs=quant(q.float(),h,n,nqpad)
        ki,ks=quant(k.float()-mean,hk,nk,nkpad)
    # Dense kernels never access raw Q or the summaries. Do not compute them.
    qp=F.pad(q,(0,0,0,nqpad-n)).contiguous() if sparse else dummy_half
    kc=(raw-mean).half().contiguous() if sparse else dummy_half
    vc=(F.pad(v.float(),(0,0,0,nkpad-nk)).reshape(b,hk,nb,64,d).sum(-2)/counts[None,None,:,None]).half().contiguous() if mode=='scalar' else dummy_half
    return TPGrouped(qp,qi,ki,F.pad(v,(0,0,0,nkpad-nk)).contiguous(),qs,ks,kc,vc,counts,n,nk,mode,query_tile)


def routes(p,prefix_tokens=874,fast_tau0=False):
    if p.mode=='dense':raise ValueError('Prepare a sparse mode before routing')
    if not fast_tau0:
        from patches.tilelang_grouped_tp import routes as original_routes
        return original_routes(p,prefix_tokens)
    b,h,np,d=p.q.shape;nb=np//64
    qc=p.q.float().reshape(b,h,nb,64,d).sum(-2)/p.counts[None,None,:,None]
    kc=p.kc.float();scores=qc@kc.transpose(-1,-2)/math.sqrt(d)
    # Tau is fixed to zero in the current policy. Retain mean/threshold math;
    # omit variance, square and sqrt operations multiplied by zero. FP16 input
    # centroids must be finite for this opt-in shortcut. Exceptional inputs
    # retain the original path by default, including its NaN propagation.
    threshold=(qc*kc.mean(-2)[...,None,:]).sum(-1)/math.sqrt(d)
    ids=torch.arange(nb,device=p.q.device)
    mandatory=((ids[:,None]-ids[None,:]).abs()<=1)|(ids[None,:]<1)
    r=((scores>threshold[...,None])|mandatory[None,None]).to(torch.int32).contiguous()
    prefix=math.ceil(prefix_tokens/64)
    r[:,:,:prefix,:]=1; r[:,:,:,:prefix]=1
    return r


@lru_cache(maxsize=64)
def kernel(b,h,hk,n,nk,d,mode='dense',causal=False,query_offset=0,query_tile=32,
           cache='group',layout='BHND',output_dtype='float16',finite_output=False):
    import tilelang
    import tilelang.language as T
    if mode not in ('dense','drop','scalar') or (mode!='dense' and (h!=hk or n!=nk or causal)):
        raise ValueError('Unsupported attention contract')
    if query_tile not in (32,64,128) or (mode!='dense' and query_tile==128):raise ValueError('Routing leaf64 must not be crossed')
    if cache not in ('group','q','both') or layout not in ('BHND','BND') or output_dtype not in ('float16','float32'):raise ValueError('Unknown kernel option')
    if finite_output and output_dtype!='float16':raise ValueError('Finite-output fusion matches the FP16 Video boundary only')
    bm=query_tile;cq=cache!='group';ck=cache=='both'
    np=math.ceil(n/max(64,bm))*max(64,bm); kp=math.ceil(nk/64)*64; nb=kp//64; ng=d//32
    q_shape=(b,h,np,d) if mode=='scalar' else (1,)
    centroid_shape=(b,hk,nb,d) if mode=='scalar' else (1,)
    count_shape=(nb,) if mode=='scalar' else (1,)
    route_shape=(b,h,math.ceil(n/64),nb) if mode!='dense' else (1,)
    output_shape=(b,h,np,d) if layout=='BHND' else (b,n,h*d)
    scale=d**-.5*math.log2(math.e)
    @T.prim_func
    def main(Q:T.Tensor(q_shape,'float16'), QI:T.Tensor((b,h,np,d),'int8'),
             KI:T.Tensor((b,hk,kp,d),'int8'), V:T.Tensor((b,hk,kp,d),'float16'),
             QS:T.Tensor((b,h,np,ng),'float32'),KS:T.Tensor((b,hk,kp,ng),'float32'),
             KC:T.Tensor(centroid_shape,'float16'),VC:T.Tensor(centroid_shape,'float16'),
             Counts:T.Tensor(count_shape,'int32'),Routes:T.Tensor(route_shape,'int32'),
             O:T.Tensor(output_shape,output_dtype)):
        with T.Kernel(T.ceildiv(n,bm),h,b,threads=128) as (bx,hy,bz):
            kh=hy//(h//hk)
            q8=T.alloc_shared((bm,d if cq else 32),'int8')
            k8=T.alloc_shared((64,d if ck else 32),'int8')
            q16=T.alloc_shared((bm,d),'float16')
            qscale=T.alloc_shared((bm,ng),'float32');kscale=T.alloc_shared((64,ng),'float32')
            v16=T.alloc_shared((64,d),'float16');p16=T.alloc_shared((bm,64),'float16')
            dot=T.alloc_fragment((bm,64),'int32');scores=T.alloc_fragment((bm,64),'float32')
            pv=T.alloc_fragment((bm,d),'float32');out=T.alloc_fragment((bm,d),'float32')
            m=T.alloc_fragment((bm,),'float32');hm=T.alloc_fragment((bm,),'float32')
            old=T.alloc_fragment((bm,),'float32');alpha=T.alloc_fragment((bm,),'float32')
            denom=T.alloc_fragment((bm,),'float32');psum=T.alloc_fragment((bm,),'float32')
            products=T.alloc_fragment((bm,d),'float32');summary=T.alloc_fragment((bm,),'float32')
            summary_shared=T.alloc_shared((bm,),'float32');beta=T.alloc_fragment((bm,),'float32')
            pooled_p=T.alloc_fragment((bm,),'float16')
            if cq:
                T.copy(QI[bz,hy,bx*bm:(bx+1)*bm,:],q8)
                T.copy(QS[bz,hy,bx*bm:(bx+1)*bm,:],qscale)
            if mode=='scalar':T.copy(Q[bz,hy,bx*bm:(bx+1)*bm,:],q16)
            T.clear(out);T.clear(denom);T.fill(m,-T.infinity('float32'))
            loop=T.min(nb,T.ceildiv(query_offset+(bx+1)*bm,64)) if causal else nb
            for kb in T.serial(loop):
                if mode=='dense' or Routes[bz,hy,(bx*bm)//64,kb]!=0:
                    T.clear(scores)
                    if ck:
                        T.copy(KI[bz,kh,kb*64:(kb+1)*64,:],k8)
                        T.copy(KS[bz,kh,kb*64:(kb+1)*64,:],kscale)
                    for g in T.serial(ng):
                        T.clear(dot)
                        if ck:
                            T.gemm(q8[0:bm,g*32:(g+1)*32],k8[0:64,g*32:(g+1)*32],dot,transpose_B=True)
                            for i,j in T.Parallel(bm,64):scores[i,j]+=T.cast(dot[i,j],'float32')*qscale[i,g]*kscale[j,g]*scale
                        else:
                            if not cq:T.copy(QI[bz,hy,bx*bm:(bx+1)*bm,g*32:(g+1)*32],q8)
                            T.copy(KI[bz,kh,kb*64:(kb+1)*64,g*32:(g+1)*32],k8)
                            if cq:T.gemm(q8[0:bm,g*32:(g+1)*32],k8,dot,transpose_B=True)
                            else:T.gemm(q8,k8,dot,transpose_B=True)
                            for i,j in T.Parallel(bm,64):scores[i,j]+=T.cast(dot[i,j],'float32')*(qscale[i,g] if cq else QS[bz,hy,bx*bm+i,g])*KS[bz,kh,kb*64+j,g]*scale
                    for i,j in T.Parallel(bm,64):
                        scores[i,j]=T.if_then_else((kb*64+j<nk)&((kb*64+j<=query_offset+bx*bm+i) if causal else True),scores[i,j],-T.infinity('float32'))
                    T.copy(m,old);T.reduce_max(scores,hm,dim=1)
                    for i in T.Parallel(bm):
                        m[i]=T.max(old[i],hm[i]);alpha[i]=T.exp2(old[i]-m[i])
                    for i,j in T.Parallel(bm,64):
                        scores[i,j]=T.exp2(scores[i,j]-m[i]);p16[i,j]=T.cast(scores[i,j],'float16')
                    T.reduce_sum(scores,psum,dim=1)
                    T.copy(V[bz,kh,kb*64:(kb+1)*64,:],v16);T.clear(pv);T.gemm(p16,v16,pv)
                    for i,x in T.Parallel(bm,d):out[i,x]=out[i,x]*alpha[i]+pv[i,x]
                    for i in T.Parallel(bm):denom[i]=denom[i]*alpha[i]+psum[i]
                elif mode=='scalar':
                    for i,x in T.Parallel(bm,d):products[i,x]=T.cast(q16[i,x],'float32')*T.cast(KC[bz,kh,kb,x],'float32')
                    T.reduce_sum(products,summary,dim=1);T.copy(summary,summary_shared)
                    for i in T.Parallel(bm):
                        old[i]=m[i];hm[i]=summary_shared[i]*scale+T.log2(T.cast(Counts[kb],'float32'))
                        m[i]=T.max(old[i],hm[i]);alpha[i]=T.exp2(old[i]-m[i]);beta[i]=T.exp2(hm[i]-m[i])
                        pooled_p[i]=T.cast(beta[i],'float16');denom[i]=denom[i]*alpha[i]+beta[i]
                    for i,x in T.Parallel(bm,d):out[i,x]=out[i,x]*alpha[i]+T.cast(pooled_p[i],'float32')*T.cast(VC[bz,kh,kb,x],'float32')
            for i,x in T.Parallel(bm,d):
                value=T.cast(T.cast(out[i,x]/denom[i],'float16'),'float32')
                stored=T.if_then_else(T.isnan(value),0.,T.min(T.max(value,-65504.),65504.)) if finite_output else value
                if layout=='BND':
                    if bx*bm+i<n:O[bz,bx*bm+i,hy*d+x]=T.cast(stored,output_dtype)
                else:O[bz,hy,bx*bm+i,x]=T.cast(stored,output_dtype)
    return tilelang.compile(main,out_idx=[10],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


def execute(p,r=None,causal=False,query_offset=0,cache='group',layout='BHND',output_dtype=torch.float16,finite_output=False):
    if not p.qi.is_cuda or torch.cuda.get_device_capability(p.qi.device)!=(7,5):
        raise ValueError('Actual SM75 execution required')
    b,h,_,d=p.qi.shape;hk=p.ki.shape[1];mode=p.mode
    _,dummy_half,dummy_int,_=geometry(b,h,hk,p.queries,p.keys,d,str(p.qi.device))
    if r is None:
        if mode!='dense':raise ValueError('Sparse routes required')
        r=dummy_int
    dtype={torch.float16:'float16',torch.float32:'float32'}.get(output_dtype)
    if dtype is None:raise ValueError('FP16/FP32 output required')
    scalar=mode=='scalar'
    out=kernel(b,h,hk,p.queries,p.keys,d,mode,causal,query_offset,p.query_tile,cache,layout,dtype,finite_output)(
        p.q if scalar else dummy_half,p.qi,p.ki,p.v,p.qs,p.ks,
        p.kc if scalar else dummy_half,p.vc if scalar else dummy_half,p.counts if scalar else dummy_int,r)
    return out if layout=='BND' else out[:,:,:p.queries]


def attention(q,k,v,mode='dense',prefix_tokens=874,query_tile=32,cache='group',
              causal=False,query_offset=0,fast_routing=False,finite_output=False):
    """Native BND boundary, rounding through FP16 even for FP32 consumers."""
    if q.dtype not in (torch.float16,torch.float32) or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise ValueError('Matching FP16 or FP32 boundary required')
    p=prepare(q.half(),k.half(),v.half(),fused=True,mode=mode,query_tile=query_tile)
    r=routes(p,prefix_tokens,fast_routing) if mode!='dense' else None
    return execute(p,r,causal,query_offset,cache,layout='BND',output_dtype=q.dtype,finite_output=finite_output)


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
