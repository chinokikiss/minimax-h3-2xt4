"""SM75 second-pass candidates: register scales, cast/pad fusion and fragment layout. No production hook installed.

The attention arithmetic is copied from the frozen optimized Group32 operator;
only thread count / GEMM warp partitioning are added. Group32 scales, leaf64,
FP16 P/V and output rounding remain unchanged. Experimental fused preparation
below preserves FP32 -> FP16 input conversion and rounded division.
"""
from functools import lru_cache
import math
import torch
import torch.nn.functional as F
from patches.tilelang_grouped_tp_optimized import TPGrouped, geometry


@lru_cache(maxsize=32)
def fused_quantizer(b,h,hk,n,nk,d,dtype,qs0,qs1,qs2,ks0,ks1,ks2,vs0,vs1,vs2,query_tile=32):
    import tilelang
    import tilelang.language as T
    nq64=math.ceil(n/64)*64;nq=math.ceil(n/max(64,query_tile))*max(64,query_tile);kp=math.ceil(nk/64)*64
    qt=nq//32;kt=kp//32;vt=kp//32;ng=d//32
    @T.prim_func
    def main(Q:T.StridedTensor((b,h,n,d),(qs0,qs1,qs2,1),dtype),
             K:T.StridedTensor((b,hk,nk,d),(ks0,ks1,ks2,1),dtype),
             V:T.StridedTensor((b,hk,nk,d),(vs0,vs1,vs2,1),dtype),
             Mean:T.Tensor((b,hk,1,d),'float32'),
             QI:T.Tensor((b,h,nq,d),'int8'),KI:T.Tensor((b,hk,kp,d),'int8'),
             VP:T.Tensor((b,hk,kp,d),'float16'),
             QS:T.Tensor((b,h,nq,ng),'float32'),KS:T.Tensor((b,hk,kp,ng),'float32')):
        with T.Kernel(qt*h+(kt+vt)*hk,b,threads=128) as (bx,bz):
            hy=T.if_then_else(bx<qt*h,bx//qt,T.if_then_else(bx<qt*h+kt*hk,(bx-qt*h)//kt,(bx-qt*h-kt*hk)//vt))
            tile=T.if_then_else(bx<qt*h,bx%qt,T.if_then_else(bx<qt*h+kt*hk,(bx-qt*h)%kt,(bx-qt*h-kt*hk)%vt))
            value=T.alloc_fragment((32,32),'float32');absolute=T.alloc_fragment((32,32),'float32')
            maximum=T.alloc_fragment((32,),'float32');scales=T.alloc_fragment((32,),'float32')
            if bx<qt*h:
                for g in T.serial(ng):
                    for i,j in T.Parallel(32,32):
                        value[i,j]=T.if_then_else(tile*32+i<n,T.cast(T.cast(Q[bz,hy,tile*32+i,g*32+j],'float16'),'float32'),0.)
                        absolute[i,j]=T.abs(value[i,j])
                    T.reduce_max(absolute,maximum,dim=1)
                    for i in T.Parallel(32):scales[i]=T.max(maximum[i]*T.float32(1./127.),1e-8)
                    for i,j in T.Parallel(32,32):
                        normal=T.call_pure_extern('float32','__fdiv_rn',value[i,j],scales[i])
                        integer=T.call_pure_extern('int32','__float2int_rn',normal)
                        QI[bz,hy,tile*32+i,g*32+j]=T.cast(T.max(-127,T.min(127,integer)),'int8')
                    for i in T.Parallel(32):QS[bz,hy,tile*32+i,g]=T.if_then_else(tile*32+i<nq64,scales[i],0.)
            else:
                if bx<qt*h+kt*hk:
                    for g in T.serial(ng):
                        for i,j in T.Parallel(32,32):
                            value[i,j]=T.if_then_else(tile*32+i<nk,
                                T.cast(T.cast(K[bz,hy,tile*32+i,g*32+j],'float16'),'float32')-Mean[bz,hy,0,g*32+j],0.)
                            absolute[i,j]=T.abs(value[i,j])
                        T.reduce_max(absolute,maximum,dim=1)
                        for i in T.Parallel(32):scales[i]=T.max(maximum[i]*T.float32(1./127.),1e-8)
                        for i,j in T.Parallel(32,32):
                            normal=T.call_pure_extern('float32','__fdiv_rn',value[i,j],scales[i])
                            integer=T.call_pure_extern('int32','__float2int_rn',normal)
                            KI[bz,hy,tile*32+i,g*32+j]=T.cast(T.max(-127,T.min(127,integer)),'int8')
                        for i in T.Parallel(32):KS[bz,hy,tile*32+i,g]=scales[i]
                else:
                    for i,j in T.Parallel(32,d):
                        VP[bz,hy,tile*32+i,j]=T.if_then_else(tile*32+i<nk,
                            T.cast(V[bz,hy,tile*32+i,j],'float16'),T.float16(0.))
    return tilelang.compile(main,out_idx=[4,5,6,7,8],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=32)
def cast_pad_key(b,h,n,d,dtype,s0,s1,s2):
    import tilelang
    import tilelang.language as T
    padded=math.ceil(n/64)*64
    @T.prim_func
    def main(K:T.StridedTensor((b,h,n,d),(s0,s1,s2,1),dtype),P:T.Tensor((b,h,padded,d),'float32')):
        with T.Kernel(T.ceildiv(padded*d,512),h,b,threads=128) as (bx,hy,bz):
            for i in T.Parallel(512):
                index=bx*512+i
                if index<padded*d:
                    P[bz,hy,index//d,index%d]=T.if_then_else(index//d<n,
                        T.cast(T.cast(K[bz,hy,index//d,index%d],'float16'),'float32'),T.float32(0.))
    return tilelang.compile(main,out_idx=[1],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


def prepare(q,k,v,query_tile=32,mean_mode='eager'):
    """Dense only: identical centering and operand arithmetic, one Q/K/V launch."""
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)!=(7,5):raise ValueError('Actual SM75 required')
    if q.ndim!=4 or k.shape!=v.shape or q.shape[0]!=k.shape[0] or q.shape[-1]!=k.shape[-1]:raise ValueError('BHND Q and matching K/V required')
    if q.dtype not in (torch.float16,torch.float32) or any(x.dtype!=q.dtype or x.device!=q.device or x.stride(-1)!=1 for x in (q,k,v)):
        raise ValueError('Matching FP16/FP32, contiguous final channel required')
    b,h,n,d=q.shape;_,hk,nk,_=k.shape
    if d not in (64,128) or h%hk or query_tile not in (32,64,128):raise ValueError('Unsupported Group32 geometry')
    counts,dummy_half,_,_=geometry(b,h,hk,n,nk,d,str(q.device));kp=math.ceil(nk/64)*64;nb=kp//64
    # Preserve the frozen unweighted leaf mean and its exact eager reduction.
    if mean_mode not in ('eager','castpad'):raise ValueError('Unknown mean preparation')
    padded=(cast_pad_key(b,hk,nk,d,str(k.dtype).removeprefix('torch.'),*k.stride()[:3])(k)
            if mean_mode=='castpad' else F.pad(k.half().float(),(0,0,0,kp-nk)))
    raw=padded.reshape(b,hk,nb,64,d).sum(-2)/counts[None,None,:,None]
    mean=raw.mean(-2,keepdim=True)
    qi,ki,vp,qs,ks=fused_quantizer(b,h,hk,n,nk,d,str(q.dtype).removeprefix('torch.'),
        *q.stride()[:3],*k.stride()[:3],*v.stride()[:3],query_tile)(q,k,v,mean)
    return TPGrouped(dummy_half,qi,ki,vp,qs,ks,dummy_half,dummy_half,counts,n,nk,'dense',query_tile)

@lru_cache(maxsize=64)
def kernel(b,h,hk,n,nk,d,mode='dense',causal=False,query_offset=0,query_tile=32,
           cache='group',layout='BHND',output_dtype='float16',finite_output=False,threads=128,warp_policy='square',row_scores=False,staged_scales=False):
    import tilelang
    import tilelang.language as T
    if mode not in ('dense','drop','scalar') or (mode!='dense' and (h!=hk or n!=nk or causal)):
        raise ValueError('Unsupported attention contract')
    if query_tile not in (32,64,128) or (mode!='dense' and query_tile==128):raise ValueError('Routing leaf64 must not be crossed')
    if cache not in ('group','q','both') or layout not in ('BHND','BND') or output_dtype not in ('float16','float32'):raise ValueError('Unknown kernel option')
    if finite_output and output_dtype!='float16':raise ValueError('Finite-output fusion matches the FP16 Video boundary only')
    if threads not in (128,256) or warp_policy not in ('square','row'): raise ValueError('Unknown warp configuration')
    gp=T.GemmWarpPolicy.FullRow if warp_policy=='row' else T.GemmWarpPolicy.Square
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
        with T.Kernel(T.ceildiv(n,bm),h,b,threads=threads) as (bx,hy,bz):
            kh=hy//(h//hk)
            q8=T.alloc_shared((bm,d if cq else 32),'int8')
            k8=T.alloc_shared((64,d if ck else 32),'int8')
            q16=T.alloc_shared((bm,d),'float16')
            qscale=T.alloc_shared((bm,ng),'float32');kscale=T.alloc_shared((64,ng),'float32')
            v16=T.alloc_shared((64,d),'float16');p16=T.alloc_shared((bm,64),'float16')
            dot=T.alloc_fragment((bm,64),'int32');scores=T.alloc_fragment((bm,64),'float32')
            pv=T.alloc_fragment((bm,d),'float32');out=T.alloc_fragment((bm,d),'float32')
            if row_scores:
                row_fragment=T.alloc_fragment((bm,64),'float32')
                row_max=T.alloc_fragment((bm,),'float32');row_sum=T.alloc_fragment((bm,),'float32')
                exchange=T.alloc_shared((bm,64),'float32');row_exchange=T.alloc_shared((bm,),'float32')
                T.annotate_layout({row_fragment:T.Fragment((bm,64),
                    forward_thread_fn=lambda i,j:(i%(threads//32))*32+j%32,
                    forward_index_fn=lambda i,j:(i//(threads//32))*2+j//32)})
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
                    if staged_scales and not ck:T.copy(KS[bz,kh,kb*64:(kb+1)*64,:],kscale)
                    if ck:
                        T.copy(KI[bz,kh,kb*64:(kb+1)*64,:],k8)
                        T.copy(KS[bz,kh,kb*64:(kb+1)*64,:],kscale)
                    for g in T.serial(ng):
                        T.clear(dot)
                        if ck:
                            T.gemm(q8[0:bm,g*32:(g+1)*32],k8[0:64,g*32:(g+1)*32],dot,transpose_B=True,policy=gp)
                            for i,j in T.Parallel(bm,64):scores[i,j]+=T.cast(dot[i,j],'float32')*qscale[i,g]*kscale[j,g]*scale
                        else:
                            if not cq:T.copy(QI[bz,hy,bx*bm:(bx+1)*bm,g*32:(g+1)*32],q8)
                            T.copy(KI[bz,kh,kb*64:(kb+1)*64,g*32:(g+1)*32],k8)
                            if cq:T.gemm(q8[0:bm,g*32:(g+1)*32],k8,dot,transpose_B=True,policy=gp)
                            else:T.gemm(q8,k8,dot,transpose_B=True,policy=gp)
                            for i,j in T.Parallel(bm,64):scores[i,j]+=T.cast(dot[i,j],'float32')*(qscale[i,g] if cq else QS[bz,hy,bx*bm+i,g])*(kscale[j,g] if staged_scales else KS[bz,kh,kb*64+j,g])*scale
                    for i,j in T.Parallel(bm,64):
                        scores[i,j]=T.if_then_else((kb*64+j<nk)&((kb*64+j<=query_offset+bx*bm+i) if causal else True),scores[i,j],-T.infinity('float32'))
                    T.copy(m,old)
                    if row_scores:
                        T.copy(scores,exchange);T.copy(exchange,row_fragment)
                        T.reduce_max(row_fragment,row_max,dim=1)
                        T.copy(row_max,row_exchange);T.copy(row_exchange,hm)
                    else:T.reduce_max(scores,hm,dim=1)
                    for i in T.Parallel(bm):
                        m[i]=T.max(old[i],hm[i]);alpha[i]=T.exp2(old[i]-m[i])
                    for i,j in T.Parallel(bm,64):
                        scores[i,j]=T.exp2(scores[i,j]-m[i]);p16[i,j]=T.cast(scores[i,j],'float16')
                    if row_scores:
                        T.copy(scores,exchange);T.copy(exchange,row_fragment)
                        T.reduce_sum(row_fragment,row_sum,dim=1)
                        T.copy(row_sum,row_exchange);T.copy(row_exchange,psum)
                    else:T.reduce_sum(scores,psum,dim=1)
                    T.copy(V[bz,kh,kb*64:(kb+1)*64,:],v16);T.clear(pv);T.gemm(p16,v16,pv,policy=gp)
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


def execute(p,r=None,causal=False,query_offset=0,cache='group',layout='BHND',output_dtype=torch.float16,finite_output=False,threads=128,warp_policy='square',row_scores=False,staged_scales=False):
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
    out=kernel(b,h,hk,p.queries,p.keys,d,mode,causal,query_offset,p.query_tile,cache,layout,dtype,finite_output,threads,warp_policy,row_scores,staged_scales)(
        p.q if scalar else dummy_half,p.qi,p.ki,p.v,p.qs,p.ks,
        p.kc if scalar else dummy_half,p.vc if scalar else dummy_half,p.counts if scalar else dummy_int,r)
    return out if layout=='BND' else out[:,:,:p.queries]
