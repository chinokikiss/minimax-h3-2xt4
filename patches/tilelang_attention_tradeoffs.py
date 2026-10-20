"""Opt-in SM75 Group32-QK / INT8-PV and order-preserving compact-drop candidates.

INT8 PV is a new numerical mode. Compact drop preserves sorted leaf order.
No production hook or sparse-policy changes.
"""
from functools import lru_cache
import math
import torch
import torch.nn.functional as F
from patches.tilelang_grouped_tp_optimized import TPGrouped, geometry


@lru_cache(maxsize=64)
def kernel(b,h,hk,n,nk,d,mode='dense',causal=False,query_offset=0,query_tile=32,
           cache='group',layout='BHND',output_dtype='float16',finite_output=False,threads=128,warp_policy='square',pv_int8=False,compact_drop=False):
    import tilelang
    import tilelang.language as T
    if mode not in ('dense','drop','scalar') or (mode!='dense' and (h!=hk or n!=nk or causal)):
        raise ValueError('Unsupported attention contract')
    if query_tile not in (32,64,128) or (mode!='dense' and query_tile==128):raise ValueError('Routing leaf64 must not be crossed')
    if cache not in ('group','q','both') or layout not in ('BHND','BND') or output_dtype not in ('float16','float32'):raise ValueError('Unknown kernel option')
    if finite_output and output_dtype!='float16':raise ValueError('Finite-output fusion matches the FP16 Video boundary only')
    if threads not in (128,256) or warp_policy not in ('square','row'): raise ValueError('Unknown warp configuration')
    if compact_drop and mode!='drop':raise ValueError('Compact list is drop only')
    gp=T.GemmWarpPolicy.FullRow if warp_policy=='row' else T.GemmWarpPolicy.Square
    bm=query_tile;cq=cache!='group';ck=cache=='both'
    np=math.ceil(n/max(64,bm))*max(64,bm); kp=math.ceil(nk/64)*64; nb=kp//64; ng=d//32
    q_shape=(b,h,np,d) if mode=='scalar' else (1,)
    centroid_shape=(b,hk,nb,d) if mode=='scalar' else (1,)
    count_shape=(nb,) if mode=='scalar' else (1,)
    route_shape=(b,h,math.ceil(n/64),nb) if mode!='dense' else (1,)
    output_shape=(b,h,np,d) if layout=='BHND' else (b,n,h*d)
    vtype='int8' if pv_int8 else 'float16'; pvtype='int32' if pv_int8 else 'float32'
    scale=d**-.5*math.log2(math.e)
    @T.prim_func
    def main(Q:T.Tensor(q_shape,'float16'), QI:T.Tensor((b,h,np,d),'int8'),
             KI:T.Tensor((b,hk,kp,d),'int8'), V:T.Tensor((b,hk,kp,d),vtype),
             QS:T.Tensor((b,h,np,ng),'float32'),KS:T.Tensor((b,hk,kp,ng),'float32'),
             KC:T.Tensor(centroid_shape,'float16'),VC:T.Tensor(centroid_shape,'float16'),
             Counts:T.Tensor(count_shape,'int32'),Routes:T.Tensor(route_shape,'int32'),
             VS:T.Tensor((b,hk,nb,d),'float32'),Lengths:T.Tensor((b,h,math.ceil(n/64)),'int32'),
             O:T.Tensor(output_shape,output_dtype)):
        with T.Kernel(T.ceildiv(n,bm),h,b,threads=threads) as (bx,hy,bz):
            kh=hy//(h//hk)
            q8=T.alloc_shared((bm,d if cq else 32),'int8')
            k8=T.alloc_shared((64,d if ck else 32),'int8')
            q16=T.alloc_shared((bm,d),'float16')
            qscale=T.alloc_shared((bm,ng),'float32');kscale=T.alloc_shared((64,ng),'float32')
            v16=T.alloc_shared((64,d),vtype);p16=T.alloc_shared((bm,64),vtype)
            dot=T.alloc_fragment((bm,64),'int32');scores=T.alloc_fragment((bm,64),'float32')
            pv=T.alloc_fragment((bm,d),pvtype);out=T.alloc_fragment((bm,d),'float32')
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
            if compact_drop:loop=Lengths[bz,hy,(bx*bm)//64]
            for entry in T.serial(loop):
                kb=Routes[bz,hy,(bx*bm)//64,entry] if compact_drop else entry
                if compact_drop or mode=='dense' or Routes[bz,hy,(bx*bm)//64,kb]!=0:
                    T.clear(scores)
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
                            for i,j in T.Parallel(bm,64):scores[i,j]+=T.cast(dot[i,j],'float32')*(qscale[i,g] if cq else QS[bz,hy,bx*bm+i,g])*KS[bz,kh,kb*64+j,g]*scale
                    for i,j in T.Parallel(bm,64):
                        scores[i,j]=T.if_then_else((kb*64+j<nk)&((kb*64+j<=query_offset+bx*bm+i) if causal else True),scores[i,j],-T.infinity('float32'))
                    T.copy(m,old);T.reduce_max(scores,hm,dim=1)
                    for i in T.Parallel(bm):
                        m[i]=T.max(old[i],hm[i]);alpha[i]=T.exp2(old[i]-m[i])
                    if pv_int8:
                        # Block-local normalization preserves P's resolution even when
                        # this block is far below the running online maximum.
                        for i in T.Parallel(bm):beta[i]=T.exp2(hm[i]-m[i])
                        for i,j in T.Parallel(bm,64):
                            scores[i,j]=T.exp2(scores[i,j]-hm[i])
                            rounded=T.call_pure_extern('int32','__float2int_rn',scores[i,j]*T.float32(127.))
                            p16[i,j]=T.cast(T.max(0,T.min(127,rounded)),'int8')
                    else:
                        for i,j in T.Parallel(bm,64):
                            scores[i,j]=T.exp2(scores[i,j]-m[i]);p16[i,j]=T.cast(scores[i,j],'float16')
                    T.reduce_sum(scores,psum,dim=1)
                    T.copy(V[bz,kh,kb*64:(kb+1)*64,:],v16);T.clear(pv);T.gemm(p16,v16,pv,policy=gp)
                    if pv_int8:
                        for i,x in T.Parallel(bm,d):out[i,x]=out[i,x]*alpha[i]+T.cast(pv[i,x],'float32')*VS[bz,kh,kb,x]*T.float32(1./127.)*beta[i]
                        for i in T.Parallel(bm):denom[i]=denom[i]*alpha[i]+psum[i]*beta[i]
                    else:
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
    return tilelang.compile(main,out_idx=[12],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=32)
def value_quantizer(b,h,kp,d):
    import tilelang
    import tilelang.language as T
    @T.prim_func
    def main(V:T.Tensor((b,h,kp,d),'float16'), VI:T.Tensor((b,h,kp,d),'int8'),
             S:T.Tensor((b,h,kp//64,d),'float32')):
        with T.Kernel(kp//64,h,b,threads=128) as (kb,hy,bz):
            value=T.alloc_fragment((64,d),'float32');absolute=T.alloc_fragment((64,d),'float32')
            maximum=T.alloc_fragment((d,),'float32')
            for i,j in T.Parallel(64,d):
                value[i,j]=T.cast(V[bz,hy,kb*64+i,j],'float32');absolute[i,j]=T.abs(value[i,j])
            T.reduce_max(absolute,maximum,dim=0)
            for j in T.Parallel(d):maximum[j]=T.max(maximum[j]*T.float32(1./127.),T.float32(1e-8))
            for i,j in T.Parallel(64,d):
                z=T.call_pure_extern('float32','__fdiv_rn',value[i,j],maximum[j])
                q=T.call_pure_extern('int32','__float2int_rn',z)
                VI[bz,hy,kb*64+i,j]=T.cast(T.max(-127,T.min(127,q)),'int8')
            for j in T.Parallel(d):S[bz,hy,kb,j]=maximum[j]
    return tilelang.compile(main,out_idx=[1,2],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=16)
def route_compactor(b,h,nq,nk):
    import tilelang
    import tilelang.language as T
    padded=1 << (nk-1).bit_length()
    @T.prim_func
    def main(R:T.Tensor((b,h,nq,nk),'int32'), I:T.Tensor((b,h,nq,nk),'int32'),
             L:T.Tensor((b,h,nq),'int32')):
        with T.Kernel(nq,h,b,threads=128) as (qx,hy,bz):
            prefix=T.alloc_shared((padded,),'int32')
            for i in T.Parallel(padded):prefix[i]=T.if_then_else(i<nk,T.cast(R[bz,hy,qx,i]!=0,'int32'),0)
            T.cumsum(prefix,dim=0)
            for i in T.Parallel(nk):I[bz,hy,qx,i]=nk
            T.sync_threads()
            for i in T.Parallel(nk):
                if R[bz,hy,qx,i]!=0:I[bz,hy,qx,prefix[i]-1]=i
            if T.get_thread_binding()==0:L[bz,hy,qx]=prefix[padded-1]
    return tilelang.compile(main,out_idx=[1,2],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=32)
def dummy(b,h,hk,n,nk,d,device):
    return (torch.empty((b,hk,math.ceil(nk/64),d),device=device),
            torch.empty((b,h,math.ceil(n/64)),device=device,dtype=torch.int32))


def execute(p,r=None,*,causal=False,query_offset=0,cache='group',output_dtype=torch.float16,
            threads=128,warp_policy='square',pv_int8=False,compact_drop=False,value=None,scales=None,lengths=None):
    if torch.cuda.get_device_capability(p.qi.device)!=(7,5):raise ValueError('Physical SM75 required')
    b,h,_,d=p.qi.shape;hk=p.ki.shape[1]
    _,dh,di,_=geometry(b,h,hk,p.queries,p.keys,d,str(p.qi.device))
    ds,dl=dummy(b,h,hk,p.queries,p.keys,d,str(p.qi.device))
    if pv_int8 and (value is None or scales is None):raise ValueError('INT8 V and block/channel scales required')
    if compact_drop and (r is None or lengths is None):raise ValueError('Sorted indices and lengths required')
    scalar=p.mode=='scalar';dtype=str(output_dtype).removeprefix('torch.')
    return kernel(b,h,hk,p.queries,p.keys,d,p.mode,causal,query_offset,p.query_tile,cache,'BND',dtype,False,
                  threads,warp_policy,pv_int8,compact_drop)(p.q if scalar else dh,p.qi,p.ki,value if pv_int8 else p.v,
                  p.qs,p.ks,p.kc if scalar else dh,p.vc if scalar else dh,p.counts if scalar else di,
                  r if r is not None else di,scales if pv_int8 else ds,lengths if compact_drop else dl)
