"""SM75 pooled attention: scalar centroid QK and a value outer product.

Preserves the original centroid/mean approximation while removing padded GEMMs
and 64-column softmax work for omitted leaves. FP16 PV only; no causal pooling.
"""
from functools import lru_cache
import math
import torch


@lru_cache(maxsize=32)
def scalar_pool_kernel(b,h,n,d):
    import tilelang
    import tilelang.language as T
    np=((n+63)//64)*64;nb=np//64;scale=d**-.5*math.log2(math.e)
    @T.prim_func
    def main(Q:T.Tensor((b,h,np,d),'float16'),QI:T.Tensor((b,h,np,d),'int8'),
             KI:T.Tensor((b,h,np,d),'int8'),V:T.Tensor((b,h,np,d),'float16'),
             QS:T.Tensor((b,h,np),'float32'),KS:T.Tensor((b,h,np),'float32'),
             KC:T.Tensor((b,h,nb,d),'float16'),VC:T.Tensor((b,h,nb,d),'float16'),
             Counts:T.Tensor((nb,),'int32'),Routes:T.Tensor((b,h,nb,nb),'int32'),
             O:T.Tensor((b,h,np,d),'float16'),LSE:T.Tensor((b,h,np),'float32')):
        with T.Kernel(np//32,h,b,threads=128) as (bx,hy,bz):
            q8=T.alloc_shared((32,d),'int8');q16=T.alloc_shared((32,d),'float16')
            k8=T.alloc_shared((64,d),'int8');v16=T.alloc_shared((64,d),'float16')
            p16=T.alloc_shared((32,64),'float16')
            dot=T.alloc_fragment((32,64),'int32');scores=T.alloc_fragment((32,64),'float32')
            pv=T.alloc_fragment((32,d),'float32');out=T.alloc_fragment((32,d),'float32')
            products=T.alloc_fragment((32,d),'float32');summary=T.alloc_fragment((32,),'float32')
            summary_shared=T.alloc_shared((32,),'float32');pooled_p=T.alloc_fragment((32,),'float16')
            m=T.alloc_fragment((32,),'float32');hm=T.alloc_fragment((32,),'float32')
            old=T.alloc_fragment((32,),'float32');alpha=T.alloc_fragment((32,),'float32')
            beta=T.alloc_fragment((32,),'float32');denom=T.alloc_fragment((32,),'float32')
            psum=T.alloc_fragment((32,),'float32')
            T.copy(QI[bz,hy,bx*32:(bx+1)*32,:],q8);T.copy(Q[bz,hy,bx*32:(bx+1)*32,:],q16)
            T.clear(out);T.clear(denom);T.fill(m,-T.infinity('float32'))
            for kb in T.serial(nb):
                if Routes[bz,hy,bx//2,kb]!=0:
                    T.copy(KI[bz,hy,kb*64:(kb+1)*64,:],k8);T.clear(dot)
                    T.gemm(q8,k8,dot,transpose_B=True)
                    for i,j in T.Parallel(32,64):
                        scores[i,j]=T.if_then_else(kb*64+j<n,T.cast(dot[i,j],'float32')*QS[bz,hy,bx*32+i]*KS[bz,hy,kb*64+j]*scale,-T.infinity('float32'))
                    T.copy(m,old);T.reduce_max(scores,hm,dim=1)
                    for i in T.Parallel(32):
                        m[i]=T.max(old[i],hm[i]);alpha[i]=T.exp2(old[i]-m[i])
                    for i,j in T.Parallel(32,64):
                        scores[i,j]=T.exp2(scores[i,j]-m[i]);p16[i,j]=T.cast(scores[i,j],'float16')
                    T.reduce_sum(scores,psum,dim=1)
                    T.copy(V[bz,hy,kb*64:(kb+1)*64,:],v16);T.clear(pv);T.gemm(p16,v16,pv)
                    for i,x in T.Parallel(32,d):out[i,x]=out[i,x]*alpha[i]+pv[i,x]
                    for i in T.Parallel(32):denom[i]=denom[i]*alpha[i]+psum[i]
                else:
                    for i,x in T.Parallel(32,d):
                        products[i,x]=T.cast(q16[i,x],'float32')*T.cast(KC[bz,hy,kb,x],'float32')
                    T.reduce_sum(products,summary,dim=1);T.copy(summary,summary_shared)
                    for i in T.Parallel(32):
                        old[i]=m[i]
                        hm[i]=summary_shared[i]*scale+T.log2(T.cast(Counts[kb],'float32'))
                        m[i]=T.max(old[i],hm[i]);alpha[i]=T.exp2(old[i]-m[i]);beta[i]=T.exp2(hm[i]-m[i])
                        pooled_p[i]=T.cast(beta[i],'float16')
                        denom[i]=denom[i]*alpha[i]+beta[i]
                    for i,x in T.Parallel(32,d):
                        out[i,x]=out[i,x]*alpha[i]+T.cast(pooled_p[i],'float32')*T.cast(VC[bz,hy,kb,x],'float32')
            for i,x in T.Parallel(32,d):O[bz,hy,bx*32+i,x]=T.cast(out[i,x]/denom[i],'float16')
            for i in T.Parallel(32):LSE[bz,hy,bx*32+i]=(m[i]+T.log2(denom[i]))*math.log(2)
    return tilelang.compile(main,out_idx=[10,11],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


def execute_scalar_pool(p,routes):
    if not p.q.is_cuda or torch.cuda.get_device_capability(p.q.device)!=(7,5):raise ValueError('Actual SM75 required')
    b,h,_,d=p.q.shape
    if routes.shape!=(b,h,p.counts.numel(),p.counts.numel()) or routes.dtype!=torch.int32 or routes.device!=p.q.device:
        raise ValueError('Int32 BHLL routes on input device required')
    out,lse=scalar_pool_kernel(b,h,p.tokens,d)(p.q,p.qi,p.ki,p.v,p.qs,p.ks,p.kc,p.vc,p.counts,routes)
    return out[:,:,:p.tokens],lse[:,:,:p.tokens]+p.shift
