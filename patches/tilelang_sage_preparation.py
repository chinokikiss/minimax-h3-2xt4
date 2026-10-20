"""Fused INT8 preparation with exactly the stage-2 Sage32 approximation."""
from functools import lru_cache
import torch
import torch.nn.functional as F
from patches.tilelang_sparse_attention import Prepared
from patches.tilelang_dense_preparation import geometry


@lru_cache(maxsize=32)
def quant_kernel(b,h,n,d,block,center):
    import tilelang
    import tilelang.language as T
    np=((n+63)//64)*64
    @T.prim_func
    def main(X:T.Tensor((b,h,n,d),'float16'),Mean:T.Tensor((b,h,1,d),'float32'),
             QI:T.Tensor((b,h,np,d),'int8'),Scale:T.Tensor((b,h,np),'float32')):
        with T.Kernel(T.ceildiv(np,block),h,b,threads=128) as (bx,hy,bz):
            value=T.alloc_fragment((block,d),'float32')
            absolute=T.alloc_fragment((block,d),'float32')
            rowmax=T.alloc_fragment((block,),'float32')
            maximum=T.alloc_fragment((1,),'float32')
            shared_scale=T.alloc_shared((1,),'float32')
            for i,j in T.Parallel(block,d):
                value[i,j]=T.if_then_else(bx*block+i<n,T.cast(X[bz,hy,bx*block+i,j],'float32')-(Mean[bz,hy,0,j] if center else 0.),0.)
                absolute[i,j]=T.abs(value[i,j])
            T.reduce_max(absolute,rowmax,dim=1)
            T.reduce_max(rowmax,maximum,dim=0)
            # Torch scalar division uses a float reciprocal multiplication;
            # tensor/tensor division below uses IEEE round-to-nearest division.
            for i in T.Parallel(1):shared_scale[i]=T.max(maximum[i]*T.float32(1./127.),1e-8)
            for i,j in T.Parallel(block,d):
                if bx*block+i<np:
                    normalized=T.call_pure_extern('float32','__fdiv_rn',value[i,j],shared_scale[0])
                    rounded=T.if_then_else(normalized>=0,T.floor(normalized+.5),-T.floor(-normalized+.5))
                    QI[bz,hy,bx*block+i,j]=T.cast(T.max(-127.,T.min(127.,rounded)),'int8')
            for i in T.Parallel(block):
                if bx*block+i<np:Scale[bz,hy,bx*block+i]=shared_scale[0]
    return tilelang.compile(main,out_idx=[2,3],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


def prepare_sage_fused(q,k,v,enable_gqa=False):
    if q.dtype!=torch.float16 or k.dtype!=q.dtype or v.dtype!=q.dtype:raise ValueError('FP16 inputs required')
    if q.shape[1]!=k.shape[1]:
        if not enable_gqa or q.shape[1]%k.shape[1]:raise ValueError('Invalid GQA')
        groups=q.shape[1]//k.shape[1];k=k.repeat_interleave(groups,1);v=v.repeat_interleave(groups,1)
    if q.ndim!=4 or q.shape!=k.shape or q.shape!=v.shape or q.shape[-1] not in (64,128):raise ValueError('Matching BHND required')
    if torch.cuda.get_device_capability(q.device)!=(7,5):raise ValueError('Actual T4 required')
    b,h,n,d=q.shape
    counts,routes,kc,vc,vi,vs=geometry(b,h,n,d,str(q.device))
    mean=k.float().mean(-2,keepdim=True)
    qi,qs=quant_kernel(b,h,n,d,32,False)(q.contiguous(),mean)
    ki,ks=quant_kernel(b,h,n,d,64,True)(k.contiguous(),mean)
    padded=lambda x:F.pad(x,(0,0,0,(-n)%64)).contiguous()
    return Prepared(padded(q),padded(v),qi,ki,qs,ks,kc,vc,counts,vi,vs,None,n),routes
