"""Group32 QK / FP16 PV TE prefill with compact GQA and additive/boolean masks.

Uses the existing fused Group32 preparation on SM75. Mask values are applied
in exp2 logits before online softmax. No mask classification or host readback.
The explicit SM120 kernel target is available only to arithmetic preview tests.
"""
from functools import lru_cache
import math
import torch


def normalize_mask(mask,b,h,n,nk,device):
    if mask is None:return torch.empty((1,1,1,1),device=device,dtype=torch.float32)
    if mask.device!=device:raise ValueError('Attention mask must be on the QKV device')
    if mask.ndim==2:mask=mask[None,None]
    elif mask.ndim==3:mask=mask[:,None]
    if mask.ndim!=4 or any(a not in (1,z) for a,z in zip(mask.shape,(b,h,n,nk))):
        raise ValueError('Mask must broadcast to local [B,H,Nq,Nk]')
    if mask.dtype==torch.bool:
        return torch.zeros_like(mask,dtype=torch.float32).masked_fill_(~mask,-float('inf')).contiguous()
    if not mask.is_floating_point():raise ValueError('Boolean or floating additive mask required')
    return mask.float().contiguous()


@lru_cache(maxsize=64)
def kernel(b,h,hk,n,nk,d,mask_shape,has_mask,dtype='float32',arch='sm_75'):
    import tilelang
    import tilelang.language as T
    if arch not in ('sm_75','sm_120'):raise ValueError('Unsupported target')
    bm=32;np=math.ceil(n/64)*64;kp=math.ceil(nk/64)*64;ng=d//32
    mb,mh,mq,mk=mask_shape;scale=d**-.5*math.log2(math.e)
    @T.prim_func
    def main(QI:T.Tensor((b,h,np,d),'int8'),KI:T.Tensor((b,hk,kp,d),'int8'),
             V:T.Tensor((b,hk,kp,d),'float16'),QS:T.Tensor((b,h,np,ng),'float32'),
             KS:T.Tensor((b,hk,kp,ng),'float32'),Mask:T.Tensor(mask_shape,'float32'),
             O:T.Tensor((b,n,h*d),dtype)):
        with T.Kernel(T.ceildiv(n,bm),h,b,threads=128) as (bx,hy,bz):
            kh=hy//(h//hk)
            q8=T.alloc_shared((bm,32),'int8');k8=T.alloc_shared((64,32),'int8')
            v16=T.alloc_shared((64,d),'float16');p16=T.alloc_shared((bm,64),'float16')
            dot=T.alloc_fragment((bm,64),'int32');scores=T.alloc_fragment((bm,64),'float32')
            pv=T.alloc_fragment((bm,d),'float32');out=T.alloc_fragment((bm,d),'float32')
            m=T.alloc_fragment((bm,),'float32');old=T.alloc_fragment((bm,),'float32')
            hm=T.alloc_fragment((bm,),'float32');alpha=T.alloc_fragment((bm,),'float32')
            psum=T.alloc_fragment((bm,),'float32');denom=T.alloc_fragment((bm,),'float32')
            safe=T.alloc_fragment((bm,),'float32')
            T.clear(out);T.clear(denom);T.fill(m,-T.infinity('float32'))
            for kb in T.serial(T.ceildiv(nk,64)):
                T.clear(scores)
                for g in T.serial(ng):
                    T.copy(QI[bz,hy,bx*bm:(bx+1)*bm,g*32:(g+1)*32],q8)
                    T.copy(KI[bz,kh,kb*64:(kb+1)*64,g*32:(g+1)*32],k8)
                    T.clear(dot);T.gemm(q8,k8,dot,transpose_B=True)
                    for i,j in T.Parallel(bm,64):
                        scores[i,j]+=T.cast(dot[i,j],'float32')*QS[bz,hy,bx*bm+i,g]*KS[bz,kh,kb*64+j,g]*scale
                for i,j in T.Parallel(bm,64):
                    if (bx*bm+i<n)&(kb*64+j<nk):
                        if has_mask:
                            scores[i,j]+=Mask[bz%mb,hy%mh,(bx*bm+i)%mq,(kb*64+j)%mk]*math.log2(math.e)
                    else:scores[i,j]=-T.infinity('float32')
                T.copy(m,old);T.reduce_max(scores,hm,dim=1)
                for i in T.Parallel(bm):
                    m[i]=T.max(old[i],hm[i]);safe[i]=T.if_then_else(m[i]==-T.infinity('float32'),0.,m[i])
                    alpha[i]=T.exp2(old[i]-safe[i])
                for i,j in T.Parallel(bm,64):
                    scores[i,j]=T.exp2(scores[i,j]-safe[i]);p16[i,j]=T.cast(scores[i,j],'float16')
                T.reduce_sum(scores,psum,dim=1)
                T.copy(V[bz,kh,kb*64:(kb+1)*64,:],v16);T.clear(pv);T.gemm(p16,v16,pv)
                for i,x in T.Parallel(bm,d):out[i,x]=out[i,x]*alpha[i]+pv[i,x]
                for i in T.Parallel(bm):denom[i]=denom[i]*alpha[i]+psum[i]
            for i,x in T.Parallel(bm,d):
                if bx*bm+i<n:
                    O[bz,bx*bm+i,hy*d+x]=T.cast(T.cast(T.if_then_else(denom[i]==0.,0.,out[i,x]/denom[i]),'float16'),dtype)
    return tilelang.compile(main,out_idx=[6],target={'kind':'cuda','arch':arch},pass_configs={'tl.enable_async_copy':False})


def execute_prepared(p,mask=None,output_dtype=torch.float32,*,local_preview=False):
    capability=torch.cuda.get_device_capability(p.qi.device)
    target=(12,0) if local_preview else (7,5)
    if capability!=target:raise ValueError('Physical SM75 required, or explicit local SM120 arithmetic preview')
    b,h,_,d=p.qi.shape;hk=p.ki.shape[1]
    m=normalize_mask(mask,b,h,p.queries,p.keys,p.qi.device)
    return kernel(b,h,hk,p.queries,p.keys,d,tuple(m.shape),mask is not None,
        str(output_dtype).removeprefix('torch.'),'sm_120' if local_preview else 'sm_75')(p.qi,p.ki,p.v,p.qs,p.ks,m)


def attention(q,k,v,mask=None):
    from patches.tilelang_grouped_tp_kitchen_v2 import prepare
    return execute_prepared(prepare(q,k,v,query_tile=32),mask,output_dtype=q.dtype)
