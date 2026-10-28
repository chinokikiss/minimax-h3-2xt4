"""Packed H3 QKV storage with FP32 RMS statistics, gains and split-half RoPE."""
from functools import lru_cache
import torch


@lru_cache(maxsize=32)
def kernel(rows,heads,dim,rot_dim,input_dtype,output_dtype,epsilon,arch='sm_75'):
    import tilelang
    import tilelang.language as T
    if rows<1 or heads<1 or heads%4 or dim!=128 or not 0<rot_dim<=dim or rot_dim%2:
        raise ValueError('H3 full-head RMS and even rotary prefix required')
    half=rot_dim//2;width=heads*dim
    @T.prim_func
    def main(P:T.Tensor((rows,3*width),input_dtype),F:T.Tensor((rows,half,2,2),'float32'),
             GQ:T.Tensor((dim,),'float32'),GK:T.Tensor((dim,),'float32'),
             Q:T.Tensor((rows,heads,dim),output_dtype),K:T.Tensor((rows,heads,dim),output_dtype)):
        with T.Kernel(rows,heads//4,threads=128) as (row,head_tile):
            q=T.alloc_fragment((4,dim),'float32');k=T.alloc_fragment((4,dim),'float32')
            qs=T.alloc_fragment((4,dim),'float32');ks=T.alloc_fragment((4,dim),'float32')
            qm=T.alloc_fragment((4,),'float32');km=T.alloc_fragment((4,),'float32')
            qn=T.alloc_shared((4,dim),'float32');kn=T.alloc_shared((4,dim),'float32')
            for h,d in T.Parallel(4,dim):
                q[h,d]=T.cast(P[row,(head_tile*4+h)*dim+d],'float32')
                k[h,d]=T.cast(P[row,width+(head_tile*4+h)*dim+d],'float32')
                qs[h,d]=T.call_pure_extern('float32','__fmul_rn',q[h,d],q[h,d])
                ks[h,d]=T.call_pure_extern('float32','__fmul_rn',k[h,d],k[h,d])
            T.reduce_sum(qs,qm,dim=1);T.reduce_sum(ks,km,dim=1)
            for h,d in T.Parallel(4,dim):
                qr=T.call_pure_extern('float32','rsqrtf',T.call_pure_extern('float32','__fadd_rn',qm[h]*T.float32(1./dim),T.float32(epsilon)))
                kr=T.call_pure_extern('float32','rsqrtf',T.call_pure_extern('float32','__fadd_rn',km[h]*T.float32(1./dim),T.float32(epsilon)))
                qn[h,d]=T.call_pure_extern('float32','__fmul_rn',T.call_pure_extern('float32','__fmul_rn',q[h,d],qr),GQ[d])
                kn[h,d]=T.call_pure_extern('float32','__fmul_rn',T.call_pure_extern('float32','__fmul_rn',k[h,d],kr),GK[d])
            T.sync_threads()
            for h,d in T.Parallel(4,dim):
                if d<rot_dim:
                    pair=d%half;side=d//half;other=T.if_then_else(d<half,d+half,d-half)
                    a=T.call_pure_extern('float32','__fmul_rn',qn[h,d],F[row,pair,side,side])
                    b=T.call_pure_extern('float32','__fmul_rn',kn[h,d],F[row,pair,side,side])
                    Q[row,head_tile*4+h,d]=T.cast(T.call_pure_extern('float32','__fmaf_rn',qn[h,other],F[row,pair,side,1-side],a),output_dtype)
                    K[row,head_tile*4+h,d]=T.cast(T.call_pure_extern('float32','__fmaf_rn',kn[h,other],F[row,pair,side,1-side],b),output_dtype)
                else:
                    Q[row,head_tile*4+h,d]=T.cast(qn[h,d],output_dtype);K[row,head_tile*4+h,d]=T.cast(kn[h,d],output_dtype)
    return tilelang.compile(main,out_idx=[4,5],target={'kind':'cuda','arch':arch},
        pass_configs={'tl.enable_async_copy':False})


def supported(packed,freqs,heads,dim):
    return (packed.is_cuda and packed.ndim==2 and packed.dtype in (torch.float16,torch.float32)
        and packed.is_contiguous() and packed.shape[1]==3*heads*dim and dim==128 and heads==28
        and freqs.device==packed.device and freqs.dtype==torch.float32 and freqs.is_contiguous()
        and freqs.ndim==6 and freqs.shape[:3]==(1,packed.shape[0],1)
        and freqs.shape[-2:]==(2,2) and 0<freqs.shape[-3]*2<=dim
        and torch.cuda.get_device_capability(packed.device)==(7,5))


def execute(packed,freqs,q_gain,k_gain,*,heads=28,dim=128,epsilon=1e-6,out_dtype=torch.float16):
    if not supported(packed,freqs,heads,dim):raise ValueError('Qualified SM75 H3 packed QKV/frequency layout required')
    if out_dtype not in (torch.float16,torch.float32):raise ValueError('Explicit FP16/FP32 output required')
    gains=[g.to(device=packed.device,dtype=torch.float32).reshape(dim).contiguous() for g in (q_gain,k_gain)]
    rows=packed.shape[0];rot_dim=freqs.shape[-3]*2
    return kernel(rows,heads,dim,rot_dim,str(packed.dtype).removeprefix('torch.'),
        str(out_dtype).removeprefix('torch.'),float(epsilon))(packed,freqs.reshape(rows,rot_dim//2,2,2),*gains)
