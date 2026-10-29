"""Experimental SM75 stores; FP32 islands and FP16 branch rounding remain.

No backend registration or production import. These kernels are qualified by
the accompanying operator suite before any runtime promotion is considered.
"""
from functools import lru_cache
import torch


@lru_cache(maxsize=32)
def norm_kernel(rows, width, modulation_rows, epsilon, arch='sm_75'):
    import tilelang
    import tilelang.language as T
    padded = 1 << (width-1).bit_length()
    @T.prim_func
    def main(X:T.Tensor((rows,width),'float16'),W:T.Tensor((width,),'float32'),
             Shift:T.Tensor((modulation_rows,width),'float32'),
             Scale:T.Tensor((modulation_rows,width),'float32'),
             Labels:T.Tensor((rows,),'int32'),Y:T.Tensor((rows,width),'float16')):
        with T.Kernel(rows,threads=256) as row:
            squares=T.alloc_fragment((padded,),'float32')
            total=T.alloc_fragment((1,),'float32')
            for i in T.Parallel(padded):
                value=T.if_then_else(i<width,T.cast(X[row,i],'float32'),T.float32(0.))
                squares[i]=T.call_pure_extern('float32','__fmul_rn',value,value)
            T.reduce_sum(squares,total,dim=0)
            for i in T.Parallel(width):
                mean=T.call_pure_extern('float32','__fmul_rn',total[0],T.float32(1./width))
                inv=T.call_pure_extern('float32','rsqrtf',T.call_pure_extern('float32','__fadd_rn',mean,T.float32(epsilon)))
                value=T.call_pure_extern('float32','__fmul_rn',T.cast(X[row,i],'float32'),inv)
                gain=T.call_pure_extern('float32','__fmul_rn',value,W[i])
                factor=T.call_pure_extern('float32','__fadd_rn',T.float32(1.),Scale[Labels[row],i])
                scaled=T.call_pure_extern('float32','__fmul_rn',gain,factor)
                Y[row,i]=T.cast(T.call_pure_extern('float32','__fadd_rn',scaled,Shift[Labels[row],i]),'float16')
    return tilelang.compile(main,out_idx=[5],target={'kind':'cuda','arch':arch},
                            pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=32)
def gate_kernel(rows,width,modulation_rows,arch='sm_75'):
    import tilelang
    import tilelang.language as T
    @T.prim_func
    def main(X:T.Tensor((rows,width),'float16'),B:T.Tensor((rows,width),'float16'),
             Gate:T.Tensor((modulation_rows,width),'float32'),Labels:T.Tensor((rows,),'int32')):
        with T.Kernel(rows,T.ceildiv(width,1024),threads=128) as (r,bx):
            for i in T.Parallel(1024):
                c=bx*1024+i
                if c<width:
                    value=T.call_pure_extern('float32','__fmaf_rn',T.cast(B[r,c],'float32'),Gate[Labels[r],c],T.cast(X[r,c],'float32'))
                    X[r,c]=T.cast(value,'float16')
    return tilelang.compile(main,out_idx=[],target={'kind':'cuda','arch':arch},
                            pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=64)
def column_store_kernel(rows,width,scalar_scale,arch='sm_75'):
    import tilelang
    import tilelang.language as T
    @T.prim_func
    def main(C:T.Tensor((rows,width),'int32'),A:T.Tensor((rows,1),'float32'),
             W:T.Tensor((1 if scalar_scale else width,),'float32'),Y:T.Tensor((rows,width),'float16')):
        with T.Kernel(rows,T.ceildiv(width,1024),threads=128) as (r,bx):
            for i in T.Parallel(1024):
                c=bx*1024+i
                if c<width:
                    scale=T.call_pure_extern('float32','__fmul_rn',A[r,0],W[0 if scalar_scale else c])
                    Y[r,c]=T.cast(T.call_pure_extern('float32','__fmul_rn',T.cast(C[r,c],'float32'),scale),'float16')
    return tilelang.compile(main,out_idx=[],target={'kind':'cuda','arch':arch},
                            pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=32)
def row_gate_kernel(rows,width,modulation_rows,shift,arch='sm_75'):
    import tilelang
    import tilelang.language as T
    @T.prim_func
    def main(P:T.Tensor((rows,width),'float16'),A:T.Tensor((rows,1),'float32'),
             W:T.Tensor((width,),'float32'),X:T.Tensor((rows,width),'float16'),
             Gate:T.Tensor((modulation_rows,width),'float32'),Labels:T.Tensor((rows,),'int32')):
        with T.Kernel(rows,T.ceildiv(width,1024),threads=128) as (r,bx):
            for i in T.Parallel(1024):
                c=bx*1024+i
                if c<width:
                    total=T.call_pure_extern('float32','__fmul_rn',T.cast(P[r,c],'float32'),T.float32(2.**shift))
                    restored=T.cast(total,'int32')
                    value=T.call_pure_extern('float32','__fmul_rn',T.cast(restored,'float32'),A[r,0])
                    dequant=T.call_pure_extern('float32','__fmul_rn',value,W[c])
                    # This rounding is part of the existing operator, even
                    # though the branch is never stored in a full FP16 array.
                    branch=T.cast(T.cast(dequant,'float16'),'float32')
                    X[r,c]=T.cast(T.call_pure_extern('float32','__fmaf_rn',branch,Gate[Labels[r],c],T.cast(X[r,c],'float32')),'float16')
    return tilelang.compile(main,out_idx=[],target={'kind':'cuda','arch':arch},
                            pass_configs={'tl.enable_async_copy':False})


def require_fp16(x):
    if (x.ndim!=2 or x.dtype!=torch.float16 or not x.is_contiguous() or not x.is_cuda
            or torch.cuda.get_device_capability(x.device)!=(7,5)):
        raise ValueError('Experimental kernels require contiguous FP16 matrices on physical SM75')


def norm(norm,x,shift,scale,labels):
    require_fp16(x)
    return norm_kernel(*x.shape,shift.shape[0],float(norm.eps))(
        x,norm.weight.float().contiguous(),shift.float().contiguous(),scale.float().contiguous(),labels)


def gate(x,gates,branch,labels):
    require_fp16(x)
    if branch.shape!=x.shape or branch.dtype!=x.dtype or not branch.is_contiguous():
        raise ValueError('Matching contiguous FP16 branch required')
    gate_kernel(*x.shape,gates.shape[0])(x,branch,gates.float().contiguous(),labels)
    return x


def row_gate(payload,scale,weights,x,gates,labels,local_k):
    from patches.tilelang_transport_tradeoffs import shift_for
    require_fp16(x)
    row_gate_kernel(*x.shape,gates.shape[0],shift_for(local_k))(
        payload,scale,weights.reshape(-1),x,gates.float().contiguous(),labels)
    return x
