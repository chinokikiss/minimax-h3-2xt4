"""Optional eager-ordered INT32 column dequantization, with one output store.

Unlike row dequantization, multiply row and weight scales first. Both FP32
products explicitly round, so no algebraic reassociation is permitted.
"""
from functools import lru_cache
import torch


@lru_cache(maxsize=32)
def kernel(rows,width,scalar_scale,dtype,arch):
    import tilelang
    import tilelang.language as T
    scale_width=1 if scalar_scale else width
    @T.prim_func
    def main(C:T.Tensor((rows,width),'int32'),A:T.Tensor((rows,1),'float32'),
             W:T.Tensor((scale_width,),'float32'),Y:T.Tensor((rows,width),dtype)):
        with T.Kernel(T.ceildiv(rows*width,1024),threads=128) as bx:
            for i in T.Parallel(1024):
                index=bx*1024+i
                if index<rows*width:
                    row=index//width;column=index%width
                    scale=T.call_pure_extern('float32','__fmul_rn',A[row,0],W[0 if scalar_scale else column])
                    value=T.call_pure_extern('float32','__fmul_rn',T.cast(C[row,column],'float32'),scale)
                    Y[row,column]=T.cast(value,dtype)
    return tilelang.compile(main,out_idx=[3],target={'kind':'cuda','arch':arch},
                            pass_configs={'tl.enable_async_copy':False})


def dequantize(accumulator,row_scale,weight_scale,out_dtype):
    if not accumulator.is_cuda or accumulator.dtype!=torch.int32 or accumulator.ndim!=2:
        raise ValueError('CUDA INT32 matrix required')
    rows,width=accumulator.shape
    if weight_scale.numel() not in (1,width):raise ValueError('Scalar or channel scale required')
    if out_dtype not in (torch.float16,torch.bfloat16,torch.float32):raise ValueError('Unsupported column dtype')
    capability=torch.cuda.get_device_capability(accumulator.device)
    arch='sm_'+''.join(map(str,capability))
    return kernel(rows,width,weight_scale.numel()==1,str(out_dtype).removeprefix('torch.'),arch)(
        accumulator.contiguous(),row_scale.to(dtype=torch.float32).reshape(rows,1).contiguous(),
        weight_scale.to(dtype=torch.float32).reshape(-1).contiguous())
