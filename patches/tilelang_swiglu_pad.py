"""FP32 SwiGLU/padding and opt-in owned FC2 rotation for H3 TP2.

Native mode preserves full-width FP32 ConvRot geometry. Owned FP32 mode
reduces temporary storage and is explicitly selected by session policy.
Gate/up may be separate TE tensors or strided halves of packed DiT FC1 output.
"""
from functools import lru_cache
import torch


@lru_cache(maxsize=64)
def kernel(rows,width,gate_stride,up_stride,world_size,rank,arch,block=1024,threads=128):
    import tilelang
    import tilelang.language as T
    if arch not in ('sm_75','sm_120'):raise ValueError('SM75 or explicit local SM120 preview required')
    if min(rows,width,world_size)<1 or not 0<=rank<world_size or min(gate_stride,up_stride)<width:raise ValueError('Invalid TP layout')
    if block not in (512,1024,2048) or threads not in (128,256):raise ValueError('Unsupported launch geometry')
    full=width*world_size;start=rank*width
    @T.prim_func
    def main(Gate:T.StridedTensor((rows,width),(gate_stride,1),'float32'),
             Up:T.StridedTensor((rows,width),(up_stride,1),'float32'),
             Padded:T.Tensor((rows,full),'float32')):
        with T.Kernel(T.ceildiv(rows*full,block),threads=threads) as bx:
            for i in T.Parallel(block):
                index=bx*block+i;row=index//full;col=index%full
                if index<rows*full:
                    if (col>=start)&(col<start+width):
                        gate=Gate[row,col-start]
                        exponent=T.call_pure_extern('float32','expf',-gate)
                        denominator=T.call_pure_extern('float32','__fadd_rn',T.float32(1.),exponent)
                        activated=T.call_pure_extern('float32','__fdiv_rn',gate,denominator)
                        Padded[row,col]=T.call_pure_extern('float32','__fmul_rn',activated,Up[row,col-start])
                    else:Padded[row,col]=T.float32(0.)
    return tilelang.compile(main,out_idx=[2],target={'kind':'cuda','arch':arch},pass_configs={'tl.enable_async_copy':False})


def swiglu_pad(gate,up,*,rank,world_size=2,block=1024,threads=128):
    """Return (..., local_width * world_size), with unowned columns set to +0."""
    if gate.shape!=up.shape or gate.ndim<2 or gate.shape[-1]<1 or gate.numel()==0:raise ValueError('Matching nonempty gate/up tensors required')
    if gate.dtype!=torch.float32 or up.dtype!=gate.dtype or not gate.is_cuda or up.device!=gate.device:raise ValueError('FP32 gate/up on one CUDA device required')
    if gate.stride(-1)!=1 or up.stride(-1)!=1:raise ValueError('Contiguous channel dimension required')
    if not isinstance(world_size,int) or not isinstance(rank,int) or world_size<1 or not 0<=rank<world_size:raise ValueError('Invalid TP ownership')
    capability=torch.cuda.get_device_capability(gate.device)
    arch={(7,5):'sm_75',(12,0):'sm_120'}.get(capability)
    if arch is None:raise ValueError('Only target SM75 and explicit local SM120 previews are supported')
    width=gate.shape[-1];g=gate.reshape(-1,width);u=up.reshape(-1,width)
    out=kernel(g.shape[0],width,g.stride(0),u.stride(0),world_size,rank,arch,block,threads)(g,u)
    return out.reshape(*gate.shape[:-1],width*world_size)


def swiglu_pad_packed(x,*,rank,world_size=2,block=1024,threads=128):
    """Consume packed [..., gate | up] without materializing either half."""
    if x.ndim<2 or x.shape[-1]%2:raise ValueError('Packed [gate | up] with an even final width required')
    gate,up=x.chunk(2,dim=-1)
    return swiglu_pad(gate,up,rank=rank,world_size=world_size,block=block,threads=threads)


def _rotate_padded(padded,width,world_size,groupsize):
    import comfy_kitchen as ck
    from patches import tilelang_row_quant as row
    backend=ck.backends.cuda;flat=padded.reshape(-1,width*world_size)
    if width%groupsize:raise ValueError('Whole ConvRot groups must be locally owned')
    rotation=backend._build_hadamard(groupsize,device=flat.device,dtype=flat.dtype)
    with row.label('native_full_width_rotation'):rotated=backend._rotate_activation(flat,rotation,groupsize)
    return rotated


def _quantize_rotated(rotated,width,comm,world_size):
    from patches import tilelang_row_quant as row
    rank=comm.get_rank()
    with row.label('local_maximum'):maximum=row.maximum_kernel(rotated.shape[0],width,width*world_size,rank*width)(rotated)
    comm.all_reduce(maximum,op=comm.ReduceOp.MAX)
    with row.label('external_max_quantization'):return row.quantize(rotated,maximum,width,rank*width)


def _quantize_padded(padded,width,comm,world_size,groupsize):
    rotated=_rotate_padded(padded,width,world_size,groupsize)
    del padded
    return _quantize_rotated(rotated,width,comm,world_size)


def _quantize_owned(x,comm,world_size,groupsize):
    """Opt-in FP32 group-local rotation, with unchanged global scale and SUM.

    Flattened GEMM M differs from Native; physical byte qualification covers
    specific trained projections, not every possible activation/backend.
    """
    import comfy_kitchen as ck
    from patches import tilelang_row_quant as row
    width=x.shape[-1]//2
    flat=x.reshape(-1,x.shape[-1]).contiguous()
    with row.label('owned_swiglu'):activated=swiglu_pad_packed(flat,rank=0,world_size=1)
    del flat
    h=ck.backends.cuda._build_hadamard(groupsize,device=x.device,dtype=x.dtype)
    with row.label('owned_rotation'):rotated=ck.backends.cuda._rotate_activation(activated,h,groupsize)
    del activated
    with row.label('local_maximum'):maximum=row.maximum_kernel(rotated.shape[0],width,width,0)(rotated)
    comm.all_reduce(maximum,op=comm.ReduceOp.MAX)
    with row.label('external_max_quantization'):return row.quantize(rotated,maximum,width,0)


def global_row_quantize(x,comm,world_size,groupsize=256,input_act='swiglu',variant='packed',rotation_mode='native'):
    """Plug-in quantization path for packed DiT FC2; other paths retain fallback."""
    from patches import tilelang_row_quant as previous
    if rotation_mode not in ('native','owned_fp32'):raise ValueError('Unknown FC2 rotation mode')
    if input_act!='swiglu' or x.dtype!=torch.float32 or world_size<2 or variant!='packed':
        return previous.global_row_quantize(x,comm,world_size,groupsize,input_act,variant)
    if rotation_mode=='owned_fp32' and world_size==2 and groupsize==256 and x.shape[-1]==14336:
        return _quantize_owned(x,comm,world_size,groupsize)
    flat=x.reshape(-1,x.shape[-1]).contiguous();width=flat.shape[-1]//2
    with previous.label('swiglu_and_padding'):padded=swiglu_pad_packed(flat,rank=comm.get_rank(),world_size=world_size)
    rotated=_rotate_padded(padded,width,world_size,groupsize)
    # The caller must release its owner too: deleting only the helper's
    # parameter retained full padding through MAX and output allocation.
    del padded,flat
    return _quantize_rotated(rotated,width,comm,world_size)


def _finish_row(q,scale,x_shape,dtype,weight,weight_scale,comm,bias,residual,residual_scale):
    from patches import parallel_int8 as original
    partial=original.int_mm(q,weight);comm.all_reduce(partial,op=comm.ReduceOp.SUM)
    out=original.dequantize(partial,scale,weight_scale,dtype,bias).reshape(*x_shape[:-1],weight.shape[0])
    if residual is not None:out=torch.addcmul(residual,out.to(dtype),residual_scale)
    return out.to(dtype)


def row_linear(x,weight,weight_scale,comm,world_size,groupsize=256,input_act=None,bias=None,residual=None,residual_scale=None,variant='packed',rotation_mode='native'):
    from patches import optimized_tp as old,tilelang_row_quant as previous
    width=x.shape[-1]//(2 if input_act=='swiglu' else 1)
    if input_act!='swiglu' or x.dtype!=torch.float32 or world_size<2 or variant!='packed' or not old.use_fallback_rotation(x,width*world_size,groupsize):
        return previous.row_linear(x,weight,weight_scale,comm,world_size,groupsize,input_act,bias,residual,residual_scale,variant)
    q,scale=global_row_quantize(x,comm,world_size,groupsize,input_act,variant,rotation_mode)
    return _finish_row(q,scale,x.shape,x.dtype,weight,weight_scale,comm,bias,residual,residual_scale)


def row_linear_parts(gate,up,weight,weight_scale,comm,world_size,groupsize=256,bias=None,residual=None,residual_scale=None,variant='packed'):
    """TE down projection entry point accepting separate gate/up outputs."""
    from patches import optimized_tp as old,tilelang_row_quant as previous
    if gate.shape!=up.shape:raise ValueError('Matching gate/up required')
    width=gate.shape[-1]
    if gate.dtype!=torch.float32 or world_size<2 or variant!='packed' or not old.use_fallback_rotation(gate,width*world_size,groupsize):
        activated=torch.nn.functional.silu(gate)*up
        return previous.row_linear(activated,weight,weight_scale,comm,world_size,groupsize,None,bias,residual,residual_scale,variant)
    with previous.label('swiglu_and_padding'):padded=swiglu_pad(gate,up,rank=comm.get_rank(),world_size=world_size)
    rotated=_rotate_padded(padded,width,world_size,groupsize)
    del padded
    q,scale=_quantize_rotated(rotated,width,comm,world_size)
    del rotated
    return _finish_row(q,scale,gate.shape,gate.dtype,weight,weight_scale,comm,bias,residual,residual_scale)
