"""Evidence-driven TP optimizations with native global-row INT8 semantics.

The fallback rotation consists of independent 256-channel groups. FP16 owns
local groups; FP32 retains native global GEMM geometry to preserve rounding.
Reduce one FP32 maximum per row and quantize with the native CUDA row kernel.
A padded sentinel group supplies the maximum without altering real features.
Native fused-rotation routes retain the full gather.
"""
import torch
import comfy_kitchen as ck
from patches import parallel_int8 as original

LEGACY_GATHER=original.gather
LEGACY_MM=original.int_mm
LEGACY_ROW=original.row_linear


def projected_reduce(q,scale,weight,weight_scale,comm,world_size,dtype,bias,mode,rows=256):
    """The Native INT32 accumulator is the only retained reduction."""
    if mode!='native_int32':raise ValueError('Only Native INT32 reduction is retained')
    partial=original.int_mm(q,weight)
    comm.all_reduce(partial,op=comm.ReduceOp.SUM)
    return original.dequantize(partial,scale,weight_scale,dtype,bias)


def fixed_gather(x,comm,dim):
    if dim not in (-1,x.ndim-1):return LEGACY_GATHER(x,comm,dim)
    if not comm.is_initialized():raise RuntimeError('Initialized transport required')
    x=x.contiguous()
    parts=[torch.empty_like(x) for _ in range(comm.get_world_size())]
    comm.all_gather(parts,x)
    return torch.cat(parts,dim=-1),comm.get_rank()*x.shape[-1],x.shape[-1]


def native_int_mm(q,weight):
    backend=ck.backends.cuda
    # Native Turing cuBLASLt storage/dispatch avoids torch._int_mm's per-call
    # transposed weight copy and padded activation buffer. INT32 sums remain
    # exact; reduction still occurs before native dequantization.
    q=q.contiguous();weight=weight.contiguous()
    out=torch.empty((q.shape[0],weight.shape[0]),device=q.device,dtype=torch.int32)
    wrap=backend._wrap_for_dlpack
    backend._C.cublas_gemm_int8(wrap(q),wrap(weight),wrap(out),wrap(backend.get_cublas_workspace()),torch.cuda.current_stream(q.device).cuda_stream)
    return out


def use_fallback_rotation(x,full_width,groupsize):
    backend=ck.backends.cuda
    fused=(groupsize==256 and full_width%256==0 and 256<=full_width<=backend._CONVROT_FUSED_MAX_K
           and backend._convrot_fused_shared_memory_fits(x,full_width,groupsize))
    return not fused and not backend._should_use_convrot_fused_kernel(x,full_width,groupsize)


def global_row_quantize(x,comm,world_size,groupsize,input_act):
    backend=ck.backends.cuda
    flat=x.reshape(-1,x.shape[-1]).contiguous()
    # Retain native eager activation rounding on the native fallback route.
    flat=backend._apply_input_act(flat,input_act,None,0.0)
    local_width=flat.shape[-1]
    if local_width%groupsize:raise ValueError('ConvRot groups must remain wholly owned by a rank')
    h=backend._build_hadamard(groupsize,device=flat.device,dtype=flat.dtype)
    if flat.dtype==torch.float32:
        # FP32 rotation GEMM can change reduction order when its flattened
        # row count changes. Preserve the native global GEMM geometry; zero
        # unowned groups, then discard them. This retains its operator route
        # while still replacing the large activation gather with row maxima.
        padded=torch.zeros((flat.shape[0],local_width*world_size),device=flat.device,dtype=flat.dtype)
        start=comm.get_rank()*local_width
        padded[:,start:start+local_width].copy_(flat)
        # Last reads are queued on this stream. Release each input before
        # allocating the next owned slice; keep the original full GEMM shape.
        del flat
        full_rotation=backend._rotate_activation(padded,h,groupsize)
        del padded
        rotated=full_rotation[:,start:start+local_width].contiguous()
        del full_rotation
    else:
        rotated=backend._rotate_activation(flat,h,groupsize)
    maximum=rotated.abs().amax(-1,keepdim=True).float()
    comm.all_reduce(maximum,op=comm.ReduceOp.MAX)
    # Same native row-wise CUDA quantizer, global scale, rounding and clamping.
    # The extra group is never supplied to the matrix multiplication.
    sentinel=torch.zeros((rotated.shape[0],groupsize),device=rotated.device,dtype=rotated.dtype)
    sentinel[:,0:1].copy_(maximum)
    q,scale=backend.quantize_int8_rowwise(torch.cat([rotated,sentinel],-1))
    return q[:,:local_width].contiguous(),scale


def row_linear(x,weight,weight_scale,comm,world_size,groupsize=256,input_act=None,bias=None,residual=None,residual_scale=None):
    local_width=x.shape[-1]//(2 if input_act=='swiglu' else 1)
    full_width=local_width*world_size
    fallback=use_fallback_rotation(x,full_width,groupsize)
    if world_size==1 or not fallback:
        return LEGACY_ROW(x,weight,weight_scale,comm,world_size,groupsize,input_act,bias,residual,residual_scale)
    q,scale=global_row_quantize(x,comm,world_size,groupsize,input_act)
    out=projected_reduce(q,scale,weight,weight_scale,comm,world_size,x.dtype,bias,'native_int32').reshape(*x.shape[:-1],weight.shape[0])
    if residual is not None:out=torch.addcmul(residual,out.to(x.dtype),residual_scale)
    return out.to(x.dtype)


def enable(mode='scalemax',swiglu_reduction='native_int32'):
    if mode not in ('scalemax','fixed'):raise ValueError('Unknown TP quantization mode')
    if swiglu_reduction!='native_int32':raise ValueError('FP32/INT8 reduction was retired after physical qualification; use native_int32')
    import patches.tp_qwen3vl as te
    import patches.tp_dit as dit
    original.gather=fixed_gather
    original.int_mm=native_int_mm
    te.row_linear=dit.row_linear=row_linear if mode=='scalemax' else LEGACY_ROW


def disable():
    import patches.tp_qwen3vl as te
    import patches.tp_dit as dit
    original.gather=LEGACY_GATHER;original.int_mm=LEGACY_MM
    te.row_linear=dit.row_linear=LEGACY_ROW
