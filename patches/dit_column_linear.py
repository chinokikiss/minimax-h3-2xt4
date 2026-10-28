"""Scoped eager column projections and shared TE input preparation.

The default keeps the selected backend, GEMM geometry and eager FP32 scale
order, with one output destination. Experimental SM75 routes require explicit
selection. Other backends, autograd and compiled calls retain the public API.
"""
import torch
import comfy_kitchen as ck

_DEQUANT_CHUNK_BYTES = 256 * 1024 * 1024


def _prepare_activation(x,convrot,convrot_groupsize):
    from comfy_kitchen.backends.eager import quantization as eager
    if convrot:
        if x.shape[-1] % convrot_groupsize:
            raise ValueError('ConvRot group size must divide input features')
        h=eager._build_hadamard(convrot_groupsize,device=x.device,dtype=x.dtype)
        rotated=eager._rotate_activation(x,h,convrot_groupsize)
    else:rotated=x
    return eager.quantize_int8_rowwise(rotated.reshape(-1,rotated.shape[-1]))


def _dequantize_eager(accumulator,row_scale,weight_scale,out_dtype):
    rows,width=accumulator.shape
    chunk_rows=max(1,min(rows,_DEQUANT_CHUNK_BYTES//(width*4)))
    output=torch.empty((rows,width),device=accumulator.device,dtype=out_dtype)
    weight_scale=weight_scale.reshape(1,-1)
    for start in range(0,rows,chunk_rows):
        stop=min(start+chunk_rows,rows)
        chunk=accumulator[start:stop].float()
        scales=row_scale[start:stop].to(device=chunk.device,dtype=torch.float32)*weight_scale
        chunk.mul_(scales)
        del scales
        output[start:stop].copy_(chunk)
        del chunk
    return output


def _column_view_eligible(codes, weight):
    """Physical-T4-qualified TP2 TE columns; other geometry keeps its layout."""
    return (codes.is_cuda and weight.is_cuda and codes.device == weight.device
            and codes.dtype == weight.dtype == torch.int8
            and codes.ndim == weight.ndim == 2 and codes.shape[0] > 0
            and codes.shape[1] == weight.shape[1] == 5120
            and weight.shape[0] in (512, 4096, 12800)
            and codes.is_contiguous() and weight.is_contiguous()
            and torch.cuda.get_device_capability(codes.device) == (7, 5))


def _column_accumulate(codes, weight):
    """Use a zero-copy RHS view for qualified SM75 columns, retaining all math."""
    from comfy_kitchen.backends.eager import quantization as eager
    # The [K,N] column-major view enables the measured SM75 tensorop GEMM.
    # Keep Kitchen's quantization, padding and INT32 accumulation unchanged.
    rhs = weight.T if _column_view_eligible(codes, weight) else weight.T.contiguous()
    return eager._int8_matmul_accumulate(codes, rhs)


def _dequantize_column(accumulator,row_scale,weight_scale,out_dtype,weight):
    """Use the byte-qualified TE epilogue without changing shared preparation."""
    if (accumulator.is_cuda and torch.cuda.get_device_capability(accumulator.device)==(7,5)
            and weight.shape[1]==5120 and weight.shape[0] in (512,4096,12800)
            and out_dtype==torch.float16):
        from patches.tilelang_column_epilogue import dequantize
        return dequantize(accumulator,row_scale,weight_scale,out_dtype)
    return _dequantize_eager(accumulator,row_scale,weight_scale,out_dtype)


def shared_column_linears(x,projections,*,convrot=True,convrot_groupsize=256):
    """Share deterministic eager rotation/quantization; keep each GEMM intact.

    Return None when the selected backend or execution mode must be preserved
    through its public operators. The caller retains normal module hooks there.
    """
    if torch.is_grad_enabled() or torch.compiler.is_compiling() or type(x) is not torch.Tensor:
        return None
    from comfy_kitchen.backends.eager import quantization as eager
    flat=x.reshape(-1,x.shape[-1])
    for weight,weight_scale in projections:
        call=dict(x=flat,weight=weight,weight_scale=weight_scale,bias=None,out_dtype=x.dtype,
            convrot=convrot,convrot_groupsize=convrot_groupsize,input_act=None,input_act_weight=None,
            input_act_eps=0.,residual=None,residual_scale=None)
        if ck.registry.get_implementation('int8_linear',kwargs=call) is not eager.int8_linear:return None
        if weight.ndim!=2 or weight.shape[-1]!=x.shape[-1] or weight_scale.numel() not in (1,weight.shape[0]):
            return None
    codes,row_scale=_prepare_activation(flat,convrot,convrot_groupsize)
    outputs=[]
    for weight,weight_scale in projections:
        weight=weight.to(device=x.device).contiguous()
        scale=weight_scale.to(device=x.device,dtype=torch.float32).reshape(-1)
        accumulator=_column_accumulate(codes,weight)
        output=_dequantize_column(accumulator,row_scale,scale,x.dtype,weight)
        del accumulator
        outputs.append(output.reshape(*x.shape[:-1],weight.shape[0]))
    return tuple(outputs)


def column_linear(x, weight, weight_scale, *, convrot=True,
                  convrot_groupsize=256, out_dtype=torch.float16, backend='preallocated'):
    if backend not in ('preallocated','cuda_int8_tilelang'):
        raise ValueError('Unknown column linear backend')
    kwargs = dict(convrot=convrot, convrot_groupsize=convrot_groupsize)
    if torch.is_grad_enabled() or torch.compiler.is_compiling() or type(x) is not torch.Tensor:
        return ck.int8_linear(x, weight, weight_scale, None, out_dtype, **kwargs)
    from comfy_kitchen.backends.eager import quantization as eager
    call = dict(x=x, weight=weight, weight_scale=weight_scale, bias=None,
                out_dtype=out_dtype, input_act=None, input_act_weight=None,
                input_act_eps=0., residual=None, residual_scale=None, **kwargs)
    implementation = ck.registry.get_implementation('int8_linear', kwargs=call)
    if implementation is not eager.int8_linear:
        return ck.int8_linear(x, weight, weight_scale, None, out_dtype, **kwargs)
    # Experimental routes stay scoped to the actual aligned H3 TP2 columns.
    # Other devices/shapes retain the pinned selected implementation.
    if (not x.is_cuda or torch.cuda.get_device_capability(x.device)!=(7,5)
            or weight.shape not in ((10752,5376),(14336,5376))):
        backend='preallocated'
    return _eager_column(x, weight, weight_scale, **kwargs, out_dtype=out_dtype,backend=backend)


def _eager_column(x, weight, weight_scale, *, convrot, convrot_groupsize, out_dtype,backend='preallocated'):
    from comfy_kitchen.backends.eager import quantization as eager
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError('Input and weight inner dimensions must match')
    shape = x.shape
    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError('INT8 weight scale must be scalar or per-output-channel')
    codes,row_scale=_prepare_activation(x,convrot,convrot_groupsize)
    if backend=='cuda_int8_tilelang':
        from patches.optimized_tp import native_int_mm
        accumulator=native_int_mm(codes,weight)
    else:
        accumulator = _column_accumulate(codes, weight)
    del codes
    # tilelang_exact is retained only as a private historical benchmark control;
    # column_linear and the runtime policy expose the direct route or fallback.
    if backend in ('tilelang_exact','cuda_int8_tilelang'):
        from patches.tilelang_column_epilogue import dequantize
        output=dequantize(accumulator,row_scale,weight_scale,out_dtype)
        return output.reshape(*shape[:-1],weight.shape[0])
    # One destination, no retained list of FP16 chunks or final cat copy.
    # Keep the full INT8 GEMM and eager's two FP32 multiplication boundaries.
    output=_dequantize_column(accumulator,row_scale,weight_scale,out_dtype,weight)
    del accumulator, row_scale
    return output.reshape(*shape[:-1], weight.shape[0])
