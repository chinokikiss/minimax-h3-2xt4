"""P0-A1: omit the dead FP16 output allocation on Kitchen's SM75 Turing path.

This is an intentionally narrow implementation of the pinned Kitchen 0.2.37
CUDA branch. All unsupported geometry, versions, devices and execution modes
return None so the public Kitchen registry remains the fallback.
"""
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch


@lru_cache(maxsize=1)
def _pinned_kitchen():
    try:
        return version('comfy-kitchen') == '0.2.37'
    except PackageNotFoundError:
        return False


def turing_column_a1(x, weight, weight_scale, *, convrot, convrot_groupsize, out_dtype):
    """Return the exact Kitchen Turing result or None without changing the input."""
    if (not _pinned_kitchen() or torch.is_grad_enabled()
            or torch.compiler.is_compiling() or type(x) is not torch.Tensor
            or not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2
            or not x.is_contiguous() or x.shape[0] <= 2 or x.shape[1] != 5376
            or weight.shape not in ((10752, 5376), (14336, 5376))
            or not weight.is_cuda or weight.device != x.device
            or weight.dtype != torch.int8 or not weight.is_contiguous()
            or weight_scale.numel() != weight.shape[0]
            or weight_scale.device != x.device or weight_scale.dtype != torch.float32
            or out_dtype != torch.float16 or not convrot or convrot_groupsize != 256
            or torch.cuda.get_device_capability(x.device) != (7, 5)):
        return None

    from comfy_kitchen.backends import cuda as kitchen

    m, k = x.shape
    n = weight.shape[0]
    if (not kitchen._prefer_turing_fused_int8(m, n, k)
            or not kitchen._cuda_device_should_use_turing_kernels(x.get_device())
            or not hasattr(kitchen._C, 'cutlass_turing_int8_dequant')):
        return None

    # Match the pinned Kitchen quantization branch and its math/order exactly.
    fused = (256 <= k <= kitchen._CONVROT_FUSED_MAX_K
             and kitchen._convrot_fused_shared_memory_fits(x, k, 256))
    if fused:
        codes = torch.empty((m, k), dtype=torch.int8, device=x.device)
        row_scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        activation_weight = kitchen._act_weight_arg(None, None, x.device, x.dtype)
        kitchen._C.quantize_int8_rowwise_convrot64(
            kitchen._wrap_for_dlpack(x), kitchen._wrap_for_dlpack(codes),
            kitchen._wrap_for_dlpack(row_scale), 256, False,
            kitchen._input_act_code(None), 0,
            kitchen._wrap_for_dlpack(activation_weight), 0.,
            torch.cuda.current_stream(x.device).cuda_stream)
    elif kitchen._should_use_convrot_fused_kernel(x, k, 256):
        codes, row_scale = kitchen.quantize_int8_rowwise_convrot(x, 256)
    else:
        hadamard = kitchen._build_hadamard(256, device=x.device, dtype=x.dtype)
        codes, row_scale = kitchen.quantize_and_rotate_rowwise(x, hadamard, 256)

    scale = kitchen._int8_weight_scale_arg(weight_scale, x.device)
    # Kitchen 0.2.37 previously allocated a full (m, n) FP16 'out' just before
    # this early return, but never read or returned that buffer on success.
    return kitchen._int8_linear_turing_quantized(
        codes, weight, row_scale, scale, None, out_dtype)
