"""Experimental inference-only row scheduling around the pinned Kitchen linear.

This does not install a backend or alter production defaults. ConvRot, per-row
quantization, INT8 GEMM and dequantization remain Kitchen calls. A full output
is allocated once; each bounded row result is copied into it and released.
Numerical equality must be qualified on the deployment device because changing
GEMM dimensions can change the selected implementation.
"""
import torch


def bounded_int8_linear(x, weight, weight_scale, *, chunk_rows=4096,
                        convrot=True, convrot_groupsize=256,
                        out_dtype=torch.float16, linear=None):
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    if torch.is_grad_enabled():
        raise RuntimeError('Bounded column experiment requires inference/no_grad')
    if linear is None:
        from comfy_kitchen import int8_linear
        linear = int8_linear
    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError('Expected [..., K] input and [N, K] weight')
    kwargs = dict(convrot=convrot, convrot_groupsize=convrot_groupsize)
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] <= chunk_rows:
        return linear(x, weight, weight_scale, None, out_dtype, **kwargs)
    output = torch.empty((flat.shape[0], weight.shape[0]), device=x.device, dtype=out_dtype)
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        part = linear(flat[start:stop], weight, weight_scale, None, out_dtype, **kwargs)
        output[start:stop].copy_(part)
        del part
    return output.reshape(*x.shape[:-1], weight.shape[0])


def preallocated_eager_int8_linear(x, weight, weight_scale, *, convrot=True,
                                   convrot_groupsize=256, out_dtype=torch.float16):
    """Benchmark-only eager epilogue replacement, preserving full GEMM geometry.

    Uses the pinned eager helpers and the same FP32 scale multiplication order.
    Unlike row scheduling, it retains the full INT32 accumulator. It removes
    the list of scaled outputs and its final concatenation allocation.
    """
    if torch.is_grad_enabled():
        raise RuntimeError('Preallocated column experiment requires inference/no_grad')
    from comfy_kitchen.backends.eager import quantization as eager
    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError('Expected [..., K] input and [N, K] weight')
    weight = weight.to(device=x.device).contiguous()
    scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(1, -1)
    if scale.numel() not in (1, weight.shape[0]):
        raise ValueError('Expected scalar or per-output-channel scale')
    if convrot:
        if x.shape[-1] % convrot_groupsize:
            raise ValueError('ConvRot group must divide input features')
        h = eager._build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        rotated = eager._rotate_activation(x, h, convrot_groupsize)
    else:
        rotated = x
    codes, row_scale = eager.quantize_int8_rowwise(rotated.reshape(-1, x.shape[-1]))
    del rotated
    accumulator = eager._int8_matmul_accumulate(codes, weight.T.contiguous())
    del codes
    m, n = accumulator.shape
    output = torch.empty((m, n), device=x.device, dtype=out_dtype)
    chunk_rows = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
    for start in range(0, m, chunk_rows):
        stop = min(start + chunk_rows, m)
        chunk = accumulator[start:stop].float()
        chunk_scales = row_scale[start:stop].to(device=x.device, dtype=torch.float32) * scale
        scaled = chunk * chunk_scales
        output[start:stop].copy_(scaled)
        del chunk, chunk_scales, scaled
    return output.reshape(*x.shape[:-1], weight.shape[0])
