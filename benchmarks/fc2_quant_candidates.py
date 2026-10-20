"""Benchmark adapter for the explicitly selected owned FP32 FC2 quantizer."""
from patches import tilelang_swiglu_pad as fused


def owned_rotation(x, comm, world_size=2, groupsize=256):
    # Real-arithmetic group locality permits this. GEMM M changes; numerical
    # acceptance MUST be established independently for each backend/shape.
    width = x.shape[-1] // 2
    if world_size != 2 or width % groupsize:
        raise ValueError('Only TP2 whole-group FC2 ownership is under study')
    return fused.global_row_quantize(x,comm,world_size,groupsize,rotation_mode='owned_fp32')
