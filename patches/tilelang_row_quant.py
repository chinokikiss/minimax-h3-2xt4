"""Opt-in SM75 FP32 row-linear candidates with unchanged native rotation/GEMM.

The reduced maximum must cover the whole finite input row across TP ranks.
These benchmark candidates are not installed in the production hook.
Kitchen 0.2.37 uses max*float32(1/127), floor 1e-30, native fast division,
nearbyint and [-128,127] saturation. Do not substitute the attention quantizer.
"""
from functools import lru_cache
from contextlib import contextmanager
import torch


@contextmanager
def label(name):
    torch.cuda.nvtx.range_push('row_fusion/' + name)
    try: yield
    finally: torch.cuda.nvtx.range_pop()


def _device(x):
    if not x.is_cuda or x.dtype != torch.float32 or torch.cuda.get_device_capability(x.device) != (7, 5):
        raise ValueError('Actual SM75 FP32 input required')


@lru_cache(maxsize=32)
def quantizer(rows, width, stride, start, division='fast'):
    import tilelang
    import tilelang.language as T
    if division not in ('fast', 'rn'): raise ValueError('Unknown division mode')
    if rows < 1 or width < 1 or start < 0 or start + width > stride: raise ValueError('Invalid owned slice')
    @T.prim_func
    def main(X: T.Tensor((rows, stride), 'float32'), Maximum: T.Tensor((rows, 1), 'float32'),
             Q: T.Tensor((rows, width), 'int8'), Scale: T.Tensor((rows, 1), 'float32')):
        with T.Kernel(rows, threads=256) as row:
            scale = T.alloc_local((1,), 'float32')
            maximum = T.call_pure_extern('float32', 'fminf', Maximum[row, 0], T.float32(3.4028234663852886e38))
            scale[0] = T.call_pure_extern('float32', 'fmaxf', maximum * T.float32(1. / 127.), T.float32(1.e-30))
            for col in T.Parallel(width):
                scaled = T.call_pure_extern('float32', '__fdividef' if division == 'fast' else '__fdiv_rn', X[row, start + col], scale[0])
                rounded = T.call_pure_extern('float32', 'nearbyintf', scaled)
                clipped = T.call_pure_extern('float32', 'fminf', T.float32(127.),
                    T.call_pure_extern('float32', 'fmaxf', T.float32(-128.), rounded))
                Q[row, col] = T.cast(clipped, 'int8')
            if T.get_thread_binding() == 0: Scale[row, 0] = scale[0]
    return tilelang.compile(main, out_idx=[2, 3], target={'kind': 'cuda', 'arch': 'sm_75'}, pass_configs={'tl.enable_async_copy': False})


@lru_cache(maxsize=32)
def maximum_kernel(rows, width, stride, start):
    import tilelang
    import tilelang.language as T
    if rows < 1 or width < 1 or start < 0 or start + width > stride: raise ValueError('Invalid owned slice')
    @T.prim_func
    def main(X: T.Tensor((rows, stride), 'float32'), Maximum: T.Tensor((rows, 1), 'float32')):
        with T.Kernel(rows, threads=256) as row:
            absolute = T.alloc_fragment((1, width), 'float32')
            maximum = T.alloc_fragment((1,), 'float32')
            for col in T.Parallel(width): absolute[0, col] = T.abs(X[row, start + col])
            T.reduce_max(absolute, maximum, dim=1)
            if T.get_thread_binding() == 0: Maximum[row, 0] = maximum[0]
    return tilelang.compile(main, out_idx=[1], target={'kind': 'cuda', 'arch': 'sm_75'}, pass_configs={'tl.enable_async_copy': False})


@lru_cache(maxsize=32)
def packing_kernel(rows, width, world_size, rank):
    import tilelang
    import tilelang.language as T
    if rows < 1 or width < 1 or world_size < 1 or not 0 <= rank < world_size: raise ValueError('Invalid TP partition')
    full = width * world_size
    @T.prim_func
    def main(X: T.Tensor((rows, width), 'float32'), Padded: T.Tensor((rows, full), 'float32')):
        with T.Kernel(T.ceildiv(rows * full, 1024), threads=256) as block:
            for i in T.Parallel(1024):
                idx = block * 1024 + i; row = idx // full; col = idx % full
                if row < rows:
                    Padded[row, col] = T.if_then_else((col >= rank * width) & (col < (rank + 1) * width), X[row, col - rank * width], 0.)
    return tilelang.compile(main, out_idx=[1], target={'kind': 'cuda', 'arch': 'sm_75'}, pass_configs={'tl.enable_async_copy': False})


def quantize(rotated, maximum, width=None, start=0, division='fast'):
    _device(rotated)
    width = rotated.shape[1] if width is None else width
    if rotated.ndim != 2 or not rotated.is_contiguous() or maximum.shape != (rotated.shape[0], 1) or maximum.dtype != torch.float32 or maximum.device != rotated.device:
        raise ValueError('Contiguous FP32 storage and matching global row maxima required')
    return quantizer(rotated.shape[0], width, rotated.shape[1], start, division)(rotated, maximum)


def global_row_quantize(x, comm, world_size, groupsize=256, input_act=None, variant='external'):
    import comfy_kitchen as ck
    if variant not in ('external', 'strided', 'packed'): raise ValueError('Unknown fusion variant')
    _device(x); backend = ck.backends.cuda
    flat = backend._apply_input_act(x.reshape(-1, x.shape[-1]).contiguous(), input_act, None, 0.)
    width = flat.shape[-1]; rows = flat.shape[0]; rank = comm.get_rank()
    if width % groupsize: raise ValueError('Whole ConvRot groups must be locally owned')
    h = backend._build_hadamard(groupsize, device=flat.device, dtype=flat.dtype)
    with label('padding'):
        if variant == 'packed': padded = packing_kernel(rows, width, world_size, rank)(flat)
        else:
            padded = torch.zeros((rows, width * world_size), device=flat.device)
            padded[:, rank * width:(rank + 1) * width].copy_(flat)
    # Identical full-width/native FP32 GEMM geometry and input values.
    rotated = backend._rotate_activation(padded, h, groupsize)
    del padded
    if variant == 'external':
        rotated = rotated[:, rank * width:(rank + 1) * width].contiguous()
        start = 0
        with label('local_maximum'): maximum = rotated.abs().amax(-1, keepdim=True)
    else:
        start = rank * width
        with label('local_maximum'): maximum = maximum_kernel(rows, width, width * world_size, start)(rotated)
    comm.all_reduce(maximum, op=comm.ReduceOp.MAX)
    with label('external_max_quantization'): return quantize(rotated, maximum, width, start)


def row_linear(x, weight, weight_scale, comm, world_size, groupsize=256, input_act=None,
               bias=None, residual=None, residual_scale=None, variant='external'):
    from patches import optimized_tp as old
    from patches import parallel_int8 as original
    width = x.shape[-1] // (2 if input_act == 'swiglu' else 1)
    if world_size == 1 or x.dtype != torch.float32 or not old.use_fallback_rotation(x, width * world_size, groupsize):
        return old.row_linear(x, weight, weight_scale, comm, world_size, groupsize, input_act, bias, residual, residual_scale)
    q, scale = global_row_quantize(x, comm, world_size, groupsize, input_act, variant)
    partial = original.int_mm(q, weight)
    comm.all_reduce(partial, op=comm.ReduceOp.SUM)
    out = original.dequantize(partial, scale, weight_scale, x.dtype, bias).reshape(*x.shape[:-1], weight.shape[0])
    if residual is not None: out = torch.addcmul(residual, out.to(x.dtype), residual_scale)
    return out.to(x.dtype)
