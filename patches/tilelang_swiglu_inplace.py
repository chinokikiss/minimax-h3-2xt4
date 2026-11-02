"""Production SM75 DiT FC2 P0-B: CTA-owned in-place SwiGLU ConvRot256.

The equal-byte FP16 FC1 source is reinterpreted as an FP32 rotation
workspace. Each CTA writes only the storage region it has finished reading.
The quantizer restores canonical column ordering without materializing an
independent rotated FP32 tensor. No benchmark modules are imported.
"""
from functools import lru_cache
import torch


@lru_cache(maxsize=16)
def inplace_kernel(rows, width, dtype, arch):
    import tilelang
    import tilelang.language as T

    if dtype != 'float16' or arch not in ('sm_75', 'sm_120') or width % 256:
        raise ValueError('P0-B requires exact FP16 ConvRot256 geometry')
    groups = width // 256

    @T.macro
    def transform(X, A, B, row, group):
        for i in T.Parallel(256):
            gate = T.cast(X[row, group * 256 + i], 'float32')
            up = T.cast(X[row, width + group * 256 + i], 'float32')
            denominator = T.call_pure_extern('float32', '__fadd_rn', T.float32(1.),
                T.call_pure_extern('float32', 'expf', -gate))
            activated = T.call_pure_extern('float32', '__fdiv_rn', gate, denominator)
            A[i] = T.call_pure_extern('float32', '__fmul_rn', activated, up)
        T.sync_threads()
        for stage in T.unroll(4):
            for i in T.Parallel(256):
                stride = 1 << (2 * stage)
                digit = (i // stride) % 4
                base = i - digit * stride
                a = A[base]
                b = A[base + stride]
                c = A[base + 2 * stride]
                d = A[base + 3 * stride]
                p = T.if_then_else(digit == 3, -a, a)
                q = T.if_then_else(digit == 2, -b, b)
                r = T.if_then_else(digit == 1, -c, c)
                s = T.if_then_else(digit == 0, -d, d)
                B[i] = T.call_pure_extern('float32', '__fadd_rn',
                    T.call_pure_extern('float32', '__fadd_rn',
                        T.call_pure_extern('float32', '__fadd_rn', p, q), r), s)
            T.sync_threads()
            for i in T.Parallel(256):
                A[i] = B[i]
            T.sync_threads()

    @T.prim_func
    def main(X: T.Tensor((rows, 2 * width), dtype),
             R: T.Tensor((rows, width), 'float32'),
             M: T.Tensor((rows, groups), 'float32')):
        with T.Kernel(rows, groups, threads=256) as (row, group):
            a = T.alloc_shared((256,), 'float32')
            b = T.alloc_shared((256,), 'float32')
            absolute = T.alloc_fragment((256,), 'float32')
            maximum = T.alloc_fragment((1,), 'float32')
            transform(X, a, b, row, group)
            for i in T.Parallel(256):
                value = a[i] * T.float32(1. / 16.)
                absolute[i] = T.abs(value)
                if i < 128:
                    R[row, group * 128 + i] = value
                else:
                    R[row, width // 2 + group * 128 + (i - 128)] = value
            T.reduce_max(absolute, maximum, dim=0)
            if T.get_thread_binding() == 0:
                M[row, group] = maximum[0]

    return tilelang.compile(main, out_idx=[2], target={'kind': 'cuda', 'arch': arch},
                            pass_configs={'tl.enable_async_copy': False})


@lru_cache(maxsize=16)
def remapped_quantizer(rows, width, arch, remapped=True):
    import tilelang
    import tilelang.language as T

    @T.prim_func
    def quant(R: T.Tensor((rows, width), 'float32'),
              Maximum: T.Tensor((rows, 1), 'float32'),
              Q: T.Tensor((rows, width), 'int8'),
              Scale: T.Tensor((rows, 1), 'float32')):
        with T.Kernel(rows, threads=256) as row:
            scale = T.alloc_local((1,), 'float32')
            maximum = T.call_pure_extern('float32', 'fminf', Maximum[row, 0],
                T.float32(3.4028234663852886e38))
            scale[0] = T.call_pure_extern('float32', 'fmaxf',
                maximum * T.float32(1. / 127.), T.float32(1.e-30))
            for col in T.Parallel(width):
                group = col // 256
                i = col % 256
                if remapped:
                    source = T.if_then_else(i < 128, group * 128 + i,
                        width // 2 + group * 128 + (i - 128))
                else:
                    source = col
                scaled = T.call_pure_extern('float32', '__fdividef',
                    R[row, source], scale[0])
                rounded = T.call_pure_extern('float32', 'nearbyintf', scaled)
                clipped = T.call_pure_extern('float32', 'fminf', T.float32(127.),
                    T.call_pure_extern('float32', 'fmaxf', T.float32(-128.), rounded))
                Q[row, col] = T.cast(clipped, 'int8')
            if T.get_thread_binding() == 0:
                Scale[row, 0] = scale[0]

    return tilelang.compile(quant, out_idx=[2, 3],
        target={'kind': 'cuda', 'arch': arch},
        pass_configs={'tl.enable_async_copy': False})



def prepare(x,comm,*,mode,arch,consume_input):
    """Return None unless the physically qualified DiT FC2 owner is safe."""
    if (mode != 'fused_stored' or arch != 'sm_75' or not consume_input
            or type(x) is not torch.Tensor
            or torch.is_grad_enabled() or torch.compiler.is_compiling()
            or not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2
            or x.shape[0] <= 0 or x.shape[1] != 14336
            or not x.is_contiguous() or x._base is not None
            or x.storage_offset() != 0
            or torch.cuda.get_device_capability(x.device) != (7,5)):
        return None
    rows=x.shape[0]
    width=x.shape[1]//2
    # Both views share the single owner allocation, exactly 2*width FP16
    # elements / width FP32 elements per row. There is no extra rotation.
    rotated_view=x.view(torch.float32)
    groups=inplace_kernel(rows,width,'float16',arch)(x,rotated_view)
    maximum=groups.amax(-1,keepdim=True)
    del groups
    comm.all_reduce(maximum,op=comm.ReduceOp.MAX)
    codes,scale=remapped_quantizer(rows,width,arch)(rotated_view,maximum)
    # Kernel dispatch is sequenced on the current CUDA stream; no later
    # reader needs the source. Detach only after quantization has launched.
    del rotated_view,maximum
    x.set_()
    return codes,scale
