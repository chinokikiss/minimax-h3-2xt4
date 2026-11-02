"""Experimental P0-B CTA-owned FC1 buffer reuse; never imported by production.

Run in the pinned comfyui conda environment. This local SM120 qualification is
NOT evidence of physical T4 correctness or full-H3 memory/performance.
"""
from functools import lru_cache
import argparse
import json
import statistics
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


class NoopCommunicator:
    class ReduceOp:
        MAX = 'max'

    def all_reduce(self, maximum, op):
        if op != self.ReduceOp.MAX:
            raise ValueError(op)


@torch.no_grad()
def candidate(x, comm, arch):
    width = x.shape[1] // 2
    view = x.view(torch.float32)
    groups = inplace_kernel(x.shape[0], width, 'float16', arch)(x, view)
    maximum = groups.amax(-1, keepdim=True)
    del groups
    comm.all_reduce(maximum, op=comm.ReduceOp.MAX)
    q, scale = remapped_quantizer(x.shape[0], width, arch)(view, maximum)
    del view, maximum
    return q, scale


@torch.no_grad()
def baseline(x, comm, arch):
    from patches.tilelang_swiglu_quant import fused_kernel
    rows, doubled = x.shape
    width = doubled // 2
    rotated, groups = fused_kernel(rows, width, 'float16', arch, 'stored')(x)
    maximum = groups.amax(-1, keepdim=True)
    del groups
    comm.all_reduce(maximum, op=comm.ReduceOp.MAX)
    result = remapped_quantizer(rows, width, arch, False)(rotated, maximum)
    del rotated, maximum
    return result


@torch.no_grad()
def measure(fn, x, comm, arch, repeats):
    fn(x.clone(), comm, arch)
    torch.cuda.synchronize()
    elapsed = []
    peaks = []
    for _ in range(repeats):
        work = x.clone()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        q, scale = fn(work, comm, arch)
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
        peaks.append((torch.cuda.max_memory_allocated(x.device) - before) / (1024 ** 3))
        del q, scale, work
    return {'median_ms': statistics.median(elapsed),
            'min_ms': min(elapsed), 'max_ms': max(elapsed),
            'phase_increment_peak_gib': max(peaks),
            'samples_ms': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, nargs='+', default=[128, 6814])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device('cuda:0')
    arch = 'sm_' + ''.join(map(str, torch.cuda.get_device_capability(device)))
    comm = NoopCommunicator()
    for rows in args.rows:
        source = torch.randn((rows, 14336), device=device, dtype=torch.float16)
        baseline_q, baseline_scale = baseline(source.clone(), comm, arch)
        trial_q, trial_scale = candidate(source.clone(), comm, arch)
        result = {
            'device': torch.cuda.get_device_name(device), 'arch': arch, 'rows': rows,
            'code_mismatch': int((baseline_q != trial_q).count_nonzero().item()),
            'scale_bit_mismatch': int((baseline_scale.view(torch.int32)
                                       != trial_scale.view(torch.int32)).count_nonzero().item()),
            'baseline': measure(baseline, source, comm, arch, args.repeats),
            'candidate': measure(candidate, source, comm, arch, args.repeats),
        }
        result['bit_exact'] = result['code_mismatch'] == result['scale_bit_mismatch'] == 0
        print(json.dumps(result, sort_keys=True), flush=True)
        del source, baseline_q, baseline_scale, trial_q, trial_scale


if __name__ == '__main__':
    main()
