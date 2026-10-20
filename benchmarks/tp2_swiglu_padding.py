"""Physical two-T4 SwiGLU/padding validation at audited TP2 shapes; no H3 run."""
import argparse
from contextlib import contextmanager
import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
import torch.distributed as dist
import comfy_kitchen as ck
from patches import optimized_tp as old, parallel_int8 as pi
from patches import tilelang_row_quant as previous, tilelang_swiglu_pad as fused
from tp2_timing import measure


@contextmanager
def label(name):
    torch.cuda.nvtx.range_push(name)
    try: yield
    finally: torch.cuda.nvtx.range_pop()


def save(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def eager_pad(gate, up, rank, world=2):
    activated = torch.nn.functional.silu(gate).mul_(up)
    out = torch.zeros((*gate.shape[:-1], gate.shape[-1]*world), device=gate.device)
    out[..., rank*gate.shape[-1]:(rank+1)*gate.shape[-1]].copy_(activated)
    return out


def units(rank, output):
    torch.manual_seed(20261018); records = []
    for width in (1, 257):
        base = torch.randn(7, width*2, device='cuda')
        for kind in ('normal', 'zero', 'subnormal', 'large', 'nonfinite'):
            packed = base.clone()
            if kind == 'zero': packed.zero_(); packed[0, 0] = -0.; packed[1, -1] = -0.
            elif kind == 'subnormal':
                packed[:, :width] *= 1.e-40; packed[:, width:] = 1.; packed[1, width:] = -1.
            elif kind == 'large': packed *= 100.; packed[:, 0] = -100.
            elif kind == 'nonfinite':
                packed[0, 0] = float('nan'); packed[1, 0] = float('inf'); packed[2, 0] = -float('inf')
                packed[3, -1] = float('nan'); packed[4, -1] = float('inf'); packed[5, -1] = -float('inf')
            before = packed.clone(); g, u = packed.chunk(2, -1)
            for separate in (False, True):
                gate, up = (g.contiguous(), u.contiguous()) if separate else (g, u)
                for owner in (0, 1):
                    actual = fused.swiglu_pad(gate, up, rank=owner)
                    record = dict(width=width, pattern=kind, separate=separate, owner=owner,
                                  bitwise_equal=equal(actual, eager_pad(gate, up, owner)), input_unchanged=equal(packed, before))
                    records.append(record); save(output/f'unit_rank{rank}.json', records)
                    assert record['bitwise_equal'] and record['input_unchanged'], record
    packed = torch.randn(2, 3, 14, device='cuda'); g, u = packed.chunk(2, -1)
    same = equal(fused.swiglu_pad_packed(packed, rank=1, world_size=3), eager_pad(g, u, 1, 3))
    records.append(dict(test='leading_dimensions_world3', bitwise_equal=same)); save(output/f'unit_rank{rank}.json', records)
    assert same
    print('SWIGLU_UNIT_GATES', rank, len(records), 'pass', flush=True)


class CaptureComm:
    """Record actual local/global contributions around REAL NCCL collectives."""
    ReduceOp = dist.ReduceOp
    def __init__(self): self.values = []
    def get_rank(self): return dist.get_rank()
    def all_reduce(self, x, op):
        before = x.clone(); dist.all_reduce(x, op=op)
        self.values.append((str(op), before, x.clone()))


def instrument():
    def wrap(owner, name, tag):
        original = getattr(owner, name)
        def call(*a, **kw):
            with label(tag): return original(*a, **kw)
        setattr(owner, name, call)
    for name in ('_apply_input_act', '_rotate_activation', 'quantize_int8_rowwise'):
        wrap(ck.backends.cuda, name, 'linear/'+name)
    wrap(pi, 'dequantize', 'linear/dequantize')
    def mm(*a, **kw):
        with label('linear/INT8_GEMM_to_INT32'): return old.native_int_mm(*a, **kw)
    pi.int_mm = mm
    original = dist.all_reduce
    def reduction(x, *a, **kw):
        with label('NCCL/all_reduce/'+str(x.dtype)+'/'+str(kw.get('op', ''))): return original(x, *a, **kw)
    dist.all_reduce = reduction


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path); ap.add_argument('--round', type=int, default=0)
    ap.add_argument('--profile', action='store_true'); args = ap.parse_args()
    rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.device_count() == 2 and all(torch.cuda.get_device_capability(i) == (7, 5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert os.environ.get('NCCL_P2P_DISABLE') == '1'
    assert os.environ.get('NCCL_MIN_NCHANNELS') == os.environ.get('NCCL_MAX_NCHANNELS') == '4'
    assert 'NCCL_PROTO' not in os.environ
    assert importlib.metadata.version('comfy-kitchen') == '0.2.37'
    assert importlib.metadata.version('tilelang') == '0.1.15'
    assert old.STREAM_ROWS == 0, 'This fusion validates current default scalemax, not optional streamed rotation'
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank), timeout=datetime.timedelta(seconds=75))
    instrument(); args.output.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', rank=rank, hardware=torch.cuda.get_device_name(rank), capability=[7, 5],
                  torch=torch.__version__, nccl=torch.cuda.nccl.version(), cases=[],
                  scope='Synthetic fixed-seed TP2 operator conformance and performance; real NCCL MAX/SUM. No model or pipeline generation.')
    target = args.output/f'rank{rank}.json'; save(target, result)
    if not args.profile and args.round == 0: units(rank, args.output)
    targets = [r for r in json.loads(args.manifest.read_text())['GEMMs'] if (r['component'], r['module']) in (('te', 'down_proj'), ('dit', 'fc2'))]
    assert len(targets) == 2
    for row in targets:
        component = row['component']; m, n, k = (row[a] for a in ('M', 'N', 'K'))
        assert (m, n, k) == ((353, 5120, 12800) if component == 'te' else (6814, 5376, 7168))
        torch.manual_seed(20261018 + rank + (99 if component == 'dit' else 0))
        if component == 'te':
            gate = torch.randn(m, k, device='cuda')*.25; up = torch.randn_like(gate)*.25; x = gate
        else:
            x = torch.randn(m, k*2, device='cuda')*.25; gate, up = x.chunk(2, -1)
        w = torch.randint(-127, 128, (n, k), device='cuda', dtype=torch.int8)
        # Replicated bias/residual/scale must agree across ranks; local activations/weights need not.
        torch.manual_seed(9918 + (99 if component == 'dit' else 0))
        ws = torch.full((n, 1), .001, device='cuda'); bias = torch.randn(n, device='cuda')*.01
        residual = torch.randn(m, n, device='cuda')*.01; residual_scale = torch.full((m, 1), .5, device='cuda')
        assert old.use_fallback_rotation(x, k*2, 256), 'Unexpected target rotation dispatch; no forced-route adapter permitted'
        variants = ['eager', 'packed', 'fused'] if args.round == 0 and not args.profile else ['packed', 'fused']
        if args.round % 2: variants.reverse()
        def padding(variant):
            with label(component+'/'+variant+'/padding_API'):
                if variant == 'fused': return fused.swiglu_pad(gate, up, rank=rank)
                if variant == 'eager': return eager_pad(gate, up, rank)
                return previous.packing_kernel(m, k, 2, rank)(torch.nn.functional.silu(gate).mul_(up))
        def quantization(variant, comm=dist):
            with label(component+'/'+variant+'/quantization_API'):
                if variant == 'fused':
                    if component == 'dit': return fused.global_row_quantize(x, comm, 2)
                    return fused._quantize_padded(fused.swiglu_pad(gate, up, rank=rank), k, comm, 2, 256)
                value = x if component == 'dit' else torch.nn.functional.silu(gate).mul_(up)
                act = 'swiglu' if component == 'dit' else None
                if variant == 'eager': return old.global_row_quantize(value, comm, 2, 256, act)
                return previous.global_row_quantize(value, comm, 2, 256, act, 'packed')
        def api(variant, comm=dist):
            with label(component+'/'+variant+'/API'):
                kw = dict(bias=bias, residual=residual, residual_scale=residual_scale)
                if variant == 'fused':
                    if component == 'te': return fused.row_linear_parts(gate, up, w, ws, comm, 2, **kw)
                    return fused.row_linear(x, w, ws, comm, 2, input_act='swiglu', **kw)
                value = x if component == 'dit' else torch.nn.functional.silu(gate).mul_(up)
                act = 'swiglu' if component == 'dit' else None
                if variant == 'eager': return old.row_linear(value, w, ws, comm, 2, input_act=act, **kw)
                return previous.row_linear(value, w, ws, comm, 2, input_act=act, variant='packed', **kw)
        # Reference is frozen production scalemax, including actual MAX then INT32 SUM.
        expected_pad = eager_pad(gate, up, rank)
        reference_q, reference_s = quantization('eager')
        reference_comm = CaptureComm(); expected = api('eager', reference_comm); torch.cuda.synchronize()
        assert len(reference_comm.values) == 2 and reference_comm.values[0][1].dtype == torch.float32 and reference_comm.values[1][1].dtype == torch.int32
        for variant in variants:
            name = component+'/'+variant; record = dict(component=component, module=row['module'], variant=variant, target=row)
            dist.barrier(); start = time.perf_counter(); error = None
            try:
                padded = padding(variant); q, s = quantization(variant)
                capture = CaptureComm(); actual = api(variant, capture); torch.cuda.synchronize()
                contributions = len(capture.values) == len(reference_comm.values) and all(
                    a[0] == b[0] and equal(a[1], b[1]) and equal(a[2], b[2]) for a, b in zip(capture.values, reference_comm.values))
                record.update(first_call_including_JIT_seconds=time.perf_counter()-start,
                              padded_bitwise_equal=equal(padded, expected_pad), q_bitwise_equal=equal(q, reference_q),
                              scale_bitwise_equal=equal(s, reference_s), output_bitwise_equal=equal(actual, expected),
                              local_and_global_MAX_INT32_SUM_bitwise_equal=contributions,
                              output_sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest())
                assert all(record[key] for key in record if key.endswith('bitwise_equal')), record
            except Exception as e: error = type(e).__name__+': '+str(e)
            errors = [None, None]; dist.all_gather_object(errors, error)
            if any(errors):
                record.update(status='failed', errors=errors); result['cases'].append(record); save(target, result)
                raise RuntimeError('Conformance gate failed: '+str(errors))
            del padded, q, s, actual, capture
            if args.profile:
                for _ in range(5): api(variant)
                torch.cuda.synchronize(); dist.barrier(); torch.cuda.profiler.start()
                for _ in range(3): api(variant)
                torch.cuda.synchronize()
                # Both processes finish all three calls before either ends capture.
                with label('profile/rank_completion_barrier'): dist.barrier(); torch.cuda.synchronize()
                torch.cuda.profiler.stop()
            else:
                record['padding_API'] = measure(lambda: padding(variant), repeats=15)
                record['quantization_API'] = measure(lambda: quantization(variant), repeats=15)
                torch.cuda.reset_peak_memory_stats(); before = torch.cuda.memory_allocated()
                record['API'] = measure(lambda: api(variant), repeats=15)
                record['torch_peak_extra_allocated_bytes'] = torch.cuda.max_memory_allocated()-before
            record['status'] = 'complete'; result['cases'].append(record); save(target, result)
            print('SWIGLU_CANDIDATE', rank, name, 'complete', flush=True)
        del reference_comm, expected_pad, reference_q, reference_s, expected, x, gate, up, w, ws, bias, residual, residual_scale
        gc.collect(); torch.cuda.empty_cache()
    result['status'] = 'complete'; save(target, result)
    collected = [None, None]; dist.all_gather_object(collected, result)
    if rank == 0 and not args.profile:
        summary = {}
        for component in ('te', 'dit'):
            summary[component] = {}
            for phase in ('padding_API', 'quantization_API', 'API'):
                scores = {v: max(next(r for r in item['cases'] if r['component'] == component and r['variant'] == v)[phase]['gpu_median_ms'] for item in collected) for v in variants}
                summary[component][phase] = dict(slower_rank_medians_ms=scores, fused_reduction_vs_packed_pct=100*(1-scores['fused']/scores['packed']))
        save(args.output/'summary.json', summary)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__': main()
