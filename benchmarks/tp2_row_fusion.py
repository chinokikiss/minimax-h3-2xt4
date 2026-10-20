"""Physical two-T4 external-max fusion screen; synthetic current TP2 shapes."""
import argparse
from contextlib import contextmanager
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
import torch.distributed as dist
import comfy_kitchen as ck
from patches import optimized_tp as old, parallel_int8 as pi, tilelang_row_quant as new
from tp2_timing import measure


@contextmanager
def label(name):
    torch.cuda.nvtx.range_push(name)
    try: yield
    finally: torch.cuda.nvtx.range_pop()


def save(path, value):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def instrument():
    def wrap(owner, name, tag):
        function = getattr(owner, name)
        def call(*a, **kw):
            with label(tag): return function(*a, **kw)
        setattr(owner, name, call)
    for name in ('_apply_input_act', '_rotate_activation', 'quantize_int8_rowwise'):
        wrap(ck.backends.cuda, name, 'linear/' + name)
    wrap(pi, 'dequantize', 'linear/dequantize')
    def mm(*a, **kw):
        with label('linear/INT8_GEMM_to_INT32'): return old.native_int_mm(*a, **kw)
    pi.int_mm = mm
    function = dist.all_reduce
    def reduction(x, *a, **kw):
        with label('NCCL/all_reduce/' + str(x.dtype) + '/' + str(kw.get('op', ''))): return function(x, *a, **kw)
    dist.all_reduce = reduction


def unit_gate(rank, output):
    torch.manual_seed(4096 + rank); rows = []
    for width in (1, 257, 3584, 7168):
        for kind in ('zero', 'tiny', 'normal', 'outlier', 'ties'):
            x = torch.randn((7, width), device=rank)
            if kind == 'zero': x.zero_()
            elif kind == 'tiny': x *= 1.e-35
            elif kind == 'outlier': x[:, -1] = 1.e12
            elif kind == 'ties':
                values = torch.arange(width, device=rank).remainder(253).float() - 126.5
                scale = torch.tensor(3.14159265, device=rank) * torch.tensor(1. / 127., device=rank)
                x[:] = values * scale
                x[1] = torch.nextafter(x[1], torch.full_like(x[1], float('inf')))
                x[2] = torch.nextafter(x[2], torch.full_like(x[2], -float('inf')))
            maximum = torch.full((7, 1), 3.14159265, device=rank) if kind == 'ties' else x.abs().amax(-1, keepdim=True) * 1.125
            q, s = ck.backends.cuda.quantize_int8_rowwise(torch.cat((x, maximum), -1)); q = q[:, :width]
            fast = new.quantize(x, maximum); rn = new.quantize(x, maximum, division='rn')
            row = dict(width=width, pattern=kind, fast_q_equal=torch.equal(q, fast[0]),
                       scale_equal=torch.equal(s, fast[1]), rn_q_equal=torch.equal(q, rn[0]),
                       rn_differing_elements=int((q != rn[0]).sum()))
            rows.append(row); save(output / f'unit_rank{rank}.json', rows)
            assert row['fast_q_equal'] and row['scale_equal'], row
    for width in (3584, 7168):
        x = torch.randn((7, width), device=rank)
        expected = torch.zeros((7, width * 2), device=rank); expected[:, rank * width:(rank + 1) * width] = x
        actual = new.packing_kernel(7, width, 2, rank)(x)
        maximum = new.maximum_kernel(7, width, width * 2, rank * width)(actual)
        assert torch.equal(actual, expected) and torch.equal(maximum, x.abs().amax(-1, keepdim=True))
    print('ROW_UNIT_GATE', rank, 'pass', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path); ap.add_argument('--selection', type=Path)
    ap.add_argument('--round', type=int, default=0); ap.add_argument('--profile', action='store_true'); args = ap.parse_args()
    rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.device_count() == 2 and all(torch.cuda.get_device_capability(i) == (7, 5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert os.environ.get('NCCL_P2P_DISABLE') == '1' and os.environ.get('NCCL_MIN_NCHANNELS') == os.environ.get('NCCL_MAX_NCHANNELS') == '4'
    assert 'NCCL_PROTO' not in os.environ
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank), timeout=datetime.timedelta(seconds=75)); instrument()
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', rank=rank, torch=torch.__version__, nccl=torch.cuda.nccl.version(),
                  hardware=torch.cuda.get_device_name(rank), cases=[],
                  scope='Unchanged current TP2 row contract; deterministic synthetic A/W, no real linear quality claim or full H3 pipeline.')
    target = args.output / f'rank{rank}.json'; save(target, result)
    if not args.selection and not args.profile: unit_gate(rank, args.output)
    selection = json.loads(args.selection.read_text()) if args.selection else None
    targets = [r for r in json.loads(args.manifest.read_text())['GEMMs'] if r['component'] == 'dit' and r['partition'] == 'row']
    if args.profile: targets = [r for r in targets if r['module'] == 'fc2']
    for index, row in enumerate(targets):
        # The source module name, not loop position, fixes inputs across screen/repeat/profile.
        torch.manual_seed(20261016 + rank + (99 if row['module'] == 'fc2' else 0))
        m, n, k = [row[a] for a in ('M', 'N', 'K')]; act = row['input_act']
        x = torch.randn((m, k * (2 if act == 'swiglu' else 1)), device=rank) * .25
        w = torch.randint(-127, 128, (n, k), device=rank, dtype=torch.int8)
        ws = torch.full((n, 1), .001, device=rank)
        assert old.use_fallback_rotation(x, k * 2, 256)
        dist.barrier(); reference_q, reference_s = old.global_row_quantize(x, dist, 2, 256, act)
        expected = old.row_linear(x, w, ws, dist, 2, input_act=act); torch.cuda.synchronize()
        variants = ['frozen', 'external', 'strided', 'packed']
        if selection:
            winner = selection['winners'][row['module']]['variant']; variants = [v for v in variants if v in ('frozen', winner)]
        random.Random(990 + args.round).shuffle(variants)
        for variant in variants:
            name = 'dit/' + row['module'] + '/' + variant
            def quantization():
                with label(name + '/quantization_API'):
                    return old.global_row_quantize(x, dist, 2, 256, act) if variant == 'frozen' else new.global_row_quantize(x, dist, 2, 256, act, variant)
            def api():
                with label(name + '/API'):
                    return old.row_linear(x, w, ws, dist, 2, input_act=act) if variant == 'frozen' else new.row_linear(x, w, ws, dist, 2, input_act=act, variant=variant)
            record = dict(module=row['module'], variant=variant, target=row); error = None
            dist.barrier(); start = time.perf_counter()
            try:
                q, s = quantization(); actual = api(); torch.cuda.synchronize()
                record.update(first_call_including_JIT_seconds=time.perf_counter() - start,
                              q_equal=torch.equal(q, reference_q), scale_equal=torch.equal(s, reference_s),
                              output_equal=torch.equal(actual, expected),
                              output_sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest())
                assert record['q_equal'] and record['scale_equal'] and record['output_equal'], record
            except Exception as e: error = type(e).__name__ + ': ' + str(e)
            errors = [None, None]; dist.all_gather_object(errors, error)
            if any(errors):
                record.update(status='failed', errors=errors); result['cases'].append(record); save(target, result)
                print('ROW_CANDIDATE_FAILED', rank, name, errors, flush=True); continue
            del q, s, actual
            if args.profile:
                for _ in range(5): api()
                torch.cuda.synchronize(); dist.barrier(); torch.cuda.profiler.start()
                for _ in range(3): api()
                torch.cuda.synchronize(); torch.cuda.profiler.stop()
            else:
                record['quantization_API'] = measure(quantization, repeats=15)
                torch.cuda.reset_peak_memory_stats(); before = torch.cuda.memory_allocated()
                record['API'] = measure(api, repeats=15)
                record['peak_extra_allocated_bytes'] = torch.cuda.max_memory_allocated() - before
            record['status'] = 'complete'; result['cases'].append(record); save(target, result)
            print('ROW_CANDIDATE', rank, name, 'complete', flush=True)
        del x, w, ws, reference_q, reference_s, expected; gc.collect(); torch.cuda.empty_cache()
    result['status'] = 'complete'; save(target, result)
    collected = [None, None]; dist.all_gather_object(collected, result)
    if rank == 0 and not args.profile:
        winners = {}
        for module in sorted({r['module'] for r in result['cases']}):
            maps = [{r['variant']: r for r in a['cases'] if r['module'] == module and r['status'] == 'complete'} for a in collected]
            scores = {v: max(a[v]['API']['gpu_median_ms'] for a in maps) for v in set(maps[0]) & set(maps[1])}
            v = min(scores, key=scores.get); winners[module] = dict(variant=v, slower_rank_API_ms=scores[v], all_slower_rank_API_ms=scores)
        save(args.output / 'selection.json', dict(winners=winners))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__': main()
