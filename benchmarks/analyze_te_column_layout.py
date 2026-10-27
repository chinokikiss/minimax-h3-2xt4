"""Summarize paired layout results and traces offline, without provider calls."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from benchmarks.te_column_layout import VARIANTS


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def trace_summary(path):
    events = load(path)['traceEvents']
    summaries = {}
    for name in VARIANTS:
        scopes = [e for e in events if e.get('cat') == 'user_annotation'
                  and e.get('ph') == 'X' and e.get('name') == 'TE_column_layout/' + name]
        assert len(scopes) == 1, (name, len(scopes))
        scope = scopes[0]
        kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X'
                   and scope['ts'] <= e['ts'] and e['ts'] + e['dur'] <= scope['ts'] + scope['dur']]
        assert kernels, 'Missing CUDA coverage'
        families = defaultdict(lambda: dict(events=0, duration_sum_ms=0.))
        copies = []
        for event in kernels:
            kernel = event['name']
            if 'nccl' in kernel.lower():
                family = 'nccl'
            elif 'direct_copy_kernel_cuda' in kernel and ('signed char' in kernel or 'EUlaE_' in kernel):
                # Windows Kineto retains Itanium symbols: the lambda argument
                # code "a" in EUlaE_ denotes signed char (the INT8 payload).
                family = 'int8_copy'
                copies.append(dict(name=kernel, duration_ms=event['dur'] / 1000))
            elif ('igemm' in kernel or 'i8816gemm_s8' in kernel
                  or ('nvjet_' in kernel and '_bii_' in kernel)):
                family = 'int8_gemm'
            else:
                family = 'other'
            families[family]['events'] += 1
            families[family]['duration_sum_ms'] += event['dur'] / 1000
        summaries[name] = dict(cpu_scope_ms=scope['dur'] / 1000,
                               kernel_events=len(kernels), families=dict(families),
                               int8_copy_kernels=copies,
                               limit='Separate instrumented pass; sums include overlap/waiting, not exclusive latency shares')
    return summaries


def summarize(folder):
    ranks = [load(path) for path in sorted(folder.glob('rank[01].json'))]
    assert ranks and all(r['status'] == 'complete' and r['signed_extremes_int32_exact'] for r in ranks)
    physical = all(r['physical_t4_qualification'] for r in ranks)
    if physical:
        assert len(ranks) == 2 and len({r['uuid'] for r in ranks}) == 2
        assert all(r['capability'] == [7, 5] and 'T4' in r['device'] for r in ranks)
        batch = load(folder / 'result.json')
        assert batch['status'] == 'complete' and batch['worker_exit_code'] == 0
    lengths = [c['tokens'] for c in ranks[0]['cases']]
    rows = []
    for length in lengths:
        cases = [next(c for c in rank['cases'] if c['tokens'] == length) for rank in ranks]
        for case in cases:
            assert all(c['byte_exact'] and c['finite'] and c['column_gemm_calls'] == 5
                       for c in case['byte_exact'].values())
            assert all(case['projection_byte_exact'].values())
            assert len({c['output_sha256'] for c in case['byte_exact'].values()}) == 1
        for name in VARIANTS:
            if physical:
                assert cases[0]['byte_exact'][name]['output_sha256'] == cases[1]['byte_exact'][name]['output_sha256']
            baseline = max(c['full_block']['legacy_copy']['median_wall_ms'] for c in cases)
            wall = max(c['full_block'][name]['median_wall_ms'] for c in cases)
            rows.append(dict(tokens=length, variant=name, full_block_ms=wall,
                             speedup_vs_legacy=baseline / wall,
                             qkv_ms=max(c['grouped_columns']['qkv'][name]['median_wall_ms'] for c in cases),
                             gate_up_ms=max(c['grouped_columns']['gate_up'][name]['median_wall_ms'] for c in cases),
                             full_block_incremental_peak_bytes=[c['full_block'][name]['incremental_peak_bytes'] for c in cases]))
    profiles = []
    for rank in ranks:
        assert rank['profile']['status'] == 'cuda_trace_saved'
        trace = folder / rank['profile']['file']
        if not trace.exists():
            trace = trace.with_suffix(trace.suffix + '.gz')
        scopes = trace_summary(trace)
        if physical:
            assert all(scope['families'].get('nccl', {}).get('events') == 4 for scope in scopes.values())
        assert len(rank['gemm_controls']) == 3 * len(lengths)
        assert all(c['int32_exact_all'] for c in rank['gemm_controls'])
        profiles.append(dict(rank=rank['rank'], scopes=scopes))
    return dict(status='complete', physical_t4_qualification=physical,
                topology=ranks[0]['topology'], column_selection=ranks[0]['column_selection'],
                attention=ranks[0]['attention'], rows=rows, profiles=profiles,
                all_int32_projection_and_full_block_checks_byte_exact=True,
                persistent_transposed_weight_cache_bytes=0,
                limits=['Seeded trained-block inputs; not real full-TE distributions or generation quality.',
                        'Local unsharded eager-column/Native-attention smoke differs from physical TP2 paths.',
                        'No full TE, vision, embeddings, weight I/O or H3 generation was timed.',
                        'Hardware allocations and Torch versions differ; local timings cannot predict T4 speedups.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], physical_t4=result['physical_t4_qualification'],
                          configurations=len(result['rows']))))
