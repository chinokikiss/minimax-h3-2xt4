"""Audit recorded samples and write measured tables without extrapolation."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import math

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'kaggle_repaired_output'


def main():
    manifest = json.loads((DATA / 'repaired_manifest.json').read_text())
    rows, checks = [], []
    with tarfile.open(DATA / 'repaired_source.tar.gz', 'r:gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                   for m in archive.getmembers() if m.isfile() and m.name.endswith('.py')}
    def audit_timing(value):
        samples = value['samples_ms']
        assert len(samples) == value['n'] == 5
        assert abs(statistics.median(samples) - value['median_ms']) < 1e-9
        assert all(math.isfinite(v) and v > 0 for v in samples)
        assert value['pre_timing_validation']['accuracy']['valid']
        assert value['post_timing_accuracy']['valid']
        checks.append('five finite positive samples; median recomputed; pre/post validity')
    for suite in ['te', 'dit', 'video']:
        result = json.loads((DATA / f'repaired_results/{suite}/results.json').read_text())
        env = json.loads((DATA / f'repaired_results/{suite}/environment.json').read_text())
        assert env['gpus'] == ['Tesla T4', 'Tesla T4']
        assert env['packages']['torch'] == '2.11.0+cu130'
        assert sources['benchmarks/retest.py'] == env['source_sha256']
        for name, digest in result['source_sha256'].items():
            assert sources[name] == digest
        for workload, values in result['workloads'].items():
            for mode in ['native', 'tp', 'sp']:
                if mode not in values:
                    continue
                value = values[mode]
                audit_timing(value)
                accuracy = value['post_timing_accuracy']
                if mode != 'native':
                    assert value['pre_timing_validation']['accuracy']['is_bit_exact']
                    assert accuracy['is_bit_exact'] and accuracy['rel_l2'] == 0
                rows.append({'suite': suite, 'workload': workload, 'method': mode,
                    'median_ms': value['median_ms'], 'p95_ms': value['p95_ms'],
                    'median_bootstrap_95ci_ms': value['median_bootstrap_95ci_ms'],
                    'peak_allocated_gib': value['peak_allocated_gib'],
                    'peak_reserved_gib': value['peak_reserved_gib'],
                    'is_bit_exact_vs_native': accuracy['is_bit_exact'] if mode != 'native' else None,
                    'samples_ms': value['samples_ms']})
    slabs = json.loads((DATA / 'repaired_results/slabs/results.json').read_text())
    assert slabs['status'] == 'complete'
    assert all(p['int8_tensorcore_gemm_confirmed'] and p['int8_attention_confirmed']
               and not p['fp32_sgemm_observed'] for p in slabs['native_kernel_probes'])
    for case in slabs['cases']:
        audit_timing(case['timing'])
        exact_method = case['accuracy_vs_matching_slabs']
        assert exact_method['rel_l2'] <= 1e-5 and exact_method['max_abs'] <= .002
        native = next(r for r in rows if r['suite'] == 'video' and r['method'] == 'native' and r['workload'] == case['name'])
        rows.append({'suite': 'slabs', 'workload': case['name'], 'method': case['variant'],
            'median_ms': case['timing']['median_ms'], 'p95_ms': case['timing']['p95_ms'],
            'samples_ms': case['timing']['samples_ms'],
            'median_bootstrap_95ci_ms': case['timing']['median_bootstrap_95ci_ms'],
            'native_wall_time_ratio': native['median_ms'] / case['timing']['median_ms'],
            'ratio_scope': 'same allocation, but native input resident on GPU; slab CPU input transfers included',
            'historical_best_tile_ratio': case['historical_best_2gpu_tile_time_ratio'],
            'historical_ratio_scope': 'different allocation',
            'matching_slabs': exact_method, 'full_decode_accuracy': case['accuracy_vs_full_saved_raw'],
            'strict_full_decode_gate_passed': case['strict_full_decode_gate_passed'],
            'peak_allocated_gib': case['timing']['peak_allocated_gib'],
            'peak_reserved_gib': case['timing']['peak_reserved_gib'],
            'gpu_utilization_mean_percent': case['utilization']['gpu_mean_percent'],
            'output_transfer_ms_median': case['timing']['output_transfer_ms_median'],
            'stitch_ms_median': case['timing']['stitch_ms_median'],
            'output_transfer_bytes': case['geometry']['output_transfer_bytes'],
            'slab_shapes': case['geometry']['slab_shapes']})
    audit = {'passed': True, 'timing_rows': len(rows), 'timing_samples': sum(len(r['samples_ms']) for r in rows),
             'physical_bit_exact_proposal_comparisons': sum(r['is_bit_exact_vs_native'] is True for r in rows if 'is_bit_exact_vs_native' in r),
             'checks': sorted(set(checks)), 'source_archive_sha256': hashlib.sha256((DATA / 'repaired_source.tar.gz').read_bytes()).hexdigest(),
             'overall_run_status': manifest['status'], 'rejected_paths': ['video TP failed pre-timing reference gate'],
             'unattempted_paths': ['video SP: shared TP core failed first'],
             'limits': 'suite JSON, timing arithmetic and source fingerprints audited; stored distributed rank-0 outputs are bit-exact, other ranks passed numerical gates; no saved tensor or full kernel trace audit'}
    (DATA / 'aggregate.json').write_text(json.dumps(rows, indent=2))
    (DATA / 'audit.json').write_text(json.dumps(audit, indent=2))
    table = ['# Repaired dual-T4 measured results', '',
             'TE/DiT are block-0 tests. Video is the full raw decoder. Five samples per accepted row.', '',
             '| Suite | Workload | Method | Median ms | P95 ms | Peak allocated GiB (GPU 0 / 1) | Accuracy |',
             '|---|---|---|---:|---:|---|---|']
    for row in rows:
        accuracy = ('self-validity baseline' if row['method'] == 'native' else 'bit-exact vs native')
        if row['suite'] == 'slabs':
            accuracy = f"full L2 {row['full_decode_accuracy']['rel_l2'] * 100:.3f}%; matching-slab gate passed"
        peaks = ' / '.join(f'{v:.3f}' for v in row['peak_allocated_gib'])
        table.append(f"| {row['suite']} | {row['workload']} | {row['method']} | {row['median_ms']:.2f} | {row['p95_ms']:.2f} | {peaks} | {accuracy} |")
    table += ['', 'Video TP failed before timing (L2 1.5776%, maximum error 0.2869); video SP was not attempted afterward.', '',
              'TE native timings include mask/RoPE construction; TP/SP use precomputed arguments. TE TP/SP are slower even against that larger native scope. Avoid a precise optimization speedup claim from these boundaries.', '',
              'Native video and full TP/SP use resident GPU inputs; slabs include CPU input transfers. Slab/native wall ratios therefore have slightly different input boundaries. Historical tile ratios also span allocations.', '',
              'Slabs fail the original strict full-decoder quality gate. GPU utilization averages cover validation, warmup and timing windows. Cross-device copy durations do not identify PCIe wire transport.', '',
              'One allocation, fixed native/TP/SP order and one warmup; bootstrap intervals describe sample variation within a process, not session-level uncertainty. No full-generation latency or perceptual quality is established.']
    (DATA / 'MEASURED_RESULTS.md').write_text('\n'.join(table) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
