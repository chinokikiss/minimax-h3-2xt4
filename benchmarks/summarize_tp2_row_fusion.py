"""CPU-only paired analysis; preserve per-round baselines and failed gates."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root', type=Path); args = ap.parse_args(); root = args.root
    state = json.loads((root / 'status.json').read_text()); sources = json.loads((root / 'source_hashes.json').read_text())
    assert sources == json.loads((root.parent.parent / 'submitted_sources.json').read_text())
    result = dict(status=state['status'], hardware=state.get('hardware'), cases=[], failures=[], unit_gates=[], candidates=[],
                  scope='Synthetic finite FP32 inputs at source-audited current TP2 shapes; no H3 pipeline or real-distribution linear-quality claim.')
    hashes = {}
    for path in sorted((root / 'screen').glob('unit_rank*.json')):
        rows = json.loads(path.read_text())
        result['unit_gates'].append(dict(file=path.name, tests=len(rows), all_fast_q_and_scale_equal=all(r['fast_q_equal'] and r['scale_equal'] for r in rows),
            rn_differing_elements=sum(r['rn_differing_elements'] for r in rows)))
    for round_name in ('screen', 'repeat'):
        folder = root / round_name
        if not all((folder / f'rank{i}.json').exists() for i in (0, 1)): continue
        ranks = [json.loads((folder / f'rank{i}.json').read_text()) for i in (0, 1)]
        maps = [{(r['module'], r['variant']): r for r in a['cases']} for a in ranks]
        for rank, data in enumerate(ranks):
            for row in data['cases']:
                if row['status'] != 'complete': continue
                frozen = next(r for r in data['cases'] if r['module'] == row['module'] and r['variant'] == 'frozen' and r['status'] == 'complete')
                assert row['output_sha256'] == frozen['output_sha256'], (round_name, rank, row['module'], row['variant'], 'output bytes differ')
                previous = hashes.setdefault((rank, row['module']), row['output_sha256'])
                assert previous == row['output_sha256'], 'Output bytes changed between rounds'
        for key in sorted(set(maps[0]) & set(maps[1])):
            rows = [a[key] for a in maps]
            if any(r['status'] != 'complete' for r in rows):
                result['failures'].append(dict(round=round_name, module=key[0], variant=key[1], errors=[r.get('errors') for r in rows])); continue
            record = dict(round=round_name, module=key[0], variant=key[1],
                exact_q_scale_output=all(r['q_equal'] and r['scale_equal'] and r['output_equal'] for r in rows),
                API_ms=max(r['API']['gpu_median_ms'] for r in rows),
                wall_ms=max(r['API']['wall_median_ms'] for r in rows),
                quantization_API_ms=max(r['quantization_API']['gpu_median_ms'] for r in rows),
                max_rank_peak_extra_allocated_bytes=max(r['peak_extra_allocated_bytes'] for r in rows),
                rank_API_medians_ms=[r['API']['gpu_median_ms'] for r in rows])
            result['cases'].append(record)
    for row in result['cases']:
        baseline = next(r for r in result['cases'] if r['round'] == row['round'] and r['module'] == row['module'] and r['variant'] == 'frozen')
        row['latency_reduction_percent'] = 100 * (1 - row['API_ms'] / baseline['API_ms'])
    for row in result['cases']:
        if row['round'] != 'repeat' or row['variant'] == 'frozen': continue
        screen = next(r for r in result['cases'] if r['round'] == 'screen' and r['module'] == row['module'] and r['variant'] == row['variant'])
        result['candidates'].append(dict(module=row['module'], variant=row['variant'],
            wins_both_rounds=row['latency_reduction_percent'] > 0 and screen['latency_reduction_percent'] > 0,
            screen_reduction_percent=screen['latency_reduction_percent'], repeat_reduction_percent=row['latency_reduction_percent']))
    result['recorded_stage_seconds'] = sum(r.get('seconds', 0) for r in state['commands'])
    result['all_complete_output_hashes_equal_across_variants_and_rounds'] = True
    result['accounting'] = 'Subprocess durations, not Kaggle quota billing.'
    profiled = list((root / 'profiled').glob('rank*.json'))
    for path in profiled:
        data = json.loads(path.read_text())
        for row in data['cases']:
            assert row['status'] == 'complete' and row['q_equal'] and row['scale_equal'] and row['output_equal']
            assert row['output_sha256'] == hashes[(data['rank'], row['module'])], 'Profiled output bytes changed'
    result['profiled_output_hashes_checked'] = len(profiled)
    (root / 'paired_summary.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
    qualified = (not result['failures'] and len(result['unit_gates']) == 2
                 and all(r['all_fast_q_and_scale_equal'] and r['tests'] == 20 for r in result['unit_gates'])
                 and all(r['exact_q_scale_output'] for r in result['cases']))
    if qualified:
        selected = dict(kernel='chinokiki9/h3-tp2-row-fusion', version=1, status='operator_verified_candidates',
            hardware=result['hardware'], source_sha256=sources,
            native_contract_source=json.loads((root.parent.parent / 'native_source_identity.json').read_text()),
            candidates=[r for r in result['candidates'] if r['wins_both_rounds']],
            contract='Finite FP32 activations, current TP2 DiT out_proj/FC2 shapes, global row maximum; preserve full-width native rotation and INT32 SUM before dequantization.',
            nccl=dict(protocol='automatic', min_channels=4, max_channels=4, p2p_disable=1),
            deployment_status='Opt-in variant argument; frozen production hook unchanged. Other dtypes/native fused rotation/world-size1 retain the original path.',
            no_full_H3_pipeline=True)
        (root / 'verified_candidates.json').write_text(json.dumps(selected, indent=2))
    evidence = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.log', '.csv', '.nsys-rep') and path.name != 'final_evidence_index.json':
            evidence.append(dict(file=str(path.relative_to(root)), bytes=path.stat().st_size,
                sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()))
    (root / 'final_evidence_index.json').write_text(json.dumps(evidence, indent=2))


if __name__ == '__main__': main()
