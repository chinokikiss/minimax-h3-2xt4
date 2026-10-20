"""Finalize version-bound operator evidence without any GPU activity."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root', type=Path); args = ap.parse_args(); root = args.root
    summary = json.loads((root / 'paired_summary.json').read_text())
    state = json.loads((root / 'status.json').read_text())
    source = json.loads((root / 'source_hashes.json').read_text())
    assert source == json.loads((root.parent.parent / 'submitted_sources.json').read_text())
    assert state['attention_screen_ok'] and state['attention_repeat_ok']
    assert not summary['failures']
    assert all(r['bitwise_output_equal'] and r['operands_identical'] for r in summary['attention'])
    assert all(r['output_identical_across_settings'] for r in summary['row_linear'])
    candidates = {}
    for winner in summary['confirmed_candidates']:
        assert winner['wins_both_rounds']
        family = winner['family']
        rows = json.loads((root / 'attention_screen/rank0.json').read_text())['cases']
        config = next(r['config'] for r in rows if r['family'] == family and r['name'] == winner['name'])
        candidates[family] = dict(config=config, **{k: v for k, v in winner.items() if k != 'family'})
    selected = dict(status='operator_verified_candidates', kernel='chinokiki9/h3-tp2-operator-optimize', version=1,
        hardware=state['hardware'], source_sha256=source, candidates=candidates,
        deployment_status='No production hook or global NCCL setting changed. Exact-shape operator evidence only.',
        sparse_contract='Existing drop/scalar policy. Fast tau0 routing requires finite centroids; retain original routing for exceptional inputs.',
        nccl=dict(protocol='default', channels=4, p2p_disable=1,
            qualification='Raw collective and complete current row APIs checked in one session; no independent row-setting repeat or whole-pipeline confirmation.'),
        profiling=dict(usable_attention_reports=len(list(root.glob('attention_profile.*.nsys-rep'))),
            usable_row_reports=len(list(root.glob('row_profile.*.nsys-rep'))),
            partial='Attention profiler timed out during report export; selected Video report unavailable.',
            compute_counters='Unavailable: prior ERR_NVGPUCTRPERM. No stall/achieved-occupancy claims.'),
        recorded_stage_seconds=sum(c['seconds'] for c in state['commands']),
        accounting='Recorded subprocess durations, not Kaggle quota billing.',
        no_full_H3_pipeline=True)
    (root / 'verified_candidates.json').write_text(json.dumps(selected, indent=2))
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.nsys-rep', '.log', '.csv') and p.name != 'final_evidence_index.json']
    index = [dict(file=str(p.relative_to(root)), bytes=p.stat().st_size,
                  sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()) for p in sorted(files)]
    (root / 'final_evidence_index.json').write_text(json.dumps(index, indent=2))
    print('Verified', len(candidates), 'attention families;', len(index), 'indexed evidence files; no GPU execution.')


if __name__ == '__main__': main()
