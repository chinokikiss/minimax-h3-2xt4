"""Summarize downloaded physical T4 SwiGLU artifacts without accepting partial gates."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root', type=Path); args = ap.parse_args()
    root = args.root
    state = json.loads((root/'status.json').read_text())
    assert state['status'] == 'complete' and state.get('profile_ok'), 'Incomplete physical experiment'
    assert len(state['hardware']) == 2 and all(x['capability'] == [7, 5] and 'T4' in x['name'] for x in state['hardware'])
    output = dict(hardware=state['hardware'], job_status=state['status'], profiler_complete=state.get('profile_ok', False), rounds={})
    for name in ('screen', 'repeat', 'profiled'):
        ranks = [json.loads((root/name/f'rank{i}.json').read_text()) for i in (0, 1)]
        assert all(r['status'] == 'complete' for r in ranks)
        for r in ranks:
            assert all(c['status'] == 'complete' and all(v for k, v in c.items() if k.endswith('bitwise_equal')) for c in r['cases'])
        if name != 'profiled': output['rounds'][name] = json.loads((root/name/'summary.json').read_text())
        # Replicated post-reduction output hashes must agree between physical ranks.
        hashes = [{(c['component'], c['variant']):c['output_sha256'] for c in r['cases']} for r in ranks]
        assert hashes[0] == hashes[1], 'Replicated output differs across TP ranks'
    units = [json.loads((root/'screen'/f'unit_rank{i}.json').read_text()) for i in (0, 1)]
    assert all(len(rows) == 41 and all(r['bitwise_equal'] and r.get('input_unchanged', True) for r in rows) for rows in units)
    output['bitwise_edge_gates'] = sum(map(len, units))
    output['recommendations'] = {}
    for component in ('te', 'dit'):
        changes = [output['rounds'][r][component]['API']['fused_reduction_vs_packed_pct'] for r in ('screen', 'repeat')]
        output['recommendations'][component] = dict(full_API_reductions_pct=changes,
            decision='candidate: both rounds improve; inspect wall samples and profiler before integration' if min(changes) > 0 else 'not a consistent full-API improvement; keep optional')
    (root/'validated_summary.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()
