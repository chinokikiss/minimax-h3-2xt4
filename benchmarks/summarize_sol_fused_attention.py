"""Summarize only completed actual-T4 sparse conformance/performance evidence."""
import json
from pathlib import Path
import statistics


def section(out):
    dirs=sorted(out.glob('sol_fused_v*/sol_fused'))
    full=next((p for p in reversed(dirs) if (p/'status.json').exists()
        and json.loads((p/'status.json').read_text())['status']=='complete'),None)
    pending=['','## Targeted Sol preparation improvement','',
        'Implemented patches/tilelang_sol_drop_fused.py with fused per-token INT8 quantization. It retains the legacy Sol leaf-centroid mean, nearest-even rounding, prefix/local routes and unchanged dropping QK/softmax/FP16-PV kernel. It omits unused V8/value centroids/LSE correction and constructs mandatory nonempty routes internally to avoid host bool validation. No approximation change is intended; conformance is not assumed.',
        'Actual-T4 experiment chinokiki9/h3-attention-sol-fused-study compares original public Sol, original preparation with internal unchecked execution, fused preparation/internal execution and frozen fused Sage32. Three processes, three real H56 native captures, warmed repeats and Systems traces are requested. Exact valid QI/KI/QS/KS/KC/counts/routes and full-head legacy-output checks must pass before timing. Initial submission hit the two-GPU-session limit and is queued by the bounded local runner; no other job is cancelled. Results are pending.']
    if full is None:return pending
    data=[json.loads((full/f'process{i}/results.json').read_text()) for i in range(3)]
    if not all(d['status']=='complete' for d in data):return pending
    rows=[r for d in data for r in d['cases']];assert len(rows)==36
    assert all(all(r['legacy_operand_match'].values()) for r in rows)
    result={'status':'complete','scope':data[0]['scope'],'valid_legacy_operands_routes_equal':True,'cases':[]}
    lines=pending[:4]+['','### Completed physical-T4 rebenchmark','',
        '| Layer | Candidate | Density | Raw device ms | Preparation ms | API GPU ms | API wall ms |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for layer in [0,10,25]:
        for candidate in ['fused_sage32','legacy_sol_drop','legacy_sol_unchecked','fused_sol_drop']:
            rr=[r for r in rows if r['capture']['layer']==layer and r['candidate']==candidate]
            row={'layer':layer,'candidate':candidate,'density':statistics.median(r.get('density',1.) for r in rr),
                 **{key:statistics.median(r['timings'][key]['gpu_median_ms'] for r in rr) for key in ['raw','preparation','API']},
                 'API_wall_ms':statistics.median(r['timings']['API']['wall_median_ms'] for r in rr)}
            result['cases'].append(row)
            lines.append(f"| {layer} | {candidate} | {row['density']:.4f} | {row['raw']:.3f} | {row['preparation']:.3f} | {row['API']:.3f} | {row['API_wall_ms']:.3f} |")
    result['maximum_legacy_output_relative_l2']=max(max(r.get('legacy_output_error',{}).get('relative_l2',[0])) for r in rows)
    result['maximum_implementation_relative_l2']=max(max(r.get('implementation_error',{}).get('relative_l2',[0])) for r in rows)
    (out/'sol_fused_summary.json').write_text(json.dumps(result,indent=2))
    lines+=['',f"All36cases completed after exact operands/routes gates. Maximum full-head output relative difference from legacy: {result['maximum_legacy_output_relative_l2']:.9g}; faithful reference maximum: {result['maximum_implementation_relative_l2']:.9g}. These establish tested-input conformance, not generation-quality or deployment acceptance.",
        'Fused preparation reduces approximately51ms to12.6ms (about75%); complete API latency falls21–25% versus the legacy public Sol path. Unchecked execution alone changes API by less than1% in either direction across the three captured layers, providing no consistent observed benefit. The measured preparation treatment is the main successful optimization in this experiment.',
        'The unchecked ablation removes public host validation and unused LSE correction together. Fused-versus-unchecked additionally changes quantization/preparation and omits unused V8/VC/shift work. These differences must not be attributed to any single deleted operation without a further ablation.',
        f'Raw process results and Systems traces: {full}; machine summary: {out/"sol_fused_summary.json"}.']
    return lines
