"""Retain both ranks, repeats, failures, memory, diagnostics and a per-site Pareto set."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1];EVIDENCE=Path('D:/attention_eval/tp2_p2_20261018')


def dispersion(values):
    return dict(median=float(np.median(values)),p10=float(np.percentile(values,10)),p90=float(np.percentile(values,90)),
        IQR=float(np.percentile(values,75)-np.percentile(values,25)),minimum=min(values),maximum=max(values),count=len(values))


def main():
    result=dict(scope='Physical two-T4 operator evidence only; approximate transport and attention generation quality unqualified.',
        evidence_root=str(EVIDENCE),jobs=[],comparisons=[],failures=[],pareto=[],generation=dict(status='unqualified',
            local_weights_available=True,local_GPU='One RTX 5060 Ti; cannot execute the required physical TP2/NCCL quality comparison',
            target_full_pipeline_run=False,reason='Full TP2 quality runtime and checkpoints were not staged in these bounded operator jobs. Qualification remains unperformed; prior PP attention clips do not qualify these TP2 transports.'))
    for kind in ('transport','attention','restoration','transport-confirm','scalar-profile'):
        for version in ((1,2) if kind=='transport' else (1,2,3) if kind=='attention' else (1,)):
            root=EVIDENCE/kind/('v'+str(version));p=root/'p2';entry=dict(kind=kind,version=version,evidence=str(root),raw={})
            if version>2 and not root.exists():continue
            if (p/'status.json').exists():entry['status']=json.loads((p/'status.json').read_text())
            if (root/'provider/identity.json').exists():entry['provider_identity']=json.loads((root/'provider/identity.json').read_text())
            if (p/'source_hashes.json').exists() and (root/'provider/source_hashes.json').exists():
                runtime_hashes=json.loads((p/'source_hashes.json').read_text());provider_hashes=json.loads((root/'provider/source_hashes.json').read_text())
                entry['source_identity_gate']=runtime_hashes==provider_hashes
                assert entry['source_identity_gate'],(kind,version,'provider/runtime source mismatch')
            for run in ('screen','repeat','profiled'):
                ranks=[]
                for rank in (0,1):
                    path=p/run/f'rank{rank}.json'
                    if path.exists():ranks.append(json.loads(path.read_text()))
                    unit=p/run/f'unit_rank{rank}.json'
                    if unit.exists():entry.setdefault('independent_units',{})[f'{run}_rank{rank}']=json.loads(unit.read_text())
                entry['raw'][run]=ranks
                for record in ranks:
                    for case in record['cases']:
                        if case.get('status')=='failed':result['failures'].append(dict(kind=kind,version=version,run=run,rank=record['rank'],case=case))
            if 'status' in entry and entry['status']['status']!='complete':result['failures'].append(dict(kind=kind,version=version,job_status=entry['status']))
            result['jobs'].append(entry)
            if version<2 and kind in ('transport','attention'):continue
            for run in ('screen','repeat'):
                ranks=entry['raw'][run]
                if len(ranks)!=2:continue
                families=sorted({c['family'] for r in ranks for c in r['cases']})
                for family in families:
                    maps=[{c['name']:c for c in r['cases'] if c['family']==family and c.get('status')=='complete' and c.get('API')} for r in ranks]
                    is_transport=kind.startswith('transport')
                    baseline='int32' if is_transport else 'current'
                    common=set(maps[0])&set(maps[1])
                    if baseline not in common:continue
                    base=max(m[baseline]['API']['gpu_median_ms'] for m in maps)
                    for name in sorted(common):
                        both=[m[name] for m in maps]
                        for c in both:
                            assert c['API']['repeats']==(30 if kind=='transport-confirm' else 15) and c['API']['warmup_actual_calls']==10
                        latency=max(c['API']['gpu_median_ms'] for c in both)
                        row=dict(kind=kind,version=version,run=run,family=family,name=name,slower_rank_API_median_ms=latency,
                            slower_rank_wall_median_ms=max(c['API']['wall_median_ms'] for c in both),API_reduction_vs_baseline_pct=100*(1-latency/base),
                            both_ranks=[dict(rank=r['rank'],GPU=dispersion(c['API']['gpu_ms']),wall=dispersion(c['API']['wall_ms'])) for r,c in zip(ranks,both)])
                        if is_transport:
                            row.update(max_relative_RMSE=max(c.get('diagnostics',c.get('arithmetic_diagnostics'))['relative_RMSE'] for c in both),
                                replicas_identical=both[0]['output_sha256']==both[1]['output_sha256'])
                            if all('memory' in c for c in both):
                                row.update(max_peak_allocated_bytes=max(c['memory']['peak_allocated_bytes'] for c in both),
                                    max_peak_reserved_bytes=max(c['memory']['peak_reserved_bytes'] for c in both))
                            if kind=='transport-confirm':
                                row['paired_native_minus_candidate_ms']=[dict(rank=r['rank'],**dispersion(c['paired_native_minus_candidate_ms'])) for r,c in zip(ranks,both)]
                                if name.endswith('_fused'):
                                    row['paired_unfused_minus_fused_ms']=[dict(rank=r['rank'],**dispersion([a-b for a,b in zip(m[name.removesuffix('_fused')]['API']['gpu_ms'],c['API']['gpu_ms'])])) for r,c,m in zip(ranks,both,maps)]
                            assert row['replicas_identical']
                        else:
                            row.update(slower_rank_prepared_main_ms=max(c['prepared_main']['gpu_median_ms'] for c in both),diagnostic_only=both[0]['diagnostic_only'])
                        result['comparisons'].append(row)
            screen=entry['raw']['screen'];repeat=entry['raw']['repeat']
            if len(screen)==len(repeat)==2:
                checks=[]
                for a,b in zip(screen,repeat):
                    original={(c['family'],c['name']):c for c in a['cases'] if c.get('status')=='complete'}
                    for c in b['cases']:
                        key=(c['family'],c['name'])
                        if key in original and c.get('status')=='complete':
                            passed=c['output_sha256']==original[key]['output_sha256'];checks.append(dict(rank=a['rank'],family=key[0],name=key[1],identical=passed));assert passed
                entry['fresh_process_identity']=checks
    for family in sorted({c['family'] for c in result['comparisons'] if c['kind']=='transport'}):
        rows=[c for c in result['comparisons'] if c['kind']=='transport' and c['family']==family and c['run']=='repeat']
        for c in rows:
            def dominates(a,b):
                ak=(a['slower_rank_API_median_ms'],a['max_relative_RMSE'],a['max_peak_allocated_bytes']);bk=(b['slower_rank_API_median_ms'],b['max_relative_RMSE'],b['max_peak_allocated_bytes'])
                return all(x<=y for x,y in zip(ak,bk)) and any(x<y for x,y in zip(ak,bk))
            result['pareto'].append(dict(family=family,name=c['name'],on_operator_Pareto_front=not any(dominates(other,c) for other in rows),
                latency_ms=c['slower_rank_API_median_ms'],relative_RMSE=c['max_relative_RMSE'],peak_allocated_bytes=c['max_peak_allocated_bytes'],
                quality_status='Not quality-qualified; Pareto arithmetic is not generation acceptance.'))
    for key,path in [('input_distribution',EVIDENCE/'input_distribution.json'),('offline_resources',EVIDENCE/'offline/compilation.json'),('offline_instructions',EVIDENCE/'offline/instruction_inventory.json'),
                     ('local_epilogue_preview',EVIDENCE/'local_epilogue_units.json'),('local_attention_preview',EVIDENCE/'local_attention_units.json'),
                     ('counter_probe',EVIDENCE/'counter-probe/v1/ncu_probe/status.json'),('target_ptx_inventory',EVIDENCE/'target_ptx_inventory.json'),
                     ('FC2_profiled_jitter',EVIDENCE/'transport-confirm/v1/p2/jitter_attribution.json')]:
        if path.exists():result[key]=json.loads(path.read_text())
    result['traces']=[str(p) for p in EVIDENCE.rglob('*.nsys-rep')]
    result['trace_attribution']={str(p.relative_to(EVIDENCE)):json.loads(p.read_text()) for p in EVIDENCE.rglob('trace_attribution.json')}
    result['pareto_interpretation']='Primary sequential reverse-repeat point medians only, without uncertainty dominance. A tiny latency difference is not robust evidence. FC2 paired confirmation is a separate scheduling experiment; no peak-memory remeasurement there.'
    result['dispatch_policy']=dict(production_defaults_unchanged=True,default_transport='native INT32',default_PV='FP16',
        scalar_pooling='Optional; unchanged routing. Omitted-only FP16-PV score split is the first integration candidate, subject to 78.3 MiB/rank workspace and broader identity gates.',
        approximations='Explicit opt-in only; full TP2 generation quality remains unqualified.',
        next_priority=['Scalar omitted-only score split with FP16 PV','Dense FP16 PV128 mapping','Per-site FP16 transport plus exact fused O epilogue','INT8 PV128 and INT8 transport only after generation qualification'])
    result['decisions']=[
        dict(status='Promote',candidate='No new production default',reason='Preserve native controls; new full TP2 quality gates are incomplete.'),
        dict(status='Experimental',priority=1,candidate='Scalar omitted-only scores, FP16 PV',reason='Repeated 9.74/11.25% API gain, same-fixture bitwise identity; 78.3 MiB workspace, broader integration/shape gates remain.'),
        dict(status='Experimental',priority=2,candidate='Dense FP16 PV128',reason='Repeated 2.97/3.98% gain; changed floating mapping rounding.'),
        dict(status='Experimental',priority=3,candidate='FP16 transport with fused O epilogue',reason='Exact codec contract, repeated gain; cancellation and full TP2 quality remain unqualified.'),
        dict(status='Experimental',candidate='Packed INT8 transport and dense INT8 PV128',reason='Repeated transport/PV opportunity but additional approximation; variable incremental FC2 fusion benefit.'),
        dict(status='Reject',candidate='Dense INT8 PV256 and running-max variant',reason='Repeated complete-API regression.'),
        dict(status='Reject as preferred option',candidate='Full scalar scores and omitted scalar PV8',reason='Full scores add unnecessary preparation; omitted PV8 gains only 0.1-0.25ms beyond omitted FP16.'),
        dict(status='Reject as deployable output',candidate='Unit-restoration diagnostic',reason='Changes the output contract; first variant fails finite gate.'),
        dict(status='Blocked',candidate='Measured dynamic hardware-counter root cause',reason='Current physical-T4 probe ERR_NVGPUCTRPERM.'),
        dict(status='Blocked for promotion',candidate='Full TP2 quality/end-to-end and all-layer/step generalization',reason='Not staged/run within the bounded operator scope; no current pipeline qualification.')]
    verification=EVIDENCE/'final_verification.json'
    if verification.exists():
        v=json.loads(verification.read_text())
        result['evidence_verification']={k:value for k,value in v.items() if k!='artifact_hashes'}
        result['evidence_verification']['hash_manifest']=str(verification)
    result['provider_status']=json.loads((EVIDENCE/'provider_status.json').read_text())
    result['summary']=dict(comparisons=len(result['comparisons']),retained_failure_records=len(result['failures']),systems_reports=len(result['traces']),
        source_identity_gates=sum(j.get('source_identity_gate',False) for j in result['jobs']),
        fresh_process_output_checks=sum(len(j.get('fresh_process_identity',[])) for j in result['jobs']),
        measured_event_samples=sum(c['API']['repeats'] for j in result['jobs'] for run in ('screen','repeat') for r in j['raw'][run] for c in r['cases'] if c.get('API')))
    (ROOT/'TP2_P2_RESULTS.json').write_text(json.dumps(result,indent=2))
    lines=['| Job | Version | Site | Variant | Round | Slower rank API ms | Reduction % | Peak allocated MiB | Relative RMSE % |', '|---|---|---|---|---|---:|---:|---:|---:|']
    for r in result['comparisons']:
        mem=f"{r['max_peak_allocated_bytes']/2**20:.1f}" if 'max_peak_allocated_bytes' in r else '—'
        err=f"{r['max_relative_RMSE']*100:.5f}" if 'max_relative_RMSE' in r else '—'
        lines.append(f"| {r['kind']} | {r['version']} | {r['family']} | {r['name']} | {r['run']} | {r['slower_rank_API_median_ms']:.4f} | {r['API_reduction_vs_baseline_pct']:.2f} | {mem} | {err} |")
    (EVIDENCE/'results_table.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('P2_RESULTS',len(result['comparisons']),'comparisons,',len(result['failures']),'retained failure records')


if __name__=='__main__':main()
