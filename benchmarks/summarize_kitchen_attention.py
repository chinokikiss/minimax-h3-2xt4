"""CPU paired analysis of the bounded Kitchen-inspired Group32 study."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def accuracy(rows):
    values=[v*100 for r in rows for v in r['dense_sample_error']['per_head_relative_RMSE']]
    return dict(median_head_partition_error_percent=statistics.median(values),
                worst_head_partition_error_percent=max(values),
                scope='32 sampled queries per rank/head, all keys, FP32 dense reference. Video ranks partition queries and share heads.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path)
    ap.add_argument('--kernel',default='chinokiki9/h3-tp2-kitchen-group32');ap.add_argument('--version',type=int,default=1)
    a=ap.parse_args();root=a.root
    state=json.loads((root/'status.json').read_text());source=json.loads((root/'source_hashes.json').read_text())
    assert source==json.loads((root.parent.parent/'submitted_sources.json').read_text())
    result=dict(kernel=a.kernel,version=a.version,status=state['status'],hardware=state.get('hardware'),cases=[],failures=[],candidates=[],native_comparisons=[],
                scope='Current TP2 dense attention APIs only, pre-exchanged KV. No H3 generation, sparse quality study or pipeline speedup claim.')
    for round_name in ('screen','repeat'):
        folder=root/round_name
        if not all((folder/f'rank{i}.json').exists() for i in (0,1)):continue
        ranks=[json.loads((folder/f'rank{i}.json').read_text()) for i in (0,1)]
        maps=[{(r['component'],r['name']):r for r in x['cases']} for x in ranks]
        for key in sorted(set(maps[0])&set(maps[1])):
            rows=[x[key] for x in maps]
            if any(r['status']!='complete' for r in rows):
                result['failures'].append(dict(round=round_name,component=key[0],name=key[1],errors=[r.get('errors') for r in rows]));continue
            native=rows[0]['config'].get('native',False)
            result['cases'].append(dict(round=round_name,component=key[0],name=key[1],config=rows[0]['config'],native=native,
                API_ms=max(r['API']['gpu_median_ms'] for r in rows),wall_ms=max(r['API']['wall_median_ms'] for r in rows),
                rank_medians_ms=[r['API']['gpu_median_ms'] for r in rows],
                bitwise_equal_to_current=None if native else all(r['bitwise_equal_to_current'] for r in rows),
                operands_identical=None if native else all(all(r['operand_identity'].values()) for r in rows),
                accuracy=accuracy(rows)))
    for row in result['cases']:
        baseline=next(r for r in result['cases'] if r['round']==row['round'] and r['component']==row['component'] and r['name']=='current')
        row['latency_reduction_percent']=100*(1-row['API_ms']/baseline['API_ms'])
    for row in result['cases']:
        if row['round']!='repeat' or row['native'] or row['name']=='current':continue
        screen=next(r for r in result['cases'] if r['round']=='screen' and r['component']==row['component'] and r['name']==row['name'])
        result['candidates'].append(dict(component=row['component'],name=row['name'],config=row['config'],
            wins_both_rounds=row['latency_reduction_percent']>0 and screen['latency_reduction_percent']>0,
            repeat_reduction_percent=row['latency_reduction_percent'],screen_reduction_percent=screen['latency_reduction_percent'],
            bitwise_equal_both_rounds=row['bitwise_equal_to_current'] and screen['bitwise_equal_to_current']))
    for round_name in ('screen','repeat'):
        rows=[r for r in result['cases'] if r['round']==round_name and r['component']=='video'];native=next((r for r in rows if r['native']),None)
        if native:
            group=min((r for r in rows if not r['native']),key=lambda x:x['API_ms'])
            result['native_comparisons'].append(dict(round=round_name,group32_name=group['name'],group32_ms=group['API_ms'],kitchen_ms=native['API_ms'],
                group32_overhead_percent=100*(group['API_ms']/native['API_ms']-1),group32_sample_accuracy=group['accuracy'],kitchen_sample_accuracy=native['accuracy']))
    units=[]
    for path in sorted((root/'screen').glob('unit_rank*.json')):
        tests=json.loads(path.read_text());units.append(dict(file=path.name,tests=len(tests),all_operands_identical=all(all(r['identity'].values()) for r in tests)))
    result['unit_gates']=units;result['recorded_stage_seconds']=sum(r.get('seconds',0) for r in state['commands'])
    result['profiling_commands']=[r for r in state['commands'] if 'nsys' in r['name']]
    (root/'paired_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    qualified=bool(units) and all(r['all_operands_identical'] for r in units)
    if qualified:
        (root/'verified_candidates.json').write_text(json.dumps(dict(kernel=a.kernel,version=a.version,
            status='operator_conformance_candidates',source_sha256=source,hardware=result['hardware'],
            candidates=[r for r in result['candidates'] if r['wins_both_rounds']],
            precision='Unchanged Group32 operands, FP32 scales/state and FP16 P/V. Full-output conformance rtol=1e-3, atol=2e-4; bitwise status recorded separately.',
            quality_limit='Sampled dense queries only; no whole-generation or sparse quality conclusion.',
            deployment_status='Production hooks unchanged; exact-shape operator candidates.'),indent=2))
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.csv','.nsys-rep') and p.name!='final_evidence_index.json']
    (root/'final_evidence_index.json').write_text(json.dumps([dict(file=str(p.relative_to(root)),bytes=p.stat().st_size,
        sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in sorted(files)],indent=2))


if __name__=='__main__':main()
