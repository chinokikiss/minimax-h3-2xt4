"""Version-bound CPU analysis of second-pass operator evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();root=a.root
    state=json.loads((root/'status.json').read_text());source=json.loads((root/'source_hashes.json').read_text())
    assert source==json.loads((root.parent.parent/'submitted_sources.json').read_text())
    assert len(state.get('hardware',[]))==2 and all(x['capability']==[7,5] and 'T4' in x['name'] for x in state['hardware'])
    result=dict(kernel='chinokiki9/h3-tp2-kitchen-group32-v2',version=1,status=state['status'],hardware=state['hardware'],cases=[],failures=[],candidates=[],unit_gates=[],
        scope='Real captured TP2 dense operators; compared to previous T4 winners. No pipeline, sparse policy or generation-quality result.')
    for round_name in ('screen','repeat'):
        ranks=[json.loads((root/round_name/f'rank{i}.json').read_text()) for i in (0,1)]
        if round_name=='screen':
            for r in ranks:
                units=r.get('unit_gates',[]);assert len(units)==7
                assert all(all(x.get('operand_identity',{}).values()) and x.get('bitwise_equal',True) for x in units)
                result['unit_gates'].append(dict(rank=r['rank'],count=len(units),passed=True))
        maps=[{(r['component'],r['name']):r for r in x['cases']} for x in ranks]
        for key in sorted(set(maps[0])&set(maps[1])):
            rows=[m[key] for m in maps]
            if any(x['status']!='complete' for x in rows):
                result['failures'].append(dict(round=round_name,component=key[0],name=key[1],errors=[x.get('errors') for x in rows]));continue
            native=rows[0]['config'].get('native',False)
            record=dict(round=round_name,component=key[0],name=key[1],config=rows[0]['config'],native=native,
                API_ms=max(x['API']['gpu_median_ms'] for x in rows),wall_ms=max(x['API']['wall_median_ms'] for x in rows),
                rank_API_ms=[x['API']['gpu_median_ms'] for x in rows],
                bitwise_equal_to_current=None if native else all(x['bitwise_equal_to_current'] for x in rows),
                operands_identical=None if native else all(all(x['operand_identity'].values()) for x in rows))
            for phase in ('preparation','attention_and_output'):
                if all(phase in x for x in rows):record[phase+'_rank_medians_ms']=[x[phase]['gpu_median_ms'] for x in rows]
            result['cases'].append(record)
    for row in result['cases']:
        baseline=next(x for x in result['cases'] if x['round']==row['round'] and x['component']==row['component'] and x['name']=='current')
        row['latency_reduction_percent']=100*(1-row['API_ms']/baseline['API_ms'])
    for row in result['cases']:
        if row['round']!='repeat' or row['native'] or row['name']=='current':continue
        screen=next(x for x in result['cases'] if x['round']=='screen' and x['component']==row['component'] and x['name']==row['name'])
        result['candidates'].append(dict(component=row['component'],name=row['name'],config=row['config'],
            screen_reduction_percent=screen['latency_reduction_percent'],repeat_reduction_percent=row['latency_reduction_percent'],
            wins_both_rounds=screen['latency_reduction_percent']>0 and row['latency_reduction_percent']>0,
            bitwise_equal_both_rounds=screen['bitwise_equal_to_current'] and row['bitwise_equal_to_current']))
    (root/'paired_summary.json').write_text(json.dumps(result,indent=2))
    (root/'verified_candidates.json').write_text(json.dumps(dict(source_sha256=source,hardware=state['hardware'],candidates=[x for x in result['candidates'] if x['wins_both_rounds']],
        status='operator candidates only; production hooks unchanged',precision='Exact operands; full-output rtol=.001/atol=.0002, bitwise status recorded separately. No generation semantics assessment.'),indent=2))
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.log','.nsys-rep') and p.name!='final_evidence_index.json']
    index=[]
    for p in sorted(files):
        with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        index.append(dict(file=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha))
    (root/'final_evidence_index.json').write_text(json.dumps(index,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
