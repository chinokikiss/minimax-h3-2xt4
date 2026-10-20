"""Summarize physical FC2 allocation and synchronized latency ablations."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    data=json.loads((args.directory/'results.json').read_text(encoding='utf8'))
    if not data.get('passed') or not data.get('physical_dual_t4') or not data.get('lifetime_exact'):
        raise RuntimeError('Physical completion and lifetime byte-exact gates must pass')
    numerical={}
    for name in ('before','lifetime','owned'):
        metrics=[m for case in data['cases'] if case['variant']==name for m in case['per_rank']]
        numerical[name]=dict(rank_cases=len(metrics),payload_changed=sum(m['changed_payload'] for m in metrics),
            scale_rows_changed=sum(m['changed_scale_rows'] for m in metrics),
            max_output_abs=max(m['output']['max_abs'] for m in metrics),
            max_output_rel_l2=max(m['output']['rel_l2'] for m in metrics),
            min_output_cosine=min(m['output']['cosine'] for m in metrics),
            all_bytes_exact=all(m[k]['bytes_exact'] for m in metrics for k in ('q','scale','output')))
    groups=defaultdict(list)
    for item in data['timings']:groups[item['stack'],item['rows'],item['scope'],item['variant']].append(item)
    comparisons=[]
    for (stack,rows,scope,variant),records in sorted(groups.items()):
        clocks={}
        for clock in ('wall_ms','gpu_ms'):
            samples=[max(record['per_rank'][r]['timing'][clock][i] for r in (0,1))
                     for record in records for i in range(len(record['per_rank'][0]['timing'][clock]))]
            clocks[clock]=dict(median=statistics.median(samples),minimum=min(samples),maximum=max(samples),count=len(samples),
                round_medians=[statistics.median(max(record['per_rank'][r]['timing'][clock][i] for r in (0,1))
                                                for i in range(len(record['per_rank'][0]['timing'][clock]))) for record in records])
        memory=[{k+'_mib':max(record['per_rank'][rank]['memory'][k] for record in records)/2**20
                 for k in ('entry','temporary','peak_allocated','peak_reserved')} for rank in (0,1)]
        comparisons.append(dict(stack=stack,rows=rows,scope=scope,variant=variant,timing=clocks,memory=memory))
    for row in comparisons:
        base=next(r for r in comparisons if (r['stack'],r['rows'],r['scope'],r['variant'])==(row['stack'],row['rows'],row['scope'],'before'))
        row['wall_reduction_percent']=100*(1-row['timing']['wall_ms']['median']/base['timing']['wall_ms']['median'])
    root=Path(__file__).resolve().parents[1]
    source_match={name:hashlib.sha256((root/name).read_text(encoding='utf8').encode()).hexdigest()==sha
                  for name,sha in data['source_sha256'].items()}
    result=dict(numerical=numerical,comparisons=comparisons,source_matches=source_match,
                scope=data['scope'],owned_scope=data['owned_scope'])
    (args.directory.parent/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('NUMERICAL',json.dumps(numerical),'SOURCE_MATCH',all(source_match.values()))
    for row in comparisons:
        if row['rows']==8192:
            print(row['stack'],row['scope'],row['variant'],round(row['timing']['wall_ms']['median'],4),'ms',
                  'temporary_MiB',[m['temporary_mib'] for m in row['memory']])


if __name__=='__main__':main()
