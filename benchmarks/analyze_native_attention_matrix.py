"""Offline analysis of the bounded native-Kitchen/TileLang T4 matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(batch):
    ranks=batch.get('ranks',[])
    result=dict(status=batch.get('status'),native_status=batch.get('native_status'),
        input_kind=batch.get('input_kind'),shapes={},errors=[],
        scope='Slower-rank complete-API operator timing; no communication or generation speedup')
    for rank in ranks:
        result['errors'].extend(dict(rank=rank.get('rank'),**e) for e in rank.get('errors',[]))
    names=sorted({case['shape']['name'] for rank in ranks for case in rank.get('cases',[])})
    for shape in names:
        matching={rank['rank']:next(c for c in rank.get('cases',[]) if c['shape']['name']==shape)
            for rank in ranks if any(c['shape']['name']==shape for c in rank.get('cases',[]))}
        row=dict(partitions={str(r):c['shape'] for r,c in matching.items()},backends={})
        for key in ('retained_block_fraction','retained_token_pair_fraction','tilelang_scalar_dispatch'):
            row[key]={str(r):c[key] for r,c in matching.items() if key in c}
        backends=sorted({c['name'] for m in matching.values() for c in m.get('candidates',[])})
        for backend in backends:
            timings={}
            complete=len(matching)==2
            for rank,case in matching.items():
                samples=[c for c in case.get('candidates',[]) if c['name']==backend and c['status']=='passed']
                valid={c['round']:c for c in samples}
                if set(valid)!={0,1}:complete=False
                timings[rank]=valid
            record=dict(complete_two_rank_two_round=complete,
                by_rank={str(r):{str(k):dict(event_median_ms=v['event_median_ms'],wall_median_ms=v['wall_median_ms'],
                    peak_allocated=v['peak_allocated'],peak_temporary=v['peak_temporary']) for k,v in rounds.items()}
                    for r,rounds in timings.items()})
            if complete:
                record.update(slower_rank_event_ms=max(statistics.median(v[k]['event_median_ms'] for k in (0,1)) for v in timings.values()),
                    slower_rank_wall_ms=max(statistics.median(v[k]['wall_median_ms'] for k in (0,1)) for v in timings.values()),
                    peak_allocated=max(v[k]['peak_allocated'] for v in timings.values() for k in (0,1)),
                    peak_temporary=max(v[k]['peak_temporary'] for v in timings.values() for k in (0,1)))
            row['backends'][backend]=record
        for backend,record in row['backends'].items():
            if not record['complete_two_rank_two_round']:continue
            record['paired_ratios']={}
            for target in ('kitchen_int8','native_dense_baseline','native_baseline','te_kitchen_default',
                           'tilelang_group32_dense','tilelang_drop','tilelang_scalar'):
                other=row['backends'].get(target)
                if not other or not other['complete_two_rank_two_round']:continue
                ratios=[other['by_rank'][str(r)][str(k)]['event_median_ms']/
                    record['by_rank'][str(r)][str(k)]['event_median_ms'] for r in (0,1) for k in (0,1)]
                record['paired_ratios'][target]=dict(min=min(ratios),max=max(ratios),median=statistics.median(ratios))
        result['shapes'][shape]=row
    return result


def markdown(result):
    lines=['# Native Kitchen / TileLang T4 attention matrix','',
        'Synthetic operator inputs; complete APIs include preparation, routing and output conversion. ',
        'Values below are the slower rank of each rank’s median across the two measurement rounds. ',
        'They exclude JIT/warmup and include no TP communication or full H3 pipeline.','',
        '| Shape | Backend | GPU ms | Host ms | Peak allocated GiB |',
        '|---|---|---:|---:|---:|']
    for shape,row in result['shapes'].items():
        for name,case in row['backends'].items():
            if not case['complete_two_rank_two_round']:
                lines.append(f'| {shape} | {name} | incomplete/rejected | — | — |')
            else:
                lines.append(f"| {shape} | {name} | {case['slower_rank_event_ms']:.3f} | {case['slower_rank_wall_ms']:.3f} | {case['peak_allocated']/2**30:.3f} |")
    lines+=['','Native qualification: '+str(result.get('native_status'))+'.',
        'These backends have different quantization/approximation contracts; timings do not rank generation quality.']
    if result['errors']:
        lines+=['','Rejected cases:','']+[f"- Rank {e['rank']}, {e['shape']}, {e['name']}: {e['error']}" for e in result['errors']]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raw=args.input.read_bytes();result=summarize(json.loads(raw))
    result['input_sha256']=hashlib.sha256(raw).hexdigest();args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'summary.md').write_text(markdown(result))
