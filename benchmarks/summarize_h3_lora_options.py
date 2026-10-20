"""Summarize Native and opt-in A schemes from a single physical dual-T4 run."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    data=json.loads((args.directory/'results.json').read_text(encoding='utf8'))
    if data['status']!='complete' or not data['native_exact_passed'] or not data['approximate_sanity_passed']:
        raise RuntimeError('Incomplete output gates; performance not accepted')
    groups=defaultdict(list)
    for row in data['projection_cases']:groups[row['adapter_mode'],row['rows'],row['config']].append(row)
    measurements={}
    for key,rows in groups.items():
        samples=[max(row['timing']['per_rank'][rank]['wall_ms'][i] for rank in (0,1))
                 for row in rows for i in range(row['timing']['per_rank'][0]['repeats'])]
        measurements[key]=dict(median_ms=statistics.median(samples),minimum_ms=min(samples),maximum_ms=max(samples),
            samples=len(samples),temporary_peak_mib=max(r['temporary_peak_bytes'] for row in rows for r in row['timing']['per_rank'])/2**20,
            adapter_mib=rows[0]['adapter_bytes']/2**20,
            per_round_median_ms=[row['timing']['critical_wall_median_ms'] for row in rows])
    comparison=[]
    for (stack,count,config),values in sorted(measurements.items()):
        base=measurements[stack,count,'native']
        comparison.append(dict(stack=stack,rows=count,config=config,**values,
            latency_reduction_percent=100*(1-values['median_ms']/base['median_ms'])))
    quality={}
    for config in ('native','row_partial','concat_a','both'):
        cases=[row for row in data['block_cases']+data['projection_cases'] if row['config']==config]
        quality[config]=dict(cases=len(cases),native_byte_exact=sum(r['native_bytes_exact'] for r in cases),
            all_ranks_agree=all(r['replicas_equal'] for r in cases),max_rel_l2=max(r['rel_l2'] for r in cases),
            max_abs=max(r['max_abs'] for r in cases),minimum_cosine=min(r['cosine'] for r in cases))
    root=Path(__file__).resolve().parents[1]
    matches={name:hashlib.sha256((root/name).read_text(encoding='utf8').encode()).hexdigest()==sha for name,sha in data['source_sha256'].items()}
    summary=dict(status=data['status'],physical_dual_t4=True,quality=quality,comparisons=comparison,
                 source_matches=matches,profiles=data['profiles'],scope=data['scope'])
    (args.directory.parent/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    for row in comparison:
        if row['rows']==8192:print(row['stack'],row['config'],f"{row['median_ms']:.3f}ms",f"{row['latency_reduction_percent']:.2f}% lower latency",f"adapter {row['adapter_mib']}MiB",f"temp {row['temporary_peak_mib']}MiB")
    print('SOURCE_MATCH',all(matches.values()))


if __name__=='__main__':main()
