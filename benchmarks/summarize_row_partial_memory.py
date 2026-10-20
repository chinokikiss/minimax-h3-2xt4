"""Summarize physical Row Partial memory ablations without changing references."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    d=json.loads((args.directory/'results.json').read_text(encoding='utf8'))
    if d['status']!='complete' or not d['native_exact_passed'] or not d['previous_row_exact_passed']:
        raise RuntimeError('Native/previous-row byte gates incomplete; reject performance')
    groups=defaultdict(list)
    for row in d['projection_cases']:groups[row['scope'],row['adapter_mode'],row['rows'],row['config']].append(row)
    comparison=[]
    for (scope,stack,count,config),rows in sorted(groups.items()):
        samples=[max(row['timing']['per_rank'][r]['wall_ms'][i] for r in (0,1)) for row in rows for i in range(row['timing']['per_rank'][0]['repeats'])]
        comparison.append(dict(scope=scope,stack=stack,rows=count,config=config,
            median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples),samples=len(samples),
            per_round_median_ms=[r['timing']['critical_wall_median_ms'] for r in rows],
            temporary_peak_mib=max(r['temporary_peak_bytes'] for row in rows for r in row['timing']['per_rank'])/2**20,
            absolute_peak_mib=max(r['peak_allocated_bytes'] for row in rows for r in row['timing']['per_rank'])/2**20))
    for row in comparison:
        before=next(r for r in comparison if (r['scope'],r['stack'],r['rows'],r['config'])==(row['scope'],row['stack'],row['rows'],'before'))
        row.update(latency_reduction_percent=100*(1-row['median_ms']/before['median_ms']),memory_reduction_mib=before['temporary_peak_mib']-row['temporary_peak_mib'])
    cases=d['block_cases']+d['projection_cases'];root=Path(__file__).resolve().parents[1]
    matches={name:hashlib.sha256((root/name).read_text(encoding='utf8').encode()).hexdigest()==sha for name,sha in d['source_sha256'].items()}
    summary=dict(status=d['status'],physical_dual_t4=True,total_cases=len(cases),
        native_byte_controls=sum(r['config']=='native' and r['native_bytes_exact'] for r in cases),
        previous_row_byte_controls=sum(r['config']!='native' and r['previous_row_bytes_exact'] for r in cases),
        source_matches=matches,comparisons=comparison,profiles=d['profiles'],scope=d['scope'])
    (args.directory.parent/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    for row in comparison:
        if row['rows']==8192:print(row['scope'],row['stack'],row['config'],f"{row['median_ms']:.3f}ms",f"temp {row['temporary_peak_mib']:.2f}MiB",f"change {row['memory_reduction_mib']:.2f}MiB")
    print('BYTE_CASES',len(cases),'SOURCE_MATCH',all(matches.values()))


if __name__=='__main__':main()
