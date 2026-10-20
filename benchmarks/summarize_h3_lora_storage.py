"""Summarize same-run byte-exact LoRA storage A/B; no cross-run speedups."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    data=json.loads((args.directory/'results.json').read_text())
    cases=data['block_cases']+data['projection_cases']
    if data['status']!='complete' or not all(c['native_bytes_exact'] and c['replicas_equal'] for c in cases):
        raise RuntimeError('Physical byte qualification incomplete; no performance acceptance')
    groups=defaultdict(list)
    for row in data['projection_cases']:
        groups[row['adapter_mode'],row['rows'],row['mode'],row['variant']].append(row)
    metrics={}
    for key,rows in groups.items():
        sample=[max(row['timing']['per_rank'][r]['wall_ms'][i] for r in (0,1))
                for row in rows for i in range(row['timing']['per_rank'][0]['repeats'])]
        per_rank=[dict(temporary_peak_mib=max(row['timing']['per_rank'][r]['temporary_peak_bytes'] for row in rows)/2**20,
                       absolute_peak_mib=max(row['timing']['per_rank'][r]['peak_allocated_bytes'] for row in rows)/2**20)
                  for r in (0,1)]
        metrics[key]=dict(critical_median_ms=statistics.median(sample),critical_min_ms=min(sample),critical_max_ms=max(sample),
                          sample_count=len(sample),per_rank=per_rank,
                          temporary_peak_mib=max(r['temporary_peak_mib'] for r in per_rank))
    comparison=[]
    for stack,count,mode,variant in sorted(metrics):
        if variant!='baseline':continue
        before=metrics[stack,count,mode,'baseline'];after=metrics[stack,count,mode,'optimized']
        comparison.append(dict(stack=stack,rows=count,mode=mode,baseline=before,optimized=after,
            latency_reduction_percent=100*(1-after['critical_median_ms']/before['critical_median_ms']),
            memory_reduction_mib=before['temporary_peak_mib']-after['temporary_peak_mib']))
    root=Path(__file__).resolve().parents[1]
    match={name:hashlib.sha256((root/name).read_text(encoding='utf8').encode()).hexdigest()==sha
           for name,sha in data['source_sha256'].items()}
    summary=dict(status=data['status'],physical_dual_t4=data['physical_dual_t4'],byte_exact_cases=len(cases),
        block_cases=len(data['block_cases']),ffn_cases=len(data['projection_cases']),source_matches_working_tree=match,
        comparisons=comparison,scope=data['scope'],
        no_memory_regression=all(r['memory_reduction_mib']>=0 for r in comparison),
        all_median_latencies_lower=all(r['latency_reduction_percent']>0 for r in comparison))
    # Compact trace evidence only; preserve compressed original traces.
    trace_evidence={}
    for trace in sorted(args.directory.glob('*rank*.json.gz')):
        events=json.loads(gzip.decompress(trace.read_bytes()))['traceEvents']
        ops=defaultdict(int);kernels=defaultdict(int);durations=defaultdict(float);collectives=[]
        for event in events:
            if event.get('ph')!='X':continue
            name=event.get('name','');category=event.get('cat')
            if category=='cpu_op':ops[name]+=1
            if category=='kernel':
                lower=name.lower()
                group='NCCL' if 'nccl' in lower else 'GEMM' if any(t in lower for t in ('gemm','cublas')) else 'other'
                kernels[group]+=1;durations[group]+=event['dur']/1000
                if group=='NCCL':
                    collectives.append(dict(operation='AllGather' if 'allgather' in lower else 'AllReduce',
                        trace_timestamp_us=event['ts'],duration_ms=event['dur']/1000))
        trace_evidence[trace.name]=dict(operator_counts={k:v for k,v in ops.items() if k in
            ('aten::cat','aten::copy_','aten::mul_','aten::add_') or 'allgather' in k.lower()},
            kernel_counts=dict(kernels),kernel_activity_ms=dict(durations),collective_intervals=collectives)
    summary['trace_evidence']=trace_evidence
    summary['trace_activity_caveat']='Kernel activity totals may overlap; NCCL kernel residence includes waits and is not pure link transfer time. Trace passes are outside timing samples.'
    (args.directory.parent/'summary.json').write_text(json.dumps(summary,indent=2))
    for row in comparison:
        print(row['stack'],row['rows'],row['mode'],
              f"{row['baseline']['critical_median_ms']:.3f}->{row['optimized']['critical_median_ms']:.3f}ms",
              f"{row['latency_reduction_percent']:.2f}% faster",
              f"{row['baseline']['temporary_peak_mib']:.2f}->{row['optimized']['temporary_peak_mib']:.2f}MiB")
    print('BYTE_EXACT',len(cases),'SOURCE_MATCH',all(match.values()))


if __name__=='__main__':main()
