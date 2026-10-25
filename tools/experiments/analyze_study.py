"""Canonical trace analysis: count real kernels, never GPU scope annotations."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def category(name):
    n=name.lower()
    if 'nccl' in n:
        return 'AllGather' if 'allgather' in n else 'AllReduce' if 'allreduce' in n else 'NCCL/P2P'
    if 'fmha' in n or 'flash' in n or 'attention' in n:return 'Attention'
    if 'quantize' in n or 'convrot' in n:return 'Quantization/dequantization'
    if 'i8816gemm' in n or 'igemm' in n or ('cutlass' in n and 'signed char' in n):return 'INT8 GEMM'
    if 'gemm' in n or 'gemv' in n:return 'Floating GEMM (rotation/AdaLN/other)'
    if 'copy' in n:return 'Layout/copy kernel'
    return 'Norm/activation/RoPE/residual/other'


def trace_record(path):
    events=json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
    groups=defaultdict(lambda:{'count':0,'duration_ms':0.0});devices=defaultdict(list)
    per_device=defaultdict(lambda:defaultdict(lambda:{'count':0,'duration_ms':0.0}))
    sync_calls=defaultdict(int)
    scope_ms=None
    memory=defaultdict(lambda:{'count':0,'duration_ms':0.0})
    for e in events:
        if e.get('ph')!='X':continue
        if e.get('cat')=='kernel':
            v=groups[category(e['name'])];v['count']+=1;v['duration_ms']+=e['dur']/1000
            device=str(e.get('args',{}).get('device','unknown'))
            devices[device].append((e['ts'],e['ts']+e['dur']))
            v=per_device[device][category(e['name'])];v['count']+=1;v['duration_ms']+=e['dur']/1000
        elif e.get('cat')=='gpu_memcpy':
            v=memory[e['name']];v['count']+=1;v['duration_ms']+=e['dur']/1000
        elif e.get('cat') in ('cuda_runtime','cuda_driver') and ('Synchronize' in e['name'] or 'WaitEvent' in e['name']):
            sync_calls[e['name']]+=1
        if e.get('cat')=='user_annotation' and e.get('name','').startswith('full::'):scope_ms=e['dur']/1000
    occupancy={}
    for d,intervals in devices.items():
        merged=[]
        for a,b in sorted(intervals):
            if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        busy=sum(b-a for a,b in merged);span=merged[-1][1]-merged[0][0]
        occupancy[d]={'kernel_span_ms':span/1000,'kernel_busy_union_ms':busy/1000,'between_kernel_idle_ms':(span-busy)/1000}
    return {'trace':path.relative_to(ROOT).as_posix(),'kernel_groups':dict(groups),'memcpy':dict(memory),'devices':occupancy,
        'per_device_kernel_groups':{d:dict(g) for d,g in per_device.items()},'synchronization_api_calls':dict(sync_calls),
        'profiled_cpu_call_scope_ms':scope_ms,
        'cuda_runtime_launch_calls':sum(1 for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'launch' in e.get('name','').lower()),
        'warning':'Profile perturbs timing. Kernel sums can overlap; NCCL activity includes waiting. CPU call markers may finish before asynchronous GPU work; GPU user annotations are excluded. Kernel-free gaps use each device kernel span, are not SM occupancy, and exclude memcpy engines and outer fill/drain.'}


def main():
    for phase in ['baseline','optimized','full_completion','ref2va_v4']:
        source=ROOT/'kaggle_study_output'/phase/'study_results'
        if not source.exists():continue
        if phase=='ref2va_v4':
            manifest=json.loads((source/'manifest.json').read_text())
            if manifest.get('phase')!='prescribed_ref2va_matrix':raise RuntimeError('Outputs belong to a different allocation')
        traces={p.relative_to(source).as_posix():trace_record(p) for p in source.rglob('*_trace.json.gz')}
        data={'phase':phase,'traces':traces,'tables':{}}
        for folder in ['te','dit','full_te','full_dit']:
            values=[]
            for p in (source/folder).glob('*_results.json'):
                rows=json.loads(p.read_text())
                for row in rows:
                    samples=row['wall_samples_ms']
                    if abs(statistics.median(samples)-row['median_ms'])>1e-6:raise RuntimeError('Median mismatch')
                    if not row['correctness_passed']:raise RuntimeError('Invalid timing accepted')
                    for label in ['pre_accuracy','post_accuracy']:
                        score=row[label]
                        if not(score['valid'] and score['rel_l2']<=1e-5 and score['cosine']>=.999999 and score['max_abs']<=.005):raise RuntimeError('Incorrect output accepted')
                    values.append(row)
            data['tables'][folder]=values
        target=ROOT/'kaggle_study_output'/phase/'analysis.json';target.write_text(json.dumps(data,indent=2))
        print(phase,'audited',sum(len(v) for v in data['tables'].values()),'measurements;',len(traces),'traces')
        index=['# Profiler trace index','', 'CUDA timings below are trace kernel sums, separately from unprofiled benchmark latency.','']
        for name,t in sorted(traces.items()):
            index += ['## '+name,'','[Chrome trace gzip]('+t['trace'].split(phase+'/',1)[-1]+')','',
                '| Category | Kernels | Duration sum (ms) |','|---|---:|---:|']
            for group,v in t['kernel_groups'].items():index.append(f"| {group} | {v['count']} | {v['duration_ms']:.3f} |")
            index.append('')
        (target.parent/'TRACE_INDEX.md').write_text('\n'.join(index),encoding='utf-8')


if __name__=='__main__':main()
