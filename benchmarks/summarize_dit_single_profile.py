"""CPU-only analysis of returned one-NFE artifacts; never contacts Kaggle."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile


def union_ns(intervals):
    total=0;end=-1
    for a,b in sorted(intervals):total+=max(0,b-max(a,end));end=max(end,b)
    return total


def attribute(database):
    db=sqlite3.connect(database);db.row_factory=sqlite3.Row
    try:
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        required={'StringIds','NVTX_EVENTS','CUPTI_ACTIVITY_KIND_KERNEL'}
        if not required.issubset(tables):return dict(blocker='Missing Nsight SQLite tables',tables=sorted(tables))
        names=dict(db.execute('select id,value from StringIds'))
        ranges=defaultdict(list)
        for row in db.execute('select * from NVTX_EVENTS where end>start'):
            r=dict(row);label=r.get('text') or names.get(r.get('textId'),'')
            if label.startswith(('H3_DIT/','denoise/nfe')):
                ranges[r['globalTid']].append((r['start'],r['end'],label))
        calls={};by_thread=defaultdict(list)
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
            if table not in tables:continue
            for row in db.execute('select * from '+table):
                r=dict(row);pid=r['globalTid']&~((1<<24)-1);key=(pid,r['correlationId'])
                if key not in calls:
                    r['api_name']=names.get(r.get('nameId'),'unknown');calls[key]=r
                    by_thread[r['globalTid']].append(r)
        for tid,queries in by_thread.items():
            events=sorted(ranges.get(tid,[]));position=0;active={};ends=[]
            for query in sorted(queries,key=lambda x:x['start']):
                t=query['start']
                while position<len(events) and events[position][0]<=t:
                    a,b,label=events[position];active[position]=(a,b,label)
                    heapq.heappush(ends,(b,position));position+=1
                while ends and ends[0][0]<=t:
                    _,i=heapq.heappop(ends);active.pop(i,None)
                query['in_denoise']=any(x[2].startswith('denoise/nfe') for x in active.values())
                semantic=[x for x in active.values() if x[2].startswith('H3_DIT/')]
                label=min(semantic,key=lambda x:x[1]-x[0])[2] if semantic else 'unattributed'
                query['phase']=label.split('/',3)[-1] if label.startswith('H3_DIT/') else label
        phases=defaultdict(lambda:dict(kernel_count=0,kernel_sum_ms=0.,kernel_intervals=[]))
        intervals=defaultdict(list);kernels=defaultdict(lambda:dict(count=0,sum_ms=0.));matched=0
        pid_device={};memcpy=defaultdict(lambda:dict(count=0,bytes=0,sum_ms=0.))
        for row in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
            r=dict(row);launch=calls.get((r['globalPid'],r['correlationId']))
            if not launch or not launch.get('in_denoise'):continue
            matched+=1;device=r['deviceId'];pid_device[r['globalPid']]=device
            name=names.get(r.get('shortName'),names.get(r.get('demangledName'),'unknown'))
            phase='NCCL/collective' if 'nccl' in name.lower() else launch['phase']
            p=phases[(device,phase)];p['kernel_count']+=1;p['kernel_sum_ms']+=(r['end']-r['start'])/1e6
            p['kernel_intervals'].append((r['start'],r['end']));intervals[device].append((r['start'],r['end']))
            k=kernels[(device,name)];k['count']+=1;k['sum_ms']+=(r['end']-r['start'])/1e6
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            for row in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'):
                r=dict(row);pid=r.get('globalPid');launch=calls.get((pid,r.get('correlationId')))
                if not launch or not launch.get('in_denoise'):continue
                p=memcpy[(r['deviceId'],r.get('copyKind'))];p['count']+=1;p['bytes']+=r.get('bytes',0);p['sum_ms']+=(r['end']-r['start'])/1e6
        apis=defaultdict(lambda:dict(count=0,host_sum_ms=0.))
        for (pid,_),r in calls.items():
            if r.get('in_denoise'):
                p=apis[(pid_device.get(pid),r['api_name'])];p['count']+=1;p['host_sum_ms']+=(r['end']-r['start'])/1e6
        timeline=[]
        for device,iv in sorted(intervals.items()):
            span=max(b for a,b in iv)-min(a for a,b in iv);busy=union_ns(iv)
            timeline.append(dict(device=device,span_ms=span/1e6,kernel_active_union_ms=busy/1e6,
                kernel_timeline_busy_fraction=busy/span if span else None))
        categories=[]
        for (device,phase),p in sorted(phases.items()):
            categories.append(dict(device=device,phase=phase,kernel_count=p['kernel_count'],
                kernel_sum_ms=p['kernel_sum_ms'],kernel_active_union_ms=union_ns(p['kernel_intervals'])/1e6))
        return dict(matched_denoise_kernels=matched,timeline=timeline,phases=categories,
            kernels=[dict(device=d,name=n,**v) for (d,n),v in sorted(kernels.items(),key=lambda x:x[1]['sum_ms'],reverse=True)],
            api=[dict(device=d,name=n,**v) for (d,n),v in sorted(apis.items(),key=lambda x:x[1]['host_sum_ms'],reverse=True)],
            copies=[dict(device=d,copy_kind=k,**v) for (d,k),v in memcpy.items()],
            limitations=['Process-qualified CUDA correlation IDs; only launches within denoise NVTX ranges',
                'Kernel timeline coverage is not SM occupancy or Tensor Core utilization',
                'Kernel sums can overlap; category union intervals must not be added as exclusive critical-path time',
                'Nsight Systems does not establish memory stalls, achieved occupancy or hardware bandwidth counters'])
    finally:db.close()


def summarize(root,output):
    output.mkdir(parents=True,exist_ok=True)
    path=root/'light_execution/dit_single/dit_profile';result_file=path/'result.json'
    if not result_file.exists():
        record=dict(status='missing_dit_result',blocker='No returned single-call result; inspect setup/execution receipts')
        (output/'summary.json').write_text(json.dumps(record,indent=2));return record
    result=json.loads(result_file.read_text());profilers=json.loads((path/'profilers.json').read_text()) if (path/'profilers.json').exists() else {}
    hardware_file=root/'light_execution/batch_results.json'
    hardware=json.loads(hardware_file.read_text()).get('hardware') if hardware_file.exists() else None
    ranks=[]
    for rank in result.get('ranks',[]):
        bench=rank['benchmark'];commands=[c for c in bench['commands'] if c['command']=='denoise']
        gpu=[g for g in bench['gpu'] if g['kind']=='command_gpu_span' and g['command']=='denoise']
        ranks.append(dict(rank=rank['rank'],calls=rank['calls'],denoise_commands=commands,command_gpu_spans=gpu,
            executed_backends=rank.get('denoise_backend_counts'),operator_warmup=rank.get('operator_warmup'),
            phase_cpu_counts=len(rank.get('performance_phases',{}).get('cpu',[]))))
    record=dict(status=result['status'],hardware=hardware,ranks=ranks,profilers=profilers,
        result_sha256=hashlib.sha256(result_file.read_bytes()).hexdigest(),
        sample_wrapper_seconds=result.get('sample_wrapper_seconds'),inputs=result.get('inputs'),
        quality_acceptance='One call only; no full-generation quality or latency-distribution claim')
    with tempfile.TemporaryDirectory(prefix='h3_dit_sql_',dir=output) as temp:
        database=Path(temp)/'dit.sqlite';compressed=path/'dit_single.sqlite.gz'
        if compressed.exists():
            with gzip.open(compressed,'rb') as src,database.open('wb') as dst:shutil.copyfileobj(src,dst)
            record['nsight']=attribute(database)
        elif (path/'dit_single.sqlite').exists():record['nsight']=attribute(path/'dit_single.sqlite')
        else:record['nsight']=dict(blocker='No returned SQLite export; raw Systems report/logs require review')
    devices=(hardware or {}).get('devices',[])
    record['hardware_verified']=len(devices)==2 and len({d.get('uuid') for d in devices})==2 and all(
        d.get('capability')==[7,5] and 'T4' in d.get('name','') for d in devices)
    record['profiling_status']='nsight_attributed' if record['nsight'].get('matched_denoise_kernels',0)>0 else 'partial_profiler_blocked'
    if record['status']=='complete' and not record['hardware_verified']:record['status']='hardware_unverified'
    (output/'summary.json').write_text(json.dumps(record,indent=2))
    lines=['# Single DiT-call profiling results','',f"Execution status: **{record['status']}**.",'',
        'One instrumented 51,220-token call; setup, conditioning and operator warmup are separate.','',
        '| Rank | DiT calls | CUDA command span (s) | Peak allocated (GiB) |','|---|---:|---:|---:|']
    for r in ranks:
        span=sum(x['cuda_ms'] for x in r['command_gpu_spans'])/1000
        peak=max((x['peak_allocated'] for x in r['denoise_commands']),default=0)/2**30
        lines.append(f"| {r['rank']} | {r['calls']} | {span:.3f} | {peak:.3f} |")
    phases=record['nsight'].get('phases',[])
    if phases:
        lines += ['', 'Exclusive assignment of kernels to the innermost semantic scope; summed kernel durations can overlap.', '',
            '| GPU | Phase | Kernels | Kernel duration sum (ms) |','|---|---|---:|---:|']
        for p in sorted(phases,key=lambda x:(x['device'],-x['kernel_sum_ms'])):
            lines.append(f"| {p['device']} | {p['phase']} | {p['kernel_count']} | {p['kernel_sum_ms']:.3f} |")
    else:lines += ['',f"Profiler blocker: {record['nsight'].get('blocker','No correlated kernels')}."]
    lines += ['', 'No memory-stall, occupancy or bandwidth-counter conclusions are inferred from Systems or CUDA event durations.',
        'Inspect summary.json and raw .nsys-rep/CSV evidence before making optimization decisions.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=summarize(args.input,args.output)
    print(json.dumps(dict(status=result['status'],ranks=len(result.get('ranks',[])),nsight_kernels=result.get('nsight',{}).get('matched_denoise_kernels'))))


if __name__=='__main__':main()
