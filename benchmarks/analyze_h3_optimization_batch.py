"""Summarize retained operator/bake artifacts without contacting Kaggle."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return merged


def overlap_seconds(a,b):
    a=union(a);b=union(b);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total/1e9


def bake_worker(path):
    row=read(path);events=row['events'];phases={}
    for event in events:
        phase=phases.setdefault(event['phase'],dict(calls=0,wall_seconds=0.,thread_cpu_seconds=0.,exclusive_wall_seconds=0.))
        phase['calls']+=1
        for key,source in (('wall_seconds','wall_ns'),('thread_cpu_seconds','thread_cpu_ns'),('exclusive_wall_seconds','exclusive_wall_ns')):
            phase[key]+=event[source]/1e9
    def spans(names):return [(e['start_ns'],e['end_ns']) for e in events if e['phase'] in names]
    trace=read(path.with_name('trace_check.json'))
    return dict(seconds=row['seconds'],peak_rss_kib=row['peak_rss_kib'],gpu_peak_allocated=row['gpu_peak_allocated'],
        cpu_count=row['cpu_count'],phases=phases,trace=trace,
        writer_overlap_with_merge_verification_host_spans_seconds=overlap_seconds(
            spans({'persist_layer'}),spans({'merge','independent_verification'})))


def analyze(root):
    result=read(root/'operator_batch/result.json');binding=read(root/'light_setup/delivery_binding.json')
    if result['capsule_sha256']!=binding['capsule_sha256']:raise ValueError('Source binding mismatch')
    devices=result.get('hardware',{}).get('devices',[])
    physical_t4=(len(devices)==2 and all('T4' in d['name'] and d['capability']==[7,5] for d in devices)
        and len({d['uuid'] for d in devices})==2)
    micro=[];aot=[];blockers=[];dispatch_findings=[]
    for item in result.get('micro_cases',[]):
        value=item['result'];case=value.get('case',{});trace=value.get('trace',{})
        row=dict(label=item['label'],device=item['device'],status=value['status'],exit_code=item['exit_code'],
            kind=case.get('kind'),candidate_ms=value.get('candidate',{}).get('median_ms'),
            baseline_ms=value.get('baseline',{}).get('median_ms'),speedup=value.get('speedup'),
            kernel_count=trace.get('kernel_count'),kernel_names=trace.get('names'),
            candidate_peak_allocated=value.get('candidate',{}).get('peak_allocated'),
            baseline_peak_allocated=value.get('baseline',{}).get('peak_allocated'),
            error=value.get('error',value.get('error_vs_eager')),extension_calls=value.get('extension_calls'),
            global_cuda_unchanged=value.get('global_cuda_unchanged'),
            frequency_storage_dtype=value.get('frequency_storage_dtype'),
            loaded_cuda_libraries=value.get('loaded_cuda_libraries'),
            first_candidate_seconds=value.get('first_candidate_seconds',value.get('cold_candidate_seconds')))
        micro.append(row)
        declined={k:v for k,v in (value.get('extension_calls') or {}).items() if k.endswith('/returned_False') and v}
        if declined:dispatch_findings.append(dict(label=item['label'],declined=declined,
            reason='Extension declined; returned values/timing include its fallback. Do not count this as accelerated extension qualification.'))
        if item['exit_code']!=0:blockers.append(dict(label=item['label'],error=value.get('error_message',value.get('error'))))
        if case.get('kind')=='aot':
            cache=root/'operator_batch'/item['label']/'compiled_cache.tar.gz'
            aot.append(dict(label=item['label'],preparation_seconds=value.get('preparation_seconds'),
                preparation_compile_calls=value.get('preparation_compile_calls'),
                first_execution_compile_calls=value.get('first_execution_compile_calls'),
                second_execution_compile_calls=value.get('second_execution_compile_calls'),
                first_execution_seconds=value.get('first_execution_seconds'),
                second_execution_seconds=value.get('second_execution_seconds'),
                preparation=value.get('preparation'),
                cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest() if cache.exists() else None))
    bake_path=root/'operator_batch/bake/result.json';bake=None
    if bake_path.exists():
        source=read(bake_path);cases=[]
        for case in source.get('cases',[]):
            workers=[bake_worker(root/'operator_batch/bake'/case['label']/f'rank{rank}/result.json') for rank in (0,1)]
            cases.append(dict(case,workers=workers))
        paired=[]
        for trial in (0,1):
            rows={c['pipeline']:c for c in cases if c['label'].startswith(f'trial{trial}_')}
            if len(rows)==2:
                paired.append(dict(trial=trial,serial_seconds=rows[False]['wall_including_process_startup_seconds'],
                    pipeline_seconds=rows[True]['wall_including_process_startup_seconds'],
                    speedup=rows[False]['wall_including_process_startup_seconds']/rows[True]['wall_including_process_startup_seconds']))
        bake=dict(status=source['status'],cases=cases,pairs=paired,
            median_paired_speedup=statistics.median(p['speedup'] for p in paired) if paired else None,
            limits='Six layers, instrumented AB/BA; no full bake/parent-tail projection. Host span overlap is not measured GPU utilization.')
        if source['status']!='complete':blockers.append(dict(label='bake',error=source['status']))
    else:blockers.append(dict(label='bake',error='No retained bake result'))
    if not physical_t4:blockers.append(dict(label='hardware',error='Two physical T4s not established'))
    if len(micro)!=result['case_count']:blockers.append(dict(label='micro_cases',error='Case count incomplete'))
    return dict(status=result['status'],physical_two_t4=physical_t4,capsule_sha256=result['capsule_sha256'],
        hardware=result.get('hardware'),micro_cases=micro,aot=aot,bake=bake,blockers=blockers,dispatch_findings=dispatch_findings,
        scope='Operator-only synthetic activation fixtures and six real trained bake layers; no full generation or quality acceptance',
        timing_contract='Two warmups/five GPU-event samples for micro timings. AOT compilation/first-use rehearsal are separate. Bake timings include observers.',
        phase_contract='Per-thread phase sums are inclusive unless explicitly marked exclusive; do not add overlapping worker/thread times into a critical path.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();value=analyze(args.artifacts);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'analysis.json').write_text(json.dumps(value,indent=2))
    fields=('label','device','kind','status','candidate_ms','baseline_ms','speedup','kernel_count','candidate_peak_allocated','baseline_peak_allocated')
    with (args.output/'micro_timings.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(value['micro_cases'])
    print(json.dumps(dict(status=value['status'],cases=len(value['micro_cases']),physical_two_t4=value['physical_two_t4'],blockers=value['blockers'])))


if __name__=='__main__':main()
