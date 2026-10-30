"""Offline, source-verified TE O_DIRECT analysis; accepts compressed public artifacts."""
import argparse
import ast
import base64
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path, PureWindowsPath
import statistics


def opened(path):
    if path.exists():
        return gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open(encoding='utf-8')
    return gzip.open(str(path)+'.gz', 'rt', encoding='utf-8')


def load(path):
    with opened(path) as stream:
        return json.load(stream)


def union(intervals):
    total, end = 0., -float('inf')
    for a, b in sorted(intervals):
        total += max(0., b-max(a,end))
        end = max(end,b)
    return total


def event_union(events):
    return union((e['ts'],e['ts']+e['dur']) for e in events if e.get('dur',0)>0)/1e6


def trace_summary(path):
    trace = load(path)
    events = trace['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    nccl = [e for e in kernels if 'nccl' in e['name'].lower()]
    compute = [e for e in kernels if 'nccl' not in e['name'].lower()]
    copies = [e for e in events if e.get('cat') == 'gpu_memcpy']
    scale = sorted([e for e in nccl if 'AllReduce_f32_RING' in e['name']],key=lambda e:e['ts'])
    groups = defaultdict(lambda:dict(calls=0,work_seconds=0.))
    for e in kernels:
        groups[e['name']]['calls'] += 1
        groups[e['name']]['work_seconds'] += e.get('dur',0)/1e6
    span = (max(e['ts']+e['dur'] for e in kernels+copies)-min(e['ts'] for e in kernels+copies))/1e6
    summary = dict(kernel_count=len(kernels),gpu_span_seconds=span,
        trace_base_time_nanoseconds=trace['baseTimeNanoseconds'],
        non_nccl_kernel_union_seconds=event_union(compute),nccl_union_seconds=event_union(nccl),
        memcpy_union_seconds=event_union(copies),all_gpu_activity_union_seconds=event_union(kernels+copies),
        top_kernels=sorted([dict(name=k,**v) for k,v in groups.items()],key=lambda r:r['work_seconds'],reverse=True)[:12])
    return summary, scale


def memory_peaks(samples):
    return dict(current_bytes=max(int(s['memory_current']) for s in samples),
        file_bytes=max(s['memory_stat']['file'] for s in samples),
        anon_bytes=max(s['memory_stat']['anon'] for s in samples),
        shmem_bytes=max(s['memory_stat'].get('shmem',0) for s in samples))


def phase_counters(samples,begin,end):
    before = max((s for s in samples if s['monotonic_ns'] <= begin),
                 key=lambda s:s['monotonic_ns'],default=samples[0])
    after = min((s for s in samples if s['monotonic_ns'] >= end),
                key=lambda s:s['monotonic_ns'],default=samples[-1])
    seconds = (after['monotonic_ns']-before['monotonic_ns'])/1e9
    fields = ('pgscan','pgsteal','pgscan_direct','workingset_refault_file','pgmajfault')
    return dict(sampled_seconds=seconds,
        memory_stat_delta={k:after['memory_stat'].get(k,0)-before['memory_stat'].get(k,0) for k in fields},
        psi_memory_seconds={k:(after['psi_memory'].get(k,0)-before['psi_memory'].get(k,0))/1e6 for k in ('some','full')},
        memory_peaks=memory_peaks([s for s in samples if before['monotonic_ns'] <= s['monotonic_ns'] <= after['monotonic_ns']]))


def analyze(root,receipt,output):
    submission = load(receipt)
    manifest = load(root/'download_manifest.json')
    for name, expected in manifest.items():
        path = root/PureWindowsPath(name).as_posix()
        if path.exists():
            data = path.read_bytes()
        else:
            with gzip.open(str(path)+'.gz','rb') as stream:
                data = stream.read()
        assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'],name
    code = (root/'te_direct_supervision/bounded_worker.py').read_text(encoding='utf-8')
    files = next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == 'FILES' for t in n.targets))
    expected_sources = submission['source_binding']['benchmark_sources']
    assert set(files) == set(expected_sources)
    for name, encoded in files.items():
        assert hashlib.sha256(base64.b64decode(encoded)).hexdigest() == expected_sources[name],name
    source = root/'te_direct_io'
    result = load(source/'result.json')
    assert result['status'] == 'complete' and result['conditioning_bit_exact'] and result['complete_gpu_profiles']
    assert load(root/'light_setup/setup_receipt.json')['capsule_sha256'] == submission['source_binding']['publication']['capsule_sha256']
    cases,rank_metrics = [],[]
    for label in ('a1','b1','b2','a2'):
        directory = source/label
        case = load(directory/'case.json')
        assert case['status'] == 'complete'
        with opened(directory/'os_samples.jsonl') as stream:
            samples = [json.loads(line) for line in stream]
        ranks = [load(directory/f'rank{rank}.json') for rank in (0,1)]
        row = dict(label=label,mode=case['mode'],startup_seconds=case['startup_seconds'],calls=case['calls'],
            os_epoch_delta=load(directory/'os_epoch.json')['counters'],memory_peaks=memory_peaks(samples),
            sampler_cpu_seconds=sum(s['sampler_cpu_ns'] for s in samples)/1e9,phases={})
        for phase in ('te1','te2'):
            commands = [next(c for c in rank['commands'] if c['phase'] == phase) for rank in ranks]
            row['phases'][phase] = phase_counters(samples,min(c['start_ns'] for c in commands),max(c['end_ns'] for c in commands))
        for rank,detail in enumerate(ranks):
            assert detail['dropped_io_rows'] == 0
            groups = defaultdict(lambda:dict(calls=0,work_wall_seconds=0.,thread_cpu_seconds=0.,bytes=0))
            with opened(directory/f'rank{rank}_io.jsonl.gz') as stream:
                for line in stream:
                    e = json.loads(line)
                    if e.get('layer') is None or e['phase'] != 'te1':
                        continue
                    item = groups[e['kind']]
                    item['calls'] += 1
                    item['work_wall_seconds'] += (e['end_ns']-e['start_ns'])/1e9
                    item['thread_cpu_seconds'] += e.get('thread_cpu_ns',0)/1e9
                    item['bytes'] += e.get('bytes',0)
            assert groups['file_sha256']['calls'] == groups['tensor_sha256']['calls'] == 50
            for phase in ('te1','te2'):
                blocks = [b for b in detail['blocks'] if b['phase'] == phase]
                assert len(blocks) == 50
            all_groups = load(directory/f'rank{rank}_io.summary.json')['groups']
            repeat_hashes = [g for g in all_groups if g['phase'] == 'te2' and g['kind'] in ('file_sha256','tensor_sha256')]
            assert not repeat_hashes
            cache = detail['stats']['host_cache']
            rank_metrics.append(dict(label=label,mode=case['mode'],rank=rank,te1_layer_work=dict(groups),
                te2_hash_calls=0,host_cache={k:cache[k] for k in ('preparations','hits','misses','evictions','cancelled_prefetches','peak_cache_bytes')},
                commands=detail['commands']))
        cases.append(row)
    traces = {}
    for label in ('a1','b2'):
        rows,scales = [],[]
        for rank in (0,1):
            summary,collectives = trace_summary(source/label/f'rank{rank}_trace.json.gz')
            rows.append(dict(rank=rank,**summary));scales.append(collectives)
        assert len(scales[0]) == len(scales[1]) == 100
        assert rows[0]['trace_base_time_nanoseconds'] == rows[1]['trace_base_time_nanoseconds']
        launch_skew = [abs(a['ts']-b['ts'])/1e6 for a,b in zip(*scales)]
        completion_skew = [abs(a['ts']+a['dur']-b['ts']-b['dur'])/1e6 for a,b in zip(*scales)]
        worst = max(range(100),key=lambda i:launch_skew[i])
        pair = [events[worst] for events in scales]
        origin = min(e['ts'] for e in pair)
        traces[label] = dict(ranks=rows,fp32_collective_pairs=100,
            fp32_kernel_work_seconds_by_rank=[sum(e['dur'] for e in events)/1e6 for events in scales],
            worst_pair_index=worst,
            worst_pair_kernel_intervals_ms=[dict(rank=i,start=(e['ts']-origin)/1000,duration=e['dur']/1000) for i,e in enumerate(pair)],
            sum_absolute_kernel_entry_skew_seconds=sum(launch_skew),
            max_kernel_entry_skew_seconds=max(launch_skew),
            max_paired_completion_skew_microseconds=max(completion_skew)*1e6)
    comparison = {}
    for mode in ('buffered','direct'):
        selected = [c for c in cases if c['mode'] == mode]
        first = [c['calls'][0]['wall_seconds'] for c in selected]
        repeats = [call['wall_seconds'] for c in selected for call in c['calls'] if call['repeat'] == 1 and not call['profiled']]
        comparison[mode] = dict(first_verified_seconds=first,first_verified_median_seconds=statistics.median(first),
            unprofiled_repeat_seconds=repeats,
            first_phase_scan_pages=[c['phases']['te1']['memory_stat_delta']['pgscan'] for c in selected],
            first_phase_refault_pages=[c['phases']['te1']['memory_stat_delta']['workingset_refault_file'] for c in selected])
    comparison['first_verified_direct_slowdown_percent'] = (comparison['direct']['first_verified_median_seconds']/comparison['buffered']['first_verified_median_seconds']-1)*100
    comparison['single_repeat_direct_slowdown_percent'] = (comparison['direct']['unprofiled_repeat_seconds'][0]/comparison['buffered']['unprofiled_repeat_seconds'][0]-1)*100
    summary = dict(status='complete',source_files_verified=len(manifest),supplemental_sources=expected_sources,
        hardware=result['devices'],software=result['software'],conditioning_bit_exact=True,
        conditioning_shape=cases[0]['calls'][0]['shape'],cases=cases,rank_metrics=rank_metrics,
        comparison=comparison,traces=traces,raw_io=result['raw_io'],
        limitations=['Two first-use observations and one unprofiled repeat per mode.',
            'Counter phase boundaries use approximately 0.2-second sampling; cgroup/device activity is shared.',
            'Direct adapter includes aligned bounce buffers and extra Python copies.',
            'Native overlay/direct gate does not establish physically cold storage or lower-layer cache bypass.',
            'GPU kernel intervals, especially NCCL waiting, are not achieved SM utilization.',
            'No Nsight Compute hardware-counter attribution is available from these Kineto traces.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (output/'latency.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream);writer.writerow(['case','mode','call','profiled','seconds'])
        writer.writerows([c['label'],c['mode'],x['repeat'],x['profiled'],x['wall_seconds']] for c in cases for x in c['calls'])
    plot(summary,output)
    return summary


def plot(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    colors = ['#397db8','#da7842']
    for index,mode in enumerate(('buffered','direct')):
        data = summary['comparison'][mode]
        axes[0].bar(index,data['first_verified_median_seconds'],color=colors[index],width=.65)
        axes[0].plot([index-.05,index+.05],data['first_verified_seconds'],'ko',markersize=4)
        axes[0].text(index,data['first_verified_median_seconds']+6,f"{data['first_verified_median_seconds']:.1f} s",ha='center')
        rows = [r for r in summary['rank_metrics'] if r['mode'] == mode]
        means = {k:statistics.mean(r['te1_layer_work'][k]['thread_cpu_seconds'] for r in rows)
                 for k in ('file_sha256','tensor_sha256','read_tensor','block_cpu_prepare')}
        other = means['block_cpu_prepare']-sum(means[k] for k in ('file_sha256','tensor_sha256','read_tensor'))
        assert other >= 0
        bottom = 0
        for name,value,color in [('File verification',means['file_sha256'],'#6b95c3'),('Tensor verification',means['tensor_sha256'],'#a0bedc'),
                                  ('Tensor loading',means['read_tensor'],'#d89259'),('Other preparation',other,'#cccccc')]:
            axes[1].bar(index,value,bottom=bottom,color=color,label=name if index == 0 else None,width=.65)
            bottom += value
    axes[0].set(title='First verified TE call (2 observations/mode)',ylabel='End-to-end seconds',ylim=(0,190))
    axes[1].set(title='CPU layer preparation per rank (work totals)',ylabel='Thread CPU seconds')
    for ax in axes:
        ax.set_xticks([0,1],['Buffered','O_DIRECT']);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[1].legend(fontsize=8)
    fig.suptitle('Physical 2xT4 TE, 2,528 tokens: fewer reclaim scans did not make TE faster',fontsize=12)
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for ax,label,name in zip(axes,('a1','b2'),('Buffered diagnostic','O_DIRECT diagnostic')):
        info = summary['traces'][label]
        for interval in info['worst_pair_kernel_intervals_ms']:
            rank = interval['rank']
            ax.barh(rank,interval['duration'],left=interval['start'],height=.35,color=colors[rank])
            ax.plot(interval['start'],rank,'|',color='black',markersize=14)
        ax.set(yticks=[0,1],yticklabels=['Rank 0','Rank 1'],xlabel='Milliseconds from first kernel entry',
               title=name+'\nWorst paired FP32 collective')
        ax.set_ylim(-.6,1.6);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.text(.02,.97,f"Entry skew: {info['max_kernel_entry_skew_seconds']*1000:.1f} ms\n"
                f"All paired completion skews <= {info['max_paired_completion_skew_microseconds']:.1f} us",
                transform=ax.transAxes,va='top',fontsize=9)
    fig.suptitle('Long NCCL kernel intervals include waiting for the peer to arrive',fontsize=12)
    fig.savefig(output/'collective_entry_skew.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--submission',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = analyze(args.artifacts,args.submission,args.output)
    print(json.dumps(result['comparison'],indent=2))
