"""Compact TE causality report from already captured physical CPU evidence."""
import argparse
import ast
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path

from analyze_causal_audit import duration, union, clipped, load_trace, read_json
from analyze_cpu_path import intersection


def block_key(event):
    try:
        value=ast.literal_eval(event.get('block',''))
        return value[1],value[2]
    except (ValueError, SyntaxError, IndexError, TypeError):
        return None


def analyze(folder):
    result=json.loads((folder/'result.json').read_text())
    if not (folder/'cpu_path_analysis.json').exists():
        from analyze_cpu_path import analyze as analyze_cpu
        analyze_cpu(folder,stage='reference_and_TE')
    if not (folder/'causal_analysis.json').exists():
        # Targeted captures have no coverage outside TE. Only report GPU
        # category unions inside that stage; never label other stages idle.
        rows=[]
        rank_kernels={r:load_trace(folder,r)[1] for r in (0,1)}
        for request in result['requests']:
            a,b=(request['stages']['reference_and_TE'][k] for k in ('start_ns','end_ns'))
            ranks={}
            for rank,kernels in rank_kernels.items():
                categories={e['category'] for e in kernels}
                unions={category:duration(clipped([(e['start_ns'],e['end_ns']) for e in kernels
                    if e['category']==category],a,b)) for category in categories}
                ranks[str(rank)]={'stage_category_busy_union_ms':{'reference_and_TE':unions}}
            rows.append(dict(request=request['request'],ranks=ranks))
        (folder/'causal_analysis.json').write_text(json.dumps(dict(requests=rows,scope='TE only; all other stages unobserved')))
    cpu=json.loads((folder/'cpu_path_analysis.json').read_text())
    causal=json.loads((folder/'causal_analysis.json').read_text())
    per_rank={r:read_json(folder/f'batch_r{r}_causal.json') for r in (0,1)}
    os_samples=defaultdict(list)
    sample_file=folder/'cpu_thread_samples.jsonl'
    if sample_file.exists():
        for line in sample_file.open():
            sample=json.loads(line)
            for row in sample['threads']:
                os_samples[(row['pid'],row['tid'])].append((sample['start_ns'],row['state'],row['wchan']))
    os_times={key:[s[0] for s in values] for key,values in os_samples.items()}
    output=dict(physical=result['physical'],passed=result['passed'],requests=[],blocks=[],
        limitations=['Read wall minus calling-thread CPU is nonrunning; it does not distinguish storage wait, descheduling, or GIL wait.',
                    'Scheduler run-delay disabled and native/GIL samples unavailable in this completed probe.',
                    'GPU NCCL duration includes synchronization; it is not measured link bandwidth.',
                    'Inclusive phases overlap; do not sum them as request latency.'])
    for req in result['requests']:
        a,b=(req['stages']['reference_and_TE'][k] for k in ('start_ns','end_ns'))
        request_row=dict(request=req['request'],te_stage_ms=(b-a)/1e6,ranks={})
        for rank,data in per_rank.items():
            events=[e for e in data['events'] if a<=e['start_ns']<b]
            waits=[w for w in cpu['proven_foreground_waits'] if w['rank']==rank and a<=w['start_ns']<b]
            preparation=[e for e in events if e['name']=='cpu_weight_prepare']
            te_prepare=[e for e in preparation if block_key(e) and block_key(e)[0]=='te']
            dit_prepare=[e for e in preparation if block_key(e) and block_key(e)[0]=='dit']
            dispatch=[e for e in events if e['name']=='rank_dispatch/te']
            io=[e for e in events if e['name']=='checkpoint_file_io']
            hashes=[e for e in events if e['name']=='cpu_tensor_hashes']
            d_samples=[]
            for e in io:
                samples=os_samples[(e['pid'],e['tid'])]
                times=os_times.get((e['pid'],e['tid']),[])
                i,j=bisect_left(times,e['start_ns']),bisect_left(times,e['end_ns'])
                d_samples.extend(s for s in samples[i:j] if s[1]=='D')
            categories=causal['requests'][req['request']]['ranks'][str(rank)]['stage_category_busy_union_ms']['reference_and_TE']
            request_row['ranks'][str(rank)]=dict(
                prepare_wall_union_ms=duration([(e['start_ns'],e['end_ns']) for e in te_prepare]),
                prepare_thread_cpu_ms=sum(e.get('thread_cpu_running_ns',0) for e in te_prepare)/1e6,
                read_wall_ms=sum(e['read_wall_ns'] for e in io)/1e6,
                read_thread_cpu_ms=sum(e['read_thread_cpu_ns'] for e in io)/1e6,
                logical_read_bytes=sum(e['read_bytes'] for e in io),
                hash_wall_ms=sum(e['end_ns']-e['start_ns'] for e in hashes)/1e6,
                hash_thread_cpu_ms=sum(e.get('thread_cpu_running_ns',0) for e in hashes)/1e6,
                pin_wall_ms=sum(e['end_ns']-e['start_ns'] for e in events if e['name']=='cpu_pin')/1e6,
                foreground_wait_union_ms=duration([(w['start_ns'],w['end_ns']) for w in waits]),
                producer_read_D_state_sample_count=len(d_samples),
                producer_read_D_wait_channels=dict((ch,sum(s[2]==ch for s in d_samples)) for ch in sorted({s[2] for s in d_samples})),
                speculative_dit_prepare_overlap_with_TE_dispatch_ms=duration(intersection(
                    [(e['start_ns'],e['end_ns']) for e in dit_prepare],[(e['start_ns'],e['end_ns']) for e in dispatch])),
                gpu_category_union_ms=categories)
            for index in range(50):
                selected=[e for e in events if block_key(e)==('te',index)]
                block_waits=[w for w in waits if block_key(w)==('te',index)]
                reads=[e for e in selected if e['name']=='checkpoint_file_io']
                prep=[e for e in selected if e['name']=='cpu_weight_prepare']
                gpu=[e for e in data['gpu_intervals'] if block_key(e)==('te',index) and a<=e['start_ns']<b]
                output['blocks'].append(dict(request=req['request'],rank=rank,block=index,
                    preparation_count=len(prep),read_bytes=sum(e['read_bytes'] for e in reads),
                    read_wall_ms=sum(e['read_wall_ns'] for e in reads)/1e6,
                    read_thread_cpu_ms=sum(e['read_thread_cpu_ns'] for e in reads)/1e6,
                    prepare_intervals_s=[[(e['start_ns']-req['start_ns'])/1e9,(e['end_ns']-req['start_ns'])/1e9] for e in prep],
                    foreground_wait_intervals_s=[[(e['start_ns']-req['start_ns'])/1e9,(e['end_ns']-req['start_ns'])/1e9] for e in block_waits],
                    foreground_wait_union_ms=duration([(w['start_ns'],w['end_ns']) for w in block_waits]),
                    publication_bounds=[w.get('post_ready_return_gap_bounds_ms') for w in block_waits],
                    cuda_event_intervals=[dict(name=e['name'],start_s=(e['start_ns']-req['start_ns'])/1e9,
                        end_s=(e['end_ns']-req['start_ns'])/1e9) for e in gpu]))
        output['requests'].append(request_row)
    (folder/'te_critical_path.json').write_text(json.dumps(output,separators=(',',':')))
    print(json.dumps(output['requests'],indent=2))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    analyze(parser.parse_args().folder)
