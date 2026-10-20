"""Block 2–49 dependency exposure, merging both ranks and parent GPU work."""
import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path

from analyze_causal_audit import load_trace,read_json,union,clipped,duration,IntervalIndex
from analyze_cpu_path import intersection,complement


def block(event):
    try:
        value=ast.literal_eval(event.get('block') or event.get('legacy_block') or '')
        return int(value[2]) if value[1]=='te' else None
    except (ValueError,SyntaxError,TypeError,IndexError):return None


def spans(events):return [(e['start_ns'],e['end_ns']) for e in events]


def analyze(folder):
    result=read_json(folder/'result.json')
    if not result['passed'] or not result['physical']:raise ValueError('Physical correctness required')
    traces={};kernels=[];apis=[]
    for rank in ('parent',0,1):
        trace,gpu,runtime=load_trace(folder,rank)
        traces[rank]=trace;kernels+=gpu;apis+=runtime
    # Version-1 capture has kernels only for warm request 1. Never call the
    # uncaptured first request GPU-idle, even if its CPU trace exists.
    request=next(r for r in result['requests'] if r['request']==1)
    begin,end=(request['stages']['reference_and_TE'][k] for k in ('start_ns','end_ns'))
    clocks={str(k):dict(cuda_anchor_uncertainty_ns=v.get('alignment_uncertainty_ns'),
        kineto_fit=v.get('kineto_clock_fit')) for k,v in traces.items()}
    if any(not t.get('kineto_clock_fit',{}).get('anchors') for t in traces.values()):
        raise ValueError('Missing per-process CPU/Kineto clock alignment')
    idle={d:complement(spans([k for k in kernels if k['device']==d]),begin,end) for d in (0,1)}
    both=intersection(idle[0],idle[1]);rows=[];linked=[];per_block_exposure=defaultdict(list)
    for rank in (0,1):
        trace=traces[rank];events=trace['events'];pid=trace['pid']
        api_by_correlation={e['args'].get('correlation'):e for e in apis if e['rank']==rank
            and e.get('args',{}).get('correlation') is not None}
        relevant=[e for e in events if e.get('request')==1]
        publications=defaultdict(list)
        for e in events:
            if e['name']=='future_ready_bounds':publications[e['future']].append(e)
        for index in range(2,50):
            selected=[e for e in relevant if block(e)==index]
            named=lambda name:[e for e in selected if e['name']==name]
            prepares=named('cpu_weight_prepare');staging=named('cpu_staging')
            reads=named('checkpoint_file_io');pins=named('cpu_pin')
            queued=[e for e in named('cpu_queue') if not e.get('cached') and not e.get('already_queued')]
            submit=named('cpu_staging_queue');h2d=named('h2d_enqueue')
            memcopies=[]
            for kernel in kernels:
                if kernel['rank']!=rank or kernel['category']!='memcpy':continue
                launch=api_by_correlation.get(kernel.get('args',{}).get('correlation'))
                if launch and any(e['start_ns']<=launch['start_ns']<=e['end_ns'] for e in h2d):
                    memcopies.append(kernel)
            waits=[e for e in named('future_consumer_wait') if e['tid']==pid and not e.get('ready_on_entry',True)]
            exposed=[];links=[]
            for wait in waits:
                ready=[e for e in publications.get(wait['future'],[]) if
                    e['lower_ns']<=wait['end_ns'] and e['upper_ns']>=wait['start_ns']]
                following=[e for e in h2d if e['tid']==pid and e['start_ns']>=wait['end_ns']]
                if len(ready)!=1 or not following:continue
                transfer=min(following,key=lambda e:e['start_ns'])
                intervals=intersection([(wait['start_ns'],wait['end_ns'])],both)
                if ready[0]['upper_ns']>transfer['start_ns']:continue
                exposed+=intervals;links.append(dict(future=wait['future'],
                    wait_s=[(wait[k]-request['start_ns'])/1e9 for k in ('start_ns','end_ns')],
                    readiness_s=[(ready[0][k]-request['start_ns'])/1e9 for k in ('lower_ns','upper_ns')],
                    h2d_submission_s=(transfer['start_ns']-request['start_ns'])/1e9,
                    both_gpu_idle_intersection_ms=duration(intervals)))
            linked+=exposed;per_block_exposure[index]+=exposed
            actual=[k for k in kernels if k['rank']==rank and
                block(k.get('launch',{}).get('owner') or {})==index and k['start_ns']<end and k['end_ns']>begin]
            gemm=[k for k in actual if k['category']=='gemm']
            queue_delay=(max(0,prepares[0]['start_ns']-min(e['start_ns'] for e in queued))/1e6
                if prepares and queued else None)
            stage_delay=(max(0,staging[0]['start_ns']-min(e['start_ns'] for e in submit))/1e6
                if staging and submit else None)
            rows.append(dict(rank=rank,block=index,
                prefetch_requested_s=[(e['start_ns']-request['start_ns'])/1e9 for e in queued],
                prepare_intervals_s=[[(e[k]-request['start_ns'])/1e9 for k in ('start_ns','end_ns')] for e in prepares],
                observed_cache_queue_delay_lower_bound_ms=queue_delay,
                observed_staging_queue_delay_lower_bound_ms=stage_delay,
                prepare_wall_union_ms=duration(spans(prepares)),
                prepare_thread_cpu_ms=sum(e.get('thread_cpu_running_ns',0) for e in prepares)/1e6,
                read_call_wall_ms=sum(e['read_wall_ns'] for e in reads)/1e6,
                read_thread_cpu_ms=sum(e['read_thread_cpu_ns'] for e in reads)/1e6,
                logical_read_bytes=sum(e['read_bytes'] for e in reads),
                file_intervals=[dict(start_s=(e['start_ns']-request['start_ns'])/1e9,
                    end_s=(e['end_ns']-request['start_ns'])/1e9,bytes=e['read_bytes'],source=e['source']) for e in reads],
                pin_wall_union_ms=duration(spans(pins)),foreground_wait_union_ms=duration(spans(waits)),
                linked_both_gpu_inactive_ms=duration(exposed),links=links,
                linked_both_idle_intervals_s=[[(a-request['start_ns'])/1e9,(b-request['start_ns'])/1e9]
                    for a,b in union(exposed)],
                h2d_submission_intervals_s=[[(e[k]-request['start_ns'])/1e9 for k in ('start_ns','end_ns')] for e in h2d],
                kineto_memcpy_intervals_s=[[(a-request['start_ns'])/1e9,(b-request['start_ns'])/1e9]
                    for a,b in union(spans(memcopies))],
                cuda_stream_intervals=[dict(name=e['name'],start_s=(e['start_ns']-request['start_ns'])/1e9,
                    end_s=(e['end_ns']-request['start_ns'])/1e9) for e in trace['gpu_intervals']
                    if block(e)==index and begin<=e['start_ns']<end],
                first_correlated_block_kernel_s=min(((k['start_ns']-request['start_ns'])/1e9 for k in actual),default=None),
                first_correlated_weight_gemm_s=min(((k['start_ns']-request['start_ns'])/1e9 for k in gemm),default=None),
                nccl_kernel_union_ms=duration(spans([k for k in actual if k['category']=='nccl'])),
                event_clock_semantics='CUDA-event durations valid; long-run absolute anchor drift is not bounded by initial calibration uncertainty. Use CPU-fitted Kineto memcpy/kernel timestamps for cross-clock ordering.',
                os_observations=[{k:v for k,v in e.items() if k not in ('name','block','window')}
                    for e in named('te_block_os_counters')],
                queue_submission_bounds=[{k:e[k] for k in ('name','future','lower_ns','upper_ns')}
                    for e in selected if e['name'] in ('cache_queue_bounds','executor_submit_bounds')],
                attribution='Exact Future publication plus same-thread H2D predecessor, intersected with inactivity on both physical GPUs; not a complete counterfactual DAG.'))
    pair_order=sorted(per_block_exposure,key=lambda i:duration(per_block_exposure[i]),reverse=True)
    forecasts=[]
    for gap in (0,5,15,60):
        elapsed=0.;selected=[];bytes_used={0:0,1:0};eligible=[]
        for index in range(2,50):
            pair=[r for r in rows if r['block']==index]
            sizes={r['rank']:max((e.get('bytes',0) for e in traces[r['rank']]['events']
                if e['name']=='h2d_enqueue' and e.get('request')==1 and block(e)==index),default=0) for r in pair}
            cost=max(r['prepare_wall_union_ms']+r['pin_wall_union_ms'] for r in pair)
            if cost<=0 or not all(sizes.values()):continue
            if elapsed+cost>gap*1000 or any(bytes_used[k]+v>4*2**30 for k,v in sizes.items()):break
            elapsed+=cost;selected.append(index);eligible+=per_block_exposure[index]
            for k,v in sizes.items():bytes_used[k]+=v
        forecasts.append(dict(idle_gap_s=gap,candidate_blocks=selected,
            observed_prepare_plus_pin_cost_s=elapsed/1000,per_rank_bytes=bytes_used,
            trace_covered_exposure_upper_bound_s=duration(eligible)/1000,
            claim='Conditional opportunity, not measured idle-policy performance. Uses observed active-run preparation costs, 4 GiB/rank, no GPU copy cost or foreground startup eviction; physical validation required.'))
    output=dict(physical=True,request=1,scope='Warm actual P0 request, TE blocks 2–49 only',
        te_stage_s=(end-begin)/1e9,clock_alignment=clocks,
        per_gpu_inactive_ms={str(d):duration(v) for d,v in idle.items()},
        both_gpu_inactive_ms=duration(both),linked_weight_unavailable_both_idle_union_ms=duration(linked),
        conditional_idle_candidates=forecasts,
        ranked_blocks=[dict(block=i,exposure_ms=duration(per_block_exposure[i])) for i in pair_order],blocks=rows,
        limitations=['Only warm request 1 has complete selected Kineto coverage.',
            'Legacy queue timestamps are lower bounds; targeted queue entry/exit timestamps bracket submission, not exact worker admission.',
            'File open/close bounds surround aggregated read calls. Bytes are logical, not physical backing I/O.',
            'Whole-block OS snapshots are optional; shared cgroup/device deltas overlap. Sampled wchan is not an exact blocked duration. GIL ownership and idle-prefill counterfactual throughput remain unmeasured.',
            'Initial CUDA-event anchor uncertainty does not bound long-run drift. Cross-clock inactivity and dependency intersections use CPU-fitted Kineto only.',
            'Linked exposure is a conditional optimization opportunity; overlapping ranks are unioned. It is not a promised speedup.',
            'First correlated GEMM is a conservative known weight-consumption bound, not proof of first weight access.'])
    (folder/'te_blocks_2_49.json').write_text(json.dumps(output,separators=(',',':')))
    print(json.dumps({k:v for k,v in output.items() if k not in ('blocks','ranked_blocks')},indent=2))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    analyze(parser.parse_args().folder)
