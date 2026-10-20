"""Compact OS attribution and a readable timeline from selected physical traces."""
import argparse
from collections import Counter
import csv
import html
import hashlib
import json
from pathlib import Path
import statistics


def summarize(folder):
    data=json.loads((folder/'te_blocks_2_49.json').read_text());rows=data['blocks'];ranked={}
    compact=[]
    for rank in (0,1):
        subset=[r for r in rows if r['rank']==rank];states=Counter();wchans=Counter()
        totals=Counter();memory=[];queue=[]
        for row in subset:
            observations=row['os_observations']
            assert len(observations)==1,'Exactly one whole-block producer required'
            observed=observations[0];c=observed['counters'];sched=c['schedstat']
            wall=(observed['end_ns']-observed['start_ns'])/1e9
            running,runnable=(sched[i]/1e9 for i in (0,1))
            # Snapshot reads surround the event, so residuals include bounded
            # observer uncertainty; these are not exact scheduler state traces.
            blocked=max(0.,wall-running-runnable)
            states.update(observed['sampled_waits']['states']);wchans.update(observed['sampled_waits']['wchans'])
            totals.update(wall_s=wall,running_s=running,runnable_s=runnable,
                blocked_residual_s=blocked,storage_read_bytes=c['io']['read_bytes'],
                logical_read_bytes=row['logical_read_bytes'],minor_faults=c['usage']['ru_minflt'],
                major_faults=c['usage']['ru_majflt'],observer_boundary_s=observed['observer_outside_prepare_ns']/1e9)
            memory.extend(int(observed[k]) for k in ('memory_current_before','memory_current_after'))
            bounds=[e for e in row['queue_submission_bounds'] if e['name']=='cache_queue_bounds']
            q=None
            if bounds:
                e=min(bounds,key=lambda v:v['lower_ns'])
                q=[max(0,(observed['start_ns']-e[k])/1e6) for k in ('upper_ns','lower_ns')]
                queue.append(q)
            compact.append(dict(rank=rank,block=row['block'],producer_wall_s=wall,
                running_s=running,runnable_s=runnable,blocked_residual_s=blocked,
                storage_read_bytes=c['io']['read_bytes'],logical_read_bytes=row['logical_read_bytes'],
                major_faults=c['usage']['ru_majflt'],minor_faults=c['usage']['ru_minflt'],
                queue_lower_ms=q[0] if q else None,queue_upper_ms=q[1] if q else None,
                exposed_both_idle_ms=row['linked_both_gpu_inactive_ms']))
        ranked[str(rank)]=dict(totals,states=dict(states),wchans=dict(wchans),
            d_state_sample_fraction=states['D']/sum(states.values()),
            min_observed_cgroup_bytes=min(memory),max_observed_cgroup_bytes=max(memory),
            queue_median_bounds_ms=[statistics.median(q[i] for q in queue) for i in (0,1)],
            read_wall_s=sum(r['read_call_wall_ms'] for r in subset)/1000,
            read_cpu_s=sum(r['read_thread_cpu_ms'] for r in subset)/1000,
            pin_wall_s=sum(r['pin_wall_union_ms'] for r in subset)/1000)
    result=dict(physical=True,request=1,ranks=ranked,
        actual_union_exposed_s=data['linked_weight_unavailable_both_idle_union_ms']/1000,
        method='Same producer thread, disjoint block intervals summed per rank only. Shared cgroup/device deltas never summed. D/wchan are samples, not exact durations. Running/runnable residual includes snapshot-boundary uncertainty; GIL ownership unmeasured.')
    (folder/'te_block_os_summary.json').write_text(json.dumps(result,indent=2))
    with (folder/'te_block_os.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    # All coordinates are request-relative seconds; full raw traces remain the
    # source for stream flow and CPU/GPU correlation inspection.
    width=1500;height=380;left=165;scale=(width-left-25)/data['te_stage_s'];shapes=[]
    def rect(start,end,lane,color,title):
        shapes.append(f'<rect x="{left+start*scale:.2f}" y="{40+lane*32}" width="{max(.6,(end-start)*scale):.2f}" height="20" fill="{color}"><title>{html.escape(title)}</title></rect>')
    labels=['R0 producer read','R0 first weight GEMM','R0 foreground wait','R0 H2D memcpy','R1 producer read','R1 first weight GEMM','R1 foreground wait','R1 H2D memcpy','Both inactive: weight wait']
    for lane,label in enumerate(labels):shapes.append(f'<text x="8" y="{55+lane*32}">{label}</text>')
    for tick in range(0,int(data['te_stage_s'])+1,5):
        x=left+tick*scale
        shapes.append(f'<path d="M{x:.2f} 30V330" stroke="#ddd"/><text x="{x:.2f}" y="22">{tick}s</text>')
    for row in rows:
        base=row['rank']*4;block=row['block']
        for interval in row['file_intervals']:rect(interval['start_s'],interval['end_s'],base,'#457b9d',f'Rank {row["rank"]} block {block}: read {interval["bytes"]} logical bytes')
        for interval in row['links']:rect(*interval['wait_s'],base+2,'#e9c46a',f'Block {block}, Future {interval["future"]}: foreground wait')
        for start,end in row.get('kineto_memcpy_intervals_s',[]):rect(start,end,base+3,'#2a9d8f',f'Block {block}: correlated Kineto memcpy')
        gemm=row['first_correlated_weight_gemm_s']
        if gemm is not None:rect(gemm,gemm+.01,base+1,'#7b2cbf',f'Block {block}: first correlated GEMM, not necessarily first weight access')
        for start,end in row['linked_both_idle_intervals_s']:rect(start,end,8,'#e76f51',f'Block {block}: verified Future dependency and both GPUs inactive; ranks unioned in totals')
        # Pin positions are not saved separately in compact rows. They remain
        # available in the causal archive; do not invent timeline locations.
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(row[k]) if not isinstance(row[k],float) else f"{row[k]:.3f}")}</td>' for k in compact[0])+'</tr>' for row in compact)
    page='''<!doctype html><meta charset="utf-8"><title>Physical TE causal profile</title>
<style>body{font:14px system-ui;margin:24px;color:#203040}svg{font:12px system-ui}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}p{max-width:1100px}</style>
<h1>Physical dual-T4: warm TE blocks 2–49</h1>
<p>Request-relative time, CPU-fitted Kineto clocks throughout. Hover over bars for block and dependency details. Foreground waits are not automatically GPU stalls. Orange exposure intersects verified weight Futures and inactivity on both GPUs; duplicated rank intervals are unioned in the reported total. Pinning positions omitted from compact rows; full archives retain them. CUDA-event absolute timestamps are excluded because initial calibration does not bound long-run drift.</p>
'''+f'<p>Reference+TE: {data["te_stage_s"]:.3f}s. Union of linked, both-GPU inactive intervals: {result["actual_union_exposed_s"]:.3f}s. This is not a promised speedup.</p><div class="scroll"><svg width="{width}" height="{height}">'+''.join(shapes)+'</svg></div><h2>Per-block OS evidence</h2><p>Running/runnable from Linux schedstat; residual is an estimate, not a traced scheduler state. Storage bytes are producer-thread kernel accounting. Shared cgroup/device deltas are not assigned exclusively to TE.</p><div class="scroll"><table><thead><tr>'+''.join('<th>'+k+'</th>' for k in compact[0])+'</tr></thead><tbody>'+table+'</tbody></table></div>'
    (folder/'te_causal_timeline.html').write_text(page,encoding='utf-8')
    root=Path(__file__).resolve().parent
    receipt=dict(analysis_source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest()
        for n in ('analyze_te_causal_profile.py','analyze_causal_audit.py','analyze_cpu_path.py','summarize_te_block_os.py')},
        artifacts_sha256={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in
            ('result.json','trace_archive.json','te_blocks_2_49.json','te_block_os_summary.json','te_block_os.csv','te_causal_timeline.html')})
    (folder/'te_analysis_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);summarize(parser.parse_args().folder)
