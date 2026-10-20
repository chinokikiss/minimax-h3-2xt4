"""Reuse real request traces; expose bounded dependencies, never sum waits."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path

from analyze_causal_audit import load_trace,union,clipped,duration,IntervalIndex


def scheduler_rows(folder,requests):
    path=folder/'cpu_thread_samples.jsonl';spy=folder/'gil_stack_samples.jsonl'
    if not path.exists():return []
    capability=json.loads((folder/'cpu_os_capabilities.json').read_text())
    enabled=capability.get('sched_schedstats')=='1';groups=defaultdict(list);gil=defaultdict(list)
    for line in path.open(encoding='utf-8'):
        sample=json.loads(line)
        for row in sample['threads']:groups[(row['pid'],row['tid'])].append((sample['start_ns'],row))
    if spy.exists():
        for line in spy.open(encoding='utf-8'):
            sample=json.loads(line)
            for row in sample['traces']:
                gil[(row['pid'],row.get('os_thread_id'))].append((sample['start_ns'],row))
    output=[]
    for request in requests:
        start,end=request['start_ns'],request['end_ns']
        for (pid,tid),samples in groups.items():
            inside=[s for s in samples if start<=s[0]<=end]
            if len(inside)<2:continue
            a,b=inside[0],inside[-1];span=b[0]-a[0]
            sa,sb=a[1].get('schedstat'),b[1].get('schedstat')
            running=sb[0]-sa[0] if sa and sb and sb[0]>=sa[0] else None
            runnable=sb[1]-sa[1] if enabled and sa and sb and sb[1]>=sa[1] else None
            residual=max(0,span-running-runnable) if running is not None and runnable is not None else None
            stacks=[row for stamp,row in gil[(pid,tid)] if start<=stamp<=end]
            waiting=[row for row in stacks if not row.get('owns_gil') and any(
                'take_gil' in f['name'] for f in row.get('frames',[]))]
            output.append(dict(request=request['request'],pid=pid,tid=tid,rank=a[1]['rank'],
                covered_start_ns=a[0],covered_end_ns=b[0],running_ms=running/1e6 if running is not None else None,
                runnable_ms=runnable/1e6 if runnable is not None else None,
                sleeping_or_blocked_residual_ms=residual/1e6 if residual is not None else None,
                sampled_states={state:sum(r['state']==state for _,r in inside) for state in ('R','S','D')},
                gil_owned_samples=sum(bool(r.get('owns_gil')) for r in stacks),gil_native_wait_samples=len(waiting),
                gil_total_samples=len(stacks),gil_wait_ms=None,
                semantics='Counters cover inner sampled interval only; residual includes sleeping; native GIL waits are observations, not exact duration'))
    return output


def timeline(folder,events,kernels,requests,waits):
    origin=min(r['start_ns'] for r in requests);names=[];indices={}
    def label(name):
        if name not in indices:indices[name]=len(names);names.append(name)
        return indices[name]
    rows=[];details=[]
    selected=('cpu_future_wait','future_consumer_wait','cpu_weight_prepare','cpu_staging','cpu_pin',
        'cpu_native_te_construct','cpu_native_dit_construct','condition_wait','tensor_transfer_submit',
        'checkpoint_file_io','h2d_enqueue','activation_packet','future_producer_publish')
    for e in events:
        if e['name'] not in selected:continue
        details.append(e);rows.append([f"CPU {e['rank']} tid {e['tid']}",(e['start_ns']-origin)/1e9,
            (e['end_ns']-origin)/1e9,label(e['name']),len(details)-1])
    for e in kernels:
        rows.append([f"GPU {e['device']} rank {e['rank']} stream {e['args'].get('stream','?')}",
            (e['start_ns']-origin)/1e9,(e['end_ns']-origin)/1e9,label(e['name']),-1])
    payload=json.dumps(dict(names=names,rows=rows,details=details,waits=waits),separators=(',',':')).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>CPU/GPU dependency timeline</title>
<style>body{font:14px system-ui;background:#10151e;color:#eee}pre{white-space:pre-wrap}canvas{background:#19212e}</style>
<h2>CPU/GPU unified dependency timeline</h2><p>CPU bars show inclusive wall spans, not scheduler running time. GPU lanes show actual kernels/copies. Click CPU intervals for Future IDs, readiness bounds and thread CPU counters. Unmapped or overlapping work is not asserted to be the sole cause.</p>
Start seconds <input id="lo" type="number" value="0"> Span seconds <input id="span" type="number" value="400"><button onclick="draw()">Zoom</button>
<canvas id="c" width="1500"></canvas><pre id="detail"></pre><script>
const data=PAYLOAD,lanes=[...new Set(data.rows.map(e=>e[0]))],laneIndex=new Map(lanes.map((n,i)=>[n,i])),c=document.getElementById('c'),ctx=c.getContext('2d');c.height=lanes.length*25+25;let shown=[];
function draw(){const lo=+document.getElementById('lo').value,span=+document.getElementById('span').value;if(span<=0)return;ctx.clearRect(0,0,c.width,c.height);shown=[];lanes.forEach((n,i)=>{ctx.fillStyle='#c5d5e5';ctx.fillText(n,2,i*25+18)});for(const e of data.rows){if(e[2]<lo||e[1]>lo+span)continue;const x=260+Math.max(0,e[1]-lo)/span*1230,w=Math.max(1,(Math.min(e[2],lo+span)-Math.max(e[1],lo))/span*1230),y=laneIndex.get(e[0])*25;ctx.fillStyle=`hsl(${e[3]*73%360} 55% 50%)`;ctx.fillRect(x,y,w,18);shown.push([e,x,y,w])}}
c.onclick=ev=>{const b=c.getBoundingClientRect(),x=(ev.clientX-b.left)*c.width/b.width,y=(ev.clientY-b.top)*c.height/b.height;const a=shown.filter(a=>x>=a[1]&&x<=a[1]+a[3]&&y>=a[2]&&y<=a[2]+18).sort((a,b)=>a[3]-b[3])[0];if(a){const e=a[0];document.getElementById('detail').textContent=JSON.stringify({name:data.names[e[3]],start:e[1],end:e[2],event:e[4]>=0?data.details[e[4]]:null},null,2)}};draw();</script>'''.replace('PAYLOAD',payload)
    (folder/'cpu_gpu_timeline.html').write_text(page,encoding='utf-8')


def intersection(left,right):
    a,b=union(left),union(right);i=j=0;out=[]
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if hi>lo:out.append([lo,hi])
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out


def complement(spans,start,end):
    out=[];previous=start
    for a,b in clipped(spans,start,end):
        if a>previous:out.append([previous,a])
        previous=max(previous,b)
    if previous<end:out.append([previous,end])
    return out


def analyze(folder,stage=None,make_timeline=False):
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if stage:
        result['requests']=[dict(r,start_ns=r['stages'][stage]['start_ns'],
            end_ns=r['stages'][stage]['end_ns']) for r in result['requests']]
    events=[];kernels=[];apis=[];ops=[];meta=[]
    for rank in ('parent',0,1):
        trace,gpu,runtime=load_trace(folder,rank,include_cpu_ops=True)
        events+=trace['events'];kernels+=gpu;apis+=runtime;ops+=trace.get('cpu_operations',[])
        meta.append(dict(rank=rank,pid=trace['pid'],clock_fit=trace.get('kineto_clock_fit')))
    detailed=any(e['name']=='future_consumer_wait' for e in events)
    request_rows=[];wait_rows=[];chains=[];proven=[]
    exact_publish=defaultdict(list)
    for e in events:
        if e['name']=='future_ready_bounds':exact_publish[e['future']].append(e)
    prepare_index=defaultdict(list)
    for e in events:
        if e['name'] in ('cpu_weight_prepare','cpu_staging'):prepare_index[(e['rank'],e.get('block'))].append(e)
    api_by_tid=defaultdict(list)
    for e in apis:
        if any(n in e['name'].lower() for n in ('launch','memcpy')):api_by_tid[(e['rank'],e['tid'])].append(e)
    api_indexes={key:(sorted(rows,key=lambda e:e['start_ns'])) for key,rows in api_by_tid.items()}
    api_starts={key:[e['start_ns'] for e in rows] for key,rows in api_indexes.items()}
    for q in result['requests']:
        start,end=q['start_ns'],q['end_ns']
        cpu=[e for e in events if e['start_ns']<end and e['end_ns']>start]
        gpu=[e for e in kernels if e['start_ns']<end and e['end_ns']>start]
        idle={d:complement([(e['start_ns'],e['end_ns']) for e in gpu if e['device']==d],start,end) for d in (0,1)}
        both_idle=intersection(idle[0],idle[1])
        waits=[e for e in cpu if e['name']==('future_consumer_wait' if detailed else 'cpu_future_wait')]
        foreground=defaultdict(list)
        for wait in waits:
            block=wait.get('block') or wait.get('legacy_block')
            publications=exact_publish.get(wait.get('future'),[])
            ready=next((e for e in publications if e['upper_ns']>=wait['start_ns']
                        and e['lower_ns']<=wait['end_ns']),None)
            producers=[e for e in prepare_index[(wait['rank'],block)]
                       if e['start_ns']<wait['end_ns'] and e['end_ns']>=wait['start_ns']]
            row=dict(request=q['request'],rank=wait['rank'],consumer_tid=wait['tid'],future=wait.get('future'),
                block=block,start_ns=wait['start_ns'],end_ns=wait['end_ns'],ready_on_entry=wait.get('ready_on_entry'),
                producer_publish=ready,preparation_candidates=producers,
                readiness='bounded by exact Future identity publication' if ready else 'unobserved; block/time candidates only')
            if ready:
                row['post_ready_return_gap_bounds_ms']=[max(0,wait['end_ns']-ready['upper_ns'])/1e6,
                                                       max(0,wait['end_ns']-ready['lower_ns'])/1e6]
            key=(wait['rank'],wait['tid']);position=bisect_left(api_starts.get(key,[]),wait['end_ns'])
            next_api=api_indexes.get(key,[])[position] if position<len(api_indexes.get(key,[])) else None
            pid=next(m['pid'] for m in meta if m['rank']==wait['rank'])
            # PID==TID identifies the Linux rank's foreground dispatch thread.
            # No dependence on an unrelated preparation thread's overlap.
            if wait['tid']==pid and wait.get('ready_on_entry') is False and next_api and next_api['start_ns']<end:
                exposed=intersection([(wait['start_ns'],wait['end_ns'])],both_idle)
                category='TE startup future' if block and 'te_startup' in block else 'foreground block future'
                row.update(next_same_thread_submission=next_api,all_gpu_idle_intersection_ms=duration(exposed),
                    proof='Foreground Future.result is a control-flow predecessor of the next submission on that thread; all-device inactivity intersection is conservative exposure, not a complete global DAG')
                if exposed:foreground[category]+=exposed;proven.append(row)
            wait_rows.append(row)
        threads=[]
        groups=defaultdict(list)
        for e in cpu:groups[(e['rank'],e['tid'])].append(e)
        for (rank,tid),group in groups.items():
            thread_waits=[e for e in group if e['name'] in ('future_consumer_wait','condition_wait','cuda_event/synchronize')
                          or (not detailed and (e['name']=='cpu_future_wait' or e['name'].startswith('sync/')))]
            counters=[(e['thread_cpu_start_ns'],e['thread_cpu_end_ns']) for e in group
                      if 'thread_cpu_start_ns' in e and start<=e['start_ns'] and e['end_ns']<=end]
            threads.append(dict(rank=rank,tid=tid,observed_wait_wall_union_ms=duration(clipped(
                [(e['start_ns'],e['end_ns']) for e in thread_waits],start,end)),
                instrumented_running_cpu_union_ms=duration(counters) if counters else None,
                runnable_ms=None,blocked_ms=None,gil_wait_ms=None,
                semantics='Thread CPU counters cover instrumented spans; OS running/runnable/blocked requires scheduler evidence, GIL ownership samples do not give GIL wait duration'))
        # Tensor matching is unavailable in the old captures: these are gaps,
        # not proven transfers of the same tensor or a causal D2H/H2D chain.
        for d in (0,1):
            copies=sorted((e for e in gpu if e['device']==d and e['category']=='memcpy'),key=lambda e:e['start_ns'])
            for before,after in zip(copies,copies[1:]):
                if 'dtoh' in before['name'].lower() and 'htod' in after['name'].lower() and after['start_ns']-before['end_ns']>=5_000_000:
                    activity=[e for e in cpu if e['start_ns']<after['start_ns'] and e['end_ns']>before['end_ns']
                              and (e['rank']==d or e['rank']=='parent') and e['name'] in
                              ('cpu_weight_prepare','cpu_native_te_construct','activation_packet','host_cache_take','cpu_pin')]
                    chains.append(dict(request=q['request'],device=d,start_ns=before['end_ns'],end_ns=after['start_ns'],
                        gap_ms=(after['start_ns']-before['end_ns'])/1e6,before=before['name'],after=after['name'],
                        cpu_overlaps=activity,proven_same_tensor=False))
        phase={}
        for name in ('cpu_weight_prepare','checkpoint_read/read_tensor','checkpoint_read/read','cpu_pin',
                     'cpu_native_te_construct','cpu_rank_deserialize','cpu_offline_deserialize','cpu_state_assignment'):
            rows=[e for e in cpu if e['name']==name]
            phase[name]=dict(wall_union_ms=duration(clipped([(e['start_ns'],e['end_ns']) for e in rows],start,end)),count=len(rows))
        ranked={name:duration(clipped(rows,start,end)) for name,rows in foreground.items()}
        combined=union(span for rows in foreground.values() for span in rows)
        request_rows.append(dict(request=q['request'],start_ns=start,end_ns=end,latency_ms=q['latency_ms'],
            all_gpu_inactive_union_ms=duration(both_idle),foreground_future_exposed_union_ms=duration(combined),
            per_cause_exposure_ms=ranked,threads=threads,phase_wall_unions=phase))
    report=dict(schema=1,case=result['case'],detailed=detailed,requests=request_rows,future_waits=wait_rows,
        proven_foreground_waits=proven,d2h_cpu_h2d_candidates=chains,
        missing_evidence=['Original traces have no scheduler state or GIL waiting duration',
            'Old Future identities/ready times are absent; only producer candidates can be reused',
            'All-device inactivity is a conservative intersection, not the entire end-to-end critical path',
            'CPU span wall time includes blocking and runnable delay; not useful CPU execution time'])
    report['os_thread_metrics']=scheduler_rows(folder,result['requests'])
    event_records=defaultdict(list)
    for e in events:
        if e['name'] in ('cuda_event/record','python_event/set','cuda_event_query'):
            event_records[(e.get('event'),e.get('generation',0))].append(e)
    report['event_waits']=[]
    for e in events:
        if e['name'] not in ('cuda_event_consumer_wait','cuda_event/synchronize','cuda_event/wait','python_event/wait'):continue
        producer=[p for p in event_records[(e.get('event'),e.get('generation',0))]
                  if p['start_ns']<=e['end_ns']]
        report['event_waits'].append(dict(consumer=e,producer_records=producer,
            exact_readiness_ns=None,semantics='Producer identity/generation and observed query bounds; stream enqueue duration is not GPU dependency duration'))
    (folder/'cpu_path_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['analysis_scope']=stage or 'full_request'
    if make_timeline:timeline(folder,events,kernels,result['requests'],wait_rows)
    print('CPU PATH',result['case'],'requests',len(request_rows),'waits',len(wait_rows),
          'conservative foreground exposure ms',[round(r['foreground_future_exposed_union_ms'],2) for r in request_rows],flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    parser.add_argument('--stage');parser.add_argument('--timeline',action='store_true')
    args=parser.parse_args();analyze(args.folder,args.stage,args.timeline)


if __name__=='__main__':main()
