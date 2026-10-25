"""Interval-union analysis and dependency evidence; never sum overlapping kernels.

The HTML renders CPU causality and per-process GPU lanes on a common clock.
Raw Kineto traces are retained for operator/launch/stream-flow inspection.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import statistics
import math
import gzip


def read_json(path):
    if path.exists():return json.loads(path.read_text())
    with gzip.open(path.with_suffix(path.suffix+'.gz'),'rt',encoding='utf-8') as stream:
        return json.load(stream)


class IntervalIndex:
    """Exact overlapping-span lookup, including long/nested CPU ranges."""
    def __init__(self, events):
        self.events=sorted(events,key=lambda e:e['start_ns'])
        self.starts=[e['start_ns'] for e in self.events]
        self.max_ends=[];latest=-math.inf
        for event in self.events:
            latest=max(latest,event['end_ns']);self.max_ends.append(latest)

    def overlapping(self,start,end):
        lo=bisect_right(self.max_ends,start)
        hi=bisect_left(self.starts,end)
        return [e for e in self.events[lo:hi] if e['end_ns']>start]


def union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<=start:continue
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return merged


def clipped(intervals,start,end):
    return union((max(a,start),min(b,end)) for a,b in intervals if a<end and b>start)


def duration(intervals):
    return sum(b-a for a,b in union(intervals))/1e6


def overlap(left,right):
    a,b=union(left),union(right);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total/1e6


def label(event):
    out=('causal/r%s/req=%s/stage=%s/cmd=%s/nfe=%s/%s' %
         (event['rank'],event['request'],event['stage'],event['command'],event['nfe'],event['name']))
    if event.get('block') is not None:out+='/block='+str(event['block'])
    return out


def category(name,cat):
    lower=name.lower()
    if 'memcpy' in cat or 'memcpy' in lower:return 'memcpy'
    if 'memset' in cat or 'memset' in lower:return 'memset'
    if 'nccl' in lower:return 'nccl'
    if any(x in lower for x in ('attention','fmha','flash','sdpa')):return 'attention'
    if any(x in lower for x in ('gemm','matmul','cutlass','cublas','mma')):return 'gemm'
    if any(x in lower for x in ('quant','convrot','rotate')):return 'quantization'
    if any(x in lower for x in ('transpose','permute','copy_kernel','contiguous')):return 'layout'
    return 'other_kernel'


def load_trace(folder,rank,include_cpu_ops=False):
    causal=read_json(folder/f'batch_r{rank}_causal.json')
    events=causal['events'];kernels=[];runtime=[]
    path=folder/f'batch_r{rank}_kineto.json'
    if path.exists() or path.with_suffix(path.suffix+'.gz').exists():
        raw=read_json(path)['traceEvents']
        custom=defaultdict(list);kineto=defaultdict(list)
        for event in events:custom[label(event)].append(event['start_ns'])
        for event in raw:
            if (event.get('ph')=='X' and event.get('cat')=='user_annotation'
                and event.get('name','').startswith('causal/')):
                kineto[event['name']].append(event['ts'])
        offsets=[]
        for name,starts in custom.items():
            traces=kineto.get(name,[])
            if len(starts)==len(traces):
                offsets.extend(a/1000-b for a,b in zip(sorted(starts),sorted(traces)))
        if not offsets:raise ValueError('No CPU correlation anchors: '+str(path))
        offset=statistics.median(offsets)
        causal['kineto_clock_fit']=dict(offset_us=offset,anchors=len(offsets),
            median_abs_residual_us=statistics.median(abs(x-offset) for x in offsets),
            p95_abs_residual_us=sorted(abs(x-offset) for x in offsets)[int(.95*(len(offsets)-1))])
        for event in raw:
            if event.get('ph')!='X':continue
            cat=event.get('cat','').lower()
            row=dict(name=event['name'],rank=rank,
                start_ns=int((event['ts']+offset)*1000),
                end_ns=int((event['ts']+offset+event.get('dur',0))*1000),
                args=event.get('args',{}),tid=event.get('tid'),category=category(event['name'],cat))
            if 'kernel' in cat or 'gpu_memcpy' in cat or 'gpu_memset' in cat:
                row['device']=row['args'].get('device',0 if rank=='parent' else rank)
                kernels.append(row)
            elif 'cuda_runtime' in cat or 'cuda_driver' in cat:runtime.append(row)
            elif include_cpu_ops and cat=='cpu_op':
                causal.setdefault('cpu_operations',[]).append(row)
        launch_by_correlation={e['args'].get('correlation'):e for e in runtime
                               if 'launch' in e['name'].lower() and e['args'].get('correlation') is not None}
        owner_groups=defaultdict(list)
        for event in events:
            if event['name'] in ('block_compute_enqueue','reference_encode','rank_dispatch/video_pixels','rank_dispatch/audio_channels'):
                owner_groups[(event['tid'],event['name'])].append(event)
        owner_indexes=defaultdict(list)
        for (tid,name),group in owner_groups.items():
            group.sort(key=lambda e:e['start_ns'])
            owner_indexes[tid].append(([e['start_ns'] for e in group],group))
        cpu_ops={e.get('args',{}).get('External id'):e['name'] for e in raw
                 if e.get('ph')=='X' and e.get('cat')=='cpu_op' and e.get('args',{}).get('External id') is not None}
        for kernel in kernels:
            kernel['operator']=cpu_ops.get(kernel['args'].get('External id'))
            launch=launch_by_correlation.get(kernel['args'].get('correlation'))
            if launch:
                owners=[]
                for starts,group in owner_indexes.get(launch['tid'],[]):
                    position=bisect_right(starts,launch['start_ns'])-1
                    if position>=0 and group[position]['end_ns']>=launch['start_ns']:
                        owners.append(group[position])
                owner=min(owners,key=lambda e:e['end_ns']-e['start_ns'],default=None)
                kernel['launch']=dict(start_ns=launch['start_ns'],end_ns=launch['end_ns'],
                                      tid=launch['tid'],owner=owner)
        del raw
    return causal,kernels,runtime


def analyze(folder):
    result=json.loads((folder/'result.json').read_text())
    if not result['passed']:raise ValueError('Cannot accept a failed run')
    if result['capture']=='timing':return None
    traces=[];gpu=[];apis=[];cpu=[];event_gpu=[]
    for rank in ('parent',0,1):
        causal,kernels,runtime=load_trace(folder,rank)
        traces.append({k:v for k,v in causal.items() if k not in ('events','gpu_intervals')})
        cpu+=causal['events'];gpu+=kernels;apis+=runtime;event_gpu+=causal['gpu_intervals']
    rows=[];stalls=[];loads=[]
    clock_error_ns=max([t.get('alignment_uncertainty_ns',0) for t in traces]+
                      [t.get('kineto_clock_fit',{}).get('p95_abs_residual_us',0)*1000 for t in traces])
    idle_threshold_ns=max(5_000_000,int(clock_error_ns*4))
    # Merge compute once, rather than scanning/sorting every kernel per copy.
    compute_indexes={rank:IntervalIndex(dict(start_ns=a,end_ns=b) for a,b in union(
        (e['start_ns'],e['end_ns']) for e in gpu if e['rank']==rank
        and e['category'] not in ('memcpy','memset','nccl'))) for rank in ('parent',0,1)}
    waits_by_load=defaultdict(list)
    for event in event_gpu:
        if event['name']=='gpu_dependency_wait_upper_bound':
            waits_by_load[(event['rank'],event.get('load'))].append(event)
    for transfer in (e for e in event_gpu if e['name']=='h2d'):
        rank=transfer['rank'];start,end=transfer['start_ns'],transfer['end_ns']
        compute=[(e['start_ns'],e['end_ns']) for e in compute_indexes[rank].overlapping(start,end)]
        waits=waits_by_load[(rank,transfer.get('load'))]
        loads.append(dict(**transfer,compute_overlap_ms=overlap([(start,end)],compute),
            elapsed_ms=(end-start)/1e6,consumer_waits=waits,
            conclusion='overlap is measured; uncovered transfer time is not automatically on the request critical path'))
    for request in result['requests']:
        start,end=request['start_ns'],request['end_ns']
        rcpu=[e for e in cpu if e['start_ns']<end and e['end_ns']>start]
        rgpu=[e for e in gpu if e['start_ns']<end and e['end_ns']>start]
        ranks={}
        for rank in ('parent',0,1):
            subset=[e for e in rgpu if e['rank']==rank]
            bycat=defaultdict(list);names=Counter();operators=defaultdict(list)
            rank_apis=[e for e in apis if e['rank']==rank and e['start_ns']<end and e['end_ns']>start]
            rank_cpu=[e for e in rcpu if e['rank']==rank]
            for e in subset:
                bycat[e['category']].append((e['start_ns'],e['end_ns']))
                names[e['name']]+=max(0,min(e['end_ns'],end)-max(e['start_ns'],start))/1e6
                operators[e.get('operator') or 'unmapped_operator'].append((e['start_ns'],e['end_ns']))
            stage_stats={}
            for name,stage in request['stages'].items():
                stage_stats[name]={cat:duration(clipped(spans,stage['start_ns'],stage['end_ns']))
                                  for cat,spans in bycat.items()}
            ranks[str(rank)]=dict(kernel_count=sum(e['category'] not in ('memcpy','memset') for e in subset),
                cuda_api_counts=dict(Counter(e['name'] for e in rank_apis)),
                cpu_launch_union_ms=duration(clipped([(e['start_ns'],e['end_ns']) for e in rank_apis
                    if 'launch' in e['name'].lower()],start,end)),
                cuda_sync_host_union_ms=duration(clipped([(e['start_ns'],e['end_ns']) for e in rank_apis
                    if 'synchronize' in e['name'].lower()],start,end)),
                cpu_activity_union_ms={name:duration(clipped([(e['start_ns'],e['end_ns']) for e in rank_cpu
                    if e['name']==name],start,end)) for name in (
                        'cpu_weight_prepare','cpu_pin','cpu_staging','cpu_future_wait','gpu_memory_admission','activation_packet')},
                category_busy_union_ms={cat:duration(clipped(spans,start,end)) for cat,spans in bycat.items()},
                stage_category_busy_union_ms=stage_stats,
                operator_busy_union_ms={name:duration(clipped(intervals,start,end)) for name,intervals in operators.items()},
                dominant_kernel_inclusive_ms=names.most_common(20))
        devices={}
        for device in (0,1):
            actual=[e for e in rgpu if e['device']==device]
            by_end=sorted(actual,key=lambda e:e['end_ns']);ends=[e['end_ns'] for e in by_end]
            by_start=sorted(actual,key=lambda e:e['start_ns']);starts=[e['start_ns'] for e in by_start]
            blocker_index=IntervalIndex(e for e in rcpu if (e['rank']=='parent' or e['rank']==device)
                and (e['name']=='cpu_future_wait' or e['name'].startswith('sync/')
                     or e['name'] in ('cpu_weight_prepare','cpu_staging','gpu_memory_admission','service_call/te','reference_encode')))
            dependency_index=IntervalIndex(e for e in event_gpu
                if e['name']=='gpu_dependency_wait_upper_bound' and e['rank']==device)
            busy=clipped([(e['start_ns'],e['end_ns']) for e in actual],start,end)
            compute=clipped([(e['start_ns'],e['end_ns']) for e in actual
                if e['category'] not in ('memcpy','memset','nccl')],start,end)
            transfers=clipped([(e['start_ns'],e['end_ns']) for e in actual if e['category']=='memcpy'],start,end)
            nccl=clipped([(e['start_ns'],e['end_ns']) for e in actual if e['category']=='nccl'],start,end)
            previous=start
            for begin,finish in [*busy,[end,end]]:
                if begin-previous>=idle_threshold_ns:
                    candidates=blocker_index.overlapping(previous,begin)
                    candidates.sort(key=lambda e:min(e['end_ns'],begin)-max(e['start_ns'],previous),reverse=True)
                    dependency=dependency_index.overlapping(previous,begin)
                    before=bisect_right(ends,previous)-1;after=bisect_left(starts,begin)
                    preceding=by_end[before] if before>=0 else None
                    following=by_start[after] if after<len(by_start) else None
                    confirmed=[]
                    owner=following.get('launch',{}).get('owner') if following else None
                    if owner and owner.get('block'):
                        for candidate in candidates:
                            if (candidate['name']=='cpu_future_wait' and not candidate.get('ready_on_entry',True)
                                and candidate.get('block')==owner['block']
                                and candidate['end_ns']<=following['launch']['start_ns']-clock_error_ns):
                                confirmed.append(dict(block=owner['block'],dependency=candidate,
                                    launch=following['launch'],exposed_idle_intersection_ms=
                                        max(0,min(candidate['end_ns'],begin)-max(candidate['start_ns'],previous))/1e6,
                                    evidence='Exact block future prevents its correlated compute launch until resolution; intersected with physical GPU inactivity'))
                    stalls.append(dict(request=request['request'],device=device,start_ns=previous,end_ns=begin,
                        idle_ms=(begin-previous)/1e6,preceding=preceding,following=following,
                        dependencies=dependency,cpu_candidates=candidates[:6],confirmed_loading_dependencies=confirmed,
                        conclusion='confirmed GPU inactivity; linked wait intervals shown; contention/sole cause requires dependency inspection'))
                previous=max(previous,finish)
            devices[str(device)]=dict(busy_union_ms=duration(busy),
                activity_fraction=duration(busy)/request['latency_ms'],
                compute_kernel_union_ms=duration(compute),memcpy_union_ms=duration(transfers),
                nccl_kernel_union_ms=duration(nccl),
                memcpy_compute_overlap_ms=overlap(transfers,compute),
                memcpy_nccl_overlap_ms=overlap(transfers,nccl),
                scope='all captured parent and worker contexts on this physical GPU; activity, not SM occupancy')
        hits=[e for e in rcpu if e['name']=='gpu_cache_lookup']
        host_hits=[e for e in rcpu if e['name']=='host_cache_outcome']
        waits=[(e['start_ns'],e['end_ns']) for e in rcpu if e['name']=='cpu_future_wait']
        gpuwait=[(e['start_ns'],e['end_ns']) for e in event_gpu
                 if e['name']=='gpu_dependency_wait_upper_bound' and e['start_ns']<end and e['end_ns']>start]
        samples=[s for s in result['host_monitor']['samples'] if start<=s['monotonic_ns']<=end]
        memory=dict(peak_process_tree_rss=max((s['process_tree_rss'] for s in samples),default=None),
            min_system_available=min((s['available'] for s in samples),default=None),devices={})
        for device in (0,1):
            values=[s['devices'][device] for s in samples if len(s['devices'])>device]
            estimates={}
            for field in ('rx_bytes_s','tx_bytes_s'):
                pairs=[(a,b) for a,b in zip(samples,samples[1:]) if len(a['devices'])>device and len(b['devices'])>device
                    and a['devices'][device][field] is not None and b['devices'][device][field] is not None]
                estimates[field.replace('_bytes_s','_estimated_bytes')]=sum(
                    (b['monotonic_ns']-a['monotonic_ns'])/1e9*(a['devices'][device][field]+b['devices'][device][field])/2
                    for a,b in pairs) if pairs else None
            memory['devices'][str(device)]=dict(peak_driver_bytes=max((v['used'] for v in values),default=None),
                sampled_mean_util_percent=statistics.mean(v['util_percent'] for v in values if v['util_percent'] is not None)
                    if any(v['util_percent'] is not None for v in values) else None,
                **estimates)
        rows.append(dict(request=request['request'],latency_ms=request['latency_ms'],nfe=request['nfe'],memory=memory,
            handoff_gap_ms=request['handoff_gap_ms'],stages=request['stages'],ranks=ranks,devices=devices,
            cpu_future_wait_union_ms=duration(clipped(waits,start,end)),
            gpu_dependency_wait_union_upper_bound_ms=duration(clipped(gpuwait,start,end)),
            gpu_cache=dict(hits=sum(bool(e['hit']) for e in hits),misses=sum(not e['hit'] for e in hits),
                           semantics='entry present before use/prefetch; readiness waits counted separately'),
            host_cache=dict(hits=sum(bool(e['hit']) for e in host_hits),misses=sum(bool(e['miss']) for e in host_hits)),
            logical_checkpoint_read_bytes=sum(e['bytes'] for e in rcpu if e['name']=='checkpoint_read_bytes'),
            weight_h2d_payload_bytes=sum(e['bytes'] for e in event_gpu if e['name']=='h2d' and start<=e['start_ns']<end)))
    stalls.sort(key=lambda e:e['idle_ms'],reverse=True)
    collectives=defaultdict(dict)
    for event in cpu:
        if event['name'].startswith('nccl/') and event['rank'] in (0,1):
            collectives[(event['request'],event['command'],event['collective'])][event['rank']]=event
    peers=[]
    for (request_id,command_id,sequence),pair in collectives.items():
        if len(pair)!=2:continue
        peers.append(dict(request=request_id,command=command_id,collective=sequence,
            operation=pair[0]['name'],same_operation=pair[0]['name']==pair[1]['name'],
            rank0_start_ns=pair[0]['start_ns'],rank1_start_ns=pair[1]['start_ns'],
            host_entry_skew_ms=abs(pair[0]['start_ns']-pair[1]['start_ns'])/1e6,
            scope='host call entry skew; not by itself measured NCCL wait or wire time'))
    report=dict(schema=1,passed=True,case=result['case'],capture=result['capture'],clock_fits=traces,
        analysis_source_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
        requests=rows,significant_stalls=stalls[:60],block_loading=loads,
        clock_error_bound_ns=clock_error_ns,idle_threshold_ns=idle_threshold_ns,
        collective_calls=[e for e in cpu if e['name'].startswith('nccl/')],
        correlated_collectives=peers,
        cautions=['Category unions can overlap each other; do not sum them.',
            'Kernel-name classification is heuristic; dominant names and raw traces are authoritative.',
            'CPU blocked time can overlap GPU work; it is not automatically exposed request latency.',
            'GPU event wait includes host enqueue delay; inspect Kineto gaps before attributing exposure.',
            'Overlap with NCCL proves concurrency, not performance interference.',
            'Cold means runtime-cache cold; physical storage coldness is unverified.'])
    (folder/'causal_analysis.json').write_text(json.dumps(report,indent=2))
    origin=min(e['start_ns'] for e in cpu)
    lanes=[]
    for e in cpu:
        if e['end_ns']<=e['start_ns']:continue
        lanes.append(dict(lane=f"CPU r{e['rank']} thread {e['tid']}",start=(e['start_ns']-origin)/1e9,
            end=(e['end_ns']-origin)/1e9,name=e['name'],detail=e))
    for e in gpu:
        detail={k:e[k] for k in ('name','rank','device','start_ns','end_ns','category','operator') if k in e}
        detail['args']={k:e['args'][k] for k in ('stream','correlation','External id','bytes') if k in e['args']}
        if e.get('launch'):
            launch=e['launch'];owner=launch.get('owner')
            detail['launch']={k:launch[k] for k in ('start_ns','end_ns','tid')}
            detail['launch']['owner']={k:owner[k] for k in ('name','block','request','stage','command','nfe') if k in owner} if owner else None
        lanes.append(dict(lane=f"GPU {e['device']} r{e['rank']} stream {e['args'].get('stream','?')}",
            start=(e['start_ns']-origin)/1e9,end=(e['end_ns']-origin)/1e9,name=e['category'],detail=detail))
    # A self-contained, zoomable canvas; no external CDN or remote data.
    payload=json.dumps(lanes,separators=(',',':')).replace('</','<\/')
    page='''<!doctype html><meta charset="utf-8"><title>Causal pipeline timeline</title>
<style>body{font:14px system-ui;background:#10151e;color:#eee;margin:16px}canvas{background:#19212e}input{width:130px}pre{white-space:pre-wrap}button{margin:8px}</style>
<h2>Actual request timeline</h2><p>Common host monotonic clock. CPU spans are nested; GPU lanes show actual kernels/memcpy. Click an interval for IDs and dependencies. Use raw Kineto traces for launch flows.</p>
<label>Start (s) <input id="start" type="number" value="0" step="1"></label>
<label>Span (s) <input id="span" type="number" step="1"></label><button onclick="draw()">Render / zoom</button>
<canvas id="view" width="1500"></canvas><pre id="detail"></pre><script>
const events=PAYLOAD, lanes=[...new Set(events.map(e=>e.lane))],c=document.getElementById('view'),ctx=c.getContext('2d');
c.height=lanes.length*30+30;document.getElementById('span').value=Math.ceil(events.reduce((a,e)=>Math.max(a,e.end),0));let shown=[];
function draw(){let lo=+document.getElementById('start').value,span=+document.getElementById('span').value;if(!(span>0))return;ctx.clearRect(0,0,c.width,c.height);shown=[];
lanes.forEach((l,i)=>{ctx.fillStyle='#bfc9d8';ctx.fillText(l,5,i*30+20)});
for(let e of events){if(e.end<lo||e.start>lo+span)continue;let y=lanes.indexOf(e.lane)*30+5,x=260+Math.max(0,e.start-lo)/span*1220,w=Math.max(1,(Math.min(e.end,lo+span)-Math.max(e.start,lo))/span*1220);let n=[...e.name].reduce((a,v)=>a+v.charCodeAt(0),0);ctx.fillStyle=`hsl(${n%360} 55% 50%)`;ctx.fillRect(x,y,w,20);shown.push({e,x,y,w});}}
c.onclick=function(ev){let r=c.getBoundingClientRect(),x=(ev.clientX-r.left)*c.width/r.width,y=(ev.clientY-r.top)*c.height/r.height;let item=shown.filter(v=>x>=v.x&&x<=v.x+v.w&&y>=v.y&&y<=v.y+20).sort((a,b)=>a.w-b.w)[0];if(item)document.getElementById('detail').textContent=JSON.stringify(item.e.detail,null,2)};draw();</script>'''.replace('PAYLOAD',payload)
    (folder/'timeline.html').write_text(page,encoding='utf-8')
    lines=[f"# {result['case']} causal diagnostic",'',
        'Diagnostic latency includes profiler perturbation. Compare with the matched timing pass before accepting latency conclusions.','',
        'Confirmed inactivity intervals below combine all captured contexts on each physical GPU. Dependency candidates are evidence, not proof of a sole cause.','']
    for stall in stalls[:12]:
        a=(stall['start_ns']-origin)/1e9;b=(stall['end_ns']-origin)/1e9
        dependencies=', '.join(e['name']+': '+str(e.get('block','')) for e in stall['cpu_candidates'][:3]) or 'no mapped CPU blocker'
        confirmed=stall.get('confirmed_loading_dependencies',[])
        certainty=('Confirmed block-future launch dependency, exposed idle intersection '
            +str(round(sum(e['exposed_idle_intersection_ms'] for e in confirmed),2))+'ms.'
            if confirmed else 'Sole blocker remains a hypothesis.')
        lines.append(f"- Request {stall['request']}, GPU {stall['device']}, timeline {a:.6f}–{b:.6f}s: **{stall['idle_ms']:.2f}ms inactive**. {certainty} Overlapping dependencies: {dependencies}. Evidence: `causal_analysis.json/significant_stalls` and raw rank traces.")
    lines+=['','H2D/NCCL interference, allocator pressure and CPU starvation require linked launch/dependency evidence and the matched timing pass; concurrency alone does not establish them.','',*('- '+c for c in report['cautions'])]
    (folder/'ROOT_CAUSES.md').write_text('\n'.join(lines),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    summaries=[]
    for result_path in args.root.rglob('result.json'):
        result=json.loads(result_path.read_text())
        if 'capture' not in result:continue
        if not result.get('passed'):continue
        analysis_path=result_path.parent/'causal_analysis.json'
        source_sha=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()
        cached=json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
        if result['capture']!='timing' and cached.get('analysis_source_sha256')!=source_sha:
            analyze(result_path.parent)
        summaries.append(dict(case=result['case'],capture=result['capture'],
            requests=len(result['requests']),warmup=len(result['warmup']),
            latency_ms=[r['latency_ms'] for r in result['requests']],path=str(result_path.parent)))
    comparisons=[]
    for profile in summaries:
        if profile['capture']!='kineto':continue
        match=next((s for s in summaries if s['case']==profile['case'] and s['capture']=='timing'
            and s['requests']==profile['requests'] and s['warmup']==profile['warmup']),None)
        if match:
            comparisons.append(dict(case=profile['case'],requests=profile['requests'],warmup=profile['warmup'],
                ratio=sum(profile['latency_ms'])/sum(match['latency_ms']),
                interpretation='single matched-pass perturbation estimate; order/variance not isolated'))
    (args.root/'comparison.json').write_text(json.dumps(dict(runs=summaries,profiler_overhead=comparisons),indent=2))
    print('ANALYZED actual request traces',len(summaries))


if __name__=='__main__':main()
