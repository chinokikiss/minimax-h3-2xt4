"""Summarize accepted physical timings and isolated profiler diagnostics."""
import argparse
import ast
import json
from pathlib import Path


def union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return merged


def duration(intervals):
    return sum(b-a for a,b in union(intervals))


def overlap(left,right):
    left,right=union(left),union(right)
    i=j=0;total=0
    while i<len(left) and j<len(right):
        a,b=left[i];c,d=right[j]
        total+=max(0,min(b,d)-max(a,c))
        if b<=d:i+=1
        else:j+=1
    return total


def trace_summary(path):
    events=json.loads(path.read_text())['traceEvents']
    weight_ranges={}
    for event in events:
        if event.get('ph')=='X' and event.get('dur') and event.get('name','').startswith('audit/H2D/'):
            weight_ranges.setdefault((event.get('pid'),event.get('tid')),[]).append(
                (event['ts'],event['ts']+event['dur']))
    weight_correlations=set()
    for event in events:
        if event.get('cat')=='cuda_runtime' and 'memcpy' in event.get('name','').lower():
            ranges=weight_ranges.get((event.get('pid'),event.get('tid')),[])
            if any(start<=event.get('ts',-1)<=end for start,end in ranges):
                correlation=event.get('args',{}).get('correlation')
                if correlation is not None:weight_correlations.add(correlation)
    buckets={k:[] for k in ('gemm','attention','quantization_layout','nccl','other','h2d','other_copy','memset')}
    launches=0;bytes_h2d=0;kernel_names={};weight_h2d=[];weight_h2d_bytes=0
    for e in events:
        if e.get('ph')!='X' or not e.get('dur'):continue
        name=e.get('name','');low=name.lower();cat=e.get('cat','')
        interval=(e['ts'],e['ts']+e['dur'])
        if cat=='kernel':
            launches+=1;kernel_names[name]=kernel_names.get(name,0)+e['dur']/1000
            if 'nccl' in low:key='nccl'
            elif any(t in low for t in ('attention','fmha','flash','cutlassf','softmax')):key='attention'
            elif any(t in low for t in ('gemm','matmul','sgemm','hgemm','igemm')):key='gemm'
            elif any(t in low for t in ('quantiz','convrot','hadamard','rotate','transpose','contiguous','copy_kernel')):key='quantization_layout'
            else:key='other'
            buckets[key].append(interval)
        elif cat=='gpu_memcpy':
            key='h2d' if ('htod' in low or 'h2d' in low or 'host to device' in low) else 'other_copy'
            buckets[key].append(interval)
            if key=='h2d':
                details=e.get('args',{})
                bytes_h2d+=int(details.get('bytes',details.get('Bytes',0)))
                if details.get('correlation') in weight_correlations:
                    weight_h2d.append(interval)
                    weight_h2d_bytes+=int(details.get('bytes',details.get('Bytes',0)))
        elif cat=='gpu_memset':
            buckets['memset'].append(interval)
    compute=[i for key in ('gemm','attention','quantization_layout','other') for i in buckets[key]]
    all_activity=[i for values in buckets.values() for i in values]
    span=max((b for a,b in all_activity),default=0)-min((a for a,b in all_activity),default=0)
    h2d=duration(buckets['h2d'])
    return dict(scope='one extra full-depth sampler evaluation; excluded from cold/warm timings',
        classification='name heuristics; ambiguous kernels in other; raw traces authoritative',
        kernel_launches=launches,h2d_payload_bytes=bytes_h2d,
        weight_h2d_attribution='CUDA runtime correlations contained in audit/H2D CPU ranges; other H2D remains unclassified',
        attributed_weight_h2d_bytes=weight_h2d_bytes,
        attributed_weight_h2d_union_ms=duration(weight_h2d)/1000,
        attributed_weight_h2d_compute_overlap_ms=overlap(weight_h2d,compute)/1000,
        attributed_weight_h2d_nccl_overlap_ms=overlap(weight_h2d,buckets['nccl'])/1000,
        activity_sum_ms={k:sum(b-a for a,b in v)/1000 for k,v in buckets.items()},
        h2d_union_ms=h2d/1000,h2d_compute_overlap_ms=overlap(buckets['h2d'],compute)/1000,
        h2d_nccl_overlap_ms=overlap(buckets['h2d'],buckets['nccl'])/1000,
        h2d_compute_overlap_fraction=overlap(buckets['h2d'],compute)/h2d if h2d else None,
        gpu_activity_span_ms=span/1000,kernel_copy_memset_free_ms=(span-duration(all_activity))/1000,
        top_kernels_ms=sorted(kernel_names.items(),key=lambda p:p[1],reverse=True)[:20])


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--strategies',nargs='+',choices=('sync','adaptive'),default=['sync','adaptive'])
    args=p.parse_args()
    out=dict(timings=[],rank_diagnostics=[],traces={})
    for strategy in args.strategies:
        result=json.loads((args.directory/f'{strategy}_result.json').read_text())
        if not (result['passed'] and result['physical'] and result['layers']==50):
            raise RuntimeError('Only completed physical full-depth runs can be accepted: '+strategy)
        previous={}
        for request in result['requests']:
            if not all(c['passed'] for c in request['correctness'].values()):
                raise RuntimeError('Incorrect request cannot supply performance evidence')
            out['timings'].append(dict(strategy=strategy,temperature=request['temperature'],
                end_to_end_ms=request['end_to_end_ms'],stage_ms=request['stage_ms'],
                worker_startup_ms=request.get('worker_startup_ms'),
                startup_inclusive_ms=request.get('startup_inclusive_ms'),
                all_bit_exact=all(c['is_bit_exact'] for c in request['correctness'].values())))
            for rank in request['runtime']:
                r=rank['rank'];old=previous.get(r,{});cache=rank['host_cache']
                events=rank['events'][len(old.get('events',[])):]
                windows={kind:[e['window'] for e in events if e['phase']==kind] for kind in ('te','dit')}
                evaluations=rank['calls']-old.get('calls',0)
                block_loads=sum(count for w in windows['dit'] for key,count in w['loads'].items()
                    if len(ast.literal_eval(key))==4 and ast.literal_eval(key)[1]=='dit')
                delta={k:cache[k]-old.get('host_cache',{}).get(k,0) for k in ('hits','misses','evictions','preparations','prepare_ms','wait_ms')}
                out['rank_diagnostics'].append(dict(strategy=strategy,temperature=request['temperature'],rank=r,
                    denoiser_evaluations=evaluations,dit_block_loads=block_loads,
                    dit_avoided_reload_fraction=1-block_loads/(evaluations*result['layers']) if evaluations else None,
                    gpu=rank['gpu'],host_cache_delta=delta,
                    host_hit_fraction=delta['hits']/(delta['hits']+delta['misses']) if delta['hits']+delta['misses'] else None,
                    checkpoint_read_bytes_delta=rank['checkpoint_read_bytes']-old.get('checkpoint_read_bytes',0),
                    checkpoint_read_calls_delta=rank['checkpoint_read_calls']-old.get('checkpoint_read_calls',0),
                    shared_host=cache['shared_budget'],windows={kind:dict(
                        h2d_bytes=sum(w['h2d_bytes'] for w in ws),hits=sum(w['hits'] for w in ws),
                        evictions=sum(w['evictions'] for w in ws),host_wait_ms=sum(w['host_wait_ms'] for w in ws),
                        foreground_reclaims=sum(w.get('foreground_reclaims',0) for w in ws),
                        reclaim_ms=sum(w.get('reclaim_ms',0) for w in ws),
                        lease_gap_cuda_ms=sum(x['cuda_ms'] for w in ws for x in w['audit_h2d_waits']),
                        loads={key:sum(w['loads'].get(key,0) for w in ws) for w in ws for key in w['loads']},
                        lookahead_changes=[c for w in ws for c in w['lookahead_changes']]) for kind,ws in windows.items()}))
                previous[r]=rank
        for rank in (0,1):
            trace=args.directory/f'{strategy}_rank{rank}_trace.json'
            if trace.exists():out['traces'][trace.name]=trace_summary(trace)
        out.setdefault('memory',{})[strategy]={k:v for k,v in result['host_monitor'].items() if k!='samples'}
        out['memory'][strategy]['host_budget_after_close']=result.get('host_budget_after_close')
    (args.directory/'analysis.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out['timings'],indent=2))


if __name__=='__main__':main()
