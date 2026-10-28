"""Correlated per-call Systems timing and complete allocator lifetime acceptance."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import sqlite3
from bisect import bisect_right


class RangeIndex:
    """Closed NVTX scopes at a timestamp, without a full scan per CUDA launch."""
    def __init__(self,rows):
        endpoints=defaultdict(list)
        for index,(a,b,_) in enumerate(rows):
            endpoints[a].append((index,True));endpoints[b].append((index,False))
        self.times=[];self.active=[];current=set()
        for stamp,changes in sorted(endpoints.items()):
            for index,opened in changes:
                if opened:current.add(index)
                else:current.discard(index)
            self.times.append(stamp);self.active.append(tuple(rows[i] for i in sorted(current)))
    def at(self,stamp):
        index=bisect_right(self.times,stamp)-1
        return self.active[index] if index>=0 else ()


def merge(intervals):
    result=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(b,result[-1][1]))
        else:result.append((a,b))
    return result


def duration(intervals):return sum(b-a for a,b in merge(intervals))


def attribute(database):
    db=sqlite3.connect(database);db.row_factory=sqlite3.Row
    try:
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        needed={'NVTX_EVENTS','StringIds','CUPTI_ACTIVITY_KIND_KERNEL'}
        if not needed<=tables:return dict(complete=False,blocker='Required Systems CUDA/NVTX tables missing')
        names=dict(db.execute('select id,value from StringIds'))
        calls=[];ranges=defaultdict(list)
        for row in db.execute('select * from NVTX_EVENTS where end>start'):
            text=row['text'] or names.get(row['textId'],'')
            ranges[row['globalTid']].append((row['start'],row['end'],text))
            m=re.fullmatch(r'H3_PAIR_CALL/run=(\d+)/rank=(\d+)',text)
            if m:
                calls.append(dict(run=int(m[1]),rank=int(m[2]),tid=row['globalTid'],
                    pid=row['globalTid']&~((1<<24)-1),start_ns=row['start'],end_ns=row['end'],
                    kernels=[],copies=[],memsets=[],blocks=set(),api=[],phases=defaultdict(list)))
        indices={tid:RangeIndex(rows) for tid,rows in ranges.items()}
        launches={}
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
            if table not in tables:continue
            for row in db.execute('select * from '+table):
                matches=[c for c in calls if c['tid']==row['globalTid'] and c['start_ns']<=row['start']<c['end_ns']]
                if not matches:continue
                c=matches[0];active=indices[c['tid']].at(row['start'])
                semantic=[r for r in active if r[2].startswith('H3_DIT/')]
                label=min(semantic,key=lambda r:r[1]-r[0])[2] if semantic else 'Native_pack_or_outer'
                block=next((int(m[1]) for r in active if (m:=re.search(r"/block=\('dit', (\d+), \d+\)/block_compute$",r[2]))),None)
                launches[(c['pid'],row['correlationId'])]=(c,label,block)
                c['api'].append(dict(name=names.get(row['nameId'],'unknown'),start_ns=row['start'],end_ns=row['end']))
        unmatched=[]
        for row in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
            launch=launches.get((row['globalPid'],row['correlationId']))
            if not launch:
                unmatched.append(dict(row));continue
            c,label,block=launch
            name=names.get(row['shortName'],names.get(row['demangledName'],'unknown'))
            kind='NCCL' if 'nccl' in name.lower() else 'compute'
            item=dict(start_ns=row['start'],end_ns=row['end'],name=name,device=row['deviceId'],kind=kind)
            c['kernels'].append(item)
            c['phases'][kind if kind=='NCCL' else label.split('/',3)[-1]].append((row['start'],row['end']))
            if block is not None:c['blocks'].add(block)
        # Background-thread launches can lack a main-thread NVTX correlation.
        # Include their activity in the window; leave their phase unattributed.
        for r in unmatched:
            matches=[c for c in calls if c['pid']==r['globalPid'] and c['start_ns']<=r['start']<
                     max([c['end_ns']]+[k['end_ns'] for k in c['kernels']])]
            if len(matches)!=1:continue
            c=matches[0];name=names.get(r['shortName'],names.get(r['demangledName'],'unknown'))
            c['kernels'].append(dict(start_ns=r['start'],end_ns=r['end'],name=name,
                device=r['deviceId'],kind='NCCL' if 'nccl' in name.lower() else 'compute',uncorrelated=True))
            c['phases']['unattributed_background_launch'].append((r['start'],r['end']))
        for table,key in [('CUPTI_ACTIVITY_KIND_MEMCPY','copies'),('CUPTI_ACTIVITY_KIND_MEMSET','memsets')]:
            if table not in tables:continue
            for row in db.execute('select * from '+table):
                r=dict(row);launch=launches.get((r.get('globalPid'),r.get('correlationId')))
                if not launch:continue
                launch[0][key].append(dict(start_ns=r['start'],end_ns=r['end'],bytes=r.get('bytes',0),
                    device=r.get('deviceId'),copy_kind=r.get('copyKind')))
        rows=[]
        for c in sorted(calls,key=lambda c:(c['rank'],c['run'])):
            kernels=[(r['start_ns'],r['end_ns']) for r in c['kernels']]
            compute=[(r['start_ns'],r['end_ns']) for r in c['kernels'] if r['kind']=='compute']
            collective=[(r['start_ns'],r['end_ns']) for r in c['kernels'] if r['kind']=='NCCL']
            copies=[(r['start_ns'],r['end_ns']) for r in c['copies']+c['memsets']]
            end=max([c['end_ns']]+[b for a,b in kernels+copies]);start=c['start_ns'];span=end-start
            active=merge(kernels+copies);gaps=[];cursor=start
            for a,b in active:
                if a>cursor:gaps.append((cursor,a))
                cursor=max(cursor,b)
            if cursor<end:gaps.append((cursor,end))
            devices={r['device'] for r in c['kernels']}
            longest=[]
            for a,b in sorted(gaps,key=lambda r:r[1]-r[0],reverse=True)[:25]:
                midpoint=(a+b)//2
                cpu=[r for tid,rs in ranges.items() if tid&~((1<<24)-1)==c['pid']
                     for r in rs if r[0]<=midpoint<r[1] and r[2].startswith(('H3_PAIR/','H3_DIT/'))]
                longest.append(dict(start_offset_ms=(a-start)/1e6,ms=(b-a)/1e6,
                                    overlapping_cpu_scopes=[r[2] for r in cpu]))
            rows.append(dict(run=c['run'],rank=c['rank'],devices=sorted(devices),
                trained_blocks_with_kernels=len(c['blocks']),kernel_count=len(kernels),
                uncorrelated_kernel_count=sum(k.get('uncorrelated',False) for k in c['kernels']),
                span_ms=span/1e6,compute_union_ms=duration(compute)/1e6,nccl_union_ms=duration(collective)/1e6,
                kernel_union_ms=duration(kernels)/1e6,copy_and_memset_union_ms=duration(copies)/1e6,
                all_gpu_activity_union_ms=duration(active)/1e6,idle_ms=duration(gaps)/1e6,
                compute_busy_fraction=duration(compute)/span if span else None,
                all_activity_busy_fraction=duration(active)/span if span else None,
                longest_idle_gaps=longest,
                phases={k:dict(kernel_sum_ms=sum(b-a for a,b in v)/1e6,union_ms=duration(v)/1e6,count=len(v)) for k,v in c['phases'].items()},
                copy_bytes=sum(r['bytes'] for r in c['copies']),
                cpu_cuda_api={name:dict(count=sum(r['name']==name for r in c['api']),
                    inclusive_ms=sum(r['end_ns']-r['start_ns'] for r in c['api'] if r['name']==name)/1e6)
                    for name in {r['name'] for r in c['api']}}))
        complete=(len(rows)==4 and {(r['rank'],r['run']) for r in rows}=={(a,b) for a in (0,1) for b in (1,2)}
            and all(r['trained_blocks_with_kernels']==50 and len(r['devices'])==1 for r in rows)
            and len({r['devices'][0] for r in rows if r['devices']})==2)
        return dict(complete=complete,calls=rows,blocker=None if complete else 'Both calls need all 50 blocks on both GPUs',
            limitations=['Timeline activity is not occupancy or Tensor Core utilization',
                'Overlapping kernels, collectives and copies are unions; their categories are not additive',
                'CPU scopes overlapping an idle gap are diagnostic associations, not a causal no-lookahead ablation',
                'Both calls include identical instrumentation; no uninstrumented production latency claim'])
    finally:db.close()


def analyze(root,database):
    from analyze_dit_memory_lifetimes import lifetimes
    from worker_cuda_checkpoint import PROTOCOL
    result=json.loads((root/'result.json').read_text())
    timing=attribute(database)
    memory=[]
    for rank in (0,1):
        for file in sorted((root/'memory'/f'rank{rank}').glob('*_denoise_*.json.gz')):
            with gzip.open(file,'rt') as stream:payload=json.load(stream)
            life=lifetimes(payload)
            name=f'rank{rank}_nfe{payload["nfe"]}_lifetimes.json.gz'
            with gzip.open(root/name,'wt',compresslevel=3) as stream:json.dump(life,stream)
            memory.append(dict(rank=rank,run=payload['nfe'],complete=life['complete_prefix_history'],
                events=len(payload['events']),peak=life['peak'],ooms=life['ooms'],
                unknown_free_events=life['unknown_free_events'],history_limit_reached=life['history_limit_reached'],
                marker_limit_reached=life['marker_limit_reached'],file=name))
    checkpoints=[]
    for rank in (0,1):
        path=root/f'rank{rank}_cuda_checkpoints.json'
        checkpoints.append(json.loads(path.read_text()) if path.is_file() else {})
    flushed=all(c.get('protocol')==PROTOCOL and
        {'call_1_complete','call_2_complete','worker_close'}<={v['label'] for v in c.get('records',[])} and
        all(v['status']=='complete' and v['stopped'] for v in c['records']) for c in checkpoints)
    complete=(result['status']=='complete' and timing['complete'] and len(memory)==4 and all(r['complete'] for r in memory) and flushed
                  and len(result['worker_exit'])==2 and all(r['exit_code']==0 and not r['alive'] for r in result['worker_exit']))
    phases=[]
    for rank in result.get('ranks',[]):
        for call in rank['pair_calls']:
            a,b=call['phase_cpu_range'];cpu=rank['performance_phases']['cpu'][a:b]
            a,b=call['phase_gpu_range'];gpu=rank['performance_phases']['gpu'][a:b]
            labels=sorted({r['name'] for r in cpu})
            phases.append(dict(rank=rank['rank'],run=call['run'],
                commands=[c for c in rank['benchmark']['commands'] if c.get('command')=='denoise' and c.get('nfe')==call['run']],
                command_gpu_spans=[r for r in rank['benchmark']['gpu'] if r.get('kind')=='command_gpu_span' and r.get('command')=='denoise' and r.get('nfe')==call['run']],
                observer_export_ms=sum(r['end_ns']-r['start_ns'] for r in rank['host_events']
                    if r['name']=='observer/lifetime_export' and r['run']==call['run'])/1e6,
                weight_uses=[r for r in rank['weight_uses'] if r['run']==call['run']],
                host_cache_delta={k:v-call['cache_stats_before'].get(k,0) for k,v in call['cache_stats'].items()},
                window_delta={k:v-call['window_stats_before'].get(k,0) for k,v in call['window_stats'].items()},
                phases={name:dict(calls=sum(r['name']==name for r in cpu),
                    inclusive_host_ms=sum(r['end_ns']-r['start_ns'] for r in cpu if r['name']==name)/1e6,
                    thread_cpu_ms=sum(r['thread_cpu_ns'] for r in cpu if r['name']==name)/1e6,
                    stream_span_ms=sum(r['stream_elapsed_ms'] for r in gpu if r['name']==name)) for name in labels}))
    report=dict(status='complete' if complete else 'profile_incomplete',nsight=timing,memory=memory,phase_comparison=phases,
                worker_flush_verified=flushed,
                setup={k:v for k,v in result.items() if k.endswith('_seconds')},cases=result['cases'],
                input_identity=result['input_identity'])
    (root/'analysis.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--sqlite',type=Path,required=True)
    args=p.parse_args();print(analyze(args.input,args.sqlite)['status'])
