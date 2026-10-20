"""Counterbalanced physical pilot before expanding to the full idle matrix."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from residency_experiment_matrix import summary


def checked_result(path,requests,idle,gap):
    result=json.loads(path.read_text())
    if not result['passed'] or len(result['requests'])!=requests:raise RuntimeError('Incomplete run')
    if any(r['nfe']!=4 for r in result['requests']):raise RuntimeError('Unequal NFE workload')
    if any(not all(v['passed'] for v in c.values()) for c in result['correctness']):raise RuntimeError('Native mismatch')
    if any(r['decode']['video']['strategy']!='native_aligned_query' or
           r['decode']['audio']['strategy']!='channel_parallel' for r in result['requests']):
        raise RuntimeError('Physical decoder fallback; ineligible comparison')
    if any(result['host_budget_after_close'][k]!=0 for k in ('reserved_bytes','preparing','pending_preparation_bytes')):
        raise RuntimeError('Unreleased host ownership')
    if any(r.get('idle',{}).get('errors',0) for r in result['runtime']):raise RuntimeError('Idle worker error')
    if any(e.get('window',{}).get('idle_errors',0) for r in result['runtime'] for e in r['events']):
        raise RuntimeError('Idle window error, including recovered allocation failure')
    if any(record.get('host_io',{}).get('memory_events',{}).get(k,0)
        for record in result['requests'] for k in ('oom','oom_kill','oom_group_kill')):
        raise RuntimeError('Cgroup OOM event')
    if idle and gap:
        for record in result['requests'][:-1]:
            period=record.get('idle_after',{})
            if any(r.get('errors',0) or (r.get('window') or {}).get('idle_errors',0) for r in period.get('ranks',[])):
                raise RuntimeError('Idle background failure')
            if not period.get('accepted') or not all(r.get('loads',0)>0 for r in period.get('ranks',[])):
                raise RuntimeError('Idle prefill did not actually run on both ranks')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('pilot','matrix'),default='pilot')
    args=parser.parse_args();out=args.output;root=Path(__file__).resolve().parents[1]
    out.mkdir(exist_ok=True)
    evidence=dict(status='running',phase=args.phase,physical=True,results={},failures={},
        workload='Full 50-layer TE/DiT, 512x512, 124 frames, four real NFEs, Native refs, seed1729',
        cache_coldness='Fresh managed caches per process; provider/file cache uncontrolled. Orders reversed; cold means managed-cold only.',
        instrumentation='Identical bounded counters, no Kineto in timing. Idle work outside output-ready latency; actual handoff gap retained.',
        default_promotion=False)
    save=lambda:(out/'idle_ablation.json').write_text(json.dumps(evidence,indent=2))
    save()
    def run(policy,idle,gap,repeat,requests):
        key=f'{policy}_idle{int(idle)}_gap{gap}_r{repeat}'
        folder=out/key;folder.mkdir(exist_ok=True)
        command=[sys.executable,'-u',str(root/'benchmarks/idle_residency_pipeline.py'),
            '--case','offline_p1_p2','--capture','timing','--output',str(folder),
            '--checkpoints',str(root/'checkpoint_paths.json'),'--reference',str(out/'native/native_reference'),
            '--rank-checkpoint-dir','/tmp/rank_local_tp2','--requests',str(requests),'--warmup','0','--steps','4']
        env=dict(os.environ,MINIMAX_H3_RESIDENCY_POLICY=policy,MINIMAX_H3_IDLE_RESIDENCY=str(int(idle)),
            MINIMAX_H3_IDLE_GAP=str(gap),MINIMAX_H3_PROFILE_DETAILS='0')
        print('AUDIT IDLE BEGIN',key,'requests',requests,flush=True);start=time.monotonic()
        with (out/(key+'.log')).open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                if time.monotonic()-start>2400:
                    os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('Idle pass deadline exceeded')
                time.sleep(30);print('AUDIT IDLE HEARTBEAT',key,round(time.monotonic()-start),flush=True)
        if process.returncode:raise RuntimeError('Full pipeline failed; inspect '+key+'.log')
        result=checked_result(folder/'result.json',requests,idle,gap)
        value=summary(result)
        value.update(policy=policy,idle=idle,gap=gap,repeat=repeat,seconds=time.monotonic()-start,
            request_samples=[dict(request=r['request'],te_ms=r['stages']['reference_and_TE']['host_wall_ms'],
                e2e_ms=r['latency_ms'],handoff_gap_ms=r['handoff_gap_ms'],idle_after=r.get('idle_after'),
                host_io=r.get('host_io')) for r in result['requests']],
            ranks=[dict(rank=r['rank'],idle=r.get('idle'),events=r['events'],
                host_cache=r['host_cache'],startup_cache=r['startup_cache']) for r in result['runtime']])
        evidence['results'][key]=value;save()
        print('AUDIT IDLE VERIFIED',key,'warm_TE_s',round(value['warm_te_ms']/1000,3),
            'warm_E2E_s',round(value['warm_e2e_ms']/1000,3),
            'idle_loads',[r.get('idle',{}).get('loads',0) for r in result['runtime']],flush=True)
    if args.phase=='pilot':
        # Four matched runs yield two warm observations per policy at 15 s;
        # the sustained runs independently test the zero-gap path.
        plan=[('p0',True,15,0,2),('p0',False,15,0,2),('p0',False,15,1,2),('p0',True,15,1,2),
              ('p0',True,0,0,3),('p0',False,0,0,3)]
    else:
        # Separate launch only after pilot safety/correctness has been reviewed.
        strategies=[('baseline',False),('p0',False),('p0_p1',False),('p0',True),('p0_p1',True)]
        plan=[(p,i,g,r,2) for r in range(2) for g in (0,5,15,60)
              for p,i in (strategies if r==0 else list(reversed(strategies)))]
    evidence['plan']=[dict(policy=p,idle=i,gap=g,repeat=r,requests=n) for p,i,g,r,n in plan];save()
    for policy,idle,gap,repeat,requests in plan:
        try:run(policy,idle,gap,repeat,requests)
        except Exception as error:
            evidence['failures'][f'{policy}_idle{int(idle)}_gap{gap}_r{repeat}']=str(error)
            evidence['status']='failed';save();raise
    groups={}
    for v in evidence['results'].values():
        key=f"{v['policy']}_idle{int(v['idle'])}_gap{v['gap']}"
        group=groups.setdefault(key,dict(te_ms=[],e2e_ms=[]))
        for r in v['request_samples'][1:]:group['te_ms'].append(r['te_ms']);group['e2e_ms'].append(r['e2e_ms'])
    evidence['warm_summary']={k:{metric:dict(n=len(samples),median=statistics.median(samples),
        minimum=min(samples),maximum=max(samples),p90=None if len(samples)<10 else statistics.quantiles(samples,n=10)[8])
        for metric,samples in group.items()} for k,group in groups.items()}
    evidence.update(status='complete',interpretation='Small paired pilot; no statistical speedup or stable P90 from two warm samples. Matrix and model-switch validation remain separate.')
    save();print('AUDIT IDLE COMPLETE',flush=True)


if __name__=='__main__':main()
