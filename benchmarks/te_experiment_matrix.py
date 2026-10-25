"""Bounded, Native-checked physical TE ablations on identical full requests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def summary(result):
    requests=result['requests']
    samples=result['host_monitor'].get('samples',[])
    measured=[s for s in samples if any(r['start_ns']<=s['monotonic_ns']<=r['end_ns'] for r in requests)]
    ram=dict(peak_process_tree_rss=max((s['process_tree_rss'] for s in measured),default=None),
        minimum_system_available=min((s['available'] for s in measured),default=None),
        semantics='Measured-request samples; process-tree RSS may double-count shared pages')
    return dict(passed=result['passed'],requests=len(requests),
        cold_te_ms=requests[0]['stages']['reference_and_TE']['host_wall_ms'],
        cold_e2e_ms=requests[0]['latency_ms'],
        warm_te_ms=sum(r['stages']['reference_and_TE']['host_wall_ms'] for r in requests[1:])/max(1,len(requests)-1),
        warm_e2e_ms=sum(r['latency_ms'] for r in requests[1:])/max(1,len(requests)-1),
        steady_requests_per_hour=3600000*(len(requests)-1)/sum(r['latency_ms'] for r in requests[1:]) if len(requests)>1 else None,
        ram=ram,gpu=[r.get('gpu') for r in result['runtime']],
        correctness=result['correctness'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=args.output
    evidence=dict(status='running',workload='Same full 50-layer 512x512/124-frame Native Ref2VA, 4 actual NFEs',
        cache_coldness='Fresh rank processes/managed caches per policy; OS page cache uncontrolled and shared across cases',
        profiling_scope='TE/reference only within real full requests; no GPU inactivity claims outside capture',
        order='baseline, coalesced, coalesced_parallel; confirm winner before baseline to reverse initial order',
        inference_math_changed=False,ablations={},confirmation={},selected='baseline',jobs=[],failures={})
    save=lambda:(out/'te_ablation.json').write_text(json.dumps(evidence,indent=2))
    save()
    def run(policy,phase,requests,capture='timing'):
        folder=out/phase/policy/capture;folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(root/'benchmarks/te_pipeline_ablation.py'),
            '--case','offline_p1_p2','--capture',capture,'--output',str(folder),
            '--checkpoints',str(root/'checkpoint_paths.json'),'--reference',str(out/'native/native_reference'),
            '--rank-checkpoint-dir','/tmp/rank_local_tp2','--requests',str(requests),'--warmup','0','--steps','4']
        if capture=='kineto':
            command=[sys.executable,'-u',str(root/'benchmarks/cpu_observer_supervisor.py'),'--output',str(folder),'--',*command]
        print('AUDIT TE BEGIN',phase,policy,capture,'requests',requests,flush=True)
        environment=dict(os.environ,MINIMAX_H3_TE_ABLATION=policy,MINIMAX_H3_PROFILE_DETAILS='0')
        start=time.monotonic()
        with (out/f'{phase}_{policy}_{capture}.log').open('w') as log:
            process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if time.monotonic()-start>2400:
                    # kill descendants too; each runner owns its own rank workers.
                    import psutil
                    for child in psutil.Process(process.pid).children(recursive=True):child.kill()
                    process.kill();process.wait();raise RuntimeError('TE pass deadline exceeded')
                time.sleep(30)
                print('AUDIT TE HEARTBEAT',phase,policy,capture,round(time.monotonic()-start),flush=True)
        if process.returncode:raise RuntimeError('TE pass failed: '+policy+' '+str(process.returncode))
        result=json.loads((folder/'result.json').read_text())
        if not result['passed'] or len(result['requests'])!=requests or any(r['nfe']!=4 for r in result['requests']):
            raise RuntimeError('Incomplete/unequal real request')
        if any(not all(v['passed'] for v in record.values()) for record in result['correctness']):
            raise RuntimeError('Native validation failed')
        evidence['jobs'].append(dict(policy=policy,phase=phase,capture=capture,seconds=time.monotonic()-start,passed=True));save()
        if capture=='kineto':
            subprocess.run([sys.executable,str(root/'tools/experiments/analyze_te_path.py'),str(folder)],check=True,timeout=300)
        return summary(result)
    evidence['ablations']['baseline']=run('baseline','short',2);save()
    for policy in ('coalesced','coalesced_parallel'):
        try:evidence['ablations'][policy]=run(policy,'short',2)
        except Exception as error:evidence['failures'][policy]=type(error).__name__+': '+str(error)
        save()
    baseline=evidence['ablations']['baseline']
    eligible=[p for p,v in evidence['ablations'].items() if p!='baseline' and
        v['warm_te_ms']<.97*baseline['warm_te_ms'] and v['warm_e2e_ms']<.99*baseline['warm_e2e_ms']]
    candidate=min(eligible,key=lambda p:evidence['ablations'][p]['warm_e2e_ms']) if eligible else None
    if candidate:
        for policy in (candidate,'baseline'):
            evidence['confirmation'][policy]=run(policy,'continuous',3);save()
        a,b=(evidence['confirmation'][p] for p in ('baseline',candidate))
        if b['warm_te_ms']<.97*a['warm_te_ms'] and b['warm_e2e_ms']<.99*a['warm_e2e_ms']:
            evidence['selected']=candidate
    # One targeted capture per policy. Capturing fewer layers/NFEs is never
    # used for the performance comparison; these full requests only narrow
    # the observation interval and keep exports outside request latency.
    profiles=['baseline']+([evidence['selected']] if evidence['selected']!='baseline' else [])
    for policy in profiles:
        try:run(policy,'diagnostic',1,'kineto')
        except Exception as error:evidence['failures']['profile/'+policy]=str(error)
        save()
    evidence.update(status='complete',
        acceptance='Native checked; >=3% warm TE and >=1% warm E2E reduction in both short and reverse-order confirmation. Small fixed workload; no statistical/general workload claim.')
    save();print('AUDIT TE COMPLETE selected',evidence['selected'],flush=True)


if __name__=='__main__':main()
