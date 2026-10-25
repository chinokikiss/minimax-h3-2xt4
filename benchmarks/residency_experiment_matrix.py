"""Counterbalanced full physical requests; separate targeted profile passes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from te_experiment_matrix import summary as original_summary


def summary(result):
    """Do not duplicate the runner's sample timeline in every matrix summary."""
    value=original_summary(result)
    requests=result['requests']
    samples=[s for s in result['host_monitor'].get('samples',[]) if any(
        r['start_ns']<=s['monotonic_ns']<=r['end_ns'] for r in requests)]
    value['ram']=dict(
        peak_process_tree_rss=max((s['process_tree_rss'] for s in samples),default=None),
        minimum_system_available=min((s['available'] for s in samples),default=None),
        peak_cgroup_current=max((s['cgroup']['current'] for s in samples
            if s.get('cgroup',{}).get('available')),default=None),
        semantics='Aggregate only; samples retained once in each runner result. RSS can double-count shared pages.')
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=args.output
    evidence=dict(status='running',physical=True,selected='baseline',results={},profiles={},failures={},
        workload='Full 50-layer Ref2VA, 512x512/124 frames, four actual NFEs, seed1729, same saved Native references',
        cache_coldness='Fresh managed caches per process; layered OS/provider cache not controlled. Reverse policy order confirmation required.',
        order=['baseline','p0','p0_p1','p0_p1','p0','baseline'],
        instrumentation='Identical compact counters in timing; targeted Kineto TE passes excluded from performance comparisons')
    save=lambda:(out/'residency_ablation.json').write_text(json.dumps(evidence,indent=2))
    save()
    def run(policy,phase,requests,capture='timing'):
        folder=out/phase/policy/capture;folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(root/'benchmarks/residency_pipeline_ablation.py'),
            '--case','offline_p1_p2','--capture',capture,'--output',str(folder),
            '--checkpoints',str(root/'checkpoint_paths.json'),'--reference',str(out/'native/native_reference'),
            '--rank-checkpoint-dir','/tmp/rank_local_tp2','--requests',str(requests),'--warmup','0','--steps','4']
        env=dict(os.environ,MINIMAX_H3_RESIDENCY_POLICY=policy,MINIMAX_H3_PROFILE_DETAILS='0')
        print('AUDIT RESIDENCY BEGIN',phase,policy,capture,flush=True)
        start=time.monotonic()
        with (out/f'{phase}_{policy}_{capture}.log').open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                if time.monotonic()-start>3000:
                    import signal
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise RuntimeError('Residency pass deadline exceeded')
                time.sleep(30)
                print('AUDIT RESIDENCY HEARTBEAT',phase,policy,capture,round(time.monotonic()-start),flush=True)
        if process.returncode:raise RuntimeError('Residency pass failed: '+policy+' '+str(process.returncode))
        result=json.loads((folder/'result.json').read_text())
        if not result['passed'] or len(result['requests'])!=requests or any(r['nfe']!=4 for r in result['requests']):
            raise RuntimeError('Incomplete or unequal workload')
        if any(not all(v['passed'] for v in record.values()) for record in result['correctness']):
            raise RuntimeError('Native validation failed')
        state=result['host_budget_after_close']
        if any(state[k]!=0 for k in ('reserved_bytes','preparing','pending_preparation_bytes')):
            raise RuntimeError('Host lease leaked')
        if capture=='kineto':
            subprocess.run([sys.executable,str(root/'tools/experiments/analyze_te_path.py'),str(folder)],check=True,timeout=300)
            # Preserve complete trace evidence compressed; never generate tensor dumps.
            import gzip,hashlib
            receipts=[]
            for path in folder.glob('*kineto.json'):
                sha=hashlib.sha256(path.read_bytes()).hexdigest()
                with path.open('rb') as source,gzip.open(str(path)+'.gz','wb',compresslevel=6) as target:
                    import shutil
                    shutil.copyfileobj(source,target)
                with gzip.open(str(path)+'.gz','rb') as source:
                    if hashlib.file_digest(source,'sha256').hexdigest()!=sha:raise RuntimeError('Trace compression integrity failed')
                receipts.append(dict(file=path.name,sha256=sha,bytes=path.stat().st_size));path.unlink()
            (folder/'trace_archive.json').write_text(json.dumps(receipts,indent=2))
        return summary(result)
    for phase,policies,requests in [('short',('baseline','p0','p0_p1'),2),('continuous',('p0_p1','p0','baseline'),3)]:
        for policy in policies:
            key=phase+'/'+policy
            try:evidence['results'][key]=run(policy,phase,requests)
            except Exception as error:evidence['failures'][key]=str(error)
            save()
    for policy in ('baseline','p0','p0_p1'):
        if 'continuous/'+policy not in evidence['results']:continue
        try:evidence['profiles'][policy]=run(policy,'diagnostic',2,'kineto')
        except Exception as error:evidence['failures']['diagnostic/'+policy]=str(error)
        save()
    evidence.update(status='complete',promotion='No automatic default promotion. Require Native correctness, repeatable E2E improvement, no OOM/leaks, and no increased physical backing traffic; inspect compact counters and traces.')
    save();print('AUDIT RESIDENCY COMPLETE',flush=True)


if __name__=='__main__':main()
