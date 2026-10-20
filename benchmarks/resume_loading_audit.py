"""Resume the qualified P0/P1/P2 study inside the existing manual Kaggle session.

No installs, downloads, reconversion, upload, or instance submission. Only
completed and matching cases may be reused. First failure stops the study.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/kaggle/working/pipeline_audit_results_resume')
OLD=Path('/kaggle/working/pipeline_audit_results')
JOBS=[('native','native','direct',False,900),
      ('direct_sync','sync','direct',False,1200),
      ('direct_p1','adaptive','direct',False,1200),
      ('direct_p1_p2','adaptive','direct',True,1200),
      ('offline_p1_p2','adaptive','offline',True,1200)]


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def command(args,budget,logname):
    start=time.monotonic()
    print('AUDIT RESUME BEGIN',logname,'budget_seconds',round(budget),flush=True)
    with (OUT/logname).open('w') as log:
        proc=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,start_new_session=True)
        def pump():
            for line in proc.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
        reader=threading.Thread(target=pump);reader.start()
        deadline=start+budget;rc=None
        try:
            while rc is None:
                try:rc=proc.wait(timeout=max(.1,min(30,deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    print('AUDIT RESUME PROGRESS',logname,'elapsed_seconds',round(time.monotonic()-start),flush=True)
                    if time.monotonic()>=deadline:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        rc=124
        finally:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            if rc:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            reader.join()
    print('AUDIT RESUME END',logname,'return_code',rc,'seconds',round(time.monotonic()-start,2),flush=True)
    if rc:raise RuntimeError(logname+' failed with return code '+str(rc))
    return time.monotonic()-start


def matching_result(path,strategy,loader,p2,guard):
    if not path.exists():return False
    r=json.loads(path.read_text())
    if not (r.get('passed') and r['physical'] and r['strategy']==strategy and
            (r['layers'],r['steps'],r['width'],r['height'])==(50,4,512,512)):
        return False
    if any(r['source_sha256'].get(k)!=v for k,v in guard['result_source_sha256'].items()):
        return False
    policy=r.get('loading_policy',{})
    if policy.get('loader')!=loader or policy.get('cross_request_te')!=p2:return False
    if strategy=='native':
        return len(r['requests'])==1 and all((OUT/'native_reference'/(n+'.pt')).exists()
            for n in ('conditioning','video','audio','pixels','waveform'))
    return (len(r['requests'])==2 and [q['temperature'] for q in r['requests']]==['cold','warm'] and
        all(set(q['correctness'])=={'conditioning','video','audio','pixels','waveform'} and
            all(c['passed'] and c['is_bit_exact'] for c in q['correctness'].values()) for q in r['requests']))


def main():
    import psutil
    guard=json.loads((ROOT/'semantic_results/manual_resume_qualification.json').read_text())
    if not guard['passed']:raise RuntimeError('Locally qualified frozen source required')
    for name,sha in guard['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('Qualified model source changed: '+name)
    for process in psutil.process_iter(['pid','cmdline']):
        if process.info['pid'] in (os.getpid(),os.getppid()):continue
        if any(str(ROOT/'benchmarks'/name) in (process.info['cmdline'] or [])
               for name in ('run_loading_audit.py','observe_loading_audit.py')):
            raise RuntimeError('Another audit is still running; do not overlap GPU cases')
    checkout=subprocess.check_output(['git','-C',str(ROOT/'ComfyUI'),'rev-parse','HEAD'],text=True).strip()
    if checkout!=guard['comfy_commit']:raise RuntimeError('Pinned ComfyUI checkout required')
    paths=json.loads((ROOT/'checkpoint_paths.json').read_text())
    specs=json.loads((ROOT/'semantic_results/pipeline_checkpoint_identity.json').read_text())
    for kind,spec in specs.items():
        if not Path(paths[kind]).is_file() or Path(paths[kind]).stat().st_size!=spec['bytes']:
            raise RuntimeError('Original checkpoint missing; this cell requires the existing session: '+kind)
    ticket=json.loads((ROOT/'offline_download_ticket.json').read_text())
    for name,row in ticket['objects'].items():
        p=Path('/tmp/rank_local_tp2')/name
        if not p.is_file() or p.stat().st_size!=row['bytes']:
            raise RuntimeError('Downloaded shard missing; no network retries are performed: '+name)
    OUT.mkdir(exist_ok=True)
    os.environ.update(COMFY_ROOT=str(ROOT/'ComfyUI'),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',
        NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,COLL,P2P',
        NCCL_DEBUG_FILE=str(OUT/'nccl.%h.%p.log'),MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='67108864')
    if not (OUT/'offline_restore.json').exists():
        shutil.copyfile(OLD/'offline_restore.json',OUT/'offline_restore.json')
    started=time.monotonic();manifest=dict(status='preflight',resume=True,installs=False,
        downloads=False,reconversion=False,model_operator_changes=False,jobs=[],
        gpu_budget_seconds=sum(j[4] for j in JOBS),binding_budget_seconds=1200,
        source_guard_sha256=hashlib.sha256((ROOT/'semantic_results/manual_resume_qualification.json').read_bytes()).hexdigest())
    def save():write_json(OUT/'manifest.json',manifest)
    save()
    try:
        probe="""import json,pathlib,torch,psutil
devices=[dict(name=torch.cuda.get_device_name(i),bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]
assert len(devices)==2 and all('T4' in d['name'] for d in devices)
assert torch.__version__=='2.11.0+cu130'
pathlib.Path(OUTPUT).write_text(json.dumps(dict(devices=devices,torch=torch.__version__,cuda=torch.version.cuda,ram=psutil.virtual_memory().total)))
""".replace('OUTPUT',repr(str(OUT/'hardware.json')))
        command([sys.executable,'-c',probe],60,'hardware.log')
        gpu_end=time.monotonic()+manifest['gpu_budget_seconds']
        manifest['status']='running';save()
        for label,strategy,loader,p2,budget in JOBS:
            folder=OUT if strategy=='native' else OUT/label;folder.mkdir(exist_ok=True)
            if strategy!='native' and not (folder/'native_reference').exists():
                (folder/'native_reference').symlink_to(OUT/'native_reference',target_is_directory=True)
            result_path=folder/(strategy+'_result.json')
            if matching_result(result_path,strategy,loader,p2,guard):
                manifest['jobs'].append(dict(strategy=label,passed=True,reused=True));save();continue
            if loader=='offline':
                evidence=json.loads((OUT/'offline_restore.json').read_text())
                if evidence['phase']!='bound':
                    seconds=command([sys.executable,'-u',str(ROOT/'benchmarks/restore_offline_bundle.py'),
                        '--ticket',str(ROOT/'offline_download_ticket.json'),'--directory','/tmp/rank_local_tp2',
                        '--checkpoints',str(ROOT/'checkpoint_paths.json'),'--evidence',str(OUT/'offline_restore.json'),
                        '--bind-only'],1200,'offline_binding.log')
                    gpu_end+=seconds
                manifest['offline_restore']=json.loads((OUT/'offline_restore.json').read_text());save()
            audit=[sys.executable,'-u',str(ROOT/'benchmarks/observe_loading_audit.py'),'--physical',
                '--strategy',strategy,'--steps','4','--cache-gib','3','--checkpoints',str(ROOT/'checkpoint_paths.json'),
                '--output',str(folder),'--loader',loader,'--cross-request-te','on' if p2 else 'off']
            if loader=='offline':audit+=['--rank-checkpoint-dir','/tmp/rank_local_tp2']
            if strategy!='native':audit+=['--profile']
            seconds=command([sys.executable,'-u',str(ROOT/'benchmarks/monitor_pipeline_job.py'),
                '--output',str(OUT/(label+'_pcie.json')),'--',*audit],
                max(.1,min(budget,gpu_end-time.monotonic())),label+'.log')
            if not matching_result(result_path,strategy,loader,p2,guard):
                raise RuntimeError('Unaccepted source/workload/correctness: '+label)
            manifest['jobs'].append(dict(strategy=label,seconds=seconds,passed=True,reused=False));save()
        command([sys.executable,str(ROOT/'analyze_loading_audit.py'),str(OUT)],300,'analysis.log')
        manifest.update(status='complete',total_seconds=time.monotonic()-started);save()
    except BaseException as error:
        manifest.update(status='failed',error=type(error).__name__+': '+str(error),
                        total_seconds=time.monotonic()-started);save();raise
    finally:
        with ZipFile(OUT/'summary_evidence.zip','w',ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and p.suffix in ('.json','.log','.md') and not p.name.endswith('_trace.json'):
                    z.write(p,p.relative_to(OUT))
        traces=list(OUT.rglob('*_trace.json'))
        if traces:
            with ZipFile(OUT/'profiler_traces.zip','w',ZIP_DEFLATED) as z:
                for p in traces:z.write(p,p.relative_to(OUT))
        print('AUDIT RESUME FINAL',json.dumps(manifest),flush=True)
        print('DOWNLOAD',OUT/'summary_evidence.zip',flush=True)


if __name__=='__main__':main()
