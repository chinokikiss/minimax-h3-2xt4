"""Bounded physical diagnostic: short captures gate the continuous matrix."""
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


def main():
    root=Path(__file__).resolve().parents[1]
    out=Path('/kaggle/working/causal_audit_results');out.mkdir(exist_ok=True)
    start=time.monotonic();deadline=start+28800
    manifest=dict(status='setup',scope='Full 50-layer Native Ref2VA, 512x512/124 frames, 4 actual NFEs, default auto VAEs',
        diagnostic_only=True,optimization_changes=False,jobs=[],budget_seconds=28800,
        matrix=['direct_sync','direct_p1','direct_p1_p2','offline_p1_p2'],
        short=dict(cases=['direct_p1_p2'],requests=2,warmup=0),continuous=dict(requests=4,warmup=1),
        correctness='Native exact TE/DiT/video; previously accepted CP audio rounding checked separately',
        warmup_and_export_outside_latency=True,profiled_runtime_overhead_not_subtracted=True)
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save()
    plan=json.loads((root/'diagnostic_plan.json').read_text())
    resume=plan.get('resume')
    if resume:
        for name,sha in resume['source_sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
                raise RuntimeError('Recovered inference/capture source binding mismatch')
        manifest['recovery']=resume
        manifest['remaining_matrix']=['direct_p1','direct_p1_p2','offline_p1_p2']
        save()
    def command(args,budget,logname):
        budget=min(budget,deadline-time.monotonic())
        if budget<=0:raise RuntimeError('Overall diagnostic budget exhausted')
        begin=time.monotonic()
        next_heartbeat=60
        print('AUDIT CAUSAL STAGE BEGIN',logname,'budget',round(budget),flush=True)
        with (out/logname).open('w') as log:
            proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            def pump():
                for line in proc.stdout:
                    log.write(line);log.flush()
                    if line.startswith(('AUDIT','VERIFIED','ERROR','Traceback')):print(line.rstrip(),flush=True)
            reader=threading.Thread(target=pump);reader.start()
            try:
                while proc.poll() is None:
                    elapsed=time.monotonic()-begin
                    (out/'progress.json').write_text(json.dumps(dict(stage=logname,seconds=elapsed,
                        disk_free=shutil.disk_usage('/tmp').free,time_ns=time.time_ns())))
                    if elapsed>=next_heartbeat:
                        # Parent-side telemetry outside the measured request
                        # process; no new fences, RPCs or inference callbacks.
                        print('AUDIT CAUSAL HEARTBEAT',logname,'elapsed_seconds',round(elapsed),
                              'budget_seconds',round(budget),'child_running',True,flush=True)
                        next_heartbeat=elapsed+60
                    if elapsed>budget:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        raise RuntimeError('Stage budget exceeded: '+logname)
                    time.sleep(10)
            finally:
                if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                reader.join()
        if proc.returncode:raise RuntimeError(logname+' failed: '+str(proc.returncode))
        return time.monotonic()-begin
    try:
        ticket=root/'offline_download_ticket.json'
        download_args=[sys.executable,'-u',str(root/'benchmarks/restore_offline_bundle.py'),
            '--ticket',str(ticket),'--directory','/tmp/rank_local_tp2',
            '--checkpoints',str(root/'checkpoint_paths.json'),'--evidence',str(out/'offline_restore.json')]
        if plan['offline_source']=='public':
            download_args=[sys.executable,'-u',str(root/'benchmarks/prepare_causal_offline.py'),
                '--mode','public','--directory','/tmp/rank_local_tp2',
                '--checkpoints',str(root/'checkpoint_paths.json'),'--evidence',str(out/'offline_restore.json')]
        required=sum(s['bytes'] for s in json.loads((root/'semantic_results/pipeline_checkpoint_identity.json').read_text()).values())
        required+=46*2**30+20*2**30
        if shutil.disk_usage('/tmp').free<required:raise RuntimeError('Insufficient disk headroom')
        errors=[]
        def offline_download():
            try:command([*download_args,'--download-only'],1800,'offline_download.log')
            except Exception as error:errors.append(type(error).__name__)
        thread=None
        if plan['offline_source'] in ('public','ticket'):
            if plan['offline_source']=='ticket':
                transport=json.loads(ticket.read_text())
                if min(s['expires'] for s in transport['objects'].values())<time.time()+1200:
                    raise RuntimeError('Offline download ticket expired; no repeated retry')
            thread=threading.Thread(target=offline_download);thread.start()
        command([sys.executable,'-m','pip','install','--no-cache-dir','--quiet',
            'comfy-kitchen==0.2.37','comfy-aimdo==0.5.5','safetensors==0.8.0','einops',
            'transformers','torchsde','sentencepiece','huggingface-hub==1.12.0','psutil',
            'nvidia-ml-py','av>=17','pydantic-settings','kornia','alembic','sqlalchemy',
            'simpleeval','blake3','comfy-angle'],900,'dependencies.log')
        command([sys.executable,'-m','pip','install','--no-cache-dir','--quiet','--upgrade',
            'torch==2.11.0','torchvision==0.26.0','torchaudio==2.11.0','--index-url',
            'https://download.pytorch.org/whl/cu130'],900,'torch.log')
        command(['git','clone','--filter=blob:none','https://github.com/Comfy-Org/ComfyUI.git',str(root/'ComfyUI')],300,'clone.log')
        command(['git','-C',str(root/'ComfyUI'),'checkout','d49e888586dd8ae012c0667b33466b815fee07f7'],120,'checkout.log')
        os.environ.update(COMFY_ROOT=str(root/'ComfyUI'),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',
            NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,COLL,P2P',NCCL_DEBUG_FILE=str(out/'nccl.%h.%p.log'),
            HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='67108864')
        prep='''import hashlib,json,pathlib,shutil
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
root=pathlib.Path(ROOT)
specs=json.loads((root/'semantic_results/pipeline_checkpoint_identity.json').read_text())
def download(item):
 kind,spec=item
 try:path=pathlib.Path(hf_hub_download('Comfy-Org/MiniMax-H3',spec['remote_file'],revision='e5eb578a89295337b8ff433a035929ce0279e0b6',cache_dir='/tmp/causal_models'))
 except Exception as error:raise RuntimeError('Download failed: '+kind+' '+type(error).__name__) from None
 with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
 if sha!=spec['sha256'] or path.stat().st_size!=spec['bytes']:raise RuntimeError('Checkpoint identity mismatch: '+kind)
 print('VERIFIED checkpoint',kind,flush=True);return kind,str(path)
with ThreadPoolExecutor(max_workers=2) as pool:paths=dict(pool.map(download,specs.items()))
(root/'checkpoint_paths.json').write_text(json.dumps(paths))
'''.replace('ROOT',repr(str(root)))
        command([sys.executable,'-u','-c',prep],1800,'models.log')
        offline_ready=False
        if thread:
            thread.join()
            if errors:raise RuntimeError('Offline transfer failed; inspect offline_download.log')
            command([*download_args,'--bind-only'],1200,'offline_binding.log')
            offline_ready=True
        nsys=shutil.which('nsys')
        manifest['nsys']=dict(available=bool(nsys),path=nsys,
            limitation=None if nsys else 'Nsight Systems is absent; no nsys traces will be claimed. Kineto, CUDA events, NVTX, NCCL logs and NVML remain available.')
        if nsys:
            for label,args in [('version',[nsys,'--version']),('help',[nsys,'profile','--help'])]:
                completed=subprocess.run(args,capture_output=True,text=True,timeout=30)
                (out/f'nsys_{label}.txt').write_text(completed.stdout+completed.stderr)
        manifest['status']='native_reference';save()
        if resume:
            command([sys.executable,'-u',str(root/'benchmarks/restore_causal_reference.py'),
                '--ticket',str(root/'resume_reference_ticket.json'),
                '--directory',str(out/'native/native_reference'),
                '--evidence',str(out/'reference_restore.json')],900,'reference_restore.log')
            manifest['short_trace_quality_gate']='Reused version-1 physical pilot, reanalyzed with current analyzer; source-bound and Native-checked'
            save()
        else:
            # Fresh Native numerical reference outside capture.
            command([sys.executable,'-u',str(root/'benchmarks/pipeline_residency_audit.py'),
                '--physical','--strategy','native','--layers','50','--steps','4',
                '--checkpoints',str(root/'checkpoint_paths.json'),'--output',str(out/'native')],1200,'native.log')
        manifest['status']='short_diagnostic';save()
        cases=['direct_sync','direct_p1','direct_p1_p2','offline_p1_p2']
        # Counterbalance timing/profile order between adjacent policies.
        for phase,requests,warmup in [('short',2,0),('continuous',4,1)]:
            if resume and phase=='short':continue
            manifest['status']=phase;save()
            phase_cases=['direct_p1_p2'] if phase=='short' else cases
            for position,case in enumerate(phase_cases):
                if resume and case=='direct_sync':continue
                if case=='offline_p1_p2' and not offline_ready:
                    helper=[sys.executable,'-u',str(root/'benchmarks/prepare_causal_offline.py'),
                        '--directory','/tmp/rank_local_tp2','--evidence',str(out/'offline_restore.json')]
                    if plan['offline_source']=='secret':
                        try:
                            command([*helper,'--mode','secret'],2400,'offline_secret_reuse.log')
                            manifest['offline_source']='immutable HF reuse through Kaggle HF_token'
                        except RuntimeError:
                            manifest['offline_reuse_limitation']='Kaggle HF_token reuse unavailable; sanitized helper error retained'
                            command([*helper,'--mode','regenerate'],2400,'offline_regeneration.log')
                            manifest['offline_source']='Native-verified regeneration on Kaggle after all Direct diagnostics'
                    else:
                        command([*helper,'--mode','regenerate'],2400,'offline_regeneration.log')
                        manifest['offline_source']='Native-verified regeneration on Kaggle after all Direct diagnostics'
                    offline_ready=True;save()
                for capture in (['timing','kineto'] if position%2==0 else ['kineto','timing']):
                    directory=out/phase/case/capture;directory.mkdir(parents=True,exist_ok=True)
                    args=[sys.executable,'-u',str(root/'benchmarks/causal_pipeline_audit.py'),
                        '--case',case,'--capture',capture,'--output',str(directory),
                        '--checkpoints',str(root/'checkpoint_paths.json'),
                        '--reference',str(out/'native/native_reference'),
                        '--requests',str(requests),'--warmup',str(warmup),'--steps','4']
                    if case=='offline_p1_p2':args+=['--rank-checkpoint-dir','/tmp/rank_local_tp2']
                    seconds=command(args,1800 if phase=='short' else 3600,f'{phase}_{case}_{capture}.log')
                    result=json.loads((directory/'result.json').read_text())
                    if not result['passed'] or len(result['requests'])!=requests:
                        raise RuntimeError('Incomplete diagnostic '+case)
                    if any(r['nfe']!=4 for r in result['requests']):raise RuntimeError('Actual NFE mismatch '+case)
                    if capture=='kineto':
                        command([sys.executable,str(root/'analyze_causal_audit.py'),str(directory)],900,f'{phase}_{case}_analysis.log')
                        analysis=json.loads((directory/'causal_analysis.json').read_text())
                        if any(r['ranks'][str(rank)]['kernel_count']==0 for r in analysis['requests'] for rank in (0,1)):
                            raise RuntimeError('GPU kernel trace missing; continuous run not admitted')
                        if not analysis['block_loading'] or not analysis['collective_calls']:
                            raise RuntimeError('Loading/NCCL trace missing; continuous run not admitted')
                    manifest['jobs'].append(dict(phase=phase,case=case,capture=capture,seconds=seconds,passed=True));save()
            command([sys.executable,str(root/'analyze_causal_audit.py'),str(out/phase)],600,phase+'_final_analysis.log')
            if phase=='short':
                manifest['short_trace_quality_gate']='PASS: real NFEs, all ranks, Native checks, loading and NCCL present';save()
        if nsys:
            manifest['status']='nsys_matched_request';save()
            directory=out/'nsys';directory.mkdir(exist_ok=True)
            # Nsight and Kineto do NOT share CUPTI ownership. Separate matched
            # actual requests, explicitly recorded as a different diagnostic pass.
            args=[nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--force-overwrite=true',
                '--output='+str(directory/'pipeline')]
            help_text=(out/'nsys_help.txt').read_text()
            if '--trace-fork-before-exec' in help_text:args+=['--trace-fork-before-exec=true']
            if '--wait' in help_text:args+=['--wait=all']
            args+=[sys.executable,'-u',str(root/'benchmarks/causal_pipeline_audit.py'),
                '--case','direct_p1_p2','--capture','nsys','--output',str(directory),
                '--checkpoints',str(root/'checkpoint_paths.json'),'--reference',str(out/'native/native_reference'),
                '--requests','2','--warmup','1','--steps','4']
            try:
                command(args,2400,'nsys.log');manifest['nsys']['capture']='complete'
            except RuntimeError:
                manifest['nsys']['capture']='unavailable at runtime; inspect nsys.log; Kineto evidence retained'
        command([sys.executable,str(root/'analyze_causal_audit.py'),str(out)],900,'combined_analysis.log')
        manifest.update(status='complete',seconds=time.monotonic()-start);save()
    except BaseException as error:
        manifest.update(status='failed',error=type(error).__name__+': '+str(error),seconds=time.monotonic()-start);save();raise
    finally:
        print('AUDIT CAUSAL FINAL',manifest['status'],flush=True)


if __name__=='__main__':main()
