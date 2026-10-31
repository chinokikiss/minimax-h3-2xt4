"""Stage exact production assets, bake the Turbo DiT, then run one bounded Sol execution preflight."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--scratch',type=Path,default=Path('/kaggle/temp/h3_sol_execution_preflight'))
    args=p.parse_args()
    source=args.source.resolve();args.output.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',full_pipeline=False,scope='Physical-T4 execution preflight; no full generation')
    def save():(args.output/'batch_results.json').write_text(json.dumps(state,indent=2))
    def mark(stage):state['status']=stage;save();print('SOL_PREFLIGHT_STAGE',stage,flush=True)
    stopped=threading.Event()
    def heartbeat():
        while not stopped.wait(60):print('SOL_PREFLIGHT_HEARTBEAT',state['status'],flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def run(argv,label,cap):
        import signal
        mark(label);begin=time.monotonic()
        with (args.output/(label+'.log')).open('w') as log:
            proc=subprocess.Popen(argv,cwd=source,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,start_new_session=True,bufsize=1)
            def relay():
                for line in proc.stdout:
                    log.write(line);log.flush()
                    if line.startswith(('SOL_','TP2_','REF2VA_','Traceback')):print(line,end='',flush=True)
            thread=threading.Thread(target=relay,daemon=True);thread.start()
            try:
                proc.wait(timeout=cap)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
                thread.join(timeout=5)
        state.setdefault('commands',[]).append(dict(stage=label,seconds=time.monotonic()-begin,exit_code=proc.returncode));save()
        if proc.returncode:raise RuntimeError(label+' failed; inspect preserved log')
    try:
        import torch
        import importlib.metadata as md
        from patches.legacy_runtime import validate_packages,validate_source
        validate_packages();validate_source(source/'ComfyUI')
        devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
            uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(devices)==2 and len({x['uuid'] for x in devices})==2
        assert all(x['capability']==[7,5] and 'T4' in x['name'] for x in devices)
        compiler=Path(shutil.which('nvcc') or '')
        assert compiler.is_file() and 'release 12.8' in subprocess.check_output([str(compiler),'--version'],text=True)
        os.environ.update(CUDACXX=str(compiler.resolve()),CUDA_HOME=str(compiler.resolve().parent.parent),
            CUDA_PATH=str(compiler.resolve().parent.parent),HF_HUB_DISABLE_XET='1',
            HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='90',OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2')
        state['hardware']=dict(devices=devices,torch=torch.__version__,cuda=torch.version.cuda,
            kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'));save()

        plan=json.loads(args.plan.read_text())
        turbo=plan['turbo_lora']
        assets=json.loads((source/'bundle_assets.json').read_text())
        identity=json.loads((source/'l1_model_identity.json').read_text())
        scratch=args.scratch;scratch.mkdir(exist_ok=False)

        from benchmarks.tp2_public_dit_assets import stage,download
        mark('download_dit')
        base=stage(source,scratch/'dit')
        mark('download_turbo_lora')
        lora=download(turbo,scratch/'adapter')
        mark('offline_bake')
        baked=scratch/'baked'
        run([sys.executable,'-u','-m','patches.tp2_disk_bake','--base',str(base),
             '--lora',str(lora),'1.0','--output',str(baked),'--devices','cuda:0','cuda:1'],
            'offline_bake_process',1800)
        receipt=json.loads((baked/'bake_receipt.json').read_text())
        assert receipt['verified_native_targets']==turbo['targets']
        assert receipt['devices']==['cuda:0','cuda:1']
        shutil.copyfile(baked/'bake_receipt.json',args.output/'offline_bake_receipt.json')

        mark('download_video_vae')
        v=identity['video']
        video=download(dict(repository=assets['model_repository'],revision=assets['model_revision'],
            file=v['remote_file'],bytes=v['bytes'],sha256=v['sha256']),scratch/'video_vae')
        state['assets']=dict(turbo={k:turbo[k] for k in ('repository','revision','file','bytes','sha256')},
            video=dict(file=v['remote_file'],bytes=v['bytes'],sha256=v['sha256']),
            baked_descriptor_sha256=sha(baked/'dit_baked.json'));save()

        mark('execution')
        run([sys.executable,'-u',str(source/'benchmarks/sol_execution_preflight.py'),
             '--checkpoint',str(baked/'dit_baked.json'),'--video',str(video),
             '--output',str(args.output/'execution')],'execution_process',1800)
        result=json.loads((args.output/'execution/result.json').read_text())
        assert result['status']=='complete' and all(result['acceptance'].values())
        state.update(status='complete',acceptance=result['acceptance'])
    except BaseException as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error))
        (args.output/'exception.txt').write_text(traceback.format_exc());save();raise
    finally:
        stopped.set();save()


if __name__=='__main__':main()
