"""Stage public immutable inputs, bake eight full experts, run one long request."""
import argparse
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
import traceback


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bake-only',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--base',type=Path); parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--bank-output',type=Path)
    args=parser.parse_args(); source=args.source.resolve()
    sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    # Initialize the validated legacy argument parser without leaking this
    # harness's flags into Native modules imported during staging/baking.
    sys.argv=sys.argv[:1]
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages(); validate_source(source/'ComfyUI')
    import torch
    if (torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5)
            or 'T4' not in torch.cuda.get_device_name(i) for i in (0,1))):
        raise RuntimeError('This full PTD batch requires actual two-T4 hardware')
    torch.set_grad_enabled(False); torch.set_num_threads(2)
    if args.bake_only:
        from patches.ptd.bank import bake
        bake(args.base,args.checkpoint,args.bank_output,layers=50,allow_pruned_adaln=True,
             runtime_layout=False,split_auxiliary=False)
        return
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',commands=[],full_pipeline=True,request_count=1,
               online_lora=False,split_auxiliary=False,fp32_sidecar=False)
    def save(): (output/'batch_results.json').write_text(json.dumps(state,indent=2))
    def mark(stage,**extra):
        state.update(status=stage,**extra); save(); print('PTD_E2E_STAGE',stage,flush=True)
    def run(argv,label,cap):
        mark(label); start=time.monotonic()
        with (output/(label+'.log')).open('w') as log:
            process=subprocess.Popen(argv,cwd=source,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,start_new_session=True,bufsize=1)
            def relay():
                for line in process.stdout:
                    log.write(line); log.flush()
                    if line.startswith(('PTD_','BASE_','L1_','Traceback')): print(line,end='',flush=True)
            thread=threading.Thread(target=relay,daemon=True); thread.start()
            try: process.wait(timeout=cap)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=5)
                thread.join(timeout=5)
                state['commands'].append(dict(stage=label,seconds=time.monotonic()-start,exit_code=process.returncode))
                save()
            if process.returncode: raise RuntimeError(label+' failed; preserve the stage log')
    stopped=threading.Event()
    def heartbeat():
        while not stopped.wait(60): print('PTD_E2E_HEARTBEAT',state['status'],flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        from importlib.metadata import version
        compiler=Path(shutil.which('nvcc') or '')
        if not compiler.is_file() or 'release 12.8' not in subprocess.check_output([str(compiler),'--version'],text=True):
            raise RuntimeError('Validated preinstalled CUDA12.8 compiler required')
        os.environ.update(CUDACXX=str(compiler.resolve()),CUDA_HOME=str(compiler.resolve().parent.parent),
            CUDA_PATH=str(compiler.resolve().parent.parent),HF_HUB_DISABLE_XET='1',
            HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='90',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='67108864')
        state['hardware']=dict(torch=torch.__version__,cuda=torch.version.cuda,
            kitchen=version('comfy-kitchen'),tilelang=version('tilelang'),
            devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
                          uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in (0,1)])
        if len({d['uuid'] for d in state['hardware']['devices']})!=2: raise RuntimeError('Distinct GPUs required')
        plan=json.loads((source/'ptd_e2e_plan.json').read_text())
        assets=json.loads((source/'bundle_assets.json').read_text())
        identity=json.loads((source/'l1_model_identity.json').read_text())
        from benchmarks.ptd_e2e_contract import validate,resource_plan
        validate(plan); state['request']=plan
        scratch=Path('/tmp/h3_ptd_e2e_20261029'); scratch.mkdir(exist_ok=False)
        (scratch/'.owned_by_ptd_e2e').write_text('One authorized full PTD run; preserve partial bank on error')
        import psutil
        host_limit=psutil.virtual_memory().total
        for cgroup in (Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')):
            if cgroup.is_file():
                value=cgroup.read_text().strip()
                if value.isdigit(): host_limit=min(host_limit,int(value))
        resources=resource_plan(assets['auxiliary']['dit']['native_header'],assets,identity,
                               free_bytes=shutil.disk_usage(scratch).free,host_limit=host_limit)
        state['resource_plan']=resources; save()
        images=[]
        from benchmarks.qualify_anime_inputs import download_image
        for row in plan['reference_images']:
            folder=source/'bench_assets/l1_current'; folder.mkdir(parents=True,exist_ok=True)
            download_image(row,folder); images.append(str(folder/row['file']))
        mark('stage_te_and_dit_base')
        from benchmarks.prepare_tp2_bundle_assets import prepare
        paths,_=prepare(source,scratch,output)
        mark('download_full_ptd_on_t4')
        from patches.ptd.checkpoint import download_on_t4
        checkpoint=download_on_t4(scratch/'adapter')
        from patches.ptd.bank import disk_plan,PTDBank
        bank_output=scratch/'eight_hop_tp2'
        disk=disk_plan(paths['dit'],bank_output,layers=50,runtime_layout=False,split_auxiliary=False)
        state['bake_disk_plan']=disk; save()
        if disk['free_bytes']<disk['required_free_bytes']: raise OSError('Full PTD bank disk admission failed')
        run([sys.executable,'-u',str(source/'benchmarks/run_ptd_e2e_batch.py'),
             '--source',str(source),'--output',str(output),'--bake-only',
             '--base',paths['dit'],'--checkpoint',str(checkpoint),'--bank-output',str(bank_output)],
             'offline_eight_hop_bake',6300)
        bank=PTDBank(bank_output/'bank.json')
        from benchmarks.run_ptd_e2e import validate_bank
        state['persistent_bank']=validate_bank(bank)
        receipts=output/'bank_receipts'; receipts.mkdir()
        shutil.copyfile(bank.path,receipts/'bank.json')
        for hop,(directory,_) in enumerate(bank.hops):
            shutil.copyfile(directory/'receipt.json',receipts/f'hop_{hop:02d}.json')
        paths['bank']=str(bank.path)
        mark('download_vaes')
        from huggingface_hub import hf_hub_download
        from benchmarks.tp2_public_dit_assets import recover
        for kind in ('video','audio'):
            expected=identity[kind]
            path=Path(recover(lambda attempt:hf_hub_download(assets['model_repository'],expected['remote_file'],
                revision=assets['model_revision'],token=False,local_dir=str(scratch/'vaes')),label=kind+'_vae'))
            if path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:
                raise ValueError('Immutable VAE checksum mismatch: '+kind)
            paths[kind]=str(path)
        request=plan|dict(paths=paths,images=images,output=str(output/'generation'),bank_sha256=bank.digest,lora_stack=[])
        (output/'request_config.json').write_text(json.dumps(request,indent=2))
        run([sys.executable,'-u',str(source/'benchmarks/run_ptd_e2e.py'),
             '--config',str(output/'request_config.json')],'full_e2e',3600)
        result=json.loads((output/'generation/result.json').read_text())
        if result['status']!='complete': raise RuntimeError('PTD E2E incomplete')
        mark('complete',completed_hops=len(result['sampler_steps']),media=result['media'])
    except BaseException as error:
        state.update(status='failed',error=repr(error)); (output/'exception.txt').write_text(traceback.format_exc()); save(); raise
    finally: stopped.set()


if __name__=='__main__': main()
