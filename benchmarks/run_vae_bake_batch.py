"""Two T4-only experiments: VAE checkpoint A/B and instrumented offline baking."""
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


def qualify(source, inputs):
    import torch
    plan=json.loads((source/'vae_bake_plan.json').read_text())
    found=list(inputs.rglob('video_latents.pt'))
    assert len(found)==1
    latent=found[0]
    expected=plan['latent']
    assert latent.stat().st_size==expected['bytes'] and sha(latent)==expected['sha256']
    z=torch.load(latent,map_location='cpu',weights_only=True)['video']
    assert list(z.shape)==expected['shape']==[1,24,107,34,60]
    assert torch.isfinite(z).all()
    assert hashlib.sha256(z.contiguous().view(torch.uint8).numpy()).hexdigest()==expected['tensor_sha256']
    for label,row in plan['vae_models'].items():
        header=json.loads((source/(label+'_header.json')).read_text())
        assert sum(k.startswith('decoder.transformer_blocks.') and k.endswith('.scale1') for k in header)==36
        assert row['revision']=='e5eb578a89295337b8ff433a035929ce0279e0b6'
        assert all(header[f'decoder.transformer_blocks.{i}.attn.to_qkv.weight']['dtype']==('I8' if label=='int8_convrot' else 'F16') for i in range(36))
    return plan,latent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--cpu-preflight',action='store_true')
    args=parser.parse_args();args.output.mkdir(exist_ok=True)
    source=args.source.resolve();sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    state=dict(status='starting',full_h3_pipeline=False,request_count=0)
    def save():(args.output/'result.json').write_text(json.dumps(state,indent=2))
    def mark(stage):state['status']=stage;save();print('VAE_BAKE_STAGE',stage,flush=True)
    stopped=threading.Event()
    def heartbeat():
        while not stopped.wait(60):print('VAE_BAKE_HEARTBEAT',state['status'],flush=True)
    threading.Thread(target=heartbeat,daemon=True).start();save()
    try:
        plan,latent=qualify(source,args.input_root)
        if args.cpu_preflight:
            import comfy.cli_args
            comfy.cli_args.args.cpu=True
            import unittest
            suite=unittest.defaultTestLoader.loadTestsFromNames(['benchmarks.test_vae_bake_experiment'])
            tests=unittest.TextTestRunner(verbosity=2).run(suite)
            assert tests.wasSuccessful()
            delivery=json.loads((source.parent/'h3_light_distribution'/'delivery_manifest.json').read_text())
            state.update(status='cpu_input_ready',tests=tests.testsRun,input_sha256=sha(latent),
                         source_capsule_sha256=delivery['capsule_sha256']);return
        import torch,importlib.metadata as md
        devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(devices)==2 and len({d['uuid'] for d in devices})==2
        assert all('T4' in d['name'] and d['capability']==[7,5] for d in devices)
        state['hardware']=dict(devices=devices,torch=torch.__version__,cuda=torch.version.cuda,kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'))
        from patches.legacy_runtime import validate_packages,validate_source
        validate_packages();validate_source(source/'ComfyUI')
        compiler=Path(shutil.which('nvcc') or '')
        assert compiler.is_file() and 'release 12.8' in subprocess.check_output([str(compiler),'--version'],text=True)
        os.environ.update(CUDACXX=str(compiler.resolve()),CUDA_HOME=str(compiler.resolve().parent.parent),CUDA_PATH=str(compiler.resolve().parent.parent),
            HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='90',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MALLOC_ARENA_MAX='2')
        from tp2_public_dit_assets import download,stage
        scratch=Path('/kaggle/temp/h3_vae_bake_20261028');scratch.mkdir(exist_ok=False)
        def run(module,argv,label,cap):
            mark(label);begin=time.monotonic()
            with (args.output/(label+'.log')).open('w') as log:
                result=subprocess.run([sys.executable,'-u','-m',module,*argv],cwd=source,stdout=log,stderr=subprocess.STDOUT,timeout=cap)
            state.setdefault('commands',[]).append(dict(stage=label,seconds=time.monotonic()-begin,exit_code=result.returncode));save()
            if result.returncode:raise RuntimeError(label+' failed; inspect retained log')
        mark('download_video_checkpoints');models={}
        for label,row in plan['vae_models'].items():models[label]=row|dict(local_path=str(download(row,scratch/'vaes')))
        manifest=args.output/'models.json';manifest.write_text(json.dumps(models,indent=2))
        run('benchmarks.run_vae_checkpoint_ab',['--latents',str(latent),'--models',str(manifest),'--output',str(args.output/'vae_ab')],'vae_checkpoint_ab',1800)
        assert json.loads((args.output/'vae_ab'/'result.json').read_text())['status']=='complete'
        run('benchmarks.probe_bake_cuda',['--output',str(args.output/'cuda_probe')],'cuda_capture_gate',240)
        mark('download_public_dit_bundle');base=stage(source,scratch/'dit')
        adapter=download(plan['lora'],scratch/'adapter')
        run('benchmarks.profile_tp2_disk_bake',['--base',str(base),'--lora',str(adapter),'--output',str(scratch/'baked'),
            '--profile',str(args.output/'bake_profile'),'--expected',str(source/'bake_expected.json')],'offline_bake_profile',1800)
        profile=json.loads((args.output/'bake_profile'/'result.json').read_text())
        state.update(status=profile['status'],quality_acceptance='Two videos await user assessment; no TE/DiT generation rerun')
    except BaseException as error:
        state.update(status='failed',error=repr(error));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:stopped.set();save()


if __name__=='__main__':main()
