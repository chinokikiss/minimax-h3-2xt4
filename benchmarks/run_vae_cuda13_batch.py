"""Bounded CUDA13 VAE attribution after its separate CPU qualification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=args.source.resolve();sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    state=dict(status='starting',full_pipeline=False,dit_calls=0)
    def save():(args.output/'result.json').write_text(json.dumps(state,indent=2))
    def run(label,extra,cap):
        state['status']=label;save();start=time.monotonic();print('VAE_CU13_STAGE',label,flush=True)
        with (args.output/(label+'.log')).open('w') as log:
            process=subprocess.run([sys.executable,'-u',str(source/'benchmarks/run_vae_dispatch_ablation.py'),
                '--source',str(source),'--output',str(args.output/label),*extra],cwd=source,
                stdout=log,stderr=subprocess.STDOUT,timeout=cap)
        state.setdefault('commands',[]).append(dict(stage=label,exit_code=process.returncode,seconds=time.monotonic()-start));save()
        if process.returncode:raise RuntimeError(label+' failed; inspect retained log')
    try:
        # Gate failure stops before 8 GB of model download or any real decode.
        run('physical_gate',['--gate-only'],180)
        from run_vae_bake_batch import qualify
        plan,latents=qualify(source,Path('/kaggle/input'))
        from tp2_public_dit_assets import download
        scratch=Path('/kaggle/temp/h3_vae_cu13_weights');scratch.mkdir(exist_ok=False)
        models={}
        os.environ.update(HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='90')
        state['status']='immutable_video_weights';save()
        for label,row in plan['vae_models'].items():
            print('VAE_CU13_WEIGHT',label,flush=True)
            models[label]=row|dict(local_path=str(download(row,scratch)))
        manifest=args.output/'models.json';manifest.write_text(json.dumps(models,indent=2))
        run('block_factorial',['--latents',str(latents),'--models',str(manifest)],1200)
        result=json.loads((args.output/'block_factorial/result.json').read_text())
        assert result['status']=='complete'
        state.update(status='complete',cases=len(result['cases']),scope='Video decoder real-activation block replay only; no full video')
    except BaseException as error:
        state.update(status='failed',error=repr(error));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:save()


if __name__=='__main__':main()
