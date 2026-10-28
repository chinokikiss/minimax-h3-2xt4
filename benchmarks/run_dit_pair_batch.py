"""CPU-qualify first, then one bounded bake and exactly two profiled denoisers."""
import argparse
import gzip
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-preflight',action='store_true');args=p.parse_args()
    source=args.source.resolve();args.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    sys.argv=['dit-pair','--cpu'] if args.cpu_preflight else ['dit-pair']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import torch
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(source/'ComfyUI')
    state=dict(status='starting',full_pipeline=False,denoiser_calls_requested=2)
    def save():(args.output/'batch_results.json').write_text(json.dumps(state,indent=2))
    def mark(stage):state['status']=stage;save();print('DIT_PAIR_STAGE',stage,flush=True)
    stopped=threading.Event()
    def heartbeat():
        while not stopped.wait(60):print('DIT_PAIR_HEARTBEAT',state['status'],flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def run(argv,label,cap):
        mark(label);start=time.monotonic()
        with (args.output/(label+'.log')).open('w') as log:
            process=subprocess.run(argv,cwd=source,stdout=log,stderr=subprocess.STDOUT,timeout=cap)
        state.setdefault('commands',[]).append(dict(stage=label,seconds=time.monotonic()-start,exit_code=process.returncode));save()
        if process.returncode:raise RuntimeError(label+' failed; preserved log contains the blocker')
    try:
        if args.cpu_preflight:
            import unittest
            from dit_pair_contract import fixture,identity
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(['benchmarks.test_dit_pair','benchmarks.test_dit_download']))
            assert result.wasSuccessful()
            plan=json.loads((source.parent/'h3_light_distribution/delivery_manifest.json').read_text())
            state.update(status='pair_cpu_ready',tests=result.testsRun,input_identity=identity(fixture()),
                         source_capsule_sha256=plan['capsule_sha256']);return
        import importlib.metadata as md
        devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
                      uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(devices)==2 and len({d['uuid'] for d in devices})==2 and all('T4' in d['name'] and d['capability']==[7,5] for d in devices)
        state['hardware']=dict(devices=devices,torch=torch.__version__,cuda=torch.version.cuda,kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'))
        compiler=Path(shutil.which('nvcc') or '');assert compiler.is_file()
        assert 'release 12.8' in subprocess.check_output([str(compiler),'--version'],text=True)
        os.environ.update(CUDACXX=str(compiler.resolve()),CUDA_HOME=str(compiler.resolve().parent.parent),CUDA_PATH=str(compiler.resolve().parent.parent),
            HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='90',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MALLOC_ARENA_MAX='2')
        from launch_dit_profile import find_tool
        nsys=find_tool('nsys')
        if not nsys:raise RuntimeError('Nsight Systems unavailable; stop before model download or attribution')
        run([sys.executable,'-u',str(source/'benchmarks/qualify_worker_cuda_trace.py'),'--nsys',nsys,
             '--output',str(args.output/'capture_gate'),'--checkpoint-restart'],'cuda_capture_gate',240)
        assert json.loads((args.output/'capture_gate/capture_gate.json').read_text())['passed']
        from tp2_public_dit_assets import stage,download
        scratch=Path('/kaggle/temp/h3_dit_pair_20261028');scratch.mkdir(exist_ok=False)
        mark('public_dit_download');base=stage(source,scratch/'dit')
        plan=json.loads((source/'vae_bake_plan.json').read_text())
        adapter=download(plan['lora'],scratch/'adapter')
        run([sys.executable,'-u','-m','patches.tp2_disk_bake','--base',str(base),'--lora',str(adapter),'1.0',
             '--output',str(scratch/'baked')],'offline_bake_setup',1500)
        receipt=json.loads((scratch/'baked/bake_receipt.json').read_text());expected=json.loads((source/'bake_expected.json').read_text())
        assert receipt['shards'].keys()==expected['shards'].keys() and all(v['state_sha256']==expected['shards'][k] for k,v in receipt['shards'].items())
        assert receipt['auxiliary_sha256']==expected['auxiliary_sha256'] and receipt['verified_native_targets']==208
        (args.output/'offline_bake_receipt.json').write_text(json.dumps(receipt,indent=2))
        trace=args.output/'dit_pair'
        run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--trace-fork-before-exec=true',
             '--flush-on-cudaprofilerstop=true','--wait=all','-o',str(trace),sys.executable,'-u',
             str(source/'benchmarks/profile_dit_pair.py'),'--checkpoint',str(scratch/'baked/dit_baked.json'),
             '--output',str(args.output/'profile')],'two_denoisers',2400)
        database=args.output/'dit_pair.sqlite'
        run([nsys,'export','--type=sqlite','--output',str(database),str(trace.with_suffix('.nsys-rep'))],'nsight_export',120)
        run([sys.executable,'-u',str(source/'benchmarks/analyze_dit_pair.py'),'--input',str(args.output/'profile'),
             '--sqlite',str(database)],'analysis',240)
        analysis=json.loads((args.output/'profile/analysis.json').read_text())
        state['analysis_status']=analysis['status']
        with database.open('rb') as src,gzip.open(database.with_suffix('.sqlite.gz'),'wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst,8*2**20)
        database.unlink()
        state.update(status=analysis['status'],counters='Nsight Systems timeline only; no hardware-counter/occupancy claims')
    except BaseException as error:
        state.update(status='failed',error=repr(error));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:stopped.set();save()


if __name__=='__main__':main()
