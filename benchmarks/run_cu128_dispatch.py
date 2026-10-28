"""CPU contract gate or bounded isolated cu128 operator cases; no model I/O."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-preflight',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=args.source.resolve();sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    sys.argv=['cu128-dispatch','--cpu'] if args.cpu_preflight else ['cu128-dispatch']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import torch
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(source/'ComfyUI')
    from case_cu128_dispatch import cases
    state=dict(status='starting',full_pipeline=False,model_downloads=0,fixture='Synthetic per-operator inputs at actual TP2/VAE shapes; not generation quality validation')
    def save():(args.output/'result.json').write_text(json.dumps(state,indent=2))
    try:
        manifest=json.loads((source.parent/'h3_light_distribution/delivery_manifest.json').read_text())
        state['capsule_sha256']=manifest['capsule_sha256']
        if args.cpu_preflight:
            import unittest
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('benchmarks.test_cu128_candidate'))
            assert result.wasSuccessful()
            import comfy_kitchen as ck
            from patches.kitchen_cu128_candidate import OPERATIONS
            assert all(hasattr(ck.backends.cuda,name) and callable(getattr(ck,name)) for name in OPERATIONS)
            state.update(status='cu128_dispatch_cpu_ready',tests=result.testsRun,cases=len(cases()));return
        devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(devices)==2 and len({d['uuid'] for d in devices})==2 and all('T4' in d['name'] and d['capability']==[7,5] for d in devices)
        assert torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8'
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).splitlines()
        assert all(int(v.split('.')[0])>=580 for v in driver),'Older drivers are outside this current-Kaggle qualification'
        compiler=Path(__import__('shutil').which('nvcc') or '').resolve()
        assert compiler.is_file() and 'release 12.8' in subprocess.check_output([str(compiler),'--version'],text=True,timeout=10)
        os.environ.update(CUDACXX=str(compiler),CUDA_HOME=str(compiler.parent.parent),CUDA_PATH=str(compiler.parent.parent))
        state.update(status='physical_cases',hardware=dict(devices=devices,driver=driver,torch=torch.__version__,cuda=torch.version.cuda));save()
        def worker(device,items):
            rows=[]
            for case in items:
                out=args.output/case['label'];out.mkdir()
                path=out/'case.json';path.write_text(json.dumps(case));start=time.monotonic()
                with (out/'process.log').open('w') as log:
                    try:
                        result=subprocess.run([sys.executable,'-u',str(source/'benchmarks/case_cu128_dispatch.py'),
                            '--source',str(source),'--case',str(path),'--device',str(device),'--output',str(out)],
                            cwd=source,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                        code=result.returncode
                    except subprocess.TimeoutExpired:code=-1
                value=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else dict(status='worker_failed_without_result')
                rows.append(dict(label=case['label'],device=device,exit_code=code,seconds=time.monotonic()-start,result=value))
                print('CU128_OPERATOR_CASE',case['label'],code,value['status'],flush=True)
            return rows
        work=cases()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(worker,0,work[::2]);b=pool.submit(worker,1,work[1::2])
            state['cases']=a.result()+b.result()
        state.update(status='complete' if all(r['exit_code']==0 for r in state['cases']) else 'operator_blockers_found',
                     acceptance='Dispatch/kernel/error evidence only. No pipeline precision promotion or global CUDA enable.')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:save()


if __name__=='__main__':main()
