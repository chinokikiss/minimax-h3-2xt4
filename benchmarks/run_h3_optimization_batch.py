"""One CPU gate or a bounded T4 batch; no complete H3 generation or API polling."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback


def cases():
    from case_cu128_dispatch import cases as dispatch_cases
    rows=[]
    for case in dispatch_cases():
        if case['label'].startswith('vae_') and case['kind']=='linear' or case['kind']=='dequant':
            rows.append(dict(case,pipeline_default=True,label='default_'+case['label']))
        elif case['kind']=='conv':rows.append(case)
        elif case['kind']=='normpad' or case['label']=='vae_rope_partial':
            rows.append(dict(case,pipeline_default=True,legacy_dispatch=True,label='default_'+case['label']))
    for n in (1,33,261,4097,51220,59134):rows.append(dict(kind='rope',label=f'fp32_rope_{n}',rows=n))
    rows.extend((dict(kind='rope',label='fp32_rope_stress_partial',rows=33,stress=True,rot=96),
                 dict(kind='rope',label='fp32_rope_strict',rows=261,dtype='float32')))
    for stage,n in (('te',2392),('dit',59134)):
        for rank in (0,1):rows.append(dict(kind='aot',label=f'aot_{stage}_rank{rank}',stage=stage,rows=n,rank=rank))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-preflight',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=args.source.resolve();sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    sys.argv=['h3-optimization-batch','--cpu'] if args.cpu_preflight else ['h3-optimization-batch']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(source/'ComfyUI')
    manifest=json.loads((source.parent/'h3_light_distribution/delivery_manifest.json').read_text())
    state=dict(status='starting',capsule_sha256=manifest['capsule_sha256'],full_pipeline=False,
        micro_cases=[],case_count=len(cases()),model_scope='Six selected trained layers for bake AB/BA; no TE/DiT generation',
        private_inputs=False)
    lock=threading.Lock()
    def save():(args.output/'result.json').write_text(json.dumps(state,indent=2))
    try:
        if args.cpu_preflight:
            import unittest
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(
                ['benchmarks.test_h3_optimization_batch','benchmarks.test_cu128_candidate']))
            assert result.wasSuccessful()
            import patches.h3_operator_aot,patches.h3_vae_operators,patches.tilelang_fp32_rms_rope
            import profile_bake_pipeline
            state.update(status='optimization_cpu_ready',tests=result.testsRun);return
        import torch,importlib.metadata as md
        devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
            uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(devices)==2 and len({d['uuid'] for d in devices})==2 and all('T4' in d['name'] and d['capability']==[7,5] for d in devices)
        assert torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8'
        state['hardware']=dict(devices=devices,torch=str(torch.__version__),cuda=torch.version.cuda,
            kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'))
        compiler=Path(shutil.which('nvcc') or '').resolve();assert compiler.is_file()
        version=subprocess.check_output([str(compiler),'--version'],text=True,timeout=10)
        assert 'release 12.8' in version
        os.environ.update(CUDACXX=str(compiler),CUDA_HOME=str(compiler.parent.parent),CUDA_PATH=str(compiler.parent.parent),
            HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='90',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MALLOC_ARENA_MAX='2')
        state.update(status='isolated_operator_cases',compiler=version);save()
        def worker(device,items):
            for case in items:
                out=args.output/case['label'];out.mkdir();path=out/'case.json';path.write_text(json.dumps(case))
                script='case_h3_optimization.py' if case['kind'] in ('rope','aot') and not case.get('legacy_dispatch') else 'case_cu128_dispatch.py'
                start=time.monotonic()
                with (out/'process.log').open('w') as log:
                    try:
                        process=subprocess.run([sys.executable,'-u',str(source/'benchmarks'/script),'--source',str(source),
                            '--case',str(path),'--device',str(device),'--output',str(out)],cwd=source,stdout=log,
                            stderr=subprocess.STDOUT,timeout=480 if case['kind']=='aot' else 180)
                        code=process.returncode
                    except subprocess.TimeoutExpired:code=-1
                value=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else dict(status='no_result')
                with lock:
                    state['micro_cases'].append(dict(label=case['label'],device=device,exit_code=code,
                        seconds=time.monotonic()-start,result=value));save()
                print('H3_BATCH_OPERATOR',case['label'],code,value['status'],flush=True)
        work=cases()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(worker,0,work[::2]);b=pool.submit(worker,1,work[1::2]);a.result();b.result()
        state['status']='public_layer_download';save()
        from tp2_public_dit_assets import stage,download
        scratch=Path('/kaggle/temp/h3_optimization_batch_20261028');scratch.mkdir(exist_ok=False)
        base=stage(source,scratch/'dit',layers=(0,1,24,25,48,49))
        plan=json.loads((source/'vae_bake_plan.json').read_text());adapter=download(plan['lora'],scratch/'adapter')
        state['status']='bake_ab_ba';save();start=time.monotonic()
        with (args.output/'bake_batch.log').open('w') as log:
            process=subprocess.run([sys.executable,'-u',str(source/'benchmarks/profile_bake_pipeline.py'),
                '--base',str(base),'--lora',str(adapter),'--scratch',str(scratch/'bake_cases'),
                '--output',str(args.output/'bake'),'--expected',str(source/'bake_expected.json')],
                cwd=source,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
        state['bake_seconds_including_process_startup']=time.monotonic()-start
        state['bake']=json.loads((args.output/'bake/result.json').read_text()) if (args.output/'bake/result.json').exists() else dict(status='no_result')
        state['status']='complete' if process.returncode==0 and all(c['exit_code']==0 for c in state['micro_cases']) else 'operator_blockers_found'
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:save()


if __name__=='__main__':main()
