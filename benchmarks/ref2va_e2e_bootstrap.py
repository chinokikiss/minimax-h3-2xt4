"""Source-only Kaggle job, full checkpoints downloaded on Kaggle, S1 -> S2 -> L1."""
import hashlib
import importlib.metadata
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

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/kaggle/working/ref2va_e2e_results')
OUT.mkdir(exist_ok=True)
STATE=dict(status='setup',commands=[],execution_order=['S1','S2','L1'],
    numerical_scope='S1/S2 exact comparison; accepted approximate operators, no new dense Native generation',
    l1_resolution='User confirmed 864x480/362 frames takes priority over the conflicting 100k-token condition',
    profiling='Bounded CUDA events and NVTX on actual requests; CPU wall time for disk/preparation; no extra NFEs',
    timing='Downloads, service startup and explicit S2 preload outside foreground; MP4/WAV export inside foreground')
DEADLINE=time.monotonic()+28000


def save():
    (OUT/'status.json').write_text(json.dumps(STATE,indent=2))


def run(argv,label,cap):
    STATE['status']=label;save()
    start=time.monotonic()
    budget=min(cap,DEADLINE-start)
    if budget<10:raise TimeoutError('Overall budget exhausted')
    print('REF2VA_START',label,'budget_seconds',round(budget),flush=True)
    with (OUT/(label+'.log')).open('w') as log:
        proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            start_new_session=True,cwd=ROOT,text=True)
        def pump():
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith(('REF2VA_STAGE','REF2VA_LAYOUT','REF2VA_RANK_COMMAND','Traceback')):
                    print(line.rstrip(),flush=True)
        reader=threading.Thread(target=pump,daemon=True);reader.start()
        heartbeat=60
        try:
            while proc.poll() is None:
                elapsed=time.monotonic()-start
                (OUT/'progress.json').write_text(json.dumps(dict(stage=label,seconds=elapsed,
                    disk_free=shutil.disk_usage('/tmp').free,updated_ns=time.time_ns())))
                if elapsed>=heartbeat:
                    print('REF2VA_HEARTBEAT',label,'seconds',round(elapsed),flush=True)
                    heartbeat=elapsed+60
                if elapsed>budget:raise TimeoutError(label)
                time.sleep(10)
            if proc.returncode:
                reader.join(timeout=5)
                print((OUT/(label+'.log')).read_text(errors='replace')[-7000:],flush=True)
                raise RuntimeError(label+' failed, exit '+str(proc.returncode))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            reader.join(timeout=5)
            STATE['commands'].append(dict(stage=label,seconds=time.monotonic()-start,exit_code=proc.returncode));save()
    print('REF2VA_DONE',label,'seconds',round(time.monotonic()-start,2),flush=True)


def main():
    plan=json.loads((ROOT/'ref2va_e2e_plan.json').read_text())
    STATE['execution_order']=list(plan)
    if any(c.get('offline_lora') for c in plan.values()):
        STATE.update(lora_loading='offline Native merge + full-width INT8 requantization',
            numerical_scope='Independent Native offline-merged weight verification; not online-bypass equality',
            profiling='Allocator alloc/free history, static storage inventories, phase markers and representative tensor shapes; profiled latency includes instrumentation overhead')
    save()
    try:
        os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_DEBUG='WARN',
            NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4',PYTHONUNBUFFERED='1',
            HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
            MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='67108864',MINIMAX_H3_PROFILE_DETAILS='1',
            COMFY_ROOT=str(ROOT/'ComfyUI'))
        os.environ.pop('NCCL_PROTO',None)
        expected_lock=json.loads((ROOT/'runtime/local_conda.json').read_text())
        expected_digest=hashlib.sha256(json.dumps(expected_lock,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        run([sys.executable,str(ROOT/'benchmarks/frozen_kaggle_setup.py'),
             '--root',str(ROOT),'--offline-install','--expected-manifest-sha256',expected_digest],'frozen_user_runtime',300)
        STATE['runtime_lock']=json.loads((ROOT/'runtime/setup_receipt.json').read_text())
        compiler=Path(shutil.which('nvcc')).resolve()
        version=subprocess.check_output([str(compiler),'--version'],text=True)
        if 'release 12.8' not in version:raise RuntimeError('CUDA12.8 compiler required for validated SM75 kernels')
        os.environ.update(CUDACXX=str(compiler),CUDA_HOME=str(compiler.parent.parent),CUDA_PATH=str(compiler.parent.parent))
        import torch
        STATE['hardware']=[dict(name=torch.cuda.get_device_name(i),sm=list(torch.cuda.get_device_capability(i)),
            bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]
        assert len(STATE['hardware'])==2 and all(d['sm']==[7,5] and 'T4' in d['name'] for d in STATE['hardware'])
        STATE['packages']={k:importlib.metadata.version(k) for k in ('torch','comfy-kitchen','tilelang','transformers','av')}
        STATE['compiler']=dict(path=str(compiler),version=version)
        save()
        import requests
        assets=Path('/tmp/ref2va_e2e_assets');assets.mkdir(exist_ok=True)
        images=[]
        for name in ('red_superboy_on_city_roof.png','mecha_dragon_lightning.png'):
            response=requests.get('https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/'+name,timeout=90)
            response.raise_for_status();path=assets/name;path.write_bytes(response.content);images.append(str(path))
        model_dir=Path('/tmp/ref2va_e2e_models');model_dir.mkdir(exist_ok=True)
        specs=json.loads((ROOT/'semantic_results/pipeline_checkpoint_identity.json').read_text())
        required=sum(s['bytes'] for s in specs.values())+6*2**30
        if any(c.get('offline_lora') for c in plan.values()):required+=specs['dit']['bytes']+3*2**30
        if shutil.disk_usage('/tmp').free<required:raise RuntimeError('Insufficient local disk headroom')
        from huggingface_hub import hf_hub_download
        from concurrent.futures import ThreadPoolExecutor
        paths={}
        STATE.update(status='checkpoint_downloads');save()
        def download(item):
            kind,spec=item;begin=time.monotonic()
            path=Path(hf_hub_download('Comfy-Org/MiniMax-H3',spec['remote_file'],
                revision='e5eb578a89295337b8ff433a035929ce0279e0b6',cache_dir=str(model_dir),token=False))
            with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
            if path.stat().st_size!=spec['bytes'] or sha!=spec['sha256']:raise RuntimeError('Checkpoint identity mismatch '+kind)
            print('REF2VA_CHECKPOINT_VERIFIED',kind,flush=True)
            return kind,path,dict(bytes=path.stat().st_size,sha256=sha,seconds=time.monotonic()-begin)
        with ThreadPoolExecutor(max_workers=2) as pool:
          for kind,path,receipt in pool.map(download,specs.items()):
            paths[kind]=str(path)
            STATE.setdefault('downloads',{})[kind]=receipt;save()
        for name in plan:
            # Equal cold kernel-cache state for the S1/S2 comparison. Reusing
            # S1's compiled TileLang cache would confound the idle-residency A/B.
            kernel_cache=Path('/tmp/ref2va_kernel_caches')/name
            driver_cache=Path('/tmp/ref2va_driver_caches')/name
            kernel_cache.mkdir(parents=True,exist_ok=True);driver_cache.mkdir(parents=True,exist_ok=True)
            os.environ.update(TILELANG_CACHE_DIR=str(kernel_cache),CUDA_CACHE_PATH=str(driver_cache))
            if name=='L1':
                STATE['status']='download_lora';save()
                paths['lora']=hf_hub_download('ZhengmingYu/DMAD',
                    'minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',
                    revision='439136a1dd4f435b8137a084f73d558d2cecda2a',cache_dir=str(model_dir),token=False)
            if plan[name].get('offline_lora'):
                merged=Path('/tmp/h3_l1_offline_merged.safetensors')
                run([sys.executable,'-u',str(ROOT/'benchmarks/merge_h3_lora.py'),
                    '--base',paths['dit'],'--lora',paths['lora'],'1.0','--output',str(merged),
                    '--verify-native'],'offline_merge_native_verified',2400)
                receipt=json.loads(merged.with_suffix('.safetensors.json').read_text())
                if not receipt.get('native_independent_verification') or receipt['verified_native_targets']<200:
                    raise RuntimeError('Full Native offline verification is incomplete')
                (OUT/'offline_merge_receipt.json').write_text(json.dumps(receipt,indent=2))
                STATE['offline_merge']=dict(targets=receipt['verified_native_targets'],
                    original_base=paths['dit'],merged_bytes=merged.stat().st_size,
                    scope='Setup cost outside request latency; base checkpoint unmodified')
                paths['dit']=str(merged);save()
                run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    str(ROOT/'benchmarks/verify_offline_tp_checkpoint.py'),'--checkpoint',str(merged),
                    '--output',str(OUT/'offline_tp_native_gate.json')],'offline_tp_native_gate',300)
            if plan[name].get('memory_profile'):
                os.environ.update(MINIMAX_H3_GPU_MEMORY_PROFILE='1',
                    MINIMAX_H3_MEMORY_DIR=str(OUT/name/'gpu_memory'))
            # Targeted advisory eviction only; no system-wide cache drop and no
            # model deletion. This is not proof of cold backing storage.
            for path in paths.values():
                with open(path,'rb') as stream:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            config=plan[name] | dict(paths=paths.copy(),output=str(OUT/name),
                images=images[:2 if name=='L1' else 1],
                coldness='Fresh process/service; per-checkpoint POSIX_FADV_DONTNEED advisory, not verified physical cold storage')
            target=ROOT/(name+'.json');target.write_text(json.dumps(config))
            try:
                run([sys.executable,'-u',str(ROOT/'benchmarks/ref2va_e2e_run.py'),'--config',str(target)],name,10800)
            finally:
                if plan[name].get('memory_profile'):
                    run([sys.executable,str(ROOT/'benchmarks/analyze_gpu_memory_overhead.py'),
                        '--directory',str(OUT/name/'gpu_memory')],'memory_analysis',180)
            result=json.loads((OUT/name/'result.json').read_text())
            if result['status']!='complete':raise RuntimeError(name+' did not complete')
            if name=='S2':
                baseline=json.loads((OUT/'S1/result.json').read_text())
                checks={key:baseline['outputs'][key]==result['outputs'][key] for key in baseline['outputs']}
                (OUT/'s1_s2_correctness.json').write_text(json.dumps(checks,indent=2))
                if not all(checks.values()):raise AssertionError('Idle residency changed outputs; stop before L1')
            STATE.setdefault('completed',[]).append(name);save()
        STATE['status']='complete';save()
    except BaseException as error:
        STATE.update(status='failed',error_type=type(error).__name__)
        (OUT/'exception.txt').write_text(traceback.format_exc());save()
        raise


if __name__=='__main__':main()
