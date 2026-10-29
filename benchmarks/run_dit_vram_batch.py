"""Fresh CPU gate / bounded SM75 operator batch, with isolated CUDA workers."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import unittest

SECTIONS=('norm_gate','column','row_epilogue','attention','block')
FILES=('dit_vram_kernels.py','dit_vram_candidates.py','run_dit_vram_case.py','run_dit_vram_batch.py','test_dit_vram_candidates.py')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cpu-preflight',action='store_true')
    args=parser.parse_args();sys.argv=sys.argv[:1];source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    import comfy.cli_args
    if args.cpu_preflight:comfy.cli_args.args.cpu=True
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(source/'ComfyUI')
    state=dict(status='starting',full_pipeline=False,no_private_inputs=True,no_lora_download=True,
               source_hashes={name:hashlib.sha256((source/'benchmarks'/name).read_bytes()).hexdigest() for name in FILES})
    manifest=json.loads((source.parent/'h3_light_distribution/delivery_manifest.json').read_text())
    state['capsule_sha256']=manifest['capsule_sha256']
    def save():(out/'result.json').write_text(json.dumps(state,indent=2))
    try:
        import comfy_kitchen as ck
        state['kitchen_split_api']={name:str(inspect.signature(getattr(ck,name))) if callable(getattr(ck,name,None)) else None
            for name in ('prequantize_int8_attention','int8_attention_from_prequantized')}
        assets=json.loads((source/'bundle_assets.json').read_text())
        assert assets['publication']['commit']=='d3e6a4b131056d2c626a01acf6ba48123a885783'
        selected={name:row for name,row in assets['objects'].items() if name in ('dit.block00.rank0.safetensors','dit.block00.rank1.safetensors')}
        assert len(selected)==2
        state['trained_weight_inputs']=selected
        if args.cpu_preflight:
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('benchmarks.test_dit_vram_candidates'))
            assert result.wasSuccessful() and result.testsRun==4
            state.update(status='vram_cpu_ready',tests=result.testsRun,gpu_used=False,weights_fetched=False)
            return
        import torch
        torch.set_grad_enabled(False)
        devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
                      uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        if len(devices)!=2 or len({d['uuid'] for d in devices})!=2 or any('T4' not in d['name'] or d['capability']!=[7,5] for d in devices):
            raise RuntimeError('Two distinct physical T4 devices required')
        compiler=Path(shutil.which('nvcc') or '')
        compiler_version=subprocess.check_output([str(compiler),'--version'],text=True,timeout=10) if compiler.is_file() else ''
        if torch.version.cuda!='12.8' or 'release 12.8' not in compiler_version:
            raise RuntimeError('This cu128 experiment requires native CUDA12.8 and its existing compiler; no upgrade')
        state.update(hardware=dict(devices=devices,torch=str(torch.__version__),cuda=torch.version.cuda),compiler=compiler_version,
            status='download_two_trained_shards',commands=[],budget_seconds=1800)
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            NCCL_P2P_DISABLE='1',NCCL_DEBUG='WARN',NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4',
            CUDACXX=str(compiler.resolve()),CUDA_HOME=str(compiler.resolve().parent.parent),
            TILELANG_CACHE_DIR='/kaggle/temp/h3_dit_vram_cache',HF_HUB_DISABLE_XET='1',
            HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='90')
        scratch=Path('/tmp/h3_dit_vram_weights_20261029');scratch.mkdir(exist_ok=False)
        if shutil.disk_usage(scratch).free<3*2**30:raise OSError('3 GiB scratch required for this bounded operator test')
        from benchmarks.tp2_public_dit_assets import download
        def fetch(item):
            name,row=item
            return download(dict(file=name,repository=assets['publication']['repo'],revision=assets['publication']['commit'],**row),scratch)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(fetch,selected.items()))
        state['weights_downloaded_bytes']=sum(row['bytes'] for row in selected.values());save()
        # Traces are worker-side and validated against a minimal CUDA kernel.
        # Tool availability is recorded without installing profilers or probing
        # protected NCU counters again. Timing runs have no active profiler.
        state['profilers']={name:shutil.which(name) for name in ('nsys','ncu')}
        state['profilers']['active']='torch.profiler with per-process CUDA capture gate'
        deadline=time.monotonic()+1680
        for section,cap in zip(SECTIONS,(240,300,240,300,480)):
            remaining=min(cap,deadline-time.monotonic())
            if remaining<30:
                state['commands'].append(dict(section=section,status='budget_skipped'));continue
            state['status']=section;save();print('DIT_VRAM_STAGE',section,flush=True)
            section_out=out/section;section_out.mkdir()
            argv=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                  str(source/'benchmarks/run_dit_vram_case.py'),'--source',str(source),'--weights',str(scratch),
                  '--output',str(section_out),'--section',section]
            if section=='block':
                excluded={}
                def valid(name,predicate):
                    try:
                        values=[json.loads((out/name/f'rank{r}/result.json').read_text()) for r in (0,1)]
                        return all(v['status']=='complete' and predicate(v) for v in values)
                    except (OSError,ValueError,KeyError):return False
                if not valid('column',lambda v:v['exact_qualified'] and v['signed_extreme_int32_exact']):
                    excluded.update({n:'Prepared/chunked column byte qualification failed' for n in ('ownership','fc1_4k','fc1_8k','fc1_16k','combined_8k')})
                if not valid('norm_gate',lambda v:v['gate_error']['byte_exact']):
                    excluded.update({n:'Norm/gate qualification failed' for n in ('norm_gate','combined_8k')})
                if not valid('row_epilogue',lambda v:v['error']['byte_exact']):excluded['combined_8k']='Row epilogue byte qualification failed'
                if any(state['kitchen_split_api'].values()) and not valid('attention',lambda v:v['error']['byte_exact']):
                    excluded.update({n:'Split attention qualification failed' for n in ('ownership','combined_8k')})
                eligibility=out/'excluded_block_variants.json';eligibility.write_text(json.dumps(excluded,indent=2))
                argv+=['--eligibility',str(eligibility)]
            start=time.monotonic()
            with (section_out/'worker.log').open('w') as log:
                process=subprocess.Popen(argv,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                    code=-9
            state['commands'].append(dict(section=section,exit_code=code,seconds=time.monotonic()-start));save()
            print('DIT_VRAM_SECTION_END',section,code,flush=True)
        state['status']='complete' if len(state['commands'])==len(SECTIONS) and all(c.get('exit_code')==0 for c in state['commands']) else 'partial_or_failed'
    except BaseException as error:
        state.update(status='failed',error=repr(error));(out/'exception.txt').write_text(traceback.format_exc());raise
    finally:save();print('DIT_VRAM_FINAL',state['status'],flush=True)


if __name__=='__main__':main()
