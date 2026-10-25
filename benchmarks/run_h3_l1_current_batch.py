"""Pinned model downloads, dual-GPU offline bake, then one current-infra L1 run.

Environment setup is completed by a separate 600-second supervised phase.
Only this invocation's TemporaryDirectory download cache is released; user
checkpoints, shared caches and historical experiment artifacts are untouched.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def profile_modes(requested):
    if requested=='tilelang_scalar':return ('tilelang_scalar','kitchen')
    if requested=='kitchen':return ('kitchen',)
    raise ValueError('Unsupported profile backend')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dit-profile',action='store_true',help='Real input preparation, then exactly one profiled DiT NFE; no decoders')
    parser.add_argument('--dit-attention',choices=('kitchen','tilelang_scalar'),default='kitchen',
        help='Explicit single-call profiling backend; E2E keeps its frozen plan')
    parser.add_argument('--capture-gate',type=Path)
    args=parser.parse_args()
    if not args.dit_profile and args.dit_attention!='kitchen':parser.error('--dit-attention requires --dit-profile')
    if args.dit_profile and (not args.capture_gate or not json.loads(args.capture_gate.read_text()).get('passed')):
        parser.error('Passed minimal worker CUDA capture required before model downloads')
    source=args.source;output=args.output;output.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',commands=[],downloads={},request_count=1)
    stop=threading.Event();batch_started=time.monotonic()
    def heartbeat():
        while not stop.wait(60):
            print('L1_BATCH_HEARTBEAT',state['status'],round(time.monotonic()-batch_started),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def save():(output/'batch_results.json').write_text(json.dumps(state,indent=2))
    def mark(stage,**extra):
        state.update(status=stage,**extra);save();print('L1_BATCH_STAGE',stage,flush=True)
    def run(argv,label,cap):
        mark(label);started=time.monotonic()
        with (output/(label+'.log')).open('w') as log:
            process=subprocess.Popen(argv,cwd=source,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                start_new_session=True,text=True,bufsize=1)
            def relay():
                for line in process.stdout:
                    log.write(line);log.flush()
                    if line.startswith(('L1_','REF2VA_','MERGED','Traceback')):print(line,end='',flush=True)
            reader=threading.Thread(target=relay,daemon=True);reader.start();heartbeat=60
            try:
                while process.poll() is None:
                    elapsed=time.monotonic()-started
                    if elapsed>cap:raise TimeoutError(label)
                    if elapsed>=heartbeat:
                        print('L1_HEARTBEAT',label,round(elapsed),flush=True);heartbeat=elapsed+60
                    time.sleep(2)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                reader.join(timeout=5)
                state['commands'].append(dict(stage=label,seconds=time.monotonic()-started,exit_code=process.returncode));save()
            if process.returncode:
                print((output/(label+'.log')).read_text(errors='replace')[-5000:],flush=True)
                raise RuntimeError(label+' failed: '+str(process.returncode))
    try:
        plan=json.loads((source/'l1_current_plan.json').read_text())
        identity=json.loads((source/'l1_model_identity.json').read_text())
        assets=json.loads((source/'l1_assets_identity.json').read_text())
        os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='90',
            MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='67108864',PYTHONUNBUFFERED='1')
        scratch=Path(tempfile.mkdtemp(prefix='h3_l1_current_',dir='/kaggle/temp'))
        weights=scratch/'weights';weights.mkdir()
        mark('verify_reference_images')
        images=[]
        for record in assets['images']:
            path=source/'bench_assets/l1_current'/record['file']
            if not path.exists():
                import requests
                response=requests.get('https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/'+record['file'],timeout=60)
                response.raise_for_status()
                if len(response.content)!=record['bytes'] or hashlib.sha256(response.content).hexdigest()!=record['sha256']:
                    raise RuntimeError('Reference image download identity changed')
                path=scratch/record['file'];path.write_bytes(response.content)
            if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise RuntimeError('Reference image identity changed')
            images.append(str(path))
        state['reference_images']=assets['images'];save()
        from huggingface_hub import hf_hub_download
        def download(kind,spec,cache,repository='Comfy-Org/MiniMax-H3',revision='e5eb578a89295337b8ff433a035929ce0279e0b6'):
            print('L1_DOWNLOAD_START',kind,spec['bytes'],flush=True);started=time.monotonic()
            path=Path(hf_hub_download(repository,spec['remote_file'],revision=revision,cache_dir=str(cache),token=False))
            digest=sha(path)
            if path.stat().st_size!=spec['bytes'] or digest!=spec['sha256']:raise RuntimeError('Model identity mismatch: '+kind)
            return path,dict(bytes=path.stat().st_size,sha256=digest,revision=revision,seconds=time.monotonic()-started)
        if args.dit_profile and (source/'bundle_assets.json').exists():
            from prepare_tp2_bundle_assets import prepare
            mark('reuse_base_tp2_bundle')
            paths,record=prepare(source,scratch,output);state['bundle']=record;save()
            # Adapters remain separate inputs. The Dataset/public bundle is
            # always unmerged and never contains DMAD or a replacement LoRA.
            selected=plan.get('profile_loras',[dict(assets['lora'],strength=1.)]);stack=[]
            for i,spec in enumerate(selected):
                path,row=download('lora_'+str(i),spec,scratch/'adapter_downloads',spec['repository'],spec['revision'])
                stack.append((str(path),spec['strength']));state['downloads']['lora_'+str(i)]=row;save()
            path,row=download('video',identity['video'],scratch/'video_downloads')
            paths['video']=str(path);state['downloads']['video']=row;save()
            config=plan|dict(paths=paths,images=images,output=str(output/'L1'),lora_stack=stack,
                bundle_offline_lora=True,operators={'dit_attention':args.dit_attention},
                source_note='Immutable unmerged TP2 bundle; bounded CPU Native LoRA preparation; paired one-call controls')
            config_path=output/'request_config.json';config_path.write_text(json.dumps(config,indent=2))
            private_inputs=scratch/'single_dit_inputs.pt';profile=output/'dit_profile'
            try:
                run([sys.executable,'-u',str(source/'benchmarks/profile_dit_single.py'),
                    '--config',str(config_path),'--inputs',str(private_inputs),'--output',str(profile),
                    '--prepare-only'],'real_input_preparation',600)
                modes=profile_modes(args.dit_attention)
                comparison={}
                for mode in modes:
                    target=profile if mode==args.dit_attention else output/'kitchen_control'
                    target.mkdir(parents=True,exist_ok=True)
                    if target!=profile:shutil.copyfile(profile/'preparation.json',target/'preparation.json')
                    config['operators']['dit_attention']=mode
                    case_config=output/(mode+'_config.json');case_config.write_text(json.dumps(config,indent=2))
                    try:
                        run([sys.executable,'-u',str(source/'benchmarks/launch_dit_profile.py'),
                            '--config',str(case_config),'--inputs',str(private_inputs),
                            '--capture-gate',str(args.capture_gate),'--output',str(target)],mode+'_profile',2000)
                        result=json.loads((target/'result.json').read_text())
                        if result['status']!='complete':raise RuntimeError('Incomplete paired profile: '+mode)
                        comparison[mode]=dict(status=result['status'],sample_wrapper_seconds=result['sample_wrapper_seconds'],
                            inputs=result['inputs'],result_path=str(target/'result.json'),calls=[r['calls'] for r in result['ranks']])
                    except Exception as error:
                        # Preserve one failed case and try the other backend once.
                        # There is no automatic retry of a denoiser call.
                        comparison[mode]=dict(status='failed',error=repr(error),result_path=str(target/'result.json'))
                    (output/'comparison.json').write_text(json.dumps(comparison,indent=2))
                if any(r['status']!='complete' for r in comparison.values()):raise RuntimeError('At least one backend profile failed; see comparison.json')
                if len(comparison)==2 and comparison['kitchen']['inputs']!=comparison['tilelang_scalar']['inputs']:
                    raise RuntimeError('Paired backend inputs changed')
                mark('complete',denoiser_calls=len(modes),calls_per_backend=1,full_pipeline=False,
                    decoders_executed=False,persistent_baked_bytes=0,profiles=list(comparison))
            finally:private_inputs.unlink(missing_ok=True)
            return
        # Stage models to avoid holding the original and baked DiT alongside
        # the 27 GB TE download at the same time.
        peak=max(2*identity['dit']['bytes']+assets['lora']['bytes'],
            sum(spec['bytes'] for spec in identity.values()))+4*2**30
        if shutil.disk_usage(scratch).free<peak:raise OSError('Insufficient scratch for staged offline baking: '+str(peak))
        mark('download_base_and_lora',scratch_required_bytes=peak)
        with tempfile.TemporaryDirectory(prefix='owned_bake_downloads_',dir=scratch) as download_cache:
            base,base_receipt=download('dit',identity['dit'],download_cache)
            lora_spec=assets['lora'];lora,lora_receipt=download('lora',lora_spec,download_cache,lora_spec['repository'],lora_spec['revision'])
            state['downloads'].update(dit=base_receipt,lora=lora_receipt);save()
            merged=weights/'h3_l1_offline_merged.safetensors'
            run([sys.executable,'-u','-m','patches.offline_lora','--base',str(base),'--lora',str(lora),'1.0',
                '--output',str(merged),'--verify-native','--devices','cuda:0','cuda:1'],'offline_bake',1800)
            receipt=json.loads(merged.with_suffix('.safetensors.json').read_text())
            if (not receipt.get('native_independent_verification') or receipt['verified_native_targets']!=208
                    or receipt['devices']!=['cuda:0','cuda:1']):raise RuntimeError('Incomplete dual-GPU offline bake verification')
            if sha(base)!=identity['dit']['sha256']:raise RuntimeError('Original scratch base changed during baking')
            (output/'offline_merge_receipt.json').write_text(json.dumps(receipt,indent=2))
            state['baked_checkpoint']=dict(bytes=merged.stat().st_size,sha256=sha(merged),verified_native_targets=208,
                scope='Native FP32 sequential offline merge and full-width ConvRot requantization; not online-LoRA equality')
        state['owned_original_download_cache_released']=not Path(download_cache).exists();save()
        mark('download_te_and_vaes')
        paths={'dit':str(merged)}
        def remaining(kind):
            path,record=download(kind,identity[kind],scratch/'inference_downloads');return kind,path,record
        with ThreadPoolExecutor(max_workers=2) as pool:
            for kind,path,record in pool.map(remaining,('te','video') if args.dit_profile else ('te','video','audio')):
                paths[kind]=str(path);state['downloads'][kind]=record;save()
        config=plan|dict(paths=paths,images=images,output=str(output/'L1'))
        if args.dit_profile:
            config.update(operators={'dit_attention':args.dit_attention},
                source_note='One DiT call with explicit attention backend; same real L1 conditioning/reference/noise')
        (output/'request_config.json').write_text(json.dumps(config,indent=2))
        if args.dit_profile:
            private_inputs=scratch/'single_dit_inputs.pt';profile=output/'dit_profile'
            try:
                run([sys.executable,'-u',str(source/'benchmarks/profile_dit_single.py'),
                    '--config',str(output/'request_config.json'),'--inputs',str(private_inputs),
                    '--output',str(profile),'--prepare-only'],'real_input_preparation',450)
                run([sys.executable,'-u',str(source/'benchmarks/launch_dit_profile.py'),
                    '--config',str(output/'request_config.json'),'--inputs',str(private_inputs),
                    '--capture-gate',str(args.capture_gate),'--output',str(profile)],'single_dit_profile',2600)
                result=json.loads((profile/'result.json').read_text())
                if result['status']!='complete':raise RuntimeError('Single DiT profile incomplete')
                mark('complete',denoiser_calls=1,full_pipeline=False,decoders_executed=False)
            finally:
                private_inputs.unlink(missing_ok=True)
            return
        run([sys.executable,'-u',str(source/'benchmarks/run_h3_l1_current_e2e.py'),
            '--config',str(output/'request_config.json')],'L1_generation',2400)
        result=json.loads((output/'L1/result.json').read_text())
        if result['status']!='complete':raise RuntimeError('Long-sequence request incomplete')
        mark('complete',generation_quality_acceptance='Pending visual/audio review')
    except BaseException as error:
        state.update(status='failed',error=repr(error));(output/'exception.txt').write_text(traceback.format_exc());save();raise
    finally:stop.set()


if __name__=='__main__':main()
