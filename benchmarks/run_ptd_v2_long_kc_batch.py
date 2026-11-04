"""Full real PTD V2 eight-hop KC E2E: mounted baked experts; stage only TE and VAEs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--proofs', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=7800)
    args=p.parse_args()
    if not 1200 <= args.timeout <= 10800:raise ValueError('Bounded timeout required')
    source=args.source.resolve()
    sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    sys.argv=sys.argv[:1]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',full_pipeline=True,trained_hops=8,
        no_online_lora=True,ptd_source='nine_verified_mounts',
        ptd_io_workers_per_rank=1,ptd_payload_rehash=False,
        ptd_http_staging=False,ptd_http_range2_hostcache=os.getenv('H3_PTD_HTTP_RANGE2_HOSTCACHE')=='1',
        ptd_http_range2_debug=os.getenv('H3_PTD_HTTP_RANGE2_DEBUG')=='1',
        local_ptd_copy=False,commands=[],stages={})
    def save(): (output/'batch_results.json').write_text(json.dumps(state,indent=2,default=str))
    def mark(stage):
        state['status']=stage;save();print('PTD_V2_E2E_STAGE',stage,flush=True)
    def run(argv,stage,timeout):
        mark(stage);started=time.monotonic()
        with (output/(stage+'.log')).open('w',encoding='utf-8') as f:
            process=subprocess.Popen(argv,cwd=source,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,start_new_session=True,bufsize=1)
            def relay():
                for line in process.stdout:
                    f.write(line);f.flush()
                    if line.startswith(('PTD_','TP2_','L1_','Traceback','REF2VA_')):
                        print(line,end='',flush=True)
            thread=threading.Thread(target=relay,daemon=True);thread.start()
            try:process.wait(timeout=timeout)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                thread.join(timeout=5)
                state['commands'].append(dict(stage=stage,seconds=time.monotonic()-started,
                    exit_code=process.returncode))
                save()
        if process.returncode:raise RuntimeError(stage+' failed; see preserved log')
    started=time.monotonic();save()
    try:
        import torch
        from importlib.metadata import version
        from benchmarks.ptd_v2_long_kc_contract import validate
        from benchmarks.run_ptd_v2_2nfe_http4_t4 import check_mounts,assemble_diagnostic_manifest
        from patches.legacy_runtime import validate_packages,validate_source
        from patches.h3_operator_policy import H3OperatorPolicy
        from patches.h3_operator_dependencies import configure_compiler
        from patches.ptd.bank_v2 import PTDBankV2
        validate_packages();validate_source(source/'ComfyUI')
        devices=[dict(name=torch.cuda.get_device_name(i),
                capability=list(torch.cuda.get_device_capability(i)),
                uuid=str(torch.cuda.get_device_properties(i).uuid))
                for i in range(torch.cuda.device_count())]
        if len(devices)!=2 or len({v['uuid'] for v in devices})!=2 or any(
                v['capability']!=[7,5] or 'T4' not in v['name'] for v in devices):
            raise RuntimeError('Two distinct physical Tesla T4 SM75 devices required')
        state['hardware']=dict(devices=devices,torch=torch.__version__,
            cuda=torch.version.cuda,kitchen=version('comfy-kitchen'),tilelang=version('tilelang'))
        mark('compiler_preflight')
        state['compiler']=configure_compiler(H3OperatorPolicy(
            te_attention='kitchen_causal',dit_attention='kitchen'))
        if state['compiler'] is None:raise RuntimeError('PTD Native TileLang compiler preflight missing')
        os.environ.update(HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
            HF_HUB_DOWNLOAD_TIMEOUT='90',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2')
        plan=json.loads(args.plan.read_text(encoding='utf-8'))
        state['request_contract']=validate(plan)
        proofs=json.loads(args.proofs.read_text(encoding='utf-8'))
        if set(proofs)!={'shared'}|{f'hop_{i:02d}' for i in range(8)}:
            raise ValueError('Exactly nine qualified immutable PTD Dataset mounts required')
        mark('bank_mount_receipt_admission')
        roots=check_mounts(proofs,state['stages'])
        manifest=output/'bank_v2.json'
        assemble_diagnostic_manifest(proofs,roots,manifest)
        bank=PTDBankV2(manifest,part_roots=tuple(sorted((k,str(v)) for k,v in roots.items())),
            verify_payloads=False)
        if bank.layers!=50 or len(bank.hops)!=8:
            raise ValueError('Trained full 50-layer eight-hop Bank v2 required')
        state['bank']=dict(recipe_sha256=bank.recipe_sha,manifest_sha256=sha(manifest),
            layers=bank.layers,hops=len(bank.hops),parts=list(roots),
            payload_sha256_recomputed=False)
        mark('reference_delivery')
        from benchmarks.qualify_anime_inputs import download_image
        image_root=source/'bench_assets/l1_current';image_root.mkdir(parents=True,exist_ok=True)
        images=[]
        for ref in plan['reference_images']:
            download_image(ref,image_root);images.append(str(image_root/ref['file']))
        assets=json.loads((source/'bundle_assets.json').read_text(encoding='utf-8'))
        identity=json.loads((source/'l1_model_identity.json').read_text(encoding='utf-8'))
        scratch=Path('/kaggle/temp/h3_ptd_v2_09mp_e2e_te_vae')
        scratch.mkdir(parents=True,exist_ok=False)
        required=sum(row['bytes'] for name,row in assets['objects'].items() if name.startswith('te.'))
        required+=assets['auxiliary']['te']['bytes']
        required+=sum(identity[k]['bytes'] for k in ('video','audio'))+5*2**30
        free=shutil.disk_usage(scratch).free
        state['scratch']=dict(path=str(scratch),free_bytes=free,required_bytes=required,
            no_170gb_bank_copy=True)
        if free<required:raise OSError('TE/VAE scratch capacity insufficient')
        from huggingface_hub import hf_hub_download
        from benchmarks.run_h3_l1_turbo_batch import download_with_retry
        def download(repo,revision,name,record,target):
            result=Path(download_with_retry(
                lambda:hf_hub_download(repo,name,revision=revision,token=False,
                    local_dir=str(target)),name,
                lambda event:print('PTD_TE_RETRY',json.dumps(event),flush=True)))
            if result.stat().st_size!=record['bytes']:
                raise ValueError('Immutable TE/VAE byte-size mismatch: '+name)
            return result
        mark('te_100_shard_download')
        root=scratch/'base_bundle';root.mkdir()
        for file in (source/'bundle_provenance').glob('*.json'):
            shutil.copyfile(file,root/file.name)
        if sha(root/'bundle.json')!=assets['publication']['bundle_sha256']:
            raise ValueError('Pinned TE bundle descriptor mismatch')
        objects=[(name,row) for name,row in assets['objects'].items() if name.startswith('te.')]
        if len(objects)!=100:raise ValueError('Expected 100 TE shards')
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(download,assets['publication']['repo'],
                assets['publication']['commit'],name,row,root) for name,row in objects]
            for n,f in enumerate(as_completed(futures),1):
                f.result()
                if n%20==0:print('PTD_TE_READY',n,flush=True)
        mark('te_auxiliary_ranges')
        import requests
        from benchmarks.checkpoint_ranges import assemble
        aux=assets['auxiliary']['te'];orig=assets['source_identity']['te']
        url='https://huggingface.co/'+assets['model_repository']+'/resolve/'+assets['model_revision']+'/'+orig['remote_file']
        def ranged(start,end):
            expected=f'bytes {start}-{end-1}/{orig["bytes"]}'
            for attempt in range(3):
                try:
                    with requests.get(url,params={'download':'true'},
                        headers={'Range':f'bytes={start}-{end-1}','Accept-Encoding':'identity'},
                        timeout=(15,90),stream=True) as response:
                        if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                            raise ValueError('TE auxiliary invalid HTTP range')
                        data=bytearray()
                        for chunk in response.iter_content(4*2**20):
                            data.extend(chunk)
                            if len(data)>end-start:raise ValueError('TE auxiliary oversized range')
                        if len(data)!=end-start:raise ValueError('TE auxiliary short range')
                        return data
                except (requests.RequestException,ValueError):
                    if attempt==2:raise
                    time.sleep(1<<attempt)
        aux_path=scratch/'te_aux.safetensors';assemble(aux,aux_path,ranged)
        from patches.tp2_bundle import DESCRIPTOR
        te=scratch/'te_bundle.json'
        te.write_text(json.dumps(dict(format=DESCRIPTOR,kind='te',
            bundle_root=root.name,bundle_sha256=assets['publication']['bundle_sha256'],
            source={k:orig[k] for k in ('bytes','sha256')},
            native_header=aux['native_header'],native_header_sha256=aux['native_header_sha256'],
            auxiliary_file=aux_path.name,auxiliary_sha256=aux['sha256']),indent=2))
        mark('vae_download')
        paths=dict(te=str(te),dit=str(manifest))
        for kind in ('video','audio'):
            paths[kind]=str(download(assets['model_repository'],
                assets['model_revision'],identity[kind]['remote_file'],identity[kind],
                scratch/'vaes'))
        config=plan|dict(paths=paths,part_roots={k:str(v) for k,v in roots.items()},
            images=images,output=str(output/'generation'),lora_stack=[])
        (output/'request_config.json').write_text(json.dumps(config,indent=2))
        remaining=args.timeout-(time.monotonic()-started)
        if remaining<1200:raise TimeoutError('Insufficient E2E budget remaining before sampler')
        run([sys.executable,'-u',str(source/'benchmarks/run_ptd_v2_long_kc_e2e.py'),
            '--config',str(output/'request_config.json')], 'full_ptd_e2e',int(remaining))
        result=json.loads((output/'generation/result.json').read_text(encoding='utf-8'))
        if result['status']!='complete' or len(result.get('sampler_steps',[]))!=8:
            raise RuntimeError('Eight-hop full PTD sample not complete')
        state.update(status='complete',completed_hops=8,media=result['media'],
            total_seconds=time.monotonic()-started)
        save()
    except BaseException as error:
        state.update(status='failed',failed_stage=state['status'],error=repr(error))
        (output/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
        save();raise

if __name__=='__main__':main()
