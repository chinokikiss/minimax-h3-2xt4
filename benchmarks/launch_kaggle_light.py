"""Execution phase after the separate direct-pip/source/API setup gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
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
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--distribution',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('e2e','operators','dit-profile'),required=True)
    p.add_argument('--private-input',type=Path)
    p.add_argument('--dit-attention',choices=('kitchen','sol'),default='kitchen')
    args=p.parse_args()
    if args.mode!='dit-profile' and args.dit_attention!='kitchen':p.error('--dit-attention requires --mode dit-profile')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',mode=args.mode,commands=[])
    def save():(out/'batch_results.json').write_text(json.dumps(state,indent=2))
    def run(command,name,timeout):
        state['stage']=name;save();print('LIGHT_RUN_START',name,flush=True);start=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,text=True)
            def relay():
                for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            thread=threading.Thread(target=relay,daemon=True);thread.start()
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);code=proc.wait(timeout=5)
            thread.join(timeout=5)
        state['commands'].append(dict(name=name,exit_code=code,seconds=time.monotonic()-start));save()
        if code:raise RuntimeError(name+' failed: '+str(code))
    try:
        import torch,comfy_kitchen as ck
        import importlib.metadata as md
        from patches.legacy_runtime import validate_packages,validate_source
        validate_packages();validate_source(args.source/'ComfyUI')
        devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
            uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(devices)==2 and len({d['uuid'] for d in devices})==2 and all(d['capability']==[7,5] and 'T4' in d['name'] for d in devices)
        assert torch.__version__=='2.11.0+cu128','Native GPU Torch changed; do not silently substitute'
        assert ck.int8_attention_is_available() and ck.backends.cuda._EXT_AVAILABLE
        nvcc=shutil.which('nvcc');assert nvcc,'Preinstalled compiler missing'
        compiler=subprocess.check_output([nvcc,'--version'],text=True,timeout=10);assert 'release 12.8' in compiler
        nvcc=Path(nvcc).resolve();os.environ.update(CUDACXX=str(nvcc),CUDA_HOME=str(nvcc.parent.parent),CUDA_PATH=str(nvcc.parent.parent))
        state['hardware']=dict(devices=devices,torch=torch.__version__,cuda=torch.version.cuda,kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'))
        state['compiler']=compiler;save()
        if args.mode=='dit-profile':
            from launch_dit_profile import find_tool
            nsys=find_tool('nsys')
            if not nsys:raise RuntimeError('Nsight Systems unavailable; blocked before model downloads')
            run([sys.executable,'-u',str(args.source/'benchmarks/qualify_worker_cuda_trace.py'),
                '--nsys',nsys,'--output',str(out/'worker_capture'),'--checkpoint-restart'],'worker_capture_gate',120)
            run([sys.executable,'-u',str(args.source/'benchmarks/run_h3_l1_current_batch.py'),
                '--source',str(args.source),'--output',str(out/'dit_single'),'--dit-profile',
                '--capture-gate',str(out/'worker_capture/capture_gate.json'),
                '--dit-attention',args.dit_attention],'dit_single',4600)
        elif args.mode=='e2e':
            run([sys.executable,'-u',str(args.source/'benchmarks/run_h3_l1_current_batch.py'),
                '--source',str(args.source),'--output',str(out/'e2e')],'long_e2e',4800)
        else:
            locator=json.loads(args.private_input.read_text());manifest=locator['manifest']
            from kaggle_secrets import UserSecretsClient
            secret=None
            for name in ('HF_TOKEN','HF_token'):
                try:secret=UserSecretsClient().get_secret(name);break
                except Exception:pass
            if not secret:raise RuntimeError('Enable Kaggle Secret HF_TOKEN for private captured input; never embed the token in code')
            from huggingface_hub import hf_hub_download
            path=Path(hf_hub_download(locator['repo'],manifest['file'],repo_type='dataset',revision=locator['revision'],token=secret,
                cache_dir='/kaggle/temp/h3_private_input_cache'))
            del secret
            if sha(path)!=manifest['compressed_sha256']:raise RuntimeError('Private capture checksum mismatch')
            assets=Path('/kaggle/temp/h3_component_assets');assets.mkdir(exist_ok=True)
            with gzip.open(path,'rb') as source,(assets/'first_block.pt').open('wb') as target:shutil.copyfileobj(source,target,4*2**20)
            if sha(assets/'first_block.pt')!=manifest['uncompressed_sha256']:raise RuntimeError('Unpacked capture mismatch')
            plans=json.loads((args.distribution/'component_asset_plans.json').read_text())
            from checkpoint_ranges import assemble
            import requests
            def one(item):
                name,plan=item;print('LIGHT_ASSET_START',name,flush=True)
                url='https://huggingface.co/'+plan['repository']+'/resolve/'+plan['revision']+'/'+plan['remote_file']
                def read(a,b):
                    with requests.get(url,params={'download':'true','offset':str(a)},headers={'Range':f'bytes={a}-{b-1}'},stream=True,timeout=(15,90)) as response:
                        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {a}-{b-1}/{plan["source_bytes"]}':
                            raise RuntimeError('Immutable model range mismatch')
                        data=bytearray()
                        for chunk in response.iter_content(4*2**20):
                            data.extend(chunk)
                            if len(data)>b-a:raise RuntimeError('Oversized range')
                        return data
                assemble(plan,assets/name,read);print('LIGHT_ASSET_READY',name,flush=True)
            with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,plans.items()))
            launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
            run([*launcher,str(args.source/'benchmarks/run_t4_row_partial_combined.py'),'--assets',str(assets),
                '--output',str(out/'combined_rows.json')],'combined_row_partial',300)
            run([sys.executable,str(args.source/'benchmarks/run_t4_reference_demand_vae.py'),'--assets',str(assets),
                '--output',str(out/'reference_demand_vae.json')],'reference_demand_vae',150)
            nsys=shutil.which('nsys')
            if not nsys and Path('/opt/nvidia').exists():nsys=next((str(p) for p in Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
            state['profiling']=dict(nsys=nsys,ncu='Historical counter-permission blocker; not reprobed')
            if nsys:
                run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--force-overwrite=true',
                    '-o',str(out/'combined_rows_nsys'),*launcher,str(args.source/'benchmarks/run_t4_row_partial_combined.py'),
                    '--assets',str(assets),'--trace-only','--output',str(out/'systems_rows.json')],'combined_nsight_systems',75)
                run([nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum','--format','csv',
                    '--output',str(out/'combined_rows_stats'),str(out/'combined_rows_nsys.nsys-rep')],'nsight_statistics',20)
        state['status']='completed_not_generation_acceptance';save()
    except BaseException as error:
        state.update(status='failed',error=repr(error));(out/'exception.txt').write_text(traceback.format_exc());save();raise


if __name__=='__main__':main()
