"""Prerequisites + bounded real T4 native/candidate generation screening."""
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


def main():
    out=Path('/kaggle/working/generation_screen');out.mkdir(exist_ok=True)
    began=time.monotonic();deadline=began+19800
    state={'status':'prerequisites','quality_status':'not_measured','performance_scope':'native two-device block placement quality screen; not TP/NCCL deployment'}
    def save():(out/'status.json').write_text(json.dumps(state,indent=2))
    save()
    assets=[p for p in Path('/kaggle/input').rglob('source_identity.json')
        if (p.parent/'comfy_master.zip').exists() or any(p.parent.rglob('nodes_minimax_h3.py'))]
    assert len(assets)==1,assets
    data=assets[0].parent;identity=json.loads(assets[0].read_text())
    root=Path('/tmp/h3_quality_source');root.mkdir(exist_ok=True)
    try:
        for name,destination in [('comfy_master.zip',root/'ComfyUI-master'),('experiment_source.zip',root)]:
            if (data/name).exists():
                assert hashlib.sha256((data/name).read_bytes()).hexdigest()==identity['bundle_sha256'][name]
                with zipfile.ZipFile(data/name) as archive:archive.extractall(destination)
            else:
                # Kaggle may expand ZIP attachments when processing a dataset.
                # Individual source SHA256 verification below remains mandatory.
                marker='nodes_minimax_h3.py' if name=='comfy_master.zip' else 'capture_real_attention.py'
                matches=list(data.rglob(marker));assert len(matches)==1,(marker,matches)
                source=matches[0].parent.parent
                shutil.copytree(source,destination,dirs_exist_ok=True)
        for name,sha in identity['comfy_master_sha256'].items():assert hashlib.sha256((root/'ComfyUI-master'/name).read_bytes()).hexdigest()==sha,name
        for name,sha in identity['workspace_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
        (out/'source_identity.json').write_text(json.dumps(identity,indent=2))
        overlays=json.loads(Path('/kaggle/working/benchmark_overrides.json').read_text())
        overlay_hashes={}
        for name,content in overlays.items():
            destination=root/name
            if not destination.resolve().is_relative_to(root.resolve()):raise RuntimeError('Overlay leaves source directory')
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(content)
            overlay_hashes[name]=hashlib.sha256(content.encode()).hexdigest()
        (out/'benchmark_override_hashes.json').write_text(json.dumps(overlay_hashes,indent=2))
        commands=[]
        def run(argv,name,timeout):
            start=time.monotonic()
            timeout=min(timeout,deadline-start)
            if timeout<=0:raise RuntimeError('Whole job budget exhausted')
            command=dict(argv=argv,timeout_seconds=timeout,exit_code=None)
            try:
                with (out/(name+'.log')).open('w') as log:
                    proc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,env=os.environ.copy(),cwd=root)
                command['exit_code']=proc.returncode
            finally:
                command['seconds']=time.monotonic()-start;commands.append(command)
                (out/'commands.json').write_text(json.dumps(commands,indent=2))
            print('GENERATION_STAGE',name,proc.returncode,flush=True)
            if proc.returncode:raise RuntimeError(name+' failed; inspect saved log')
        run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','tilelang==0.1.15','comfy-kitchen==0.2.37',
             'comfy-aimdo==0.5.5','safetensors==0.8.0','einops','transformers==5.16.1','torchsde','sentencepiece',
             'huggingface-hub==1.12.0','psutil','nvidia-ml-py','av==19.0.1','pydantic-settings','kornia',
             'alembic','sqlalchemy','simpleeval','blake3','comfy-angle'],'dependencies',900)
        compiler=Path(shutil.which('nvcc')).resolve()
        compiler_version=subprocess.check_output([str(compiler),'--version'],text=True)
        if 'release 12.8' not in compiler_version:raise RuntimeError('Required SM75 CUDA12.8 compiler unavailable')
        os.environ.update(CUDACXX=str(compiler),CUDA_HOME=str(compiler.parent.parent),CUDA_PATH=str(compiler.parent.parent))
        state['compiler']=dict(path=str(compiler),version=compiler_version);save()
        run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--upgrade','torch==2.11.0',
             'torchvision==0.26.0','torchaudio==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'],'cu130_runtime',1200)
        import torch
        state['hardware']={'devices':[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
            vram=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())],
            'torch':torch.__version__,'packages':{name:importlib.metadata.version(name) for name in ['tilelang','comfy-kitchen','transformers','safetensors','av']}}
        assert len(state['hardware']['devices'])==2 and all(x['capability']==[7,5] for x in state['hardware']['devices'])
        os.environ.update(COMFY_ROOT=str(root/'ComfyUI-master'),HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
        # Source imports and model metadata are verified before large downloads.
        code="import sys;sys.path[:0]=["+repr(str(root/'ComfyUI-master'))+","+repr(str(root))+","+repr(str(root/'benchmarks'))+"];import comfy_kitchen,comfy.ops,comfy.ldm.minimax.model,comfy.ldm.minimax.vae,comfy.text_encoders.minimax;from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo;print('IMPORT_PASS')"
        run([sys.executable,'-c',code],'source_imports',120)
        if Path('/kaggle/working/generation_schedule.json').exists():
            os.environ['H3_GROUPED_GATE']='1'
            sys.path.insert(0,str(root/'benchmarks'))
            from quality_next_driver import prerequisites
            prerequisites(root,out)
        run([sys.executable,str(root/'benchmarks/quality_first_runtime_probe.py'),str(out/'runtime_compatibility.json')],'runtime_compatibility',300)
        with (out/'pip_freeze.txt').open('w') as log:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=log,check=True,timeout=30)
        specs=json.loads((root/'semantic_results/pipeline_checkpoint_identity.json').read_text())
        required=sum(x['bytes'] for x in specs.values())+8*2**30
        state['disk_free']=shutil.disk_usage('/tmp').free
        if state['disk_free']<required:raise RuntimeError('Insufficient /tmp disk for exact checkpoint set plus bounded scratch')
        from huggingface_hub import hf_hub_download
        paths={};aliases={'te':'qwen3vl_32b_minimax_h3_int8_convrot.safetensors','dit':'MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
            'video':'minimax_h3_video_vae_int8_convrot.safetensors','audio':'minimax_h3_audio_vae_fp32_clean.safetensors'}
        model_dir=Path('/tmp/h3_quality_models');model_dir.mkdir(exist_ok=True)
        for kind,spec in specs.items():
            state.update(status='checkpoint_download',downloading=kind);save()
            path=Path(hf_hub_download('Comfy-Org/MiniMax-H3',spec['remote_file'],revision='e5eb578a89295337b8ff433a035929ce0279e0b6',cache_dir='/tmp/h3_model_cache'))
            with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
            if sha!=spec['sha256'] or path.stat().st_size!=spec['bytes']:raise RuntimeError('Checkpoint SHA mismatch '+kind)
            alias=model_dir/aliases[kind]
            if not alias.exists():alias.symlink_to(path)
            paths[kind]=dict(path=str(alias),sha256=sha,bytes=path.stat().st_size)
            state['checkpoints']=paths;save();print('CHECKPOINT_VERIFIED',kind,flush=True)
        workload=json.loads((root/'bench_assets/tokyo_ref2va.json').read_text())
        workload.update(comfy_root=str(root/'ComfyUI-master'),model_directory=str(model_dir),
            capture_layers={'te':[],'dit':[],'video':[]},save_denoiser_tensors=False,save_pixel_tensor=False,
            seed_global_rng=True,dit_loading='resident_two_device')
        if Path('/kaggle/working/generation_schedule.json').exists():
            from quality_next_driver import run_schedule
            run_schedule(root,out,workload,run,state,save)
            return
        policies={'native':{},'sage32_dit':{'dit':{'mode':'sage32'}},
            'scalar_tau0_dit':{'dit':{'mode':'scalar_pool','tau':0.,'prefix_tokens':874,'expected_tokens':6814}},
            'drop_tau0_dit':{'dit':{'mode':'drop','tau':0.,'prefix_tokens':874,'expected_tokens':6814}}}
        state.update(status='quality_screen',runs=[]);save()
        with (out/'telemetry.csv').open('w') as log:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,clocks.sm,pstate,utilization.gpu,memory.used,power.draw','--format=csv','-lms','500'],stdout=log)
            try:
                for seed in [20261013,20261014]:
                    for mode,policy in policies.items():
                        name=f'{mode}_seed{seed}';directory=out/name
                        config=workload|{'attention_policy':policy,'seed':seed,'output_directory':str(directory),
                            'qualify_resident_loading':mode=='native' and seed==20261013}
                        cfg=out/(name+'.json');cfg.write_text(json.dumps(config,indent=2))
                        state['current_run']=name;save()
                        run([sys.executable,'-u',str(root/'benchmarks/quality_first_generation.py'),'--config',str(cfg)],name,3600)
                        run([sys.executable,str(root/'benchmarks/quality_first_video_metrics.py'),'--video',str(directory/'native_video.mp4'),
                            '--reference',str(root/'bench_assets/tokyo_neon.jpg'),'--prompt',config['prompt'],'--output',str(directory/'video_metrics.json')],
                            name+'_flow_metrics',120)
                        state['runs'].append({'name':name,'status':'generated','quality':'unrated'});save()
            finally:monitor.terminate();monitor.wait(timeout=15)
        state.update(status='screen_generated',quality_status='metrics and blinded human review pending');save()
    except Exception as error:
        # Never print signed download URLs or substitute another GPU.
        state.update(status='blocked',blocker=type(error).__name__,reason=re.sub(r'https?://\S+','[URL]',str(error)),
            traceback_frames=[dict(file=f.filename,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)])
        save();print('GENERATION_BLOCKED',state['blocker'],state['reason'],flush=True)
        raise SystemExit(2)


if __name__=='__main__':main()
