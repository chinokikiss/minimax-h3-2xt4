"""Submit-ready, bounded physical VAE profiling; never touches existing jobs."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
SLUG='chinokiki9/minimax-h3-vae-kv-profile'


def main():
    proof=json.loads((ROOT/'semantic_results/vae_kv_profile_local/results.json').read_text())
    assert proof['completed'] and proof['passed'] and not proof['physical']
    assert len(proof['cases'])==21
    for name,sha in proof['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    paths=list((ROOT/'patches').glob('*.py'))
    paths += [ROOT/'benchmarks'/name for name in ('profile_slab_kv.py','retest.py','logical_dual_gpu.py')]
    paths += [ROOT/'semantic_results/vae_kv_profile_local/results.json',ROOT/'slab_results/saved_references/manifest.json']
    paths += [ROOT/'slab_results/saved_references'/('input_'+name+'.pt') for name in ('768x768_1F','768x1344_2F')]
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for path in sorted(paths):archive.add(path,arcname=path.relative_to(ROOT).as_posix())
    payload=base64.b64encode(stream.getvalue()).decode()
    code='''import base64, io, json, os, pathlib, signal, subprocess, sys, tarfile, time
ROOT=pathlib.Path('/kaggle/working/vae_kv_source');ROOT.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/vae_kv_profile');OUT.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:
    archive.extractall(ROOT,filter='data')
manifest={'status':'setup','setup_budget_seconds':900,'gpu_budget_seconds':1800,'total_budget_seconds':2700,
          'automatic_retry':False,'new_job_only':True,'raw_decoder_only':True,'physical_required':'2x Tesla T4',
          'correctness':'same-allocation Native outputs, exactness required for accepted speedups',
          'modes':['native','independent','global_rope','shared_kv_global','full_query_control','query_block_control','aligned']}
def save(): (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
save();deadline=time.monotonic()+900
def run(args,label,budget):
    log=OUT/(label+'.log');start=time.monotonic();position=0
    with log.open('w') as stream:
        proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            elapsed=time.monotonic()-start
            if elapsed>budget:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise TimeoutError(label+' exceeded its bounded budget')
            text=log.read_text(errors='replace');new=text[position:];position=len(text)
            for line in new.splitlines():
                if line.startswith('VAE '):print(line,flush=True)
            print('VAE PROGRESS',label,round(elapsed),flush=True);time.sleep(15)
        text=log.read_text(errors='replace')
        for line in text[position:].splitlines():
            if line.startswith('VAE '):print(line,flush=True)
        if proc.returncode:
            print(text[-6000:],flush=True);raise RuntimeError(label+' failed with exit '+str(proc.returncode))
def setup(args,label):run(args,label,max(1,deadline-time.monotonic()))
try:
    # Match the qualified Native runtime; no alternate attention/quantization.
    setup([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0','torchvision==0.26.0','torchaudio==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'],'torch_install')
    setup([sys.executable,'-m','pip','install','--quiet','comfy-kitchen==0.2.37','comfy-aimdo==0.5.5','safetensors==0.8.0','huggingface-hub==1.12.0','einops','transformers','torchsde','sentencepiece'],'dependencies')
    comfy=ROOT/'ComfyUI'
    setup(['git','clone','--filter=blob:none','https://github.com/Comfy-Org/ComfyUI.git',str(comfy)],'comfy_clone')
    setup(['git','-C',str(comfy),'checkout','d49e888586dd8ae012c0667b33466b815fee07f7'],'comfy_pin')
    os.environ.update(COMFY_ROOT=str(comfy),HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
                      MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='67108864',OMP_NUM_THREADS='2',
                      NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,COLL',NCCL_DEBUG_FILE=str(OUT/'nccl.%h.%p.log'),
                      NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
    download="from huggingface_hub import hf_hub_download; import pathlib; p=hf_hub_download('Comfy-Org/MiniMax-H3','vae/minimax_h3_video_vae_int8_convrot.safetensors',revision='e5eb578a89295337b8ff433a035929ce0279e0b6'); pathlib.Path('/kaggle/working/vae_kv_profile/checkpoint_path.txt').write_text(p)"
    setup([sys.executable,'-c',download],'video_checkpoint_download')
    # Explicit topology evidence; transport routing is not guessed from copy timings.
    setup(['nvidia-smi','topo','-m'],'topology')
    setup(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv'],'hardware')
    manifest.update(status='running');save()
    checkpoint=(OUT/'checkpoint_path.txt').read_text()
    run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
         str(ROOT/'benchmarks/profile_slab_kv.py'),'--checkpoint',checkpoint,'--output',str(OUT),
         '--warmups','1','--iterations','3'],'execution',1800)
    results=json.loads((OUT/'results.json').read_text())
    assert results['completed'] and results['physical']
    manifest.update(status='complete',accepted_candidates=[{'workload':r['workload'],'mode':r['mode']} for r in results['cases'] if r['accepted']]);save()
except Exception as exc:
    manifest.update(status='failed',error=type(exc).__name__+': '+str(exc));save();raise
'''.replace('PAYLOAD',repr(payload))
    compile(code,'vae-kv-profile-notebook','exec')
    folder=ROOT/'kaggle_vae_kv_profile';folder.mkdir(exist_ok=True)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[
        dict(id='scope',cell_type='markdown',metadata={},source=['# Physical dual-T4 VAE KV profiling\nOne bounded job. Three trained full-decoder workloads, including 512px/124 frames. Approximate controls never earn accepted speedups.\n']),
        dict(id='profile',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    (folder/'profile.ipynb').write_text(json.dumps(notebook,indent=1))
    meta=dict(id=SLUG,title='MiniMax H3 VAE KV Profile',code_file='profile.ipynb',language='python',kernel_type='notebook',
              is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,dataset_sources=[],competition_sources=[],model_sources=[],kernel_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    write=dict(kernel=SLUG,local_qualified=True,local_candidate_cases=3,source_sha256=proof['source_sha256'],
               setup_budget_seconds=900,gpu_budget_seconds=1800,iterations=3,warmups=1,
               source_payload_bytes=len(stream.getvalue()),existing_pipeline_job_unchanged=True)
    (folder/'build_manifest.json').write_text(json.dumps(write,indent=2))
    print('BUILT',folder,'payload bytes',len(stream.getvalue()))


if __name__=='__main__':main()
