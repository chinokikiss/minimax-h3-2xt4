"""Freeze a locally executed cache gate into one bounded dual-T4 allocation."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def main():
    proof = ROOT/'semantic_results/host_cache_ordered/result.json'
    result = json.loads(proof.read_text())
    if not result['passed'] or result['physical'] or not result.get('post_decode_ordering_passed'):
        raise RuntimeError('A passing local execution is required')
    for name, sha in result['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('Source changed after local execution: '+name)
    plans = json.loads((ROOT/'repaired_fixtures/checkpoint_plans.json').read_text())
    for kind, plan in plans.items():
        with (ROOT/f'repaired_fixtures/{kind}_block0.safetensors').open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=plan['sha256']:
                raise RuntimeError('Local fixture differs from remote range plan')
    files = list((ROOT/'patches').glob('*.py')) + [ROOT/'benchmarks'/name for name in
        ['simulate_host_cache.py','logical_dual_gpu.py','retest.py','fixture_inputs.py',
         'te_dit_study.py','checkpoint_ranges.py']]
    files += [proof,ROOT/'repaired_fixtures/checkpoint_plans.json']
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive,mode='w:gz') as tar:
        for path in files:
            tar.add(path,arcname=path.relative_to(ROOT).as_posix())
    payload = base64.b64encode(archive.getvalue()).decode()
    code = '''import base64,hashlib,io,json,os,pathlib,signal,subprocess,sys,tarfile,time,threading
root=pathlib.Path('/kaggle/working/cache_source');root.mkdir(exist_ok=True)
out=pathlib.Path('/kaggle/working/cache_results');out.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as tar:tar.extractall(root,filter='data')
start=time.monotonic();setup_end=start+720
manifest=dict(status='setup',setup_budget_seconds=720,gpu_budget_seconds=480,retry=False,
              scope='Two-request host cache correctness/admission gate, not a performance benchmark')
def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
save()
def command(args,budget,logname):
    begin=time.monotonic()
    with (out/logname).open('w') as log:
        proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        def pump():
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith(('CACHE','VERIFIED','Traceback','ERROR')): print(line.rstrip(),flush=True)
        thread=threading.Thread(target=pump);thread.start()
        try: rc=proc.wait(timeout=max(1,budget))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
        if rc:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        thread.join()
    if rc:raise RuntimeError(logname+' failed; return code '+str(rc))
    return time.monotonic()-begin
try:
    command([sys.executable,'-m','pip','install','--quiet','comfy-kitchen==0.2.37','comfy-aimdo==0.5.5','safetensors==0.8.0','einops','transformers','torchsde','sentencepiece','huggingface-hub==1.12.0','psutil'],setup_end-time.monotonic(),'dependencies.log')
    command([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0','torchvision==0.26.0','torchaudio==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'],setup_end-time.monotonic(),'torch.log')
    command(['git','clone','--filter=blob:none','https://github.com/Comfy-Org/ComfyUI.git',str(root/'ComfyUI')],setup_end-time.monotonic(),'clone.log')
    command(['git','-C',str(root/'ComfyUI'),'checkout','d49e888586dd8ae012c0667b33466b815fee07f7'],setup_end-time.monotonic(),'checkout.log')
    os.environ.update(COMFY_ROOT=str(root/'ComfyUI'),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P',NCCL_DEBUG_FILE=str(out/'nccl.%h.%p.log'),HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
    prep="""import hashlib,json,pathlib,sys
from huggingface_hub import hf_hub_download
root=pathlib.Path(ROOT_VALUE);sys.path.insert(0,str(root/'benchmarks'))
from checkpoint_ranges import download
plans=json.loads((root/'repaired_fixtures/checkpoint_plans.json').read_text())
paths={}
for kind,plan in plans.items():
    target=pathlib.Path('/tmp')/(kind+'_cache_block0.safetensors')
    download(plan,target);paths[kind]=str(target)
    print('VERIFIED',kind,plan['sha256'],flush=True)
video=pathlib.Path(hf_hub_download('Comfy-Org/MiniMax-H3','vae/minimax_h3_video_vae_int8_convrot.safetensors',revision='e5eb578a89295337b8ff433a035929ce0279e0b6',cache_dir='/tmp/cache_models'))
with video.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
if video.stat().st_size!=2811065184 or sha!='52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e':raise RuntimeError('Video checkpoint identity mismatch')
paths['video']=str(video);print('VERIFIED video',sha,flush=True)
(root/'checkpoint_paths.json').write_text(json.dumps(paths))
""".replace('ROOT_VALUE',repr(str(root)))
    command([sys.executable,'-u','-c',prep],setup_end-time.monotonic(),'download.log')
    manifest.update(status='running',setup_seconds=time.monotonic()-start);save()
    manifest['gpu_seconds']=command([sys.executable,'-u',str(root/'benchmarks/simulate_host_cache.py'),'--physical','--checkpoints',str(root/'checkpoint_paths.json'),'--output',str(out)],480,'gate.log')
    result=json.loads((out/'result.json').read_text())
    if not result.get('passed'):raise RuntimeError('Gate did not pass')
    manifest.update(status='complete',total_seconds=time.monotonic()-start);save()
except Exception as error:
    manifest.update(status='failed',error=type(error).__name__+': '+str(error),total_seconds=time.monotonic()-start);save();raise
finally:
    # Large comparator tensors are diagnostics, not deliverable artifacts.
    golden=out/'goldens.pt'
    if golden.exists():golden.unlink()
    print('CACHE FINAL',json.dumps(manifest),flush=True)
'''.replace('PAYLOAD',repr(payload))
    compile(code,'host_cache_gate','exec')
    folder=ROOT/'kaggle_host_cache';folder.mkdir(exist_ok=True)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(id='cache-gate',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    (folder/'cache_gate.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/minimax-h3-host-cache-gate',title='MiniMax H3 Host Cache Gate',code_file='cache_gate.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
    print('Frozen cache gate:',folder/'cache_gate.ipynb')


if __name__=='__main__':
    main()
