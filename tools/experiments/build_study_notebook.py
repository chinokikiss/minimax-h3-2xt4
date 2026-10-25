"""Frozen, bounded dual-T4 profiler jobs. Reuse exact trained range fixtures."""
import base64
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]
SLUG='chinokiki9/minimax-h3-te-dit-parallel-study'


def main():
    paths=list((ROOT/'patches').glob('*.py'))
    paths += [ROOT/'benchmarks'/n for n in ['retest.py','te_dit_study.py','fixture_inputs.py','checkpoint_ranges.py']]
    paths += [ROOT/'repaired_fixtures/checkpoint_plans.json']
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for p in sorted(paths):tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    payload=base64.b64encode(stream.getvalue()).decode()
    code='''import base64, io, json, os, pathlib, signal, subprocess, sys, tarfile, time
root=pathlib.Path('/kaggle/working/study_source');root.mkdir(exist_ok=True)
archive=base64.b64decode(PAYLOAD)
pathlib.Path('/kaggle/working/study_source.tar.gz').write_bytes(archive)
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:tar.extractall(root,filter='data')
started=time.monotonic();manifest={'phase':'baseline_profile','status':'setup','jobs':[],'retry':False}
out=pathlib.Path('/kaggle/working/study_results');out.mkdir(exist_ok=True)
def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
save()
def cmd(args): subprocess.run(args,check=True,timeout=max(1,600-(time.monotonic()-started)))
try:
    cmd([sys.executable,'-m','pip','install','--quiet','comfy-kitchen==0.2.37','comfy-aimdo==0.5.5','safetensors==0.8.0','einops','transformers','torchsde','sentencepiece','huggingface-hub==1.12.0'])
    cmd([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0','torchvision==0.26.0','torchaudio==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'])
    cmd(['git','clone','--filter=blob:none','https://github.com/Comfy-Org/ComfyUI.git',str(root/'ComfyUI')])
    cmd(['git','-C',str(root/'ComfyUI'),'checkout','d49e888586dd8ae012c0667b33466b815fee07f7'])
    os.environ['COMFY_ROOT']=str(root/'ComfyUI')
    os.environ['NCCL_P2P_DISABLE']='1';os.environ['NCCL_IB_DISABLE']='1'
    os.environ['NCCL_DEBUG']='INFO';os.environ['NCCL_DEBUG_SUBSYS']='INIT,GRAPH,COLL,P2P'
    os.environ['NCCL_DEBUG_FILE']=str(out/'nccl.%h.%p.log')
    prep="import sys,json,pathlib;root=pathlib.Path(ROOT);sys.path.insert(0,str(root/'benchmarks'));from checkpoint_ranges import download;plans=json.loads((root/'repaired_fixtures/checkpoint_plans.json').read_text());[(download(plans[k],root/'repaired_fixtures'/ (k+'_block0.safetensors')),print('VERIFIED',k,flush=True)) for k in ['te','dit']]".replace('ROOT',repr(str(root)))
    cmd([sys.executable,'-u','-c',prep])
    manifest.update(status='running',setup_seconds=time.monotonic()-started);save()
    for kind in ['te','dit']:
        target=out/kind;target.mkdir(exist_ok=True)
        start=time.monotonic()
        with (target/'execution.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(root/'benchmarks/te_dit_study.py'),'--kind',kind,'--checkpoint',str(root/'repaired_fixtures'/(kind+'_block0.safetensors')),'--output',str(target)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=proc.wait(timeout=240)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
            if rc:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        print((target/'execution.log').read_text(),flush=True)
        manifest['jobs'].append({'kind':kind,'returncode':rc,'seconds':time.monotonic()-start});save()
    manifest.update(status='complete' if all(j['returncode']==0 for j in manifest['jobs']) else 'partial_failure',total_seconds=time.monotonic()-started)
    manifest['files']=[p.relative_to(pathlib.Path('/kaggle/working')).as_posix() for p in out.rglob('*') if p.is_file() and p.suffix!='.pt'];save()
except Exception as e:
    manifest.update(status='failed',error=type(e).__name__+': '+str(e));save();raise
'''.replace('PAYLOAD',repr(payload))
    compile(code,'study_notebook','exec')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'id':'run','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    folder=ROOT/'kaggle_study';folder.mkdir(exist_ok=True)
    (folder/'study.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    (folder/'kernel-metadata.json').write_text(json.dumps({'id':SLUG,'title':'MiniMax H3 TE DiT Parallel Study','code_file':'study.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}),encoding='utf-8')
    print('Frozen notebook bytes', (folder/'study.ipynb').stat().st_size)


if __name__=='__main__':main()
