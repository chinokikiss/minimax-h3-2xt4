"""One frozen dual-T4 allocation: targeted block ablations and full models."""
import base64
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from tools.experiments.build_study_notebook import ROOT, SLUG


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-only',action='store_true')
    args=parser.parse_args()
    proofs=[ROOT/'semantic_results'/n for n in [
        'te_dit_study_scalemax_t4route.json',
        'full_study_te_dit_s512_m8_scalemax_t4route.json']]
    for p in proofs:
        value=json.loads(p.read_text())
        if not value['passed']:raise RuntimeError('Local semantic execution did not pass: '+p.name)
        for name,sha in value['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Changed after semantic execution: '+name)
    paths=list((ROOT/'patches').glob('*.py'))
    paths += [ROOT/'benchmarks'/n for n in ['retest.py','te_dit_study.py','fixture_inputs.py','full_models.py','full_study.py']]
    paths += proofs+[ROOT/'semantic_results/full_checkpoint_identity.json']
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for p in sorted(paths):tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    payload=base64.b64encode(stream.getvalue()).decode()
    code='''import base64, io, json, math, os, pathlib, signal, subprocess, sys, tarfile, time
root=pathlib.Path('/kaggle/working/study_source');root.mkdir(exist_ok=True)
archive=base64.b64decode(PAYLOAD)
pathlib.Path('/kaggle/working/study_source.tar.gz').write_bytes(archive)
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:tar.extractall(root,filter='data')
started=time.monotonic();setup_end=started+1200
manifest={'phase':'optimized_blocks_and_full_models','status':'setup','setup_budget_seconds':1200,'gpu_budget_seconds':2400,'jobs':[],'retry':False}
out=pathlib.Path('/kaggle/working/study_results');out.mkdir(exist_ok=True)
def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
save()
def cmd(args):subprocess.run(args,check=True,timeout=max(1,setup_end-time.monotonic()))
try:
    cmd([sys.executable,'-m','pip','install','--quiet','comfy-kitchen==0.2.37','comfy-aimdo==0.5.5','safetensors==0.8.0','einops','transformers','torchsde','sentencepiece','huggingface-hub==1.12.0'])
    cmd([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0','torchvision==0.26.0','torchaudio==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'])
    cmd(['git','clone','--filter=blob:none','https://github.com/Comfy-Org/ComfyUI.git',str(root/'ComfyUI')])
    cmd(['git','-C',str(root/'ComfyUI'),'checkout','d49e888586dd8ae012c0667b33466b815fee07f7'])
    os.environ['COMFY_ROOT']=str(root/'ComfyUI')
    os.environ['NCCL_P2P_DISABLE']='1';os.environ['NCCL_IB_DISABLE']='1'
    os.environ['NCCL_DEBUG']='INFO';os.environ['NCCL_DEBUG_SUBSYS']='INIT,GRAPH,P2P'
    os.environ['NCCL_DEBUG_FILE']=str(out/'nccl.%h.%p.log')
    os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    prep="""import hashlib,json,pathlib
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
root=pathlib.Path(ROOT_VALUE)
specs=json.loads((root/'semantic_results/full_checkpoint_identity.json').read_text())
def download(item):
    kind,spec=item
    path=pathlib.Path(hf_hub_download('Comfy-Org/MiniMax-H3',spec['remote_file'],revision='e5eb578a89295337b8ff433a035929ce0279e0b6',cache_dir='/tmp/study_models'))
    if path.stat().st_size!=spec['bytes']:raise RuntimeError('Full checkpoint size differs')
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if sha!=spec['sha256']:raise RuntimeError('Full checkpoint differs from local trained model')
    print('VERIFIED full checkpoint',kind,sha,flush=True)
    return kind,str(path)
with ThreadPoolExecutor(max_workers=2) as pool:paths=dict(pool.map(download,specs.items()))
(root/'checkpoint_paths.json').write_text(json.dumps(paths))
""".replace('ROOT_VALUE',repr(str(root)))
    cmd([sys.executable,'-u','-c',prep])
    checkpoints=json.loads((root/'checkpoint_paths.json').read_text())
    manifest.update(status='running',setup_seconds=time.monotonic()-started);save()
    gpu_end=time.monotonic()+2400
    def job(name,args,budget):
        remaining=min(budget,gpu_end-time.monotonic())
        if remaining<=0:raise TimeoutError('GPU study budget exhausted')
        target=out/name.split('/')[0];target.mkdir(exist_ok=True)
        logpath=target/(name.split('/')[-1]+'.log')
        print('START',name,'budget',int(remaining),flush=True)
        begin=time.monotonic()
        with logpath.open('w') as log:
            proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
            if rc:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        print(logpath.read_text(),flush=True)
        manifest['jobs'].append({'name':name,'returncode':rc,'seconds':time.monotonic()-begin});save()
        return rc
    selected={}
    for kind in ['te','dit']:
        block=out/kind
        for mode in ['native','tp_legacy','tp_fixed','tp_scalemax','tp_p2p']:
            job(kind+'/'+mode,[sys.executable,'-u',str(root/'benchmarks/te_dit_study.py'),'--kind',kind,'--checkpoint',checkpoints[kind],'--output',str(block),'--modes',mode],180)
        candidates=[]
        for mode in ['tp_fixed','tp_scalemax','tp_p2p']:
            path=block/(mode+'_results.json')
            if not path.exists():continue
            rows=json.loads(path.read_text())
            if len(rows)!=4 or not all(r['correctness_passed'] for r in rows):continue
            target=[r for r in rows if r['batch']==1 and r['length']!=37]
            candidates.append({'mode':mode,'geomean_ms':math.exp(sum(math.log(r['median_ms']) for r in target)/len(target))})
        if candidates:selected[kind]=min(candidates,key=lambda c:c['geomean_ms'])['mode']
        (block/'selection.json').write_text(json.dumps({'correctness_filtered_candidates':candidates,'selected':selected.get(kind),'criterion':'geomean latency of two batch-1 longer block workloads; confirm full-model performance separately'},indent=2))
    for kind in ['te','dit']:
        full=out/('full_'+kind);full.mkdir(exist_ok=True)
        if job('full_'+kind+'/native',[sys.executable,'-u',str(root/'benchmarks/full_study.py'),'--kind',kind,'--checkpoint',checkpoints[kind],'--output',str(full),'--mode','native'],350):continue
        for mode in ['pp_serial_balanced','tp_optimized','pp_overlap']:
            if mode=='tp_optimized' and kind not in selected:continue
            invocation=[sys.executable,'-u',str(root/'benchmarks/full_study.py'),'--kind',kind,'--checkpoint',checkpoints[kind],'--output',str(full),'--mode',mode]
            if mode=='tp_optimized':invocation+=['--tp-variant',selected[kind]]
            job('full_'+kind+'/'+mode,invocation,450)
    manifest.update(status='complete' if all(j['returncode']==0 for j in manifest['jobs']) else 'partial_failure',total_seconds=time.monotonic()-started)
    manifest['files']=[p.relative_to(pathlib.Path('/kaggle/working')).as_posix() for p in out.rglob('*') if p.is_file() and p.suffix!='.pt'];save()
except Exception as e:
    manifest.update(status='failed',error=type(e).__name__+': '+str(e),total_seconds=time.monotonic()-started)
    manifest['files']=[p.relative_to(pathlib.Path('/kaggle/working')).as_posix() for p in out.rglob('*') if p.is_file() and p.suffix!='.pt'];save();raise
'''.replace('PAYLOAD',repr(payload))
    if args.full_only:
        # Only orchestration changes: execute the exact locally validated
        # model and timing code, with room for complete 50-layer workloads.
        begin=code.index('    selected={}\n')
        end=code.index("    for kind in ['te','dit']:\n        full=",begin)
        selection={}
        for kind in ['te','dit']:
            path=ROOT/'kaggle_study_output/optimized/study_results'/kind/'selection.json'
            selection[kind]=json.loads(path.read_text())['selected']
            if selection[kind] not in ['tp_fixed','tp_scalemax','tp_p2p']:
                raise RuntimeError('No physically validated optimized TP selection')
        code=code[:begin]+'    selected='+repr(selection)+'\n'+code[end:]
        code=code.replace("'phase':'optimized_blocks_and_full_models'","'phase':'full_models_completion'")
        code=code.replace("'gpu_budget_seconds':2400","'gpu_budget_seconds':3600")
        code=code.replace('gpu_end=time.monotonic()+2400','gpu_end=time.monotonic()+3600')
        code=code.replace("'--mode','native'],350)","'--mode','native'],600)")
        code=code.replace('job(\'full_\'+kind+\'/\'+mode,invocation,450)','job(\'full_\'+kind+\'/\'+mode,invocation,600)')
        code=code.replace("        print(logpath.read_text(),flush=True)","        print('FINISHED',name,'returncode',rc,'seconds',round(time.monotonic()-begin,2),flush=True)\n        print(logpath.read_text(),flush=True)")
    compile(code,'optimized_study','exec')
    folder=ROOT/'kaggle_study';folder.mkdir(exist_ok=True)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'id':'run','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    (folder/'study.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    print('Frozen optimized notebook bytes',(folder/'study.ipynb').stat().st_size)


if __name__=='__main__':main()
