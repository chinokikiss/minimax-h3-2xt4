"""Package the exact working-tree harness and proposals for a private Kaggle run."""
import base64
import argparse
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
COMFY_SHA = "d49e888586dd8ae012c0667b33466b815fee07f7"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',default='kaggle_retest')
    parser.add_argument('--id',default='chinokiki9/minimax-h3-2xt4-benchmark-retest')
    parser.add_argument('--title',default='MiniMax-H3 2xT4 Benchmark Retest')
    parser.add_argument('--suites',default='audio,video,video_tiling,te,dit,te_stack,dit_stack')
    parser.add_argument('--rounds',type=int,default=3)
    args = parser.parse_args()
    folder = ROOT / args.folder
    folder.mkdir(exist_ok=True)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for directory in ["benchmarks", "patches"]:
            for path in sorted((ROOT / directory).glob("*.py")):
                archive.add(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))
    payload = base64.b64encode(stream.getvalue()).decode()
    code = '''import base64, json, os, pathlib, subprocess, sys, tarfile, io, time, traceback
ROOT = pathlib.Path('/kaggle/working/retest_source')
ROOT.mkdir(exist_ok=True)
payload = PAYLOAD
archive = base64.b64decode(payload)
(pathlib.Path('/kaggle/working') / 'retest_source.tar.gz').write_bytes(archive)
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
    tar.extractall(ROOT,filter='data')
def command(args, **kwargs):
    print('RUN', ' '.join(args),flush=True)
    subprocess.run(args,check=True,**kwargs)
command([sys.executable,'-m','pip','install','--quiet','comfy-kitchen==0.2.37','comfy-aimdo==0.5.5','safetensors==0.8.0','huggingface-hub==1.12.0','einops','transformers','torchsde','sentencepiece'])
command([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0','torchvision==0.26.0','torchaudio==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'])
comfy = ROOT / 'ComfyUI'
command(['git','clone','--filter=blob:none','https://github.com/Comfy-Org/ComfyUI.git',str(comfy)])
command(['git','-C',str(comfy),'checkout',COMFY_SHA])
os.environ['COMFY_ROOT'] = str(comfy)
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
command(['nvidia-smi'])
command([sys.executable,str(ROOT/'benchmarks/retest_smoke.py')])
manifest = {'comfy_sha':COMFY_SHA,'jobs':[],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
manifest_path = pathlib.Path('/kaggle/working/retest_manifest.json')
for round_index in range(ROUND_COUNT):
    for suite in SUITE_LIST:
        if suite == 'video_profile' and round_index > 0:
            continue
        out = pathlib.Path('/kaggle/working/retest_results') / f'round_{round_index}' / suite
        job = {'round':round_index,'suite':suite,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        start = time.perf_counter()
        out.mkdir(parents=True,exist_ok=True)
        invocation = [sys.executable,'-u',str(ROOT/'benchmarks/retest_profile.py'),'--output',str(out)] if suite == 'video_profile' else [sys.executable,'-u',str(ROOT/'benchmarks/retest.py'),'--suite',suite,'--output',str(out),'--round',str(round_index),'--warmups','3','--iterations','10']
        with (out/'execution.log').open('w') as log:
            process = subprocess.Popen(invocation,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                print(line,end='',flush=True)
                log.write(line)
                log.flush()
            job['returncode'] = process.wait()
        job['elapsed_seconds'] = time.perf_counter()-start
        manifest['jobs'].append(job)
        manifest_path.write_text(json.dumps(manifest,indent=2))
        print('JOB_RESULT',json.dumps(job),flush=True)
manifest['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
manifest_path.write_text(json.dumps(manifest,indent=2))
print('RETEST_FINISHED',flush=True)
'''.replace('PAYLOAD',repr(payload)).replace('COMFY_SHA',repr(COMFY_SHA)).replace('ROUND_COUNT',str(args.rounds)).replace('SUITE_LIST',repr(args.suites.split(',')))
    notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[{"cell_type":"markdown","metadata":{},"source":["# Corrected benchmark rerun\nReal checkpoints, identical inputs, raw synchronized timings, explicit correctness gates. Existing results are retained separately.\n"]},{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code.splitlines(keepends=True)}]}
    (folder/'retest.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    metadata = json.loads((ROOT/'kaggle_kernel/kernel-metadata.json').read_text())
    metadata.update(id=args.id, code_file='retest.ipynb',title=args.title)
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(folder)

if __name__ == '__main__':
    main()
