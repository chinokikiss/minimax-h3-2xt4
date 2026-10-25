"""Prepare trained component assets on CPU from immutable, byte-checked ranges."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'benchmarks'))
from checkpoint_ranges import make_plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',type=Path,default=Path('D:/minimaxH3'))
    parser.add_argument('--fixtures',type=Path,default=Path('D:/attention_eval/t4_followup_20261021/component_fixtures'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use fresh submitted source directories')
    args.output.mkdir(parents=True)
    revision='e5eb578a89295337b8ff433a035929ce0279e0b6'
    rows=[('te_block0.safetensors','qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
            'Comfy-Org/MiniMax-H3',revision,'text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'),
          ('dit_blocks.safetensors','MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
            'Comfy-Org/MiniMax-H3',revision,'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors'),
          ('refiner_attention.safetensors','MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
            'Comfy-Org/MiniMax-H3',revision,'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors'),
          ('lora_blocks.safetensors','loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',
            'ZhengmingYu/DMAD','439136a1dd4f435b8137a084f73d558d2cecda2a',
            'minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors'),
          ('video_vae.safetensors','minimax_h3_video_vae_int8_convrot.safetensors',
            'Comfy-Org/MiniMax-H3',revision,'vae/minimax_h3_video_vae_int8_convrot.safetensors'),
          ('audio_vae.safetensors','minimax_h3_audio_vae_fp32_clean.safetensors',
            'Comfy-Org/MiniMax-H3',revision,'vae/minimax_h3_audio_vae_fp32.safetensors')]
    plans={}
    for filename,original,repo,commit,remote in rows:
        fixture=args.fixtures/filename
        if not fixture.exists():fixture=args.models/original
        plan=make_plan(args.models/original,fixture,remote)
        plan.update(repository=repo,revision=commit);plans[filename]=plan
    source=(ROOT/'benchmarks/checkpoint_ranges.py').read_text(encoding='utf-8')
    code='''import concurrent.futures,gzip,hashlib,io,json,pathlib,requests,shutil,tarfile,time
OUT=pathlib.Path('/kaggle/working/component_assets');OUT.mkdir(exist_ok=True)
state={'status':'preparing','scope':'CPU-only immutable trained component weights and retained inputs; no GPU/model inference','assets':{}}
def save():(OUT/'assets_receipt.json').write_text(json.dumps(state,indent=2))
save()
'''+source+'\nPLANS='+repr(plans)+'\n'
    code+='''
def one(item):
    name,plan=item;start=time.monotonic()
    url='https://huggingface.co/'+plan['repository']+'/resolve/'+plan['revision']+'/'+plan['remote_file']
    def read(a,b):
        with requests.get(url,params={'download':'true','offset':str(a)},
            headers={'Range':f'bytes={a}-{b-1}'},stream=True,timeout=(15,90)) as response:
            if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {a}-{b-1}/{plan["source_bytes"]}':
                raise RuntimeError('Immutable range mismatch: '+name)
            data=bytearray()
            for part in response.iter_content(4*2**20):
                data.extend(part)
                if len(data)>b-a:raise RuntimeError('Oversized range')
            return data
    assemble(plan,OUT/name,read)
    return name,dict(sha256=plan['sha256'],bytes=plan['bytes'],seconds=time.monotonic()-start,
                     repository=plan['repository'],revision=plan['revision'],remote_file=plan['remote_file'])
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(one,item) for item in PLANS.items()]
        for job in concurrent.futures.as_completed(jobs):
            name,record=job.result();state['assets'][name]=record;save();print('COMPONENT_ASSET_READY',name,flush=True)
    inputs=list(pathlib.Path('/kaggle/input').rglob('input_manifest.json'))
    if len(inputs)!=1:raise RuntimeError('Attach exactly one retained input dataset')
    manifest=json.loads(inputs[0].read_text());target=OUT/'first_block.pt'
    with gzip.open(inputs[0].parent/manifest['file'],'rb') as source,target.open('wb') as destination:
        shutil.copyfileobj(source,destination,4*2**20)
    with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==manifest['uncompressed_sha256'] and target.stat().st_size==manifest['uncompressed_bytes']
    state['assets']['first_block.pt']=dict(sha256=digest,bytes=target.stat().st_size,provenance=manifest['provenance'])
    state['status']='ready'
except BaseException as error:
    state.update(status='blocked',error=repr(error));raise
finally:save();print('COMPONENT_ASSETS_FINAL',state['status'],flush=True)
'''
    ast.parse(code)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')),
        cells=[dict(id='trained-cpu-assets',cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
    (args.output/'experiment.ipynb').write_text(json.dumps(notebook))
    (args.output/'kernel-metadata.json').write_text(json.dumps(dict(id='pamcks/h3-trained-component-cpu-assets',
        title='H3 Trained Component CPU Assets',code_file='experiment.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=True,
        dataset_sources=['pamcks/h3-retained-component-inputs-20261021'],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
    (args.output/'asset_plans.json').write_text(json.dumps(plans,indent=2))
    (args.output/'submission_binding.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(code.encode()).hexdigest(),asset_sha256={n:p['sha256'] for n,p in plans.items()}),indent=2))
    print(args.output)


if __name__=='__main__':main()
