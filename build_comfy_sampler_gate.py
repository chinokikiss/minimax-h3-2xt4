"""Freeze passed local sampler sources into one short physical dual-T4 gate."""
import base64
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile

ROOT=Path(__file__).resolve().parent


def range_plan(original, remote_file):
    from benchmarks.checkpoint_ranges import read_header
    header, raw=read_header(original)
    selected={}
    ranges=[]
    offset=0
    prefix='model.layers.' if '/text_encoders/' in '/'+remote_file else 'blocks.'
    for name,info in header.items():
        if name=='__metadata__':
            selected[name]=info
            continue
        if name.startswith(prefix) and int(name[len(prefix):].split('.')[0])>=2:
            continue
        start,end=info['data_offsets'];size=end-start
        selected[name]={**info,'data_offsets':[offset,offset+size]}
        ranges.append({'key':name,'start':len(raw)+start,'end':len(raw)+end,'target':offset})
        offset+=size
    encoded=json.dumps(selected,separators=(',',':')).encode()
    encoded+=b' '*(-len(encoded)%8)
    raw=struct.pack('<Q',len(encoded))+encoded
    digest=hashlib.sha256(raw)
    with Path(original).open('rb') as stream:
        for item in ranges:
            stream.seek(item['start'])
            remaining=item['end']-item['start']
            while remaining:
                chunk=stream.read(min(remaining,64*2**20))
                if not chunk:raise IOError('Incomplete trained checkpoint')
                digest.update(chunk);remaining-=len(chunk)
            item['target']+=len(raw)
    return dict(remote_file=remote_file,source_bytes=Path(original).stat().st_size,
        bytes=len(raw)+offset,sha256=digest.hexdigest(),header_base64=base64.b64encode(raw).decode(),ranges=ranges)


def main():
    for layers in (2,50):
        proof=ROOT/f'semantic_results/comfy_sampler_{layers}/result.json'
        result=json.loads(proof.read_text())
        if not result['passed'] or result['physical']:
            raise RuntimeError('Both local sampler scopes must pass before submission')
        for name,sha in result['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                raise RuntimeError('Changed since local execution: '+name)
    plans={kind:range_plan(original,remote) for kind,original,remote in [
        ('te','D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors','text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'),
        ('dit','D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors','diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors')]}
    plan_path=ROOT/'semantic_results/comfy_sampler_range_plans.json'
    plan_path.write_text(json.dumps(plans),encoding='utf-8')
    files=list((ROOT/'patches').glob('*.py'))+[ROOT/'benchmarks'/name for name in
        ['qualify_comfy_sampler.py','checkpoint_ranges.py','full_models.py','ref2va_workloads.py',
         'fixture_inputs.py','retest.py','logical_dual_gpu.py']]
    files += [plan_path,ROOT/'semantic_results/comfy_sampler_2/result.json',ROOT/'semantic_results/comfy_sampler_50/result.json']
    archive=io.BytesIO()
    with tarfile.open(fileobj=archive,mode='w:gz') as tar:
        for path in files:tar.add(path,arcname=path.relative_to(ROOT).as_posix())
    payload=base64.b64encode(archive.getvalue()).decode()
    # Reuse the proven bounded setup/cleanup framework; independent slug.
    previous=json.loads((ROOT/'kaggle_host_cache/cache_gate.ipynb').read_text())
    code=''.join(previous['cells'][0]['source'])
    start=code.index('base64.b64decode(')+len('base64.b64decode(')
    end=code.index(')),mode=',start)
    code=code[:start]+repr(payload)+code[end:]
    code=code.replace('/kaggle/working/cache_source','/kaggle/working/sampler_source')
    code=code.replace('/kaggle/working/cache_results','/kaggle/working/sampler_results')
    code=code.replace('Two-request host cache correctness/admission gate, not a performance benchmark',
        'Native ComfyUI conditioning/sampling/cancellation/VAE handoff on two trained TP layers and full Native vision/VAE; no latency benchmark')
    code=code.replace('repaired_fixtures/checkpoint_plans.json','semantic_results/comfy_sampler_range_plans.json')
    code=code.replace("root/'benchmarks/simulate_host_cache.py'", "root/'benchmarks/qualify_comfy_sampler.py'")
    code=code.replace("'--physical','--checkpoints'", "'--physical','--layers','2','--include-video','--checkpoints'")
    code=code.replace("'--output',str(out)","'--output',str(out/'result.json')")
    code=code.replace('CACHE','SAMPLER')
    code=code.replace("('SAMPLER','VERIFIED','Traceback','ERROR')",
        "('SAMPLER','VERIFIED','NATIVE','TP ','Native ComfyUI','Traceback','ERROR')")
    probe = "import json,pathlib,torch; devices=[dict(name=torch.cuda.get_device_name(i),bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]; assert len(devices)==2 and all('T4' in d['name'] for d in devices),devices; pathlib.Path('/kaggle/working/sampler_results/hardware.json').write_text(json.dumps(dict(devices=devices,torch=torch.__version__,cuda=torch.version.cuda)))"
    boundary="    manifest.update(status='running',setup_seconds=time.monotonic()-start);save()"
    code=code.replace(boundary,
        "    command([sys.executable,'-c',"+repr(probe)+"],setup_end-time.monotonic(),'hardware.log')\n"+boundary)
    compile(code,'sampler_gate','exec')
    folder=ROOT/'kaggle_comfy_sampler';folder.mkdir(exist_ok=True)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=previous['metadata'],
        cells=[dict(id='sampler-gate',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    (folder/'sampler_gate.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    metadata=json.loads((ROOT/'kaggle_host_cache/kernel-metadata.json').read_text())
    metadata.update(id='chinokiki9/minimax-h3-comfyui-sampler-gate',title='MiniMax H3 ComfyUI Sampler Gate',code_file='sampler_gate.ipynb')
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(folder)


if __name__=='__main__':main()
