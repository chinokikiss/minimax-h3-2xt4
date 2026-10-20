"""Source-only physical dual-T4 job; no local weight upload or full downloads."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'benchmarks'))
from checkpoint_ranges import read_header


def plan(path,repo,revision,remote_file,select):
    header,prefix=read_header(path);entries={};offset=0;ranges=[]
    for key in sorted(k for k in header if k!='__metadata__' and select(k)):
        info=header[key];a,b=info['data_offsets'];size=b-a
        entries[key]=dict(dtype=info['dtype'],shape=info['shape'],data_offsets=[offset,offset+size])
        ranges.append(dict(key=key,start=len(prefix)+a,end=len(prefix)+b,destination=offset));offset+=size
    if not entries:raise ValueError('Empty fixture selection')
    raw=json.dumps(entries,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    prefix=struct.pack('<Q',len(raw))+raw;digest=hashlib.sha256(prefix)
    with Path(path).open('rb') as stream:
        for item in ranges:
            item['target']=len(prefix)+item.pop('destination');stream.seek(item['start'])
            left=item['end']-item['start']
            while left:
                data=stream.read(min(left,4*2**20))
                if not data:raise IOError('Incomplete local identity source')
                digest.update(data);left-=len(data)
    return dict(repo=repo,revision=revision,remote_file=remote_file,source_bytes=Path(path).stat().st_size,
                bytes=len(prefix)+offset,sha256=digest.hexdigest(),header_base64=base64.b64encode(prefix).decode(),ranges=ranges)


def main(native_order=False,storage=False,a_options=False,row_memory=False):
    native_order=native_order or storage or a_options or row_memory
    proofs={}
    if native_order:
        proof_names=('h3_lora_storage_single.json','h3_lora_storage_multi.json','h3_lora_storage_long.json') if storage else ('h3_lora_native_order_single.json','h3_lora_native_order_multi.json','h3_lora_native_order_long.json')
        if a_options:proof_names=('h3_lora_options_exact_single.json','h3_lora_options_exact_multi.json','h3_lora_options_exact_long.json')
        if row_memory:proof_names=('h3_row_memory_exact_single.json','h3_row_memory_exact_multi.json','h3_row_memory_exact_long.json')
        for name in proof_names:
            proof=json.loads((ROOT/'semantic_results'/name).read_text())
            if not proof['passed'] or not all(r.get('is_bit_exact',False) for r in proof['cases'] if r.get('required',True)):
                raise RuntimeError('Local byte qualification must pass before T4 submission: '+name)
            for source,sha in proof['source_sha256'].items():
                if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=sha:
                    raise RuntimeError('Source changed after local gate: '+source)
            proofs[name]=proof
    if a_options:
        name='h3_lora_options_local/results.json'
        proof=json.loads((ROOT/'semantic_results'/name).read_text())
        if not proof['passed'] or not proof['native_exact_passed']:raise RuntimeError('Optional A local gate must pass first')
        for source,sha in proof['source_sha256'].items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=sha:raise RuntimeError('Source changed after optional gate: '+source)
        proofs[name]=proof
    if row_memory:
        for name in ('h3_row_memory_local/results.json','h3_row_memory_payload.json'):
            proof=json.loads((ROOT/'semantic_results'/name).read_text())
            if not proof['passed'] or (name.endswith('results.json') and not proof['previous_row_exact_passed']):raise RuntimeError('Row memory local gate must pass first')
            for source,sha in proof['source_sha256'].items():
                if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=sha:raise RuntimeError('Source changed after row memory gate: '+source)
            proofs[name]=proof
    prefixes=('blocks.0.','blocks.25.','blocks.49.') if native_order else ('blocks.0.',)
    lora_prefixes=tuple('diffusion_model.'+p for p in prefixes)
    plans=dict(dit=plan('D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
        'Comfy-Org/MiniMax-H3','e5eb578a89295337b8ff433a035929ce0279e0b6',
        'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',lambda k:k.startswith(prefixes) or k=='adaln_t_table'),
        lora=plan('D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',
        'ZhengmingYu/DMAD','439136a1dd4f435b8137a084f73d558d2cecda2a',
        'minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',lambda k:k.startswith(lora_prefixes)))
    names=['benchmarks/physical_h3_reduction.py','benchmarks/fetch_h3_physical_fixtures.py','benchmarks/checkpoint_ranges.py',
           'benchmarks/fixture_inputs.py','benchmarks/retest.py','benchmarks/merge_h3_lora.py',
           'patches/optimized_tp.py','patches/parallel_int8.py','patches/quantized_collectives.py','patches/h3_lora.py',
           'patches/windowed_tp.py','patches/weight_residency.py','patches/diagnostic_metrics.py',
           'patches/tp_dit.py','patches/tp_qwen3vl.py','patches/rank_local_int8.py']
    if native_order:names.append('benchmarks/physical_h3_lora_repair.py')
    if storage:names+=['benchmarks/physical_h3_lora_storage.py','benchmarks/native_order_lora_baseline.py']
    if a_options:names+=['benchmarks/h3_lora_options_audit.py','benchmarks/logical_dual_gpu.py','patches/comfy_tp_runtime.py','custom_nodes/minimax_h3_tp2/__init__.py']
    if row_memory:names+=['benchmarks/row_partial_memory_audit.py','benchmarks/row_partial_memory_baseline.py','benchmarks/logical_dual_gpu.py','patches/comfy_tp_runtime.py','custom_nodes/minimax_h3_tp2/__init__.py']
    sources={n:(ROOT/n).read_text(encoding='utf8') for n in names}
    for n,s in sources.items():ast.parse(s,filename=n)
    hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}
    code='''import pathlib,subprocess,sys,os,time,json,signal,traceback,shutil
ROOT=pathlib.Path('/kaggle/working/h3_source');ROOT.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/h3_reduction_results');OUT.mkdir(exist_ok=True)
deadline=time.monotonic()+1740
state=dict(status='setup',budget_seconds=1740,automatic_retry=False,commands=[],
    scope='Physical 2xT4 learned FFN and compact block0 LoRA diagnostics; no full pipeline claim; source-only upload')
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
def run(argv,label,cap):
    seconds=min(cap,deadline-time.monotonic())
    if seconds<10:raise TimeoutError('Physical run budget exhausted')
    entry=dict(label=label,cap_seconds=seconds);start=time.monotonic()
    print('PHYSICAL_STAGE_START',label,flush=True)
    with (OUT/(label+'.log')).open('w') as log:
        p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:entry['exit_code']=p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=6)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
            entry['status']='timeout'
    entry['seconds']=time.monotonic()-start;state['commands'].append(entry);save()
    print('PHYSICAL_STAGE_END',label,entry,flush=True)
    if entry.get('exit_code')!=0:
        print((OUT/(label+'.log')).read_text(errors='replace')[-7000:],flush=True)
        raise RuntimeError('Physical stage failed: '+label)
'''
    for n,s in sources.items():code+=f'p=ROOT/{n!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({s!r})\n'
    code+=f"(ROOT/'fixture_plans.json').write_text(json.dumps({plans!r},indent=2))\n"
    code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
    if native_order:code+=f"(OUT/'local_qualification.json').write_text(json.dumps({proofs!r},indent=2))\n"
    code+='''save()
try:
    os.environ.update(OMP_NUM_THREADS='2',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_DEBUG='WARN',
                      PYTHONUNBUFFERED='1',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
                      COMFY_ROOT=str(ROOT/'ComfyUI'))
    run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir',
        'comfy-kitchen==0.2.37','comfy-aimdo==0.5.5','safetensors==0.8.0','huggingface-hub==1.12.0',
        'einops','transformers','torchsde','sentencepiece','psutil'],'dependencies',240)
    run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--upgrade',
        'torch==2.11.0','torchvision==0.26.0','torchaudio==2.11.0',
        '--index-url','https://download.pytorch.org/whl/cu130'],'runtime',480)
    run(['git','clone','--filter=blob:none','https://github.com/Comfy-Org/ComfyUI.git',str(ROOT/'ComfyUI')],'comfy_clone',120)
    run(['git','-C',str(ROOT/'ComfyUI'),'checkout','d49e888586dd8ae012c0667b33466b815fee07f7'],'comfy_pin',60)
    run([sys.executable,'-u',str(ROOT/'benchmarks/fetch_h3_physical_fixtures.py'),str(ROOT/'fixture_plans.json'),str(ROOT/'fixtures')],'pinned_fixtures',360)
    state['status']='physical_benchmark';save()
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(ROOT/'benchmarks/physical_h3_reduction.py'),'--checkpoint',str(ROOT/'fixtures/dit.safetensors'),
        '--lora',str(ROOT/'fixtures/lora.safetensors'),'--output',str(OUT),'--profile']
    run(command,'dual_t4',600)
    result=json.loads((OUT/'results.json').read_text())
    state.update(status=result['status'],native_exact_passed=result.get('native_exact_passed'),
                 projection_cases=len(result['projection_cases']),block_cases=len(result['block_cases']))
    # Checkpoint fixtures and the offline merged test artifact are derived,
    # reproducible intermediates. Keep hashes/receipts and compact traces only.
    for p in (OUT/'offline_block0.safetensors',ROOT/'fixtures/dit.safetensors',ROOT/'fixtures/lora.safetensors'):
        if p.is_file():p.unlink()
except BaseException as error:
    state.update(status='failed',error_type=type(error).__name__)
    (OUT/'exception.txt').write_text(traceback.format_exc())
finally:save();print('PHYSICAL_FINAL',state['status'],flush=True)
'''
    ast.parse(code)
    if native_order:
        code=code.replace('h3_reduction_results','h3_lora_repair_results')
        code=code.replace("str(ROOT/'benchmarks/physical_h3_reduction.py')","str(ROOT/'benchmarks/physical_h3_lora_repair.py')")
        code=code.replace('Physical 2xT4 learned FFN and compact block0 LoRA diagnostics; no full pipeline claim; source-only upload',
            'Locally byte-qualified Native-order LoRA on physical 2xT4; blocks0/25/49 and long learned FFN; no full pipeline claim')
        ast.parse(code)
    if storage:
        code=code.replace('h3_lora_repair_results','h3_lora_storage_results').replace("str(ROOT/'benchmarks/physical_h3_lora_repair.py')","str(ROOT/'benchmarks/physical_h3_lora_storage.py')")
        ast.parse(code)
    if a_options:
        code=code.replace('h3_lora_repair_results','h3_lora_options_results')
        code=code.replace("str(ROOT/'benchmarks/physical_h3_lora_repair.py'),'--checkpoint'",
                          "str(ROOT/'benchmarks/h3_lora_options_audit.py'),'--physical','--checkpoint'")
        ast.parse(code)
    if row_memory:
        code=code.replace('h3_lora_repair_results','h3_row_memory_results')
        code=code.replace("str(ROOT/'benchmarks/physical_h3_lora_repair.py'),'--checkpoint'",
                          "str(ROOT/'benchmarks/row_partial_memory_audit.py'),'--physical','--checkpoint'")
        ast.parse(code)
    folder=ROOT/('kaggle_h3-row-partial-memory' if row_memory else 'kaggle_h3-optional-a-schemes' if a_options else 'kaggle_h3-native-order-lora-storage' if storage else 'kaggle_h3-native-order-lora-repair' if native_order else 'kaggle_h3-optional-int8-lora');folder.mkdir(exist_ok=True)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(id='h3-physical-reduction',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    (folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(
        id='chinokiki9/h3-row-partial-memory' if row_memory else 'chinokiki9/h3-optional-a-schemes' if a_options else 'chinokiki9/h3-native-order-lora-storage' if storage else 'chinokiki9/h3-native-order-lora-repair' if native_order else 'chinokiki9/h3-optional-int8-and-lora-physical-t4',
        title='H3 Row Partial Memory' if row_memory else 'H3 Optional A Schemes' if a_options else 'H3 Native Order LoRA Storage' if storage else 'H3 Native Order LoRA Repair' if native_order else 'H3 Optional INT8 and LoRA Physical T4',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,
        enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
    (folder/'fixture_plans.json').write_text(json.dumps(plans,indent=2))
    print('SOURCE_ONLY_JOB',str(folder),'download_bytes',sum(p['bytes'] for p in plans.values()))


if __name__=='__main__':main(native_order='--native-order' in sys.argv,storage='--storage' in sys.argv,a_options='--a-options' in sys.argv,row_memory='--row-memory' in sys.argv)
