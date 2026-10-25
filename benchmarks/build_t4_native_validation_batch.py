"""Freeze one bounded physical-T4 batch after CPU core/assets receipts pass."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
import build_kaggle_native_torch_jobs as source_builder


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True);parser.add_argument('--assets-receipt',type=Path,required=True)
    parser.add_argument('--budget-seconds',type=int,default=1500)
    args=parser.parse_args()
    cpu=json.loads(args.preflight.read_text());assets=json.loads(args.assets_receipt.read_text())
    if cpu['status']!='cpu_ready' or assets['status']!='ready':raise RuntimeError('CPU preparation has not passed')
    if args.output.exists():raise FileExistsError('Never overwrite submitted sources')
    args.output.mkdir(parents=True)
    source_builder.BENCHMARKS+=['run_t4_native_core_gates.py','run_physical_layer_bake.py','run_physical_h3_components.py']
    data,hashes=source_builder.capsule();digest=hashlib.sha256(data).hexdigest()
    code='''import base64,hashlib,importlib.util,io,json,os,pathlib,shutil,signal,subprocess,sys,tarfile,time,traceback
OUT=pathlib.Path('/kaggle/working/t4_validation');OUT.mkdir(exist_ok=True)
SRC=pathlib.Path('/kaggle/temp/h3_t4_source');SRC.mkdir(parents=True,exist_ok=True)
state={'status':'preparing','scope':'Bounded physical TP2 component/operator tests; no H3 sampler or full generation','commands':[]}
def save():(OUT/'batch_results.json').write_text(json.dumps(state,indent=2))
'''
    code+='payload=base64.b64decode('+repr(base64.b64encode(data).decode())+')\n'
    code+=f"assert hashlib.sha256(payload).hexdigest()=={digest!r}\n"
    code+="(OUT/'source_capsule.tar.gz.bin').write_bytes(payload)\n"
    code+="with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:archive.extractall(SRC,filter='data')\n"
    code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
    code+=f"EXPECTED_CPU_RECEIPT={hashlib.sha256(args.preflight.read_bytes()).hexdigest()!r}\n"
    code+=f"EXPECTED_ASSET_RECEIPT={hashlib.sha256(args.assets_receipt.read_bytes()).hexdigest()!r}\n"
    code+=f'BUDGET={args.budget_seconds!r}\n'
    code+='''
deadline=time.monotonic()+BUDGET
env=dict(os.environ,OMP_NUM_THREADS='2',NCCL_P2P_DISABLE='1',NCCL_DEBUG='WARN',
 NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4',PYTHONUNBUFFERED='1',
 TILELANG_CACHE_DIR='/kaggle/temp/h3_tilelang_cache',CUDA_CACHE_PATH='/kaggle/temp/h3_cuda_cache')
env.pop('NCCL_PROTO',None)
def run(command,name,cap):
    remaining=min(cap,deadline-time.monotonic());entry=dict(name=name,argv=command)
    if remaining<10:entry['status']='budget_skipped';state['commands'].append(entry);save();return False
    start=time.monotonic()
    with (OUT/(name+'.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:entry['exit_code']=process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            entry.update(exit_code=-9,status='timeout')
    entry['seconds']=time.monotonic()-start;state['commands'].append(entry);save()
    print('T4_VALIDATION_STAGE',name,entry.get('exit_code'),flush=True)
    if entry.get('exit_code')!=0:print((OUT/(name+'.log')).read_text(errors='replace')[-6000:],flush=True)
    return entry.get('exit_code')==0
save()
try:
    receipts=list(pathlib.Path('/kaggle/input').rglob('preflight_receipt.json'))
    matching=[p for p in receipts if hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED_CPU_RECEIPT]
    if len(matching)!=1:raise RuntimeError('Mounted CPU result differs from reviewed receipt')
    cpu=json.loads(matching[0].read_text());assert cpu['status']=='cpu_ready'
    cpu_sources=json.loads((matching[0].parent/'source_hashes.json').read_text())
    gpu_sources=json.loads((OUT/'source_hashes.json').read_text())
    for name,expected in cpu_sources.items():
        if name.startswith('patches/') and gpu_sources.get(name)!=expected:
            raise RuntimeError('Runtime source changed after CPU preflight: '+name)
    asset_files=list(pathlib.Path('/kaggle/input').rglob('assets_receipt.json'))
    matching_assets=[p for p in asset_files if hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED_ASSET_RECEIPT]
    if len(matching_assets)!=1:raise RuntimeError('Mounted trained assets differ from reviewed receipt')
    asset_receipt=json.loads(matching_assets[0].read_text());assert asset_receipt['status']=='ready'
    ASSETS=matching_assets[0].parent
    # All assets were prepared and hashed on CPU; mounted data are read-only.
    state['asset_identity']=asset_receipt
    bundles=list(pathlib.Path('/kaggle/input').rglob('local_conda.json'))
    assert len(bundles)==1
    setup_spec=importlib.util.spec_from_file_location('frozen',bundles[0].parent/'frozen_kaggle_setup.py')
    frozen=importlib.util.module_from_spec(setup_spec);setup_spec.loader.exec_module(frozen)
    frozen.prepare_source(bundles[0].parent,SRC)
    prep_spec=importlib.util.spec_from_file_location('native_prep',SRC/'benchmarks/prepare_kaggle_native_torch.py')
    prep=importlib.util.module_from_spec(prep_spec);prep_spec.loader.exec_module(prep)
    base=prep.inventory();minor=f'{sys.version_info.major}.{sys.version_info.minor}'
    wheels=cpu['target_wheelsets'].get(minor)
    if wheels is None:raise RuntimeError('CPU-prepared core wheels unavailable for native GPU Python '+minor)
    archive=matching[0].parent/wheels['archive']
    with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==wheels['sha256']
    wheelhouse=pathlib.Path('/kaggle/temp/h3_t4_core_wheels');wheelhouse.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:') as tar:tar.extractall(wheelhouse,filter='data')
    overlay=pathlib.Path('/kaggle/temp/h3_t4_core_overlay')
    started=time.monotonic()
    prep.install_overlay(wheelhouse,overlay,cpu['core_pins'],OUT/'offline_core_install.log',wheels['wheels_sha256'])
    state['offline_install_seconds']=time.monotonic()-started
    assert prep.inventory()==base,'Native Kaggle packages were modified'
    profile=dict(profile='kaggle-native-torch-pinned-core-T4',packages=dict(cpu['core_pins']),native_base_packages=base,
                 cpu_preflight_torch=cpu['api']['torch'],python=sys.version)
    for name in ('torch','torchvision','torchaudio','numpy'):
        if name in base:profile['packages'][name]=base[name]
    (OUT/'runtime_profile.json').write_text(json.dumps(profile,indent=2))
    env.update(H3_RUNTIME_LOCK=str(OUT/'runtime_profile.json'),COMFY_ROOT=str(SRC/'ComfyUI'),
        PYTHONPATH=os.pathsep.join([str(overlay),str(SRC),str(SRC/'ComfyUI'),str(SRC/'benchmarks')]))
    nvcc=shutil.which('nvcc')
    if nvcc:
        path=pathlib.Path(nvcc).resolve();state['compiler']=subprocess.check_output([str(path),'--version'],text=True)
        env.update(CUDACXX=str(path),CUDA_HOME=str(path.parent.parent),CUDA_PATH=str(path.parent.parent))
    else:state['compiler_blocker']='No preinstalled nvcc; no GPU-time installer or compiler substitution'
    probe="""import torch,json,importlib.metadata as md,comfy_kitchen as ck
from patches.legacy_runtime import validate_packages,validate_source
import os
validate_packages();validate_source(os.environ['COMFY_ROOT'])
d=dict(torch=torch.__version__,cuda=torch.version.cuda,devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())],kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'),kitchen_cuda_extension=ck.backends.cuda._EXT_AVAILABLE)
print('T4_DEVICE_RECEIPT '+json.dumps(d))
assert len(d['devices'])==2 and all(x['capability']==[7,5] and 'T4' in x['name'] for x in d['devices'])
assert ck.int8_attention_is_available() and ck.backends.cuda._EXT_AVAILABLE
"""
    assert run([sys.executable,'-c',probe],'device_and_core_gate',60),'Native GPU core gate failed'
    line=next(line for line in (OUT/'device_and_core_gate.log').read_text().splitlines() if line.startswith('T4_DEVICE_RECEIPT '))
    state['hardware']=json.loads(line.split(' ',1)[1]);save()
    launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    gates=[*launcher,str(SRC/'benchmarks/run_t4_native_core_gates.py'),'--full-shapes']
    attention=run([*gates,'--section','attention','--output',str(OUT/'attention_gates.json')],
        'attention_operator_gates',360) if nvcc else False
    rows=run([*gates,'--section','rows','--output',str(OUT/'row_gates.json')],
        'row_operator_gates',300) if nvcc else False
    run([*gates,'--section','residency','--output',str(OUT/'residency_gates.json')],'physical_adaptive_residency',120)
    state['operator_gates']=dict(attention=attention,rows=rows);save()
    flow=[*launcher,str(SRC/'benchmarks/run_h3_physical_fp16_flow.py')]
    if attention and rows:
        run([*flow,'te','--checkpoint',str(ASSETS/'te_block0.safetensors'),'--tokens','353',
             '--flows','native_fp32_flow','fp16_fold_c64','--lifetime-ablation','--profile',
             '--output',str(OUT/'te_flow.json')],'te_fp16_and_lifetimes',180)
    if rows:
        common=['--checkpoint',str(ASSETS/'dit_blocks.safetensors'),'--fixture',str(ASSETS/'first_block.pt')]
        run([*flow,'dit',*common,'--tokens','0','--layer','0','--flows','native_fp32_flow','fp16_fold_c64',
             '--profile','--output',str(OUT/'dit_full_flow.json')],'dit_full_fp16',180)
        run([*flow,'dit',*common,'--tokens','513','--layer','49','--lora',str(ASSETS/'lora_blocks.safetensors'),
             '--output',str(OUT/'dit_late_lora_flow.json')],'dit_late_lora_fp16',180)
        run([*flow,'dit',*common,'--tokens','513','--layer','24','--flows','native_fp32_flow','fp16_fold_c64',
             '--output',str(OUT/'dit_mid_flow.json')],'dit_mid_fp16',120)
    components=[sys.executable,str(SRC/'benchmarks/run_physical_h3_components.py')]
    run([*components,'refiner','--assets',str(ASSETS),'--output',str(OUT/'refiner.json')],'native_refiner',120)
    run([sys.executable,str(SRC/'benchmarks/run_physical_layer_bake.py'),
         '--checkpoint',str(ASSETS/'dit_blocks.safetensors'),'--lora',str(ASSETS/'lora_blocks.safetensors'),
         '--output',str(OUT/'layer_bake')],'dual_gpu_layer_bake',300)
    run([*components,'vae','--assets',str(ASSETS),'--output',str(OUT/'vae_idle.json')],'vae_idle_and_native_regressions',420)
    nsys=shutil.which('nsys')
    if not nsys and pathlib.Path('/opt/nvidia').exists():
        nsys=next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
    state['profiling']=dict(nsys=nsys,ncu_hardware_counters='Not collected; historical ERR_NVGPUCTRPERM, no repeated GPU counter probe without changed permissions')
    if nsys and attention and rows and deadline-time.monotonic()>90:
        run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--force-overwrite=true',
             '-o',str(OUT/'te_nsys'),*launcher,str(SRC/'benchmarks/run_h3_physical_fp16_flow.py'),'te',
             '--checkpoint',str(ASSETS/'te_block0.safetensors'),'--tokens','353','--repeats','3',
             '--flows','native_fp32_flow','fp16_fold_c64','--output',str(OUT/'te_nsys_flow.json')],'te_nsight_systems',90)
    elif not nsys:state['profiling']['nsys_blocker']='Nsight Systems executable unavailable'
    required=[e for e in state['commands'] if e['name']!='te_nsight_systems']
    state['status']='passed' if attention and rows and all(e.get('exit_code')==0 for e in required) else 'partial_or_failed'
except BaseException as error:
    state.update(status='blocked',error=repr(error));(OUT/'exception.txt').write_text(traceback.format_exc())
finally:
    state['seconds_used']=BUDGET-max(0,deadline-time.monotonic());save()
    print('T4_VALIDATION_FINAL',state['status'],flush=True)
'''
    ast.parse(code)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')),
        cells=[dict(id='native-core-t4-batch',cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
    (args.output/'experiment.ipynb').write_text(json.dumps(notebook))
    (args.output/'kernel-metadata.json').write_text(json.dumps(dict(id='pamcks/h3-native-torch-t4-validation',
        title='H3 Native Torch T4 Validation',code_file='experiment.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,docker_image_pinning_type='original',
        dataset_sources=['pamcks/h3-frozen-runtime-20261021'],
        kernel_sources=['pamcks/h3-native-torch-cpu-preflight','pamcks/h3-trained-component-cpu-assets'],
        competition_sources=[],model_sources=[]),indent=2))
    (args.output/'source_binding.json').write_text(json.dumps(dict(capsule_sha256=digest,files_sha256=hashes,
        cpu_receipt_sha256=hashlib.sha256(args.preflight.read_bytes()).hexdigest(),
        assets_receipt_sha256=hashlib.sha256(args.assets_receipt.read_bytes()).hexdigest()),indent=2))
    print(args.output)


if __name__=='__main__':main()
