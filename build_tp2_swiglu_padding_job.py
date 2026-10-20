"""Bounded two-T4 SwiGLU plus padding validation; no datasets or H3 models."""
import ast
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
BASE = Path('D:/attention_eval/operator_optimization_tp2_20261018/swiglu_padding')
OUT = Path('D:/attention_eval/operator_optimization_tp2_20261018/swiglu_padding'); OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((BASE / 'tp2_operator_manifest.json').read_text())
for name, sha in manifest['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, 'Current TP source changed; regenerate audit'
names = ['benchmarks/tp2_swiglu_padding.py', 'benchmarks/tp2_timing.py', 'patches/tilelang_row_quant.py',
         'patches/optimized_tp.py', 'patches/parallel_int8.py', 'patches/tilelang_swiglu_pad.py']
sources = {name: (ROOT / name).read_text(encoding='utf8') for name in names}
hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
for name, source in sources.items(): ast.parse(source, filename=name)
code = '''import pathlib,subprocess,sys,shutil,time,json,os,signal,traceback
SRC=pathlib.Path('/kaggle/working/source');SRC.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/swiglu_padding');OUT.mkdir(exist_ok=True)
deadline=time.monotonic()+560
state={'status':'running','budget_seconds':560,'commands':[],
       'scope':'Current TP2 TE down_proj and DiT FC2 only; SwiGLU padding, quantization and full row APIs with real NCCL. Synthetic inputs, no model construction, dataset upload, H3 generation or sparse policy change.'}
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
def remaining():return deadline-time.monotonic()
def run(argv,name,cap):
    timeout=min(cap,remaining())
    if timeout<12:state['commands'].append({'name':name,'status':'budget_skipped'});save();return False
    start=time.monotonic();entry={'name':name,'cap_seconds':timeout}
    with (OUT/(name+'.log')).open('w') as log:
        p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:entry['exit_code']=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=6)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=4)
            entry['status']='timeout'
    entry['seconds']=time.monotonic()-start;state['commands'].append(entry);save()
    print('SWIGLU_PADDING_STAGE',name,entry,flush=True)
    if entry.get('exit_code')!=0:print('STAGE_ERROR_TAIL',name,(OUT/(name+'.log')).read_text(errors='replace')[-6000:],flush=True)
    return entry.get('exit_code')==0
'''
for name, source in sources.items(): code += f'p=SRC/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({source!r})\n'
code += f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
code += f"(OUT/'tp2_operator_manifest.json').write_text(json.dumps({manifest!r},indent=2))\n"
code += '''save()
try:
    nvcc=pathlib.Path(shutil.which('nvcc')).resolve()
    state['compiler']=subprocess.check_output([str(nvcc),'--version'],text=True)
    assert 'release 12.8' in state['compiler'],'Pinned SM75 compiler unavailable'
    os.environ.update(CUDACXX=str(nvcc),CUDA_HOME=str(nvcc.parent.parent),CUDA_PATH=str(nvcc.parent.parent),
        OMP_NUM_THREADS='2',NCCL_P2P_DISABLE='1',NCCL_DEBUG='WARN',NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4',PYTHONUNBUFFERED='1')
    os.environ.pop('NCCL_PROTO',None)
    assert run([sys.executable,'-m','pip','install','--quiet','tilelang==0.1.15','comfy-kitchen==0.2.37'],'dependencies',180)
    assert run([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'],'runtime',240)
    import torch
    state['hardware']=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())]
    state['torch']=torch.__version__;save()
    assert len(state['hardware'])==2 and all(d['capability']==[7,5] and 'T4' in d['name'] for d in state['hardware'])
    launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    base=[*launcher,str(SRC/'benchmarks/tp2_swiglu_padding.py'),'--manifest',str(OUT/'tp2_operator_manifest.json')]
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,name,clocks.sm,utilization.gpu,memory.used,power.draw','--format=csv','-lms','500'],stdout=(OUT/'telemetry.csv').open('w'))
    try:
        screen_ok=run([*base,'--output',str(OUT/'screen')],'screen',min(230,max(30,remaining()-155)))
        state['screen_ok']=screen_ok;save()
        if screen_ok:
            state['repeat_ok']=run([*base,'--round','1','--output',str(OUT/'repeat')],'repeat',75)
            nsys=shutil.which('nsys') or next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
            state['nsys']=nsys;state['profile_ok']=False
            if nsys and remaining()>50:
                state['profile_ok']=run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                    '--capture-range=cudaProfilerApi','--capture-range-end=repeat','--force-overwrite=true','-o',str(OUT/'swiglu_profile'),
                    *base,'--profile','--output',str(OUT/'profiled')],'nsys',75)
            else:state['profiling_blocker']='Nsight Systems unavailable or insufficient remaining bounded budget'
        state['status']='complete' if screen_ok and state.get('repeat_ok') and state.get('profile_ok') else 'partial'
    finally:monitor.terminate();monitor.wait(timeout=10)
except Exception as e:
    state['status']='blocked';state['error']=type(e).__name__+': '+str(e)
    (OUT/'exception.txt').write_text(traceback.format_exc())
finally:save();print('SWIGLU_PADDING_FINAL',state['status'],flush=True)
'''
ast.parse(code)
folder = ROOT / 'kaggle_h3-tp2-swiglu-padding'; folder.mkdir(exist_ok=True)
notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                cells=[dict(id='tp2-swiglu-padding', cell_type='code', metadata={}, execution_count=None, outputs=[], source=code.splitlines(keepends=True))])
(folder / 'experiment.ipynb').write_text(json.dumps(notebook), encoding='utf8')
(folder / 'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-tp2-swiglu-padding', title='H3 TP2 SwiGLU Padding',
    code_file='experiment.ipynb', language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False,
    enable_internet=True, dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[]), indent=2))
(OUT / 'submitted_sources.json').write_text(json.dumps(hashes, indent=2)); print(folder)
