"""Freeze a <=20 minute private TP2 operator batch, no full model/pipeline."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('D:/attention_eval/operator_optimization_tp2_20261016')
FOLDER=ROOT/'kaggle_h3-tp2-operator-screen';FOLDER.mkdir(exist_ok=True)
names=['benchmarks/tp2_attention_operators.py','benchmarks/tp2_gemm_operators.py','benchmarks/tp2_timing.py','benchmarks/real_t4_attention_study.py',
       'benchmarks/attention_root_causes.py','patches/tilelang_grouped_tp.py','patches/tilelang_grouped_fused.py',
       'patches/tilelang_grouped_qk.py','patches/tilelang_sparse_attention.py','patches/tilelang_dense_preparation.py',
       'patches/optimized_tp.py','patches/parallel_int8.py']
sources={n:(ROOT/n).read_text(encoding='utf8') for n in names}
for name,source in sources.items():ast.parse(source,filename=name)
manifest=json.loads((OUT/'tp2_operator_manifest.json').read_text())
code='''import pathlib,subprocess,sys,shutil,time,json,os,signal,traceback
SRC=pathlib.Path('/kaggle/working/source');SRC.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/tp2_operators');OUT.mkdir(exist_ok=True)
deadline=time.monotonic()+1140
state={'status':'running','scope':'TP2 operators only; no checkpoint download, no H3 pipeline','commands':[],
       'budget_seconds':1140,'profiling_limit':'Nsight Compute counters previously denied ERR_NVGPUCTRPERM; no memory-stall/occupancy claims from Systems alone.'}
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
def run(argv,name,cap):
    remaining=min(cap,deadline-time.monotonic())
    if remaining<10:state['commands'].append({'name':name,'status':'budget_skipped'});save();return False
    start=time.monotonic();entry={'name':name,'cap_seconds':remaining}
    with (OUT/(name+'.log')).open('w') as log:
        p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:entry['exit_code']=p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
            entry['status']='timeout'
    entry['seconds']=time.monotonic()-start;state['commands'].append(entry);save()
    print('OPERATOR_STAGE',name,entry,flush=True)
    return entry.get('exit_code')==0
'''
for name,source in sources.items():
    code+=f'p=SRC/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({source!r})\n'
hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}
code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
code+=f"(OUT/'tp2_operator_manifest.json').write_text(json.dumps({manifest!r},indent=2))\n"
code+='''save()
try:
    # Keep the known SM75-capable compiler while using Kitchen's required runtime.
    nvcc=pathlib.Path(shutil.which('nvcc')).resolve()
    state['compiler']=subprocess.check_output([str(nvcc),'--version'],text=True)
    assert 'release 12.8' in state['compiler'],'SM75 CUDA12.8 compiler unavailable'
    os.environ.update(CUDACXX=str(nvcc),CUDA_HOME=str(nvcc.parent.parent),CUDA_PATH=str(nvcc.parent.parent),
                      OMP_NUM_THREADS='2',NCCL_P2P_DISABLE='1',NCCL_DEBUG='WARN',PYTHONUNBUFFERED='1')
    assert run([sys.executable,'-m','pip','install','--quiet','tilelang==0.1.15','comfy-kitchen==0.2.37'],'dependencies',180)
    assert run([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'],'cu130_runtime',420)
    # Import only after package replacement; the worker processes are fresh.
    import torch
    state['hardware']=[{'name':torch.cuda.get_device_name(i),'capability':list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]
    state['torch']=torch.__version__;save()
    assert len(state['hardware'])==2 and all(d['capability']==[7,5] and 'T4' in d['name'] for d in state['hardware'])
    mounts=list(pathlib.Path('/kaggle/input').rglob('capture_manifest.json'));assert len(mounts)==1
    launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    gemm=[*launcher,str(SRC/'benchmarks/tp2_gemm_operators.py'),'--manifest',str(OUT/'tp2_operator_manifest.json')]
    attn=[*launcher,str(SRC/'benchmarks/tp2_attention_operators.py'),'--data',str(mounts[0].parent)]
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,name,clocks.sm,utilization.gpu,memory.used,power.draw','--format=csv','-lms','500'],stdout=(OUT/'telemetry.csv').open('w'))
    try:
        gemm_ok=run([*gemm,'--output',str(OUT/'gemm')],'gemm',180)
        attn_ok=run([*attn,'--output',str(OUT/'attention')],'attention',480)
        nsys=shutil.which('nsys') or next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
        state['nsys']=nsys;save()
        # One Systems session per family, all ranks/children; compile first calls
        # are outside cudaProfilerApi capture ranges in each worker.
        if nsys:
            for name,base,valid in [('gemm',gemm,gemm_ok),('attention',attn,attn_ok)]:
                if not valid:continue
                run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                     '--capture-range=cudaProfilerApi','--capture-range-end=repeat','--force-overwrite=true',
                     '-o',str(OUT/(name+'_profile')),*base,'--profile','--output',str(OUT/(name+'_profiled'))],name+'_nsys',180)
        else:state['nsys_blocker']='Nsight Systems executable unavailable'
    finally:monitor.terminate();monitor.wait(timeout=10)
    state['status']='complete' if gemm_ok and attn_ok else 'partial'
except Exception as e:
    state['status']='blocked';state['error']=type(e).__name__+': '+str(e)
    (OUT/'exception.txt').write_text(traceback.format_exc())
finally:save();print('OPERATOR_FINAL',state['status'],flush=True)
'''
ast.parse(code)
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(id='tp2-operators',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(FOLDER/'experiment.ipynb').write_text(json.dumps(nb),encoding='utf8')
(FOLDER/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-tp2-operator-screen',title='H3 TP2 Operator Screen',
    code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
    dataset_sources=['chinokiki9/h3-tp2-operator-fixtures'],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
(OUT/'submitted_sources.json').write_text(json.dumps(hashes,indent=2))
print(FOLDER)
