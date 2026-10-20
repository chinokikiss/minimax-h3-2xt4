"""Single bounded T4 session: protocols, current row APIs, attention shortlist."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path('D:/attention_eval/operator_optimization_tp2_20261016')
OUT=BASE/'operator_pass2';OUT.mkdir(exist_ok=True)
manifest=json.loads((BASE/'tp2_operator_manifest.json').read_text())
for name,sha in manifest['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,'TP source changed: regenerate shape audit before submission'
names=['benchmarks/tp2_attention_optimize.py','benchmarks/tp2_nccl_microbench.py','benchmarks/tp2_row_protocol.py',
       'benchmarks/tp2_gemm_operators.py','benchmarks/tp2_timing.py','benchmarks/real_t4_attention_study.py','benchmarks/attention_root_causes.py',
       'patches/tilelang_grouped_tp.py','patches/tilelang_grouped_tp_optimized.py','patches/tilelang_grouped_fused.py',
       'patches/tilelang_grouped_qk.py','patches/tilelang_sparse_attention.py','patches/tilelang_dense_preparation.py',
       'patches/optimized_tp.py','patches/parallel_int8.py']
sources={name:(ROOT/name).read_text(encoding='utf8') for name in names}
for name,source in sources.items():ast.parse(source,filename=name)
code='''import pathlib,subprocess,sys,shutil,time,json,os,signal,traceback
SRC=pathlib.Path('/kaggle/working/source');SRC.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/operator_pass2');OUT.mkdir(exist_ok=True)
deadline=time.monotonic()+690
state={'status':'running','budget_seconds':690,'commands':[],
       'scope':'Actual TP2 operators only; no model weights, no full H3 run, no sparse-versus-dense quality study',
       'profiling_limit':'Nsight Compute previously denied hardware counters; no new counter attempt.'}
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
    print('OPERATOR_STAGE',name,entry,flush=True)
    if entry.get('exit_code')!=0:
        print('STAGE_ERROR_TAIL',name,(OUT/(name+'.log')).read_text(errors='replace')[-4000:],flush=True)
    return entry.get('exit_code')==0
'''
for name,source in sources.items():code+=f'p=SRC/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({source!r})\n'
hashes={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
code+=f"(OUT/'tp2_operator_manifest.json').write_text(json.dumps({manifest!r},indent=2))\n"
code+='''save()
try:
    nvcc=pathlib.Path(shutil.which('nvcc')).resolve()
    state['compiler']=subprocess.check_output([str(nvcc),'--version'],text=True)
    assert 'release 12.8' in state['compiler'],'Pinned SM75 compiler unavailable'
    os.environ.update(CUDACXX=str(nvcc),CUDA_HOME=str(nvcc.parent.parent),CUDA_PATH=str(nvcc.parent.parent),
        OMP_NUM_THREADS='2',NCCL_P2P_DISABLE='1',NCCL_DEBUG='WARN',PYTHONUNBUFFERED='1')
    for name in ('NCCL_PROTO','NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS'):os.environ.pop(name,None)
    assert run([sys.executable,'-m','pip','install','--quiet','tilelang==0.1.15','comfy-kitchen==0.2.37'],'dependencies',180)
    assert run([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'],'runtime',240)
    import torch
    state['hardware']=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())]
    state['torch']=torch.__version__;save()
    assert len(state['hardware'])==2 and all(d['capability']==[7,5] and 'T4' in d['name'] for d in state['hardware'])
    data=list(pathlib.Path('/kaggle/input').rglob('capture_manifest.json'));assert len(data)==1
    launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    comm=[*launcher,str(SRC/'benchmarks/tp2_nccl_microbench.py'),'--manifest',str(OUT/'tp2_operator_manifest.json')]
    row=[*launcher,str(SRC/'benchmarks/tp2_row_protocol.py'),'--manifest',str(OUT/'tp2_operator_manifest.json')]
    attn=[*launcher,str(SRC/'benchmarks/tp2_attention_optimize.py'),'--data',str(data[0].parent)]
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,name,clocks.sm,utilization.gpu,memory.used,power.draw','--format=csv','-lms','500'],stdout=(OUT/'telemetry.csv').open('w'))
    settings=[]
    try:
        for proto in ('default','Simple'):
            folder=OUT/('comm_'+proto)
            if run([*comm,'--protocol',proto,'--output',str(folder)],'comm_'+proto,45):settings.append((proto,None,folder))
        def score(setting):
            return max(next(r['timings']['gpu_median_ms'] for r in json.loads((setting[2]/f'rank{i}.json').read_text())['cases'] if r['name']=='dit_INT32_SUM') for i in range(2))
        if settings:
            best=min(settings,key=score)
            if remaining()>350:
                proto=best[0];folder=OUT/'comm_channels4'
                if run([*comm,'--protocol',proto,'--channels','4','--output',str(folder)],'comm_channels4',45):settings.append((proto,4,folder))
            best=min(settings,key=score)
            (OUT/'communication_selection.json').write_text(json.dumps({'winner':{'protocol':best[0],'channels':best[1]},
                'slower_rank_SUM_ms':[{'protocol':s[0],'channels':s[1],'ms':score(s)} for s in settings]},indent=2))
        else:best=('default',None,None)
        row_settings=[('default',None)]
        if best[:2]!=('default',None):row_settings.append(best[:2])
        row_ok=[]
        for index,(proto,channels) in enumerate(row_settings):
            argv=[*row,'--protocol',proto,'--output',str(OUT/f'row{index}')]
            if channels:argv+=['--channels',str(channels)]
            if run(argv,'row'+str(index),55):row_ok.append((index,proto,channels))
        # Integer SUM protocol changes must preserve every output byte.
        if len(row_ok)==2:
            for rank in range(2):
                a=json.loads((OUT/f'row0/rank{rank}.json').read_text());b=json.loads((OUT/f'row1/rank{rank}.json').read_text())
                assert [r['output_sha256'] for r in a['cases']]==[r['output_sha256'] for r in b['cases']],'Row protocol altered output'
        screen_ok=run([*attn,'--output',str(OUT/'attention_screen')],'attention_screen',min(280,max(30,remaining()-100)))
        selection=OUT/'attention_screen/selection.json'
        repeat_ok=False
        if screen_ok and selection.exists() and remaining()>90:
            repeat_ok=run([*attn,'--selection',str(selection),'--round','1','--output',str(OUT/'attention_repeat')],'attention_repeat',100)
        nsys=shutil.which('nsys') or next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
        state['nsys']=nsys;state['attention_screen_ok']=screen_ok;state['attention_repeat_ok']=repeat_ok;save()
        if nsys and screen_ok and selection.exists() and remaining()>65:
            run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',
                 '--capture-range-end=repeat','--force-overwrite=true','-o',str(OUT/'attention_profile'),*attn,
                 '--selection',str(selection),'--profile','--output',str(OUT/'attention_profiled')],'attention_nsys',90)
        if nsys and row_ok and remaining()>45:
            index,proto,channels=row_ok[-1];argv=[*row,'--protocol',proto,'--profile','--output',str(OUT/'row_profiled')]
            if channels:argv+=['--channels',str(channels)]
            run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',
                 '--capture-range-end=repeat','--force-overwrite=true','-o',str(OUT/'row_profile'),*argv],'row_nsys',60)
        state['status']='complete' if screen_ok and row_ok else 'partial'
    finally:monitor.terminate();monitor.wait(timeout=10)
except Exception as e:
    state['status']='blocked';state['error']=type(e).__name__+': '+str(e)
    (OUT/'exception.txt').write_text(traceback.format_exc())
finally:save();print('OPERATOR_FINAL',state['status'],flush=True)
'''
ast.parse(code)
folder=ROOT/'kaggle_h3-tp2-operator-optimize';folder.mkdir(exist_ok=True)
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    cells=[dict(id='tp2-optimize',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(folder/'experiment.ipynb').write_text(json.dumps(nb),encoding='utf8')
(folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-tp2-operator-optimize',title='H3 TP2 Operator Optimize',
    code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
    dataset_sources=['chinokiki9/h3-tp2-operator-fixtures'],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
(OUT/'submitted_sources.json').write_text(json.dumps(hashes,indent=2));print(folder)
