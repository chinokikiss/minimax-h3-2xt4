"""Private T4 job for profiler-driven preparation and QK accuracy candidates."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2];folder=root/'kaggle_real_t4_improvements';folder.mkdir(exist_ok=True)
files=['patches/tilelang_sparse_attention.py','patches/tilelang_dense_preparation.py','patches/tilelang_grouped_qk.py',
       'benchmarks/attention_root_causes.py','benchmarks/real_t4_attention_study.py','benchmarks/real_t4_targeted_improvements.py',
       'benchmarks/test_dense_preparation_contract.py']
code='''import json,pathlib,subprocess,sys,shutil
SRC=pathlib.Path('/kaggle/working/improvement_source');SRC.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/improvement_results');OUT.mkdir(exist_ok=True)
mounts=list(pathlib.Path('/kaggle/input').rglob('capture_manifest.json'));assert len(mounts)==1,mounts;DATA=mounts[0].parent
subprocess.run([sys.executable,'-m','pip','install','tilelang==0.1.15'],check=True,timeout=600)
'''
manifest={}
for name in files:
    content=(root/name).read_text(encoding='utf-8');manifest[name]=hashlib.sha256(content.encode()).hexdigest()
    code+=f"p=SRC/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({content!r})\n"
code+=f"(OUT/'source_manifest.json').write_text(json.dumps({manifest!r},indent=2))\n"
code+='''commands=[]
def run(args,name,timeout=3600):
    with (OUT/(name+'.log')).open('w') as log:
        p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    commands.append(dict(name=name,argv=args,exit_code=p.returncode))
    (OUT/'commands.json').write_text(json.dumps(commands,indent=2))
    print('IMPROVEMENT_STAGE',name,p.returncode,flush=True)
    assert p.returncode==0,(name,(OUT/(name+'.log')).read_text()[-6000:])
run([sys.executable,str(SRC/'benchmarks/test_dense_preparation_contract.py')],'preparation_contract',120)
base=[sys.executable,str(SRC/'benchmarks/real_t4_targeted_improvements.py'),'--data',str(DATA)]
with (OUT/'hardware_telemetry.csv').open('w') as telemetry:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,clocks.sm,clocks.mem,pstate,utilization.gpu,utilization.memory,memory.used,power.draw',
        '--format=csv','-lms','100'],stdout=telemetry,stderr=subprocess.STDOUT)
    try:run([*base,'--output',str(OUT/'benchmarks')],'real_targeted_improvements')
    finally:monitor.terminate();monitor.wait(timeout=15)
nsys=shutil.which('nsys') or next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
if nsys:
    for mode in ['baseline','fast_dense','group32']:
        trace=OUT/mode
        run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',
             '--capture-range-end=stop','--force-overwrite=true','-o',str(trace),*base,'--profile',mode,
             '--output',str(OUT/'profiles'/mode)],'nsys_'+mode)
        run([nsys,'export','--type','sqlite','--force-overwrite=true','-o',str(trace.with_suffix('.sqlite')),
             str(trace.with_suffix('.nsys-rep'))],'export_'+mode,240)
        run([nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum','--format','csv',
             str(trace.with_suffix('.nsys-rep'))],'stats_'+mode,240)
(OUT/'status.json').write_text(json.dumps(dict(status='complete',scope='diagnostic heads; full deployment study remains pending',
    counter_blocker='ERR_NVGPUCTRPERM confirmed separately'),indent=2))
'''
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
(folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-real-t4-targeted-improvements',title='H3 Real T4 Targeted Improvements',
    code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,
    enable_internet=True,dataset_sources=['chinokiki9/h3-real-attention-diagnostic-heads'],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
print(folder)
