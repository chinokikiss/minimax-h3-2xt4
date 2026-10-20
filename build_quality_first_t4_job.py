"""Freeze bounded stage-1/2 suite into a private actual-T4 Kaggle notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--sparse',action='store_true');parser.add_argument('--targeted',action='store_true');args=parser.parse_args()
slug='h3-quality-first-t4-'+('pooling' if args.sparse else 'quantization')
folder=ROOT/('kaggle_'+slug);folder.mkdir(exist_ok=True)
files=['patches/tilelang_sparse_attention.py','patches/tilelang_dense_preparation.py','patches/tilelang_sage_qk.py','patches/tilelang_sage_preparation.py',
       'patches/tilelang_grouped_qk.py','patches/tilelang_scalar_pool.py','benchmarks/attention_root_causes.py',
       'benchmarks/real_t4_attention_study.py','benchmarks/quality_first_t4_microbench.py','benchmarks/test_sage_qk_contract.py']
code="""import pathlib,json,subprocess,sys,shutil
SRC=pathlib.Path('/kaggle/working/source');SRC.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/quality_first');OUT.mkdir(exist_ok=True)
mounts=list(pathlib.Path('/kaggle/input').rglob('capture_manifest.json'));assert len(mounts)==1,mounts
DATA=mounts[0].parent
subprocess.run([sys.executable,'-m','pip','install','tilelang==0.1.15'],check=True,timeout=600)
"""
manifest={}
for name in files:
    content=(ROOT/name).read_text(encoding='utf8');manifest[name]=hashlib.sha256(content.encode()).hexdigest()
    code+=f'p=SRC/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({content!r})\n'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
code+=f"(OUT/'source_manifest.json').write_text(json.dumps(dict(base_commit={commit!r},source_sha256={manifest!r},working_tree_modified=True),indent=2))\n"
code+='''commands=[]
def run(argv,name,timeout=3600):
    with (OUT/(name+'.log')).open('w') as log:p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    commands.append(dict(argv=argv,exit_code=p.returncode));(OUT/'commands.json').write_text(json.dumps(commands,indent=2))
    print('STAGE',name,p.returncode,flush=True)
    assert p.returncode==0,(name,(OUT/(name+'.log')).read_text()[-8000:])
run([sys.executable,str(SRC/'benchmarks/test_sage_qk_contract.py')],'cpu_contract',120)
base=[sys.executable,str(SRC/'benchmarks/quality_first_t4_microbench.py'),'--data',str(DATA)]
'''
if args.targeted:code+="base+=['--modes','sage_warp32_fused,sage_cuda_warp32_contract']\n"
code+='''
with (OUT/'telemetry.csv').open('w') as log:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,name,clocks.sm,clocks.mem,pstate,utilization.gpu,memory.used,power.draw','--format=csv','-lms','200'],stdout=log)
    try:run([*base,'--output',str(OUT/'unprofiled')],'microbench')
    finally:monitor.terminate();monitor.wait(timeout=15)
nsys=shutil.which('nsys') or next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
'''
mode='scalar_pool' if args.sparse else ('sage_warp32_fused' if args.targeted else 'sage_warp32')
code+=f'mode={mode!r}\n'
code+='''if nsys:
    trace=OUT/mode
    run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
        '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',str(trace),
        *base,'--output',str(OUT/'profiled'),'--profile',mode],'nsys')
    run([nsys,'export','--type','sqlite','--force-overwrite=true','-o',str(trace.with_suffix('.sqlite')),str(trace.with_suffix('.nsys-rep'))],'export',240)
    run([nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum','--format','csv',str(trace.with_suffix('.nsys-rep'))],'stats',240)
(OUT/'status.json').write_text(json.dumps(dict(status='complete',generation_quality='not_measured',nsys=nsys,counters='provider blocked'),indent=2))
'''
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf8')
(folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/'+slug,title=slug.replace('-',' ').title(),
    code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
    dataset_sources=['chinokiki9/'+('h3-sparse-failure-diagnostic-head' if args.sparse else 'h3-real-attention-diagnostic-heads')],
    competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
print(folder)
