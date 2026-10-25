"""T4 rebenchmark of source-identified Sol preparation/synchronization costs."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];slug='h3-attention-sol-fused-study'
folder=ROOT/('kaggle_'+slug);folder.mkdir(exist_ok=True)
files=['benchmarks/attention_sol_drop_study.py','benchmarks/real_t4_attention_study.py','benchmarks/attention_root_causes.py',
       'patches/tilelang_sol_drop_fused.py','patches/tilelang_grouped_fused.py','patches/tilelang_grouped_qk.py',
       'patches/tilelang_sparse_attention.py','patches/tilelang_dense_preparation.py','patches/tilelang_sage_preparation.py']
code="""import pathlib,json,subprocess,sys,shutil,time
SRC=pathlib.Path('/kaggle/working/source');SRC.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/sol_fused');OUT.mkdir(exist_ok=True)
mounts=list(pathlib.Path('/kaggle/input').rglob('capture_manifest.json'));assert len(mounts)==1
DATA=mounts[0].parent
subprocess.run([sys.executable,'-m','pip','install','tilelang==0.1.15'],check=True,timeout=600)
"""
hashes={}
for name in files:
    content=(ROOT/name).read_text(encoding='utf8');hashes[name]=hashlib.sha256(content.encode()).hexdigest()
    code+=f'p=SRC/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({content!r})\n'
code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
code+='''commands=[]
def run(argv,name,timeout=7200):
    start=time.monotonic()
    with (OUT/(name+'.log')).open('w') as log:p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    commands.append(dict(argv=argv,exit_code=p.returncode,seconds=time.monotonic()-start));(OUT/'commands.json').write_text(json.dumps(commands,indent=2))
    assert p.returncode==0,name
    print('STAGE',name,'complete',flush=True)
base=[sys.executable,str(SRC/'benchmarks/attention_sol_drop_study.py'),'--data',str(DATA)]
with (OUT/'telemetry.csv').open('w') as log:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,name,clocks.sm,utilization.gpu,memory.used,power.draw','--format=csv','-lms','500'],stdout=log)
    try:
        for process in range(3):run([*base,'--process',str(process),'--output',str(OUT/f'process{process}')],f'process{process}')
    finally:monitor.terminate();monitor.wait(timeout=15)
nsys=shutil.which('nsys') or next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
if nsys:
    run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',
         '--capture-range-end=repeat','--force-overwrite=true','-o',str(OUT/'API'),*base,'--profile','--output',str(OUT/'profiled')],'nsys')
    for path in sorted(OUT.glob('API*.nsys-rep')):
        run([nsys,'export','--type','sqlite','--force-overwrite=true','-o',str(path.with_suffix('.sqlite')),str(path)],'export_'+path.stem,300)
        run([nsys,'stats','--force-export=true','--report','cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum','--format','csv',str(path)],'stats_'+path.stem,300)
(OUT/'status.json').write_text(json.dumps(dict(status='complete',scope='GPU0 native H56 Sol conformance and performance; not generation/TP2',nsys=nsys),indent=2))
'''
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(id='sol-fused-study',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(folder/'experiment.ipynb').write_text(json.dumps(nb),encoding='utf8')
(folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/'+slug,title='H3 Attention Sol Fused Study',code_file='experiment.ipynb',language='python',
    kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,dataset_sources=['chinokiki9/h3-native-dit-profile-subset'],
    competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
print(folder)
