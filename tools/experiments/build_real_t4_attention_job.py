"""Freeze the actual H3 attention replay and Nsight capture into a private job."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    folder=ROOT/'kaggle_real_t4_attention';folder.mkdir(exist_ok=True)
    files=['patches/tilelang_sparse_attention.py','benchmarks/attention_root_causes.py',
           'benchmarks/real_t4_attention_study.py','benchmarks/test_attention_root_causes.py',
           'patches/tilelang_dense_preparation.py','patches/tilelang_grouped_qk.py','patches/tilelang_scalar_pool.py',
           'benchmarks/real_t4_targeted_improvements.py','benchmarks/real_t4_scalar_pool.py']
    code='''import json, pathlib, subprocess, sys, time, shutil, traceback
SRC=pathlib.Path('/kaggle/working/h3_real_t4_source');SRC.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/h3_real_t4_results');OUT.mkdir(exist_ok=True)
mounts=list(pathlib.Path('/kaggle/input').rglob('capture_manifest.json'))
print('DATA_MOUNTS',mounts,flush=True)
assert len(mounts)==1,mounts
DATA=mounts[0].parent
subprocess.run([sys.executable,'-m','pip','install','tilelang==0.1.15'],check=True,timeout=600)
'''
    manifest={}
    for relative in files:
        content=(ROOT/relative).read_text(encoding='utf-8')
        manifest[relative]=hashlib.sha256(content.encode()).hexdigest()
        code+=f"p=SRC/{relative!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({content!r})\n"
    code+=f"(OUT/'source_manifest.json').write_text(json.dumps({manifest!r},indent=2))\n"
    code+='''study=SRC/'benchmarks/real_t4_attention_study.py'
commands=[]
def run(args,name,timeout=7200,required=True):
    start=time.monotonic()
    print('STAGE_BEGIN',name,flush=True)
    with (OUT/(name+'.log')).open('w') as log:
        proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    commands.append(dict(name=name,argv=args,exit_code=proc.returncode,seconds=time.monotonic()-start))
    (OUT/'commands.json').write_text(json.dumps(commands,indent=2))
    print('STAGE_END',name,commands[-1],flush=True)
    if required:assert proc.returncode==0,(name,(OUT/(name+'.log')).read_text()[-10000:])
run([sys.executable,str(SRC/'benchmarks/test_attention_root_causes.py')],'diagnostic_self_check',120)
base=[sys.executable,str(study),'--data',str(DATA)]
run([*base,'--output',str(OUT/'benchmarks')],'real_shape_benchmarks',5400)
nsys=shutil.which('nsys')
if not nsys:
    nsys=next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
ncu=shutil.which('ncu')
profiler={'nsys':nsys,'ncu':ncu,'counter_blocker':None}
for component,file,heads in [('te','te_layer10_call-1.pt',64),('dit','dit_layer25_call09.pt',28),('video','video_layer18_call15.pt',32)]:
    modes=['native','dense_int8','dense_int8_pv']+(['sol_tau0','sol_tau0_int8_pv','topk50'] if component=='dit' else [])
    for mode in modes:
        name=component+'_'+mode
        selected=[*base,'--file',file,'--heads',str(heads),'--mode',mode,'--profile','--repeats','3',
                  '--output',str(OUT/'profile_runs'/name)]
        if nsys:
            trace=OUT/'nsys'/name;trace.parent.mkdir(exist_ok=True)
            run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                 '--capture-range=cudaProfilerApi','--capture-range-end=stop',
                 '--force-overwrite=true','-o',str(trace),*selected],'nsys_'+name,900,False)
            rep=trace.with_suffix('.nsys-rep')
            if rep.exists():
                run([nsys,'export','--type','sqlite','--force-overwrite=true','-o',str(trace.with_suffix('.sqlite')),str(rep)],
                    'nsys_export_'+name,240,False)
                run([nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum',
                     '--format','csv',str(rep)],'nsys_stats_'+name,240,False)
        else:profiler['nsys_blocker']='Nsight Systems executable missing in this allocation'
        if ncu and component=='dit' and mode=='dense_int8':
            target=OUT/'ncu';target.mkdir(exist_ok=True)
            run([ncu,'--set','basic','--launch-count','1','--kernel-name','regex:main',
                 '--force-overwrite','-o',str(target/'dit_dense'),*selected],'ncu_attempt',900,False)
            evidence=(OUT/'ncu_attempt.log').read_text()
            if 'ERR_NVGPUCTRPERM' in evidence:
                profiler['counter_blocker']='ERR_NVGPUCTRPERM: provider denies NVIDIA performance counters'
            elif not (target/'dit_dense.ncu-rep').exists():profiler['counter_blocker']='No Nsight Compute report; see ncu_attempt.log'
(OUT/'profiler_availability.json').write_text(json.dumps(profiler,indent=2))
improvement=[sys.executable,str(SRC/'benchmarks/real_t4_targeted_improvements.py'),'--data',str(DATA),'--representatives']
run([*improvement,'--output',str(OUT/'improvements_full_heads')],'improvements_full_heads',3600)
run([*improvement,'--dit-rank-heads','28','--output',str(OUT/'improvements_rank28')],'improvements_rank28',3600)
scalar=[sys.executable,str(SRC/'benchmarks/real_t4_scalar_pool.py'),'--data',str(DATA)]
run([*scalar,'--output',str(OUT/'scalar_pool_full_heads')],'scalar_pool_full_heads',3600)
run([*scalar,'--heads','28','--output',str(OUT/'scalar_pool_rank28')],'scalar_pool_rank28',3600)
run([*base,'--output',str(OUT/'accuracy'),'--accuracy'],'real_accuracy_root_causes',10800)
(OUT/'status.json').write_text(json.dumps(dict(status='operator_study_complete',
    full_pipeline_status='pending: operator replay is not a full 2xT4 pipeline benchmark',
    accuracy_capture_scope='5 TE layers; 5 DiT layers at 3 denoising calls; 5 VAE layers across 30 decoder passes',
    performance_counter_blocker=profiler['counter_blocker']),indent=2))
'''
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
                  cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    (folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-real-t4-attention-root-causes',
        title='H3 Real T4 Attention Root Causes',code_file='experiment.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
        dataset_sources=['chinokiki9/h3-native-attention-tokyo-20261013'],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
    print(folder)


if __name__=='__main__':main()
