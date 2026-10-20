"""Small private actual-activation gate while the full dataset upload proceeds.

Original full sequence retained; one known diagnostic head per component. This
is a correctness/profiling gate and never deployment-head-count performance.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import zlib
import torch
import numpy as np
import sys

ROOT=Path(__file__).resolve().parent
CAPTURES=Path('D:/attention_eval/tokyo_ref2va_20261013')
tiny='--tiny' in sys.argv
dataset_mode='--dataset' in sys.argv
sparse_failure='--sparse-failure' in sys.argv
if sparse_failure:dataset_mode=True
small='--small' in sys.argv or tiny
suffix='_sparse_failure' if sparse_failure else ('_dataset' if dataset_mode else ('_tiny' if tiny else ('_small' if small else '')))
FOLDER=ROOT/('kaggle_real_t4_replay_gate'+suffix)
FOLDER.mkdir(exist_ok=True)
source_files=['patches/tilelang_sparse_attention.py','benchmarks/attention_root_causes.py',
              'benchmarks/real_t4_attention_study.py','benchmarks/test_attention_root_causes.py']
captures=json.loads((CAPTURES/'capture_manifest.json').read_text())['records']
chosen=[('te_layer10_call-1.pt',40),('dit_layer49_call09.pt',10),('video_layer00_call00.pt',5)]
if sparse_failure:chosen=[('dit_layer25_call09.pt',14)]
if small:chosen=[chosen[0],chosen[2]]
if tiny:chosen=chosen[:1]
records=[]
gate_dataset=Path('D:/attention_eval/t4_redo_20261013/'+('sparse_failure_dataset' if sparse_failure else 'gate_dataset'))
if dataset_mode:gate_dataset.mkdir(parents=True,exist_ok=True)
code='''import base64, hashlib, json, pathlib, subprocess, sys, zlib, shutil, numpy as np
SRC=pathlib.Path('/kaggle/working/h3_gate_source');SRC.mkdir(exist_ok=True)
DATA=pathlib.Path('/kaggle/working/h3_gate_data');DATA.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/h3_gate_results');OUT.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-m','pip','install','tilelang==0.1.15'],check=True,timeout=600)
'''
source_manifest={}
for name in source_files:
    content=(ROOT/name).read_text(encoding='utf-8')
    source_manifest[name]=hashlib.sha256(content.encode()).hexdigest()
    code+=f"p=SRC/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({content!r})\n"
for name,head in chosen:
    record=next(r for r in captures if Path(r['file']).name==name)
    full=torch.load(CAPTURES/record['file'],map_location='cpu',weights_only=True)
    group=full['q'].shape[1]//full['k'].shape[1]
    data={**full,'q':full['q'][:,head:head+1].contiguous(),
          'k':full['k'][:,head//group:head//group+1].contiguous(),
          'v':full['v'][:,head//group:head//group+1].contiguous(),
          'native_output':full['native_output'][:,head:head+1].contiguous()}
    # Clone slices so torch.save does not serialize their full original storage.
    for tensor in ['q','k','v','native_output']:data[tensor]=data[tensor].clone()
    stream=io.BytesIO();torch.save(data,stream);payload=stream.getvalue()
    if dataset_mode:(gate_dataset/name).write_bytes(payload)
    # Lossless byte-plane transport compression, with exact .pt hash on restore.
    aligned=payload+b'\0'*((-len(payload))%4)
    shuffled=np.frombuffer(aligned,dtype=np.uint8).reshape(-1,4).T.copy().tobytes()
    packed=base64.b64encode(zlib.compress(shuffled,level=6)).decode()
    code+=f"shuffled=zlib.decompress(base64.b64decode({packed!r}));payload=np.frombuffer(shuffled,dtype=np.uint8).reshape(4,-1).T.copy().tobytes()[:{len(payload)}];assert hashlib.sha256(payload).hexdigest()=={hashlib.sha256(payload).hexdigest()!r};(DATA/{name!r}).write_bytes(payload)\n"
    records.append({**record,'file':name,'q_shape':list(data['q'].shape),'k_shape':list(data['k'].shape),
        'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),
        'source_capture_sha256':record['sha256'],'source_head':head,
        'scope':'single diagnostic native head; full original sequence; not deployment performance'})
gate_manifest={'records':records}
code+=f"(DATA/'capture_manifest.json').write_text(json.dumps({gate_manifest!r},indent=2))\n"
code+=f"(OUT/'source_manifest.json').write_text(json.dumps({source_manifest!r},indent=2))\n"
code+='''base=[sys.executable,str(SRC/'benchmarks/real_t4_attention_study.py'),'--data',str(DATA)]
commands=[]
def run(args,name,timeout=1800):
    with (OUT/(name+'.log')).open('w') as log:
        p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    commands.append(dict(name=name,argv=args,exit_code=p.returncode))
    (OUT/'commands.json').write_text(json.dumps(commands,indent=2))
    print('GATE_STAGE',name,p.returncode,flush=True)
    assert p.returncode==0,(name,(OUT/(name+'.log')).read_text()[-6000:])
run([sys.executable,str(SRC/'benchmarks/test_attention_root_causes.py')],'diagnostic_self_check',120)
for file in ['te_layer10_call-1.pt','dit_layer49_call09.pt','video_layer00_call00.pt']:
    run([*base,'--file',file,'--heads','1','--repeats','7','--output',str(OUT/'benchmarks'/file[:-3])],'bench_'+file[:-3])
run([*base,'--output',str(OUT/'accuracy'),'--accuracy'],'accuracy_root_causes',3600)
nsys=shutil.which('nsys') or next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
if nsys:
    for mode in ['dense_int8','sol_tau0','dense_int8_pv']:
        trace=OUT/mode
        run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','-o',str(trace),
            *base,'--file','dit_layer49_call09.pt','--heads','1','--mode',mode,'--profile','--repeats','3',
            '--output',str(OUT/'profiles'/mode)],'nsys_'+mode)
        run([nsys,'export','--type','sqlite','--force-overwrite=true','-o',str(trace.with_suffix('.sqlite')),
             str(trace.with_suffix('.nsys-rep'))],'export_'+mode,240)
        run([nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum','--format','csv',
             str(trace.with_suffix('.nsys-rep'))],'stats_'+mode,240)
(OUT/'status.json').write_text(json.dumps(dict(status='complete',scope='three real diagnostic heads, full sequence; not deployment benchmarks',
    nsys_available=bool(nsys),ncu_blocker='ERR_NVGPUCTRPERM confirmed in capability probe'),indent=2))
'''
if small:
    code=code.replace("['te_layer10_call-1.pt','dit_layer49_call09.pt','video_layer00_call00.pt']",
                      "['te_layer10_call-1.pt','video_layer00_call00.pt']")
    code=code.replace("['dense_int8','sol_tau0','dense_int8_pv']", "['dense_int8','dense_int8_pv']")
    code=code.replace("'--file','dit_layer49_call09.pt'", "'--file','te_layer10_call-1.pt'")
    code=code.replace('three real diagnostic heads','two real diagnostic heads (TE and Video VAE)')
if tiny:
    code=code.replace("['te_layer10_call-1.pt','video_layer00_call00.pt']", "['te_layer10_call-1.pt']")
    code=code.replace('two real diagnostic heads (TE and Video VAE)', 'one real TE diagnostic head')
if dataset_mode:
    (gate_dataset/'capture_manifest.json').write_text(json.dumps(gate_manifest,indent=2))
    (gate_dataset/'dataset-metadata.json').write_text(json.dumps(dict(
        title='H3 Real Attention Diagnostic Heads',id='chinokiki9/h3-real-attention-diagnostic-heads',
        licenses=[{'name':'other'}]),indent=2))
    code='\n'.join(line for line in code.splitlines() if not line.startswith('shuffled=')
                   and not line.startswith("(DATA/'capture_manifest.json').write_text"))+'\n'
    code=code.replace('/kaggle/working/h3_gate_data','/kaggle/input/h3-real-attention-diagnostic-heads')
    code=code.replace("DATA=pathlib.Path('/kaggle/input/h3-real-attention-diagnostic-heads');DATA.mkdir(exist_ok=True)",
        "mounts=list(pathlib.Path('/kaggle/input').rglob('capture_manifest.json'));print('DATA_MOUNTS',mounts,flush=True);assert len(mounts)==1,mounts;DATA=mounts[0].parent")
if sparse_failure:
    metadata=json.loads((gate_dataset/'dataset-metadata.json').read_text())
    metadata.update(title='H3 Sparse Failure Diagnostic Head',id='chinokiki9/h3-sparse-failure-diagnostic-head')
    (gate_dataset/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
    code=code.replace("['te_layer10_call-1.pt','dit_layer49_call09.pt','video_layer00_call00.pt']", "['dit_layer25_call09.pt']")
    code=code.replace("'--file','dit_layer49_call09.pt'", "'--file','dit_layer25_call09.pt'")
    code=code.replace('three real diagnostic heads', 'one real DiT pooling-failure head')
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
notebook_path=FOLDER/'experiment.ipynb'
notebook_path.write_text(json.dumps(notebook),encoding='utf-8')
(FOLDER/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-real-t4-replay-gate'+suffix.replace('_','-'),
    title='H3 Real T4 Replay Gate'+suffix.replace('_',' ').title(),code_file='experiment.ipynb',language='python',kernel_type='notebook',
    is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
    dataset_sources=(['chinokiki9/h3-sparse-failure-diagnostic-head'] if sparse_failure else
                     (['chinokiki9/h3-real-attention-diagnostic-heads'] if dataset_mode else [])),
    competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
print('Notebook bytes',notebook_path.stat().st_size)
