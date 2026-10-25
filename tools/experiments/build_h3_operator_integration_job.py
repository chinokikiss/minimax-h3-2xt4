"""Bounded two-T4 integration gates; no model, generation, dataset or timing sweep."""
import ast
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--runtime-dataset',help='Private dataset containing the frozen user conda/source bundle and Linux wheelhouse')
parser.add_argument('--validate-only',action='store_true',help='Parse generated code without creating or submitting a notebook')
args=parser.parse_args()
if not args.runtime_dataset and not args.validate_only:parser.error('--runtime-dataset is required for a runnable notebook')
OUT=Path('D:/attention_eval/h3_conda_operator_submission_20261019')
names=[str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'patches').glob('*.py')]
names+=['custom_nodes/minimax_h3_tp2/__init__.py','benchmarks/test_h3_operator_integration.py','benchmarks/frozen_kaggle_setup.py','runtime/local_conda.json']
sources={name:(ROOT/name).read_text(encoding='utf8') for name in names}
for name,source in sources.items():
    if name.endswith('.py'):ast.parse(source,filename=name)
lock=json.loads(sources['runtime/local_conda.json'])
lock_digest=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(',',':')).encode()).hexdigest()
hashes={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
code='''import pathlib,subprocess,sys,os,time,json,traceback,signal,shutil
SRC=pathlib.Path('/kaggle/working/source');SRC.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/integration');OUT.mkdir(exist_ok=True)
deadline=time.monotonic()+540
state={'status':'running','scope':'Operator integration gates on two physical T4s. No H3 checkpoint, full generation or performance benchmark.','commands':[]}
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
def run(argv,name,cap):
    start=time.monotonic();timeout=min(cap,deadline-start);assert timeout>10
    with (OUT/(name+'.log')).open('w') as log:
        process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();rc=-9
    state['commands'].append(dict(name=name,exit_code=rc,seconds=time.monotonic()-start));save()
    print('INTEGRATION_STAGE',name,rc,flush=True)
    if rc:print((OUT/(name+'.log')).read_text(errors='replace')[-8000:],flush=True)
    return rc==0
'''
for name,source in sources.items():code+=f'p=SRC/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({source!r})\n'
code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
code+=f"EXPECTED_RUNTIME_LOCK={lock_digest!r}\n"
code+='''save()
try:
    nvcc=pathlib.Path(shutil.which('nvcc')).resolve()
    state['compiler']=subprocess.check_output([str(nvcc),'--version'],text=True)
    assert 'release 12.8' in state['compiler']
    os.environ.update(CUDACXX=str(nvcc),CUDA_HOME=str(nvcc.parent.parent),CUDA_PATH=str(nvcc.parent.parent),
        OMP_NUM_THREADS='2',NCCL_P2P_DISABLE='1',NCCL_DEBUG='WARN',NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4',PYTHONUNBUFFERED='1')
    os.environ.pop('NCCL_PROTO',None)
    assert run([sys.executable,str(SRC/'benchmarks/frozen_kaggle_setup.py'),'--root',str(SRC),'--offline-install',
        '--expected-manifest-sha256',EXPECTED_RUNTIME_LOCK],'frozen_user_runtime',90)
    os.environ['COMFY_ROOT']=str(SRC/'ComfyUI')
    state['runtime_lock']=json.loads((SRC/'runtime/setup_receipt.json').read_text());save()
    import torch
    state['hardware']=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())]
    state['torch']=torch.__version__;save()
    assert len(state['hardware'])==2 and all(x['capability']==[7,5] and 'T4' in x['name'] for x in state['hardware'])
    ok=run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(SRC/'benchmarks/test_h3_operator_integration.py'),'--t4','--distributed','--output',str(OUT/'gates.json')],'gates',300)
    state['status']='complete' if ok else 'failed'
except Exception as error:
    state['status']='blocked';state['error']=type(error).__name__+': '+str(error)
    (OUT/'exception.txt').write_text(traceback.format_exc())
finally:save();print('H3_INTEGRATION_FINAL',state['status'],flush=True)
'''
ast.parse(code)
if args.validate_only:
    print('FROZEN_OPERATOR_BUILDER_VALID',len(sources),'source files',lock_digest)
    raise SystemExit(0)
folder=ROOT/'kaggle_h3-tp2-operator-integration-frozen';folder.mkdir(exist_ok=True)
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    cells=[dict(id='integration',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf8')
(folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-tp2-operator-integration-frozen',title='H3 TP2 Operator Integration Frozen Runtime',
    code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,
    enable_internet=True,dataset_sources=[args.runtime_dataset],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
OUT.mkdir(exist_ok=True)
for name,source in sources.items():
    path=OUT/'submitted_source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source,encoding='utf8')
(OUT/'submitted_sources.json').write_text(json.dumps(hashes,indent=2));print(folder)
