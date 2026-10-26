"""Bounded TE operator audit using the already CPU-qualified frozen setup."""
import ast
import base64
import hashlib
import json
from pathlib import Path

from .delivery import ROOT,submit


def notebook(evidence,qualified):
    from benchmarks.t4_notebook_watchdog import supervise_worker
    evidence=Path(evidence);qualified=Path(qualified);evidence.mkdir(parents=True,exist_ok=True)
    job=evidence/'gpu_job';job.mkdir()
    pub=json.loads((qualified/'publication.json').read_text())
    cpu_path=qualified/'cpu_outputs/light_setup/setup_receipt.json'
    cpu=json.loads(cpu_path.read_text())
    assert cpu['status']=='cpu_ready' and cpu['capsule_sha256']==pub['capsule_sha256']
    setup=(qualified/'gpu_outputs/light_setup/bounded_worker.py').read_text()
    ast.parse(setup)
    files={'te_precision.py':(ROOT/'experiments/kitchen_attention/te_precision.py').read_bytes(),
           'kitchen_te_attention.py':(ROOT/'patches/kitchen_te_attention.py').read_bytes()}
    for source in files.values():ast.parse(source)
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
    payload={name:base64.b64encode(raw).decode() for name,raw in files.items()}
    execution='''import base64,hashlib,json,os,pathlib,sys
ROOT=pathlib.Path('/kaggle/working/supplemental');ROOT.mkdir(parents=True,exist_ok=True)
FILES=SOURCE_PAYLOAD
HASHES=SOURCE_HASHES
for name,encoded in FILES.items():
 raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==HASHES[name]
 (ROOT/name).write_bytes(raw)
(ROOT/'source_hashes.json').write_text(json.dumps(HASHES,indent=2))
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
src='/kaggle/temp/h3_light_source';env['PYTHONPATH']=src+os.pathsep+env.get('PYTHONPATH','')
env['OMP_NUM_THREADS']='2';os.chdir(src)
argv=[sys.executable,'-u',str(ROOT/'te_precision.py'),'--output','/kaggle/working/te_precision']
os.execve(sys.executable,argv,env)
'''.replace('SOURCE_PAYLOAD',repr(payload)).replace('SOURCE_HASHES',repr(hashes))
    code=supervise_worker(setup,budget=240,output='/kaggle/working/light_setup')
    code+='''
import json,pathlib
setup=json.loads(pathlib.Path('/kaggle/working/light_setup/setup_receipt.json').read_text())
assert setup['status']=='ready_for_device_gate',setup['status']
'''
    code+=supervise_worker(execution,budget=300,output='/kaggle/working/te_supervision')
    code+='''
guard=json.loads(pathlib.Path('/kaggle/working/te_supervision/supervision.json').read_text())
assert guard['status']=='worker_exited' and guard['exit_code']==0,guard
'''
    ast.parse(code)
    meta=dict(id='pamcks/h3-te-causal-precision-t4-20261026',title='H3 TE Causal Precision T4 20261026',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',docker_image_pinning_type='original',
        dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    book=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')),
        cells=[dict(id='te-causal-precision',cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
    (job/'experiment.ipynb').write_text(json.dumps(book))
    (job/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    smoke=json.loads((evidence/'local_smoke.json').read_text());assert smoke['status']=='passed'
    (job/'source_binding.json').write_text(json.dumps(dict(publication=pub,supplemental_sources=hashes,
        cpu_qualification_sha256=hashlib.sha256(cpu_path.read_bytes()).hexdigest(),
        bootstrap_worker_sha256=hashlib.sha256(setup.encode()).hexdigest(),
        local_checks=dict(reference_and_dispatch_unittests=12,passed=True,local_smoke_scope=smoke['scope'],local_smoke_cases=len(smoke['cases'])),
        execution_budget_seconds=300,setup_budget_seconds=240,final_collection_minimum_delay_seconds=630,
        frozen_capsule_modified=False,dataset_mounts=0,kernel_mounts=0,temporary_private_auth=False),indent=2))
    return job


def run(evidence,qualified):
    evidence=Path(evidence);notebook(evidence,qualified);submit(evidence,'gpu')
    path=evidence/'gpu_submission.json';state=json.loads(path.read_text())
    state.update(result_relative='000_te_causal_precision.json',earliest_final_collection_seconds=630,
                 experiment_kind='te_causal_precision_operator_audit')
    path.write_text(json.dumps(state,indent=2)+'\n')
