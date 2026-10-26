"""Reuse the CPU-qualified frozen bootstrap with explicit supplemental sources."""
import ast
import base64
import hashlib
import json
from pathlib import Path

from .delivery import ROOT


def notebook(evidence,qualified):
    from benchmarks.t4_notebook_watchdog import supervise_worker
    evidence=Path(evidence);qualified=Path(qualified);evidence.mkdir(parents=True,exist_ok=True)
    job=evidence/'gpu_job';job.mkdir()
    pub=json.loads((qualified/'publication.json').read_text())
    cpu=json.loads((qualified/'cpu_outputs/light_setup/setup_receipt.json').read_text())
    assert cpu['status']=='cpu_ready' and cpu['capsule_sha256']==pub['capsule_sha256']
    setup=(qualified/'gpu_outputs/light_setup/bounded_worker.py').read_text()
    ast.parse(setup)
    files={name:(ROOT/path).read_bytes() for name,path in {
        'graph.py':'experiments/kitchen_attention/graph.py',
        'graph_bench.py':'experiments/kitchen_attention/graph_bench.py',
        'kitchen_te_attention.py':'patches/kitchen_te_attention.py'}.items()}
    hashes={n:hashlib.sha256(raw).hexdigest() for n,raw in files.items()}
    for raw in files.values():ast.parse(raw)
    payload={n:base64.b64encode(raw).decode() for n,raw in files.items()}
    execution='''import base64,hashlib,json,os,pathlib,sys
ROOT=pathlib.Path('/kaggle/working/supplemental');ROOT.mkdir(parents=True,exist_ok=True)
FILES=PAYLOAD
HASHES=SOURCE_HASHES
for name,encoded in FILES.items():
 raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==HASHES[name]
 (ROOT/name).write_bytes(raw)
(ROOT/'source_hashes.json').write_text(json.dumps(HASHES,indent=2))
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
src='/kaggle/temp/h3_light_source'
env['PYTHONPATH']=src+os.pathsep+env.get('PYTHONPATH','')
env['OMP_NUM_THREADS']='2';os.chdir(src)
argv=[sys.executable,'-u',str(ROOT/'graph_bench.py'),'--output','/kaggle/working/graph_matrix']
os.execve(sys.executable,argv,env)
'''.replace('PAYLOAD',repr(payload)).replace('SOURCE_HASHES',repr(hashes))
    code=supervise_worker(setup,budget=240,output='/kaggle/working/light_setup')
    code+='''
import json,pathlib
setup=json.loads(pathlib.Path('/kaggle/working/light_setup/setup_receipt.json').read_text())
assert setup['status']=='ready_for_device_gate',setup['status']
'''
    code+=supervise_worker(execution,budget=360,output='/kaggle/working/graph_supervision')
    code+='''
guard=json.loads(pathlib.Path('/kaggle/working/graph_supervision/supervision.json').read_text())
assert guard['status']=='worker_exited' and guard['exit_code']==0,guard
'''
    ast.parse(code)
    meta=dict(id='pamcks/h3-kitchen-dense-graph-t4-20261025',title='H3 Kitchen Dense Graph T4 20261025',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',docker_image_pinning_type='original',
        dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    book=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')),
        cells=[dict(id='kitchen-graph',cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
    (job/'experiment.ipynb').write_text(json.dumps(book))
    (job/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (job/'source_binding.json').write_text(json.dumps(dict(publication=pub,supplemental_sources=hashes,
        cpu_qualification_sha256=hashlib.sha256((qualified/'cpu_outputs/light_setup/setup_receipt.json').read_bytes()).hexdigest(),
        bootstrap_worker_sha256=hashlib.sha256(setup.encode()).hexdigest(),
        execution_budget_seconds=360,setup_budget_seconds=240,final_collection_minimum_delay_seconds=690,
        frozen_capsule_modified=False,dataset_mounts=0,kernel_mounts=0,temporary_private_auth=False),indent=2))
    return job
