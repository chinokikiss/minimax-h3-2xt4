"""Build independent private cu130/SM75 operator smoke Kaggle job.

No dataset, H3 weights, PTD bank, LoRA, existing kernel overwrite or submission.
"""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from benchmarks.build_ptd_v2_bake_job import DEFAULT_BASE,_baseline_commit
from benchmarks.source_overlay_closure import overlay_files,assert_closed
from benchmarks.t4_notebook_watchdog import supervise_worker

ROOT=Path(__file__).resolve().parents[1]
ID='pamcks/h3-cuda130-t4-sm75-all-operator-smoke-20261104'
SEEDS=(
    'benchmarks/run_h3_cuda130_t4_operator_smoke.py',
    'patches/h3_operator_dependencies.py',
    'patches/h3_operator_policy.py',
    'patches/h3_operator_aot.py',
    'patches/kitchen_attention.py',
    'patches/kitchen_te_attention.py',
    'patches/sol_attention.py',
    'experiments/kitchen_attention/build.py',
    'experiments/kitchen_attention/derive.py',
    'experiments/kitchen_attention/attention.py',
    'patches/tilelang_column_epilogue.py',
    'patches/tilelang_row_quant.py',
    'patches/tilelang_swiglu_quant.py',
    'patches/tilelang_transport_tradeoffs.py',
    'patches/tilelang_transport_p2.py',
    'patches/tilelang_fp32_rms_rope.py',
    'patches/tilelang_dit_fusion.py',
)


def build(output,base=DEFAULT_BASE):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Do not overwrite any Kaggle operator test run')
    base=Path(base).resolve()
    commit=_baseline_commit(base)
    names,closure=overlay_files(commit,SEEDS)
    assert_closed(commit,names,SEEDS)
    members={name:(ROOT/name).read_bytes() for name in names}
    hashes={name:hashlib.sha256(value).hexdigest() for name,value in members.items()}
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for name,data in sorted(members.items()):
            item=tarfile.TarInfo(name);item.size=len(data);item.mtime=0
            archive.addfile(item,io.BytesIO(data))
    blob=stream.getvalue();sha=hashlib.sha256(blob).hexdigest()
    notebook=json.loads(base.read_text(encoding='utf-8'))
    original=''.join(notebook['cells'][0]['source'])
    marker="assert receipt['status']=='ready_for_device_gate'\n"
    if original.count(marker)!=1:raise RuntimeError('Frozen Kaggle setup boundary changed')
    setup=original[:original.index(marker)+len(marker)]
    inject=f"""
import base64,hashlib,io,json,pathlib,tarfile,os,subprocess,sys
SOURCE=pathlib.Path('/kaggle/temp/h3_light_source')
HASHES={hashes!r}
DATA=base64.b64decode({base64.b64encode(blob).decode('ascii')!r})
if hashlib.sha256(DATA).hexdigest()!={sha!r}:raise ValueError('cu130 source capsule hash mismatch')
with tarfile.open(fileobj=io.BytesIO(DATA),mode='r:gz') as archive:
    if set(archive.getnames())!=set(HASHES):raise ValueError('cu130 source capsule manifest mismatch')
    for member in archive:
        if not member.isfile() or member.name.startswith('/') or '..' in pathlib.PurePosixPath(member.name).parts:
            raise ValueError('Unsafe source capsule entry')
        data=archive.extractfile(member).read()
        if hashlib.sha256(data).hexdigest()!=HASHES[member.name]:raise ValueError('source mismatch: '+member.name)
        dest=SOURCE/member.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
print('CU130_OPERATOR_SOURCE_READY',len(HASHES),flush=True)
"""
    worker="""import json,os,pathlib,sys
env=dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
env['H3_ALL_OPS_CUDA130_T4_QUALIFY']='1'
env['PYTHONUNBUFFERED']='1'
env['OMP_NUM_THREADS']='2'
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_h3_cuda130_t4_operator_smoke.py'),
      '--source',str(source),'--device','0',
      '--output','/kaggle/working/h3_cu130_sm75_all_operators']
os.execve(sys.executable,argv,env)
"""
    code=setup+inject+supervise_worker(worker,budget=3600,
                output='/kaggle/working/h3_cu130_sm75_supervision')
    ast.parse(code)
    if re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',code):
        raise ValueError('Token-like string in cu130 operator job')
    notebook['cells'][0]['source']=code.splitlines(keepends=True)
    metadata=json.loads((base.parent/'kernel-metadata.json').read_text())
    metadata.update(id=ID,title='H3 CUDA130 T4 SM75 All Operator Smoke 20261104',
        is_private=True,enable_gpu=True,enable_internet=True,
        dataset_sources=[],model_sources=[],competition_sources=[],kernel_sources=[],
        machine_shape='NvidiaTeslaT4')
    output.mkdir(parents=True)
    (output/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    evidence=dict(format='h3-cu130-sm75-all-operator-smoke-v1',status='prepared_unsubmitted',
        kaggle_id=ID,immutable_baseline_commit=commit,
        independently_submitted=True,physical_gpu='Tesla T4 SM75',physical_device_index=0,
        source_sha256=sha,source_files=hashes,dependency_closure=closure,
        notebook_sha256=hashlib.sha256((output/'experiment.ipynb').read_bytes()).hexdigest(),
        dataset_sources=[],no_model=True,no_full_generation=True,supervisor_seconds=3600,
        cold_source_compilation=True,runtime_execution_required=True,
        not_a_full_production_qualification=True)
    (output/'source_binding.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(dict(id=ID,sources=len(names),closure=len(closure),
        notebook_bytes=(output/'experiment.ipynb').stat().st_size,
        source_sha256=sha,output=str(output)),indent=2))
    return evidence


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path,default=DEFAULT_BASE)
    args=p.parse_args();build(args.output,args.base)


if __name__=='__main__':main()
