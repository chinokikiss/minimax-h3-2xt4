"""Reuse the passed frozen core setup; bind the new, CPU-tested TE experiment."""
import argparse
import ast
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from t4_notebook_watchdog import supervise_worker

ROOT=Path(__file__).resolve().parents[1]
NAMES=('te_parallel_matrix.py','run_te_parallel_batch.py','test_te_parallel_matrix.py')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qualified',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--resume-from',type=Path,help='Retain fully validated points from a collected previous attempt')
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args();args.evidence.mkdir(parents=True,exist_ok=True)
    job=args.evidence/'gpu_job'
    if job.exists():raise FileExistsError('Preserve previously built/submitted jobs')
    pub=json.loads((args.qualified/'source_publication_v2.json').read_text())
    cpu_path=args.qualified/'cpu_outputs/light_setup/setup_receipt.json'
    cpu=json.loads(cpu_path.read_text())
    assert cpu['status']=='cpu_ready' and cpu['capsule_sha256']==pub['capsule_sha256']
    assert cpu['review_suite']['passed'] and cpu['core_pins']['comfy-kitchen']=='0.2.37' and cpu['core_pins']['tilelang']=='0.1.13'
    checks_dir=args.resume_from or args.evidence
    defaults_path=checks_dir/'defaults_cpu_final.json'
    if not defaults_path.exists():defaults_path=checks_dir/'defaults_cpu.json'
    defaults=json.loads(defaults_path.read_text())
    smoke=json.loads((checks_dir/'local_single_smoke.json').read_text())
    assert defaults['passed'] and smoke['status']=='passed'
    assert all(item['public_shard_byte_exact'] for item in json.loads((checks_dir/'local_asset_identity.json').read_text()))
    lengths=[128,353,1306,4096];resume=None
    if args.resume_from:
        from te_parallel_matrix import completed_lengths
        rank_paths=[args.resume_from/f'gpu_outputs/te_matrix/rank{rank}.json' for rank in (0,1)]
        skipped=completed_lengths([json.loads(path.read_text()) for path in rank_paths])
        assert skipped,'No complete previous point to resume'
        lengths=[n for n in lengths if n not in skipped];assert lengths,'No unfinished points'
        resume=dict(kernel=json.loads((args.resume_from/'gpu_submission.json').read_text())['kernel'],
            skipped_lengths=skipped,previous_rank_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in rank_paths],
            note='Retained points belong to the previous allocation; comparisons remain paired within each allocation')
        assert json.loads((args.evidence/'local_padding_smoke.json').read_text())['status']=='passed'
    files={name:(ROOT/'benchmarks'/name).read_bytes() for name in NAMES}
    for value in files.values():ast.parse(value)
    credential=re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_\-]{20,})')
    assert not any(credential.search(value) for value in files.values())
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
    manifest=json.loads((args.qualified/'distribution_v2/delivery_manifest.json').read_text())
    runtime_names=('tp_qwen3vl','h3_operator_policy','h3_tensor_flow','tilelang_group32_masked',
        'kitchen_te_attention','optimized_tp','tilelang_transport_p2','tilelang_swiglu_quant','dit_column_linear','parallel_int8')
    runtime_hashes={f'patches/{name}.py':hashlib.sha256((ROOT/'patches'/f'{name}.py').read_bytes()).hexdigest() for name in runtime_names}
    assert all(manifest['source_hashes'][name]==digest for name,digest in runtime_hashes.items()),'TE runtime differs from the CPU-qualified frozen source'
    # Run against these exact source bytes; do not install/change any packages.
    with (args.evidence/'harness_cpu.log').open('w') as log:
        check=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'benchmarks'),
            '-p','test_te_parallel_matrix.py','-v'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    assert check.returncode==0,'New TE harness CPU checks failed'
    count=int(re.search(r'Ran (\d+) tests?',(args.evidence/'harness_cpu.log').read_text()).group(1))
    local=dict(passed=True,tests=count,source_hashes=hashes,defaults_cpu=defaults,single_gpu_smoke=smoke)
    (args.evidence/'harness_cpu.json').write_text(json.dumps(local,indent=2))
    setup=(args.qualified/'gpu_outputs/light_setup/bounded_worker.py').read_text()
    assert setup.count("'--mode','vae-parallel'")==1
    setup=setup.replace("'--mode','vae-parallel'","'--mode','operators'")
    ast.parse(setup)
    code="import json,pathlib\nOUT=pathlib.Path('/kaggle/working/light_setup');OUT.mkdir(exist_ok=True)\n"
    code+="(OUT/'cpu_qualification.json').write_text("+repr(json.dumps(cpu))+')\n'
    code+=supervise_worker(setup,budget=300,output='/kaggle/working/light_setup')
    code+="\nassert json.loads((OUT/'setup_receipt.json').read_text())['status']=='ready_for_device_gate'\n"
    execution='''import base64,hashlib,json,os,pathlib,sys
SUPP=pathlib.Path('/kaggle/working/te_sources');SUPP.mkdir(exist_ok=True)
FILES=SOURCE_FILES
HASHES=SOURCE_HASHES
for name,encoded in FILES.items():
 raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==HASHES[name]
 (SUPP/name).write_bytes(raw)
(SUPP/'source_hashes.json').write_text(json.dumps(HASHES,indent=2))
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
src='/kaggle/temp/h3_light_source';env['PYTHONPATH']=src+os.pathsep+env.get('PYTHONPATH','');os.chdir(src)
argv=[sys.executable,'-u',str(SUPP/'run_te_parallel_batch.py'),'--source',src,'--output','/kaggle/working/te_matrix','--lengths',*LENGTH_ARGUMENTS]
os.execve(sys.executable,argv,env)
'''.replace('SOURCE_FILES',repr({name:base64.b64encode(raw).decode() for name,raw in files.items()})).replace('SOURCE_HASHES',repr(hashes)).replace('LENGTH_ARGUMENTS',repr([str(n) for n in lengths]))
    code+=supervise_worker(execution,budget=900,output='/kaggle/working/te_supervision')
    code+="\nguard=json.loads(pathlib.Path('/kaggle/working/te_supervision/supervision.json').read_text())\nassert guard['status']=='worker_exited' and guard['exit_code']==0\n"
    ast.parse(code);job.mkdir()
    metadata=dict(id='pamcks/h3-te-attention-tp2-matrix-20261027',title='H3 TE Attention TP2 Matrix 20261027',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',docker_image_pinning_type='original',
        dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')),
        cells=[dict(id='te-parallel-matrix',cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
    (job/'experiment.ipynb').write_text(json.dumps(notebook))
    (job/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    binding=dict(publication=pub,supplemental_sources=hashes,cpu_qualification_sha256=hashlib.sha256(cpu_path.read_bytes()).hexdigest(),
        local_harness_qualification=local,bootstrap_sha256=hashlib.sha256(setup.encode()).hexdigest(),
        frozen_capsule_modified=False,te_runtime_unchanged=True,te_runtime_hashes=runtime_hashes,private_auth=False,
        dataset_mounts=0,kernel_mounts=0,setup_budget_seconds=300,execution_budget_seconds=900,executed_lengths=lengths,resume=resume)
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2))
    print('Built:',job,flush=True)
    if not args.submit:return
    receipt=args.evidence/'gpu_submission.json'
    state=dict(kernel=metadata['id'],submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='submitting',
        returncode=None,source_binding=binding,final_collection_count=0,earliest_final_collection_seconds=1500,
        experiment_kind='te_parallel_matrix',result_relative='te_matrix/result.json',run_timeout_seconds=1500,
        notebook_sha256=hashlib.sha256((job/'experiment.ipynb').read_bytes()).hexdigest())
    with receipt.open('x') as stream:json.dump(state,stream,indent=2)
    try:
        result=subprocess.run([sys.executable,'-m','kaggle','kernels','push','-p',str(job),
            '--accelerator','NvidiaTeslaT4','-t','1500'],capture_output=True,text=True,timeout=180)
        output=result.stdout+result.stderr
        state.update(returncode=result.returncode,status='accepted' if result.returncode==0 else 'submission_failed')
        version=re.search(r'Kernel version (\d+)',output)
        if version:state['version']=int(version.group(1))
    except subprocess.TimeoutExpired:
        output='Submission acknowledgement timed out; no automatic retry.'
        state.update(returncode=-1,status='acceptance_uncertain')
    (args.evidence/'gpu_submission_cli.log').write_text(output,encoding='utf-8')
    receipt.write_text(json.dumps(state,indent=2));print(output[-1500:],flush=True)


if __name__=='__main__':main()
