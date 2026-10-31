"""Build one source-bound physical-T4 Sol execution preflight from the completed baseline notebook.

The completed baseline supplies only the frozen environment bootstrap.  Current
runtime/preflight sources are dependency-closed against that baseline and hash
bound inside the notebook.  This job performs no full H3 generation.
"""
import argparse
import ast
import base64
import datetime as dt
import hashlib
import io
import json
import lzma
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.t4_notebook_watchdog import supervise_worker
from benchmarks.source_overlay_closure import overlay_files as closed_overlay_files,assert_closed

CORE_OVERLAY=[
    'benchmarks/sol_execution_preflight.py',
    'benchmarks/run_sol_execution_preflight_batch.py',
    'benchmarks/dit_pair_contract.py',
    'benchmarks/e2e_attention_validation.py',
    'benchmarks/ref2va_e2e_run.py',
    'patches/attention_schedule.py',
    'patches/sol_attention.py',
    'patches/sol_sm75_backend.py',
    'patches/h3_operator_policy.py',
    'patches/h3_operator_aot.py',
    'patches/h3_operator_dependencies.py',
    'patches/comfy_tp_runtime.py',
    'patches/comfy_sampler.py',
    'patches/parallel_vae.py',
]
RETIRED=[
    'patches/tilelang_group32_masked.py','patches/tilelang_te_preparation.py',
    'patches/tilelang_attention_p2.py','patches/tilelang_attention_tradeoffs.py',
    'patches/tilelang_dense_preparation.py','patches/tilelang_grouped_fused.py',
    'patches/tilelang_grouped_qk.py','patches/tilelang_grouped_tp.py',
    'patches/tilelang_grouped_tp_kitchen.py','patches/tilelang_grouped_tp_kitchen_v2.py',
    'patches/tilelang_grouped_tp_optimized.py','patches/tilelang_sage_preparation.py',
    'patches/tilelang_sage_qk.py','patches/tilelang_scalar_pool.py',
    'patches/tilelang_sol_drop_fused.py','patches/tilelang_sparse_attention.py',
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline-job',type=Path,required=True)
    p.add_argument('--baseline-request',type=Path,required=True)
    p.add_argument('--baseline-result',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()

    baseline_job=args.baseline_job.resolve()
    baseline_binding=json.loads((baseline_job/'source_binding.json').read_text(encoding='utf-8'))
    baseline_commit=baseline_binding['publication']['commit']
    overlay,closure=closed_overlay_files(baseline_commit,CORE_OVERLAY)
    assert_closed(baseline_commit,overlay,CORE_OVERLAY)

    baseline_plan=json.loads(args.baseline_request.read_text(encoding='utf-8'))
    baseline_result=json.loads(args.baseline_result.read_text(encoding='utf-8'))
    assert baseline_result['status']=='complete' and len(baseline_result['sampler_steps'])==8
    assert (baseline_plan['width'],baseline_plan['height'],baseline_plan['frames'],
            baseline_plan['steps'],baseline_plan['seed'])==(1216,704,243,8,20261020)
    turbo=baseline_plan['turbo_lora']
    for key in ('repository','revision','file','bytes','sha256','targets'):
        assert key in turbo
    plan=dict(name='sol_execution_preflight_20261030',turbo_lora=turbo,
        baseline_request_sha256=hashlib.sha256(args.baseline_request.read_bytes()).hexdigest(),
        baseline_result_sha256=hashlib.sha256(args.baseline_result.read_bytes()).hexdigest(),
        execution_contract=dict(
            dit_tokens=59134,dit_calls=2,dit_schedule='1:kitchen,2:sol@1.0',
            video_attention='kitchen',video_latent=[1,24,12,16,16],
            quality_claim=False,performance_claim=False))

    hashes={}
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:') as archive:
        for name in overlay:
            raw=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
            info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0;info.uid=0;info.gid=0;info.uname='';info.gname=''
            archive.addfile(info,io.BytesIO(raw))
    archive_raw=lzma.compress(buf.getvalue(),preset=9|lzma.PRESET_EXTREME)
    archive_b85=base64.b85encode(archive_raw).decode()
    archive_sha256=hashlib.sha256(archive_raw).hexdigest()

    args.evidence.mkdir(parents=True,exist_ok=True)
    job=args.evidence/'gpu_job'
    if job.exists():raise FileExistsError(job)
    job.mkdir()

    nb=json.loads((baseline_job/'experiment.ipynb').read_text(encoding='utf-8'))
    assert len(nb['cells'])==1
    old=''.join(nb['cells'][0]['source'])
    marker="print('T4_SUPERVISOR notebook_cell_started',flush=True)"
    assert old.count(marker)>=2
    prefix=old[:old.rfind(marker)]

    execution=r'''import base64,hashlib,io,json,lzma,os,pathlib,sys,tarfile
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
out=pathlib.Path('/kaggle/working/sol_execution_preflight');out.mkdir()
manifest=json.loads(pathlib.Path('/kaggle/temp/h3_light_distribution/delivery_manifest.json').read_text())
for name,expected in manifest['source_hashes'].items():
    path=source/name
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Frozen baseline source changed: '+name
hashes=HASHES
archive_raw=base64.b85decode(OVERLAY_ARCHIVE)
assert hashlib.sha256(archive_raw).hexdigest()==OVERLAY_ARCHIVE_SHA256
with tarfile.open(fileobj=io.BytesIO(lzma.decompress(archive_raw)),mode='r:') as archive:
    for member in archive.getmembers():
        rel=pathlib.PurePosixPath(member.name)
        assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
        target=source.joinpath(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
        payload=archive.extractfile(member);assert payload is not None
        target.write_bytes(payload.read())
for name,expected in hashes.items():
    target=source/name
    assert target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==expected,'Overlay hash mismatch: '+name
for name in RETIRED:
    target=source/name
    if target.exists():target.unlink()
(out/'source_overlay.json').write_text(json.dumps(dict(
    baseline_capsule_sha256=manifest['capsule_sha256'],overlay_hashes=hashes,retired=RETIRED,
    scope='Dependency-closed current runtime for physical T4 Sol execution preflight only'),indent=2))
plan=PLAN
(out/'plan.json').write_text(json.dumps(plan,indent=2))
argv=[sys.executable,'-u',str(source/'benchmarks/run_sol_execution_preflight_batch.py'),
    '--source',str(source),'--output',str(out),'--plan',str(out/'plan.json'),
    '--scratch','/kaggle/temp/h3_sol_execution_preflight_20261030']
os.execve(sys.executable,argv,env)
'''.replace('OVERLAY_ARCHIVE_SHA256',repr(archive_sha256)).replace(
        'OVERLAY_ARCHIVE',repr(archive_b85)).replace('HASHES',repr(hashes)).replace(
        'RETIRED',repr(RETIRED)).replace('PLAN',repr(plan))

    code=prefix+supervise_worker(execution,budget=6000,output='/kaggle/working/sol_execution_preflight_supervision')
    code+="assert json.loads(pathlib.Path('/kaggle/working/sol_execution_preflight/batch_results.json').read_text())['status']=='complete'\n"
    ast.parse(code)
    assert not re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_\-]{20,})',code)
    nb['cells'][0]['source']=code.splitlines(keepends=True)
    (job/'experiment.ipynb').write_text(json.dumps(nb),encoding='utf-8')
    if (job/'experiment.ipynb').stat().st_size>=900_000:
        raise ValueError('Notebook overlay exceeds the proven SaveKernel size envelope')

    meta=json.loads((baseline_job/'kernel-metadata.json').read_text(encoding='utf-8'))
    meta.update(id='pamcks/h3-sol-execution-preflight-t4-20261030',
        title='H3 Sol Execution Preflight T4 20261030')
    assert meta['is_private'] and meta['dataset_sources']==meta['kernel_sources']==[]
    (job/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

    binding=dict(
        baseline_source_binding=baseline_binding,
        baseline_request_sha256=plan['baseline_request_sha256'],
        baseline_result_sha256=plan['baseline_result_sha256'],
        overlay_hashes=hashes,overlay_dependency_closure=list(closure),
        overlay_baseline_commit=baseline_commit,overlay_archive_sha256=archive_sha256,
        overlay_archive_bytes=len(archive_raw),retired_paths=RETIRED,
        execution_contract=plan['execution_contract'],turbo_lora=turbo,
        scope='Physical T4 execution-semantic preflight; no full generation, quality or performance claim',
        full_pipeline_authorized=False,gpu_count=2,zero_mounts=True,
        status_polling=False,live_log_polling=False)
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    (args.evidence/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    receipt=dict(status='ready',job=str(job),notebook_bytes=(job/'experiment.ipynb').stat().st_size,
        notebook_sha256=hashlib.sha256((job/'experiment.ipynb').read_bytes()).hexdigest(),
        source_binding=binding)
    (args.evidence/'build_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))
    if not args.submit:return

    submission=args.evidence/'gpu_submission.json'
    state=dict(kernel=meta['id'],status='submitting',
        submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_binding=binding,
        run_timeout_seconds=6300,final_collection_count=0,
        result_relative='sol_execution_preflight/execution/result.json',
        notebook_sha256=receipt['notebook_sha256'])
    with submission.open('x',encoding='utf-8') as f:json.dump(state,f,indent=2)
    proc=subprocess.run([sys.executable,str(ROOT/'benchmarks/kaggle_tls12_runner.py'),
        '--cli','kernels','push','-p',str(job)],capture_output=True,text=True,timeout=180)
    output=proc.stdout+proc.stderr
    state.update(status='accepted' if proc.returncode==0 else 'acceptance_uncertain',
        returncode=proc.returncode,response=output)
    m=re.search(r'Kernel version (\d+)',output)
    if m:state['version']=int(m.group(1))
    u=re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',output)
    if u:state['url']=u.group(0)
    submission.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    (args.evidence/'gpu_submission_cli.log').write_text(output,encoding='utf-8')
    print(output[-2000:])
    if proc.returncode:raise RuntimeError('Preflight submission was not accepted; no automatic retry')


if __name__=='__main__':main()
