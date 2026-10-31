"""Build one production Sol-v2 0.85 MP / 10 s long-sequence E2E from the completed baseline notebook.

The previous successful notebook supplies the frozen environment bootstrap and
reference delivery. This builder replaces only the execution worker, overlays
the current source-locked experimental Sol-v2 path, and preserves the exact workload
except for the explicitly requested attention policy.
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
from benchmarks.turbo_challenge_contract import validate
from benchmarks.source_overlay_closure import overlay_files as closed_overlay_files,assert_closed

CORE_OVERLAY=[
    'patches/attention_schedule.py',
    'patches/sol_attention.py',
    'patches/sol_sm75_backend.py',
    'patches/h3_operator_policy.py',
    'patches/h3_operator_aot.py',
    'patches/h3_operator_dependencies.py',
    'patches/comfy_tp_runtime.py',
    'patches/runtime_diagnostics.py',
    'patches/parallel_vae.py',
    'benchmarks/e2e_attention_validation.py',
    'benchmarks/ref2va_e2e_run.py',
    'benchmarks/run_h3_l1_turbo_e2e.py',
    'benchmarks/run_h3_l1_turbo_batch.py',
    'benchmarks/turbo_challenge_contract.py',
    'custom_nodes/minimax_h3_tp2/__init__.py',
    'custom_nodes/minimax_h3_tp2/web/loader_migration.js',
]
RETIRED=[
    'patches/tilelang_group32_masked.py',
    'patches/tilelang_te_preparation.py',
    'patches/tilelang_attention_p2.py',
    'patches/tilelang_attention_tradeoffs.py',
    'patches/tilelang_dense_preparation.py',
    'patches/tilelang_grouped_fused.py',
    'patches/tilelang_grouped_qk.py',
    'patches/tilelang_grouped_tp.py',
    'patches/tilelang_grouped_tp_kitchen.py',
    'patches/tilelang_grouped_tp_kitchen_v2.py',
    'patches/tilelang_grouped_tp_optimized.py',
    'patches/tilelang_sage_preparation.py',
    'patches/tilelang_sage_qk.py',
    'patches/tilelang_scalar_pool.py',
    'patches/tilelang_sol_drop_fused.py',
    'patches/tilelang_sparse_attention.py',
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
    plan=json.loads(args.baseline_request.read_text(encoding='utf-8'))
    result=json.loads(args.baseline_result.read_text(encoding='utf-8'))
    assert result['status']=='complete' and len(result['sampler_steps'])==8
    assert (plan['width'],plan['height'],plan['frames'],plan['steps'],plan['seed'])==(1216,704,243,8,20261020)
    plan['operators']=dict(
        te_attention='kitchen_causal',
        dit_attention='kitchen',
        dit_attention_schedule='2-7:sol@1.0',
        sol_tau=1.0,
        sol_topology_v2=True,
        sol_sync_free_lut=True,
        sol_collect_route_stats=False,
        sol_temporal_radius=1,
        sol_local_radius_tokens=64,
        sol_protect_frame_boundaries=True,
        video_attention='kitchen',
    )
    plan['name']='turbo_085mp_10s_sol_v2_production'
    plan['source_note']=(
        'Exact completed 0.85MP regular Turbo workload inputs replayed against the promoted production Sol-v2 path. '
        'Production Sol semantics: native D64/D128, H3 token-space temporal/local routing, exact generated-audio prefix, '
        'frame-boundary protection, sync-free LUT, and metadata-only runtime checkpoint admission. '
        'TE Kitchen causal; DiT NFE1/NFE8 Kitchen and NFE2-7 Sol-v2 All-INT8 tau=1.0; Video VAE Kitchen INT8. '
        'Route-stat collection is disabled for timed production measurement. Prompt, references, seed, LoRA, sampler, shifts and geometry unchanged.'
    )
    validate(plan)

    hashes={}
    archive_buffer=io.BytesIO()
    with tarfile.open(fileobj=archive_buffer,mode='w:') as archive:
        for name in overlay:
            raw=(ROOT/name).read_bytes()
            hashes[name]=hashlib.sha256(raw).hexdigest()
            info=tarfile.TarInfo(name)
            info.size=len(raw);info.mtime=0;info.uid=0;info.gid=0;info.uname='';info.gname=''
            archive.addfile(info,io.BytesIO(raw))
    archive_raw=lzma.compress(archive_buffer.getvalue(),preset=9|lzma.PRESET_EXTREME)
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
out=pathlib.Path('/kaggle/working/sol_v2_production_e2e');out.mkdir()
manifest=json.loads(pathlib.Path('/kaggle/temp/h3_light_distribution/delivery_manifest.json').read_text())
for name,expected in manifest['source_hashes'].items():
    path=source/name
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Frozen baseline source changed: '+name
hashes=HASHES
archive_raw=base64.b85decode(OVERLAY_ARCHIVE)
assert hashlib.sha256(archive_raw).hexdigest()==OVERLAY_ARCHIVE_SHA256
with tarfile.open(fileobj=io.BytesIO(lzma.decompress(archive_raw)),mode='r:') as archive:
    # The Kaggle Python image is part of the frozen environment, but avoid
    # depending on tarfile.extractall(filter=...), whose availability differs
    # across Python patch levels. The builder emits regular files only.
    for member in archive.getmembers():
        rel=pathlib.PurePosixPath(member.name)
        assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
        target=source.joinpath(*rel.parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        payload=archive.extractfile(member)
        assert payload is not None
        target.write_bytes(payload.read())
for name,expected in hashes.items():
    target=source/name
    assert target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==expected,'Overlay hash mismatch: '+name
for name in RETIRED:
    target=source/name
    if target.exists():target.unlink()
(out/'source_overlay.json').write_text(json.dumps(dict(
    baseline_capsule_sha256=manifest['capsule_sha256'],
    overlay_hashes=hashes,retired=RETIRED,
    scope='Production Sol-v2 long-sequence E2E overlay after physical-T4 microgate and Real-H3 attention qualification'
),indent=2))
request=REQUEST
(out/'request.json').write_text(json.dumps(request,indent=2))
argv=[sys.executable,'-u',str(source/'benchmarks/run_h3_l1_turbo_batch.py'),
      '--source',str(source),'--output',str(out),'--plan',str(out/'request.json'),
      '--scratch','/tmp/h3_sol_v2_production_20261030','--e2e-timeout','5400']
os.execve(sys.executable,argv,env)
'''.replace('OVERLAY_ARCHIVE_SHA256',repr(archive_sha256)).replace(
        'OVERLAY_ARCHIVE',repr(archive_b85)).replace('HASHES',repr(hashes)).replace(
        'RETIRED',repr(RETIRED)).replace('REQUEST',repr(plan))
    code=prefix+supervise_worker(execution,budget=7200,output='/kaggle/working/sol_v2_production_supervision')
    code+="assert json.loads(pathlib.Path('/kaggle/working/sol_v2_production_e2e/batch_results.json').read_text())['status']=='complete'\n"
    ast.parse(code)
    assert not re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_\-]{20,})',code)
    nb['cells'][0]['source']=code.splitlines(keepends=True)
    (job/'experiment.ipynb').write_text(json.dumps(nb),encoding='utf-8')
    if (job/'experiment.ipynb').stat().st_size>=900_000:
        raise ValueError('Notebook overlay exceeds the proven SaveKernel size envelope')

    meta=json.loads((baseline_job/'kernel-metadata.json').read_text(encoding='utf-8'))
    meta.update(id='pamcks/h3-sol-v2-production-e2e-20261030',
                title='H3 Sol V2 Production E2E 20261030')
    assert meta['is_private'] and meta['dataset_sources']==meta['kernel_sources']==[]
    (job/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

    binding=dict(
        baseline_source_binding=baseline_binding,
        baseline_result_sha256=hashlib.sha256(args.baseline_result.read_bytes()).hexdigest(),
        baseline_request_sha256=hashlib.sha256(args.baseline_request.read_bytes()).hexdigest(),
        overlay_hashes=hashes,
        overlay_dependency_closure=list(closure),
        overlay_baseline_commit=baseline_commit,
        overlay_archive_sha256=archive_sha256,
        overlay_archive_bytes=len(archive_raw),
        retired_paths=RETIRED,
        request_contract=validate(plan),
        attention_policy=plan['operators'],
        local_gates=dict(
            sol_contract_tests='benchmarks.test_sol_production',
            configuration_tests='benchmarks.test_h3_configuration_defaults',
            sm75_native_compile='Current native source is source-locked and AOT-compiled in the job; current full production overlay still requires physical-T4 E2E qualification',
        ),
        new_cpu_smoke_run=True,
        prior_cpu_qualification_reused_for_environment_only=False,
        changed_source_not_claimed_cpu_qualified=False,
        full_pipeline_authorized=True,
        gpu_count=2,
        zero_mounts=True,
        status_polling=False,
        live_log_polling=False,
    )
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    (args.evidence/'request.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (args.evidence/'build_receipt.json').write_text(json.dumps(dict(
        status='ready',notebook_bytes=(job/'experiment.ipynb').stat().st_size,
        notebook_sha256=hashlib.sha256((job/'experiment.ipynb').read_bytes()).hexdigest(),
        source_binding=binding),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='built',job=str(job),
        notebook_bytes=(job/'experiment.ipynb').stat().st_size,attention_policy=plan['operators']),indent=2))
    if not args.submit:return

    receipt=args.evidence/'gpu_submission.json'
    state=dict(kernel=meta['id'],status='submitting',
        submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding,run_timeout_seconds=8100,final_collection_count=0,
        result_relative='sol_v2_production_e2e/batch_results.json',
        notebook_sha256=hashlib.sha256((job/'experiment.ipynb').read_bytes()).hexdigest())
    with receipt.open('x',encoding='utf-8') as f:json.dump(state,f,indent=2)
    proc=subprocess.run([sys.executable,str(ROOT/'benchmarks/kaggle_tls12_runner.py'),
        '--cli','kernels','push','-p',str(job)],capture_output=True,text=True,timeout=180)
    output=proc.stdout+proc.stderr
    state.update(status='accepted' if proc.returncode==0 else 'acceptance_uncertain',
                 returncode=proc.returncode,response=output)
    m=re.search(r'Kernel version (\d+)',output)
    if m:state['version']=int(m.group(1))
    u=re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',output)
    if u:state['url']=u.group(0)
    receipt.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    (args.evidence/'gpu_submission_cli.log').write_text(output,encoding='utf-8')
    print(output[-2000:])
    if proc.returncode:raise RuntimeError('Kaggle submission was not accepted; no automatic retry')


if __name__=='__main__':main()
