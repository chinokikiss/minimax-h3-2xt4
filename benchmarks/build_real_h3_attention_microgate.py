"""Build one Real-H3 first-attention microgate from the completed 0.85 MP / 10 s baseline notebook.

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
    'benchmarks/real_h3_attention_microgate.py',
    'benchmarks/run_real_h3_attention_microgate_batch.py',
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
    p.add_argument('--workspace-pool',action='store_true',help='experimental stream-local routing workspace')
    p.add_argument('--bitmap-ab',action='store_true',help='same-captured-QKV Sol v2 original vs bitmap LUT')
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
    plan['name']='turbo_085mp_10s_real_h3_attention_microgate'
    plan['source_note']=(
        'Exact completed 0.85MP/10s Turbo request is used only to produce the first real post-RMS/post-RoPE DiT Q/K/V. '
        'Execution aborts at the first attention before any DiT attention result propagates. '
        'Four contiguous rank-local heads are replayed in-place through Kitchen INT8, old Sol tau1 compact/sync-free, '
        'and Sol-v2 topology-aware tau 1.0/0.7/0.5. No full denoise trajectory, VAE decode, or media export.'
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
env['H3_SOL_WORKSPACE_POOL']='1' if OPT_WORKSPACE_POOL else '0'
env['H3_SOL_LUT_BITMAP']='0'
env['H3_REAL_SOL_BITMAP_AB']='1' if OPT_BITMAP_AB else '0'
source=pathlib.Path('/kaggle/temp/h3_light_source')
out=pathlib.Path('/kaggle/working/real_h3_attention_microgate');out.mkdir()
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
    scope='Real-H3 first-attention capture/replay overlay; no full-pipeline claim'
),indent=2))
request=REQUEST
(out/'request.json').write_text(json.dumps(request,indent=2))
argv=[sys.executable,'-u',str(source/'benchmarks/run_real_h3_attention_microgate_batch.py'),
      '--source',str(source),'--output',str(out),'--plan',str(out/'request.json'),
      '--scratch','/tmp/h3_real_h3_attention_microgate_20261030','--e2e-timeout','3600']
os.execve(sys.executable,argv,env)
'''.replace('OVERLAY_ARCHIVE_SHA256',repr(archive_sha256)).replace(
        'OVERLAY_ARCHIVE',repr(archive_b85)).replace('HASHES',repr(hashes)).replace(
        'RETIRED',repr(RETIRED)).replace('REQUEST',repr(plan)).replace(
        'OPT_WORKSPACE_POOL',repr(bool(args.workspace_pool))).replace(
        'OPT_BITMAP_AB',repr(bool(args.bitmap_ab)))
    code=prefix+supervise_worker(execution,budget=5400,output='/kaggle/working/real_h3_attention_microgate_supervision')
    code+="assert json.loads(pathlib.Path('/kaggle/working/real_h3_attention_microgate/batch_results.json').read_text())['status']=='complete'\n"
    ast.parse(code)
    assert not re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_\-]{20,})',code)
    nb['cells'][0]['source']=code.splitlines(keepends=True)
    (job/'experiment.ipynb').write_text(json.dumps(nb),encoding='utf-8')
    if (job/'experiment.ipynb').stat().st_size>=900_000:
        raise ValueError('Notebook overlay exceeds the proven SaveKernel size envelope')

    meta=json.loads((baseline_job/'kernel-metadata.json').read_text(encoding='utf-8'))
    if args.bitmap_ab:
        kernel_id='pamcks/h3-real-h3-attn-bitmap-20261031'
    elif args.workspace_pool:
        kernel_id='pamcks/h3-real-h3-attn-pool-20261031'
    else:
        kernel_id='pamcks/h3-real-h3-attention-microgate-20261030'
    meta.update(id=kernel_id, code_file='experiment.ipynb',
                title='H3 Real H3 Attention Microgate '+('Opt 20261031' if
                    args.workspace_pool or args.bitmap_ab else '20261030'))
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
        sol_compiler_options=dict(workspace_pool=bool(args.workspace_pool),
                                  bitmap_ab=bool(args.bitmap_ab)),
        experiment_scope='First real post-RoPE DiT attention only; operator replay on 4 local heads per rank',
        local_gates=dict(
            sol_contract_tests='benchmarks.test_sol_production',
            configuration_tests='benchmarks.test_h3_configuration_defaults',
            sm75_native_compile='Current native source is source-locked and AOT-compiled in the job; current full production overlay still requires physical-T4 E2E qualification',
        ),
        new_cpu_smoke_run=False,
        prior_cpu_qualification_reused_for_environment_only=True,
        changed_source_not_claimed_cpu_qualified=True,
        full_pipeline_authorized=False,
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
        source_binding=binding,run_timeout_seconds=6000,final_collection_count=0,
        result_relative='real_h3_attention_microgate/batch_results.json',
        notebook_sha256=hashlib.sha256((job/'experiment.ipynb').read_bytes()).hexdigest())
    with receipt.open('x',encoding='utf-8') as f:json.dump(state,f,indent=2)
    proc=subprocess.run([sys.executable,str(ROOT/'benchmarks/kaggle_tls12_runner.py'),
        '--cli','kernels','push','-p',str(job)],capture_output=True,text=True,timeout=180)
    output=proc.stdout+proc.stderr
    m=re.search(r'Kernel version (\d+)',output)
    provider_error=bool(re.search(r'(?im)^\s*(?:Kernel push error:|Error:)',output))
    accepted=proc.returncode==0 and m is not None and not provider_error
    state.update(status='accepted' if accepted else 'acceptance_uncertain',
                 returncode=proc.returncode,response=output,provider_error=provider_error)
    if m:state['version']=int(m.group(1))
    u=re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',output)
    if u:state['url']=u.group(0)
    receipt.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    (args.evidence/'gpu_submission_cli.log').write_text(output,encoding='utf-8')
    print(output[-2000:])
    if not accepted:raise RuntimeError('Kaggle submission was not accepted; no automatic retry')


if __name__=='__main__':main()
