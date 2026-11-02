"""Build one no-weight physical-T4 A/B: bidirectional vs key-only Sol prefix.

Frozen production sources are copied twice and changed only in the experiment's
candidate source tree. No Kaggle status/list/live-log calls, no submission here.
"""
from __future__ import annotations
import argparse,ast,base64,hashlib,io,json,lzma,tarfile
from pathlib import Path
from benchmarks.t4_notebook_watchdog import supervise_worker

ROOT=Path(__file__).resolve().parents[1]
SOURCE=[
    "benchmarks/sol_cond_protection_ab_probe.py",
    "patches/sol_attention.py",
    "patches/sol_sm75_backend.py",
]
NATIVE=ROOT/"patches/sol_attention_native"
SOURCE += [p.relative_to(ROOT).as_posix() for p in sorted(NATIVE.rglob("*")) if p.is_file()]
PATCH="patches/sol_attention_native/sol_sm75_preprocess.cuh"
CHANGES=(
    ("neighbor || protected_key || protected_query || topology_neighbor;",
     "neighbor || protected_key || topology_neighbor;"),
    ("neighbor||protected_key||protected_query||topology_neighbor;",
     "neighbor||protected_key||topology_neighbor;"),
)
EVIDENCE=ROOT/"docs/data/h3_sol_unidirectional_ab_20261101"
KERNEL="pamcks/h3-sol-unidirectional-protection-ab-20261101"
CAPTURE_KERNEL="pamcks/h3-real-sol-comfy-capture-20261031"


def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()


def main():
    job=EVIDENCE/"gpu_job"
    if job.exists():raise FileExistsError("Preserve existing A/B job: "+str(job))
    assert len(SOURCE)==len(set(SOURCE))
    before=(ROOT/PATCH).read_text(encoding="utf-8")
    mutated=before
    for original,candidate in CHANGES:
        assert before.count(original)==1,(original,before.count(original))
        mutated=mutated.replace(original,candidate,1)
    assert mutated.count("protected_query") == before.count("protected_query")-2
    assert "|| protected_key || topology_neighbor;" in mutated
    assert "||protected_key||topology_neighbor;" in mutated
    # Guard that only the intended two forced-exact expressions changed.
    check=mutated
    for original,candidate in CHANGES:
        assert check.count(candidate)==1
        check=check.replace(candidate,original,1)
    assert check==before
    source_hashes={}
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode="w:") as tar:
        for name in SOURCE:
            raw=(ROOT/name).read_bytes()
            source_hashes[name]=sha(raw)
            info=tarfile.TarInfo(name)
            info.size=len(raw);info.mtime=0;info.uid=info.gid=0
            info.uname=info.gname=""
            tar.addfile(info,io.BytesIO(raw))
    compressed=lzma.compress(buf.getvalue(),preset=9|lzma.PRESET_EXTREME)
    payload=base64.b85encode(compressed).decode("ascii")
    mutant_sha=sha(mutated.encode("utf-8"))
    notebook_worker=r'''import base64,hashlib,io,json,lzma,os,pathlib,statistics,subprocess,sys,tarfile,time
import torch
ROOT=pathlib.Path('/kaggle/temp/h3_sol_cond_ab')
ROOT.mkdir(parents=True,exist_ok=False)
OUT=pathlib.Path('/kaggle/working/sol_cond_protection_ab')
OUT.mkdir(parents=True,exist_ok=True)
RAW=lzma.decompress(base64.b85decode(PAYLOAD))
assert hashlib.sha256(base64.b85decode(PAYLOAD)).hexdigest()==PAYLOAD_SHA
for variant in ('bidirectional','key_only'):
    dest=ROOT/variant
    dest.mkdir()
    with tarfile.open(fileobj=io.BytesIO(RAW),mode='r:') as tf:
        for entry in tf:
            name=pathlib.PurePosixPath(entry.name)
            assert entry.isfile() and not name.is_absolute() and '..' not in name.parts
            path=dest.joinpath(*name.parts)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(tf.extractfile(entry).read())
    for name,expected in HASHES.items():
        assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==expected,(variant,name)
    if variant=='key_only':
        native=dest/PATCH
        content=native.read_text()
        for old,new in REPLACEMENTS:
            assert content.count(old)==1,(old,content.count(old))
            content=content.replace(old,new,1)
        native.write_text(content)
        assert hashlib.sha256(native.read_bytes()).hexdigest()==MUTANT_SHA
assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(7,5)
import shutil
nvcc=shutil.which('nvcc')
assert nvcc and 'release 12.8' in subprocess.check_output([nvcc,'--version'],text=True)
# Optional immutable source: already saved real post-RMS/post-RoPE H3 QKV.
# If the Kaggle source kernel's failed v1 output is not mounted, the result is
# explicitly marked synthetic-only; never label it as real-model accuracy.
captures=list(pathlib.Path('/kaggle/input').rglob('rank0_real_qkv.pt'))
capture_root=next((p.parent for p in captures if (p.parent/'rank1_real_qkv.pt').is_file()),None)
state=dict(status='running',gpu=torch.cuda.get_device_name(0),capability=[7,5],
    captured_real_data=capture_root is not None,capture_kernel=CAPTURE_KERNEL,
    capture_path=str(capture_root) if capture_root else None,
    native_patch_sha256_baseline=HASHES[PATCH],
    native_patch_sha256_candidate=MUTANT_SHA,
    scope='single physical T4; same frozen real QKV when mounted, plus synthetic 4head real geometry',
    order=['bidirectional','key_only','key_only','bidirectional'],runs={})
def save():(OUT/'summary.json').write_text(json.dumps(state,indent=2))
save()
order=state['order']
for index,variant in enumerate(order):
    variant_out=OUT/('run'+str(index)+'_'+variant)
    env=dict(os.environ)
    env['PYTHONPATH']=str(ROOT/variant)
    env['H3_SOL_WORKSPACE_POOL']='0'
    env['H3_SOL_LUT_BITMAP']='0'
    env['H3_SOL_CACHE']='/kaggle/temp/h3_sol_cond_ab_cache'
    cmd=[sys.executable,'-u',str(ROOT/variant/'benchmarks/sol_cond_protection_ab_probe.py'),
        '--variant',variant,'--out',str(variant_out),'--repeat-id',str(index)]
    if capture_root:cmd+=['--capture-root',str(capture_root)]
    begin=time.monotonic()
    proc=subprocess.run(cmd,cwd=ROOT/variant,env=env,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,timeout=750)
    (OUT/('run'+str(index)+'_stdout.log')).write_text(proc.stdout)
    state['runs'][str(index)]=dict(variant=variant,exit_code=proc.returncode,
        seconds=time.monotonic()-begin,path=str(variant_out.relative_to(OUT)))
    save()
    print('SOL_COND_AB_RUN',index,variant,proc.returncode,proc.stdout[-1600:],flush=True)
    if proc.returncode:raise RuntimeError('Failed A/B '+variant+' run '+str(index))
    state['runs'][str(index)]['result']=json.loads((variant_out/'result.json').read_text())
    save()
# Accurate paired output comparison, including route difference scope.
def metric(a,b):
    a=a.float();b=b.float()
    diff=a-b
    return dict(rel_l2=float(torch.linalg.vector_norm(diff)/
        torch.linalg.vector_norm(b).clamp_min(1.e-12)),
        mae=float(diff.abs().mean()),max_abs=float(diff.abs().max()),
        finite=bool(torch.isfinite(a).all()),exact_equal=bool(torch.equal(a,b)))
def med_ms(ids,case,phase,key):
    return statistics.median(state['runs'][str(i)]['result']['cases'][case][phase][key]
        for i in ids)
comparison={}
for name in ['synthetic_4head_full_length']+(['rank0_real','rank1_real'] if capture_root else []):
    timing={}
    for phase in ('prepare','execute','complete'):
        basems=med_ms((0,3),name,phase,'wall_median_ms')
        onems=med_ms((1,2),name,phase,'wall_median_ms')
        bgpu=med_ms((0,3),name,phase,'gpu_event_median_ms')
        ogpu=med_ms((1,2),name,phase,'gpu_event_median_ms')
        timing[phase]=dict(bidirectional_wall_ms=basems,key_only_wall_ms=onems,
            speedup_wall=basems/onems,bidirectional_gpu_ms=bgpu,key_only_gpu_ms=ogpu,
            speedup_gpu=bgpu/ogpu)
    base=state['runs']['0']['result']['cases'][name]['routed']
    one=state['runs']['1']['result']['cases'][name]['routed']
    data=dict(timing=timing,baseline_routes=base,key_only_routes=one,
        pair_reduction=base['total']-one['total'],
        global_route_density_before=base['total']/base['all_pairs'],
        global_route_density_after=one['total']/one['all_pairs'])
    if capture_root:
        a=torch.load(OUT/('run0_bidirectional/'+name+'_comparison.pt'),
            map_location='cpu',weights_only=True)
        b=torch.load(OUT/('run1_key_only/'+name+'_comparison.pt'),
            map_location='cpu',weights_only=True)
        pb=base['prefix_blocks'];start=4974
        assert a['output'].shape==b['output'].shape
        assert a['mask'].shape==b['mask'].shape
        assert torch.equal(a['mask'][...,pb:,:],b['mask'][...,pb:,:]),'video routing changed'
        assert torch.equal(a['mask'][...,:,:pb],b['mask'][...,:,:pb]),'prefix K routing changed'
        assert torch.all(a['mask']>=b['mask']),'candidate added exact pairs'
        assert torch.equal(a['output'][:,:,start:],b['output'][:,:,start:]),'video output changed'
        assert a['dense_indices']==b['dense_indices'] and torch.equal(a['dense'],b['dense'])
        denseidx=torch.tensor(a['dense_indices'],dtype=torch.long)
        before=a['output'][:,:,denseidx]
        after=b['output'][:,:,denseidx]
        dense=a['dense']
        prefix_sel=(denseidx<start).nonzero().flatten()
        video_sel=(denseidx>=start).nonzero().flatten()
        data['precision']=dict(
            whole_vs_bidirectional=metric(b['output'],a['output']),
            prefix_vs_bidirectional=metric(b['output'][:,:,:start],a['output'][:,:,:start]),
            audio_vs_bidirectional=metric(b['output'][:,:,4164:start],a['output'][:,:,4164:start]),
            video_vs_bidirectional=metric(b['output'][:,:,start:],a['output'][:,:,start:]),
            sampled_dense_queries=len(denseidx),
            sampled_fp32_dense=dict(
                bidirectional_all=metric(before,dense),
                key_only_all=metric(after,dense),
                bidirectional_prefix=metric(before[:,:,prefix_sel],dense[:,:,prefix_sel]),
                key_only_prefix=metric(after[:,:,prefix_sel],dense[:,:,prefix_sel]),
                bidirectional_video=metric(before[:,:,video_sel],dense[:,:,video_sel]),
                key_only_video=metric(after[:,:,video_sel],dense[:,:,video_sel])))
    comparison[name]=data
state['comparison']=comparison
state['status']='complete' if capture_root else 'complete_synthetic_only'
save()
print('SOL_COND_PROTECTION_AB_COMPLETE',state['status'],json.dumps({
    n:{p:row['timing'][p]['speedup_wall'] for p in ('prepare','execute','complete')}
    for n,row in comparison.items()}),flush=True)
'''
    for key,value in (
        ("PAYLOAD_SHA",repr(sha(compressed))),("PAYLOAD",repr(payload)),
        ("HASHES",repr(source_hashes)),("PATCH",repr(PATCH)),
        ("REPLACEMENTS",repr(CHANGES)),("MUTANT_SHA",repr(mutant_sha)),
        ("CAPTURE_KERNEL",repr(CAPTURE_KERNEL)),
    ):
        import re
        notebook_worker=re.sub(r"\b"+key+r"\b",lambda _match:value,notebook_worker)
    ast.parse(notebook_worker)
    code=supervise_worker(notebook_worker,budget=3300,
        output='/kaggle/working/sol_cond_protection_ab_supervision')
    code+="assert json.loads(pathlib.Path('/kaggle/working/sol_cond_protection_ab/summary.json').read_text())['status'] in ('complete','complete_synthetic_only')\n"
    ast.parse(code)
    notebook=dict(nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(name="python3",language="python",
            display_name="Python 3")),
        cells=[dict(id="sol-cond-protection-ab",cell_type="code",
            execution_count=None,metadata={},outputs=[],
            source=code.splitlines(keepends=True))])
    encoded=json.dumps(notebook).encode()
    assert len(encoded)<900_000
    metadata=dict(id=KERNEL,title="H3 Sol Bidirectional vs One-Way Condition Protection AB",
        code_file="experiment.ipynb",language="python",kernel_type="notebook",
        is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,
        machine_shape="NvidiaTeslaT4",docker_image_pinning_type="original",
        dataset_sources=[],kernel_sources=[CAPTURE_KERNEL],
        competition_sources=[],model_sources=[])
    binding=dict(scope="zero-weight single-physical-T4 experimental A/B",
        baseline_native_sha256=source_hashes[PATCH],
        candidate_native_sha256=mutant_sha,changed_cuda_expressions=2,
        production_sources_untouched=True,
        source_hashes=source_hashes,
        source_archive_sha256=sha(compressed),notebook_sha256=sha(encoded),
        capture_kernel=CAPTURE_KERNEL,
        capture_rank_sha256={
            "0":"4c9116fd7f603539c4e49e7e440c034c893e246f8649371b21376483fafd34e8",
            "1":"9138f989a127295555ace8ca3914e007c0f3d6d9cd5f26f296c48789ad80bc91"},
        references_no_model_weights=True,source_kernel_mount_required_for_real_accuracy=True,
        status_polling=False,live_log_polling=False,automatic_retry=False,
        result_relative="sol_cond_protection_ab/summary.json")
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    job.mkdir(parents=True,exist_ok=False)
    (job/"experiment.ipynb").write_bytes(encoded)
    (job/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2))
    (job/"source_binding.json").write_text(json.dumps(binding,indent=2))
    (EVIDENCE/"build_receipt.json").write_text(json.dumps(dict(
        status="ready",kernel=KERNEL,notebook_bytes=len(encoded),
        notebook_sha256=sha(encoded),baseline_native_sha256=source_hashes[PATCH],
        candidate_native_sha256=mutant_sha,changed_cuda_expressions=2,
        source_files=len(SOURCE),capture_kernel=CAPTURE_KERNEL,
        no_gpu_submission_yet=True),indent=2))
    print(json.dumps(dict(status="built",kernel=KERNEL,notebook_bytes=len(encoded),
        candidate_native_sha256=mutant_sha,source_files=len(SOURCE)),indent=2))


if __name__=="__main__":
    main()
