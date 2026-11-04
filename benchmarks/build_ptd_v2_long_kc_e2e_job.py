"""Build private full eight-hop PTD Bank v2 1280x704/10s all-KC Kaggle job.
Preserves nine independently qualified immutable Dataset mounts; no uploads,
no second full-bank bake, no 170GB scratch copy or duplicate PTD payload SHA.
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

from benchmarks.build_ptd_v2_2nfe_http4_job import (
    ROOT,DEFAULT_BASE,SOURCE_FILES,_proven_bank,_gzip_bundle,_baseline_commit
)
from benchmarks.build_h3_t4_fp16_512_sol_all_8nfe_1mp_10s import REFERENCE_STAGE
from benchmarks.source_overlay_closure import overlay_files,assert_closed
from benchmarks.t4_notebook_watchdog import supervise_worker
from benchmarks.ptd_v2_long_kc_contract import from_previous,validate

KAGGLE_ID='pamcks/h3-ptd-v2-e2e-0-9mp-10s-all-kc-20261104'
TITLE='H3 PTD V2 E2E 0.9MP 10s All KC 20261104'
FORMAT='h3-ptd-v2-long-kc-e2e-delivery-v1'
EVIDENCE=ROOT/'docs/data/h3_ptd_v2_long_kc_e2e_20261104'
INPUT=ROOT/'docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102/request.json'
EXTRA=(
    'benchmarks/ptd_v2_long_kc_contract.py',
    'benchmarks/run_ptd_v2_long_kc_e2e.py',
    'benchmarks/run_ptd_v2_long_kc_batch.py',
    'benchmarks/run_ptd_e2e.py',
    'benchmarks/run_h3_l1_turbo_e2e.py',
    'benchmarks/run_h3_l1_turbo_batch.py',
    'benchmarks/ref2va_e2e_run.py',
    'benchmarks/ptd_e2e_contract.py',
    'benchmarks/e2e_attention_validation.py',
    'benchmarks/qualify_anime_inputs.py',
    'benchmarks/checkpoint_ranges.py',
    'patches/staged_vae.py',
    'patches/parallel_vae.py',
    'patches/comfy_sampler.py',
    'patches/h3_operator_policy.py',
    'patches/h3_operator_dependencies.py',
    'patches/h3_tensor_flow.py',
    'patches/ptd/nodes.py',
)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(base=DEFAULT_BASE,output=None, *, range2_hostcache=False):
    base=Path(base).resolve()
    variant_tag = ('h3_ptd_v2_09mp_range2_hostcache_20261104' if range2_hostcache
                   else 'h3_ptd_v2_09mp_10s_kc_e2e_20261104')
    variant_kernel = ('pamcks/h3-ptd-v2-range2-hostcache-09mp-20261104'
                      if range2_hostcache else KAGGLE_ID)
    variant_title = ('H3 PTD V2 Range2 Hostcache 09mp 20261104'
                     if range2_hostcache else TITLE)
    variant_evidence = (ROOT/'docs/data/h3_ptd_v2_range2_hostcache_e2e_20261104'
                        if range2_hostcache else EVIDENCE)
    if re.sub(r'[^a-z0-9]+','-',variant_title.lower()).strip('-') != variant_kernel.split('/',1)[1]:
        raise ValueError('Kaggle title/Kernel ID must resolve to the same slug')
    output=Path(output or (variant_evidence/'gpu_job')).resolve()
    if output.exists():raise FileExistsError('Never overwrite a Kaggle experiment: '+str(output))
    manifest,proofs=_proven_bank()
    plan=from_previous(json.loads(INPUT.read_text(encoding='utf-8')))
    contract=validate(plan)
    pinned_commit=_baseline_commit(base)
    seeds=tuple(sorted(set(SOURCE_FILES+EXTRA)))
    names,closure=overlay_files(pinned_commit,seeds)
    assert_closed(pinned_commit,names,seeds)
    src_map={name:(ROOT/name).read_bytes() for name in names}
    hashes={name:hashlib.sha256(value).hexdigest() for name,value in src_map.items()}
    capsule=_gzip_bundle(src_map)
    capsule_sha=hashlib.sha256(capsule).hexdigest()
    payloads={
        'proofs.json':(json.dumps(proofs,sort_keys=True,separators=(',',':'))+'\n').encode(),
        'request.json':(json.dumps(plan,sort_keys=True,separators=(',',':'))+'\n').encode(),
    }
    binding_payload=_gzip_bundle(payloads)
    binding_sha=hashlib.sha256(binding_payload).hexdigest()
    nb=json.loads(base.read_text(encoding='utf-8'))
    original=''.join(nb['cells'][0]['source'])
    marker="assert receipt['status']=='ready_for_device_gate'\n"
    if original.count(marker)!=1:raise ValueError('Frozen baseline notebook setup marker changed')
    setup=original[:original.index(marker)+len(marker)]
    inject=f"""
import base64,hashlib,io,json,os,pathlib,sys,tarfile,time
SOURCE=pathlib.Path('/kaggle/temp/h3_light_source')
BIND_ROOT=pathlib.Path('/kaggle/working/{variant_tag}')
BIND_ROOT.mkdir(parents=True,exist_ok=False)
source=SOURCE
out=BIND_ROOT
def unpack_locked(blob,digest,hashes,root):
    raw=base64.b64decode(blob)
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Source capsule digest mismatch')
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        if set(archive.getnames())!=set(hashes):raise ValueError('Source capsule inventory changed')
        for member in archive:
            target=pathlib.PurePosixPath(member.name)
            if not member.isfile() or target.is_absolute() or '..' in target.parts:
                raise ValueError('Unsafe capsule member')
            data=archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=hashes[member.name]:
                raise ValueError('Source member digest mismatch: '+member.name)
            destination=root/member.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(data)
SOURCE_HASHES={hashes!r}
unpack_locked({base64.b64encode(capsule).decode('ascii')!r},
              {capsule_sha!r},SOURCE_HASHES,SOURCE)
METADATA_HASHES={dict((k,hashlib.sha256(v).hexdigest()) for k,v in payloads.items())!r}
unpack_locked({base64.b64encode(binding_payload).decode('ascii')!r},
              {binding_sha!r},METADATA_HASHES,BIND_ROOT)
(BIND_ROOT/'notebook_admission.json').write_text(json.dumps(dict(
    format={FORMAT!r},full_pipeline=True,ptd_hops=8,dit_attention='kitchen',
    payload_sha256_repeated=False,provider_pre_kernel_dataset_attach_timing_unobservable=True,
    qualified_dataset_identities={dict((k,v['published_identity']) for k,v in proofs.items())!r},
    source_sha256={capsule_sha!r}),indent=2))
"""
    # Reference is fetched from immutable commit-pinned public mirror if baseline
    # source does not already include the 7.7MB image; character stays private.
    inject+=REFERENCE_STAGE
    worker=r"""import json,os,pathlib,sys
env=dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
env['H3_PTD_E2E_ALLOW_UNVALIDATED_CUDA13']='1'
source=pathlib.Path('/kaggle/temp/h3_light_source')
root=pathlib.Path('/kaggle/working/h3_ptd_v2_09mp_10s_kc_e2e_20261104')
argv=[sys.executable,'-u',str(source/'benchmarks/run_ptd_v2_long_kc_batch.py'),
      '--source',str(source),'--output',str(root/'trial'),
      '--plan',str(root/'request.json'),'--proofs',str(root/'proofs.json'),
      '--timeout','8400']
os.execve(sys.executable,argv,env)
"""
    if range2_hostcache:
        worker=worker.replace("env['H3_PTD_E2E_ALLOW_UNVALIDATED_CUDA13']='1'",
            "env['H3_PTD_E2E_ALLOW_UNVALIDATED_CUDA13']='1'\nenv['H3_PTD_HTTP_RANGE2_HOSTCACHE']='1'\nenv['H3_PTD_HTTP_RANGE2_DEBUG']='1'")
    worker=worker.replace('h3_ptd_v2_09mp_10s_kc_e2e_20261104',variant_tag)
    code=setup+inject+supervise_worker(worker,budget=10200,
        output='/kaggle/working/'+variant_tag+'_supervision')
    code+=("assert json.loads(pathlib.Path('/kaggle/working/"+variant_tag+
          "/trial/batch_results.json').read_text())['status']=='complete'\n")
    ast.parse(code)
    if re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_-]{20,})',code):
        raise ValueError('Potential credential accidentally embedded in source')
    nb['cells'][0]['source']=code.splitlines(keepends=True)
    meta=json.loads((base.parent/'kernel-metadata.json').read_text(encoding='utf-8'))
    meta.pop('id_no',None)
    meta.update(id=variant_kernel,title=variant_title,code_file='experiment.ipynb',
        is_private=True,enable_gpu=True,enable_internet=True,
        machine_shape='NvidiaTeslaT4',kernel_sources=[],model_sources=[],
        competition_sources=[],dataset_sources=[proofs[p]['dataset']
            for p in ('shared',)+(tuple(f'hop_{i:02d}' for i in range(8)))])
    output.mkdir(parents=True)
    (output/'experiment.ipynb').write_text(json.dumps(nb),encoding='utf-8')
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    record=dict(format=FORMAT,status='prepared_not_submitted',
        kernel=variant_kernel,notebook_sha256=digest(output/'experiment.ipynb'),
        notebook_bytes=(output/'experiment.ipynb').stat().st_size,
        source_archive_sha256=capsule_sha,source_archive_bytes=len(capsule),
        frozen_base_commit=pinned_commit,source_files=len(names),
        changed_dependency_closure=list(closure),plan_contract=contract,
        prompt_sha256=plan['prompt_sha256'],reference_sha256=[v['sha256'] for v in plan['reference_images']],
        trained_hops=8,layers=50,full_e2e=True,attention='all_eight_kitchen',
        trained_video_shift=12.0,trained_audio_shift=3.0,
        gpu_hardware='2xTeslaT4',dataset_sources=meta['dataset_sources'],
        remote_proof_identifiers={k:v['published_identity'] for k,v in proofs.items()},
        recipe_sha256=manifest['recipe_sha256'],source_overlays_hashes=hashes,
        no_online_lora=True,no_full_payload_sha256=True,
        mounts_not_http_staged=True,io_workers_per_rank=1,
        http_range2_to_hostweightcache=range2_hostcache,
        http_range2_debug=range2_hostcache,
        experimental_cuda13_opt_in='H3_PTD_E2E_ALLOW_UNVALIDATED_CUDA13',
        worker_timeout=8400,notebook_supervisor_timeout=10200)
    (output/'source_binding.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (output.parent/'request.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (output.parent/'build_receipt.json').write_text(json.dumps(
        dict(status='ready',notebook_sha256=record['notebook_sha256'],
            notebook_bytes=record['notebook_bytes'],kernel=KAGGLE_ID),indent=2),encoding='utf-8')
    return record

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path)
    p.add_argument('--base',type=Path,default=DEFAULT_BASE)
    p.add_argument('--range2-hostcache',action='store_true',help='Opt-in HTTP Range2 direct tensor host-cache PTD E2E')
    args=p.parse_args()
    result=build(args.base,args.output,range2_hostcache=args.range2_hostcache)
    print(json.dumps(dict(status=result['status'],kernel=result['kernel'],
        source_files=result['source_files'],notebook_bytes=result['notebook_bytes'],
        notebook_sha256=result['notebook_sha256']),indent=2))

if __name__=='__main__':main()
