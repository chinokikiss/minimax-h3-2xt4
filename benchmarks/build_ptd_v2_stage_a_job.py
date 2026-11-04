"""Private Kaggle Stage A: two trained PTD blocks, eight fixture hops, tiny latent.

Derive the notebook from the previously qualified frozen GPU setup. This
builder only generates source-bound artifacts; it never submits or polls.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re

from build_ptd_v2_bake_job import ROOT, DEFAULT_BASE, _tar_bytes
from source_overlay_closure import overlay_files as closed_overlay_files, assert_closed
from t4_notebook_watchdog import supervise_worker

CORE_OVERLAY = (
    'patches/tp2_bundle.py',
    'patches/comfy_tp_runtime.py',
    'patches/ptd/__init__.py',
    'patches/ptd/bank_v2.py',
    'patches/ptd/bake_v2.py',
    'patches/ptd/composite_lora.py',
    'patches/ptd/publish_v2.py',
    'patches/ptd/checkpoint.py',
    'patches/ptd/mapping.py',
    'patches/ptd/runtime.py',
    'patches/ptd/queue.py',
    'patches/ptd/sampler.py',
    'patches/ptd/schedule.py',
    'patches/ptd/reference_sources.json',
    'benchmarks/run_ptd_v2_stage_a_t4.py',
    'benchmarks/run_ptd_tp2_t4.py',
    'benchmarks/ptd_t4_assets.py',
    'benchmarks/ptd_worker_probe.py',
    'benchmarks/tp2_public_dit_assets.py',
    'benchmarks/checkpoint_ranges.py',
)
KERNEL_ID = 'pamcks/h3-ptd-bank-v2-stage-a-two-block-t4-20261102'
OUT = '/kaggle/working/ptd_v2_stage_a_20261102'


def _baseline_commit(base):
    binding_path = Path(base).resolve().parent / 'source_binding.json'
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    commit = binding.get('baseline_commit')
    if commit is None:
        commit = binding.get('baseline_source_binding', {}).get('publication', {}).get('commit')
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError('Frozen Stage A baseline has no immutable publication commit')
    return commit


def _closed_overlay(base):
    commit = _baseline_commit(base)
    files, closure = closed_overlay_files(commit, CORE_OVERLAY)
    assert_closed(commit, files, CORE_OVERLAY)
    if 'patches/rank_local_int8.py' not in files:
        raise RuntimeError('Stage A source closure omitted rank_local_int8 API dependency')
    return commit, files, closure


def build(base, destination):
    base = Path(base).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Stage A GPU job already exists: preserve immutable attempts')
    baseline_commit, overlay, closure = _closed_overlay(base)
    rows = {name: (ROOT / name).read_bytes() for name in overlay}
    proofs = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(rows.items())}
    blob = _tar_bytes(rows)
    digest = hashlib.sha256(blob).hexdigest()
    baseline = json.loads(base.read_text(encoding='utf-8'))
    original = ''.join(baseline['cells'][0]['source'])
    boundary = "assert receipt['status']=='ready_for_device_gate'\n"
    if original.count(boundary) != 1 or 'bundle_assets.json' not in original[:original.index(boundary)]:
        raise ValueError('Unqualified frozen setup boundary or missing bundle asset manifest')
    setup = original[:original.index(boundary) + len(boundary)]
    injected = f"""
# Immutable explicit v2-only Stage A overlay after the frozen setup.
import base64,hashlib,io,json,pathlib,tarfile
SOURCE=pathlib.Path('/kaggle/temp/h3_light_source')
OVERLAY_SHA256={digest!r}
FILES={json.dumps(proofs, sort_keys=True)}
DATA=base64.b64decode({base64.b64encode(blob).decode('ascii')!r})
assert hashlib.sha256(DATA).hexdigest()==OVERLAY_SHA256
with tarfile.open(fileobj=io.BytesIO(DATA),mode='r:gz') as archive:
    assert set(archive.getnames())==set(FILES)
    for item in archive:
        assert item.isfile() and not item.name.startswith('/') and '..' not in pathlib.PurePosixPath(item.name).parts
        raw=archive.extractfile(item).read()
        assert hashlib.sha256(raw).hexdigest()==FILES[item.name]
        path=SOURCE/item.name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==FILES[item.name]
print('PTD_V2_STAGE_A_OVERLAY_READY',OVERLAY_SHA256,len(FILES),flush=True)
# Fresh-process import rehearsal: never trust modules cached by the frozen setup.
import os,subprocess,sys
probe_env=dict(os.environ)
probe_env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
probe_code="import sys\\nfrom pathlib import Path\\nsource=Path('/kaggle/temp/h3_light_source')\\nsys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]\\nimport patches.tp2_bundle\\nimport patches.ptd.bank_v2\\nimport patches.ptd.bake_v2\\nimport patches.ptd.runtime\\nimport benchmarks.run_ptd_v2_stage_a_t4\\n"
probe=subprocess.run([sys.executable,'-c',probe_code],env=probe_env,capture_output=True,text=True,timeout=90)
if probe.returncode:
    raise RuntimeError('PTD v2 Stage A source import gate failed: '+probe.stderr[-6000:])
print('PTD_V2_STAGE_A_SOURCE_IMPORT_GATE_PASS',flush=True)
"""
    execute = f"""import json,os,pathlib,sys
env=dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
cmd=[sys.executable,'-u',str(source/'benchmarks/run_ptd_v2_stage_a_t4.py'),
     '--source',str(source),'--output',{OUT!r}]
os.execve(sys.executable,cmd,env)
"""
    code = setup + injected + supervise_worker(
        execute, budget=1200, output='/kaggle/working/ptd_v2_stage_a_supervision')
    code += f"""assert json.loads(pathlib.Path({(OUT+'/result.json')!r}).read_text())['status']=='complete'
"""
    ast.parse(code)
    if re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})', code):
        raise ValueError('Potential credential pattern in Stage A notebook')
    baseline['cells'][0]['source'] = code.splitlines(keepends=True)
    metadata = json.loads((base.parent / 'kernel-metadata.json').read_text(encoding='utf-8'))
    metadata.update(id=KERNEL_ID, title='H3 PTD Bank v2 Stage A Two Block T4 20261102',
                    enable_gpu=True, enable_internet=True, is_private=True,
                    dataset_sources=[], kernel_sources=[], model_sources=[],
                    competition_sources=[], machine_shape='NvidiaTeslaT4')
    destination.mkdir(parents=True)
    notebook_path = destination / 'experiment.ipynb'
    notebook_path.write_text(json.dumps(baseline), encoding='utf-8')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2),
                                                     encoding='utf-8')
    binding = dict(format='h3-ptd-v2-stage-a-delivery-v2', id=KERNEL_ID,
                   baseline_notebook_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                   baseline_commit=baseline_commit, core_overlay=list(CORE_OVERLAY),
                   source_dependency_closure=list(closure),
                   source_overlay_sha256=digest, source_overlays=proofs,
                   gpu_count=2, trained_ptd_layers=2, trained_ptd_hops=8,
                   full_release_download=False, full_release_sha256_verified=False,
                   production_deployment=False, no_te_vae=True,
                   tiny_latent=True, max_worker_seconds=1200,
                   source_frozen=True, no_dataset_upload=True,
                   no_kaggle_status_polling=True)
    (destination / 'source_binding.json').write_text(json.dumps(binding, indent=2),
                                                     encoding='utf-8')
    print(json.dumps(dict(output=str(destination), id=KERNEL_ID,
                          notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                          overlay_sha256=digest, source_files=len(proofs),
                          notebook_bytes=notebook_path.stat().st_size)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.output)


if __name__ == '__main__':
    main()
