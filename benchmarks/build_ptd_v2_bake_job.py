"""Build one private physical-T4 PTD Bank v2 single-hop GPU job.

Uses the already-proven frozen Kaggle setup notebook, then source-bound,
reviewed v2 overlays. Never changes the baseline notebook or uploads itself.
"""
import argparse
import ast
import base64
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CORE_OVERLAY = (
    'patches/tp2_bundle.py',
    'patches/ptd/__init__.py',
    'patches/ptd/bank_v2.py',
    'patches/ptd/bake_v2.py',
    'patches/ptd/composite_lora.py',
    'patches/ptd/publish_v2.py',
    'patches/ptd/checkpoint.py',
    'patches/ptd/mapping.py',
    'patches/ptd/reference_sources.json',
    'benchmarks/preflight_ptd_v2_bake.py',
    'benchmarks/run_ptd_v2_bake_t4.py',
    'benchmarks/tp2_public_dit_assets.py',
    'benchmarks/checkpoint_ranges.py',
)
DEFAULT_BASE = ROOT / ('docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102_v2'
                       '/gpu_job/experiment.ipynb')


def _baseline_commit(base):
    binding_path = Path(base).resolve().parent / 'source_binding.json'
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    commit = binding.get('baseline_commit')
    if commit is None:
        commit = binding.get('baseline_source_binding', {}).get('publication', {}).get('commit')
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError('Frozen PTD bake baseline has no immutable publication commit')
    return commit


def _closed_overlay(base):
    try:
        from benchmarks.source_overlay_closure import overlay_files, assert_closed
    except ImportError:
        from source_overlay_closure import overlay_files, assert_closed
    commit = _baseline_commit(base)
    files, closure = overlay_files(commit, CORE_OVERLAY)
    assert_closed(commit, files, CORE_OVERLAY)
    if 'patches/rank_local_int8.py' not in files:
        raise RuntimeError('PTD bake source closure omitted rank_local_int8 API dependency')
    return commit, files, closure


def _tar_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def build(base, output, *, id='pamcks/h3-ptd-v2-hop0-bake-t4-20261102'):
    base = Path(base).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite PTD GPU job')
    baseline_commit, overlay, closure = _closed_overlay(base)
    archive = {name: (ROOT / name).read_bytes() for name in overlay}
    # Changes outside this exact file set remain those of the frozen baseline.
    sha = {name: hashlib.sha256(payload).hexdigest()
           for name, payload in sorted(archive.items())}
    blob = _tar_bytes(archive)
    digest = hashlib.sha256(blob).hexdigest()
    notebook = json.loads(base.read_text(encoding='utf-8'))
    original = ''.join(notebook['cells'][0]['source'])
    end = "assert receipt['status']=='ready_for_device_gate'\n"
    if original.count(end) != 1:
        raise ValueError('Previously qualified notebook setup boundary changed')
    setup = original[:original.index(end) + len(end)]
    if 'bundle_assets.json' not in setup or 'checkpoint_ranges.py' not in setup:
        raise ValueError('Baseline lacks immutable full Native bundle source metadata')
    inject = f"""
# Reviewed PTD v2 source overlay, applied after unchanged frozen CPU setup.
import base64,hashlib,io,json,pathlib,tarfile
SOURCE=pathlib.Path('/kaggle/temp/h3_light_source')
OVERLAY_SHA256={digest!r}
OVERLAY_FILE_SHA256={json.dumps(sha, sort_keys=True)}
OVERLAY_BYTES=base64.b64decode({base64.b64encode(blob).decode('ascii')!r})
assert hashlib.sha256(OVERLAY_BYTES).hexdigest()==OVERLAY_SHA256
with tarfile.open(fileobj=io.BytesIO(OVERLAY_BYTES),mode='r:gz') as arc:
    assert set(arc.getnames())==set(OVERLAY_FILE_SHA256)
    for member in arc:
        assert member.isfile() and not member.name.startswith('/') and '..' not in pathlib.PurePosixPath(member.name).parts
        payload=arc.extractfile(member).read()
        assert hashlib.sha256(payload).hexdigest()==OVERLAY_FILE_SHA256[member.name]
        path=SOURCE/member.name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(payload)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==OVERLAY_FILE_SHA256[member.name]
print('PTD_V2_REVIEWED_OVERLAY_READY',OVERLAY_SHA256,len(OVERLAY_FILE_SHA256),flush=True)
# Fresh-process import rehearsal against the exact frozen source + overlay.
import os,subprocess,sys
probe_env=dict(os.environ)
probe_env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
probe_code="import sys\\nfrom pathlib import Path\\nsource=Path('/kaggle/temp/h3_light_source')\\nsys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]\\nimport patches.tp2_bundle\\nimport patches.ptd.bank_v2\\nimport patches.ptd.bake_v2\\nimport patches.ptd.checkpoint\\nimport benchmarks.run_ptd_v2_bake_t4\\n"
probe=subprocess.run([sys.executable,'-c',probe_code],env=probe_env,capture_output=True,text=True,timeout=90)
if probe.returncode:
    raise RuntimeError('PTD v2 Hop0 source import gate failed: '+probe.stderr[-6000:])
print('PTD_V2_HOP0_SOURCE_IMPORT_GATE_PASS',flush=True)
"""
    from t4_notebook_watchdog import supervise_worker
    execution = """import json,os,pathlib,sys
env=dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_ptd_v2_bake_t4.py'),
      '--source',str(source),'--output','/kaggle/working/ptd_v2_hop0_bake_t4_20261102']
os.execve(sys.executable,argv,env)
"""
    supervisor = supervise_worker(execution, budget=9000,
                                  output='/kaggle/working/ptd_v2_hop0_supervision')
    code = setup + inject + supervisor
    code += """assert json.loads(pathlib.Path('/kaggle/working/ptd_v2_hop0_bake_t4_20261102/result.json').read_text())['status']=='complete'
"""
    ast.parse(code)
    if re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})', code):
        raise ValueError('Secret-pattern in generated notebook')
    notebook['cells'][0]['source'] = code.splitlines(keepends=True)
    meta = json.loads((base.parent / 'kernel-metadata.json').read_text(encoding='utf-8'))
    meta.update(id=id, title='H3 PTD v2 Hop0 Bake T4 20261102',
                enable_gpu=True, enable_internet=True, is_private=True,
                dataset_sources=[], kernel_sources=[], model_sources=[],
                competition_sources=[], machine_shape='NvidiaTeslaT4')
    output.mkdir(parents=True)
    nb_path = output / 'experiment.ipynb'
    nb_path.write_text(json.dumps(notebook), encoding='utf-8')
    (output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2),
                                                 encoding='utf-8')
    binding = dict(format='h3-ptd-v2-hop0-kaggle-delivery-v2',
                   scope='physical dual-T4 full 50-layer one-hop source qualification',
                   original_notebook_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                   original_notebook=str(base), baseline_commit=baseline_commit,
                   core_overlay=list(CORE_OVERLAY), source_dependency_closure=list(closure),
                   immutable_original_setup=True, reviewed_overlay_sha256=digest,
                   reviewed_overlay_files=sha,
                   output_is_dataset=False, published_part=False,
                   physical_devices=2, ptd_hops_baked=1, adapted_layers=50,
                   gpu_supervision_budget_seconds=9000, data_sources=[],
                   kaggle_id=id, no_status_polling=True)
    (output / 'source_binding.json').write_text(json.dumps(binding, indent=2),
                                                encoding='utf-8')
    print(json.dumps(dict(output=str(output), id=id, notebook_sha256=hashlib.sha256(
        nb_path.read_bytes()).hexdigest(), overlay_sha256=digest,
        overlay_files=len(sha), notebook_bytes=nb_path.stat().st_size)), flush=True)
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.output)


if __name__ == '__main__':
    main()
