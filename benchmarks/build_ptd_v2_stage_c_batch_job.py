"""Generate (do not submit) a bounded independent-T4 Stage C production Hop batch.

The opt-in --retire-after-upload-ack trades local scratch recovery against
remote processing risk; separately mounted remote SHA256 remains mandatory.
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

from benchmarks.build_ptd_v2_bake_job import (
    DEFAULT_BASE, CORE_OVERLAY as BAKE_OVERLAY, _baseline_commit)
from benchmarks.build_ptd_v2_stage_c_dual_hop_job import _tar_bytes_deterministic
from benchmarks.source_overlay_closure import overlay_files, assert_closed
from benchmarks.t4_notebook_watchdog import supervise_worker

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = BAKE_OVERLAY + (
    'benchmarks/run_ptd_v2_stage_c_publish_t4.py',
    'benchmarks/run_ptd_v2_stage_c_dual_hop_t4.py',
    'benchmarks/run_ptd_v2_stage_c_batch_t4.py',
    'docs/data/h3_ptd_v2_stage_c_20261103/license_pin/LICENSE',
)


def build(base, output, *, hops, replace_hops=(), retire_after_upload_ack=False):
    hops = tuple(hops)
    replace_hops = tuple(replace_hops)
    if (not hops or len(hops) != len(set(hops))
            or any(type(h) is not int or not 0 <= h < 8 for h in hops)
            or not set(replace_hops).issubset(hops)):
        raise ValueError('Invalid or duplicate batch Hops / replacement Hops')
    if len(hops) > 2 and not retire_after_upload_ack:
        raise ValueError('Batch >2 requires explicit scratch reclamation policy')
    base, output = Path(base).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing Notebook delivery')
    commit = _baseline_commit(base)
    files, closure = overlay_files(commit, SOURCE_FILES)
    assert_closed(commit, files, SOURCE_FILES)
    if 'patches/rank_local_int8.py' not in files:
        raise RuntimeError('Missing frozen-commit rank_local_int8 dependency')
    archive = {name: (ROOT / name).read_bytes() for name in files}
    file_hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in archive.items()}
    compressed = _tar_bytes_deterministic(archive)
    digest = hashlib.sha256(compressed).hexdigest()
    notebook = json.loads(base.read_text(encoding='utf-8'))
    original = ''.join(notebook['cells'][0]['source'])
    marker = "assert receipt['status']=='ready_for_device_gate'\n"
    if original.count(marker) != 1:
        raise ValueError('Frozen Stage B setup marker changed')
    setup = original[:original.index(marker) + len(marker)]
    injection = f"""
import ast,base64,hashlib,io,json,pathlib,tarfile,subprocess,sys,os
SOURCE=pathlib.Path('/kaggle/temp/h3_light_source')
OVERLAY_SHA256={digest!r}
FILE_HASHES={json.dumps(file_hashes,sort_keys=True)}
COMPRESSED=base64.b64decode({base64.b64encode(compressed).decode('ascii')!r})
assert hashlib.sha256(COMPRESSED).hexdigest()==OVERLAY_SHA256
with tarfile.open(fileobj=io.BytesIO(COMPRESSED),mode='r:gz') as arc:
    assert set(arc.getnames())==set(FILE_HASHES)
    for member in arc:
        assert member.isfile() and not member.name.startswith('/') and '..' not in pathlib.PurePosixPath(member.name).parts
        data=arc.extractfile(member).read()
        assert hashlib.sha256(data).hexdigest()==FILE_HASHES[member.name]
        target=SOURCE/member.name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==FILE_HASHES[member.name]
env=dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
check=subprocess.run([sys.executable,'-c',"import sys;sys.path[:0]=['/kaggle/temp/h3_light_source','/kaggle/temp/h3_light_source/ComfyUI'];import benchmarks.run_ptd_v2_stage_c_batch_t4"],env=env,capture_output=True,text=True,timeout=90)
if check.returncode:
    raise RuntimeError('Stage C batch source overlay cold import failed: '+check.stderr[-3000:])
print('STAGE_C_BATCH_SOURCE_READY',OVERLAY_SHA256,len(FILE_HASHES),flush=True)
"""
    suffix = ''.join(str(h) for h in hops)
    work = f'/kaggle/working/h3_ptd_v2_stage_c_batch_{suffix}_20261103'
    scratch = f'/kaggle/temp/h3_ptd_v2_stage_c_batch_{suffix}_20261103'
    execution = f"""import json,os,pathlib,sys
env=dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_ptd_v2_stage_c_batch_t4.py'),
    '--source',str(source),'--output',{work!r},'--scratch',{scratch!r},
    '--hops',*map(str,{list(hops)!r})]
if {bool(replace_hops)!r}: argv.extend(['--replace-hops',*map(str,{list(replace_hops)!r})])
if {retire_after_upload_ack!r}: argv.append('--retire-after-upload-ack')
os.execve(sys.executable,argv,env)
"""
    supervisor = supervise_worker(execution, budget=21600,
        output=f'/kaggle/working/h3_ptd_v2_stage_c_batch_{suffix}_supervision')
    code = setup + injection + supervisor
    code += f"assert json.loads(pathlib.Path({(work+'/result.json')!r}).read_text())['status']=='upload_requests_accepted_remote_readback_pending'\n"
    ast.parse(code)
    if re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})', code):
        raise ValueError('Generated source contains a secret-like pattern')
    notebook['cells'][0]['source'] = code.splitlines(keepends=True)
    metadata = json.loads((base.parent/'kernel-metadata.json').read_text(encoding='utf-8'))
    kernel_id = f'pamcks/h3-ptd-v2-stage-c-batch-{suffix}-20261103'
    metadata.update(id=kernel_id,title=f'H3 PTD v2 Stage C Batch {suffix} 20261103',
        enable_gpu=True,enable_internet=True,is_private=True,
        dataset_sources=[],kernel_sources=[],model_sources=[],
        competition_sources=[],machine_shape='NvidiaTeslaT4')
    output.mkdir(parents=True)
    nb = output / 'experiment.ipynb'
    nb.write_text(json.dumps(notebook), encoding='utf-8')
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    binding = dict(format='h3-ptd-v2-stage-c-bounded-delivery-v1',
        baseline_commit=commit, source_notebook_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        overlay_sha256=digest, overlay_files=file_hashes,
        dependency_closure=list(closure),
        notebook_sha256=hashlib.sha256(nb.read_bytes()).hexdigest(),
        hops=list(hops), replace_hops=list(replace_hops),
        retire_after_upload_ack=retire_after_upload_ack,
        remote_readback='pending',
        kernel_id=kernel_id, private_only=True, no_status_polling=True,
        dataset_ids=[f'pamcks/h3-ptd-v2-hop-{h:02d}-20261103' for h in hops])
    (output/'source_binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hops', nargs='+', type=int, required=True)
    parser.add_argument('--replace-hops', nargs='*', type=int, default=[])
    parser.add_argument('--retire-after-upload-ack', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output, hops=args.hops,
                           replace_hops=args.replace_hops,
                           retire_after_upload_ack=args.retire_after_upload_ack),
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
