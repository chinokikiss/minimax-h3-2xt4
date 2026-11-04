"""Build immutable, private Stage C one-hop Kaggle T4 publication notebooks.

Replicates the qualified frozen setup and overlays only dependency-closed
current sources. This is a build operation, not a Kaggle submission.
"""
import argparse
import ast
import base64
import hashlib
import gzip
import io
import json
from pathlib import Path
import re
import tarfile

from benchmarks.build_ptd_v2_bake_job import (
    DEFAULT_BASE, CORE_OVERLAY as BAKE_OVERLAY,
    _baseline_commit)
from benchmarks.source_overlay_closure import overlay_files, assert_closed
from benchmarks.t4_notebook_watchdog import supervise_worker

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = BAKE_OVERLAY + (
    'benchmarks/run_ptd_v2_stage_c_publish_t4.py',
    'docs/data/h3_ptd_v2_stage_c_20261103/license_pin/LICENSE',
)
CAMPAIGN = '20261103'


def _tar_bytes_deterministic(files):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w:') as archive:
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode='wb', mtime=0, filename='') as stream:
        stream.write(raw.getvalue())
    return compressed.getvalue()


def build(base, output, *, hop=0, replace_existing=False):
    if type(hop) is not int or not 0 <= hop < 8:
        raise ValueError('Stage C hop must be 0..7')
    if type(replace_existing) is not bool:
        raise ValueError('replace_existing must be explicit bool')
    base, output = Path(base).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing Stage C Kaggle job')
    commit = _baseline_commit(base)
    files, closure = overlay_files(commit, SOURCE_FILES)
    assert_closed(commit, files, SOURCE_FILES)
    if 'patches/rank_local_int8.py' not in files:
        raise RuntimeError('Stage C frozen overlay lacks changed rank_local_int8 dependency')
    archive = {name: (ROOT / name).read_bytes() for name in files}
    file_hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in archive.items()}
    compressed = _tar_bytes_deterministic(archive)
    digest = hashlib.sha256(compressed).hexdigest()
    notebook = json.loads(base.read_text(encoding='utf-8'))
    original = ''.join(notebook['cells'][0]['source'])
    marker = "assert receipt['status']=='ready_for_device_gate'\n"
    if original.count(marker) != 1:
        raise ValueError('Frozen setup boundary differs from qualified Stage B')
    setup = original[:original.index(marker) + len(marker)]
    injection = f"""
# Stage C one-Hop source closure over immutable frozen Kaggle setup.
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
probe="import sys\\nfrom pathlib import Path\\ns=Path('/kaggle/temp/h3_light_source')\\nsys.path[:0]=[str(s),str(s/'ComfyUI'),str(s/'benchmarks')]\\nimport patches.tp2_bundle\\nimport patches.ptd.bank_v2\\nimport patches.ptd.bake_v2\\nimport patches.ptd.publish_v2\\nimport benchmarks.run_ptd_v2_stage_c_publish_t4\\n"
check=subprocess.run([sys.executable,'-c',probe],env=env,capture_output=True,text=True,timeout=90)
if check.returncode:
    raise RuntimeError('Stage C frozen-overlay cold import failed: '+check.stderr[-3000:])
print('STAGE_C_SOURCE_OVERLAY_READY',OVERLAY_SHA256,len(FILE_HASHES),flush=True)
"""
    work = f'/kaggle/working/h3_ptd_v2_stage_c_hop_{hop:02d}_{CAMPAIGN}'
    scratch = f'/kaggle/temp/h3_ptd_v2_stage_c_hop_{hop:02d}_{CAMPAIGN}'
    execution = f"""import json,os,pathlib,sys
env=dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_ptd_v2_stage_c_publish_t4.py'),
    '--source',str(source),'--hop',{str(hop)!r},
    '--output',{work!r},'--scratch',{scratch!r}]
if {replace_existing!r}: argv.append('--replace-existing')
os.execve(sys.executable,argv,env)
"""
    supervisor = supervise_worker(execution, budget=21600,
        output=f'/kaggle/working/h3_ptd_v2_stage_c_hop_{hop:02d}_supervision')
    code = setup + injection + supervisor
    code += f"""assert json.loads(pathlib.Path({(work+'/result.json')!r}).read_text())['status']=='upload_requests_accepted_remote_readback_pending'
"""
    ast.parse(code)
    if re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})', code):
        raise ValueError('Generated notebook contains a secret-like value')
    notebook['cells'][0]['source'] = code.splitlines(keepends=True)
    meta = json.loads((base.parent/'kernel-metadata.json').read_text(encoding='utf-8'))
    kernel_id = f'pamcks/h3-ptd-v2-stage-c-hop-{hop:02d}-{CAMPAIGN}'
    meta.update(id=kernel_id,title=f'H3 PTD v2 Stage C Hop {hop:02d} {CAMPAIGN}',
        enable_gpu=True,enable_internet=True,is_private=True,
        dataset_sources=[],kernel_sources=[],model_sources=[],
        competition_sources=[],machine_shape='NvidiaTeslaT4')
    output.mkdir(parents=True)
    nb=output/'experiment.ipynb'
    nb.write_text(json.dumps(notebook),encoding='utf-8')
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    binding=dict(format='h3-ptd-v2-stage-c-hop-delivery-v1',
        baseline_commit=commit, source_notebook_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        core_overlay=list(SOURCE_FILES), dependency_closure=list(closure),
        overlay_sha256=digest, overlay_files=file_hashes,
        notebook_sha256=hashlib.sha256(nb.read_bytes()).hexdigest(),
        hop=hop, kernel_id=kernel_id, dataset_ids=(
            [f'pamcks/h3-ptd-v2-shared-{CAMPAIGN}'] if hop==0 else []
        )+[f'pamcks/h3-ptd-v2-hop-{hop:02d}-{CAMPAIGN}'],
        private_only=True, publish_mode='replace_delete_old_versions' if replace_existing else 'create_new',
        publication_ack_is_not_remote_proof=True,
        physical_gpu_count=2, no_status_polling=True,
        supervisor_budget_seconds=21600)
    (output/'source_binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),kernel_id=kernel_id,hop=hop,
        overlay_files=len(file_hashes),overlay_sha256=digest,
        notebook_sha256=binding['notebook_sha256'])),flush=True)
    return binding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=DEFAULT_BASE)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hop',type=int,default=0)
    p.add_argument('--replace-existing',action='store_true',
                   help='Explicitly delete old Dataset versions when uploading a replacement')
    args=p.parse_args()
    build(args.base,args.output,hop=args.hop,replace_existing=args.replace_existing)


if __name__=='__main__':
    main()
