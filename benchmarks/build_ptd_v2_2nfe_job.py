"""Build one immutable private Kaggle two-NFE PTD v2 mounted-Dataset I/O trial.

Uses independent Stage C remote SHA256 proofs as preflight evidence. Builds a
small diagnostic Bank manifest from proved local receipts; it does not rehash
169.5 GB, publish a production Bank, upload, delete or version any Dataset.
The returned notebook is only a candidate until separately submitted.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from benchmarks.build_ptd_v2_bake_job import DEFAULT_BASE, CORE_OVERLAY, _baseline_commit
from benchmarks.source_overlay_closure import overlay_files, assert_closed
from benchmarks.t4_notebook_watchdog import supervise_worker
from patches.ptd.schedule import hop_sigmas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/data/h3_ptd_v2_stage_c_20261103'
BATCH = (DATA / 'gpu_final_batch_0007_v1_verified_pull'
         / 'h3_ptd_v2_stage_c_batch_01234567_20261103')
PROOFS = (DATA / 'readback_eight_hops_final_v1'
          / 'h3_ptd_v2_stage_c_eight_hop_readback_20261103')
SHARED = (DATA / 'gpu_final_hop00_v1'
          / 'h3_ptd_v2_stage_c_hop_00_20261103')
SHARED_PROOF = (DATA / 'readback_final_shared_v1'
                / 'h3_ptd_v2_readback_shared/part_readback_proof.json')
FORMAT = 'h3-ptd-v2-two-nfe-kaggle-delivery-v1'
KAGGLE_ID = 'pamcks/h3-ptd-v2-2nfe-io-v4-20261103'
SOURCE_FILES = CORE_OVERLAY + (
    'benchmarks/run_ptd_v2_2nfe_t4.py',
    'benchmarks/dit_pair_contract.py',
    'patches/ptd/runtime.py',
    'patches/ptd/sampler.py',
    'patches/comfy_tp_runtime.py',
    'patches/runtime_diagnostics.py',
    'docs/data/h3_ptd_v2_stage_c_20261103/license_pin/LICENSE',
)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _proven_bank():
    """Check remote proofs and source receipts locally, without opening private payloads."""
    original = json_load(SHARED/'shared_receipt.json')
    recipe = original['recipe']
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if (original.get('format') != 'h3-ptd-tp2-part-v2'
            or original.get('status') != 'complete'
            or recipe.get('layers') != 50 or recipe.get('hops') != 8
            or recipe.get('steps') != 32 or recipe.get('pruned_adaln_approximation') is not True
            or original['recipe_sha256'] != recipe_sha):
        raise ValueError('Original Shared Bank v2 recipe is incomplete')
    sources = {'shared': (SHARED/'shared_receipt.json', SHARED_PROOF)}
    sources.update({
        f'hop_{i:02d}': (BATCH/f'hop_{i:02d}_receipt.json',
                         PROOFS/f'hop_{i:02d}_readback_proof.json')
        for i in range(8)})
    result = json_load(PROOFS/'result.json')
    if (result.get('status') != 'passed_remote_sha256'
            or set(result.get('parts', {})) != set(sources) - {'shared'}):
        raise ValueError('Eight-Hop independent mounted SHA256 summary is not PASS')
    parts, proofs, receipts = {}, {}, {}
    for part_id, (receipt_path, proof_path) in sources.items():
        receipt = json_load(receipt_path)
        proof = json_load(proof_path)
        handle = 'pamcks/h3-ptd-v2-' + ('shared' if part_id == 'shared'
                                       else part_id.replace('_', '-')) + '-20261103'
        expected_pin = handle + '/versions/1'
        total_files = 3 if part_id == 'shared' else 202
        total_bytes = 58693088 if part_id == 'shared' else 21179555374
        if (receipt.get('format') != 'h3-ptd-tp2-part-v2'
                or receipt.get('status') != 'complete'
                or receipt.get('recipe_sha256') != recipe_sha
                or proof.get('format') != 'h3-ptd-tp2-part-readback-v1'
                or proof.get('status') != 'verified'
                or proof.get('part') != part_id
                or proof.get('recipe_sha256') != recipe_sha
                or proof.get('published_identity') != expected_pin
                or proof.get('receipt_sha256') != digest(receipt_path)
                or proof.get('source_receipt_sha256') != digest(receipt_path)
                or proof.get('payload_files') != total_files
                or proof.get('payload_bytes') != total_bytes
                or proof.get('layers') != 50
                or (part_id != 'shared' and
                    (receipt.get('hop') != int(part_id[-2:])
                     or result['parts'][part_id]['payload_tree_sha256']
                     != proof['payload_tree_sha256']))):
            raise ValueError('Independent remote proof and source receipt mismatch: ' + part_id)
        if part_id != 'shared':
            admission = json_load(BATCH/f'{part_id}_local_admission.json')
            if (admission.get('receipt_sha256') != proof['receipt_sha256']
                    or admission.get('payload_tree_sha256') != proof['payload_tree_sha256']
                    or admission.get('payload_bytes') != total_bytes):
                raise ValueError('Bake write-time admission differs from remote proof: ' + part_id)
            if len(receipt.get('shards', {})) != 100:
                raise ValueError('Full 100-rank-shard receipt required: ' + part_id)
        parts[part_id] = dict(root=part_id, identity=expected_pin)
        proofs[part_id] = dict(dataset=handle, published_identity=expected_pin,
            receipt_sha256=proof['receipt_sha256'],
            payload_tree_sha256=proof['payload_tree_sha256'],
            payload_files=total_files, payload_bytes=total_bytes)
        receipts[part_id] = receipt

    hops = []
    for hop in range(8):
        part_id = f'hop_{hop:02d}'
        record = receipts[part_id]
        shards = {stem: dict(row, part=part_id, file=stem+'.safetensors',
                             manifest=stem+'.json')
                  for stem, row in record['shards'].items()}
        hops.append(dict(hop=hop, status='complete', source_recipe_sha256=recipe_sha,
                         auxiliary=dict(record['auxiliary'], part=part_id), shards=shards))
    shared = receipts['shared']
    manifest = dict(format='h3-ptd-tp2-bank-v2', status='complete',
        online_lora=False, full_model=True, recipe=recipe, recipe_sha256=recipe_sha,
        parts=parts,
        shared=dict(part='shared',aux=shared['aux'],aux_sha256=shared['aux_sha256'],
                    aux_bytes=shared['aux_bytes'],native_header=shared['native_header'],
                    block_plan=shared['block_plan']),
        hops=hops, sigmas=hop_sigmas(8,32,12.).tolist())
    return manifest, proofs


def _gzip_bundle(payloads):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for name, data in sorted(payloads.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mtime = 0
            archive.addfile(entry, io.BytesIO(data))
    return stream.getvalue()


def build(base=DEFAULT_BASE, output=None, *, measure_throughput=False):
    base = Path(base).resolve()
    if output is None:
        raise ValueError('Explicit output path required')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Do not overwrite existing private Kaggle evidence')
    manifest, proofs = _proven_bank()
    source_commit = _baseline_commit(base)
    source_files = SOURCE_FILES + (('benchmarks/probe_ptd_mounted_io_throughput.py',)
                                    if measure_throughput else ())
    names, closure = overlay_files(source_commit, source_files)
    assert_closed(source_commit, names, source_files)
    source_map = {name:(ROOT/name).read_bytes() for name in names}
    source_hashes = {name:hashlib.sha256(data).hexdigest()
                     for name,data in source_map.items()}
    source_blob = _gzip_bundle(source_map)
    source_sha = hashlib.sha256(source_blob).hexdigest()
    # Receipts already live inside each independently SHA256-qualified mount.
    # Reconstruct the ephemeral manifest from mounted receipt metadata to stay
    # below Kaggle's 1-MB Notebook source limit.
    payloads = {
        'proofs.json': (json.dumps(proofs, sort_keys=True, separators=(',', ':'))+'\n').encode()
    }
    metadata_blob = _gzip_bundle(payloads)
    metadata_sha = hashlib.sha256(metadata_blob).hexdigest()

    notebook = json_load(base)
    original = ''.join(notebook['cells'][0]['source'])
    marker = "assert receipt['status']=='ready_for_device_gate'\n"
    if original.count(marker) != 1:
        raise ValueError('Frozen Kaggle baseline setup boundary changed')
    setup = original[:original.index(marker)+len(marker)]
    inject = f"""
import base64, hashlib, io, json, pathlib, tarfile, os, subprocess, sys, time
TRIAL_CELL_READY = time.monotonic()
SOURCE = pathlib.Path('/kaggle/temp/h3_light_source')
BIND_ROOT = pathlib.Path('/kaggle/working/ptd_v2_2nfe_io_20261103')
BIND_ROOT.mkdir(parents=True, exist_ok=False)
def unpack_locked(blob, digest, hashes, root):
    data = base64.b64decode(blob)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('Immutable PTD 2NFE source capsule SHA256 mismatch')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as arc:
        if set(arc.getnames()) != set(hashes):
            raise ValueError('Frozen capsule member manifest changed')
        for member in arc:
            if not member.isfile() or member.name.startswith('/') or '..' in pathlib.PurePosixPath(member.name).parts:
                raise ValueError('Unsafe file in frozen PTD 2NFE capsule')
            data = arc.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != hashes[member.name]:
                raise ValueError('Frozen capsule member SHA mismatch: '+member.name)
            target = root/member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
SOURCE_HASHES = {source_hashes!r}
unpack_locked({base64.b64encode(source_blob).decode('ascii')!r},
              {source_sha!r},SOURCE_HASHES,SOURCE)
META_HASHES = {dict((name,hashlib.sha256(data).hexdigest()) for name,data in payloads.items())!r}
unpack_locked({base64.b64encode(metadata_blob).decode('ascii')!r},
              {metadata_sha!r},META_HASHES,BIND_ROOT)
(BIND_ROOT/'cell_setup_metrics.json').write_text(json.dumps(dict(
    cell_setup_finished_monotonic=time.monotonic(),
    trial_cell_ready_monotonic=TRIAL_CELL_READY,
    note='Kaggle pre-kernel input attach/transfer is not timed by notebook code'),indent=2))
env = dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
probe = subprocess.run([sys.executable,'-c',
    "import sys;sys.path[:0]=['/kaggle/temp/h3_light_source','/kaggle/temp/h3_light_source/ComfyUI'];import benchmarks.run_ptd_v2_2nfe_t4;import patches.ptd.runtime"],
    env=env,capture_output=True,text=True,timeout=120)
if probe.returncode:
    raise RuntimeError('Full-50-layer PTD 2NFE cold source import failed: '+probe.stderr[-2500:])
print('PTD_2NFE_SOURCE_READY',len(SOURCE_HASHES),flush=True)
"""
    worker = """import json, os, pathlib, sys
env = dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
# Isolated opt-in for this 2-NFE I/O diagnostic; validated CUDA 12.8 wins
# whenever available. Do not propagate this to production H3 workloads.
env['H3_PTD_2NFE_ALLOW_UNVALIDATED_CUDA13'] = '1'
source = pathlib.Path('/kaggle/temp/h3_light_source')
bind = pathlib.Path('/kaggle/working/ptd_v2_2nfe_io_20261103')
argv = [sys.executable,'-u',str(source/'benchmarks/run_ptd_v2_2nfe_t4.py'),
        '--source',str(source),
        '--manifest',str(bind/'bank_v2.json'),
        '--proofs',str(bind/'proofs.json'),
        '--output',str(bind/'trial'),'--io-workers','1']
os.execve(sys.executable,argv,env)
"""
    if measure_throughput:
        # Original NFE executes first. Probe untouched Hop02..07 after the
        # worker exits to keep NFE timings comparable and avoid cache warmup.
        # Preserve NFE failure status, but still attempt the throughput probe.
        trial_and_probe = """import subprocess
trial_code = subprocess.run(argv,env=env,check=False).returncode
probe_argv = [sys.executable,'-u',str(source/'benchmarks/probe_ptd_mounted_io_throughput.py'),
              '--proofs',str(bind/'proofs.json'),
              '--output',str(bind/'mounted_io_throughput.json'),
              '--max-seconds','600']
probe_code = subprocess.run(probe_argv,env=env,check=False).returncode
raise SystemExit(trial_code or probe_code)
"""
        worker = worker.replace('os.execve(sys.executable,argv,env)\n',trial_and_probe)
        if 'os.execve(sys.executable,argv,env)' in worker:
            raise RuntimeError('PTD throughput worker still contains terminal execve')
    supervisor = supervise_worker(worker,budget=3000,
        output='/kaggle/working/ptd_v2_2nfe_io_supervision_20261103')
    code = setup + inject + supervisor
    ast.parse(code)
    if re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',code):
        raise ValueError('Source capsule leaked token-looking string')
    notebook['cells'][0]['source'] = code.splitlines(keepends=True)
    kaggle_meta = json_load(base.parent/'kernel-metadata.json')
    kaggle_meta.update(id=KAGGLE_ID,title=('H3 PTD V2 2NFE IO V4 Throughput v2 20261103'
                                                 if measure_throughput else 'H3 PTD V2 2NFE IO V4 20261103'),
        enable_gpu=True,enable_internet=True,is_private=True,
        dataset_sources=[proofs[p]['dataset'] for p in ('shared',)+(tuple(f'hop_{i:02d}' for i in range(8)))],
        machine_shape='NvidiaTeslaT4',kernel_sources=[],model_sources=[],
        competition_sources=[])
    output.mkdir(parents=True)
    nb = output/'experiment.ipynb'
    nb.write_text(json.dumps(notebook), encoding='utf-8')
    (output/'kernel-metadata.json').write_text(json.dumps(kaggle_meta,indent=2),
                                               encoding='utf-8')
    binding = dict(format=FORMAT,status='prepared_not_submitted',
        id=KAGGLE_ID,notebook_sha256=digest(nb),source_sha256=source_sha,
        metadata_sha256=metadata_sha,baseline_commit=source_commit,
        source_files=len(names),changed_dependency_closure=len(closure),
        proof_identifiers={k:v['published_identity'] for k,v in proofs.items()},
        recipe_sha256=manifest['recipe_sha256'],layer_count=50,
        dataset_sources=kaggle_meta['dataset_sources'],
        nfe=2,full_trained_grid_steps=8,io_workers_per_rank=1,
        no_full_remote_rehash=True,no_dataset_upload_or_version_change=True,
        gpu_job_submitted=False,source_readback_evidence_only=True,
        compiler_policy='prefer_12_8_else_explicit_unqualified_13_0_for_2nfe_only',
        provider_pre_kernel_attach_timing_unobservable=True,
        mounted_io_throughput_probe=measure_throughput,
        throughput_scope=('post-2nfe unique-region mounted payload reads; '
                          '1/4/8/16 reader P10/P50/P90, no provider download')
                         if measure_throughput else None)
    (output/'source_binding.json').write_text(json.dumps(binding,indent=2),
                                              encoding='utf-8')
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--measure-mounted-throughput',action='store_true')
    args = parser.parse_args()
    result = build(args.base,args.output,
                   measure_throughput=args.measure_mounted_throughput)
    print(json.dumps(dict(id=result['id'],status=result['status'],
                          sources=len(result['dataset_sources']),
                          source_files=result['source_files'],
                          notebook_sha256=result['notebook_sha256']),indent=2))


if __name__ == '__main__':
    main()
