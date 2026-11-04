"""Cold-import the exact PTD notebook overlay against its pinned frozen source.

This is a local, no-GPU, no-network, no-Kaggle-submission delivery gate.
It reconstructs the source from git rather than importing the developer worktree.
"""
import argparse
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    'patches.tp2_bundle',
    'patches.ptd.bank_v2',
    'patches.ptd.bake_v2',
    'patches.ptd.runtime',
)


def _extract_regular(archive, destination):
    for member in archive:
        relative = Path(member.name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unqualified archive member: ' + member.name)
        if member.isdir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            raise ValueError('Unqualified archive member: ' + member.name)
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as stream:
            path.write_bytes(stream.read())


def _notebook_overlay(job):
    binding = json.loads((job / 'source_binding.json').read_text(encoding='utf-8'))
    notebook = json.loads((job / 'experiment.ipynb').read_text(encoding='utf-8'))
    script = ''.join(notebook['cells'][0]['source'])
    ast.parse(script)
    assignments = {}
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    # Both builders embed a literal base64-encoded tarball as the b64decode argument.
    field = ('DATA' if 'DATA' in assignments else
             'OVERLAY_BYTES' if 'OVERLAY_BYTES' in assignments else
             'COMPRESSED')
    encoded_call = assignments[field]
    if not (isinstance(encoded_call, ast.Call) and len(encoded_call.args) == 1):
        raise ValueError('Notebook no longer embeds a literal overlay archive')
    encoded = ast.literal_eval(encoded_call.args[0])
    blob = base64.b64decode(encoded, validate=True)
    expected_digest = binding.get('source_overlay_sha256',
                                  binding.get('reviewed_overlay_sha256',
                                              binding.get('overlay_sha256')))
    if hashlib.sha256(blob).hexdigest() != expected_digest:
        raise ValueError('Notebook archive differs from source binding')
    proofs = binding.get('source_overlays',
                         binding.get('reviewed_overlay_files', binding.get('overlay_files')))
    if not isinstance(proofs, dict) or 'patches/rank_local_int8.py' not in proofs:
        raise ValueError('Required current rank_local_int8.py missing from delivered overlay')
    return binding, blob, proofs


def qualify(job):
    binding, blob, proofs = _notebook_overlay(job)
    commit = binding['baseline_commit']
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError('Frozen git revision is not pinned')
    with tempfile.TemporaryDirectory(prefix='ptd-v2-frozen-import-') as tmp:
        source = Path(tmp) / 'source'
        source.mkdir()
        archive = subprocess.run(
            ['git', 'archive', '--format=tar', commit, 'patches', 'benchmarks',
             'experiments', 'custom_nodes'],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode='r:') as files:
            _extract_regular(files, source)
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as files:
            members = files.getmembers()
            if set(m.name for m in members) != set(proofs):
                raise ValueError('Embedded overlay file inventory mismatch')
            for member in members:
                data = files.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != proofs[member.name]:
                    raise ValueError('Embedded overlay hash mismatch: ' + member.name)
            _extract_regular(files, source)
        modules = list(MODULES)
        if 'benchmarks/run_ptd_v2_stage_c_publish_t4.py' in proofs:
            modules.append('benchmarks.run_ptd_v2_stage_c_publish_t4')
        elif 'benchmarks/run_ptd_v2_stage_a_t4.py' in proofs:
            modules.append('benchmarks.run_ptd_v2_stage_a_t4')
        elif 'benchmarks/run_ptd_v2_bake_t4.py' in proofs:
            modules.append('benchmarks.run_ptd_v2_bake_t4')
        if 'benchmarks/run_ptd_v2_stage_c_dual_hop_t4.py' in proofs:
            modules.append('benchmarks.run_ptd_v2_stage_c_dual_hop_t4')
        # Explicitly pin PYTHONPATH and cwd to the reconstructed tree.
        # A fresh interpreter cannot accidentally reuse already-imported project modules.
        import_script = (
            "import sys; "
            "import comfy.options; comfy.options.enable_args_parsing(); "
            "import comfy.cli_args; "
            + '; '.join('import ' + name for name in modules)
            + "; from patches.rank_local_int8 import check_state_schema; "
            "print('PTD_V2_FROZEN_IMPORT_PASS')"
        )
        environment = dict(os.environ)
        environment['PYTHONPATH'] = os.pathsep.join(
            [str(source), str(ROOT / 'ComfyUI'), str(source / 'benchmarks')])
        process = subprocess.run(
            [sys.executable, '-c', import_script],
            cwd=source, env=environment, text=True, capture_output=True, timeout=100)
        if process.returncode:
            raise RuntimeError('Frozen overlay import failed:\n' + process.stdout + process.stderr)
        return dict(status='passed', commit=commit, files=len(proofs),
                    imported=modules, stdout=process.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.job.resolve()), indent=2))


if __name__ == '__main__':
    main()
