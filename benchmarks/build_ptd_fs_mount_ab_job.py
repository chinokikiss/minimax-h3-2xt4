"""Build separate CPU-only publisher and reader Kaggle kernels for synthetic FS mount A/B.

The publisher uploads two identical synthetic 1GiB datasets (private vs public).
The reader MUST be submitted separately only after both exact resources exist.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

PROBE = "benchmarks/probe_ptd_fs_mount_ab.py"
PREFIX = "pamcks/h3-ptd-fsmount-ab"
HANDLES = [
    "pamcks/h3-ptd-fsmount-ab-private-20261103",
    "pamcks/h3-ptd-fsmount-ab-public-20261103",
]


def build(phase, destination):
    if phase not in ("publish", "read"):
        raise ValueError("Unknown phase")
    root = Path(__file__).resolve().parents[1]
    source = (root / PROBE).read_bytes()
    ast.parse(source)
    probe_sha = hashlib.sha256(source).hexdigest()
    kernel = f"{PREFIX}-{'publisher' if phase == 'publish' else 'reader'}-cpu-20261103"
    binding = {
        "format": "h3-ptd-fsmount-public-private-ab-builder-v1",
        "phase": phase, "kernel_id": kernel, "cpu_only": True, "gpu": False,
        "no_real_model_weights": True, "no_credential_values_embedded": True,
        "source_sha256": probe_sha, "dataset_sources": HANDLES if phase == "read" else [],
        "public_dataset_synthetic_only": True,
        "publisher_only_submits_create_requests": phase == "publish",
        "reader_only_uses_pre_attached_datasets": phase == "read",
        "read_cold_candidate_MiB_each": 1024 if phase == "read" else 0,
        "read_total_workers": [1, 4] if phase == "read" else [],
        "no_kaggle_status_or_list_polling": True,
    }
    output = f"/kaggle/working/h3_ptd_fs_mount_ab_{phase}_20261103"
    script = f'''import base64, hashlib, json, os, pathlib, subprocess, sys, time
BASE=pathlib.Path({output!r})
BASE.mkdir(parents=True,exist_ok=True)
BINDING={binding!r}
(BASE/'source_binding.json').write_text(json.dumps(BINDING,indent=2),encoding='utf-8')
if list(pathlib.Path('/dev').glob('nvidia[0-9]*')):
 raise RuntimeError('CPU-only experiment expected, GPU detected')
data=base64.b64decode({base64.b64encode(source).decode()!r})
if hashlib.sha256(data).hexdigest()!=BINDING['source_sha256']:
 raise RuntimeError('Frozen probe source checksum mismatch')
source=BASE/'probe_ptd_fs_mount_ab.py'
source.write_bytes(data)
cmd=[sys.executable,'-u',str(source),'--phase',{phase!r},'--output',str(BASE/'measurement')]
begin=time.monotonic()
try:
 with (BASE/'worker_stdout.log').open('w') as out, (BASE/'worker_stderr.log').open('w') as err:
  process=subprocess.run(cmd,stdout=out,stderr=err,timeout={3300 if phase == 'publish' else 1200},check=False,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 state=dict(status='completed' if process.returncode==0 else 'failed',
            exit_code=process.returncode,elapsed_seconds=time.monotonic()-begin)
except subprocess.TimeoutExpired:
 state=dict(status='timeout',elapsed_seconds=time.monotonic()-begin)
(BASE/'supervision.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
print('H3_FS_MOUNT_AB_FINAL',state,flush=True)
if state['status']!='completed':
 raise RuntimeError('FS mount A/B worker failed; inspect bounded output artifacts')
'''
    ast.parse(script)
    metadata = {
        "id": kernel,
        "title": f"H3 PTD FSMount AB {'Publisher' if phase == 'publish' else 'Reader'} CPU 20261103",
        "code_file": "experiment.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": phase == "publish",
        "docker_image_pinning_type": "original",
        "dataset_sources": binding["dataset_sources"], "kernel_sources": [],
        "competition_sources": [], "model_sources": [],
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "experiment.py").write_text(script, encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    binding["notebook_sha256"] = hashlib.sha256(script.encode()).hexdigest()
    (destination / "source_binding.json").write_text(
        json.dumps(binding, indent=2), encoding="utf-8")
    return binding


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("publish", "read"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.phase, a.output), indent=2))


if __name__ == "__main__":
    main()
