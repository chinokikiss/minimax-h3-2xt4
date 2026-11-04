"""Build an immutable single-Overlay HTTP Range-to-RAM CPU Kaggle benchmark."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

SOURCES = (
    "benchmarks/probe_ptd_preadv_stripe_cpu.py",
    "benchmarks/probe_ptd_overlay_http_range_ram_cpu.py",
)
SOURCE_DATASET = "pamcks/h3-ptd-v2-hop-00-20261103"
KERNEL_ID = "pamcks/h3-ptd-overlay-range-ram-cpu-20261104"
NOTEBOOK_OUT = "/kaggle/working/h3_ptd_overlay_range_ram_cpu_20261104"


def build(output):
    root = Path(__file__).resolve().parents[1]
    capsule = {}
    hashes = {}
    for relative in SOURCES:
        raw = (root / relative).read_bytes()
        ast.parse(raw, filename=relative)
        name = Path(relative).name
        capsule[name] = base64.b64encode(raw).decode("ascii")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    binding = {
        "format": "h3-ptd-overlay-http-range-ram-cpu-delivery-v1",
        "kernel_id": KERNEL_ID,
        "source_sha256": hashes,
        "dataset": SOURCE_DATASET,
        "dataset_version": 1,
        "dataset_path": "payload/aux_overlay.safetensors",
        "expected_bytes": 1544161056,
        "cpu_only": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_mutations": False,
        "weight_publication": False,
        "http_disk_staging": False,
        "raw_object_http_range": True,
        "single_preallocated_ram_buffer": True,
        "range_workers": [1, 2, 4, 8],
        "range_trials_per_worker": 2,
        "mounted_preadv_trials": 2,
        "order": ["preadv1", "range1", "range2", "range4", "range8",
                  "range8", "range4", "range2", "range1", "preadv1"],
        "full_file_receipt_sha256_each_case": True,
        "kaggle_status_list_live_log_polling": False,
        "automatic_retry": False,
        "worker_timeout_seconds": 1300,
        "supervisor_timeout_seconds": 1400,
    }
    source = f'''import ast,base64,hashlib,json,os,pathlib,subprocess,sys,time
OUT=pathlib.Path({NOTEBOOK_OUT!r})
OUT.mkdir(parents=True,exist_ok=True)
BINDING={binding!r}
(OUT/'source_binding.json').write_text(json.dumps(BINDING,indent=2))
if list(pathlib.Path('/dev').glob('nvidia[0-9]*')):
 raise RuntimeError('This trial is CPU-only; GPU devices must be disabled')
CAPSULE={capsule!r}
HASHES={hashes!r}
for name,payload in CAPSULE.items():
 raw=base64.b64decode(payload)
 if hashlib.sha256(raw).hexdigest()!=HASHES[name]:
  raise RuntimeError('Frozen benchmark source SHA256 mismatch: '+name)
 ast.parse(raw,filename=name)
 (OUT/name).write_bytes(raw)
started=time.monotonic()
try:
 with (OUT/'worker_stdout.log').open('w') as stdout,(OUT/'worker_stderr.log').open('w') as stderr:
  proc=subprocess.run(
   [sys.executable,'-u',str(OUT/'probe_ptd_overlay_http_range_ram_cpu.py'),
    '--output',str(OUT/'measurement')],
   cwd=str(OUT),stdout=stdout,stderr=stderr,timeout=1300,check=False,
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 result_file=OUT/'measurement'/'result.json'
 measurement=json.loads(result_file.read_text()) if result_file.exists() else {{}}
 state=dict(status='complete' if proc.returncode==0 else 'failed',
            measurement_status=measurement.get('status','missing'),
            worker_exit_code=proc.returncode,elapsed_seconds=time.monotonic()-started)
except subprocess.TimeoutExpired:
 state=dict(status='timeout',elapsed_seconds=time.monotonic()-started)
(OUT/'supervision.json').write_text(json.dumps(state,indent=2))
print('H3_PTD_OVERLAY_RANGE_RAM_FINAL',state,flush=True)
if state['status']!='complete':
 raise RuntimeError('Bounded CPU Range-to-RAM experiment did not finish; partial artifacts retained')
'''
    ast.parse(source)
    metadata = {
        "id": KERNEL_ID, "title": "H3 PTD Overlay Range RAM CPU 20261104",
        "code_file": "experiment.py", "language": "python",
        "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "docker_image_pinning_type": "original",
        "dataset_sources": [SOURCE_DATASET], "kernel_sources": [],
        "competition_sources": [], "model_sources": [],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.py").write_text(source, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    binding["notebook_sha256"] = hashlib.sha256(
        (output / "experiment.py").read_bytes()).hexdigest()
    (output / "source_binding.json").write_text(
        json.dumps(binding, indent=2) + "\n", encoding="utf-8")
    return binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
