"""Build frozen private Kaggle CPU Notebook for PTD Overlay fair transport A/B."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

SOURCES = (
    "benchmarks/probe_ptd_preadv_stripe_cpu.py",
    "benchmarks/probe_ptd_overlay_http4_ab.py",
)
HOPS = (2, 3, 4, 5, 6, 7, 0, 1)
DATASETS = [f"pamcks/h3-ptd-v2-hop-{hop:02d}-20261103" for hop in HOPS]
KERNEL_ID = "pamcks/h3-ptd-overlay-preadv-vs-http4-cpu-20261104"
NOTEBOOK_OUT = "/kaggle/working/h3_ptd_overlay_preadv_vs_http4_cpu_20261104"


def build(output):
    root = Path(__file__).resolve().parents[1]
    capsules, checks = {}, {}
    for relative in SOURCES:
        binary = (root / relative).read_bytes()
        ast.parse(binary, filename=relative)
        name = Path(relative).name
        capsules[name] = base64.b64encode(binary).decode("ascii")
        checks[name] = hashlib.sha256(binary).hexdigest()
    binding = {
        "format": "h3-ptd-overlay-preadv-vs-http4-cpu-job-v1",
        "kernel_id": KERNEL_ID, "frozen_sources_sha256": checks,
        "datasets": DATASETS, "enable_gpu": False, "enable_internet": True,
        "dataset_mutations": False,
        "comparison": "1 tensor_preadv sequential vs 4 kagglehub HTTP downloads",
        "bytes_per_arm": "4 x published aux_overlay.safetensors; exact validated in runtime",
        "crossover": True, "post_HTTP_RAM_read": True,
        "http_local_staging": "/kaggle/temp (owned scratch, reclaimed per case)",
        "status_list_live_log_polling": False, "automatic_retry": False,
        "outer_supervisor_timeout_seconds": 900,
        "dataset_scope": "8 immutable PTD hop overlays only; other weights not read",
    }
    source = f'''import ast, base64, hashlib, json, os, pathlib, subprocess, sys, time
OUT=pathlib.Path({NOTEBOOK_OUT!r})
OUT.mkdir(parents=True, exist_ok=True)
BINDING={binding!r}
(OUT/'source_binding.json').write_text(json.dumps(BINDING,indent=2))
if list(pathlib.Path('/dev').glob('nvidia[0-9]*')):
 raise RuntimeError('This experiment must run on a CPU-only Kaggle notebook')
FILES={capsules!r}
CHECKS={checks!r}
for name,payload in FILES.items():
 raw=base64.b64decode(payload)
 if hashlib.sha256(raw).hexdigest()!=CHECKS[name]:
  raise RuntimeError('Frozen source integrity check failed: '+name)
 ast.parse(raw,filename=name)
 (OUT/name).write_bytes(raw)
os.environ['DISABLE_KAGGLE_CACHE']='1'
started=time.monotonic()
try:
 with (OUT/'worker_stdout.log').open('w') as stdout,(OUT/'worker_stderr.log').open('w') as stderr:
  run=subprocess.run(
   [sys.executable,'-u',str(OUT/'probe_ptd_overlay_http4_ab.py'),
    '--output',str(OUT/'measurement')],
   stdout=stdout,stderr=stderr,timeout=900,check=False,
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',DISABLE_KAGGLE_CACHE='1'))
 state=dict(status='completed' if run.returncode==0 else 'failed',
            exit_code=run.returncode,elapsed_seconds=time.monotonic()-started)
except subprocess.TimeoutExpired:
 state=dict(status='timeout',elapsed_seconds=time.monotonic()-started)
(OUT/'supervision.json').write_text(json.dumps(state,indent=2))
print('H3_PTD_OVERLAY_AB_FINAL',state,flush=True)
if state['status']!='completed':
 raise RuntimeError('Bounded Overlay A/B failed; partial records preserved')
'''
    ast.parse(source)
    metadata = {
        "id": KERNEL_ID,
        "title": "H3 PTD Overlay preadv1 vs HTTP4 CPU 20261104",
        "code_file": "experiment.py", "language": "python",
        "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "docker_image_pinning_type": "original",
        "dataset_sources": DATASETS,
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
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
