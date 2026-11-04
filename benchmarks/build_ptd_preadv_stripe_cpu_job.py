"""Freeze a CPU-only mounted PTD preadv/stripe I/O ablation into a private Kaggle script.

No GPU kernels, model generation, dataset mutation or status polling.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

SOURCE = "benchmarks/probe_ptd_preadv_stripe_cpu.py"
KERNEL_ID = "pamcks/h3-ptd-v2-preadv-stripe-cpu-20261104"
HOPS = [f"pamcks/h3-ptd-v2-hop-{hop:02d}-20261103" for hop in range(2, 8)]


def build(output):
    root = Path(__file__).resolve().parents[1]
    payload = (root / SOURCE).read_bytes()
    ast.parse(payload, filename=SOURCE)
    digest = hashlib.sha256(payload).hexdigest()
    binding = {
        "format": "h3-ptd-preadv-stripe-kaggle-cpu-v1",
        "kernel_id": KERNEL_ID,
        "source_sha256": digest,
        "cpu_only": True, "enable_gpu": False,
        "enable_internet": False, "dataset_mutations": False,
        "no_model_inference": True, "no_weight_publication": True,
        "real_ptd_hop_datasets": HOPS,
        "block_cases": 11, "block_unique_shards": 22,
        "overlay_cases": 6, "overlay_unique_hops": 6,
        "strategies": ["tensor_pread", "tensor_preadv", "stripe_preadv"],
        "workers": [1, 2, 4, 8],
        "stripe_MiB": [4, 8],
        "block_case_deadline_seconds": 110,
        "overlay_case_deadline_seconds": 165,
        "worker_measurement_budget_seconds": 1280,
        "supervisor_deadline_seconds": 1500,
        "integrity": "Full-file SHA256 compared to each published Hop receipt",
        "cache_caveat": "Unique files where possible; mounted read is not the same as remote API download.",
        "no_status_list_live_log_polling": True,
        "no_automatic_retry": True,
    }
    script = f'''import base64, hashlib, json, os, pathlib, subprocess, sys, time
OUT=pathlib.Path("/kaggle/working/h3_ptd_preadv_stripe_cpu_20261104")
OUT.mkdir(parents=True, exist_ok=True)
BINDING={binding!r}
(OUT/"source_binding.json").write_text(json.dumps(BINDING,indent=2))
if list(pathlib.Path("/dev").glob("nvidia[0-9]*")):
 raise RuntimeError("A CPU-only Kaggle notebook is required")
source=base64.b64decode({base64.b64encode(payload).decode()!r})
if hashlib.sha256(source).hexdigest()!=BINDING["source_sha256"]:
 raise RuntimeError("Frozen experiment source mismatch")
probe=OUT/"probe_ptd_preadv_stripe_cpu.py"
probe.write_bytes(source)
started=time.monotonic()
try:
 with (OUT/"worker_stdout.log").open("w") as stdout,(OUT/"worker_stderr.log").open("w") as stderr:
  run=subprocess.run([sys.executable,"-u",str(probe),"--output",str(OUT/"measurement")],
                     stdout=stdout,stderr=stderr,timeout=1500,check=False,
                     env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))
 state=dict(status="completed" if run.returncode==0 else "failed",
            exit_code=run.returncode,elapsed_seconds=time.monotonic()-started)
except subprocess.TimeoutExpired:
 state=dict(status="timeout",elapsed_seconds=time.monotonic()-started)
(OUT/"supervision.json").write_text(json.dumps(state,indent=2))
print("PTD_PREADV_STRIPE_FINAL",state,flush=True)
if state["status"]!="completed":
 raise RuntimeError("PTD CPU I/O experiment blocked/failed; partial artifacts preserved")
'''
    ast.parse(script)
    metadata = {
        "id": KERNEL_ID,
        "title": "H3 PTD V2 Preadv Stripe Mounted IO CPU 20261104",
        "code_file": "experiment.py", "language": "python",
        "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "docker_image_pinning_type": "original",
        "dataset_sources": HOPS,
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.py").write_text(script, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    binding["notebook_sha256"] = hashlib.sha256(
        (output / "experiment.py").read_bytes()).hexdigest()
    (output / "source_binding.json").write_text(
        json.dumps(binding, indent=2), encoding="utf-8")
    return binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
