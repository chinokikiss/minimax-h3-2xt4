"""Freeze and build CPU-only Kaggle HTTP-direct vs GCSFuse benchmark Notebook."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

SOURCE_FILES = (
    "benchmarks/probe_ptd_fs_mount_max_v2.py",
    "benchmarks/probe_ptd_http_direct_ab.py",
)
# The isolated synthetic sources were rejected by Kaggle at v1 push.
# Only previously verified attached PTD Hop02..07 sources are required.
SYNTH = []
REAL = [f"pamcks/h3-ptd-v2-hop-{i:02d}-20261103" for i in range(2, 8)]
KERNEL_ID = "pamcks/h3-ptd-http-direct-vs-gcsfuse-cpu-20261103"


def build(output):
    root = Path(__file__).resolve().parents[1]
    contents = {}
    sha = {}
    for name in SOURCE_FILES:
        payload = (root / name).read_bytes()
        ast.parse(payload, filename=name)
        contents[Path(name).name] = base64.b64encode(payload).decode("ascii")
        sha[Path(name).name] = hashlib.sha256(payload).hexdigest()
    binding = {
        "format": "h3-ptd-http-direct-vs-gcsfuse-cpu-notebook-v1",
        "kernel_id": KERNEL_ID,
        "frozen_sources_sha256": sha,
        "cpu_only": True, "gpu_used": False, "dataset_mutations": False,
        "enable_internet": True,
        "disable_kaggle_cache": "1",
        "minimum_kagglehub_version_for_output_dir": "0.4.3",
        "datasets": REAL,
        "real_rank_streams": 12,
        "real_http_workers": [1, 4, 8, 16],
        "HTTP_real_case_deadlines_seconds": {"1": 35, "4": 40, "8": 45, "16": 55},
        "HTTP_synthetic_case_deadline_seconds": 25,
        "mounted_case_deadline_seconds": 35,
        "measurement_budget_seconds": 320,
        "outer_supervisor_timeout_seconds": 420,
        "early_stop_after_any_real_http_timeout_or_failure": True,
        "early_stop_after_two_consecutive_below_30pct_best": True,
        "comparison_caveats": [
            "Only completed HTTP bytes divided by HTTP case walltime qualify as application download throughput.",
            "Network RX one-second windows count other interface traffic and are diagnostics, not exact per-file payload.",
            "Local file read may hit page cache; not a cold NVMe result.",
            "HTTP API may be unavailable or require authentication for private Datasets.",
            "This does not change mounting or publish/update any Dataset.",
        ],
        "optional_synthetic_mounts": False,
        "no_status_list_live_log_polling": True,
        "no_automatic_retry": True,
    }
    notebook = f'''import base64, hashlib, json, os, pathlib, subprocess, sys, time
OUT=pathlib.Path('/kaggle/working/h3_ptd_http_direct_vs_gcsfuse_cpu_20261103')
OUT.mkdir(parents=True, exist_ok=True)
BINDING={binding!r}
(OUT/'source_binding.json').write_text(json.dumps(BINDING,indent=2))
if list(pathlib.Path('/dev').glob('nvidia[0-9]*')):
 raise RuntimeError('CPU-only Notebook required')
CAPSULE={contents!r}
CHECKS={sha!r}
for name,payload in CAPSULE.items():
 raw=base64.b64decode(payload)
 if hashlib.sha256(raw).hexdigest()!=CHECKS[name]:
  raise RuntimeError('Frozen source checksum mismatch '+name)
 (OUT/name).write_bytes(raw)
os.environ['DISABLE_KAGGLE_CACHE']='1'
started=time.monotonic()
try:
 with (OUT/'worker_stdout.log').open('w') as stdout,(OUT/'worker_stderr.log').open('w') as stderr:
  run=subprocess.run([sys.executable,'-u',str(OUT/'probe_ptd_http_direct_ab.py'),
                      '--output',str(OUT/'measurement')],
                      stdout=stdout,stderr=stderr,timeout=420,check=False,
                      env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
                               DISABLE_KAGGLE_CACHE='1'))
 state=dict(status='completed' if run.returncode==0 else 'failed',
            exit_code=run.returncode,elapsed_seconds=time.monotonic()-started)
except subprocess.TimeoutExpired:
 state=dict(status='timeout',elapsed_seconds=time.monotonic()-started)
(OUT/'supervision.json').write_text(json.dumps(state,indent=2))
print('H3_PTD_HTTP_DIRECT_FINAL',state,flush=True)
if state['status']!='completed':
 raise RuntimeError('Bounded CPU HTTP A/B worker failed; partial evidence saved')
'''
    ast.parse(notebook)
    metadata = {
        "id": KERNEL_ID,
        "title": "H3 PTD HTTP Direct vs GCSFuse CPU 20261103",
        "code_file": "experiment.py",
        "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "docker_image_pinning_type": "original",
        "dataset_sources": SYNTH + REAL,
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.py").write_text(notebook, encoding="utf-8")
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
