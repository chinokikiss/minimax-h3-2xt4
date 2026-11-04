"""Build a private CPU-only Kaggle benchmark over two synthetic and six real PTD mounts."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

SOURCE = "benchmarks/probe_ptd_fs_mount_max_v2.py"
SYNTH = ["pamcks/h3-ptd-fsmount-ab-private-20261103",
         "pamcks/h3-ptd-fsmount-ab-public-20261103"]
REAL = [f"pamcks/h3-ptd-v2-hop-{i:02d}-20261103" for i in range(2, 8)]
KERNEL_ID = "pamcks/h3-ptd-fsmount-max-adaptive-cpu-20261103"


def build(output):
    root = Path(__file__).resolve().parents[1]
    source = (root / SOURCE).read_bytes()
    ast.parse(source)
    sha = hashlib.sha256(source).hexdigest()
    binding = {
        "format": "h3-ptd-fsmount-max-cpu-notebook-v2",
        "kernel_id": KERNEL_ID,
        "source_sha256": sha,
        "cpu_only": True, "enable_gpu": False, "enable_internet": False,
        "actual_model_code_executed": False, "dataset_mutations": False,
        "synthetic_dataset_handles": SYNTH, "real_ptd_hop_dataset_handles": REAL,
        "real_hop_rank_streams": 12,
        "real_worker_profiles": [1, 2, 4, 8, 16, 32],
        "synthetic_worker_profiles": [1],
        "real_per_case_windows": 2, "real_window_MiB": 96,
        "real_chunk_MiB": 2,
        "real_per_case_cold_candidate_MiB": 192,
        "target_MBps": 800.0,
        "no_status_list_live_log_polling": True,
        "early_stop_after_pilot_timeout": True,
        "per_case_deadlines_seconds": {"synthetic": 45, "pilot": 90, "confirm": 150},
        "outer_supervisor_deadline_seconds": 900,
        "caveat": "Mounted application source-read throughput is NOT provider download speed.",
    }
    script = f'''import base64, hashlib, json, os, pathlib, subprocess, sys, time
OUT=pathlib.Path('/kaggle/working/h3_ptd_fsmount_max_v2_20261103')
OUT.mkdir(parents=True,exist_ok=True)
BINDING={binding!r}
(OUT/'source_binding.json').write_text(json.dumps(BINDING,indent=2))
if list(pathlib.Path('/dev').glob('nvidia[0-9]*')):
 raise RuntimeError('CPU-only Notebook required')
source=base64.b64decode({base64.b64encode(source).decode()!r})
if hashlib.sha256(source).hexdigest()!=BINDING['source_sha256']:
 raise RuntimeError('Frozen source SHA256 mismatch')
probe=OUT/'probe_ptd_fs_mount_max.py'
probe.write_bytes(source)
started=time.monotonic()
try:
 with (OUT/'worker_stdout.log').open('w') as stdout, (OUT/'worker_stderr.log').open('w') as stderr:
  run=subprocess.run([sys.executable,'-u',str(probe),'--output',str(OUT/'measurement')],
   stdout=stdout,stderr=stderr,timeout=900,check=False,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 state=dict(status='completed' if run.returncode==0 else 'failed',
            exit_code=run.returncode,elapsed_seconds=time.monotonic()-started)
except subprocess.TimeoutExpired:
 state=dict(status='timeout',elapsed_seconds=time.monotonic()-started)
(OUT/'supervision.json').write_text(json.dumps(state,indent=2))
print('PTD_FSMOUNT_MAX_FINAL',state,flush=True)
if state['status']!='completed':
 raise RuntimeError('CPU mounted-source benchmark worker failed; inspect bounded partial result')
'''
    ast.parse(script)
    meta = {
        "id": KERNEL_ID,
        "title": "H3 PTD FSMount Max Adaptive CPU 20261103",
        "code_file": "experiment.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "docker_image_pinning_type": "original",
        "dataset_sources": SYNTH + REAL,
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.py").write_text(script, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    binding["script_sha256"] = hashlib.sha256((output / "experiment.py").read_bytes()).hexdigest()
    (output / "source_binding.json").write_text(json.dumps(binding, indent=2), encoding="utf-8")
    return binding


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.output), indent=2))
