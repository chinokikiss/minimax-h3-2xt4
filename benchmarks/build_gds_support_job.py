"""Prepare a private, offline, bounded GDS check. Does not submit any job."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

from benchmarks.t4_notebook_watchdog import supervise_worker


def build(output):
    root = Path(__file__).resolve().parents[1]
    raw = (root / "benchmarks/probe_gds_support.py").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(raw).decode()
    worker = '''import base64,hashlib,pathlib,subprocess,sys
OUT=pathlib.Path('/kaggle/working/gds_support');OUT.mkdir(parents=True,exist_ok=True)
source=base64.b64decode(ENCODED)
assert hashlib.sha256(source).hexdigest()==DIGEST
probe=OUT/'probe_gds_support.py';probe.write_bytes(source)
result=subprocess.run([sys.executable,'-B',str(probe),'--gpu-io','--devices','0','1',
                       '--scratch','/tmp','--output',str(OUT/'result.json')],
                      stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=110)
(OUT/'probe.stdout.txt').write_text(result.stdout)
(OUT/'probe.stderr.txt').write_text(result.stderr)
assert result.returncode==0,'Probe process failed; inspect saved diagnostics'
'''.replace("ENCODED", repr(encoded)).replace("DIGEST", repr(digest))
    script = supervise_worker(worker, budget=135, output="/kaggle/working/gds_supervision")
    ast.parse(script)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(id="pamcks/h3-gds-support-check-2026-10-27",
                    title="H3 GDS support check 2026-10-27",
                    code_file="experiment.py", language="python", kernel_type="script",
                    is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
                    machine_shape="NvidiaTeslaT4", docker_image_pinning_type="original",
                    dataset_sources=[], kernel_sources=[], competition_sources=[], model_sources=[])
    plan = dict(status="prepared_not_submitted", gpu_exception_authorized=False,
                purpose="Verify direct GDS read/write support on Kaggle scratch, not throughput",
                probe_sha256=digest, script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                internal_budget_seconds=135, per_device_deadline_seconds=40,
                devices=[0, 1], payload_bytes_per_device=4 * 1024 * 1024,
                scratch="/tmp", no_packages_installed=True,
                no_h3_imports_or_models=True, no_credentials=True, no_network=True,
                fallback_allowed=False, status_or_live_log_polling=False,
                final_collection="Once after the bounded batch; no automatic resubmission",
                source_scope="Standalone environment probe; frozen H3 source and core pins untouched")
    for name, value in (("kernel-metadata.json", metadata), ("probe_plan.json", plan)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    (output / "experiment.py").write_text(script, encoding="utf-8")
    print(json.dumps({"prepared": str(output.resolve()), **plan}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
