"""Build a private CPU-only, six-Hop PTD mounted-I/O validation Notebook.

No Dataset creation/update, GPU, model imports or network installation.
The separate submission is operator-controlled; this builder does not submit.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "benchmarks/probe_ptd_dataset_io_validation.py"
FORMAT = "h3-ptd-mounted-io-validation-delivery-v1"
KERNEL_ID = "pamcks/h3-ptd-mounted-io-validation-cpu-20261103"
DATASETS = [f"pamcks/h3-ptd-v2-hop-{hop:02d}-20261103"
            for hop in range(2, 8)]


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve all prior Kaggle job candidates and receipts")
    source = PROBE.read_bytes()
    ast.parse(source)
    digest = hashlib.sha256(source).hexdigest()
    encoded = base64.b64encode(source).decode("ascii")
    code = f"""
import base64, hashlib, json, os, pathlib, subprocess, sys, time
OUT = pathlib.Path('/kaggle/working/h3_ptd_io_validation_cpu_20261103')
OUT.mkdir(parents=True, exist_ok=False)
SOURCE = base64.b64decode({encoded!r})
EXPECTED_SHA256 = {digest!r}
if hashlib.sha256(SOURCE).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError('PTD IO validation source capsule SHA256 mismatch')
if list(pathlib.Path('/dev').glob('nvidia[0-9]*')):
    raise RuntimeError('CPU-only mounted-I/O job must not have an allocated GPU')
probe = OUT / 'probe.py'
probe.write_bytes(SOURCE)
(OUT/'source_identity.json').write_text(json.dumps(dict(
    source_sha256=EXPECTED_SHA256, scope='six immutable PTD Hop mounts, CPU-only',
    dataset_sources={DATASETS!r}, no_dataset_upload=True,
    no_gpu=True, no_full_payload_hash=True), indent=2))
started = time.monotonic()
try:
    process = subprocess.run(
        [sys.executable, '-u', str(probe), '--output', str(OUT/'measurement')],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
        capture_output=True, text=True, timeout=1320, check=False)
    (OUT/'worker_stdout.log').write_text(process.stdout[-32000:], encoding='utf-8')
    (OUT/'worker_stderr.log').write_text(process.stderr[-12000:], encoding='utf-8')
    state = dict(status='completed' if process.returncode == 0 else 'failed',
                 exit_code=process.returncode, elapsed_seconds=time.monotonic()-started)
except subprocess.TimeoutExpired as exc:
    state = dict(status='timeout', elapsed_seconds=time.monotonic()-started,
                 error_type=type(exc).__name__)
(OUT/'supervision.json').write_text(json.dumps(state, indent=2))
print('PTD_IO_VALIDATION_FINAL', state, flush=True)
if state['status'] != 'completed':
    raise RuntimeError('PTD mounted-I/O worker did not complete; inspect persisted partial measurements')
"""
    ast.parse(code)
    notebook = dict(
        nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name="Python 3", language="python",
                                      name="python3")),
        cells=[dict(cell_type="code", execution_count=None, metadata={},
                    outputs=[], source=code.splitlines(keepends=True))],
    )
    metadata = dict(
        id=KERNEL_ID, title="H3 PTD Mounted IO Validation CPU 20261103",
        code_file="experiment.ipynb", language="python", kernel_type="notebook",
        is_private=True, enable_gpu=False, enable_tpu=False,
        enable_internet=False, docker_image_pinning_type="original",
        dataset_sources=DATASETS, kernel_sources=[],
        competition_sources=[], model_sources=[],
    )
    output.mkdir(parents=True)
    notebook_path = output / "experiment.ipynb"
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2),
                                                  encoding="utf-8")
    binding = dict(
        format=FORMAT, status="prepared_not_submitted",
        kernel_id=KERNEL_ID, notebook_sha256=hashlib.sha256(
            notebook_path.read_bytes()).hexdigest(),
        probe_sha256=digest, dataset_sources=DATASETS,
        cpu_only=True, gpu=False, no_download_or_upload=True,
        model_execution=False, test_schedule="1,4,8,16,16,8,4,1",
        main_window_MiB=192, main_windows_per_case=4,
        balanced_hops=list(range(2, 8)),
        compare_modes=["balanced_8m", "balanced_1m", "hop02_8m",
                       "repeat_region_8m"],
        case_timeout_seconds=115, overall_measurement_budget_seconds=1200,
        output_relative="h3_ptd_io_validation_cpu_20261103/measurement/result.json",
        caveat="Mounted file read rate, not provider-side Dataset download",
    )
    (output / "source_binding.json").write_text(json.dumps(binding, indent=2),
                                                 encoding="utf-8")
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
