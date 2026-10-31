"""Build the isolated Star7-derived SM75 Sol All-INT8 CUDA library."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / "native"


def source_manifest():
    rows = {}
    for p in sorted(NATIVE.rglob("*")):
        if p.is_file():
            rows[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    rows["STAR7_LICENSE"] = hashlib.sha256((ROOT / "STAR7_LICENSE").read_bytes()).hexdigest()
    return rows


def build(output: Path, nvcc=None):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    compiler = str(nvcc or shutil.which("nvcc") or "")
    if not compiler:
        raise RuntimeError("Existing nvcc is required")
    version = subprocess.check_output([compiler, "--version"], text=True, timeout=15)
    if "V12.8.93" not in version:
        raise RuntimeError("Qualified SM75 experiment requires CUDA 12.8.93 nvcc")
    library = output / "h3_sol_all_int8_sm75.so"
    toolkit = Path(compiler).resolve().parent.parent
    command = [compiler, "-shared", "-O3", "--use_fast_math", "-std=c++17",
               "-arch=sm_75", "--cudart", "static", "-Xcompiler=-fPIC,-O2",
               "-Xlinker=--exclude-libs,ALL", "-I", str(NATIVE),
               "-I", str(toolkit / "include")]
    for name in ("lib64", "lib"):
        directory = toolkit / name
        if directory.is_dir():
            command += ["-L", str(directory)]
    command += ["-o", str(library), str(NATIVE / "sla_sm75_sparse.cu")]
    started = time.monotonic()
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    log = proc.stdout.decode("utf-8", errors="replace")
    (output / "compile.log").write_text(log, encoding="utf-8")
    receipt = dict(status="passed" if proc.returncode == 0 else "failed", target="sm_75",
        compiler_version=version, argv=command, seconds=time.monotonic()-started,
        sources=source_manifest(), artifact_sha256=hashlib.sha256(library.read_bytes()).hexdigest() if proc.returncode == 0 else None,
        artifact_bytes=library.stat().st_size if proc.returncode == 0 else None,
        scope="Experimental operator build only; no production dispatch changes")
    (output / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if proc.returncode:
        raise RuntimeError("Sol All-INT8 SM75 build failed")
    return library


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--nvcc")
    a=p.parse_args();print(build(a.output,a.nvcc))
