"""Bounded GDS capability probe; no installs, model loading, or H3 imports.

Default mode only inspects the environment and never allocates a CUDA tensor.
--gpu-io explicitly requests a 4-MiB strict-GDS roundtrip on each named GPU.
Kaggle GPU execution needs the separately authorized setup-check exception.
"""
from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types


def command(argv, timeout=10, env=None, stdout_limit=24000):
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                errors="replace", timeout=timeout, env=env)
        return {"returncode": result.returncode,
                "stdout": result.stdout[:stdout_limit], "stderr": result.stderr[:4000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": type(error).__name__, "message": str(error)}


def gdscheck_path():
    found = shutil.which("gdscheck.py")
    if found:
        return found
    if sys.platform != "linux":
        return None
    for base in (Path("/usr/local/cuda"), *sorted(Path("/usr/local").glob("cuda-*"))):
        for relative in ("gds/tools/gdscheck.py", "tools/gdscheck.py"):
            candidate = base / relative
            if candidate.is_file():
                return str(candidate)
    return None


def metadata(scratch):
    result = {"platform": platform.platform(), "python": sys.executable,
              "scratch": str(scratch), "cuda_tensors_allocated": False,
              "packages": {}, "gdscheck_path": gdscheck_path()}
    for name in ("torch", "nvidia-cufile-cu12", "nvidia-cufile-cu13"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    result["nvidia_smi"] = command([
        "nvidia-smi", "--query-gpu=name,driver_version,pci.bus_id", "--format=csv,noheader"])
    if sys.platform == "linux":
        result["nvidia_fs"] = {
            "sys_module": Path("/sys/module/nvidia_fs").exists(),
            "device": Path("/dev/nvidia-fs").exists(),
            "stats": Path("/proc/driver/nvidia-fs/stats").exists()}
        result["filesystem"] = command([
            "findmnt", "--json", "--target", str(scratch), "--output", "TARGET,FSTYPE,SOURCE"])
        result["cufile_library"] = ctypes.util.find_library("cufile")
    try:
        import torch
        import torch.cuda.gds as gds
        result["torch_cuda_build"] = torch.version.cuda
        result["gds_module"] = inspect.getfile(gds)
        names = ("_gds_register_buffer", "_gds_register_handle",
                 "_gds_load_storage", "_gds_save_storage")
        # gds.py can install Python error stubs; hasattr alone gives a false positive.
        result["native_gds_bindings"] = {
            name: callable(getattr(torch._C, name, None)) and
            not inspect.isfunction(getattr(torch._C, name, None)) for name in names}
        if sys.platform == "win32":
            proxy = types.SimpleNamespace()
            try:
                # The platform guard runs before opening a file. A proxy avoids
                # the partially initialized GdsFile destructor on rejection.
                gds.GdsFile.__init__(proxy, "__gds_no_file_opened__", os.O_RDONLY)
            except Exception as error:
                result["platform_rejection"] = {
                    "type": type(error).__name__, "message": str(error)}
            result["probe_file_opened"] = hasattr(proxy, "fd")
    except Exception as error:
        result["torch_probe_error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def gpu_worker(scratch, device):
    """Called in an isolated process so its CUDA/file operations have a deadline."""
    with tempfile.TemporaryDirectory(prefix="h3_gds_probe_", dir=scratch) as folder:
        folder = Path(folder)
        config = {"properties": {"allow_compat_mode": False,
                                  "force_compat_mode": False},
                  "logging": {"dir": str(folder), "level": "TRACE"}}
        config_path = folder / "cufile.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        os.environ.update(CUFILE_ENV_PATH_JSON=str(config_path),
                          CUFILE_ALLOW_COMPAT_MODE="false", CUFILE_FORCE_COMPAT_MODE="false",
                          CUFILE_LOGGING_LEVEL="TRACE", CUFILE_LOGFILE_MAX_SIZE_MB="1",
                          CUFILE_LOGFILE_PATH=str(folder / "cufile.log"))
        result = {"device": device, "bytes": 4 * 1024 * 1024,
                  "compatibility_fallback_allowed": False, "config": config}
        checker = gdscheck_path()
        if checker:
            result["gdscheck"] = command([sys.executable, checker, "-p"], timeout=12)
        try:
            import torch
            import torch.cuda.gds as gds
            with torch.cuda.device(device), torch.inference_mode():
                source = torch.arange(result["bytes"], dtype=torch.int32,
                                      device=f"cuda:{device}").to(torch.uint8)
                destination = torch.empty_like(source)
                torch.cuda.synchronize(device)
                io_file = gds.GdsFile(str(folder / "roundtrip.bin"), os.O_CREAT | os.O_RDWR)
                try:
                    io_file.save_storage(source.untyped_storage())
                    io_file.load_storage(destination.untyped_storage())
                    torch.cuda.synchronize(device)
                    result["equal"] = bool(torch.equal(source, destination))
                finally:
                    del io_file
                result["device_name"] = torch.cuda.get_device_name(device)
            checker_text = result.get("gdscheck", {}).get("stdout", "")
            disabled = re.search(r"properties\.(?:use|allow)_compat_mode\s*:\s*false",
                                 checker_text, re.IGNORECASE) is not None
            log = folder / "cufile.log"
            log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            posix = re.search(r"cufile\s+IO\s+mode\s*:\s*POSIX", log_text, re.IGNORECASE) is not None
            result.update(compatibility_disabled_verified=disabled, posix_fallback_observed=posix,
                          cufile_log_sha256=hashlib.sha256(log_text.encode()).hexdigest(),
                          cufile_log_tail=log_text[-4000:])
            if result["equal"] and disabled and not posix:
                result["status"] = "strict_gds_roundtrip_passed"
            elif result["equal"] and not posix:
                result["status"] = "roundtrip_passed_direct_mode_unconfirmed"
            else:
                result["status"] = "direct_io_not_verified"
        except Exception as error:
            result.update(status="direct_io_failed", error=type(error).__name__, message=str(error))
            log = folder / "cufile.log"
            if log.exists():
                result["cufile_log_tail"] = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--gpu-io", action="store_true")
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--worker-device", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    scratch = args.scratch.resolve(strict=True)
    if not scratch.is_dir():
        parser.error("--scratch must be an existing directory")
    if args.worker_device is not None:
        print(json.dumps(gpu_worker(scratch, args.worker_device)))
        return
    started = time.monotonic()
    result = {"schema": 1, "mode": "gpu_io" if args.gpu_io else "metadata_only",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "package_installation": False, "h3_components_executed": False,
              "metadata": metadata(scratch), "direct_io": []}
    if sys.platform == "win32":
        result.update(status="unsupported_on_this_platform", direct_gds_supported=False)
    elif sys.platform != "linux":
        result.update(status="unsupported_by_this_probe", direct_gds_supported=None)
    elif args.gpu_io:
        for device in args.devices:
            executed = command([sys.executable, "-B", str(Path(__file__).resolve()),
                                "--scratch", str(scratch), "--worker-device", str(device)],
                               timeout=40, stdout_limit=64000)
            try:
                child = json.loads(executed.get("stdout", "").strip().splitlines()[-1])
            except (ValueError, IndexError):
                child = {"device": device, "status": "worker_failed_or_timed_out", "process": executed}
            result["direct_io"].append(child)
        passed = all(row.get("status") == "strict_gds_roundtrip_passed" for row in result["direct_io"])
        result.update(status="strict_gds_verified" if passed else "direct_gds_not_verified",
                      direct_gds_supported=True if passed else None)
    else:
        result.update(status="metadata_only_direct_io_unverified", direct_gds_supported=None)
    result["elapsed_seconds"] = time.monotonic() - started
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
