"""Fair CPU-only PTD Overlay benchmark: one tensor_preadv vs four HTTP workers.

Both arms process the exact same eight immutable 1.54GB aux_overlay
safetensors, once per transport. First-touch phase uses disjoint four-file
groups; second phase swaps transport and group. No inference or GPU.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import probe_ptd_preadv_stripe_cpu as mounted

FORMAT = "h3-ptd-v2-overlay-1-tensor-preadv-vs-4-http-cpu-v1"
HOPS = (2, 3, 4, 5, 6, 7, 0, 1)
OVERLAY_NAME = "aux_overlay.safetensors"
SCRATCH_ROOT = Path("/kaggle/temp/h3_ptd_overlay_http4_ab_20261104")
MIB = 1 << 20
DEADLINES = {"mounted": 170, "http": 205}
TOTAL_BUDGET = 780
MAX_SCRATCH_MARGIN = 1024 * MIB


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def plans():
    first = HOPS[:4]
    second = HOPS[4:]
    return [
        {"id": "phase1_mount_A", "method": "mounted", "hops": first},
        {"id": "phase1_http4_B", "method": "http", "hops": second},
        {"id": "phase2_http4_A", "method": "http", "hops": first},
        {"id": "phase2_mount_B", "method": "mounted", "hops": second},
    ]


def verify_design():
    cases = plans()
    assert [x["method"] for x in cases] == ["mounted", "http", "http", "mounted"]
    assert set(cases[0]["hops"]).isdisjoint(cases[1]["hops"])
    assert set(cases[0]["hops"]) | set(cases[1]["hops"]) == set(HOPS)
    assert cases[0]["hops"] == cases[2]["hops"]
    assert cases[1]["hops"] == cases[3]["hops"]
    assert len(HOPS) == len(set(HOPS)) == 8
    return True


def collect_sources():
    found = {}
    mounts = {}
    for hop in HOPS:
        handle = mounted.HANDLE.format(hop)
        root = mounted.locate(handle)
        receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        if receipt.get("hop") != hop:
            raise ValueError("Wrong immutable PTD Hop receipt " + handle)
        aux = receipt.get("auxiliary", {})
        if aux.get("file") != OVERLAY_NAME:
            raise ValueError("Incorrect overlay name in receipt " + handle)
        path = root / OVERLAY_NAME
        size = path.stat().st_size
        if size != aux["bytes"] or size < 1000 * MIB:
            raise ValueError("Overlay size/inventory mismatch " + handle)
        found[str(hop)] = {
            "hop": hop, "handle": handle, "dataset_path": "payload/" + OVERLAY_NAME,
            "path": str(path), "size": size,
            "expected_sha256": mounted.checksum(aux["sha256"]),
            "stem": "hop{:02d}_aux_overlay".format(hop),
        }
        mounts[str(hop)] = {
            "handle": handle, "path": str(path), "bytes": size,
            "filesystem": mounted.mount_type(root),
        }
    if len({x["size"] for x in found.values()}) != 1:
        raise ValueError("Overlays differ in length; case needs byte-balanced plan")
    return found, mounts


def checked_local_path(value, owned):
    actual = Path(value).resolve()
    owned = Path(owned).resolve()
    if os.path.commonpath((str(actual), str(owned))) != str(owned):
        raise RuntimeError("HTTP output escaped dedicated scratch directory")
    if not actual.is_file():
        raise FileNotFoundError("KaggleHub did not return an output file")
    filesystem = mounted.mount_type(actual)
    if filesystem and filesystem["filesystem"] in ("fuse.gcsfuse", "nfs", "nfs4"):
        raise RuntimeError("HTTP output is network-mounted, not local")
    return actual, filesystem


def run_mounted(case, sources, output):
    samples = []
    started = time.monotonic()
    for hop in case["hops"]:
        spec = sources[str(hop)]
        sample = mounted.measure_file(spec, "tensor_preadv", 1, 8)
        samples.append({"hop": hop, **sample})
        save(output, {"status": "running", "samples": samples,
                      "case": case, "source": "gcsfuse tensor_preadv"})
        print("OVERLAY_PREADV_VALIDATED", hop, round(sample["MBps"], 3), flush=True)
    elapsed = time.monotonic() - started
    total = sum(sample["file_bytes"] for sample in samples)
    actual_read = sum(sample["payload_bytes"] for sample in samples)
    read_seconds = sum(sample["io_seconds"] for sample in samples)
    state = {
        "status": "complete", "case": case, "method": "mounted_tensor_preadv_1",
        "files_verified": len(samples), "bytes_verified": total,
        "payload_bytes_read": actual_read, "elapsed_seconds": elapsed,
        "application_MBps": total / 1e6 / elapsed,
        "tensor_only_MBps": actual_read / 1e6 / read_seconds,
        "tensor_only_seconds": read_seconds,
        "hash_seconds": sum(sample["hash_seconds"] for sample in samples),
        "samples": samples,
        "caveat": "Wall throughput includes header parsing, allocation and checksum.",
    }
    save(output, state)
    return state


def run_http(case, sources, output, scratch):
    os.environ["DISABLE_KAGGLE_CACHE"] = "1"
    import kagglehub
    from kagglehub.config import is_kaggle_cache_disabled
    if not is_kaggle_cache_disabled():
        raise RuntimeError("KaggleHub cache-disable switch was not respected")
    if "output_dir" not in inspect.signature(kagglehub.dataset_download).parameters:
        raise RuntimeError("kagglehub.dataset_download lacks output_dir")
    if scratch.exists():
        raise FileExistsError("Refusing to reuse benchmark scratch")
    scratch.mkdir(parents=True)
    expected = sum(sources[str(h)]["size"] for h in case["hops"])
    free = shutil.disk_usage(scratch).free
    if free < expected + MAX_SCRATCH_MARGIN:
        raise RuntimeError("Insufficient local scratch for four full HTTP Overlays: free={} need={}".format(
            free, expected + MAX_SCRATCH_MARGIN))

    completed = []
    state = {"status": "running", "case": case, "method": "kagglehub_http_4",
             "expected_bytes": expected, "disable_kaggle_cache": True,
             "kagglehub_version": importlib.metadata.version("kagglehub"),
             "http_files": completed}
    save(output, state)

    def download_one(number, hop):
        entry = sources[str(hop)]
        directory = scratch / ("file_{:02d}".format(number))
        directory.mkdir()
        begun = time.monotonic()
        returned = kagglehub.dataset_download(
            entry["handle"], path=entry["dataset_path"], output_dir=str(directory))
        downloaded, fs = checked_local_path(returned, directory)
        actual_size = downloaded.stat().st_size
        if actual_size != entry["size"]:
            raise ValueError("HTTP file length mismatches published receipt for Hop" + str(hop))
        return {
            "hop": hop, "file_bytes": actual_size, "download_seconds": time.monotonic() - begun,
            "path": str(downloaded), "local_filesystem": fs,
        }

    download_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="overlay-http4") as pool:
        tasks = [pool.submit(download_one, i, hop)
                 for i, hop in enumerate(case["hops"])]
        for task in as_completed(tasks):
            row = task.result()
            completed.append(row)
            save(output, state)
            print("OVERLAY_HTTP_COMPLETED", row["hop"], round(row["download_seconds"], 3), flush=True)
    http_seconds = time.monotonic() - download_start
    total = sum(row["file_bytes"] for row in completed)
    if len(completed) != 4 or total != expected:
        raise RuntimeError("Incomplete four-file HTTP payload")
    state.update(http_seconds=http_seconds, http_MBps=total / 1e6 / http_seconds)
    save(output, state)

    local_samples = []
    ram_start = time.monotonic()
    # One final payload buffer at a time; HTTP4 download is never claimed to be RAM-direct.
    for row in sorted(completed, key=lambda item: item["hop"]):
        entry = dict(sources[str(row["hop"])], path=row["path"])
        sample = mounted.measure_file(entry, "tensor_preadv", 1, 8)
        local_samples.append({"hop": row["hop"], **sample})
        state["local_ram_samples"] = local_samples
        save(output, state)
    local_total_seconds = time.monotonic() - ram_start
    read_seconds = sum(x["io_seconds"] for x in local_samples)
    state.update(
        status="complete", files_verified=len(local_samples), bytes_verified=total,
        local_read_wall_seconds=local_total_seconds,
        local_tensor_only_seconds=read_seconds,
        local_hash_seconds=sum(x["hash_seconds"] for x in local_samples),
        disk_to_ram_MBps=sum(x["payload_bytes"] for x in local_samples) / 1e6 / read_seconds,
        disk_then_ram_total_seconds=http_seconds + local_total_seconds,
        disk_then_ram_MBps=total / 1e6 / (http_seconds + local_total_seconds),
        caveats=[
            "HTTP measured four concurrent distinct 1.54GB Overlay files, not one file split by HTTP ranges.",
            "KaggleHub output_dir downloads to instance disk, not directly to RAM.",
            "Immediate local read can hit page cache, and its SHA256 time is recorded separately.",
        ],
    )
    save(output, state)
    return state


def worker(case, source_path, result_path, scratch):
    sources = json.loads(Path(source_path).read_text())
    result_path = Path(result_path)
    try:
        if case["method"] == "mounted":
            run_mounted(case, sources, result_path)
        elif case["method"] == "http":
            run_http(case, sources, result_path, Path(scratch))
        else:
            raise ValueError("Invalid benchmark method")
    except BaseException as exc:
        current = (json.loads(result_path.read_text()) if result_path.exists() else {})
        current.update(status="failed", case=case, error_type=type(exc).__name__,
                       error=str(exc)[:1200], traceback=traceback.format_exc()[-4500:])
        save(result_path, current)
        raise


def cleanup_owned(scratch):
    scratch = Path(scratch)
    if not scratch.exists():
        return False
    parent = SCRATCH_ROOT.resolve()
    target = scratch.resolve()
    if target.parent != parent or not target.name.startswith("phase"):
        raise RuntimeError("Refusing scratch cleanup outside this benchmark run")
    shutil.rmtree(target)
    return True


def master(output):
    if not verify_design():
        raise RuntimeError("Case design self-test failed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "format": FORMAT, "status": "starting", "cpu_only": True,
        "gpu_used": False, "dataset_mutations": False, "inference": False,
        "physical_payload": "eight immutable aux_overlay.safetensors",
        "case_order": plans(), "results": [],
        "fairness": [
            "Same eight full-size overlays, each measured once by each transport.",
            "First-touch comparison uses two independent equal-byte groups of four Hops.",
            "Crossover exchanges the same groups and reads using the other transport.",
            "Mounted arm reads four full overlays serially with tensor_preadv workers=1.",
            "HTTP arm downloads four full overlays concurrently with four HTTP calls.",
            "Disk-to-RAM local reading after HTTP is measured separately and included in CPU-ready throughput.",
            "Cold cache cannot be enforced reliably on Kaggle mounted FUSE; crossover is diagnostic.",
            "No GPU, no weights altered, no production runtime change.",
        ],
        "deadlines": DEADLINES, "total_budget_seconds": TOTAL_BUDGET,
    }
    save(output / "result.json", state)
    try:
        sources, mounts = collect_sources()
        state["mounts"] = mounts
        save(output / "sources.json", sources)
        state["expected_bytes_per_case"] = sum(sources[str(h)]["size"] for h in HOPS[:4])
        if SCRATCH_ROOT.exists():
            raise FileExistsError("Dedicated benchmark scratch already exists; refusing overwrite")
        SCRATCH_ROOT.mkdir(parents=True)
        save(output / "result.json", state)
    except BaseException as exc:
        state.update(status="blocked_preflight", error=repr(exc))
        save(output / "result.json", state)
        raise

    begun = time.monotonic()
    script = Path(__file__).resolve()
    for case in plans():
        limit = DEADLINES[case["method"]]
        if time.monotonic() - begun + 4 > TOTAL_BUDGET:
            summary = {"id": case["id"], "status": "skipped_budget"}
        else:
            result_file = output / (case["id"] + ".json")
            scratch = SCRATCH_ROOT / case["id"]
            argv = [sys.executable, "-u", str(script), "--worker-case",
                    json.dumps(case), "--sources", str(output / "sources.json"),
                    "--case-output", str(result_file), "--scratch", str(scratch)]
            launched = time.monotonic()
            try:
                process = subprocess.run(argv, capture_output=True, text=True,
                                         timeout=limit, check=False)
                status = "complete" if process.returncode == 0 else "failed"
                diagnostic = (process.stdout + "\n" + process.stderr)[-2200:]
            except subprocess.TimeoutExpired as exc:
                status = "timeout"
                diagnostic = repr(exc)[-1000:]
            summary = {
                "id": case["id"], "method": case["method"], "hops": case["hops"],
                "status": status, "supervisor_wall_seconds": time.monotonic() - launched,
                "deadline_seconds": limit, "diagnostic_tail": diagnostic,
            }
            if result_file.exists():
                detail = json.loads(result_file.read_text())
                summary["case_status"] = detail.get("status")
                for key in ("application_MBps", "tensor_only_MBps", "http_MBps",
                            "disk_then_ram_MBps", "elapsed_seconds", "http_seconds",
                            "local_read_wall_seconds", "disk_then_ram_total_seconds",
                            "bytes_verified", "files_verified", "error_type", "error"):
                    if key in detail:
                        summary[key] = detail[key]
                if detail.get("status") != "complete" and status == "complete":
                    summary["status"] = "failed_incomplete_record"
            if case["method"] == "http":
                try:
                    summary["scratch_reclaimed"] = cleanup_owned(scratch)
                except Exception as exc:
                    summary["cleanup_warning"] = str(exc)[:350]
            print("OVERLAY_AB_CASE", case["id"], summary["status"], flush=True)
        state["results"].append(summary)
        save(output / "result.json", state)
    state["elapsed_seconds"] = time.monotonic() - begun
    state["status"] = ("complete" if all(x["status"] == "complete" for x in state["results"])
                       else "complete_with_unusable_cases")
    save(output / "result.json", state)
    print("OVERLAY_AB_FINISHED", state["status"], flush=True)


def selftest():
    verify_design()
    assert len({hop for case in plans() for hop in case["hops"]}) == 8
    print("H3_PTD_OVERLAY_AB_SELFTEST_PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--worker-case")
    parser.add_argument("--sources")
    parser.add_argument("--case-output")
    parser.add_argument("--scratch")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    elif args.worker_case:
        if not all((args.sources, args.case_output, args.scratch)):
            parser.error("Worker case requires sources, case-output and scratch")
        worker(json.loads(args.worker_case), args.sources, args.case_output, args.scratch)
    elif args.output:
        master(args.output)
    else:
        parser.error("Specify --self-test or --output")


if __name__ == "__main__":
    main()
