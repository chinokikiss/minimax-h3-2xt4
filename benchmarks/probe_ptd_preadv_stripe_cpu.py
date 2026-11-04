"""Bounded CPU-only safetensors preadv/striped-read experiment on mounted PTD Bank v2.

Read actual independent block shards and full per-Hop auxiliary overlays. Each
read reconstructs exactly one safetensors file in a single final payload buffer,
then validates its SHA256 against the immutable published receipt. This does
not run H3 inference or infer CUDA/TP2 performance from mounted-source I/O.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
try:
    import resource
except ImportError:  # Windows can still parse plans and build frozen notebooks.
    resource = None
import struct
import subprocess
import sys
import tempfile
import time

MIB = 1 << 20
HANDLE = "pamcks/h3-ptd-v2-hop-{:02d}-20261103"
HOPS = tuple(range(2, 8))
FORMAT = "h3-ptd-preadv-stripe-mounted-cpu-v1"
SHA = re.compile(r"^[0-9a-f]{64}$")
BLOCK_PROFILES = (
    ("tensor_pread", 1, 8), ("tensor_pread", 4, 8),
    ("tensor_pread", 8, 8), ("tensor_preadv", 1, 8),
    ("tensor_preadv", 4, 8), ("tensor_preadv", 8, 8),
    ("stripe_preadv", 2, 8), ("stripe_preadv", 4, 8),
    ("stripe_preadv", 8, 8), ("stripe_preadv", 4, 4),
    ("stripe_preadv", 8, 4),
)
# Six distinct, independently published Hop overlays; no repeated remote file.
OVERLAY_PROFILES = (
    ("tensor_pread", 1, 8), ("tensor_preadv", 1, 8),
    ("stripe_preadv", 2, 8), ("stripe_preadv", 4, 8),
    ("stripe_preadv", 8, 8), ("stripe_preadv", 8, 4),
)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def checksum(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("Invalid immutable SHA256 in published PTD receipt")
    return value


def locate(handle):
    owner, slug = handle.split("/")
    base = Path("/kaggle/input")
    candidates = (base / slug, base / "datasets" / owner / slug,
                  base / "datasets" / slug, base / owner / slug)
    roots = [candidate for parent in candidates
             for candidate in (parent, parent / "payload")
             if (candidate / "receipt.json").is_file()]
    if len(roots) != 1:
        raise FileNotFoundError(
            "Expected exactly one mounted PTD receipt for " + handle +
            "; found " + str(len(roots)))
    return roots[0]


def mount_type(path):
    target = str(path.resolve())
    best = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        lhs, sep, rhs = line.partition(" - ")
        if not sep:
            continue
        fields = lhs.split()
        if len(fields) < 5:
            continue
        point = fields[4].replace("\\040", " ")
        if target != point and not target.startswith(point.rstrip("/") + "/"):
            continue
        attrs = rhs.split()
        if len(attrs) < 2:
            continue
        if best is None or len(point) > len(best["mountpoint"]):
            best = {"mountpoint": point, "filesystem": attrs[0],
                    "source": attrs[1]}
    return best


def sources():
    found = {}
    mounts = {}
    for hop in HOPS:
        handle = HANDLE.format(hop)
        root = locate(handle)
        receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        if receipt.get("hop") != hop or len(receipt.get("shards", {})) != 100:
            raise ValueError("Incomplete immutable PTD Hop receipt " + handle)
        aux = receipt["auxiliary"]
        if aux["file"] != "aux_overlay.safetensors":
            raise ValueError("Unexpected PTD auxiliary name")
        overlay = root / aux["file"]
        if overlay.stat().st_size != aux["bytes"]:
            raise ValueError("Incorrect overlay byte size")
        blocks = {}
        for layer in range(22 if hop == 2 else 0):
            stem = "dit.block{:02d}.rank0".format(layer)
            entry = receipt["shards"][stem]
            file = root / (stem + ".safetensors")
            if file.stat().st_size != entry["bytes"]:
                raise ValueError("Incorrect rank shard size " + stem)
            blocks[layer] = {
                "path": str(file), "expected_sha256": checksum(entry["file_sha256"]),
                "size": entry["bytes"], "stem": stem,
            }
        found[hop] = {"overlay": {"path": str(overlay),
                      "expected_sha256": checksum(aux["sha256"]),
                      "size": aux["bytes"], "stem": "aux_overlay"},
                      "blocks": blocks}
        mounts[str(hop)] = {"handle": handle, "root": str(root),
                            "filesystem": mount_type(root),
                            "overlay_bytes": aux["bytes"],
                            "block_count": len(blocks)}
    return found, mounts


def safetensors_plan(path):
    path = Path(path)
    stat = path.stat()
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Short safetensors prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length < 10 * MIB:
            raise ValueError("Unsafe safetensors header length")
        header_bytes = stream.read(length)
        if len(header_bytes) != length:
            raise ValueError("Incomplete safetensors header")
    meta = json.loads(header_bytes)
    ranges = []
    for name, item in meta.items():
        if name == "__metadata__":
            continue
        start, end = item["data_offsets"]
        if not 0 <= start < end <= stat.st_size - (8 + length):
            raise ValueError("Invalid safetensors tensor range")
        ranges.append((start, end, name))
    ranges.sort()
    cursor = 0
    for start, end, _ in ranges:
        if start != cursor:
            raise ValueError("Overlapping or non-covering tensor ranges")
        cursor = end
    payload_bytes = stat.st_size - (8 + length)
    if cursor != payload_bytes or not ranges:
        raise ValueError("Safetensors payload coverage mismatch")
    return prefix + header_bytes, payload_bytes, ranges, (stat.st_size, stat.st_mtime_ns)


def tasks_for(payload_bytes, ranges, mode, stripe_mib):
    if mode.startswith("tensor_"):
        return [(start, end - start) for start, end, _ in ranges]
    if mode == "stripe_preadv":
        step = stripe_mib * MIB
        return [(start, min(step, payload_bytes - start))
                for start in range(0, payload_bytes, step)]
    raise ValueError("Unknown experiment strategy")


def transfer(fd, destination, file_offset, offset, size, mode):
    end = offset + size
    chunk = 8 * MIB
    total = 0
    while offset < end:
        extent = min(chunk, end - offset)
        if mode.endswith("preadv"):
            if not hasattr(os, "preadv"):
                raise RuntimeError("os.preadv is unavailable; Linux required")
            got = os.preadv(fd, [destination[offset:offset + extent]],
                            file_offset + offset)
        else:
            if not hasattr(os, "pread"):
                raise RuntimeError("os.pread is unavailable; Linux required")
            data = os.pread(fd, extent, file_offset + offset)
            got = len(data)
            destination[offset:offset + got] = data
        if got <= 0:
            raise IOError("Unexpected EOF in positional checkpoint read")
        total += got
        offset += got
    return total


def memory_snapshot():
    result = {}
    try:
        rows = Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
        values = {key: int(value) for key, value in
                  (line.split() for line in rows)}
        for name in ("anon", "file", "workingset_refault_file",
                     "pgscan", "pgmajfault"):
            if name in values:
                result[name] = values[name]
    except (OSError, ValueError):
        pass
    return result


def measure_file(file, mode, workers, stripe_mib):
    path = Path(file["path"])
    if path.stat().st_size != file["size"]:
        raise RuntimeError("Immutable PTD source size mismatch")
    header, payload_bytes, ranges, signature = safetensors_plan(path)
    destination = bytearray(payload_bytes)
    view = memoryview(destination)
    chunks = tasks_for(payload_bytes, ranges, mode, stripe_mib)
    before = memory_snapshot()
    started = time.perf_counter()
    fd = os.open(path, os.O_RDONLY)
    try:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="ptd-stripe") as pool:
            futures = [
                pool.submit(transfer, fd, view, len(header), offset, size, mode)
                for offset, size in chunks
            ]
            counts = [future.result() for future in futures]
    finally:
        os.close(fd)
    io_seconds = time.perf_counter() - started
    after = memory_snapshot()
    if sum(counts) != payload_bytes:
        raise IOError("Partial checkpoint read")
    sha_started = time.perf_counter()
    digest = hashlib.sha256(header)
    digest.update(view)
    actual = digest.hexdigest()
    hash_seconds = time.perf_counter() - sha_started
    if actual != file["expected_sha256"]:
        raise ValueError("Mounted checkpoint full-file SHA256 mismatch: " + file["stem"])
    if (path.stat().st_size, path.stat().st_mtime_ns) != signature:
        raise RuntimeError("Immutable PTD checkpoint mutated during I/O")
    maxrss_kib = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                  if resource is not None else None)
    view.release()
    return {"stem": file["stem"], "file_bytes": file["size"],
            "payload_bytes": payload_bytes, "task_count": len(chunks),
            "tensor_count": len(ranges), "io_seconds": io_seconds,
            "MBps": payload_bytes / 1e6 / io_seconds,
            "MiBps": payload_bytes / MIB / io_seconds,
            "hash_seconds": hash_seconds, "full_sha256_valid": True,
            "peak_rss_KiB": maxrss_kib,
            "cgroup_delta": {key: after[key] - before[key]
                             for key in before.keys() & after.keys()}}


def run_case(case, output):
    result = {"id": case["id"], "kind": case["kind"],
              "strategy": case["strategy"], "workers": case["workers"],
              "stripe_MiB": case["stripe_MiB"], "status": "running",
              "samples": []}
    save(output, result)
    for file in case["files"]:
        sample = measure_file(file, case["strategy"], case["workers"],
                              case["stripe_MiB"])
        result["samples"].append(sample)
        save(output, result)
        print("PTD_STRIPE_CASE_SAMPLE", case["id"], file["stem"],
              round(sample["MBps"], 3), flush=True)
    samples = result["samples"]
    total_bytes = sum(sample["payload_bytes"] for sample in samples)
    total_seconds = sum(sample["io_seconds"] for sample in samples)
    result.update(status="passed", validated_files=len(samples),
                  validated_bytes=total_bytes,
                  weighted_MBps=total_bytes / 1e6 / total_seconds,
                  weighted_MiBps=total_bytes / MIB / total_seconds)
    save(output, result)


def case_plans(found):
    result = []
    for index, (mode, workers, stripe) in enumerate(BLOCK_PROFILES):
        result.append({
            "id": "block_{:02d}_{}_w{}_s{}".format(index, mode, workers, stripe),
            "kind": "block", "strategy": mode, "workers": workers,
            "stripe_MiB": stripe,
            # Each profile has two disjoint ~187MiB rank shards from Hop02.
            "files": [found[2]["blocks"][2 * index],
                      found[2]["blocks"][2 * index + 1]],
            "timeout_seconds": 110,
        })
    for index, (mode, workers, stripe) in enumerate(OVERLAY_PROFILES):
        hop = HOPS[index]
        result.append({
            "id": "overlay_{:02d}_hop{}_{}_w{}_s{}".format(
                index, hop, mode, workers, stripe),
            "kind": "overlay", "strategy": mode, "workers": workers,
            "stripe_MiB": stripe, "files": [found[hop]["overlay"]],
            "timeout_seconds": 165,
        })
    return result


def synthetic_selftest():
    if not hasattr(os, "pread") or not hasattr(os, "preadv"):
        raise RuntimeError("Linux positional I/O APIs are required")
    with tempfile.TemporaryDirectory(prefix="ptd_stripe_smoke_") as tmp:
        file = Path(tmp) / "fixture.safetensors"
        values = (bytes(range(256)) * 32000, bytes(range(128)) * 19000,
                  bytes(range(17)) * 190)
        offsets = (0, len(values[0]), len(values[0]) + len(values[1]))
        fields = {
            "weight0": {"dtype": "U8", "shape": [len(values[0])],
                        "data_offsets": [offsets[0], offsets[1]]},
            "weight1": {"dtype": "U8", "shape": [len(values[1])],
                        "data_offsets": [offsets[1], offsets[2]]},
            "weight2": {"dtype": "U8", "shape": [len(values[2])],
                        "data_offsets": [offsets[2], offsets[2] + len(values[2])]},
        }
        header = json.dumps(fields, separators=(",", ":")).encode()
        file.write_bytes(struct.pack("<Q", len(header)) + header + b"".join(values))
        expected = hashlib.sha256(file.read_bytes()).hexdigest()
        info = {"path": str(file), "size": file.stat().st_size,
                "expected_sha256": expected, "stem": "fixture"}
        for mode, workers, stripe in (
                ("tensor_pread", 1, 8), ("tensor_pread", 4, 8),
                ("tensor_preadv", 1, 8), ("tensor_preadv", 4, 8),
                ("stripe_preadv", 2, 4), ("stripe_preadv", 8, 4)):
            sample = measure_file(info, mode, workers, stripe)
            if not sample["full_sha256_valid"]:
                raise AssertionError("Synthetic I/O SHA256 mismatch")
        bad = dict(info, expected_sha256="0" * 64)
        try:
            measure_file(bad, "stripe_preadv", 2, 4)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered checksum was accepted")
    return True


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "format": FORMAT, "status": "starting", "cpu_only": True,
        "gpu_used": False, "dataset_mutations": False,
        "real_model_inference": False, "full_model_weights_loaded": False,
        "source": "immutable Kaggle Dataset mounted safetensors",
        "scope": ("Full-file payload reads with receipt SHA256; no extra "
                  "download, model computation, hash readback or full E2E."),
        "caveats": [
            "A unique shard or Hop overlay is used per measurement where possible.",
            "Different Hops have different content; results remain subject to cache and mirror variability.",
            "The reader uses one final payload buffer per file, not the entire model.",
            "Thread executor startup is inside the case timing; this is an I/O ablation, not production host-cache timing.",
            "Full-file SHA256 is computed after I/O and excluded from reported MBps.",
            "A CPU-only mounted-source reading benchmark cannot establish 800MB/s throughput or E2E speedup.",
        ],
        "cases": [], "mounts": {}, "selftest": None,
    }
    save(output / "result.json", state)
    try:
        state["selftest"] = synthetic_selftest()
        save(output / "result.json", state)
        found, mounts = sources()
        state["mounts"] = mounts
        cases = case_plans(found)
        state["planned_cases"] = [
            {"id": case["id"], "kind": case["kind"],
             "workers": case["workers"], "stripe_MiB": case["stripe_MiB"],
             "strategy": case["strategy"], "timeout_seconds": case["timeout_seconds"],
             "file_count": len(case["files"])}
            for case in cases
        ]
        save(output / "result.json", state)
    except BaseException as error:
        state["status"] = "blocked_preflight"
        state["error_type"] = type(error).__name__
        state["error_message"] = str(error)[:600]
        save(output / "result.json", state)
        raise
    state["status"] = "running"
    save(output / "result.json", state)
    started = time.monotonic()
    budget = 1280
    for case in cases:
        if time.monotonic() - started >= budget:
            state["cases"].append({"id": case["id"], "status": "skipped_budget"})
            save(output / "result.json", state)
            continue
        plan = output / (case["id"] + ".plan.json")
        result = output / (case["id"] + ".result.json")
        save(plan, case)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--case", str(plan), "--case-output", str(result)],
                capture_output=True, text=True, timeout=case["timeout_seconds"],
                check=False)
            status = "passed" if proc.returncode == 0 else "failed"
            diagnostic = (proc.stdout + "\n" + proc.stderr)[-1000:]
        except subprocess.TimeoutExpired:
            status, diagnostic = "timeout", "Bounded subprocess timeout"
        summary = {
            "id": case["id"], "kind": case["kind"], "status": status,
            "workers": case["workers"], "strategy": case["strategy"],
            "stripe_MiB": case["stripe_MiB"],
            "elapsed_seconds": time.monotonic() - t0,
            "diagnostic_tail": diagnostic,
        }
        if result.exists():
            partial = json.loads(result.read_text())
            summary["validated_files"] = len(partial.get("samples", []))
            if partial.get("status") == "passed":
                summary["weighted_MBps"] = partial["weighted_MBps"]
                summary["weighted_MiBps"] = partial["weighted_MiBps"]
            elif status == "passed":
                summary["status"] = "failed_incomplete_receipt"
        state["cases"].append(summary)
        save(output / "result.json", state)
        print("PTD_STRIPE_CASE", case["id"], summary["status"],
              round(summary.get("weighted_MBps", 0.), 3), flush=True)
        if status == "failed":
            # An integrity or implementation failure is not a performance datapoint.
            state["integrity_or_case_failure"] = True
    state["elapsed_seconds"] = time.monotonic() - started
    failed = any(row["status"].startswith("failed") for row in state["cases"])
    incomplete = any(row["status"] in ("timeout", "skipped_budget")
                     for row in state["cases"])
    state["status"] = "failed" if failed else (
        "partial_bounded" if incomplete else "complete")
    save(output / "result.json", state)
    print("PTD_STRIPE_FINAL", state["status"], len(state["cases"]), flush=True)
    if failed:
        raise RuntimeError("At least one strategy failed; inspect per-case receipts")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--case-output", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps({"synthetic_sha256": synthetic_selftest()}))
    elif args.case:
        if args.case_output is None:
            parser.error("--case-output is required")
        run_case(json.loads(args.case.read_text()), args.case_output)
    elif args.output:
        main(args.output)
    else:
        parser.error("--output, --case or --selftest is required")


if __name__ == "__main__":
    cli()
