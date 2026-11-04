"""Bounded post-2NFE Kaggle-mounted Dataset payload throughput probe.

This measures application-visible reads from already mounted files. It cannot
observe or infer Kaggle's provider-side, pre-notebook Dataset download/staging.
No payload hashing, Dataset publication, GPU allocation, or runtime weight bake.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import statistics
import time

FORMAT = "h3-ptd-mounted-payload-throughput-v1"
TARGET_MBPS = 800.0
PROFILES = (1, 4, 8, 16)
WINDOWS = 8
WINDOW_BYTES = 512 * 2**20
CHUNK_BYTES = 8 * 2**20


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    point = (len(ordered) - 1) * fraction
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def _process_io():
    try:
        return {key: int(value) for line in Path("/proc/self/io").read_text().splitlines()
                for key, value in [line.split(":", 1)]}
    except (OSError, ValueError):
        return {}


def _memory_stat():
    try:
        return {key: int(value) for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
                for key, value in [line.split()]}
    except (OSError, ValueError):
        return {}


def _mounted_root(dataset):
    slug = dataset.split("/", 1)[1]
    choices = (Path("/kaggle/input") / slug,
               Path("/kaggle/input/datasets") / dataset,
               Path("/kaggle/input/datasets") / slug,
               Path("/kaggle/input") / dataset)
    found = [p / "payload" for p in choices if (p / "payload").is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one mounted Dataset payload: {dataset}; got {len(found)}")
    return found[0].resolve()


def _sources(proofs):
    """Use untouched Hop02..07 after the two NFE; avoid repeat byte ranges."""
    rows = []
    for hop in range(2, 8):
        part = f"hop_{hop:02d}"
        root = _mounted_root(proofs[part]["dataset"])
        receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        if receipt.get("hop") != hop or len(receipt.get("shards", {})) != 100:
            raise ValueError("Mounted PTD Bank shard inventory changed: " + part)
        for stem in sorted(receipt["shards"]):
            if not stem.startswith("dit.block") or ".rank" not in stem:
                raise ValueError("Unexpected PTD rank filename: " + stem)
            path = root / (stem + ".safetensors")
            size = receipt["shards"][stem]["bytes"]
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError("PTD mounted shard size mismatch: " + str(path))
            rows.append((path, size))
    return rows


class UniqueRegions:
    def __init__(self, sources):
        self.sources = list(sources)
        self.index = 0
        self.offset = 0
        self.total_available = sum(size for _, size in self.sources)

    def segments(self, amount, chunk_bytes):
        result = []
        left = amount
        while left:
            if self.index >= len(self.sources):
                raise ValueError("Not enough non-overlapping PTD bytes to measure throughput")
            path, size = self.sources[self.index]
            take = min(left, chunk_bytes, size - self.offset)
            if take <= 0:
                self.index += 1
                self.offset = 0
                continue
            result.append((path, self.offset, take))
            self.offset += take
            left -= take
            if self.offset == size:
                self.index += 1
                self.offset = 0
        return result


def _read_exact(task):
    fd, offset, size = task
    data = os.pread(fd, size, offset)
    if len(data) != size:
        raise IOError(f"Mounted Dataset short read at offset {offset}: {len(data)}/{size}")
    return len(data)


def _delta(after, before, key):
    return after[key] - before[key] if key in after and key in before else None


def measure(sources, *, profiles=PROFILES, windows=WINDOWS,
            window_bytes=WINDOW_BYTES, chunk_bytes=CHUNK_BYTES,
            target_mbps=TARGET_MBPS, max_seconds=600, output=None):
    """Report unique-region read throughput, never provider network throughput."""
    if (not hasattr(os, "pread") or not sources or not profiles or
            any(type(v) is not int or v < 1 or v > 32 for v in profiles) or
            not 2 <= windows <= 64 or not 0 < chunk_bytes <= 16 * 2**20 or
            not 0 < window_bytes <= 2**30 or window_bytes % chunk_bytes != 0 or
            not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("Invalid bounded mounted-I/O throughput probe contract")
    cursor = UniqueRegions(sources)
    required = len(profiles) * windows * window_bytes
    if cursor.total_available < required:
        raise ValueError(f"Insufficient unique shard bytes: {cursor.total_available} < {required}")
    report = dict(format=FORMAT, status="running",
                  target_MBps=target_mbps, profiles=list(profiles),
                  windows_per_profile=windows, window_bytes=window_bytes,
                  chunk_bytes=chunk_bytes, unique_requested_bytes=required,
                  available_unique_bytes=cursor.total_available,
                  scope="already-mounted Dataset regular-file os.pread; not Kaggle download",
                  cache_control="unique non-overlapping byte ranges; OS cache state not fully controlled",
                  page_cache_drop=False, payload_sha256=False, gpu_operations=False,
                  source_hops=[2, 3, 4, 5, 6, 7], readings=[])
    def save():
        if output:
            Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    started = time.monotonic()
    try:
        for workers in profiles:
            windows_data = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for index in range(windows):
                    if time.monotonic() - started >= max_seconds:
                        raise TimeoutError("Mounted-I/O throughput budget exceeded")
                    regions = cursor.segments(window_bytes, chunk_bytes)
                    opened = {}
                    try:
                        for path, _, _ in regions:
                            if path not in opened:
                                opened[path] = os.open(path, os.O_RDONLY)
                        tasks = [(opened[path], offset, size) for path, offset, size in regions]
                        io_before, memory_before = _process_io(), _memory_stat()
                        begin = time.perf_counter()
                        transferred = sum(pool.map(_read_exact, tasks))
                        elapsed = time.perf_counter() - begin
                        io_after, memory_after = _process_io(), _memory_stat()
                    finally:
                        for fd in opened.values():
                            os.close(fd)
                    if transferred != window_bytes or elapsed <= 0:
                        raise AssertionError("Incomplete mounted-payload throughput sample")
                    row = dict(profile_workers=workers, window=index,
                               application_read_bytes=transferred,
                               wall_seconds=elapsed,
                               MBps=transferred / 1e6 / elapsed,
                               MiBps=transferred / 2**20 / elapsed,
                               proc_read_bytes_delta=_delta(io_after, io_before, "read_bytes"),
                               cgroup_file_bytes_delta=_delta(memory_after, memory_before, "file"),
                               cgroup_refault_file_delta=_delta(memory_after, memory_before, "workingset_refault_file"),
                               cgroup_pgscan_delta=_delta(memory_after, memory_before, "pgscan"))
                    windows_data.append(row)
                    report["readings"].append(row)
                    save()
            speeds = [row["MBps"] for row in windows_data]
            summary = dict(workers=workers, samples=len(speeds),
                           mean_MBps=statistics.mean(speeds),
                           min_MBps=min(speeds), max_MBps=max(speeds),
                           p10_MBps=_percentile(speeds, 0.1),
                           p50_MBps=_percentile(speeds, 0.5),
                           p90_MBps=_percentile(speeds, 0.9),
                           wall_weighted_MBps=sum(row["application_read_bytes"] for row in windows_data) /
                                              1e6 / sum(row["wall_seconds"] for row in windows_data))
            summary["passes_800MBps_p10"] = summary["p10_MBps"] >= target_mbps
            report.setdefault("summary", []).append(summary)
            print("PTD_MOUNTED_IO_PROFILE", json.dumps(summary, sort_keys=True), flush=True)
            save()
        report["current_1_worker_passes"] = next(
            (item["passes_800MBps_p10"] for item in report["summary"] if item["workers"] == 1), None)
        report["any_profile_passes"] = any(item["passes_800MBps_p10"] for item in report["summary"])
        report["status"] = "completed"
        report["elapsed_seconds"] = time.monotonic() - started
        save()
        print("PTD_MOUNTED_IO_FINAL", report["status"], report["current_1_worker_passes"],
              report["any_profile_passes"], flush=True)
        return report
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__,
                      error_message=str(exc), elapsed_seconds=time.monotonic() - started)
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proofs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise FileExistsError("Preserve prior throughput measurements and require existing output parent")
    proofs = json.loads(args.proofs.read_text(encoding="utf-8"))
    if set(proofs) != {"shared"} | {f"hop_{i:02d}" for i in range(8)}:
        raise ValueError("Expected immutable 9-part PTD Dataset proof inventory")
    measure(_sources(proofs), output=args.output, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
