"""CPU-only PTD mounted-source filesystem and max-concurrency experiment.

Two identical 1GiB synthetic Datasets determine public/private mount type.
Six real immutable PTD Hop02..07 inputs simulate twelve independent rank streams.
All reads are bounded and cold-candidate, not guaranteed cache misses or provider
download traffic. No GPU, downloads, Dataset mutations, hashes of large weights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

MIB = 1 << 20
HOPS = tuple(range(2, 8))
PUBLIC_PRIVATE = {
    "private": "pamcks/h3-ptd-fsmount-ab-private-20261103",
    "public": "pamcks/h3-ptd-fsmount-ab-public-20261103",
}
SYNTH_MANIFEST_SHA256 = "f34480d1ca029b9ebbf6940bd9f25627f6393ddfd42f673a2aa131df29d3314e"
HOP_HANDLES = {hop: f"pamcks/h3-ptd-v2-hop-{hop:02d}-20261103" for hop in HOPS}
WORKERS = (1, 2, 4, 8, 16, 32)
FORMAT = "h3-ptd-fsmount-max-balanced-cpu-v1"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def percentile(values, p):
    if not values:
        return None
    a = sorted(values)
    x = (len(a) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return a[lo] + (a[hi] - a[lo]) * (x - lo)


def assert_cpu():
    if list(Path("/dev").glob("nvidia[0-9]*")):
        raise RuntimeError("CPU-only Notebook required")


def candidate_roots(handle):
    owner, slug = handle.split("/")
    base = Path("/kaggle/input")
    candidates = [base / slug, base / "datasets" / owner / slug,
                  base / "datasets" / slug, base / owner / slug]
    return [p for c in candidates for p in (c, c / "payload")]


def brief_layout():
    """Bounded, read-only diagnostic; never traverse weight contents."""
    base = Path("/kaggle/input")
    report = {"top": [], "matching": {}}
    try:
        report["top"] = sorted(p.name for p in base.iterdir())[:40]
    except OSError as error:
        report["error_type"] = type(error).__name__
    for handle in list(PUBLIC_PRIVATE.values()) + list(HOP_HANDLES.values()):
        matching = []
        for p in candidate_roots(handle):
            try:
                if p.is_dir():
                    matching.append({"path": str(p),
                                     "entries": sorted(x.name for x in p.iterdir())[:12]})
            except OSError:
                continue
        report["matching"][handle] = matching
    return report


def locate(handle, marker):
    found = [p for p in candidate_roots(handle) if (p / marker).is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one attached {handle} with {marker}, found {len(found)}")
    return found[0]


def mount_type(path):
    resolved = str(path.resolve())
    best = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        fields = left.split()
        if len(fields) < 6:
            continue
        mountpoint = fields[4].replace("\\040", " ")
        if resolved != mountpoint and not resolved.startswith(mountpoint.rstrip("/") + "/"):
            continue
        detail = right.split()
        if len(detail) < 2:
            continue
        if best is None or len(mountpoint) > len(best["mountpoint"]):
            best = {"mountpoint": mountpoint, "filesystem": detail[0],
                    "source": detail[1], "options": fields[5],
                    "super_options": detail[2] if len(detail) > 2 else ""}
    if best is None:
        raise ValueError(f"Mount type unknown for {path}")
    return best


def memory_stat():
    try:
        rows = (Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
        data = dict(line.split() for line in rows)
        return {n: int(data[n]) for n in
                ("file", "workingset_refault_file", "pgscan", "pgmajfault") if n in data}
    except (OSError, ValueError):
        return {}


class Cursor:
    def __init__(self, files):
        self.files = list(files)
        self.index = 0
        self.offset = 0

    def take(self, amount, chunk):
        tasks = []
        while amount:
            if self.index == len(self.files):
                raise ValueError("Insufficient non-overlapping source ranges")
            name, length = self.files[self.index]
            n = min(amount, chunk, length - self.offset)
            if n <= 0:
                self.index += 1
                self.offset = 0
                continue
            tasks.append(dict(path=str(name), offset=self.offset, size=n))
            self.offset += n
            amount -= n
            if self.offset == length:
                self.index += 1
                self.offset = 0
        return tasks


def synthetic_sources():
    files = {}
    mounts = {}
    for privacy, handle in PUBLIC_PRIVATE.items():
        root = locate(handle, "manifest.json")
        manifest_file = root / "manifest.json"
        checksum = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        if checksum != SYNTH_MANIFEST_SHA256:
            raise ValueError(f"Unexpected source-manifest SHA256 for {privacy}")
        manifest = json.loads(manifest_file.read_text())
        if not manifest.get("synthetic_only") or len(manifest.get("shards", [])) != 32:
            raise ValueError("Synthetic Dataset metadata mismatch")
        pairs = []
        for entry in manifest["shards"]:
            path = root / entry["name"]
            length = int(entry["bytes"])
            if length != 32 * MIB or path.stat().st_size != length:
                raise ValueError(f"Unexpected {privacy} synthetic shard size")
            pairs.append((str(path), length))
        files[privacy] = pairs
        mounts[privacy] = {"handle": handle, "root": str(root),
                           "fs": mount_type(root),
                           "manifest_sha256": checksum, "bytes": sum(n for _, n in pairs)}
    return files, mounts


def hop_sources():
    files = {}
    mounts = {}
    for hop, handle in HOP_HANDLES.items():
        root = locate(handle, "receipt.json")
        receipt = json.loads((root / "receipt.json").read_text())
        shards = receipt.get("shards", {})
        if receipt.get("hop") != hop or len(shards) != 100:
            raise ValueError(f"Hop{hop:02d} receipt invalid")
        for rank in (0, 1):
            pairs = []
            for stem, descriptor in sorted(shards.items()):
                if not stem.startswith("dit.block") or ".rank" not in stem:
                    raise ValueError("Unexpected PTD rank-shard name")
                tail = stem.rsplit(".rank", 1)[1]
                if tail != str(rank):
                    continue
                path = root / (stem + ".safetensors")
                size = int(descriptor["bytes"])
                if size <= 0 or path.stat().st_size != size:
                    raise ValueError(f"Hop{hop:02d} rank{rank} shard size mismatch")
                pairs.append((str(path), size))
            if len(pairs) != 50:
                raise ValueError(f"Hop{hop:02d} rank{rank}: expected 50 shards, got {len(pairs)}")
            files[(hop, rank)] = pairs
        mounts[str(hop)] = {"handle": handle, "root": str(root),
                            "fs": mount_type(root),
                            "logical_bytes": sum(n for rank in (0, 1) for _, n in files[(hop, rank)])}
    return files, mounts


def build_cases(synthetic, real):
    cases = []
    for privacy, workers in (("private", 1), ("public", 1),
                             ("public", 4), ("private", 4)):
        cursor = Cursor(synthetic[privacy])
        if workers == 4:
            cursor.take(512 * MIB, 8 * MIB)
        windows = []
        for _ in range(8):
            windows.append(cursor.take(64 * MIB, 8 * MIB))
        cases.append({"id": f"synthetic_{privacy}_{workers}",
                      "group": "synthetic", "workers": workers,
                      "window_bytes": 64 * MIB, "windows": windows})
    # Real workload: six Dataset mounts * two rank streams, equal shares per window.
    # 4MiB positional reads, enough outstanding tasks to saturate up to 32 threads.
    streams = sorted(real)
    if len(streams) != 12:
        raise ValueError("Exactly 12 Hop-rank streams required")
    cursors = {key: Cursor(real[key]) for key in streams}
    for workers in WORKERS:
        windows = []
        for _ in range(8):
            tasks = []
            for key in streams:
                for item in cursors[key].take(16 * MIB, 4 * MIB):
                    item["hop"], item["rank"] = key
                    tasks.append(item)
            assert sum(task["size"] for task in tasks) == 192 * MIB
            assert len(tasks) >= workers
            windows.append(tasks)
        cases.append({"id": f"ptd_12rank_{workers:02d}",
                      "group": "ptd", "workers": workers,
                      "window_bytes": 192 * MIB, "windows": windows})
    return cases


def case_worker(case, output):
    rows = []
    result = {"id": case["id"], "group": case["group"],
              "workers": case["workers"], "status": "running", "windows": rows}
    save(output, result)
    with ThreadPoolExecutor(max_workers=case["workers"]) as executor:
        for index, tasks in enumerate(case["windows"]):
            fds = {}
            try:
                for item in tasks:
                    if item["path"] not in fds:
                        fds[item["path"]] = os.open(item["path"], os.O_RDONLY)
                def operation(item):
                    started = time.perf_counter()
                    result_bytes = os.pread(
                        fds[item["path"]], item["size"], item["offset"])
                    if len(result_bytes) != item["size"]:
                        raise IOError("Short mounted Dataset read")
                    return len(result_bytes), time.perf_counter() - started
                before = memory_stat()
                started = time.perf_counter()
                completed = list(executor.map(operation, tasks))
                elapsed = time.perf_counter() - started
                after = memory_stat()
            finally:
                for fd in fds.values():
                    os.close(fd)
            total = sum(b for b, _ in completed)
            if total != case["window_bytes"]:
                raise AssertionError("Window byte accounting mismatch")
            row = {"index": index, "read_bytes": total,
                   "wall_seconds": elapsed, "MBps": total / 1e6 / elapsed,
                   "chunk_latency_max_seconds": max(t for _, t in completed),
                   "cgroup_delta": {k: after[k] - before[k] for k in before.keys() & after.keys()}}
            rows.append(row)
            save(output, result)
            print("PTD_MAX_IO_WINDOW", case["id"], index, round(row["MBps"], 3), flush=True)
    rates = [row["MBps"] for row in rows]
    result.update(status="complete", samples=len(rates),
                  p10_MBps=percentile(rates, .1),
                  p50_MBps=percentile(rates, .5),
                  p90_MBps=percentile(rates, .9),
                  max_MBps=max(rates),
                  wall_weighted_MBps=sum(row["read_bytes"] for row in rows) /
                      1e6 / sum(row["wall_seconds"] for row in rows),
                  stable_ge_800_MBps=percentile(rates, .1) >= 800)
    save(output, result)


def master(output):
    assert_cpu()
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "format": FORMAT, "status": "starting", "cpu_only": True, "gpu_used": False,
        "scope": "mounted Dataset application-level pread throughput; not pre-Notebook transfer",
        "caveats": [
            "Unique byte ranges are cold candidates but OS/FUSE/provider cache misses are not proven.",
            "PTD mode models six Hop mounts and two rank streams each, without GPU compute or weight decoding.",
            "Per-case timeouts censor slow profiles; incomplete cases cannot qualify maximum stable throughput.",
            "Application reads are not independent evidence of Kaggle provider network download bandwidth.",
            "Only 1GiB synthetic controls; these do not establish all ~21GiB per-Hop effects.",
        ],
        "mounts": {}, "cases": [], "summaries": {},
        "max_sustained_complete_profile": None,
    }
    save(output / "result.json", state)
    state["layout"] = brief_layout()
    save(output / "result.json", state)
    synthetic, real = {}, {}
    try:
        synthetic, syn_mounts = synthetic_sources()
        state["mounts"]["synthetic"] = syn_mounts
    except Exception as exc:
        state["synthetic_error_type"] = type(exc).__name__
        state["synthetic_error_message"] = str(exc)[:400]
    save(output / "result.json", state)
    try:
        real, real_mounts = hop_sources()
        state["mounts"]["ptd"] = real_mounts
    except Exception as exc:
        state["ptd_error_type"] = type(exc).__name__
        state["ptd_error_message"] = str(exc)[:400]
    save(output / "result.json", state)
    if not synthetic and not real:
        state["status"] = "blocked_no_mounts"
        save(output / "result.json", state)
        return
    cases = []
    if synthetic:
        synthetic_cases = build_cases(synthetic, {
            (hop, rank): [("/unused", 4 * 1024 * MIB)]
            for hop in HOPS for rank in (0, 1)})
        cases.extend(case for case in synthetic_cases if case["group"] == "synthetic")
    if real:
        real_cases = build_cases({
            privacy: [("/unused", 1024 * MIB)]
            for privacy in PUBLIC_PRIVATE}, real)
        cases.extend(case for case in real_cases if case["group"] == "ptd")
    all_tasks = set()
    for case in cases:
        for window in case["windows"]:
            for t in window:
                key = (t["path"], t["offset"], t["size"])
                if key in all_tasks:
                    raise AssertionError("Duplicate cold-candidate byte range")
                all_tasks.add(key)
    state["status"] = "running"
    state["planned"] = [{"id": c["id"], "workers": c["workers"],
                         "windows": len(c["windows"]),
                         "read_bytes": sum(sum(t["size"] for t in w) for w in c["windows"])}
                        for c in cases]
    save(output / "result.json", state)
    started = time.monotonic()
    for case in cases:
        if time.monotonic() - started > 2250:
            state["cases"].append({"id": case["id"], "status": "skipped_budget"})
            save(output / "result.json", state)
            continue
        path = output / (case["id"] + ".json")
        case_plan = output / (case["id"] + "_plan.json")
        save(case_plan, case)
        begin = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--case", str(case_plan), "--case-output", str(path)],
                capture_output=True, text=True, timeout=210, check=False)
            status = "complete" if proc.returncode == 0 else "failed"
            tail = (proc.stdout + "\n" + proc.stderr)[-1000:]
        except subprocess.TimeoutExpired:
            status, tail = "timeout", "Isolated 210-second case timeout"
        summary = {"id": case["id"], "workers": case["workers"],
                   "group": case["group"], "status": status,
                   "seconds": time.monotonic() - begin, "diagnostic": tail}
        if path.exists():
            data = json.loads(path.read_text())
            rates = [w["MBps"] for w in data.get("windows", [])]
            summary["completed_windows"] = len(rates)
            if rates:
                summary.update(samples=len(rates), min_MBps=min(rates),
                               max_MBps=max(rates), p10_MBps=percentile(rates, .1),
                               p50_MBps=percentile(rates, .5),
                               p90_MBps=percentile(rates, .9),
                               weighted_MBps=sum(w["read_bytes"] for w in data["windows"]) /
                                   1e6 / sum(w["wall_seconds"] for w in data["windows"]),
                               qualified=status == "complete" and len(rates) == 8)
        state["cases"].append(summary)
        save(output / "result.json", state)
        print("PTD_MAX_IO_CASE", case["id"], status, summary.get("completed_windows"), flush=True)
    real_complete = [
        item for item in state["cases"]
        if item["group"] == "ptd" and item.get("qualified")
    ]
    if real_complete:
        best = max(real_complete, key=lambda r: r["weighted_MBps"])
        state["max_sustained_complete_profile"] = {
            "workers": best["workers"], "MBps": best["weighted_MBps"],
            "p10_MBps": best["p10_MBps"],
            "meets_800_MBps_p10": best["p10_MBps"] >= 800,
        }
    state["status"] = ("complete" if all(
        item["status"] == "complete" for item in state["cases"]) else "partial")
    state["elapsed_seconds"] = time.monotonic() - started
    save(output / "result.json", state)
    print("PTD_MAX_IO_DONE", state["status"], flush=True)


def selftest():
    sample = {privacy: [(f"{privacy}.bin", 1024 * MIB)] for privacy in PUBLIC_PRIVATE}
    real = {(hop, rank): [(f"hop{hop}_rank{rank}.bin", 2 * 1024 * MIB)]
            for hop in HOPS for rank in (0, 1)}
    cases = build_cases(sample, real)
    assert len(cases) == 10
    seen = set()
    for case in cases:
        assert len(case["windows"]) == 8
        for window in case["windows"]:
            assert sum(t["size"] for t in window) == case["window_bytes"]
            assert len(window) >= case["workers"]
            for task in window:
                key = (task["path"], task["offset"], task["size"])
                assert key not in seen
                seen.add(key)
    assert sum(sum(t["size"] for w in c["windows"] for t in w) for c in cases) == (
        4 * 512 * MIB + 6 * 1536 * MIB)
    assert percentile([1, 2, 3, 4], .1) == 1.3
    print("PTD_MAX_IO_SELFTEST_PASS", len(cases), len(seen))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    p.add_argument("--case", type=Path)
    p.add_argument("--case-output", type=Path)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        selftest()
    elif a.case:
        if not a.case_output:
            p.error("Need --case-output")
        case_worker(json.loads(a.case.read_text()), a.case_output)
    elif a.output:
        master(a.output)
    else:
        p.error("Specify output, case or self-test")


if __name__ == "__main__":
    main()
