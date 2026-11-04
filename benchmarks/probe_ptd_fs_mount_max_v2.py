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
FORMAT = "h3-ptd-fsmount-max-adaptive-cpu-v2"


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
    """2x64MiB synthetic pilots; 2x96MiB per real worker; 8x96MiB best confirmation."""
    cases = []
    for privacy in ("private", "public"):
        cursor = Cursor(synthetic[privacy])
        windows = [cursor.take(64 * MIB, 8 * MIB) for _ in range(2)]
        cases.append({"id": f"synthetic_{privacy}_1",
                      "group": "synthetic", "workers": 1,
                      "window_bytes": 64 * MIB, "windows": windows})
    streams = sorted(real)
    if len(streams) != 12:
        raise ValueError("Exactly 12 Hop-rank streams required")
    cursors = {key: Cursor(real[key]) for key in streams}

    def fresh_windows(count):
        windows = []
        for _ in range(count):
            tasks = []
            for key in streams:
                for item in cursors[key].take(8 * MIB, 2 * MIB):
                    item["hop"], item["rank"] = key
                    tasks.append(item)
            assert sum(t["size"] for t in tasks) == 96 * MIB
            assert len(tasks) == 48
            windows.append(tasks)
        return windows

    for workers in WORKERS:
        cases.append({"id": f"ptd_pilot_{workers:02d}",
                      "group": "ptd_pilot", "workers": workers,
                      "window_bytes": 96 * MIB, "windows": fresh_windows(2)})
    cases.append({"id": "ptd_best_confirm", "group": "ptd_confirm",
                  "workers": 1, "window_bytes": 96 * MIB,
                  "windows": fresh_windows(8)})
    return cases


def early_stop_reason(summary, best_before_MBps, severe_drop_streak):
    """Hard-stop higher worker counts after a timeout, failure or repeated collapse."""
    if summary["status"] == "timeout":
        return "pilot_timeout"
    if summary["status"] != "complete":
        return "pilot_failed"
    if best_before_MBps is not None and severe_drop_streak >= 2:
        return "two_consecutive_below_30pct_of_best"
    return None



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
        "scope": "mounted Dataset application pread; not provider-side download",
        "caveats": [
            "Unique byte ranges are cold candidates, not proven physical cold reads.",
            "PTD scans 6 Hop mounts x 2 ranks without GPU kernels; pilot windows are short.",
            "Early stop after any pilot timeout can miss a non-monotonic higher-worker recovery.",
            "Only the best completed profile receives 8-window P10 confirmation.",
            "Measured speed is an application mounted-read rate, not a Kaggle API ZIP download rate.",
        ],
        "scheduling": {
            "synthetic_cases": 2, "synthetic_windows_each": 2,
            "ptd_pilot_workers": list(WORKERS), "ptd_pilot_windows_each": 2,
            "ptd_best_confirmation_windows": 8,
            "synthetic_case_timeout_seconds": 45,
            "ptd_pilot_timeout_seconds": 90,
            "ptd_confirmation_timeout_seconds": 150,
            "max_measurement_budget_seconds": 720,
            "early_stop_on_any_pilot_timeout": True,
            "early_stop_on_two_consecutive_below_best_fraction": 0.30,
        },
        "mounts": {}, "cases": [], "max_sustained_complete_profile": None,
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
    fake_real = {(hop, rank): [("/unused_real", 4 * 1024 * MIB)]
                 for hop in HOPS for rank in (0, 1)}
    fake_synthetic = {privacy: [("/unused_synthetic", 1024 * MIB)]
                      for privacy in PUBLIC_PRIVATE}
    if synthetic:
        cases.extend(c for c in build_cases(synthetic, fake_real)
                     if c["group"] == "synthetic")
    if real:
        cases.extend(c for c in build_cases(fake_synthetic, real)
                     if c["group"] in ("ptd_pilot", "ptd_confirm"))
    seen = set()
    for case in cases:
        for window in case["windows"]:
            for item in window:
                key = (item["path"], item["offset"], item["size"])
                if key in seen:
                    raise AssertionError("Duplicate cold-candidate source byte range")
                seen.add(key)
    state["status"] = "running"
    state["planned"] = [
        {"id": c["id"], "workers": c["workers"],
         "windows": len(c["windows"]),
         "planned_bytes": sum(sum(t["size"] for t in window)
                              for window in c["windows"])}
        for c in cases]
    save(output / "result.json", state)

    started = time.monotonic()
    best = None
    stop = None
    severe_drop_streak = 0
    for original in cases:
        case = original
        group = case["group"]
        if time.monotonic() - started > 720:
            state["cases"].append({"id": case["id"], "group": group,
                                   "status": "skipped_budget"})
            save(output / "result.json", state)
            continue
        if group == "ptd_pilot" and stop is not None:
            state["cases"].append({"id": case["id"], "group": group,
                                   "status": "skipped_early_stop", "reason": stop})
            save(output / "result.json", state)
            continue
        if group == "ptd_confirm":
            if best is None:
                state["cases"].append({"id": case["id"], "group": group,
                                       "status": "skipped_no_completed_pilot"})
                save(output / "result.json", state)
                continue
            case = dict(case, workers=best["workers"])
        timeout = (45 if group == "synthetic" else
                   90 if group == "ptd_pilot" else 150)
        path = output / (case["id"] + ".json")
        plan_path = output / (case["id"] + "_plan.json")
        save(plan_path, case)
        before_best = best["weighted_MBps"] if best is not None else None
        begin = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--case", str(plan_path), "--case-output", str(path)],
                capture_output=True, text=True, timeout=timeout, check=False)
            status = "complete" if proc.returncode == 0 else "failed"
            diag = (proc.stdout + "\n" + proc.stderr)[-1100:]
        except subprocess.TimeoutExpired:
            status, diag = "timeout", f"Hard case deadline of {timeout}s"
        summary = {"id": case["id"], "group": group,
                   "workers": case["workers"], "status": status,
                   "seconds": time.monotonic() - begin, "diagnostic": diag}
        if path.exists():
            data = json.loads(path.read_text())
            windows = data.get("windows", [])
            rates = [w["MBps"] for w in windows]
            summary["completed_windows"] = len(windows)
            if rates:
                summary.update(
                    samples=len(rates), min_MBps=min(rates),
                    max_MBps=max(rates), p10_MBps=percentile(rates, .1),
                    p50_MBps=percentile(rates, .5),
                    p90_MBps=percentile(rates, .9),
                    weighted_MBps=sum(w["read_bytes"] for w in windows) /
                        1e6 / sum(w["wall_seconds"] for w in windows),
                    qualified=status == "complete" and len(rates) == len(case["windows"]))
        if group == "ptd_pilot":
            if status == "complete" and summary.get("qualified"):
                if before_best is not None and summary["weighted_MBps"] < 0.30 * before_best:
                    severe_drop_streak += 1
                else:
                    severe_drop_streak = 0
                if best is None or summary["weighted_MBps"] > best["weighted_MBps"]:
                    best = summary.copy()
            else:
                severe_drop_streak = 0
            reason = early_stop_reason(summary, before_best, severe_drop_streak)
            if reason:
                stop = reason
                state["early_stop"] = {"reason": stop, "at_workers": case["workers"]}
        if group == "ptd_confirm" and summary.get("qualified"):
            state["max_sustained_complete_profile"] = {
                "workers": case["workers"], "MBps": summary["weighted_MBps"],
                "p10_MBps": summary["p10_MBps"],
                "meets_800_MBps_p10": summary["p10_MBps"] >= 800,
                "samples": summary["samples"],
            }
        state["cases"].append(summary)
        state["best_pilot_profile"] = best
        save(output / "result.json", state)
        print("PTD_MAX_IO_V2_CASE", case["id"], status,
              summary.get("completed_windows"), flush=True)
    state["elapsed_seconds"] = time.monotonic() - started
    state["status"] = ("partial_early_stop" if state.get("early_stop")
                       else "complete" if all(
                           c["status"] == "complete" for c in state["cases"])
                       else "partial")
    save(output / "result.json", state)
    print("PTD_MAX_IO_V2_DONE", state["status"], flush=True)



def selftest():
    syn = {privacy: [(f"{privacy}.bin", 1024 * MIB)] for privacy in PUBLIC_PRIVATE}
    real = {(hop, rank): [(f"hop{hop}_rank{rank}.bin", 2048 * MIB)]
            for hop in HOPS for rank in (0, 1)}
    cases = build_cases(syn, real)
    assert len(cases) == 9
    assert [c["workers"] for c in cases if c["group"] == "ptd_pilot"] == list(WORKERS)
    assert sum(c["group"] == "ptd_confirm" for c in cases) == 1
    seen = set()
    for case in cases:
        expected_windows = (8 if case["group"] == "ptd_confirm" else 2)
        assert len(case["windows"]) == expected_windows
        for window in case["windows"]:
            assert sum(t["size"] for t in window) == case["window_bytes"]
            assert len(window) >= case["workers"]
            for task in window:
                key = (task["path"], task["offset"], task["size"])
                assert key not in seen
                seen.add(key)
    assert sum(sum(t["size"] for w in c["windows"] for t in w)
               for c in cases) == 2176 * MIB
    assert early_stop_reason({"status": "timeout"}, None, 0) == "pilot_timeout"
    assert early_stop_reason({"status": "failed"}, None, 0) == "pilot_failed"
    assert early_stop_reason({"status": "complete"}, 100, 2) == "two_consecutive_below_30pct_of_best"
    assert early_stop_reason({"status": "complete"}, 100, 1) is None
    print("PTD_MAX_IO_V2_SELFTEST_PASS", len(cases), len(seen))



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
