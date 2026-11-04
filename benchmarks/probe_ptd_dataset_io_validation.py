"""CPU-only, balanced PTD Dataset mounted-read validation.

No model loading, payload hashing, uploads, GPU usage or provider-side download timing.
Every cold-candidate byte range is unique. Slow cases are isolated in bounded
subprocesses so an NFS stall cannot prevent later concurrency profiles.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HOPS = tuple(range(2, 8))
MIB = 2 ** 20
WINDOW = 192 * MIB
TARGET_MBPS = 800.0
DATASET_PREFIX = "pamcks/h3-ptd-v2-hop-"
SCHEDULE = (
    ("balanced_8m", 1, 4), ("balanced_8m", 4, 4),
    ("balanced_8m", 8, 4), ("balanced_8m", 16, 4),
    ("balanced_8m", 16, 4), ("balanced_8m", 8, 4),
    ("balanced_8m", 4, 4), ("balanced_8m", 1, 4),
    ("balanced_1m", 4, 2), ("balanced_1m", 4, 2),
    ("hop02_8m", 4, 2),
)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    x = (len(ordered) - 1) * fraction
    lo, hi = math.floor(x), math.ceil(x)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def mounted_sources():
    groups = {}
    for hop in HOPS:
        dataset = f"{DATASET_PREFIX}{hop:02d}-20261103"
        slug = dataset.split("/")[1]
        roots = [
            Path("/kaggle/input") / slug,
            Path("/kaggle/input/datasets") / dataset,
            Path("/kaggle/input/datasets") / slug,
            Path("/kaggle/input") / dataset,
        ]
        found = [root / "payload" for root in roots if (root / "payload").is_dir()]
        if len(found) != 1:
            raise FileNotFoundError(f"One mounted payload required for {dataset}; found {len(found)}")
        root = found[0]
        receipt = read_json(root / "receipt.json")
        shards = receipt.get("shards", {})
        if receipt.get("hop") != hop or len(shards) != 100:
            raise ValueError(f"Invalid Hop {hop:02d} receipt inventory")
        files = []
        for stem in sorted(shards):
            if not stem.startswith("dit.block") or ".rank" not in stem:
                raise ValueError("Unexpected rank shard name")
            path = root / (stem + ".safetensors")
            size = int(shards[stem]["bytes"])
            if size <= 0 or not path.is_file() or path.stat().st_size != size:
                raise ValueError("Missing or inconsistent rank shard: " + str(path))
            files.append((str(path), size))
        groups[hop] = files
    return groups


class Cursor:
    def __init__(self, files):
        self.files = files
        self.index = 0
        self.offset = 0
        self.available = sum(size for _, size in files)

    def take(self, amount, chunk, hop):
        segments = []
        while amount:
            if self.index == len(self.files):
                raise ValueError(f"Hop {hop:02d} has insufficient distinct ranges")
            name, size = self.files[self.index]
            count = min(amount, chunk, size - self.offset)
            if not count:
                self.index += 1
                self.offset = 0
                continue
            segments.append(dict(path=name, offset=self.offset, size=count, hop=hop))
            self.offset += count
            amount -= count
            if self.offset == size:
                self.index += 1
                self.offset = 0
        return segments


def build_plan(groups):
    cursors = {hop: Cursor(groups[hop]) for hop in HOPS}
    cases = []
    first = None
    for idx, (mode, workers, count) in enumerate(SCHEDULE):
        chunk = MIB if mode == "balanced_1m" else 8 * MIB
        hops = (2,) if mode == "hop02_8m" else HOPS
        windows = []
        for _ in range(count):
            tasks = []
            for hop in hops:
                tasks.extend(cursors[hop].take(WINDOW // len(hops), chunk, hop))
            if sum(item["size"] for item in tasks) != WINDOW:
                raise AssertionError("Incorrect balanced window total")
            if len(hops) == 6 and any(
                sum(t["size"] for t in tasks if t["hop"] == hop) != WINDOW // 6
                for hop in HOPS
            ):
                raise AssertionError("Window did not balance all six Hop mounts")
            windows.append(tasks)
        if idx == 0:
            first = windows[0]
        cases.append(dict(id=f"case_{idx:02d}", mode=mode, workers=workers,
                          windows=windows, repeat_region=False))
    cases.append(dict(id="case_11", mode="repeat_region_8m", workers=4,
                      windows=[first, first], repeat_region=True))
    return dict(format="h3-ptd-mounted-io-balanced-plan-v1", cases=cases,
                source_hops=list(HOPS),
                cold_candidate_bytes=sum(
                    sum(t["size"] for window in c["windows"] for t in window)
                    for c in cases if not c["repeat_region"]),
                repeat_region_bytes=2 * WINDOW)


def _memory_stat():
    try:
        rows = (Path("/sys/fs/cgroup/memory.stat").read_text(encoding="utf-8")
                .splitlines())
        data = dict((a, int(b)) for a, b in (line.split() for line in rows))
        return {k: data.get(k) for k in
                ("file", "workingset_refault_file", "pgscan", "pgmajfault")}
    except (OSError, ValueError):
        return {}


def _nfs_reads():
    try:
        rows = Path("/proc/self/mountstats").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    current = None
    found = {}
    for row in rows:
        if row.startswith("device "):
            current = next(
                (hop for hop in HOPS
                 if f"h3-ptd-v2-hop-{hop:02d}-20261103" in row), None)
        elif current is not None and row.strip().startswith("READ:"):
            try:
                found[str(current)] = [int(value) for value in
                                       row.split(":", 1)[1].split()]
            except ValueError:
                pass
    return found


def _nfs_delta(after, before):
    return {
        hop: [a - b for a, b in zip(values, before[hop])]
        for hop, values in after.items()
        if hop in before and len(values) == len(before[hop])
    }


def _memory_delta(after, before):
    return {key: after[key] - before[key] for key in after.keys() & before.keys()
            if after[key] is not None and before[key] is not None}


def _mount_types():
    try:
        rows = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    found = {}
    for line in rows:
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        mount = left.split()[4]
        for hop in HOPS:
            if f"h3-ptd-v2-hop-{hop:02d}-20261103" not in mount:
                continue
            parts = right.split()
            options = left.split()[5] + "," + (parts[2] if len(parts) > 2 else "")
            selected = [s for s in options.split(",") if s.startswith(
                ("rsize=", "vers=", "timeo=", "retrans="))]
            found[str(hop)] = dict(fs=parts[0], options=selected)
    return found


def run_case(case, output):
    windows = []
    result = dict(id=case["id"], mode=case["mode"], workers=case["workers"],
                  repeat_region=case["repeat_region"], status="running",
                  windows=windows, mount_types=_mount_types())
    save(output, result)
    with ThreadPoolExecutor(max_workers=case["workers"]) as pool:
        for index, window in enumerate(case["windows"]):
            fds = {}
            for task in window:
                if task["path"] not in fds:
                    fds[task["path"]] = os.open(task["path"], os.O_RDONLY)
            def read_one(task):
                begin = time.perf_counter()
                data = os.pread(fds[task["path"]], task["size"], task["offset"])
                elapsed = time.perf_counter() - begin
                if len(data) != task["size"]:
                    raise IOError("Short mounted read")
                return dict(hop=task["hop"], seconds=elapsed,
                            bytes=len(data))
            try:
                mem_before, nfs_before = _memory_stat(), _nfs_reads()
                begin = time.perf_counter()
                completed = list(pool.map(read_one, window))
                seconds = time.perf_counter() - begin
                mem_after, nfs_after = _memory_stat(), _nfs_reads()
            finally:
                for fd in fds.values():
                    os.close(fd)
            total = sum(item["bytes"] for item in completed)
            if total != WINDOW:
                raise AssertionError("Window byte total mismatch")
            durations = [item["seconds"] for item in completed]
            row = dict(index=index, wall_seconds=seconds,
                       read_bytes=total, MBps=total / 1e6 / seconds,
                       chunk_count=len(completed), chunk_seconds_p50=percentile(durations, .5),
                       chunk_seconds_p90=percentile(durations, .9),
                       chunk_seconds_p99=percentile(durations, .99),
                       chunk_seconds_max=max(durations),
                       hop_bytes={str(hop): sum(t["bytes"] for t in completed
                                               if t["hop"] == hop) for hop in HOPS},
                       nfs_read_rpc_delta=_nfs_delta(nfs_after, nfs_before),
                       cgroup_delta=_memory_delta(mem_after, mem_before))
            windows.append(row)
            save(output, result)
            print("PTD_IO_WINDOW", case["id"], index, round(row["MBps"], 3),
                  "max_chunk_s", round(row["chunk_seconds_max"], 3), flush=True)
    rates = [row["MBps"] for row in windows]
    result.update(status="completed", samples=len(rates),
                  p10_MBps=percentile(rates, .1),
                  p50_MBps=percentile(rates, .5),
                  wall_weighted_MBps=sum(row["read_bytes"] for row in windows) /
                      1e6 / sum(row["wall_seconds"] for row in windows))
    save(output, result)


def run_master(output, *, max_seconds=1200, case_timeout=115):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(format="h3-ptd-mounted-io-validation-v1", status="starting",
                  scope="already-mounted PTD Dataset application reads; not provider download",
                  target_MBps=TARGET_MBPS, source_hops=list(HOPS), cases=[],
                  caveats=[
                      "Unique non-overlapping ranges are cold candidates, not proven uncached storage reads.",
                      "NFS mountstats and cgroup counters are observational; their absence is not proof of zero I/O.",
                      "Timeout-censored cases do not qualify performance, and provider pre-kernel attach is unobserved.",
                  ])
    save(output / "result.json", report)
    groups = mounted_sources()
    plan = build_plan(groups)
    save(output / "plan.json", plan)
    report["source_bytes"] = {str(hop): sum(size for _, size in groups[hop])
                              for hop in HOPS}
    report["status"] = "running"
    save(output / "result.json", report)
    started = time.monotonic()
    for case in plan["cases"]:
        remaining = max_seconds - (time.monotonic() - started)
        if remaining < 20:
            report["cases"].append(dict(id=case["id"], status="skipped_budget"))
            save(output / "result.json", report)
            continue
        case_path = output / (case["id"] + ".json")
        job = output / (case["id"] + "_plan.json")
        save(job, case)
        limit = min(case_timeout, remaining - 5)
        begin = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--case-plan", str(job), "--case-output", str(case_path)],
                capture_output=True, text=True, timeout=limit, check=False)
            status = "completed" if proc.returncode == 0 else "failed"
            diagnostic = (proc.stdout + "\n" + proc.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            status, diagnostic = "timeout", "Case exceeded its isolated deadline"
        report["cases"].append(dict(id=case["id"], status=status,
                                    seconds=time.monotonic() - begin,
                                    diagnostic=diagnostic))
        if case_path.exists():
            case_result = read_json(case_path)
            if case_result.get("windows"):
                report["cases"][-1]["completed_windows"] = len(case_result["windows"])
        save(output / "result.json", report)
        print("PTD_IO_CASE", case["id"], status, flush=True)
    measurements = {}
    for case in plan["cases"]:
        path = output / (case["id"] + ".json")
        if not path.exists():
            continue
        result = read_json(path)
        for row in result.get("windows", []):
            key = result["mode"] + ":" + str(result["workers"])
            measurements.setdefault(key, []).append(row)
    summaries = {}
    for key, rows in measurements.items():
        rates = [row["MBps"] for row in rows]
        summaries[key] = dict(samples=len(rates),
                              p10_MBps=percentile(rates, .1),
                              p50_MBps=percentile(rates, .5),
                              min_MBps=min(rates),
                              weighted_MBps=sum(r["read_bytes"] for r in rows) /
                                  1e6 / sum(r["wall_seconds"] for r in rows),
                              passes_800MBps_p10=(len(rates) >= 8 and
                                                  percentile(rates, .1) >= TARGET_MBPS))
    report["summaries"] = summaries
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = ("complete" if all(
        row["status"] == "completed" for row in report["cases"])
        else "partial")
    save(output / "result.json", report)
    print("PTD_IO_FINAL", report["status"], summaries, flush=True)


def _selftest():
    groups = {h: [(f"hop{h}.bin", 3 * 1024 * MIB)] for h in HOPS}
    plan = build_plan(groups)
    assert len(plan["cases"]) == len(SCHEDULE) + 1
    seen = set()
    for case in plan["cases"]:
        for window in case["windows"]:
            assert sum(t["size"] for t in window) == WINDOW
            if case["repeat_region"]:
                continue
            for t in window:
                key = (t["path"], t["offset"], t["size"])
                assert key not in seen
                seen.add(key)
    assert plan["cold_candidate_bytes"] == sum(
        len(c["windows"]) * WINDOW for c in plan["cases"] if not c["repeat_region"])
    print("PTD_IO_SELFTEST_PASS", len(seen), len(plan["cases"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-plan", type=Path)
    parser.add_argument("--case-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _selftest()
    elif args.case_plan:
        if not args.case_output:
            parser.error("--case-output required")
        run_case(read_json(args.case_plan), args.case_output)
    elif args.output:
        run_master(args.output)
    else:
        parser.error("--output required")


if __name__ == "__main__":
    main()
