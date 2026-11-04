"""CPU-only Kaggle Dataset mounted-FUSE vs direct-HTTP throughput A/B.

No GPU, no Dataset mutations, no production Bank changes, no full 21GB Hop
downloads. HTTP cases use fresh local output directories and are subprocess-
bounded, with early-stop of higher worker counts on timeout/failure/collapse.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

try:
    from benchmarks import probe_ptd_fs_mount_max_v2 as mounted
except ModuleNotFoundError:
    import probe_ptd_fs_mount_max_v2 as mounted

MIB = 1 << 20
FORMAT = "h3-ptd-http-direct-vs-gcsfuse-cpu-v1"
WORKERS = (1, 4, 8, 16)
HTTP_DEADLINES = {1: 35, 4: 40, 8: 45, 16: 55}
SYNTH_TIMEOUT = 25
MOUNT_TIMEOUT = 35
TOTAL_BUDGET = 320
SCRATCH_ROOT = Path("/kaggle/temp/h3_ptd_http_direct_ab_20261103")
MIN_FREE_MARGIN = 512 * MIB


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    x = (len(ordered) - 1) * fraction
    low, high = math.floor(x), math.ceil(x)
    return ordered[low] + (ordered[high] - ordered[low]) * (x - low)


def network_rx_bytes():
    """Diagnostic interface RX, not an exact Kaggle HTTP payload counter."""
    counts = {}
    try:
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            name, columns = line.split(":", 1)
            name = name.strip()
            if name == "lo":
                continue
            values = columns.split()
            if len(values) >= 1:
                counts[name] = int(values[0])
    except (OSError, ValueError):
        pass
    return sum(counts.values())


def sample_network(stop, samples):
    while not stop.is_set():
        samples.append((time.monotonic(), network_rx_bytes()))
        stop.wait(1.0)
    samples.append((time.monotonic(), network_rx_bytes()))


def summarize_network(samples):
    rates = [(b[1] - a[1]) / 1e6 / (b[0] - a[0])
             for a, b in zip(samples, samples[1:])
             if b[0] > a[0] and b[1] >= a[1]]
    return {
        "scope": "all_non_loopback_interface_rx_including_HTTP_control_and_other_traffic",
        "seconds_sampled": len(rates),
        "rx_MB": (samples[-1][1] - samples[0][1]) / 1e6 if len(samples) > 1 else 0,
        "p10_MBps": percentile(rates, 0.1) if len(rates) >= 8 else None,
        "p50_MBps": percentile(rates, 0.5) if len(rates) >= 8 else None,
        "min_MBps": min(rates) if rates else None,
        "rates_MBps": rates,
        "p10_qualified": len(rates) >= 8,
    }


def case_plan(tasks, workers, case_id):
    return {
        "id": case_id,
        "workers": workers,
        "expected_bytes": sum(item["bytes"] for item in tasks),
        "tasks": tasks,
    }


def select_real_file_tasks(real):
    """Round-robin files across 6 Hops and 2 TP ranks; never repeat byte files."""
    candidates = []
    for index in range(50):
        for hop in mounted.HOPS:
            for rank in (0, 1):
                files = real[(hop, rank)]
                if index >= len(files):
                    continue
                name, size = files[index]
                if 32 * MIB <= size <= 512 * MIB:
                    candidates.append({
                        "handle": mounted.HOP_HANDLES[hop],
                        "dataset_path": "payload/" + Path(name).name,
                        "source_path": name,
                        "bytes": size,
                        "hop": hop, "rank": rank,
                    })
    if len(candidates) < sum(max(2, n) for n in WORKERS):
        raise RuntimeError("Insufficient real PTD rank files for disjoint HTTP cases")
    plans, start = [], 0
    for workers in WORKERS:
        count = max(2, workers)
        tasks = candidates[start:start + count]
        start += count
        plans.append(case_plan(tasks, workers, f"real_http_{workers:02d}"))
    return plans


def select_synthetic_tasks(synthetic):
    cases = []
    for privacy in ("public", "private"):
        name, size = synthetic[privacy][0]
        cases.append(case_plan([{
            "handle": mounted.PUBLIC_PRIVATE[privacy],
            "dataset_path": "payload/" + Path(name).name,
            "source_path": name,
            "bytes": size,
        }], 1, "synthetic_http_" + privacy))
    return cases


def early_stop_reason(summary, best_before, weak_streak):
    if summary["status"] == "timeout":
        return "hard_timeout_skip_higher_workers"
    if summary["status"] != "complete":
        return "failed_or_auth_blocked_skip_higher_workers"
    if best_before is not None and weak_streak >= 2:
        return "two_consecutive_below_30pct_of_best"
    return None


def ensure_local_path(path, scratch):
    actual = Path(path).resolve()
    parent = scratch.resolve()
    if os.path.commonpath([str(actual), str(parent)]) != str(parent):
        raise RuntimeError("HTTP resolver did not return the explicitly owned scratch path")
    fs = mounted.mount_type(actual)
    if fs["filesystem"] in ("fuse.gcsfuse", "nfs", "nfs4"):
        raise RuntimeError("HTTP resolver still returned a network-mounted path")
    return actual, fs


def one_http(case, result_path, scratch):
    """Download one file per task, in unique paths, with a hard parent deadline."""
    mounted.assert_cpu()
    os.environ["DISABLE_KAGGLE_CACHE"] = "1"
    import kagglehub
    from kagglehub.config import is_kaggle_cache_disabled
    if not is_kaggle_cache_disabled():
        raise RuntimeError("Kaggle shared-cache disable flag is ineffective")
    if "output_dir" not in inspect.signature(kagglehub.dataset_download).parameters:
        raise RuntimeError("Installed kagglehub lacks output_dir; need >=0.4.3")
    scratch = Path(scratch)
    if scratch.exists():
        raise FileExistsError("Refusing to reuse an existing HTTP trial scratch path")
    scratch.mkdir(parents=True)
    observations = []
    state = {
        "status": "running", "case_id": case["id"],
        "workers": case["workers"],
        "kagglehub_version": importlib.metadata.version("kagglehub"),
        "disable_kaggle_cache": True, "output_dir_active": True,
        "scratch": str(scratch),
        "expected_bytes": case["expected_bytes"], "finished": observations,
        "http_transport_not_gcsfuse_mount": True,
    }
    save(result_path, state)
    free = shutil.disk_usage(scratch).free
    if free < int(case["expected_bytes"] * 1.2) + MIN_FREE_MARGIN:
        state.update(status="blocked_space", available_bytes=free)
        save(result_path, state)
        return

    def download_one(number, task):
        directory = scratch / f"file_{number:02d}"
        directory.mkdir()
        began = time.monotonic()
        local = kagglehub.dataset_download(
            task["handle"], path=task["dataset_path"], output_dir=str(directory))
        actual, fs = ensure_local_path(local, directory)
        measured_bytes = actual.stat().st_size
        if measured_bytes != task["bytes"]:
            raise ValueError("HTTP file bytes differ from mounted receipt/stat")
        return {
            "task": number, "handle": task["handle"],
            "path": task["dataset_path"], "source_bytes": measured_bytes,
            "seconds": time.monotonic() - began,
            "local_path": str(actual), "filesystem": fs["filesystem"],
        }

    samples = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_network, args=(stop, samples), daemon=True)
    sampler.start()
    begin = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=case["workers"]) as pool:
            futures = [pool.submit(download_one, n, item)
                       for n, item in enumerate(case["tasks"])]
            for future in as_completed(futures):
                row = future.result()
                observations.append(row)
                save(result_path, state)
    finally:
        stop.set()
        sampler.join(timeout=2)
    elapsed = time.monotonic() - begin
    transferred = sum(row["source_bytes"] for row in observations)
    if len(observations) != len(case["tasks"]) or transferred != case["expected_bytes"]:
        raise AssertionError("Incomplete HTTP download accounting")
    state.update(
        status="complete", elapsed_seconds=elapsed,
        completed_bytes=transferred,
        effective_MBps=transferred / 1e6 / elapsed,
        network_diagnostic=summarize_network(samples),
    )
    save(result_path, state)
    # Actual instance-disk read check after HTTP, separate from HTTP timing.
    def read_local(row):
        size = min(128 * MIB // len(observations), row["source_bytes"])
        with open(row["local_path"], "rb", buffering=0) as handle:
            blob = os.pread(handle.fileno(), size, 0)
        if len(blob) != size:
            raise IOError("Short local read")
        return size
    for workers in (1, min(4, len(observations)), min(8, len(observations))):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            read_bytes = sum(pool.map(read_local, observations))
        duration = time.monotonic() - started
        state.setdefault("local_read_tests", []).append({
            "workers": workers, "bytes": read_bytes,
            "seconds": duration, "MBps": read_bytes / 1e6 / duration,
            "cache_caveat": "May hit OS page cache immediately after download",
        })
        save(result_path, state)


def one_mounted(plan, result_path):
    mounted.case_worker(plan, result_path)


def run_case(script_path, case, output, kind, timeout, scratch=None):
    plan_file = output / (case["id"] + "_plan.json")
    case_file = output / (case["id"] + ".json")
    save(plan_file, case)
    command = [sys.executable, "-u", str(script_path),
               "--case", str(plan_file), "--case-output", str(case_file),
               "--kind", kind]
    if scratch is not None:
        command += ["--scratch", str(scratch)]
    began = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, check=False)
        status = "complete" if result.returncode == 0 else "failed"
        diagnostic = (result.stdout + "\n" + result.stderr)[-1800:]
    except subprocess.TimeoutExpired:
        status = "timeout"
        diagnostic = f"Hard subprocess timeout {timeout}s; child download threads terminated"
    summary = {
        "id": case["id"], "kind": kind, "workers": case["workers"],
        "status": status, "elapsed_seconds": time.monotonic() - began,
        "deadline_seconds": timeout, "diagnostic_tail": diagnostic,
    }
    if case_file.is_file():
        data = json.loads(case_file.read_text())
        summary["partial_case_status"] = data["status"]
        summary["completed_files"] = len(data.get("finished", []))
        if data.get("status") != "complete" and status == "complete":
            status = data["status"]
            summary["status"] = status
        for key in ("effective_MBps", "completed_bytes", "network_diagnostic",
                    "local_read_tests", "elapsed_seconds"):
            if key in data:
                summary[key] = data[key]
        if kind == "mounted":
            windows = data.get("windows", [])
            if windows:
                summary["completed_windows"] = len(windows)
                summary["mounted_weighted_MBps"] = (
                    sum(w["read_bytes"] for w in windows) /
                    1e6 / sum(w["wall_seconds"] for w in windows))
    if scratch is not None and scratch.exists():
        # Only this run's unique scratch case directory, never mounted datasets.
        safe_parent = SCRATCH_ROOT.resolve()
        target = scratch.resolve()
        if target.parent != safe_parent:
            raise RuntimeError("Refusing cleanup outside own scratch root")
        shutil.rmtree(target)
        summary["owned_scratch_reclaimed"] = True
    return summary


def master(out):
    mounted.assert_cpu()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    state = {
        "format": FORMAT, "status": "starting",
        "cpu_only": True, "gpu_used": False, "dataset_mutations": False,
        "Kaggle_HTTP_direct_vs_mounted_read": True,
        "no_kaggle_status_list_live_log_polling": True,
        "scope": [
            "GCSFuse baseline: application os.pread of mounted model weights",
            "HTTP: actual downloaded output file bytes divided by case walltime",
            "Network RX 1s samples include other traffic and are diagnostic only",
            "Local read immediately after HTTP may hit page cache and is not a cold NVMe test",
        ],
        "deadlines_seconds": {
            "mounted": MOUNT_TIMEOUT, "synthetic": SYNTH_TIMEOUT,
            "real": HTTP_DEADLINES, "outer": 420, "measurement_budget": TOTAL_BUDGET,
        },
        "early_stop": None, "cases": [], "mounts": {},
    }
    save(out / "result.json", state)
    began = time.monotonic()
    synthetic = {}
    try:
        synthetic, synth_mounts = mounted.synthetic_sources()
        state["mounts"]["synthetic"] = synth_mounts
    except Exception as error:
        state["optional_synthetic_unavailable"] = {
            "error_type": type(error).__name__, "detail": str(error)[:320]}
    try:
        real, real_mounts = mounted.hop_sources()
        state["mounts"]["real"] = real_mounts
    except Exception as error:
        state.update(status="blocked_real_hop_mounts", error_type=type(error).__name__,
                     error_detail=str(error)[:320])
        save(out / "result.json", state)
        return
    if SCRATCH_ROOT.exists():
        state.update(status="blocked_unsafe_scratch_reuse")
        save(out / "result.json", state)
        return
    SCRATCH_ROOT.mkdir(parents=True)
    state["scratch_root"] = str(SCRATCH_ROOT)
    save(out / "result.json", state)
    script = Path(__file__).resolve()
    fake_synth = {privacy: [("/unused", 1024*MIB)] for privacy in mounted.PUBLIC_PRIVATE}
    mounted_plan = next(c for c in mounted.build_cases(fake_synth, real)
                        if c["id"] == "ptd_pilot_01")
    mounted_plan = dict(mounted_plan, id="real_mounted_01")
    plans = [("mounted", mounted_plan, MOUNT_TIMEOUT)]
    if synthetic:
        plans += [("http", c, SYNTH_TIMEOUT) for c in select_synthetic_tasks(synthetic)]
    plans += [("http", c, HTTP_DEADLINES[c["workers"]]) for c in select_real_file_tasks(real)]
    best = None
    weak_streak = 0
    skip_higher = None
    for kind, case, limit in plans:
        is_real = case["id"].startswith("real_http")
        if time.monotonic() - began + limit > TOTAL_BUDGET:
            summary = {"id": case["id"], "status": "skipped_total_budget"}
        elif is_real and skip_higher:
            summary = {"id": case["id"], "status": "skipped_early_stop",
                       "reason": skip_higher}
        else:
            scratch = SCRATCH_ROOT / case["id"] if kind == "http" else None
            best_before = best
            try:
                summary = run_case(script, case, out, kind, limit, scratch)
            except Exception as error:
                summary = {"id": case["id"], "status": "failed_supervisor",
                           "error_type": type(error).__name__,
                           "error_detail": str(error)[:320]}
            if is_real:
                if summary["status"] == "complete" and summary.get("effective_MBps"):
                    if best_before is not None and summary["effective_MBps"] < best_before * 0.30:
                        weak_streak += 1
                    else:
                        weak_streak = 0
                    best = max(best or 0.0, summary["effective_MBps"])
                reason = early_stop_reason(summary, best_before, weak_streak)
                if reason:
                    skip_higher = reason
                    state["early_stop"] = {"reason": reason, "at_workers": case["workers"]}
        state["cases"].append(summary)
        save(out / "result.json", state)
        print("H3_PTD_HTTP_DIRECT_CASE", case["id"], summary["status"], flush=True)
    state["elapsed_seconds"] = time.monotonic() - began
    state["best_complete_real_http_MBps"] = best
    state["status"] = ("complete_with_early_stop" if skip_higher
                       else "complete_or_partial")
    save(out / "result.json", state)
    print("H3_PTD_HTTP_DIRECT_DONE", state["status"], flush=True)


def selftest():
    real = {(hop, rank): [(f"hop{hop:02d}_rank{rank}_{i}.bin", 128*MIB)
                         for i in range(50)]
            for hop in mounted.HOPS for rank in (0, 1)}
    plans = select_real_file_tasks(real)
    assert [p["workers"] for p in plans] == list(WORKERS)
    assert len({(t["handle"], t["dataset_path"]) for p in plans for t in p["tasks"]}) == 30
    assert [len(p["tasks"]) for p in plans] == [2, 4, 8, 16]
    assert early_stop_reason({"status": "timeout"}, None, 0).startswith("hard_timeout")
    assert early_stop_reason({"status": "failed"}, None, 0).startswith("failed")
    assert early_stop_reason({"status": "complete"}, 100, 2) == "two_consecutive_below_30pct_of_best"
    assert early_stop_reason({"status": "complete"}, 100, 1) is None
    assert summarize_network([(0.0, 0), (1.0, 2000000)])["p10_MBps"] is None
    print("H3_PTD_HTTP_DIRECT_SELFTEST_PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--case-output", type=Path)
    parser.add_argument("--kind", choices=("mounted", "http"))
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    elif args.case:
        if not args.case_output or not args.kind:
            parser.error("Case requires output and kind")
        case = json.loads(args.case.read_text())
        if args.kind == "http":
            if args.scratch is None:
                parser.error("HTTP case requires scratch")
            one_http(case, args.case_output, args.scratch)
        else:
            one_mounted(case, args.case_output)
    elif args.output:
        master(args.output)
    else:
        parser.error("Need output, case, or self-test")


if __name__ == "__main__":
    main()
