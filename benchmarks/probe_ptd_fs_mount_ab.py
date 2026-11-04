"""CPU-only Kaggle mount A/B: publish identical synthetic public/private Datasets, then read.

Two independent kernels are mandatory: the reader binds exact Dataset inputs only
after the publisher's creation requests have materialized. Never upload H3 data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

MIB = 1 << 20
SHARDS = 32
SHARD_BYTES = 32 * MIB
CHUNK = 8 * MIB
WINDOW = 64 * MIB
WINDOWS_PER_CASE = 8
PUBLISH_HANDLES = {
    "private": "pamcks/h3-ptd-fsmount-ab-private-20261103",
    "public": "pamcks/h3-ptd-fsmount-ab-public-20261103",
}
FORMAT = "h3-ptd-fsmount-public-private-ab-20261103-v1"


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    i = (len(ordered) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (i - lo)


def assert_cpu():
    if list(Path("/dev").glob("nvidia[0-9]*")):
        raise RuntimeError("CPU-only verification; GPU devices detected")


def prepare_source(root):
    root.mkdir(parents=True, exist_ok=False)
    payload = root / "source"
    payload.mkdir()
    shards = []
    complete = hashlib.sha256()
    for idx in range(SHARDS):
        name = f"rank_{idx:03d}.bin"
        digest = hashlib.sha256()
        with (payload / name).open("xb") as fh:
            for _ in range(SHARD_BYTES // (4 * MIB)):
                blob = os.urandom(4 * MIB)
                fh.write(blob)
                digest.update(blob)
                complete.update(blob)
            fh.flush()
            os.fsync(fh.fileno())
        shards.append({"name": name, "bytes": SHARD_BYTES, "sha256": digest.hexdigest()})
    manifest = {
        "format": FORMAT, "synthetic_only": True,
        "generator": "os.urandom; generated once, hardlinked to two staging trees",
        "bytes": SHARDS * SHARD_BYTES,
        "concatenated_sha256": complete.hexdigest(), "shards": shards,
    }
    save(payload / "manifest.json", manifest)
    return payload, manifest


def staging(root, source, privacy):
    handle = PUBLISH_HANDLES[privacy]
    directory = root / ("stage_" + privacy)
    inner = directory / "payload"
    inner.mkdir(parents=True)
    for f in source.iterdir():
        os.link(f, inner / f.name)
    save(directory / "dataset-metadata.json", {
        "id": handle,
        "title": f"H3 PTD synthetic FS mount A/B {privacy} 20261103",
        "subtitle": "Synthetic opaque bytes for Kaggle filesystem backend validation only",
        "licenses": [{"name": "CC0-1.0"}],
    })
    return directory


def publish(output):
    assert_cpu()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "format": FORMAT, "phase": "publish", "status": "running",
        "cpu_only": True, "gpu_used": False,
        "real_h3_weights_or_activations": False,
        "public_content": "synthetic os.urandom bytes only",
        "target_bytes_per_dataset": SHARDS * SHARD_BYTES,
        "handles": PUBLISH_HANDLES, "requests": {},
        "provider_version_and_remote_byte_integrity": "unverified",
        "no_status_or_list_polling": True,
    }
    save(output / "result.json", result)
    base = Path("/kaggle/temp/h3_ptd_fs_mount_ab_20261103")
    if base.exists():
        raise FileExistsError("Refusing to overwrite existing A/B scratch directory")
    source, manifest = prepare_source(base)
    result["source_manifest_sha256"] = hashlib.sha256(
        (source / "manifest.json").read_bytes()).hexdigest()
    result["source_concatenated_sha256"] = manifest["concatenated_sha256"]
    save(output / "source_manifest.json", manifest)
    save(output / "result.json", result)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    try:
        api.authenticate()  # Notebook's native auth only; never embed local tokens.
    except Exception as exc:
        result.update(status="blocked_auth", auth_error_type=type(exc).__name__)
        save(output / "result.json", result)
        raise RuntimeError("Kaggle notebook-native authentication unavailable") from None
    for privacy in ("private", "public"):
        folder = staging(base, source, privacy)
        started = time.monotonic()
        result["requests"][privacy] = {
            "state": "upload_request_started", "handle": PUBLISH_HANDLES[privacy],
            "public": privacy == "public",
        }
        save(output / "result.json", result)
        try:
            response = api.dataset_create_new(
                str(folder), public=privacy == "public",
                quiet=True, convert_to_csv=False, dir_mode="tar")
            if getattr(response, "error", None):
                raise RuntimeError("Kaggle API returned a nonempty error")
        except Exception as exc:
            result["requests"][privacy].update(
                state="failed", error_type=type(exc).__name__,
                seconds=time.monotonic() - started)
            result["status"] = "partial_or_failed"
            save(output / "result.json", result)
            raise RuntimeError("Dataset create request failed for " + privacy) from None
        result["requests"][privacy].update(
            state="accepted_request_not_readback_proof",
            seconds=time.monotonic() - started)
        save(output / "result.json", result)
    result["status"] = "requests_accepted_reader_pending"
    save(output / "result.json", result)
    print("H3_AB_PUBLISH_REQUESTS_ACCEPTED", list(result["requests"]), flush=True)


def mount_root(handle):
    owner, slug = handle.split("/")
    base = Path("/kaggle/input")
    paths = [base / slug, base / "datasets" / owner / slug,
             base / "datasets" / slug, base / owner / slug]
    found = [p / "payload" for p in paths if (p / "payload" / "manifest.json").is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected unique attached payload {slug}; found {len(found)}")
    return found[0]


def mount_info(root):
    target = str(root.resolve())
    selected = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        parts = left.split()
        mount = parts[4].replace("\\040", " ")
        if target != mount and not target.startswith(mount.rstrip("/") + "/"):
            continue
        info = right.split()
        if selected is None or len(mount) > len(selected["mountpoint"]):
            selected = {
                "mountpoint": mount, "filesystem": info[0],
                "source": info[1] if len(info) > 1 else None,
                "mount_options": parts[5],
                "super_options": info[2] if len(info) > 2 else "",
            }
    if not selected:
        raise RuntimeError("No mountinfo entry for attached Dataset")
    return selected


def memory_stat():
    try:
        values = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
        return {k: int(values[k]) for k in
                ("file", "workingset_refault_file", "pgscan", "pgmajfault") if k in values}
    except (OSError, ValueError):
        return {}


def windows_plan(root, workers):
    begin = 0 if workers == 1 else SHARDS * SHARD_BYTES // 2
    windows = []
    for window in range(WINDOWS_PER_CASE):
        segments = []
        for i in range(WINDOW // CHUNK):
            absolute = begin + window * WINDOW + i * CHUNK
            shard_index, offset = divmod(absolute, SHARD_BYTES)
            assert shard_index < SHARDS
            segments.append((str(root / f"rank_{shard_index:03d}.bin"), offset, CHUNK))
        windows.append(segments)
    return windows


def one_case(root, workers, output):
    windows = windows_plan(root, workers)
    data = {"status": "running", "workers": workers, "windows": []}
    save(output, data)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, tasks in enumerate(windows):
            fds = {path: os.open(path, os.O_RDONLY) for path, _, _ in tasks}
            def one(item):
                path, offset, length = item
                start = time.perf_counter()
                blob = os.pread(fds[path], length, offset)
                if len(blob) != length:
                    raise IOError("Short Dataset read")
                return len(blob), time.perf_counter() - start
            before = memory_stat()
            start = time.perf_counter()
            try:
                done = list(executor.map(one, tasks))
            finally:
                for fd in fds.values():
                    os.close(fd)
            elapsed = time.perf_counter() - start
            after = memory_stat()
            row = {
                "index": index, "read_bytes": sum(x[0] for x in done),
                "wall_seconds": elapsed,
                "MBps": sum(x[0] for x in done) / 1e6 / elapsed,
                "chunk_max_seconds": max(x[1] for x in done),
                "cgroup_delta": {key: after[key] - before[key] for key in before.keys() & after.keys()},
            }
            if row["read_bytes"] != WINDOW:
                raise AssertionError("Window did not read all planned bytes")
            data["windows"].append(row)
            save(output, data)
            print("H3_AB_READ", root.parent.name, workers, index, round(row["MBps"], 2), flush=True)
    rates = [w["MBps"] for w in data["windows"]]
    data.update(
        status="complete", count=len(rates), p10_MBps=percentile(rates, .1),
        p50_MBps=percentile(rates, .5), p90_MBps=percentile(rates, .9),
        min_MBps=min(rates),
        weighted_MBps=sum(w["read_bytes"] for w in data["windows"]) / 1e6 /
            sum(w["wall_seconds"] for w in data["windows"]),
        meets_800_MBps_p10=percentile(rates, .1) >= 800,
    )
    save(output, data)


def read(output):
    assert_cpu()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "format": FORMAT, "phase": "read", "status": "running",
        "target_MBps_p10": 800,
        "caveats": [
            "Mounted file application-layer throughput; not pre-notebook Dataset download.",
            "Unique offset ranges are cold candidates; OS, host, FUSE and remote caches are not ruled out.",
            "One publisher request acknowledgement is not a remote byte integrity proof.",
            "This synthetic 1GiB sample does not establish full 21GiB Hop performance.",
        ], "mounts": {}, "cases": {},
    }
    save(output / "result.json", result)
    roots = {}
    for privacy, handle in PUBLISH_HANDLES.items():
        root = mount_root(handle)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("format") != FORMAT or not manifest.get("synthetic_only"):
            raise ValueError("Attached Dataset is not the expected synthetic fixture")
        if len(manifest.get("shards", [])) != SHARDS:
            raise ValueError("Unexpected synthetic shard inventory")
        if any(s["bytes"] != SHARD_BYTES or
               (root / s["name"]).stat().st_size != SHARD_BYTES
               for s in manifest["shards"]):
            raise ValueError("Synthetic size/inventory mismatch")
        roots[privacy] = root
        result["mounts"][privacy] = {
            "handle": handle, "path": str(root),
            "fs": mount_info(root),
            "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        }
    if result["mounts"]["private"]["manifest_sha256"] != result["mounts"]["public"]["manifest_sha256"]:
        raise ValueError("Public/private synthetic source manifests do not match")
    save(output / "result.json", result)
    # Interleave privacy and concurrency to reduce time-order bias.
    for privacy, workers in (("private", 1), ("public", 1),
                             ("public", 4), ("private", 4)):
        key = f"{privacy}_{workers}"
        path = output / (key + ".json")
        started = time.monotonic()
        try:
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--case-root", str(roots[privacy]), "--workers", str(workers),
                   "--case-output", str(path)]
            proc = subprocess.run(cmd, timeout=210, check=False,
                                  capture_output=True, text=True)
            status = "complete" if proc.returncode == 0 else "failed"
            detail = (proc.stdout + "\n" + proc.stderr)[-1500:]
        except subprocess.TimeoutExpired:
            status, detail = "timeout", "Case hit 210s deadline"
        result["cases"][key] = {
            "status": status, "seconds": time.monotonic() - started,
            "diagnostic": detail,
        }
        if path.exists():
            measurements = json.loads(path.read_text())
            result["cases"][key]["completed_windows"] = len(measurements.get("windows", []))
            if status == "complete":
                result["cases"][key]["p10_MBps"] = measurements["p10_MBps"]
                result["cases"][key]["weighted_MBps"] = measurements["weighted_MBps"]
        save(output / "result.json", result)
    result["status"] = "complete" if all(
        r["status"] == "complete" for r in result["cases"].values()) else "partial"
    save(output / "result.json", result)
    print("H3_AB_READER_FINAL", result["status"], flush=True)


def selftest():
    assert len(PUBLISH_HANDLES) == 2 and SHARDS * SHARD_BYTES == 1024 * MIB
    for workers in (1, 4):
        case = windows_plan(Path("/tmp/synthetic"), workers)
        assert len(case) == WINDOWS_PER_CASE
        ranges = [(path, start, length) for window in case for path, start, length in window]
        assert len(ranges) == len(set(ranges))
        assert sum(x[2] for x in ranges) == 512 * MIB
    assert percentile([1, 2, 3, 4], .1) == 1.3
    print("PTD_FS_MOUNT_AB_SELFTEST_PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("publish", "read"))
    p.add_argument("--output", type=Path)
    p.add_argument("--case-root", type=Path)
    p.add_argument("--workers", type=int, choices=(1, 4))
    p.add_argument("--case-output", type=Path)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        selftest()
    elif a.case_root is not None:
        if not a.case_output or not a.workers:
            p.error("Case mode needs workers and output")
        one_case(a.case_root, a.workers, a.case_output)
    elif a.output and a.phase == "publish":
        publish(a.output)
    elif a.output and a.phase == "read":
        read(a.output)
    else:
        p.error("Select phase/output or self-test")


if __name__ == "__main__":
    main()
