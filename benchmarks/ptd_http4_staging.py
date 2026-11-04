"""Experiment-only 4-worker HTTP producer for PTD's live 2NFE window.

Pre-fetches two immutable Hop parts in the background while physical TP2 consumes
completed files; never does a repeated full-payload SHA256 pass. Actual tensor
I/O reads the local cache and therefore still incurs local disk I/O; this is
HTTP/GPU overlap, not HTTP-to-GPU zero-copy. Applies only to an opt-in trial.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import time

GIB = 1 << 30
SCRATCH = Path("/kaggle/temp/h3_ptd_2nfe_http4_20261103")
RESERVE_BYTES = 4 * GIB
MAX_HOP = 2
WORKERS = 4


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def plan_two_hops(roots, proofs):
    parts = {}
    total = 0
    for hop in range(MAX_HOP):
        part = f"hop_{hop:02d}"
        root = Path(roots[part])
        manifest = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        if manifest.get("hop") != hop or len(manifest.get("shards", {})) != 100:
            raise ValueError("Expected original verified 50-layer TP2 part: " + part)
        heavy = [{"name": manifest["auxiliary"]["file"],
                  "bytes": int(manifest["auxiliary"]["bytes"])}]
        heavy.extend({"name": name + ".safetensors", "bytes": int(meta["bytes"])}
                     for name, meta in sorted(manifest["shards"].items()))
        metadata = ["receipt.json"] + [
            name + ".json" for name in sorted(manifest["shards"])]
        if len(heavy) != 101 or len(metadata) != 101:
            raise ValueError("Expected 202 independently published Hop payload members")
        for item in heavy:
            if item["bytes"] <= 0 or (root / item["name"]).stat().st_size != item["bytes"]:
                raise ValueError("Mounted PTD shard size mismatch: " + part + "/" + item["name"])
        if any(not (root / name).is_file() for name in metadata):
            raise FileNotFoundError("Missing small PTD shard metadata: " + part)
        if part not in proofs:
            raise ValueError("Missing independently qualified part identity")
        total += sum(item["bytes"] for item in heavy)
        parts[part] = {
            "handle": proofs[part]["dataset"], "mounted_root": str(root),
            "heavy": heavy, "metadata": metadata,
            "expected_heavy_bytes": sum(item["bytes"] for item in heavy),
        }
    return parts, total


def stage_two_hops(roots, proofs, result_path, *, workers=WORKERS, scratch=SCRATCH):
    """Return rank/aux roots backed by temporary local files, while retaining 7 other mounts."""
    if workers != WORKERS:
        raise ValueError("Only exact 4-worker HTTP A/B is qualified")
    os.environ["DISABLE_KAGGLE_CACHE"] = "1"
    import kagglehub
    from kagglehub.config import is_kaggle_cache_disabled
    if not is_kaggle_cache_disabled():
        raise RuntimeError("KaggleHub shared-cache bypass not enabled")
    if "output_dir" not in inspect.signature(kagglehub.dataset_download).parameters:
        raise RuntimeError("KaggleHub HTTP output_dir unsupported")
    scratch = Path(scratch)
    parts, expected_total = plan_two_hops(roots, proofs)
    if scratch.exists():
        raise FileExistsError("Refusing to reuse existing HTTP-staging scratch")
    available = shutil.disk_usage(scratch.parent).free
    state = {
        "format": "h3-ptd-2nfe-http4-stage-v1",
        "status": "preflight", "workers_total": workers,
        "network_path": "kagglehub HTTP with DISABLE_KAGGLE_CACHE=1",
        "payload_sha256_repeated": False,
        "metadata_receipt_sha256_separate_from_payload": True,
        "kagglehub_version": importlib.metadata.version("kagglehub"),
        "expected_transfer_bytes": expected_total, "available_before_bytes": available,
        "reserve_bytes": RESERVE_BYTES,
        "staged_parts": {part: {"handle": value["handle"],
                               "expected_heavy_bytes": value["expected_heavy_bytes"]}
                         for part, value in parts.items()},
        "file_events": [], "completed_bytes": 0, "completed_files": 0,
    }
    save(result_path, state)
    if available < int(expected_total * 1.04) + RESERVE_BYTES:
        state["status"] = "blocked_insufficient_scratch"
        save(result_path, state)
        raise OSError("Insufficient scratch for two full local PTD Hops with 4-GiB reserve")

    scratch.mkdir(parents=True)
    parts_root = scratch / "parts"
    downloads = scratch / "inflight"
    parts_root.mkdir()
    downloads.mkdir()
    state["status"] = "running"
    save(result_path, state)

    # Small immutable JSON and receipt metadata are copied directly from
    # independently qualified Dataset mounts. No full byte SHA sweep.
    for part, spec in parts.items():
        root = parts_root / part
        root.mkdir()
        for name in spec["metadata"]:
            src = Path(spec["mounted_root"]) / name
            dest = root / name
            shutil.copyfile(src, dest)
            if src.stat().st_size != dest.stat().st_size:
                raise IOError("Small manifest metadata copy mismatch")
    state["metadata_copied"] = True
    save(result_path, state)

    jobs = [(part, spec, entry)
            for part, spec in parts.items()
            for entry in spec["heavy"]]
    started = time.monotonic()
    state['download_started_monotonic'] = started
    save(result_path, state)

    def transfer(number, part, spec, entry):
        owner = downloads / f"{number:03d}"
        owner.mkdir()
        relative = "payload/" + entry["name"]
        began = time.monotonic()
        downloaded = kagglehub.dataset_download(
            spec["handle"], path=relative, output_dir=str(owner))
        path = Path(downloaded).resolve()
        if os.path.commonpath((str(path), str(owner.resolve()))) != str(owner.resolve()):
            raise RuntimeError("HTTP resolver returned non-owned output path")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("Downloaded PTD shard size differs from proven receipt")
        dest = parts_root / part / entry["name"]
        if dest.exists():
            raise FileExistsError("PTD local stage target already exists")
        os.link(path, dest)
        if dest.stat().st_size != entry["bytes"]:
            raise IOError("Stage hardlink byte size mismatch")
        # A bank reader must never observe a partially downloaded shard.
        ready = dest.with_name(dest.name + '.ready')
        ready.write_text('complete\\n', encoding='ascii')
        # Remove only this run's exact owned transient folder; keep the
        # stage hardlink and preserve all mounted Kaggle Dataset files.
        shutil.rmtree(owner)
        return {"part": part, "file": entry["name"], "bytes": entry["bytes"],
                "seconds": time.monotonic() - began}

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ptd-http4')
    futures = []
    try:
        futures = [pool.submit(transfer, i, part, spec, entry)
                   for i, (part, spec, entry) in enumerate(jobs)]
        # A stalled or nonproductive server must not occupy the Kaggle slot
        # indefinitely. Fail the experiment after 600 seconds of stage time.
        for future in as_completed(futures, timeout=600):
            row = future.result()
            state["completed_files"] += 1
            state["completed_bytes"] += row["bytes"]
            current = state["staged_parts"][row["part"]]
            current["files"] = current.get("files", 0) + 1
            current["bytes"] = current.get("bytes", 0) + row["bytes"]
            state["last_completed"] = row
            save(result_path, state)
    except BaseException as exc:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        state.update(status="failed", error_type=type(exc).__name__,
                     error_message=str(exc)[:320],
                     elapsed_seconds=time.monotonic() - started)
        save(result_path, state)
        raise
    else:
        pool.shutdown(wait=True)

    if state["completed_files"] != len(jobs) or state["completed_bytes"] != expected_total:
        raise RuntimeError("Incomplete HTTP staged dataset accounting")
    for part, spec in parts.items():
        root = parts_root / part
        for entry in spec["heavy"]:
            if (root / entry["name"]).stat().st_size != entry["bytes"]:
                raise RuntimeError("HTTP staged shard disappeared or was truncated")
    elapsed = time.monotonic() - started
    state.update(status="complete", elapsed_seconds=elapsed,
                 download_finished_monotonic=time.monotonic(),
                 completed_files=len(jobs), completed_bytes=expected_total,
                 effective_MBps=expected_total / 1e6 / elapsed,
                 local_filesystem="overlay_or_instance_scratch",
                 counted_throughput="HTTP downloaded file bytes / wall time",
                 note="Asynchronous HTTP/GPU pipeline with local disk cache; no repeated payload SHA256.")
    save(result_path, state)
    result_roots = dict(roots)
    for part in parts:
        result_roots[part] = parts_root / part
    return result_roots, state
