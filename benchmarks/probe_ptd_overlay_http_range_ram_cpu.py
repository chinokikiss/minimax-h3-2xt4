"""Single-overlay, CPU-only Kaggle Dataset HTTP Range -> one RAM buffer A/B.

All arms reconstruct the exact same 1,544,161,056-byte safetensors file.
HTTP arms use concurrent disjoint byte ranges against one signed raw object URL;
mounted arm uses one os.preadv reader and tensor-aligned payload spans.
No full-file HTTP disk staging, GPU, dataset mutation, or automatic retry.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse

MIB = 1 << 20
CHUNK = 8 * MIB
SOURCE_HANDLE = "pamcks/h3-ptd-v2-hop-00-20261103"
DATASET_VERSION = 1
RELATIVE_PATH = "payload/aux_overlay.safetensors"
EXPECTED_BYTES = 1_544_161_056
RANGE_WORKERS = (1, 2, 4, 8)
FORMAT = "h3-ptd-overlay-single-file-http-range-ram-v1"
ROUND_ORDER = (1, 2, 4, 8, 8, 4, 2, 1)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_error(exc):
    message = re.sub(r"https?://[^\s'\"<>]+", "<redacted-url>", str(exc))
    message = re.sub(r"(?i)(token|signature|credential|authorization)=[^\s&]+",
                     r"\1=<redacted>", message)
    return {"type": type(exc).__name__, "message": message[:800]}


def ranges_for(total, workers):
    if not isinstance(workers, int) or not 1 <= workers <= 8 or total < workers:
        raise ValueError("Invalid Range partition")
    result = [(total * i // workers, total * (i + 1) // workers)
              for i in range(workers)]
    if result[0][0] != 0 or result[-1][1] != total:
        raise AssertionError("Incomplete Range partition")
    if any(b <= a or (i and a != result[i - 1][1])
           for i, (a, b) in enumerate(result)):
        raise AssertionError("Overlapping/gapped Range partition")
    return result


def check_response(response, start, end, total):
    expected = "bytes {}-{}/{}".format(start, end - 1, total)
    if response.status_code != 206:
        raise ValueError("Range must return HTTP 206, got " + str(response.status_code))
    if response.headers.get("Content-Range") != expected:
        raise ValueError("Content-Range mismatch, expected " + expected +
                         ", received " + str(response.headers.get("Content-Range")))
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != end - start:
        raise ValueError("Content-Length disagrees with requested Range")
    encoding = response.headers.get("Content-Encoding")
    if encoding and encoding.lower() != "identity":
        raise ValueError("Compressed HTTP transfer cannot directly fill raw buffer")


def source_record():
    import probe_ptd_preadv_stripe_cpu as mounted
    root = mounted.locate(SOURCE_HANDLE)
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    aux = receipt["auxiliary"]
    if receipt.get("hop") != 0 or aux.get("file") != "aux_overlay.safetensors":
        raise ValueError("Wrong PTD Hop00 auxiliary receipt")
    path = root / aux["file"]
    byte_count = int(aux["bytes"])
    if byte_count != EXPECTED_BYTES or path.stat().st_size != byte_count:
        raise ValueError("Mounted Overlay size differs from frozen experiment")
    sha256 = mounted.checksum(aux["sha256"])
    header, payload_bytes, tensor_ranges, signature = mounted.safetensors_plan(path)
    if len(header) + payload_bytes != byte_count:
        raise ValueError("Mounted safetensors does not cover the file")
    return {"handle": SOURCE_HANDLE, "version": DATASET_VERSION,
            "relative_path": RELATIVE_PATH, "mount": str(path),
            "bytes": byte_count, "sha256": sha256,
            "header_bytes": len(header), "header": header,
            "tensor_ranges": tensor_ranges, "signature": signature}


def resolve_raw_file_url():
    """Use Kaggle's authenticated redirect; never persist or print its signed URL."""
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    request = ApiDownloadDatasetRequest()
    request.owner_slug = "pamcks"
    request.dataset_slug = "h3-ptd-v2-hop-00-20261103"
    request.file_name = RELATIVE_PATH
    request.dataset_version_number = DATASET_VERSION
    request.raw = True
    started = time.perf_counter()
    response = build_kaggle_client().datasets.dataset_api_client.download_dataset(request)
    try:
        response.raise_for_status()
        url = response.url
        parts = urlparse(url)
        if parts.scheme != "https" or not parts.netloc:
            raise ValueError("Kaggle did not return an HTTPS object URL")
        if parts.path.lower().endswith((".zip", ".tar", ".tar.gz")):
            raise ValueError("Kaggle returned a compressed archive, not raw Overlay")
        result = {"resolve_seconds": time.perf_counter() - started,
                  "response_status": response.status_code,
                  "raw_flag": True, "object_host": parts.hostname,
                  "api_version": DATASET_VERSION}
        return url, result
    finally:
        response.close()


def http_headers(start, end):
    return {"Range": "bytes={}-{}".format(start, end - 1),
            "Accept-Encoding": "identity", "Cache-Control": "no-transform"}


def probe_raw_range(url, source):
    import requests
    count = min(4096, source["bytes"])
    with requests.get(url, headers=http_headers(0, count), stream=True,
                      timeout=(15, 90)) as response:
        check_response(response, 0, count, source["bytes"])
        response.raw.decode_content = False
        data = response.raw.read(count)
        if len(data) != count:
            raise IOError("Short Range source probe")
    with open(source["mount"], "rb") as stream:
        expected = stream.read(count)
    if data != expected:
        raise ValueError("HTTP Range is not the mounted raw safetensors object")
    return {"bytes_checked": count, "status": 206, "matches_mounted_prefix": True}


def download_ranges_to_ram(url, target, total, workers):
    """No HTTP file or full temporary bytes object; workers write disjoint memoryview spans."""
    import requests
    plan = ranges_for(total, workers)

    def one(index, start, end):
        started = time.perf_counter()
        with requests.get(url, headers=http_headers(start, end), stream=True,
                          timeout=(15, 90)) as response:
            check_response(response, start, end, total)
            response.raw.decode_content = False
            pos = start
            while pos < end:
                section = target[pos:min(pos + CHUNK, end)]
                try:
                    got = response.raw.readinto(section)
                finally:
                    section.release()
                if not got:
                    raise IOError("Premature HTTP Range EOF")
                pos += got
            if pos != end:
                raise IOError("Incomplete HTTP Range transfer")
        return {"index": index, "start": start, "end": end,
                "bytes": pos - start, "seconds": time.perf_counter() - started}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="range-to-ram") as pool:
        futures = [pool.submit(one, index, a, b)
                   for index, (a, b) in enumerate(plan)]
        pieces = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - started
    if sum(piece["bytes"] for piece in pieces) != total:
        raise IOError("HTTP Range completed-byte accounting mismatch")
    return elapsed, sorted(pieces, key=lambda item: item["index"])


def mounted_tensor_preadv(source, target):
    """One Linux reader, safetensors tensor spans, direct writes to final RAM."""
    if not hasattr(os, "preadv"):
        raise RuntimeError("Linux os.preadv is required for mounted arm")
    path = source["mount"]
    header = source["header"]
    started = time.perf_counter()
    target[:len(header)] = header
    fd = os.open(path, os.O_RDONLY)
    try:
        for start, end, _ in source["tensor_ranges"]:
            pos = start + len(header)
            stop = end + len(header)
            while pos < stop:
                section = target[pos:min(pos + CHUNK, stop)]
                try:
                    got = os.preadv(fd, [section], pos)
                finally:
                    section.release()
                if got <= 0:
                    raise IOError("Mounted preadv returned short data")
                pos += got
    finally:
        os.close(fd)
    elapsed = time.perf_counter() - started
    stat = os.stat(path)
    if (stat.st_size, stat.st_mtime_ns) != tuple(source["signature"]):
        raise RuntimeError("Mounted source mutated during the experiment")
    return elapsed, {"tensor_count": len(source["tensor_ranges"]),
                     "payload_bytes": source["bytes"] - len(header),
                     "header_bytes": len(header), "preadv_workers": 1}


def checksum_ram(target, expected):
    started = time.perf_counter()
    actual = hashlib.sha256(target).hexdigest()
    seconds = time.perf_counter() - started
    if actual != expected:
        raise ValueError("Full-file RAM SHA256 differs from mounted published receipt")
    return seconds


def memory_snapshot():
    result = {}
    try:
        rows = Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
        values = dict((key, int(value)) for key, value in (line.split() for line in rows))
        result = {key: values[key] for key in ("anon", "file", "pgmajfault",
                                                "workingset_refault_file") if key in values}
    except (OSError, ValueError):
        pass
    return result


def case_summary(method, workers, transfer, hash_time, total, details, order):
    return {"order": order, "method": method, "workers": workers,
            "status": "complete", "bytes": total, "full_sha256_valid": True,
            "transfer_seconds": transfer, "hash_seconds": hash_time,
            "verified_ram_ready_seconds": transfer + hash_time,
            "transfer_MBps": total / 1e6 / transfer,
            "verified_ram_MBps": total / 1e6 / (transfer + hash_time),
            "details": details}


def aggregate(cases, size):
    by_method = {}
    for row in cases:
        if row.get("status") != "complete":
            continue
        key = row["method"] + "_" + str(row["workers"])
        bucket = by_method.setdefault(key, {"cases": 0, "total_bytes": 0,
                                              "transfer_seconds": 0.0,
                                              "verified_ram_ready_seconds": 0.0})
        bucket["cases"] += 1
        bucket["total_bytes"] += size
        bucket["transfer_seconds"] += row["transfer_seconds"]
        bucket["verified_ram_ready_seconds"] += row["verified_ram_ready_seconds"]
    for value in by_method.values():
        value["transfer_MBps"] = value["total_bytes"] / 1e6 / value["transfer_seconds"]
        value["verified_ram_MBps"] = value["total_bytes"] / 1e6 / value["verified_ram_ready_seconds"]
    return by_method


def experiment(output):
    output = Path(output)
    started = time.perf_counter()
    state = {"format": FORMAT, "status": "preflight", "cpu_only": True,
             "gpu_used": False, "source_dataset": SOURCE_HANDLE,
             "dataset_version": DATASET_VERSION, "source_relative_path": RELATIVE_PATH,
             "http_disk_staging": False, "single_preallocated_ram_buffer": True,
             "automatic_retry": False, "order": [],
             "cases": [], "source_identity": "mounted receipt + exact version raw HTTP"}
    save(output / "result.json", state)
    source = source_record()
    state.update(file_bytes=source["bytes"], expected_sha256=source["sha256"],
                 mount_path=source["mount"], original_mount_signature=source["signature"],
                 memory_before=memory_snapshot())
    save(output / "result.json", state)
    allocation_started = time.perf_counter()
    final_bytes = bytearray(source["bytes"])
    view = memoryview(final_bytes)
    state["preallocation_seconds"] = time.perf_counter() - allocation_started
    state["memory_after_allocation"] = memory_snapshot()
    url, redirect = resolve_raw_file_url()
    state["raw_url_resolution"] = redirect
    state["range_probe"] = probe_raw_range(url, source)
    save(output / "result.json", state)
    state["status"] = "running"
    order = [("mounted_preadv", 1)] + [("http_range_ram", w) for w in ROUND_ORDER] + [("mounted_preadv", 1)]
    state["order"] = [method + "_" + str(workers) for method, workers in order]
    for position, (method, workers) in enumerate(order):
        started_case = time.perf_counter()
        try:
            if method == "mounted_preadv":
                transfer, details = mounted_tensor_preadv(source, view)
            else:
                transfer, pieces = download_ranges_to_ram(url, view, source["bytes"], workers)
                details = {"ranges": pieces, "full_disk_files": 0}
            hash_time = checksum_ram(view, source["sha256"])
            row = case_summary(method, workers, transfer, hash_time, source["bytes"],
                               details, position)
        except BaseException as exc:
            row = {"order": position, "method": method, "workers": workers,
                   "status": "failed", "case_wall_seconds": time.perf_counter() - started_case,
                   "error": safe_error(exc)}
        state["cases"].append(row)
        state["aggregate"] = aggregate(state["cases"], source["bytes"])
        save(output / "result.json", state)
        print("OVERLAY_RANGE_RAM_CASE", position, method, workers,
              row["status"], round(row.get("verified_ram_MBps", 0), 3), flush=True)
    view.release()
    del final_bytes
    state["status"] = ("complete" if all(x["status"] == "complete" for x in state["cases"])
                       else "complete_with_failed_cases")
    state["elapsed_seconds"] = time.perf_counter() - started
    state["memory_after"] = memory_snapshot()
    save(output / "result.json", state)
    print("OVERLAY_RANGE_RAM_FINISHED", state["status"], flush=True)


def selftest():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    payload = bytes((i * 31 + 9) % 251 for i in range(131_117))
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            header = self.headers.get("Range", "")
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", header)
            if not match:
                self.send_error(400)
                return
            a, b = map(int, match.groups())
            if a > b or b >= len(payload):
                self.send_error(416)
                return
            data = payload[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Range", "bytes {}-{}/{}".format(a, b, len(payload)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = "http://127.0.0.1:{}/test".format(server.server_address[1])
        target = memoryview(bytearray(len(payload)))
        for workers in RANGE_WORKERS:
            elapsed, pieces = download_ranges_to_ram(url, target, len(payload), workers)
            if not elapsed > 0 or len(pieces) != workers or bytes(target) != payload:
                raise AssertionError("HTTP Range direct-RAM selftest failed")
        target.release()
        class Bad:
            status_code = 200
            headers = {"Content-Range": "bytes 0-9/10"}
        try:
            check_response(Bad(), 0, 10, 10)
        except ValueError:
            pass
        else:
            raise AssertionError("HTTP 200 was incorrectly accepted")
        for workers in RANGE_WORKERS:
            p = ranges_for(1_544_161_056, workers)
            if sum(b - a for a, b in p) != 1_544_161_056:
                raise AssertionError("Range coverage mismatch")
    finally:
        server.shutdown()
        server.server_close()
    print("OVERLAY_RANGE_RAM_SELFTEST_PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    elif args.output:
        try:
            experiment(args.output)
        except BaseException as exc:
            result_file = args.output / "result.json"
            state = (json.loads(result_file.read_text(encoding="utf-8"))
                     if result_file.exists() else {"format": FORMAT})
            state.update(status="blocked_or_failed", fatal_error=safe_error(exc))
            save(result_file, state)
            print("OVERLAY_RANGE_RAM_BLOCKED", state["fatal_error"], flush=True)
    else:
        parser.error("Specify --output or --self-test")


if __name__ == "__main__":
    main()
