# H3 PTD Overlay — HTTP Range ×1/2/4/8 to RAM vs tensor_preadv ×1 (2026-11-04)

Status: **private Kaggle CPU Kernel v1 accepted; execution and measurements pending**.

- Kernel: https://www.kaggle.com/code/pamcks/h3-ptd-overlay-range-ram-cpu-20261104
- Provider ACK: `Kernel version 1 successfully pushed.` (CLI exit 0).
- One immutable published source only: `pamcks/h3-ptd-v2-hop-00-20261103` version 1, `payload/aux_overlay.safetensors`, 1,544,161,056 bytes.
- No T4, no full HTTP file written to disk, no Dataset modification, no production PTD Runtime edits.

## Exact experiment

The CPU process preallocates **one full-size bytearray** and reuses its memoryview in every case. Every HTTP worker receives a disjoint byte range of the same source object, with `Range: bytes=start-end`, directly calls `response.raw.readinto(memoryview slice)`, and verifies HTTP **206**, exact `Content-Range`, valid content length, and identity transfer encoding. A Kaggle SDK `raw=True` request obtains an authenticated redirect URL in memory; no signed URL or credential is recorded. A 4-KiB Range probe must match the mounted raw safetensors prefix before any full case.

The comparison arm uses one Linux `os.preadv` reader over safetensors tensor-aligned spans of the mounted **same file**, copied into the same preallocated final RAM buffer. No O_DIRECT or drop_caches is used.

Order, including reversed HTTP-order second pass:

1. Mounted preadv1.
2. HTTP Range1, Range2, Range4, Range8.
3. HTTP Range8, Range4, Range2, Range1.
4. Mounted preadv1.

Every completed case runs **full-file SHA256** on the in-RAM bytearray, compared with the original immutable Hop00 receipt. Recorded per case: transfer wall seconds and MB/s; SHA256 seconds; verified RAM-ready wall seconds and MB/s; exact range byte accounting; one-run memory snapshots. Aggregate figures weight by total processed bytes and elapsed time rather than unweighted mean speed.

## Validation and evidence

- Local actual HTTP Range-to-RAM test with a threaded range-capable server: **PASS** for workers 1/2/4/8.
- HTTP 200 (Range ignored) rejection and range coverage checks: **PASS**.
- Frozen two-source SHA256 and Notebook AST/metadata validation: **PASS**.
- Notebook SHA256: `ed6f85c5db0424ce4b4c6cb77481bc374d5392e35ec1464d9ca20c823e9f3def`.
- Builder: `benchmarks/build_ptd_overlay_http_range_ram_cpu_job.py`.
- Probe: `benchmarks/probe_ptd_overlay_http_range_ram_cpu.py`.
- Frozen candidate: `docs/data/h3_ptd_overlay_range_ram_cpu_20261104/job_v1_local/`.
- Submission receipt: `docs/data/h3_ptd_overlay_range_ram_cpu_20261104/submission_v1.json`.

Actual metrics will be at `/kaggle/working/h3_ptd_overlay_range_ram_cpu_20261104/measurement/result.json` when the Kernel completes. Signed URL validity, raw-object Range support, network throughput and CPU memory availability are **not yet validated on Kaggle**. Failure cases and preflight blockers are recorded without falling back to disk or declaring numeric performance.

No Kaggle status/list/live-log polling or automatic submission retries. A push ACK is not experimental success.
