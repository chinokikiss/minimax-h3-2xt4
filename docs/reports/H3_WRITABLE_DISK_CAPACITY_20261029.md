# Kaggle writable disk capacity probe — 2026-10-29

The user requested actual writable-space measurements after PTD's storage
failure. This is a standalone **CPU-only** experiment. It imports only Python's
standard library, installs nothing, loads no H3/core operators, uses no model
weights or credentials, and mounts no datasets or notebook outputs. It does
not qualify a Torch ABI or change the validated H3 environment.

The probe inventories mount topology, byte-level `df`, and visible block-device
sizes. It groups writable directories by `st_dev`: aliases such as `/tmp` and
`/kaggle/temp` are filled once, and cannot be counted as independent disks.
Read-only mounts and memory filesystems are excluded. Visible unmounted devices
are listed, without mounting, formatting or writing their raw device nodes.

For each selected mounted disk, it retains fresh 128-MiB chunks of random data,
requires `fsync`, and reads every chunk back to verify its SHA256. Its memory
buffer is bounded to 8 MiB per payload; all earlier verified chunks remain on
disk until that disk's measurement ends. `fallocate`, `truncate`, sparse files
and endpoint-only writes are not used as capacity proof.

The result records successfully synced-and-verified bytes, failing operation,
errno, unqualified partial-file size, mount state, reported free space before
and at failure, cleanup errors and remaining space. A file whose sync/readback
fails never counts as durable capacity. ENOSPC/EDQUOT are observed boundaries;
EIO or emergency read-only transitions remain writeback faults, rather than an
invented exact quota.

The report filesystem is tested first with 512 MiB of free space retained for
receipts. Root overlay storage is tested last. Each disk has a 512-GiB byte cap
and 1,500-second bound, within a 3,000-second probe budget and 3,150-second
watchdog. A cap, deadline or report-headroom stop is explicitly a **lower bound**,
not the disk's maximum. Another instance, including a T4 instance, can expose
different storage and requires its own admission.

Only exact probe-owned files in the newly created directory are unlinked;
there is no recursive deletion or cleanup of unrelated files. Failure evidence
is written before cleanup. Eight local contracts cover real small writes and
readback, ENOSPC and EIO accounting, fsync propagation, timeout, alias handling,
report-space preservation and unrelated-file preservation. They also run on
Kaggle before any large write.

Sources:

- `benchmarks/kaggle_write_capacity.py`
- `benchmarks/test_kaggle_write_capacity.py`
- `benchmarks/build_kaggle_write_capacity_job.py`

Submission/source receipts will be retained under
`docs/data/h3_writable_disk_capacity_20261029/`. No provider status/list/live-log
polling is used; results are collected separately after completion.

## Submission

Kaggle acknowledged **version 1** of
[H3 Writable Disk Capacity CPU 20261029](https://www.kaggle.com/code/pamcks/h3-writable-disk-capacity-cpu-20261029).
The embedded probe and contracts are bound to immutable source commit
`1a79945db6d0b6ddd3e63191e0547363ee903c45` and individual SHA256 hashes.
GPU acceleration and internet access are disabled; no source/model downloads or
dependency installation are needed by the submitted script. Its watchdog is
3,150 seconds and the provider timeout is 3,300 seconds.

Only the submission acknowledgement has been received. Actual writable-capacity
results are **pending**, with final collection count zero. No scheduling state
or live logs have been queried. Source binding, exact submitted script, metadata
and acknowledgement receipts are archived in the data folder above.

The first receipt's script digest covered LF-normalized script text, while the
Windows file contains CRLF. `cpu_v1_source_verification.json` records both hashes,
verifies the exact archived file and embedded source hashes, and retains the
original receipt. The builder now records file bytes and canonical text hashes
separately. This bookkeeping correction did not alter or resubmit the CPU job.
