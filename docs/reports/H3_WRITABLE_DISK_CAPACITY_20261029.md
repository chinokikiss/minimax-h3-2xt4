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

Final artifacts were collected once and the embedded source bytes and source
binding match the submission. The probe completed in 1,647.97 seconds; the
watchdog exited successfully. All eight CPU contracts passed. No scheduling
state or live logs were queried. Raw results, source, logs, readback hashes and
download hashes are archived under `docs/data/h3_writable_disk_capacity_20261029/cpu_v1_final/`.

The first receipt's script digest covered LF-normalized script text, while the
Windows file contains CRLF. `cpu_v1_source_verification.json` records both hashes,
verifies the exact archived file and embedded source hashes, and retains the
original receipt. The builder now records file bytes and canonical text hashes
separately. This bookkeeping correction did not alter or resubmit the CPU job.

## Actual result on this CPU instance

| Filesystem and representative path | Verified additional retained payload | Stop condition | Other writable aliases |
|---|---:|---|---|
| `/dev/loop2`, ext4, `/kaggle/working` | 19.000 GiB (20,401,094,656 bytes) | Report-space reserve: 514.22 MiB remained | `/kaggle/huggingface`, `/kaggle/nbdev` |
| Root overlay, `/tmp` | 94.000 GiB (100,931,731,456 bytes) | `ENOSPC`, errno 28 on the next chunk | `/var/tmp`, `/root`, `/` |

The 19-GiB result is a deliberately stopped lower bound, not the ext4 maximum:
the filesystem reports 19.518 GiB total and the probe retained its report
reserve. The root-overlay result is an observed writeback boundary with
128-MiB measurement granularity. The failing 128-MiB file is excluded even
though its apparent size reached the requested length. Neither number is an
exact account quota or a reservation for a future instance.

The root overlay advertised about 1,048.81 GiB free before filling, and still
954.69 GiB free when `ENOSPC` occurred. Thus `df` does not measure the writable
capacity available to this container. Changing `/tmp` to `/root` or `/var/tmp`
does not provide another filesystem. No additional independent mounted writable
disk was selected. `/kaggle/temp` was absent in the standalone CPU probe; it was
not qualified as a third storage pool.

All probe-owned files were removed, with zero cleanup errors, and free space
returned to its initial level apart from small report files. This is consistent
with the earlier T4 bake failing near 93 GiB despite much larger advertised free
space, but does not establish identical quotas across CPU and T4 instances.
Planning a roughly 170-GiB full PTD bank from `df` alone remains unsupported;
future full-model work needs a bounded storage plan and instance-specific
admission, rather than another alias of the root overlay.
