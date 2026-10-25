# Kaggle mounted storage read probe — 2026-10-24

One private CPU-only Kaggle job completed with status **complete**. It ran for 144.53 seconds inside the benchmark process. The frozen ComfyUI source identity and four tiny CPU contract tests passed. No package was installed, no H3 component/model was executed, and no GPU was requested.

Job: [pamcks/h3-kaggle-storage-read-probe-2026-10-24, version 2](https://www.kaggle.com/code/pamcks/h3-kaggle-storage-read-probe-2026-10-24). Version 1 stopped before benchmarking because the bootstrap omitted the separate frozen source extraction. Version 2 corrected that extraction after exact-archive local validation. The first version-2 final-download connection failed with a TLS EOF before any artifacts arrived; the same final collection was resumed successfully. No status/list/live-log polling or additional compute submission was used for recovery.

## Filesystems visible in this allocation

| Path | Filesystem | Mount source | Mount access | Reported capacity GiB | Reported free GiB | Device ID |
|---|---|---|---|---:|---:|---|
| `/` | overlay | `overlay` | rw | 8156.88 | 1048.73 | 53 |
| `/kaggle/input` | ext4 | `/dev/loop2` | ro | 19.52 | 19.50 | 1794 |
| `/kaggle/working` | ext4 | `/dev/loop2` | rw | 19.52 | 19.50 | 1794 |
| `/kaggle/temp` | overlay | `overlay` | rw | 8156.88 | 1048.73 | 53 |
| `/tmp` | overlay | `overlay` | rw | 8156.88 | 1048.73 | 53 |
| `/dev/shm` | tmpfs | `shm` | rw | 14.00 | 14.00 | 75 |
| `/kaggle/input/datasets/cdeotte/melanoma-512x512` | nfs | NFS dataset export | ro | 25600.00 | 7069.86 | major:minor `0:52` |

Paths sharing a device ID expose the same filesystem device; they must not be counted as separate capacity. Exposed block-device names do not establish physical HDD/SSD hardware. Filesystem free space is separate from Kaggle saved-output quotas. `/dev/shm` is a RAM filesystem when reported as tmpfs.

The actual dataset files are on the nested NFS mount, not on the ext4 filesystem reported for the empty `/kaggle/input` parent. The NFS server reports 25 TiB total backing capacity; that is not a private user allocation. `/kaggle/working` has only 19.52 GiB of formatted capacity in this job, so the approximately 41-GiB H3 bundle cannot all fit there. `/kaggle/temp` and `/tmp` share the root overlay, which reports 8,156.88 GiB total and about 1,048.73 GiB free at inventory time; these host filesystem figures do not reserve that space for this notebook. The exposed block inventory includes a 128-GiB `sda`, an 8-TiB `sdb` marked read-only, and loop devices; these are not additional writable notebook volumes. This job had a 30-GiB cgroup RAM limit and four CPU cores. The 14-GiB `/dev/shm` ceiling is RAM-backed and consumes the same memory budget.

## Read measurements

| Path | First pass MiB/s | Immediate reread MiB/s | After DONTNEED hint MiB/s | Two-reader aggregate MiB/s |
|---|---:|---:|---:|---:|
| `/kaggle/input` | 207.6 | 5161.8 | 197.8 | 331.6 |
| `/kaggle/working` | 5246.0 | 5148.4 | 3716.4 | 4072.8 |
| `/kaggle/temp` | 5042.0 | 5102.4 | 3459.0 | 4241.4 |
| `/tmp` | 5266.1 | 5425.7 | 3518.9 | 4147.0 |

| Path | Random 4-KiB MiB/s / p95 ms | Strided 12,800-byte MiB/s / p95 ms | Two-reader strided aggregate MiB/s |
|---|---:|---:|---:|
| `/kaggle/input` | 2.4 / 3.4608 | 56.9 / 1.7570 | 18.6 |
| `/kaggle/working` | 141.2 / 0.0364 | 1593.3 / 0.0153 | 504.0 |
| `/kaggle/temp` | 128.9 / 0.0422 | 1506.5 / 0.0232 | 437.8 |
| `/tmp` | 104.3 / 0.0471 | 1383.4 / 0.0214 | 430.7 |

Rates are binary MiB/s. Each sequential phase is one bounded pass, not a median across independent sessions. Concurrency reports aggregate throughput over disjoint byte ranges, not the sum of separately timed rates. Range p95 values describe individual measured calls. Partial phases are labeled and do not represent a complete corpus sweep.

The first dataset pass began with 0% observed inode page residency and took 9.87 seconds for 2 GiB. The immediate reread began at 100% residency and took 0.397 seconds. After DONTNEED, only 0.0067% of dataset pages remained resident and the next pass took 10.35 seconds. This supports a roughly 198–208 MiB/s dataset path when those inode pages are absent in this allocation. It still does not identify the physical disk or prove remote caches were cold. Writable copies began with 100% inode residency because they had just been written; their first-pass values are cached reads. Even their 3,459–3,716 MiB/s hinted results may benefit from caches beneath the container filesystem.

Two sequential readers increased dataset aggregate throughput to 332 MiB/s. Two Python threads reduced the strided result from 56.9 to 18.6 MiB/s, with the same slowdown direction on scratch paths; this probe therefore does not justify blindly adding reader concurrency for tiny calls. Thread/GIL scheduling, disrupted read-ahead and filesystem behavior are not isolated by this test.

## Corpus, cache state and interpretation limits

The mounted public dataset was `cdeotte/melanoma-512x512`. The selected corpus was 2.000 GiB in 17 file ranges. It is a binary filesystem proxy; no H3 weight shard was used. Identical source bytes were copied to writable paths, fsynced, then read through the same benchmark. Copies were SHA256 checked after timing. Copy/hash/fsync time is excluded from read rates and is not presented as pure write bandwidth.

Preparation took 23.76 seconds for the working copy, 16.37 seconds for the temp copy and 16.30 seconds for the `/tmp` copy. For a single read, staging adds work. For repeated scans whose working set exceeds available page cache, staging selected shards into verified ephemeral scratch is worth evaluating. At 208 MiB/s, a hypothetical single sequential pass over 41 GiB would take about 3.4 minutes; this arithmetic extrapolation excludes model parsing, CPU LoRA work, GPU transfer and contention.

First pass and DONTNEED hints do not establish physically cold storage. Kernel, filesystem, host and remote caches can remain warm. This is CPU filesystem I/O, not H3 load latency or GPU-instance qualification.

Cache residency observations and per-process I/O deltas are retained for every phase in the JSON receipt. A successful inode cache hint or positive caller `read_bytes` does not prove all lower storage layers were cold. A 2-GiB probe can fit in cache; it cannot qualify repeated scans of the approximately 41-GiB H3 bundle under model memory pressure. This CPU allocation also does not establish T4-instance storage performance or H3 end-to-end loading latency.

In this allocation, dataset storage was usable for sequential loading but expensive for small scattered reads: random 4-KiB calls achieved only 2.37 MiB/s with a 3.46-ms p95. Favor contiguous pre-sharded bundle reads and avoid repeatedly reconstructing row shards through thousands of tiny dataset reads. The observed table applies to this allocation; cache eviction and model preparation add separate costs. Changing the storage source does not remove the duplicated CPU LoRA merging found in the H3 engineering review.

## Reproduction and evidence

- Driver: [kaggle_storage_probe.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/benchmarks/kaggle_storage_probe.py>); builder: [build_kaggle_storage_probe.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/benchmarks/build_kaggle_storage_probe.py>).
- Four CPU contract tests cover exact byte partitioning, sequential/range counts, deadlines, copy hashes and mount selection.
- The matching fresh capsule used immutable GitHub source and verified its archive/capsule hashes; H3 core libraries were not loaded or installed.
- Installed local Kaggle CLI 2.2.4 parses a version suffix but does not forward that version in its final-output request. No version after 2 was submitted; returned driver/source hashes match the recorded version-2 submission.
- Raw final outputs and SHA256 download manifest: `D:\attention_eval\kaggle_storage_probe_20261024\output_v2_resume`.
- [Machine-readable result](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/docs/data/kaggle_storage_probe_20261024.json>).
- Native package versions, raw mountinfo, df/lsblk results, path capacities and phase-level cache/error data are preserved in that receipt.
