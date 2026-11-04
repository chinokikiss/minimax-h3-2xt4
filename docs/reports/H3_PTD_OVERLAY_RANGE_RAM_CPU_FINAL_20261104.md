# H3 PTD Overlay HTTP Range-to-RAM CPU — final v1 results (2026-11-04)

**Result: COMPLETE, 10/10 cases; 10/10 full-file SHA256 PASS.** No GPU. Version-bound private Kaggle CPU Kernel: https://www.kaggle.com/code/pamcks/h3-ptd-overlay-range-ram-cpu-20261104 (version 1).

Source: the **same** immutable Hop00 `payload/aux_overlay.safetensors`, 1,544,161,056 bytes, pinned Dataset version 1, receipt SHA256 `16b4405637ea4d5e18a98f3cff0db31ee73f41427a939dd3b8666cac66eeb23e`. Raw HTTP redirect reached `storage.googleapis.com`; the 4-KiB HTTP Range probe returned 206 and matched mounted source bytes. Every full case also matched the immutable full-file SHA256.

## Aggregated throughput

Two cases per method, 3,088,322,112 bytes per arm. Weighted throughput = total validated bytes divided by summed elapsed time; MB/s is decimal.

| Method | Transfer seconds, two cases | Transfer MB/s | Full SHA-verified RAM-ready seconds | Verified RAM MB/s |
| --- | ---: | ---: | ---: | ---: |
| HTTP Range ×1 | 28.044 | 110.123 | 41.544 | 74.338 |
| **HTTP Range ×2** | **15.161** | **203.699** | **27.810** | **111.049** |
| HTTP Range ×4 | 26.226 | 117.756 | 37.085 | 83.276 |
| HTTP Range ×8 | 24.575 | 125.668 | 35.056 | 88.097 |
| Mounted tensor_preadv ×1 | 17.833 | 173.182 | 29.526 | 104.596 |

**Important: the two-case mounted aggregation is heavily warm-cache biased.** Treat the mounted row above as a recorded aggregate, not a fair remote cold-read estimate.

## Per-case transfer throughput

| Experiment order | Method | Transfer seconds | Transfer MB/s | SHA256 seconds | Verified RAM-ready MB/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Mounted preadv ×1, first pass | 17.289 | 89.316 | 6.639 | 64.535 |
| 1 | HTTP Range ×1 | 14.284 | 108.102 | 8.652 | 67.324 |
| 2 | HTTP Range ×2 | 8.111 | 190.385 | 6.807 | 103.509 |
| 3 | HTTP Range ×4 | 14.114 | 109.406 | 4.943 | 81.027 |
| 4 | HTTP Range ×8 | 10.948 | 141.048 | 5.117 | 96.118 |
| 5 | HTTP Range ×8 | 13.627 | 113.313 | 5.363 | 81.311 |
| 6 | HTTP Range ×4 | 12.112 | 127.487 | 5.916 | 85.654 |
| 7 | HTTP Range ×2 | 7.050 | 219.015 | 5.842 | 119.775 |
| 8 | HTTP Range ×1 | 13.760 | 112.221 | 4.848 | 82.983 |
| 9 | Mounted preadv ×1, last pass | 0.544 | 2838.618 | 5.055 | 275.809 |

Mounted first versus last read: **17.289s vs 0.544s (31.8×)**. Cgroup file cache changed from about 25 MB before the experiment to about 1.61 GB at the end, consistent with effective cache residency of this full-size file. The first mounted case is more indicative of remote cold-read behavior, but the experiment cannot guarantee perfectly matched cold cache or server/network state.

HTTP ×2 was the fastest **in both independent passes**, 190.385 and 219.015 MB/s, whereas ×1 was 108.102/112.221, ×4 was 109.406/127.487, and ×8 was 141.048/113.313 MB/s. Two-range aggregate transfer throughput is **1.85×** HTTP ×1, **1.73×** HTTP ×4 and **1.62×** HTTP ×8. Compared with the first mounted read only, ×2 yields **2.28×** aggregated transfer throughput (or **2.13×** when comparing first pass against first pass). This is an observed experiment comparison, not a guaranteed deployment speedup.

## Scope and interpretation

- **One preallocated bytearray** was reused by every case, with HTTP bytes written through `response.raw.readinto(memoryview)` directly into disjoint spans. No full Overlay HTTP staging file was created. Memory allocation took 1.869s, reported separately.
- Full-file SHA256 was deliberately performed after each transfer for experimental integrity. In production PTD inference, payload re-hashing is disabled by policy; hence **transfer MB/s**, not SHA-verified MB/s, is the more relevant transport figure. HTTP ×2 spent approximately **7.58s transferring + 6.32s SHA256** per Overlay.
- The lack of monotonic benefit from 4/8 connections is a measured fact, **not** proof of which particular server, TLS, CPU, proxy or network limit caused it.
- 203.699 MB/s is roughly 25.5% of the desired 800 MB/s target; ×2 does not qualify that target.
- This probes **one 1.544-GB Overlay**, not the numerous rank shard files or all eight 21.18-GB PTD Hop parts, and it does not measure producer/consumer overlap against live GPU inference.
- Do not compare the previous four-**independent-file** KaggleHub HTTP-to-disk benchmark directly with this one-file concurrent Range-to-RAM benchmark as though they had identical source/cache/path semantics.

## Execution evidence

- `supervision.json`: status complete, measurement_status complete, worker exit 0, supervisor elapsed 182.700s.
- `measurement/result.json`: status complete, 10 completed cases, full SHA256 on each, measured elapsed 182.055s, HTTP redirect resolved in 0.890s.
- `worker_stderr.log`: empty.
- CPU-only; no Dataset mutation or PTD production modification.
- Final collection was one version-pinned `kaggle kernels output .../1`; no status/list/live-log polling or resubmission.
- Raw final artifacts: `docs/data/h3_ptd_overlay_range_ram_cpu_20261104/final_v1/h3_ptd_overlay_range_ram_cpu_20261104/`.

**Architecture takeaway:** use HTTP Range ×2 as the next opt-in candidate for a bounded PTD Overlay prefetch experiment, while retaining mounted direct-reading as the existing default until a complete Hop/window-overlap E2E measurement validates the benefit.
