# H3 PTD V2 Range2 HostWeightCache 0.901MP 10s E2E — 2026-11-04

## Outcome

**COMPLETE** on Kaggle Kernel `pamcks/h3-ptd-v2-range2-hostcache-09mp-20261104`, version 1. The immutable notebook submission receipt is stored locally at `docs/data/h3_ptd_v2_range2_hostcache_e2e_20261104/gpu_submission_v1.json`; final media and raw execution artifacts are intentionally not committed to the public Git repository.

- Hardware: 2 × Tesla T4 (SM75), PyTorch 2.11.0+cu130, CUDA 13.0.
- Workload: MiniMax-H3 Ref2VA, 1280 × 704 (0.90112 MP), 243 decoded frames, 240 exported frames at 24 FPS / 10 s, 8 PTD hops / 8 NFE, all Kitchen DiT attention.
- Weight path: opt-in HTTP Range ×2 directly into CPU Tensor storage adopted by HostWeightCache; no whole-overlay disk staging; no repeated runtime payload SHA256. `H3_PTD_HTTP_RANGE2_DEBUG=1`, global `debug=False`.
- Finished with video/audio exports, two nonfatal legacy TE column-view diagnostic warnings. No quality-equivalence claim (pruned AdaLN approximation enabled); visual/audio review remains pending.

## Wall-clock timings

| Stage | Seconds |
| --- | ---: |
| VAE metadata | 0.737 |
| Reference + TE | 145.623 |
| PTD 8-NFE sampling | 1466.205 |
| Video decode | 114.807 |
| Audio decode | 3.310 |
| Media export | 16.999 |
| E2E command (including other overhead) | 1819.17 |
| Supervisor total | 2144.93 |

## HTTP Range ×2 instrumentation

These are **per-rank sums of actual HTTP requests**, not wall-clock aggregate network capacity. HTTP MB/s uses transferred bytes divided by summed Range-GET transport durations; Tensor-ready MB/s also includes URL resolution and CPU Tensor fill. The per-Hop series in the raw result is cumulative and must not be added across hops.

| Metric | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| HTTP bytes | 101,681,876,992 | 78,560,776,192 |
| Requests | 866 | 822 |
| Range-GET transport seconds | 919.929 | 791.239 |
| Tensor-ready seconds | 1135.246 | 1008.460 |
| HTTP transfer MB/s | **110.532** | **99.288** |
| Tensor-ready MB/s | **89.568** | **77.902** |
| Batch p50 download MB/s | 100.385 | 100.496 |
| Batch p95 download MB/s | 124.421 | 116.567 |

Both ranks combined transferred **180,242,653,184 bytes** across **1,688 requests**; the transport-time-weighted mean is approximately **105.33 MB/s**, *not* concurrent aggregate throughput. The tensor-ready-time-weighted mean is approximately **84.08 MB/s**. The 800 MB/s target was not reached.

## Comparison and limitations

The prior mounted-I/O PTD V4 sampled 8 NFE in 1497.46 s and spent approximately 1811 s in the E2E command, versus 1466.21 s and 1819.17 s here. The sampling-stage improvement (31.25 s, 2.1%) did **not** translate into an E2E latency reduction. The runs were not tightly controlled for external I/O conditions, caching or scheduling, so no causal speedup can be assigned to Range2.

Video decode used `native_temporal_pairs` and audio decode `native`, both without fallback. Media sizes: H.264 MP4 13,284,048 bytes; WAV 1,280,044 bytes. The final result reported `status=complete`; actual pixel/waveform tensors passed finite checks. Final artifact files and per-Hop telemetry remain in the local Kaggle archive, not in Git.
