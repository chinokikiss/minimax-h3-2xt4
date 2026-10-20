# TE critical path — physical evidence and completed ablations

The completed physical CPU probe establishes checkpoint preparation as the
dominant TE bottleneck for this request. It does not establish Python GIL
contention or speculative DiT preparation as the main cause. The physical
loading ablations are complete. Coalesced loading improved the initial pair,
but failed the predefined TE improvement threshold in reverse-order
confirmation. The existing baseline remains selected; no candidate is promoted.

The workload is the same full 50-layer Native Ref2VA request used previously:
512×512, 124 output frames, deterministic gradient reference, seed 1729, four
actual denoising evaluations, TP2 TE/DiT, Native-aligned video query partition,
and Channel Parallel audio. Measured conditioning is `[1,279,5120]`. These are
full-request tests, not resident TE block latency or a larger token sweep.

## Completed CPU probe

The [physical CPU audit](https://www.kaggle.com/code/chinokiki9/minimax-h3-cpu-critical-path-audit)
completed on dual T4. Both requests in both passes passed Native checks.
TE/latents/video are bit exact; the pre-existing accepted audio rounding
contract remains separate. The first request has fresh managed caches and
includes VAE startup. OS page-cache coldness is uncontrolled.

| Pass | Request | Reference + TE | End-to-end |
|---|---:|---:|---:|
| Timing, observers disabled | 0 | 126.06 s | 414.46 s |
| Timing, observers disabled | 1 | 98.97 s | 359.36 s |
| CPU + full-request Kineto | 0 | 113.62 s | 390.98 s |
| CPU + full-request Kineto | 1 | 96.37 s | 361.23 s |

The profile pass follows the timing pass on the same allocation; storage/cache
state differs. Its lower cold latency is not negative profiler overhead. Warm
end-to-end latency increased about 0.52%, but two observations do not isolate
instrumentation overhead from variation. Diagnostic durations are not used as
performance wins.

| TE evidence, rank 0 / rank 1 | Cold diagnostic request | Warm diagnostic request |
|---|---:|---:|
| Checkpoint read call wall time | 56.81 / 57.33 s | 91.23 / 90.96 s |
| Calling-thread CPU inside reads | 11.52 / 11.75 s | 10.84 / 10.86 s |
| Logical bytes read during TE stage | 14.95 / 15.01 GB | 12.83 / 12.82 GB |
| First-load tensor checksum wall time | 37.50 / 37.23 s | 0 / 0 s |
| Foreground Future wait union | 95.12 / 93.40 s | 79.89 / 81.58 s |
| Pinning wall time | 2.37 / 2.28 s | 2.53 / 2.60 s |
| Useful GEMM + attention kernel unions* | 0.91 / 0.52 s | 0.60 / 0.89 s |
| NCCL kernel union | 0.094 / 44.70 s | 26.18 / 0.105 s |

*Separate category unions are descriptive, not a non-overlapping total. CPU
phases are inclusive and overlap. Never add rank waits, nested preparation,
I/O, pinning and GPU durations to reconstruct request latency.

Evidence: `kaggle_cpu_audit_output/cpu_audit_results/continuous/offline_p1_p2/kineto/`
contains `te_critical_path.json` (200 rank/request/block records),
`cpu_path_analysis.json`, `causal_analysis.json`, the per-rank custom traces,
`cpu_thread_samples.jsonl`, and tool capabilities. Raw Kineto remains on Kaggle;
this investigation downloaded only missing CPU evidence and reused the GPU
analysis rather than another multi-GB full capture.

## Confirmed causes, bounds and hypotheses

1. **Checkpoint readiness blocks TE consumers.** Exact Future identities link
   consumers to the preparation producer and readiness brackets. On warm
   request 1, rank 1 waits for block 44 at 83.807–85.818 s after request start;
   its preparation takes about 1.95 s wall / 0.22 s thread CPU. Similar waits
   repeat across the stack. Rank 0 block 32 waits at 60.364–62.348 s. Read
   callbacks dominate the producer's wall time. The OS samples repeatedly
   observe that producer in uninterruptible `D` state inside read calls,
   predominantly `folio_wait_bit_common`: 733 / 763 warm observations on
   ranks 0 / 1. This supports waiting for file pages, rather than attributing
   the complete delay to useful CPU execution. These are samples, not exact
   durations; physical storage traffic, page-cache misses, device throughput
   limits and storage/network origin remain unmeasured.
2. **Native integrity verification adds cold CPU work.** All 50 first-load TE
   tensor hash calls cost about 37 core-seconds per rank and then disappear on
   the warm request. Verification is retained. Moving it outside measured
   cold latency without accounting for that work would be an unfair comparison.
3. **TE's short compute cannot conceal a slow serial producer.** TE's useful
   GPU kernels take a small fraction of its 96 s warm stage. DiT's longer
   attention/GEMM execution can run concurrently with preparation, explaining
   why comparable producer waits have far less both-GPUs-idle exposure there.
   The old 150.64 s Future wall union was not 150.64 s removable latency.
4. **NCCL can mask lack of useful progress.** The waiting rank spends 44.70 s
   cold / 26.18 s warm in TE NCCL kernels while its peer has little NCCL time.
   Matched collective host-entry skew reaches 1.141 s cold and 0.729 s warm.
   In the warm request, the latter entries occur at 67.525 s rank 0 and
   68.254 s rank 1. These support rank readiness imbalance. NCCL occupancy is
   not useful-model computation or proof of wire congestion. A complete
   kernel-level dependency diagnosis remains bounded by clock-fit uncertainty.
   Re-reading the preserved, compressed four-request Offline trace confirms
   the masking directly. Request 3 has 87.58 s of foreground TE Future waits:
   25.86 s coincide with both GPUs having no activity, but 86.81 s coincide
   with no non-NCCL model-kernel progress. NCCL-only occupancy accounts for
   60.00 s of this wait-associated exposure. One such interval is
   24.267–26.026 s after that TE stage starts: rank 0 awaits block 11 at
   24.205–26.002 s and rank 1 awaits block 12 at 24.260–26.247 s, each a
   control-flow predecessor of the next `cudaMemcpyAsync` weight submission.
   In the same request, DiT has
   51.52 s of foreground waits, of which only 6.11 s have no useful kernel
   progress; 4.41 s of those overlap NCCL. This confirms substantially more
   overlap with real computation in DiT. It does not prove that removing a
   wait removes the same amount of end-to-end latency. Reproduction:
   `python analyze_useful_progress.py kaggle_causal_resume_output/causal_audit_results/continuous/offline_p1_p2/kineto`;
   evidence is its `useful_progress.json`. Rank/thread/device intervals are
   unions, and memcpy/memset/NCCL are excluded from model-kernel progress.
5. **Speculative DiT interference was not observed during TE dispatch.** The
   overlap of actual DiT preparations with `rank_dispatch/te` is zero in both
   requests/ranks. Current code starts DiT lookahead after TE encoding returns.
   This does not establish behavior for every workload or the subsequent
   transport handoff, but it rules out that proposed overlap in this capture.

Runnable delay is unavailable because `sched_schedstats=0`; `perf` is absent.
The previous py-spy capture failed because native unwinding cannot be combined
with `--nonblocking`. That incompatible option was removed. The corrected probe still failed to inspect the Kaggle interpreter, reporting
`Failed to find a python interpreter in the .data section`. It yielded no
usable GIL samples. No exact GIL waiting duration is claimed.

The host cache also prepared 115 / 116 TE blocks across only 100 actual uses
per rank, with frequent evictions. Some work is speculative and spans request
handoffs. This establishes extra preparation; blaming a particular eviction
for an exposed stall still requires its producer/consumer interval. Preserving
a larger TE prefix and beginning block lookahead during video decoding are
follow-on candidates, not measured wins in this submission.

## Implemented candidates and physical acceptance

`patches/te_loading.py` adds opt-in candidates:

- **Coalesced:** one sequential read/allocation per rank-local offline TE
  block, with disjoint tensor views. Original manifest, source identity,
  configuration, tensor hash and Native assignment checks remain intact.
- **Coalesced + bounded parallel:** at most two pageable TE preparation
  workers per rank; DiT and startup groups keep single-worker preparation.
  In-flight payloads count against cache capacity; construction reservations
  share the original HostBudget. Pinning/H2D remain in the original window.
- A locally qualified priority variant disables speculative DiT preparation
  until sampling. It is not in the physical matrix because the completed
  capture shows zero overlap with TE dispatch. It is not a production default.

Both loading approaches pass the local compact trained Native full-request
checks. The bounded variant plus targeted profiling matches all five preserved
uninstrumented outputs bit-for-bit and exits cleanly. These local logical-rank
tests qualify semantics, not T4 speed. A nested profiler ownership defect found
during qualification was fixed before submission: only the scope/thread that
opens a profiler may close it.

The [physical ablation job](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-critical-path-ablations)
version 1 is **COMPLETE**, checked on 2026-10-14. Physical GPU identity was
guarded before accepting results. Twelve unprofiled full requests and one
separate diagnostic request passed Native correctness checks: conditioning,
latents and video pixels are exact; waveform differences remain within the
previously accepted Channel Parallel rounding contract.

| Pass / policy | Cold reference + TE | Cold end-to-end | Warm reference + TE | Warm end-to-end |
|---|---:|---:|---:|---:|
| Initial baseline, 2 requests | 125.686 s | 410.976 s | 99.235 s | 364.114 s |
| Initial coalesced, 2 requests | 111.621 s | 379.990 s | 95.160 s | 354.236 s |
| Initial coalesced + parallel, 2 requests | 108.379 s | 385.236 s | 95.672 s | 354.715 s |
| Confirmation coalesced, 3 continuous requests | 111.209 s | 373.581 s | 96.061 s | 352.254 s |
| Confirmation baseline, 3 continuous requests | 112.341 s | 377.552 s | 96.941 s | 356.333 s |

Warm values in the initial pass are one request; confirmation values average
the final two requests. Managed caches start fresh per policy, while OS page
cache is uncontrolled. Cold results include startup and cannot isolate loader
benefits. The initial coalesced reduction was 4.11% for TE and 2.71% end-to-end;
parallel preparation gave 3.59% and 2.58%, without beating coalesced alone.
Coalesced advanced to winner-before-baseline confirmation, which reduced TE
only **0.91%** and end-to-end **1.14%**. Acceptance requires at least 3% warm TE
and 1% warm end-to-end improvement in both comparisons. It failed that gate.
Small fixed-workload samples do not establish generalized or statistical gains.

The targeted baseline diagnostic request completed and passed correctness
(114.244 s reference + TE; 385.588 s end-to-end). Subsequent report generation
failed with `ImportError: cannot import name 'read_json'`: the notebook bundled
an older `analyze_causal_audit.py` alongside the newer TE analyzer. This is an
analysis packaging defect, not an inference failure. The builder now includes
that dependency explicitly for future submissions; version 1 remains frozen.
The saved capture can be analyzed locally without repeating GPU inference.
Py-spy still cannot inspect this interpreter (`Failed to find a python
interpreter in the .data section`); exact GIL wait duration remains unavailable.
Targeted captures have no kernel coverage outside TE/reference.

Evidence is in `kaggle_te_critical_path_output/te_latency_results/`: per-pass
`result.json`, Native verdicts, `te_ablation.json`, `te_matrix.log`, and diagnostic
tool capabilities. Performance results are complete; new diagnostic analysis
is partial. Normal inference retains its existing loading policy.

## Cleanup delivered

Detailed event histories are disabled in normal inference and bounded when
enabled; stage history is bounded to 16 entries. Benchmark kernel profiling
and HTML timelines are explicitly opt-in. Thirty completed trace files were
SHA256-verified through gzip round trips, reducing 11.873 GiB to 1.157 GiB
without losing evidence. Three obsolete local debug markers were removed.
See `ARTIFACT_RETENTION.md` for restore and retention commands. Reference
results, checkpoints, active experiments and historical reproduction scripts
remain intact.
