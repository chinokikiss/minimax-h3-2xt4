# TE blocks 2–49: causal profile

Status: physical targeted diagnostic COMPLETE, two full Native-checked requests.
P0 is retained; P1 stays opt-in. Idle performance validation is separate.
The canonical workspace `TE_RESIDENCY_P01_RESULTS.md` and current TE/DiT/VAE
lifecycle were reviewed. No inference scheduling changes preceded this analysis.

## Targeted physical result: file readiness, not CPU run-queue delay

[Physical diagnostic](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-block-critical-path),
version 1: 50 TE/DiT layers, 512×512/124 frames, four actual NFEs, unchanged P0.
Both requests pass exact conditioning/latent/video checks and the existing CP
audio contract. No decoder fallback; all managed/pending preparation balances
are zero after close. This is diagnostic latency, not an optimization benchmark.

Warm reference+TE is **103.452 s**. Both GPUs have no captured kernel/memcpy
activity for **68.443 s**; **61.017 s** of that union intersects verified block
2–49 staging-Future dependencies. Rank overlaps are counted once. Clock fits
use 1,910/3,008/3,009 CPU annotations, with p95 residuals 24.5/30.2/30.9 µs.
Only warm TE has complete Kineto coverage. CUDA-event durations are useful,
but their initial absolute anchor does not bound drift over the preceding
request; cross-clock ordering and inactivity use CPU-fitted Kineto instead.

| Blocks 2–49, serial producer | Rank 0 | Rank 1 |
|---|---:|---:|
| Whole-block OS observation wall | 90.917 s | 91.466 s |
| Running, Linux schedstat | 10.565 s | 10.619 s |
| Runnable but not scheduled | 0.344 s | 0.298 s |
| Remaining off-CPU time, approximate | 80.008 s | 80.550 s |
| Buffered read wall / calling-thread CPU | 88.824 / 9.007 s | 89.167 / 9.039 s |
| Producer-thread storage read accounting | 11.649 GB | 11.656 GB |
| Logical payload read | 11.711 GB | 11.711 GB |
| Caller major faults | 0 | 0 |
| Samples in uninterruptible sleep, D | 784/903, 86.8% | 802/915, 87.7% |
| `folio_wait_bit_common` samples | 786/903 | 801/915 |
| Pinning wall | 1.867 s | 1.742 s |

These serial producer intervals can be summed within one rank. The two ranks
must not be summed into request latency. OS snapshots bracket the narrower
prepare interval, so the off-CPU residual has boundary uncertainty. Sample
counts identify the dominant wait state; they are not exact blocked durations.
Kernel stacks are denied in this container. GIL ownership remains unmeasured;
the observed D/folio waits and small runnable delay do not support GIL or CPU
run-queue saturation as the dominant explanation for these read stalls.

Confirmed causal chain: required shard read → worker waiting for file-page
readiness → staging Future unavailable → H2D submission delayed → no work on
either GPU over the linked interval. Almost all logical payload requires
kernel-accounted storage reads. Large caller minor faults include destination
allocation; zero caller major faults does not establish a warm file cache.

The full warm request records 23.462 million cgroup file refaults, 23.603 million
kswapd page steals and 33,950 direct steals. Observed TE boundary memory reaches
29.83 GiB against the established 30 GiB limit; request OOM/kill/max-event deltas
are zero. This supports background cache churn rather than caller direct reclaim
as the principal mechanism in this capture. Counters cover the shared cgroup,
not TE exclusively. Active whole device `sdb` records 48.671 GB reads across the
whole request; do not add its partitions, loop or dm layers to that traffic.

Queue backlog is real: bounded enqueue-to-worker median is approximately
13.452 s on rank 0 and 23.232 s on rank 1. The producer spends its time waiting
on folios rather than runnable. Increasing worker count is therefore not an
evidence-based fix; queue residence overlaps the same underlying producer work.
Pin/H2D savings alone cannot account for the roughly 61 s dependency exposure.

For example, block 4/rank 0 waits at request-relative **12.956–14.789 s**,
its Future is published at **14.787640–14.787814 s**, and H2D is submitted at
**14.790056 s**. Rank 1 waits **13.139–14.998 s**, publication is
**14.996457–14.996732 s**, H2D submission **14.999058 s**. Their linked
both-idle union is **1.653428 s**, not the sum of both waits.

The actionable order is: (1) test bounded GPU reuse during real think time;
(2) preserve useful CPU/file residency while measuring background+foreground
I/O together; (3) address remaining reader/storage stalls only through controlled
ablations. Do not expand concurrency or P1 scope based on inclusive wait totals.

With this capture, conditional 4 GiB/rank opportunities are ≤2.60 s for a 5 s
gap (blocks 2–3), ≤10.64 s for 15 s (2–8), and ≤25.72 s for 60 s (2–18).
These omit H2D, startup eviction and contention; the implemented policy also
prefills blocks 0–1 and may therefore cover less. They are not predicted gains.

Reproduction:

```powershell
python analyze_te_causal_profile.py kaggle_te_causal_profile_output/te_causal_results/pipeline
python summarize_te_block_os.py kaggle_te_causal_profile_output/te_causal_results/pipeline
```

Evidence is in that directory: `te_blocks_2_49.json` (96 per-rank/block rows),
`te_block_os_summary.json`, `te_block_os.csv`, `te_causal_timeline.html`, and six
SHA-verified trace archives. The archives occupy 10.75 MB rather than 129.54 MB
raw. Whole-block snapshot overhead outside prepare is 0.253/0.261 s per rank;
total Kineto/hook overhead is not isolated by these numbers. Timing pilot runs
omit Kineto and the OS observer. Nsight Systems is absent on this allocation;
no nsys trace or scheduler/GIL ownership claim is made.

## Earlier saved-trace analysis (separate allocation)

## Measured dependency exposure

The actual warm request has a 99.647 s reference+TE stage. Parent and both rank
Kineto timelines are merged on the host monotonic clock. Clock fits use
1,910/3,009/3,010 matching CPU annotations; p95 residuals are 15.6/29.0/28.9 µs.
CUDA anchor uncertainty is 18.9–25.9 µs. Only warm request 1 has complete GPU
coverage; the uncaptured first request is never labelled idle.

Both physical GPUs are inactive for a union of **80.887 s**. Of that,
**74.219 s** intersects exact foreground staging-Future waits for blocks 2–49,
with matching Future publication and subsequent same-thread H2D submission.
Ranks are unioned, not added. This is a confirmed submission dependency with
conservative GPU exposure, not a full counterfactual DAG or guaranteed speedup.

## Ranked findings

1. **Serialized checkpoint preparation/read readiness dominates.** Across
   blocks 2–49, rank 0/1 read-call wall totals are 85.811/86.352 s, but calling
   thread CPU is 8.713/8.809 s. Preparation totals are 87.799/88.259 s with
   10.074/10.127 s CPU. The earlier physical file-read investigation establishes
   page loss/refault → fresh backing I/O → folio waits on its allocation.
   This capture confirms nonrunning read time and foreground dependency;
   per-block storage-vs-scheduler attribution still needs OS counters.
2. **Queue backlog is largely a consequence of the serial producer.** Cache
   request-to-preparation observed delay has median ~22.0/22.2 s; staging
   executor delay ~1.84/1.85 s. Enqueue timestamps were emitted after submit,
   so these are observed lower bounds. They must not be added to read time or
   interpreted as proven CPU saturation. Adding executor workers alone is not
   justified, especially given earlier unsuccessful parallel-preparation tests.
3. **Pinning and H2D are secondary, though not free.** Pin spans total
   1.969/2.044 s for these blocks. Per-block H2D stream intervals, launches and
   first correlated GEMMs are retained. A GEMM is a known weight-consumption
   bound; it is not necessarily the first instruction accessing a norm weight.
4. **Cross-rank skew/NCCL can mask inactivity.** Local inactive unions differ
   (81.906/96.007 s), so a single-rank Future duration or NCCL runtime is not
   end-to-end idle. The 74.219 s estimate requires both GPUs inactive.

Shared device traffic, cgroup refaults and PSI are not attributed exclusively
to an individual TE block. Buffered misses need not cause caller major faults;
memory PSI can include page-readiness waits. GIL/runnable delays remain unknown.

## Bounded opportunity, not a performance claim

A trace-based candidate plan for blocks 2 onward, capped at 4 GiB per rank,
gives the following **conditional upper bounds**. It assumes observed active-run
preparation costs remain applicable during idle. It omits copy cost, startup
eviction, cancellation and contention, so actual gains may be lower or zero.

| Think time | Candidate blocks/rank | Observed preparation+pin cost | Covered exposed wait |
|---|---|---:|---:|
| 0 s | None | 0 s | 0 s |
| 5 s | 2–3 | 3.28 s | ≤2.16 s |
| 15 s | 2–9 | 14.77 s | ≤12.83 s |
| 60 s | 2–18 | 31.50 s | ≤28.47 s |

The full 74.219 s is outside a realistic 4 GiB idle-cache claim. The purpose of
the physical idle-gap ablation is to measure how much of the smaller candidate
opportunity survives memory pressure and real request handoff.

## Evidence and reproduction

`python analyze_te_causal_profile.py kaggle_residency_p01_output/residency_results/diagnostic/p0/kineto`
produces `te_blocks_2_49.json`: 96 rank/block rows, prefetch/worker/read/pin/H2D
intervals, exact Future links, correlated kernels, clock fits and unioned GPU
exposure. The regression test proves parent work/other-rank activity exclude
overlapped waits and duplicated rank waits are counted only once.

That saved capture lacked bounded queue submission, per-block OS deltas and
sampled worker wait states. The new targeted capture above supplies them at
whole-block boundaries, without per-tensor mincore or extra NFEs. Exact GIL
ownership, scheduler switch traces and counterfactual savings remain unavailable.
Full traces are selected diagnostics; normal inference and timing use compact metrics.
