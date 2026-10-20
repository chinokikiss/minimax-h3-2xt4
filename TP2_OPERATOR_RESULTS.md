# TP2 operator screen: physical T4 results

**Current user decision:** preserve the current zero-shot sparse routing and
scalar-pooling schemes. Do not use the historical L1/L2 diagnostics below to
judge generation quality or drive sparse precision changes. Acceptance concerns
generation-distribution and semantic deviation, which requires full H3 runs and
is deferred while T4 time is insufficient. Current work is performance only;
see `TP2_OPERATOR_PASS2.md` for the newer confirmed T4 improvements and
`TP2_OFFLINE_OPTIMIZATION.md` for their offline preparation. The T4 results below
remain historical evidence.

2026-10-16. Initial screen completed on two simultaneous Tesla T4 SM75 ranks,
Torch2.11.0+cu130, nvcc12.8, TileLang0.1.15 and Kitchen0.2.37. No model weights
were downloaded and no H3 pipeline was run. The verified target map is in
`TP2_OPERATOR_OPTIMIZATION.md` and the machine-readable manifest.

## Attention optimization implemented and measured

`patches/tilelang_grouped_tp.py` adds native GQA, rectangular global-KV attention,
and Group32 DiT drop/scalar pooling. It compares eager preparation, fused
Group32 quantization with the original group loads, and fused preparation with
full Q/K caching. The fused quantizer is reused from the earlier audited module;
its integration now handles the actual TP partitions without KV head replication.

The table uses **the slower rank's median complete attention API**, including
casts, preparation, routing where applicable and conversion to BND/native dtype.
Both ranks execute concurrently. This is not request latency or pipeline speedup.

| Actual TP2 attention | Eager Group32 ms | Fused quantization/group loads ms | Full Q/K cache ms | Best latency reduction vs eager |
|---|---:|---:|---:|---:|
| TE Q32/KV4, N353, D128 | 1.113 | 0.786 | 0.819 | 29.4% |
| DiT H28, N6814, D128, dense | 93.316 | 81.693 | 95.264 | 12.5% |
| DiT same, drop | 68.542 | 57.914 | 59.403 | 15.5% |
| DiT same, scalar pooling | 87.455 | 67.802 | 68.304 | 22.5% |
| Video H32, Q1024/773, K1797, D64 | 4.084 | 2.593 | 2.565 | 37.2% |

All30 implementation cases passed exact valid QI/KI/scale identity against eager
preparation, identical sparse routes, finite outputs, independent streaming
reference checks on three heads per rank, and density1 sparse/dense checks.
The largest sampled-head implementation relative L2 is0.00021355 (0.0214%).
Full-output cross-candidate checks also pass. This validates the implementation,
not the sparse approximation's fidelity to native attention.

Native SDPA medians are1.433ms TE,238.063ms DiT and1.432ms Video. For Video this
is an FP16 dense reference, **not the active TP2 production baseline**, which
calls Kitchen INT8 attention. The follow-up below measures that exact baseline
and the active KV-exchange method. Do not compare Group32 against SDPA and call
it a production Video speedup/regression.

## Active Video TP2 microchain confirmation

The follow-up invokes the actual `AlignedVideoDecode.exchange` method on real
local K/V slices and verifies that gathered global K/V exactly match the fixture.
It then calls the active `comfy_kitchen.int8_attention` or Group32, and performs
the same BND reshape/finite handling. It excludes QKV production, RMSNorm/RoPE,
the output projection and the rest of the decoder.

An initial fixed-count screen exposed idle-clock sensitivity after JIT. Its
v1 timings are retained as diagnostic evidence, not the comparison below.
The v2 confirmation prewarms the device for at least1second before each phase,
aligns ranks, runs10 fixed operator warm-ups and records15 repeated samples.
Compilation and loading are excluded. No extra Systems pass was launched for v2.

| Same active query partition | Attention API, slower rank ms | Exchange + attention + BND/finite, slower rank ms |
|---|---:|---:|
| Current Kitchen INT8 | 1.068 | 2.773 |
| Group32 fused, group loads | 2.414 | 4.082 |
| Group32 fused, full Q/K cache | 2.329 | 4.071 |

Standalone KV exchange is1.93–1.96ms. Group32's complete microchain is46.8%
slower than the current Kitchen baseline in this confirmation. Full Q/K caching
helps its isolated attention API slightly but changes the microchain by less
than0.3%. The different warm-up protocols make first-screen and follow-up
absolute attention times unsuitable for cross-run subtraction.

The three Video Systems ranges show four Kitchen attention-API kernels per
call versus16 for Group32. Native Q/K quantization is fused; Group32 still
launches separate mean/count/reduction/copy operations. Native attention uses
an8-by32 grid for1024 queries (128 queries per CTA), versus Group32's32-by32
grid (32 queries per CTA), which repeats KV loads across more CTAs. The core
arithmetic also differs: Kitchen uses INT8 P/V; this Group32 contract uses FP16
PV and channel-group scale accumulation. Performance comparisons do not imply
these are interchangeable precision modes.

This makes dense-only Video query tiles64/128 and staged mean/quantization
fusion high-priority candidates, alongside NCCL all-gather protocol/packing.
Preserve global key centering, true ragged counts and FP16 probability rounding.
Do not change DiT route granularity by reusing a wider query tile indiscriminately.
On this one Video boundary, worst-head SDPA-relative error is0.653% for Kitchen
and0.661% for Group32; this does not establish whole-VAE quality equivalence.
The user's Group32 policy and optional sparse modes are retained; production
dispatch has not been changed by this study.

## Profiler evidence and optimization decisions

Collected23 Systems reports:13 GEMM and10 attention capture ranges, both ranks.
CPU export/analysis uses process-qualified correlation IDs, preventing collisions
between ranks. Of2746 kernels,2725 have a matching launch/NVTX scope;21 boundary
kernels are unattributed. Some captures clip a repetition when the other rank
ends its capture, so use actual observed counts. Profiled absolute durations
also differ from event timings; use repeated unprofiled samples for selection.

Full Q/K caching is not the default optimization. D128 dense/drop launches grow
from24,576 to35,328 shared bytes per CTA. Dense registers rise from197 to206 per
thread. Scalar pooling uses32,896 versus43,648 shared bytes. These resource
increases accompany the measured cached-kernel regression; they support pruning
this variant on long D128 cases. They do not establish achieved occupancy or
memory-stall cycles. D64 Video has a small cache benefit and remains a candidate.

At current routes, retained density is57.40%/58.46% across ranks, so block
omission is42.60%/41.54%. With the same fused preparation and group loads,
drop reduces the slower-rank API from81.69 to57.91ms (29.1%), while scalar pooling
reduces it to67.80ms (17.0%). This screen **does show a sparse speedup**; the old
37.6%-omission/no-speedup observation is a different shape/quantization/path.
It must not be reused as the explanation for these new results. Scalar pooling
still performs centroid scores, reductions, exponentials and value updates on
omitted leaves. Preparation and routing also remain in the complete API.

## Current TP GEMM and communication bottlenecks

Thirteen unique local cases per rank cover17 MNK triples across the two ranks.
All INT32 cores match an independent PyTorch integer GEMM exactly. Linear
inputs/weights are synthetic at verified TP shapes, so this is performance/core
equivalence evidence, not real linear-activation quantization quality. Video
linear fixtures exclude RMSNorm weights and residual fusion; do not treat them
as full active MLP/attention blocks.

Representative rank0 unprofiled medians:

| Current DiT operation | Complete API ms | Standalone INT8 GEMM ms | Standalone INT32 SUM ms |
|---|---:|---:|---:|
| QKV column | 25.53 | 20.02 | — |
| O row | 50.70 | 6.51 | 36.63 |
| FC1 column | 32.79 | 28.69 | — |
| FC2 row | 68.01 | 14.23 | 36.58 |

Separate phase medians are non-additive. Column complete APIs and raw INT32
GEMMs can select different backend algorithms/epilogues; do not subtract their
times to estimate quantization. Both ranks' collective warm-up counts agree
for every row API, SUM and MAX phase in this completed v1 run. Subsequent runs
use the new fixed-count distributed timer to eliminate the timing-loop risk.

Systems independently identifies the row bottleneck. Across three rank0 O-row
calls: INT32 SUM109.02ms total, FP32 MAX1.58ms, GEMM10.80ms, rotation10.75ms,
quantization2.06ms, dequantization3.71ms. Across three FC2 calls: SUM108.53ms,
MAX4.47ms, activation11.98ms, rotation23.89ms, GEMM23.49ms, quantization4.17ms,
dequantization3.65ms. Additional padding, maxima, sentinel and contiguous-copy
kernels remain under the outer API scope. Every DiT row SUM carries a146.53MB
INT32 tensor per rank. Traces show a RING_LL NCCL kernel; protocol/channel
selection is now a higher-priority microbenchmark than a broad GEMM rewrite.

Next high-ROI work, in order:

1. Benchmark NCCL default versus supported Simple/LL protocols and a small
   channel shortlist at the146.53MB boundary. Preserve INT32 SUM and the actual
   no-P2P configuration. Test chunked GEMM/SUM overlap only after quantizing with
   the unchanged global FP32 rotation geometry; measure competition for T4 SMs.
2. Fuse external-global-max row quantization to remove sentinel concatenation;
   reduce activation/rotation/padding traffic. Verify native scale/rounding and
   the current FP32 global GEMM geometry before performance acceptance.
3. Reuse identical TE Q/K/V and Gate/Up preparation. DiT already packs those
   projections. Preserve native integer accumulation and post-SUM cast ordering.
4. Keep group loads for long D128 attention; test Q-only reuse/query tiling with
   a tight resource budget. Remove drop's unused V-centroid work and test direct
   BND stores through FP16 rounding. Do not change leaf64 or arithmetic precision
   and label the result a free optimization.
5. Keep a narrow shape-specific GEMM shortlist, especially column QKV/FC1.
   Rewriting every GEMM is lower ROI while row communication dominates.

## Sparse fidelity root cause: routing, not Group32 implementation

On the full real DiT layer25/call9 fixture, worst-head errors against fresh
native dense attention are0.701% for Group32 dense,51.82% for drop and51.75%
for scalar pooling. Rank0 aggregate output relative L2 is0.516%,10.81% and10.75%
respectively. These are attention-boundary errors, not video-quality scores.

A zero-GPU-cost CPU diagnostic controls the arithmetic on all56 heads using
32 uniform and32 largest-Q-norm query rows per head. This intentionally probes
outliers and is not an unbiased whole-sequence mean. For global head14:

| Controlled ablation | Sampled output relative L2 |
|---|---:|
| FP16 cast only | 0.02% |
| Group32 dense | 0.36% |
| Routing/drop with **native FP32 Q/K/V** | 59.77% |
| Group32 drop | 59.86% |
| Group32 scalar pooling | 59.81% |

The same sampled queries lose64.63% of native probability mass on average and
up to94.63%. Scalar pooling estimates only0.265% mass for the omitted leaves.
Their true grouped log-sum-exp exceeds the centroid logit plus log-count by an
average4.83 natural-log units. This directly exposes the failure: centroid
routing misses salient within-leaf keys, and scalar mean summaries severely
underestimate their exponential mass. Changing the INT8 GEMM or quantizer will
not repair that loss. Head14's native output RMS is2.58, so this is not simply
a vanishing-reference-norm denominator artifact.

The user has decided to retain both existing sparse options and their current
route policies. These operator diagnostics do not establish generation-quality
or semantic deviation. Retention changes, dense fallback rules and scalar
precision corrections are outside the current optimization scope. Any later
generation-quality assessment must use full H3 output distributions and
semantics; no new full-pipeline runs are authorized within the current budget.

## Evidence and limits

Evidence root: `D:/attention_eval/operator_optimization_tp2_20261016`.
The `v1/tp2_operators` folder contains both-rank raw samples, hardware/compiler
identity, source hashes,23 original Systems reports, CPU-exported SQLite files
and `trace_attribution.json`. `cpu_sparse_diagnostics.json` records every sampled
query and ablation. Dataset blobs were reused without new QKV uploads.
`video_v1/video_tp2` contains the actual Video exchange/attention Systems traces;
`video_v2/video_tp2` contains the warmed confirmation and raw samples. All three
Kaggle sessions completed. Their recorded command stages total about16.1minutes
of wall time, including package setup; this is not a provider billing estimate.

The fixtures are native H3 captures partitioned according to current TP2
source, not a fresh TP2 trajectory capture. They cover one TE, DiT and Video
boundary for screening. There is one process run with10 repeated measurements
per phase; finalists still need independent-process repetition. Compilation/
loading are outside those steady-state timings. Telemetry peaks at2501MiB per
GPU for this operator study; it does not estimate full-model VRAM.

Nsight Compute hardware counters remain unavailable: the prior physical-T4
probe returned `ERR_NVGPUCTRPERM`. No new counter attempt was made. Systems
evidence cannot establish achieved occupancy, Tensor Core utilization or memory
stall breakdown. No new full-pipeline quality/throughput claims are made.
