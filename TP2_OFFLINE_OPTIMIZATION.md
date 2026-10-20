# Performance work with sparse behavior frozen

2026-10-16. Current user decision supersedes the earlier sparse precision
recommendations: keep Group32 dense attention for TE/Video, and Group32 dense,
drop or scalar pooling for DiT. Do not alter the current zero-shot routes,
prefix protection, pooling formulas, quantization precision or leaf order.
Sparse generation quality must be assessed through generation-distribution
and semantic deviation in full H3 runs. That work is deferred; operator L1/L2
against dense attention is not its acceptance criterion.

This continuation used **zero new Kaggle GPU sessions** and ran no generation.
The physical-T4-tested baseline module is unchanged. New code is opt-in in
`patches/tilelang_grouped_tp_optimized.py`; production dispatch is unchanged.

## Implemented candidates

- Mode-specific preparation: drop no longer computes unused value centroids
  or materializes their FP32 V reduction. Scalar pooling still computes exactly
  the original centroids. Dense mode has no raw-Q/centroid/route placeholder
  buffers proportional to sequence length; unused kernel arguments are tiny
  read-only placeholders and disappear from generated device signatures.
- Cached counts/dummy geometry, bounded to32 entries and keyed by CUDA stream
  as well as shape/device. This avoids handing asynchronously initialized
  counts to another request stream. Prewarm caches on the capture stream before
  any future CUDA Graph capture; graph execution is not validated here.
- Optional finite-centroid tau0 routing shortcut: same scores, threshold,
  diagonal/global/prefix protections and output mask, while omitting variance/
  square-root work multiplied by zero. The default router remains the original
  path, including exceptional-input behavior. The shortcut is opt-in and needs
  finite Q/centroids; CPU checks confirm exact masks on both real DiT partitions.
- Query tiles32/64/128 and Q-only reuse, in addition to group-at-a-time/full-QK
  caching. Sparse tiles are limited to32/64 so one CTA cannot cross two distinct
  leaf64 routing rows. K leaf64, online softmax order and FP16 P/V remain fixed.
- Launch only query tiles containing real queries. For DiT6814/BQ32 this trims
 214 to213 query CTAs per head. Video BQ64 reduces rank0/rank1 grids from32/26
  to16/13 query CTAs per head. These are launch geometries, not speedup claims.
- Store BND directly, preserving FP16 output rounding before widening to FP32
  for TE/DiT. An optional Video-only FP16 finite-output fusion matches the
  placement of `nan_to_num`; it is a separate candidate because its codegen
  reports scalarization of a two-element output vector loop. It is not enabled
  by default and has no measured performance result.

All current TP2 boundaries remain: TE32Q/4KV heads; DiT28 heads and local packed
width10752; Video32 heads with1024/773 queries against1797 global keys. The
global K mean still includes the true ragged final key leaf. No local-key mean,
INT8 PV conversion, scale narrowing or different sparse selection was introduced.

The high-level candidate returns BND directly; it is not a drop-in replacement
for Kitchen's BHND-returning function. Later integration must remove the matching
caller transpose/reshape and, when finite-output fusion is selected, the redundant
caller `nan_to_num`. No such production hook was installed in this continuation.

## Validation completed without GPU execution

CPU tests cover GQA, rectangular/ragged queries, padded query-tile sizes,
mode-specific allocation, sparse tile guards and exceptional centroid handling.
On the actual captured tensors, all10 component/rank/mode cases have identical
valid QI/KI/scales; scalar VC and routes also match the frozen baseline exactly.
These are implementation-equivalence checks, not a sparse quality experiment.
They do not validate execution of the new CUDA kernels.

Eight configurations lower at the actual TP2 shapes and compile to **SM75
cubins** using the local CPU compiler. No cubin was loaded or launched.

| Candidate | Query tile | Registers/thread | Dynamic shared bytes |
|---|---:|---:|---:|
| Video rank0, group loads, direct BND | 64 | 166 | 21,504 |
| Video rank1, group loads, direct BND | 64 | 168 | 21,504 |
| Video rank0, finite-output fusion | 64 | 168 | 21,504 |
| Video rank1, Q-only reuse | 128 | 255 | 36,864 |
| TE, Q-only reuse, FP32 BND | 64 | 204 | 36,864 |
| DiT dense, Q-only reuse, FP32 BND | 32 | 185 | 28,160 |
| DiT drop, group loads, FP32 BND | 32 | 208 | 24,576 |
| DiT scalar, group loads, FP32 BND | 64 | 202 | 46,336 |

PTXAS reports no stack frame or spills for these configurations. The128-query
Video candidate reaches255 registers/thread, so64 is the first tile to test;
128 remains a higher-risk alternative. These resource figures use CUDA13.3/
Windows and cannot be directly substituted for the previous CUDA12.8/Linux
T4 launch data. They establish compilation feasibility, not speed, achieved
occupancy, memory stalls, or GPU conformance.

## Hardware continuation completed

The bounded physical-T4 continuation has now completed. See
`TP2_OPERATOR_PASS2.md` for paired screen/confirmation results, actual Linux T4
launch resources, communication measurements and profiler limitations. The
recommendations below describe the plan before that pass and are retained for
historical context.

## Original next-pass plan

First compare the frozen implementation with these candidates on identical
captured TP partitions, checking implementation equivalence and measuring the
complete preparation/routing/output API. Do not introduce a dense-relative
sparse quality gate or change the route policy. Start with Video BQ64/direct
BND, DiT drop preparation and DiT Q-only reuse; retain the existing T4 winner
unless a new candidate wins a warmed complete-operator comparison.

`benchmarks/tp2_nccl_microbench.py` is ready for a separate small communication
pass. It derives message sizes from the verified TP manifest, preserves
`NCCL_P2P_DISABLE=1`, checks collective correctness, and tests the DiT INT32 SUM,
FP32 row MAX and Video padded KV all-gather. Run fresh processes for default
versus Simple protocol; optionally test a small channel shortlist afterward.
It needs only PyTorch, with no model, Kitchen, TileLang, activation upload or
CUDA-runtime package replacement. It has not been executed.

After communication measurements, row-linear optimization should preserve
global scales and INT32 SUM before dequantization. External-max quantization
and shared TE column preparation remain follow-up work; neither is claimed
implemented or faster here. Full H3 generation-distribution/semantic checks
remain deferred to a later GPU budget.

## Reproduction and evidence

- `benchmarks/test_grouped_tp_optimized.py --real-data D:/attention_eval/t4_redo_20261013/dataset`
- `benchmarks/lower_grouped_tp_optimized.py` — static SM75 CUDA generation only.
- `benchmarks/compile_grouped_tp_offline.py` — CPU compilation/resource reports.

Outputs, source hashes, exact real-input checks and compiler logs are under
`D:/attention_eval/operator_optimization_tp2_20261016/offline_optimization`.
The earlier T4 reports/builders remain historical reproductions; their
dense-relative diagnostics are not the policy for new sparse quality work.
