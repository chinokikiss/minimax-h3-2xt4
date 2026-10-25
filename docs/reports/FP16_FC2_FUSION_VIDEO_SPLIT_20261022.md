# FP16 residual, FP32 SwiGLU/quantization, transport and Video split — 2026-10-22

This experiment implements three FP32 preparation candidates while retaining
FP16 residual storage, the original INT8 GEMM, and either Native INT32 SUM or
the existing power-of-two scaled FP16 transport. It also independently locates
the first differing stage of the 261-token trained Video decoder split.
At experiment capture, production defaults, pinned ComfyUI, local conda and
core versions were unchanged. The subsequent user-authorized default promotion
is recorded below and in the [integration follow-up](H3_FP16_FUSED_DEFAULT_20261022.md).

## Implemented operator and controls

`benchmarks/h3_fp16_fc2_experiment.py` contains the experimental operator.
It consumes FP16 packed gate/up values; TE's separate producers are concatenated
before activation, so their product is never first materialized in FP16.
FP32 preparation produces INT8 codes and a shared FP32 row scale. The existing
`parallel_int8.int_mm` implementation, installed by `optimized_tp.enable`,
computes the original INT8→INT32 GEMM. The reference reduction is INT32 SUM.
Dequantization and any LoRA addition remain FP32 until one FP16 output store.
RMSNorm and residual-flow conventions otherwise remain unchanged.

| Preparation | Computation / storage | Numerical contract |
|---|---|---|
| `control` | Existing FP32 SwiGLU + ConvRot + global row quantization | Current FP32 preparation control; DiT uses the existing owned-FP32 rotation option |
| `fused_stored` | FP16 input loads, FP32 SwiGLU + regular H4 ConvRot + group maxima in one producer; retain FP32 rotated values for quantization | Same mathematical regular ConvRot256; butterfly FP32 reduction order differs from Native SGEMM |
| `fused_recompute` | First pass computes group maxima; second pass recomputes FP32 SwiGLU/regular ConvRot and writes INT8 directly | Same global-scale dependency; avoids the full FP32 rotated tensor, at the cost of recomputation |

Kitchen uses a **regular H4 Kronecker matrix**, not the ordinary Sylvester
Hadamard. The fused kernel implements four regular H4 stages and normalization
by 1/16. A local compile/arithmetic preview compared it to Kitchen's actual
matrix, including an activation product above FP16 range; the recomputed INT8
result matched quantization of its own rotated values. This preview used SM120
and is explicitly not target timing or T4 qualification.

The TP2 maximum is still global: reduce local group maxima, AllReduce MAX across
the two ranks, then round/clamp INT8 with that shared scale. A one-pass local
scale would change the operator. Neither fused candidate removes that dependency.

Each preparation is tested with `int32` and `scaled_fp16` transport. The latter
uses the existing encode/collect/fused-dequantization codec, followed by the
same FP32 LoRA/output handling. A separately computed codec reference checks
its arithmetic. Scaled FP16 SUM is lossy relative to INT32; it is not described
as byte-exact Native transport.

## Physical matrix and measurement scope

The bounded [dual-T4 batch](https://www.kaggle.com/code/pamcks/h3-fp16-fc2-video-localization)
uses the CPU-prepared pinned-core overlay, native Kaggle Torch, frozen source
and immutable trained component assets from the previous successful gates.
It has an eight-minute internal cap, no internet, and separate processes for
the FC2 matrix and Video localization so a failed candidate does not suppress
the independent Video investigation. No full H3 sampler is run.

Three preparation choices × two transports × three components give **18 cases
per rank**:

- TE block 0 / down projection: S353, local K12800, output N5120. Inputs are
  seeded synthetic TE states, not a real TE trajectory.
- DiT block 0 / FC2: S6814, local K7168, output N5376. Inputs select 6814 rows
  from the retained S7895 first-block input; this is not a native S6814 capture.
- DiT block 49 + trained LoRA: S513, same projection dimensions, reusing the
  first-block input as a stress case, not that layer's own trajectory.

Within each component, all candidates share the same captured FP16 gate/up
input and trained INT8 weights. Whole-block results use the same residual
input, conditioning and Native FP32 activation-flow reference. No dense-FP32
replacement or sparse-quality surrogate is introduced.

Preparation, complete row API and complete trained block each receive two
warmups and seven synchronized measurements. Barriers are outside timed regions;
the aggregate uses the maximum of corresponding rank samples. First candidate
preparation/compilation time is separate. The first timed API call is also
retained, but is not mislabeled as cold compilation. GPU clocks are not locked.

Peak allocation is a **benchmark working-set** measurement: retained control
codes and the captured FP16 producer coexist consistently across candidates.
Reserved memory includes allocator history. These are not full-pipeline VRAM
peaks. Post-warm PyTorch traces cover each stress-row candidate; kernel-duration
sums are not hardware counters or critical-path attribution.

## Video localization method

The actual trained 36-block decoder runs Native and physical query-parallel
execution on 261 tokens split 256/5, using latent seed 732 and the Native latent
denormalization constants. Hooks compare all attention, MLP, final-normalization
and final-projection outputs against the corresponding Native token slice.
Raw decoder tensors and the finalized valid image frame are reported separately.

At the first differing stage, controlled replays hold the full Native input
fixed and compare full/local QKV+norm, Q/K RoPE, attention with identical Q/K/V,
output projection with identical input, FFN input projection and SwiGLU output
projection. The five-row case also pads to 128 rows as a diagnostic, then
discards dummy outputs. This is a replay experiment, not a production fix.

A source-based hypothesis is the pinned Kitchen SM75 predicate
`m <= 128 or n >= 2*k`: contractions at full M261 and local M5 can choose
different epilogues. Padding to 128 does not cross that predicate's threshold.
Actual primitive evidence, rather than that hypothesis alone, determines the
localization conclusion. The local one-block API preview exercised all replay
calls successfully; it is not used to infer T4 numerical behavior.

## Reproducibility and evidence

Local root: `D:\attention_eval\fp16_fc2_video_20261022`.

- `batch_v1`: immutable submitted notebook, metadata, per-file source hashes
  and exact CPU/assets/prior-result receipt bindings.
- `local_video_replay_preview.log`: explicitly local SM120 harness check.
- `output_v1`: version-bound final artifacts and SHA256 download manifest,
  populated by one final collection after the run window.
- `summarize_fc2_video_experiments.py`: offline device/source/hash verification,
  aligned-rank timing aggregation, numerical diagnostics and Video stage summary.

The first submission returned a non-JSON response; one source-retrieval check
returned HTTP 403. The user explicitly authorized a retry, which successfully
created version 1. No repeated status/list/live-log polling is used. Local
pre-submission verification passed **34 focused tests**.

Physical results are appended only from downloaded artifacts. No generation,
audio/video semantic, sparse-distribution or production-default acceptance is
implied by operator-level completion.

## Completed physical results

Version 1 completed in **358.94 seconds**, including the 34.50-second device/core
gate, 222.43-second FC2 process and 40.77-second Video process. Both devices
were distinct Tesla T4s, SM75. Actual software was Torch **2.11.0+cu128**,
CUDA **12.8**, Kitchen **0.2.37**, TileLang **0.1.13**. Downloaded source hashes
match the submitted capsule and the prior receipt binding. All **18 cases on
each rank** completed, with finite row and block outputs and exact agreement
with each candidate's independent codec/dequantization/LoRA reference. This
checks implementation arithmetic; it does not make lossy candidates Native-exact.

The [aggregate evidence](../data/fp16_fc2_fusion_video_split_20261022.json)
contains all aligned samples, both Video stage records, source bindings and
profiler summaries. Full version-bound artifacts remain under `output_v1`.

### Preparation, complete row and trained block latency

All times below are medians in milliseconds, taking the slower rank in each
corresponding sample. Row speedup compares with the corrected FP32-preparation /
INT32-SUM control in the same component, not the old overflowing FP16 path.

| Component | Preparation | SUM | Preparation ms | Row ms | Row speedup | Block ms |
|---|---|---|---:|---:|---:|---:|
| TE0 S353 | control | INT32 | 2.617 | 5.707 | 1.00× | 24.584 |
| TE0 S353 | control | scaled FP16 | 2.616 | 5.171 | 1.10× | 25.786 |
| TE0 S353 | stored | INT32 | 0.910 | 3.867 | 1.48× | 31.619 |
| TE0 S353 | stored | scaled FP16 | 0.482 | **1.906** | **3.00×** | **22.343** |
| TE0 S353 | recompute | INT32 | 1.342 | 4.305 | 1.33× | 24.018 |
| TE0 S353 | recompute | scaled FP16 | 0.594 | 2.007 | 2.84× | 22.743 |
| DiT0 S6814 | control | INT32 | 12.062 | 41.500 | 1.00× | 265.846 |
| DiT0 S6814 | control | scaled FP16 | 11.917 | 35.818 | 1.16× | 264.244 |
| DiT0 S6814 | stored | INT32 | 4.568 | 39.356 | 1.05× | 260.300 |
| DiT0 S6814 | stored | scaled FP16 | 5.257 | **27.727** | **1.50×** | **253.446** |
| DiT0 S6814 | recompute | INT32 | 6.820 | 36.995 | 1.12× | 261.825 |
| DiT0 S6814 | recompute | scaled FP16 | 7.030 | 35.577 | 1.17× | 262.168 |
| DiT49+LoRA S513 | control | INT32 | 1.251 | 8.458 | 1.00× | 39.012 |
| DiT49+LoRA S513 | control | scaled FP16 | 1.436 | 8.739 | 0.97× | 40.070 |
| DiT49+LoRA S513 | stored | INT32 | 0.797 | 10.935 | 0.77× | 39.033 |
| DiT49+LoRA S513 | stored | scaled FP16 | 0.427 | 6.357 | 1.33× | 37.655 |
| DiT49+LoRA S513 | recompute | INT32 | 0.999 | 9.303 | 0.91× | 42.461 |
| DiT49+LoRA S513 | recompute | scaled FP16 | 0.506 | **6.227** | **1.36×** | **37.158** |

Stored fusion with scaled transport is the strongest measured throughput
candidate for TE0 and large DiT0. Their block medians improve by **9.1%** and
**4.7%** respectively, much less than row gains because other block work remains.
Recompute with scaled transport has the lowest stress-row median and lowers
its block median by **4.8%**, but stored/recompute stress timing ranges overlap.
Recompute is primarily a memory tradeoff: on DiT0, row working-set peak falls
from about **1116 MiB** for stored+scaled to **930 MiB** for recompute+scaled,
while row latency increases from 27.73 to 35.58 ms. These are retained benchmark
allocations, not production VRAM guarantees.

Do not infer additive or independently stable improvements from these numbers.
Cases ran in fixed order in one batch, without locked clocks. For example,
identical stored preparation measured 0.797 ms in the stress INT32 case and
0.427 ms in its scaled case even though preparation itself does not depend on
the SUM choice. TE stored/INT32 block samples span 22.69–38.63 ms; the 31.62-ms
median does not establish a structural block regression. Large DiT control
row samples span 39.23–59.03 ms, versus 26.61–37.08 ms for stored+scaled. The
complete rows support further work on fusion+transport; a randomized paired
repeat is needed before assigning small differences to a particular kernel.

First preparation calls including fused compilation were **2.45–8.22 seconds**
on rank 0. They are excluded from these steady-state medians. Changing from
INT32 to scaled transport reuses the already compiled preparation; its own
first encode/epilogue work is also outside the repeated steady-state samples.

### Numerical result and overflow mechanism

FP16 gate/up storage is retained. Their SwiGLU product, regular ConvRot,
quantization scale, dequantization and LoRA arithmetic use FP32. The old
stress-case failure was a branch-product overflow before quantization, which
residual scaling alone cannot prevent; see the
[prior controlled diagnosis](T4_NATIVE_TORCH_VALIDATION_20261021.md).
This batch exercises the corrected control and its fused approximations.

| Component | Control block relative L2 vs Native FP32 flow | Fused + INT32 block relative L2 | Fused + scaled FP16 block relative L2 | Fused INT8 code disagreement |
|---|---:|---:|---:|---:|
| TE0 S353 | 0.00136602 | 0.00136599 | 0.00136683 | 0.000598% |
| DiT0 S6814 | 0.00272833 | 0.00272837 | 0.00276694 | 0.000732% |
| DiT49+LoRA S513 | 0.01515660 | 0.01515664 | 0.01515993 | 0.000286% |

Code disagreement aggregates both ranks against the same captured control
input. Stored and recompute have matching numerical diagnostics in this batch.
Fused/INT32 row relative L2 is approximately **5.9–7.1e-5**; with scaled transport
it is approximately **4.0–4.4e-4**. Thus transport rounding contributes more row
deviation than butterfly summation order here. No nonfinite row/block outputs
were observed in either rank. These component checks do not qualify late-step
generation, underflow throughout a trajectory, or sparse semantic preservation.

### What the warm row traces establish

The six stress-row candidates on each rank have post-warm CPU/CUDA traces.
Rank-0 kernel counts fall from **20** for control/INT32 to **18** for stored/INT32
and **17** for recompute/INT32; corresponding scaled cases have 21, 19 and 18.
The original `cutlass_75_tensorop_i8816gemm_s8_128x128_tn_align16` remains present:
this experiment does not replace the INT8 GEMM.

In the rank-0 trace, control/INT32 records 7.63 ms of summed kernel duration,
including 4.46 ms of NCCL activity. Recompute+scaled records 6.27 ms, including
3.26 ms of NCCL activity. The INT32-SUM kernel changes from 1.42 ms to an
FP16-SUM kernel of about 0.79 ms in the control+scaled trace. However, the
LoRA activation AllGather still contributes roughly **2.4–3.0 ms** across the
traced candidates and is unaffected by the SUM payload codec. It reconstructs
Native full input geometry before LoRA A; it is not a quantizer communication.
That remaining exchange and FP32 LoRA work limit further row acceleration.

These are one-call traced duration sums including collective waiting, not
bandwidth measurements or proof of hardware memory stalls. Rank-1 recompute /
INT32 has a particularly long collective wait, illustrating why one trace
cannot rank candidates. This batch does not collect new Nsight Compute counters
or new Nsight Systems evidence; its PyTorch traces cannot substitute for them.

## Video 261: actual first-difference localization

Both ranks compared **74 stage outputs** each: 36 attention outputs, 36 MLP
outputs, final normalization and final projection.

| Observation | Rank 0: 256 query rows | Rank 1: 5 suffix rows |
|---|---|---|
| All 36 attention outputs | Exact tensor equality | Exact tensor equality |
| All 36 MLP outputs | Exact tensor equality | Exact tensor equality |
| Final `norm_out` | Exact tensor equality | Exact tensor equality |
| Final `proj_out` | **First mismatch** | Exact tensor equality |
| `proj_out` replay with identical Native normalized input | **Mismatch reproduced** | Exact tensor equality |

The replay isolates final projection geometry: Native computes M261,
K2048→N3072, whereas rank 0 computes M256 with identical input rows and weights.
Its **145,205** differing values equal the count in the assembled raw decoder
output (786,432 elements); max absolute error is **0.00390625**. The valid last
image frame has **33,668 / 196,608** differing values, max absolute error
**0.0004472733**, reproducing the prior regression. No nonfinite values occur.
The five suffix tokens on rank 1 are discarded from valid pixel output.

This rules out accumulated attention/MLP drift and the hypothesized short-row
Kitchen contraction dispatch as the cause of this particular decoder mismatch.
The evidence identifies **shape-dependent final-projection arithmetic** as the
first cause; it does not identify a specific cuBLAS algorithm, split-K reduction
or epilogue instruction. The model declares an ordinary final Linear with
checkpoint dense weight `[3072, 2048]`, unlike its quantized transformer linears.
No block replay or padded-128 intervention was invoked because the first
difference was outside the block stack. The source hypothesis above is retained
as the pre-experiment hypothesis, not an accepted explanation.

The narrow repair to test next is to gather the final hidden/normalized rows
and evaluate `proj_out` once on rank 0 at Native M261, then reshape pixels.
This preserves the proven matching transformer partition and removes the
localized projection-shape difference. Padding rank 0 from M256 to M261 is a
cheaper counterfactual, but zero-filled suffix rows do not guarantee Native
quantizer or GEMM behavior and must be validated. Neither repair has been
measured in this batch, so **261-token query-parallel byte identity remains
unqualified**. No production attention or fallback policy was changed.

## Disposition

Keep the corrected FP32-preparation control, stored fusion and recompute fusion
as explicit experimental choices; scaled transport is wired to all three.
Stored+scaled is the next throughput candidate, and recompute+scaled is the
memory candidate. Retain INT32 SUM when its exact integer reduction is required.
Before production promotion, repeat the winning shapes in randomized paired
order, validate actual TE and late-layer trajectory inputs, and test the narrow
Video final-projection repair. Production defaults and dependency/source locks
remain unchanged by this experiment.

## Subsequent default promotion - 2026-10-22

After reviewing these results, the user requested **FP16-fix + fused_stored +
scaled FP16 transport** as the pipeline default. The production implementation
now resides in `patches/tilelang_swiglu_quant.py`, selected through
`H3OperatorPolicy`, loader defaults and both bundled workflows. FP32 LoRA
addition precedes the final FP16 store. Explicit prior preparation and INT32
transport choices remain available. See the [production follow-up](H3_FP16_FUSED_DEFAULT_20261022.md)
for exact dispatch, fallback, compatibility, local integration checks and
rollback settings. The T4 tables above still describe the immutable measured
experiment, not a new full-pipeline run or generation-quality acceptance.
