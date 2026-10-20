# TP2 row-linear fusion on T4

2026-10-16. Continuation after `TP2_OPERATOR_PASS2.md`. The scope is the current
DiT output-projection and FC2 row-linear operators on two physical TeslaT4 SM75
devices. No H3 pipeline, model weights, attention captures, new dataset upload,
generation-distribution evaluation or sparse-policy change is involved.

## Implementation and contract

`patches/tilelang_row_quant.py` provides three opt-in variants:

1. `external`: retain the current owned rotated-slice copy and local maximum,
   but quantize directly using the globally reduced row maximum. This removes
   the sentinel, concatenation, redundant quantizer maximum scan and INT8
   owned-slice copy.
2. `strided`: additionally read the owned slice directly from the full native
   rotation output, using a TileLang local absolute-maximum reduction. This
   removes the contiguous FP32 slice and the eager absolute-value temporary.
3. `packed`: additionally combine the full-width zero padding and owned-slice
   copy in one TileLang pass before the same native FP32 rotation.

All variants preserve the full global-width FP32 ConvRot GEMM geometry and its
input values. They retain global FP32 MAX, existing native INT8 GEMM, INT32 SUM
before dequantization, and native dequantization. The production hook remains
the frozen `patches/optimized_tp.py` implementation.

The candidate targets finite FP32 activations on the native fallback-rotation
route. Other dtypes, world-size1 and native fused-rotation routes fall back to
the previous row implementation. Global maxima must cover the complete row;
this is not local-scale quantization. Nonfinite-input behavior is not qualified
for these experimental variants.

The quantization contract was checked against the official tagged
[Kitchen0.2.37 CUDA implementation](https://github.com/Comfy-Org/comfy-kitchen/blob/v0.2.37/comfy_kitchen/backends/cuda/ops/int8_linear.cu)
and its build flags. For FP32 it uses a float32 reciprocal127 multiply, a1e-30
scale floor, fast division, nearby-integer rounding and saturation to[-128,127].
Those details differ from Group32 attention quantization. A rounded-division
control is included in unit probes; it is not automatically substituted.
The source archive's URL and SHA256 are saved with the experiment evidence.

## Offline preparation

Twelve target kernels (quantizer, maximum and packing; both rank offsets;
local widths3584/7168) lower and compile to SM75 cubins without loading or
executing any GPU kernel. The local Windows compiler reports no spills or stack
frame. Registers range14–62/thread. These are CPU compilation/resource checks,
not T4 performance or runtime accuracy claims.

## Physical experiment

Private Kaggle job `chinokiki9/h3-tp2-row-fusion`, version1, uses a600s provider
cap and560s internal deadline. One shared setup pins Torch2.11.0+cu130,
Kitchen0.2.37, TileLang0.1.15 and nvcc12.8. Hardware is explicitly checked as
two TeslaT4 SM75 devices. It attaches no datasets.

All variants use default NCCL protocol, four channels and P2P disabled, keeping
communication settings fixed across the experiment. Inputs and weights are
deterministic synthetic tensors at the current source-audited TP2 shapes:

| Operator | M | N | Local GEMM K | Global quantization K | Input activation |
|---|---:|---:|---:|---:|---|
| DiT out projection | 6814 | 5376 | 3584 | 7168 | None |
| DiT FC2 | 6814 | 5376 | 7168 | 14336 | SwiGLU, raw local input width14336 |

This is an implementation-conformance/performance experiment. It does not
measure real linear activation distributions or establish full-generation
quality. It does not change or judge sparse attention approximations.

The hardware unit gate covers zero, tiny, normal, outlier and half-integer
rounding inputs at widths1,257,3584,7168, plus direct checks of padding and
owned-slice maxima. Complete-operator gates compare Q, scales and final outputs
byte-for-byte with the frozen row implementation. Failed gates are retained
and skipped rather than accepted using a loose numeric tolerance.

Timed calls prewarm device clocks, align ranks, run10 fixed warm-ups and collect
15 GPU-event and wall-clock samples. Selection uses the slower rank's median
complete API. Quantization API timing is separate. Finalists are repeated in a
fresh process with another order and identical inputs. Peak extra allocated
memory is measured with the same persistent fixture tensors resident.

Nsight Systems is restricted to FC2 frozen/selected, at most two reports.
No Nsight Compute retry is scheduled because counters were previously denied.

## Results

Version1 completed on two TeslaT4 SM75 devices. All three candidates passed;
there were no failed gates. Selection favored `packed` for both modules.

Screen, slower rank's complete-API median, milliseconds:

| Operator | Frozen | External max | Direct strided slice | Packed padding + strided + external max |
|---|---:|---:|---:|---:|
| Out projection | 33.745 | 31.644 | 30.304 | 29.735 |
| FC2 | 53.955 | 50.561 | 47.243 | 46.453 |

The fresh-process repeat used identical inputs and a different order:

| Operator | Frozen API ms | Packed API ms | Latency reduction | Frozen → packed quantization API ms |
|---|---:|---:|---:|---:|
| Out projection | 33.760 | 30.209 | 10.5% | 11.475 → 8.088 |
| FC2 | 54.943 | 47.612 | 13.3% | 26.732 → 20.001 |

Quantization API includes activation, padding, native rotation, local maximum,
global MAX and quantization. It excludes INT8 GEMM, INT32 SUM and dequantization.
It is measured separately; do not add its median to medians from other phases.
Complete-API wall medians are33.828→30.301ms and55.035→47.696ms respectively.
The improvements reproduce in both rounds, but these are two processes within
one allocation, not independent repeated Kaggle sessions or a pipeline speedup.
Do not combine these percentages with a previous allocation's NCCL percentages.

## Equivalence and rounding evidence

All40 fast-division unit pattern/rank cases match native Q and scales exactly.
Both ranks also pass direct padding/max checks for both target widths. All24
timed operator configuration/rank cases (16 screen,8 confirmation) have exact
Q, scales and outputs. Final output SHA256 values match across variants and
both rounds separately for each rank/module; the full profiled outputs match
as well. This supports implementation equivalence on these inputs, not arbitrary
nonfinite values or real generation quality.

The `__fdiv_rn` control changes13,382 INT8 elements per rank; every difference
occurs in the constructed half-integer probes. Per width the differing counts
are5,317,4358 and8702. `__fdividef` matches the native CUDA contract throughout.
Thus substituting a more accurate division can alter quantized threshold
decisions and fail byte equivalence. The selected operator retains native fast
division; no error tolerance was used to accept changed Q.

## Memory result

Peak **extra PyTorch allocated memory** during complete API measurement, with
the same fixture/reference tensors resident:

| Operator | Frozen MiB | Packed MiB | Reduction |
|---|---:|---:|---:|
| Out projection | 466.641 | 372.641 | 94.000MiB /20.1% |
| FC2 | 1117.922 | 931.602 | 186.320MiB /16.7% |

The memory result is identical in screen and repeat. Even `external` has the
same lower peak: the candidate releases the padded input before creating any
owned contiguous slice, whereas the frozen expression creates the slice while
padding and full rotation output are still live. The full native rotation
remains the peak allocation boundary. Removing later slice/sentinel buffers
helps traffic and latency without further lowering that boundary in this test.
These figures are not device-wide VRAM, allocator reserved memory or a full H3
capacity estimate.

## Profiler evidence and remaining bottlenecks

Both planned FC2 Systems reports are usable, with three complete calls per
device. Process-qualified correlation IDs attribute all captured GPU kernels.
The frozen API launches17 kernels/call; `packed` launches10. It removes eager
absolute-value, slice, sentinel and concatenation work while retaining the same
native rotation, INT8 GEMM, collective and dequantization calls.

First-device diagnostic kernel averages, milliseconds per captured call:

| Work | Frozen | Packed |
|---|---:|---:|
| Outer buffer/reduction/copy kernels + native quantizer | 11.324 | replaced by the next three rows |
| Combined padding/copy | included above | 2.609 |
| Owned-slice absolute maximum | included above | 0.731 |
| External-global-max quantizer | included above | 1.016 |
| Native FP32 rotation | 7.734 | 8.037 |
| Native INT8 GEMM | 8.139 | 7.610 |
| INT32 SUM | 18.679 | 19.162 |
| Eager SwiGLU | 3.997 | 3.985 |
| Native dequantization | 1.219 | 1.215 |

The changed buffer/quantization work falls from11.324 to4.356ms/call, accounting
for about7ms of removed GPU work. Unchanged native kernels vary slightly between
captures. This is not evidence that GEMM or rotation was optimized, and no
collective speedup is claimed. Global MAX timings include rank-arrival waiting.
Profiled averages are diagnostic and are not the warmed API medians above.

Measured Linux T4 launch resources for the new FC2 kernels:

| Kernel | Threads | Registers/thread | Dynamic shared bytes | Reported local bytes/thread |
|---|---:|---:|---:|---:|
| External-max quantization | 256 | 62 | 0 | 0 |
| Local absolute maximum | 256 | 40 | 1024 | 0 |
| Combined padding | 256 | 16 | 0 | 0 |

The largest remaining individual phase is the roughly19ms INT32 SUM, followed
by rotation and GEMM near8ms each. Kernel interval coverage is about99.9%, but
that includes NCCL kernels and waiting; it is not achieved occupancy or useful
SM utilization. Nsight Compute counters remain unavailable, so this study does
not diagnose measured memory stalls or Tensor Core utilization. Output projection
was timed and validated but not separately profiled, keeping this batch small.

## Decision and next work

Keep `packed` as the verified finite-FP32 operator candidate for the two current
DiT shapes. Call `tilelang_row_quant.row_linear(..., variant='packed')` explicitly;
the production `optimized_tp` hook and TE/Video routes were not changed.
The other variants remain useful ablations, with external max alone giving
about6.2–6.3% screen improvement before the additional memory-traffic changes.

The next fusion target is the two eager FC2 SwiGLU kernels plus padding. It
must reproduce the native intermediate FP32 rounding and activation formula;
an algebraically simplified sigmoid/multiply is not automatically equivalent.
Rotation replacement and communication restructuring are larger changes requiring
separate gates. The present pass deliberately preserves their current TP2
boundaries. Sparse policy and scalar pooling are unaffected.

The five recorded stages total347.23s, including197.18s of shared package/runtime
setup; screen93.52s, confirmation25.68s and Systems30.85s. The notebook's own
final event is near371s. These are recorded durations, not Kaggle quota billing.
No retry, full H3 run or additional GPU job was needed.

## Reproduction and artifacts

Builder: `build_tp2_row_fusion_job.py`; hardware harness:
`benchmarks/tp2_row_fusion.py`. Offline lowering/compilation:
`benchmarks/lower_row_quant.py` and
`benchmarks/compile_grouped_tp_offline.py --input <offline-folder>`.
CPU analysis uses `benchmarks/summarize_tp2_row_fusion.py` and the existing
process-safe trace export/attribution scripts.

Version-bound raw results, all15-sample distributions, source hashes, rounding
probes, two Systems reports and local SQLite exports are under
`D:/attention_eval/operator_optimization_tp2_20261016/row_quantization/v1/row_fusion`.
`paired_summary.json`, `verified_candidates.json` and `final_evidence_index.json`
record the selection, qualifications and SHA256 evidence index. The five local
submitted source texts still match the executed notebook; all21 downloaded
artifacts pass collection SHA256 verification.
