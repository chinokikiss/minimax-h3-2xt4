# TP2 operator optimization, second physical-T4 pass

2026-10-16. Operator-only continuation. No H3 model construction, model-weight
download, denoising, video generation or audio generation is part of this job.
The existing dense, drop and optional scalar-pooling approximations are retained.
Sparse-versus-dense L1/L2 is not a quality gate. Full-generation distribution and
semantic validation remain deferred.

## Reproduction and measurement contract

Private Kaggle job: `chinokiki9/h3-tp2-operator-optimize`, version1.
Builder: `build_tp2_operator_optimization_job.py`. The provider timeout is720s;
the notebook enforces a690s internal deadline, bounded subprocess stages and
process-group cleanup. Runtime installation is shared across all experiments.
This is one bounded session, not an open-ended autotuning sweep.

Evidence root:
`D:/attention_eval/operator_optimization_tp2_20261016/operator_pass2`.
The submitted source hashes and current TP2 manifest are retained. The builder
checks the actual TP source hashes before submission. Torch2.11.0+cu130,
Kitchen0.2.37, TileLang0.1.15, and nvcc12.8 are pinned; execution requires two
actual TeslaT4 devices with compute capability7.5. No SM120 fallback is allowed.

Attention inputs are existing real H3 captures sliced according to the current
TP2 implementation. They are not a newly captured TP2 denoising trajectory.
DiT V preserves the packed local-QKV token stride10752. Video uses local query
lengths1024/773 with global1797-key attention. Its KV exchange is excluded from
the new attention screen and measured separately as a raw collective.

Row-linear inputs/weights are deterministic synthetic tensors at the verified
current DiT O/FC2 MNK shapes. The harness calls the existing row-linear API,
including native FP32 rotation geometry, global FP32 row MAX, INT8 GEMM,
INT32 SUM and native dequantization. This is not real linear-activation quality
validation. Outputs are byte-hashed separately on each rank across NCCL settings.

Both ranks run simultaneously. Every timed case prewarms GPU clocks locally for
at least1s, aligns ranks, performs10 identical operator warm-ups, and collects15
samples. Loading/JIT is recorded separately. Comparisons use the slower rank's
median complete-API time. Separate phase medians must not be added to infer a
microchain latency. Screening configuration order is randomized, and finalists
are repeated in a fresh process with another order if the budget allows.

Candidate acceptance requires exact valid QI/KI/scales and sparse routes; scalar
centroids must also match. A same-operator output check detects implementation
regressions. It does not judge the approximation against native dense attention.
No new INT8 PV, narrower scales, different route policy or softmax leaf ordering
is introduced.

Nsight Systems captures finalists only. Nsight Compute hardware counters were
previously denied by Kaggle (`ERR_NVGPUCTRPERM`), so no additional quota is spent
retrying them. Kernel intervals and launch resource usage do not establish
achieved occupancy, memory-stall causes or Tensor Core utilization.

## Results

All15 configurations completed on both ranks, with no failed candidates. All50
timed configuration/rank cases (screen plus confirmation) had bitwise-identical
outputs to their corresponding frozen operator and identical quantized operands,
scales and routes. Scalar centroids also matched. This is evidence for these
fixtures, not a claim of bitwise equality on every possible input.

The baseline already uses fused Group32 preparation and the previously selected
load strategy. These gains are incremental to that baseline, not comparisons
with eager preparation or native attention.

| Family and selected configuration | Screen frozen → selected ms | Fresh-process frozen → selected ms | Confirmation latency reduction |
|---|---:|---:|---:|
| TE, BQ32/group loads/direct BND | 0.826 → 0.658 | 0.900 → 0.672 | 25.3% |
| DiT dense, BQ32/Q-only reuse/direct BND | 75.363 → 69.935 | 79.886 → 73.004 | 8.6% |
| DiT drop, BQ32/pruned preparation/tau0/direct BND | 52.308 → 49.493 | 54.974 → 50.614 | 7.9% |
| DiT scalar, BQ64/tau0/direct BND | 66.146 → 52.437 | 69.416 → 54.446 | 21.6% |
| Video, BQ64/group loads/direct BND | 2.484 → 2.100 | 2.511 → 2.110 | 16.0% |

Absolute timings drift between rounds; each reduction above uses its own round's
baseline. Fifteen samples are not15 independent Kaggle sessions. The direction
was reproduced in a fresh process, with a different candidate order, on the same
allocated devices. TE has visible host/launch jitter; do not promise a fixed25.3%
reduction under a complete model workload.

No changes were made to the production attention hook or global NCCL environment.
The tested operator implementation is
`patches/tilelang_grouped_tp_optimized.py`. `verified_candidates.json` records
the selected arguments, hashes and qualifications. Its BND-returning interface
requires removing the caller's matching transpose/reshape when integrated.
Video's selected path retains external `nan_to_num`.

## What the profiler supports

Nine of ten attention reports and both row-linear reports were exported and
analyzed locally, without further GPU activity. The attention process completed
its conformance runs, but the90s profiler stage expired during report export.
The selected Video report is unavailable; its speed result is supported by both
unprofiled rounds, not by a new Video-winner trace. Some second-rank captures also
end before the final queued kernel. Counts below use the first device's three
fully observed calls, rather than assuming both devices always captured three.

| API | Preparation launches/call | Routing launches/call | Attention + output launches/call | Total |
|---|---:|---:|---:|---:|
| TE frozen → selected | 18 → 13 | 0 | 3 → 1 | 21 → 14 |
| DiT dense frozen → selected | 18 → 13 | 0 | 3 → 1 | 21 → 14 |
| DiT drop frozen → selected | 28 → 17 | 32 → 21 | 3 → 1 | 63 → 39 |
| DiT scalar frozen → selected | 28 → 23 | 32 → 21 | 3 → 1 | 63 → 45 |

Direct BND/native-dtype stores remove two output conversion launches in these
TE/DiT traces while retaining the original FP16 rounding boundary. Geometry
reuse and mode-specific preparation remove allocations/fills; drop does not
compute unused V centroids. The tau0 shortcut removes variance/square-root work
that the current policy multiplies by zero. It preserves observed routes,
including the874-token prefix protection and existing mandatory leaves.
Its finite-centroid precondition must remain explicit; default routing retains
the original exceptional-input behavior.

For DiT dense, the profiled first-device attention/output phase changes from
67.609 to61.840ms/call, while preparation remains5.834→5.827ms. Q-only reuse
keeps Q and its scales across all key leaves without caching all K groups.
Measured launch resources increase from197 to220 registers/thread and24,576
to28,160 dynamic shared bytes. The complete API wins despite that resource cost;
the earlier full Q/K cache was not selected.

For drop, preparation changes8.905→6.546ms/call, routing1.291→1.234ms, and
attention/output39.141→37.276ms in the diagnostic trace. The screen's otherwise
similar BQ32 candidate with original routing takes49.562ms versus49.493ms for
the tau0 shortcut: the shortcut alone is a small, unresolved difference, not
the explanation for the complete drop improvement.

For scalar, preparation remains8.877→8.868ms and routing1.273→1.226ms, while
attention/output falls53.236→39.569ms/call. The BQ32 shortcut-only screen gains
less than1%; BQ64 is the major improvement. It processes a full64-query route
leaf per CTA instead of two half-leaf CTAs, while retaining64-key leaf order.
Actual launch resources rise from195 to255 registers/thread and32,896 to46,336
dynamic shared bytes. This is a resource-heavy winner with little register
headroom; it does not justify another blind tile-size expansion.

These profiled phase numbers are diagnostic averages from a separate capture,
not the warmed median API measurements and not quantities to add across runs.
They support the conclusion that the long DiT attention loop remains the main
attention cost. They do not isolate memory stalls, barriers or Tensor Core issue
efficiency without unavailable hardware counters. Local Windows cubin resource
estimates differed from these Linux T4 launch resources; use the latter here.

The Video finite-output fusion was screened but not selected:2.130ms versus
2.100ms for BQ64 with external finite handling. The compiler warns that a
two-element vector loop was scalarized. That warning is a possible explanation,
not a measured stall cause. Removing a launch alone did not win this comparison.
Do not promote that fusion or claim the new Group32 Video path beats the active
Kitchen INT8 backend; this batch compares Group32 implementations only.

## TP2 communication and current row-linear APIs

The146,530,176-byte DiT INT32 SUM,27,256-byte FP32 MAX and8MiB/rank padded Video
KV all-gather are derived from the verified TP2 manifest. Their correctness is
checked before timing. Default protocol leaves `NCCL_PROTO` unset.

| Setting | DiT INT32 SUM ms | FP32 row MAX ms | Video raw padded KV all-gather ms |
|---|---:|---:|---:|
| Default protocol/channels | 20.619 | 0.088 | 1.435 |
| Forced Simple/default channels | 20.789 | 0.096 | 1.474 |
| Default protocol/four channels | 18.409 | 0.086 | 1.446 |

Forcing Simple is not useful in this allocation. Four channels reduce SUM
latency10.7%, with a small Video difference within the measured sample ranges.
The complete unchanged row-linear APIs corroborate the large-message result:

| Current TP2 row API | Default channels ms | Four channels ms | Latency reduction |
|---|---:|---:|---:|
| DiT out projection | 35.154 | 32.760 | 6.8% |
| DiT FC2 | 53.953 | 51.630 | 4.3% |

All output hashes match across settings for both modules and ranks. This is one
paired setting comparison, not an independently repeated communication campaign.
The candidate uses `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, keeps
`NCCL_P2P_DISABLE=1`, and leaves protocol automatic. NCCL settings affect the
communicator; they cannot be switched per operator after initialization. Keep
this as an explicit deployment candidate until a representative microchain is
rechecked. No whole-pipeline speedup is inferred.

The selected-setting Systems reports still show RING_LL INT32 SUM kernels.
On the first device, the out projection spends18.055ms/call in SUM,5.680ms in
FP32 rotation,5.124ms in INT8 GEMM,5.084ms in outer reductions/copies/packing,
0.862ms in row quantization and1.199ms in dequantization. FC2 spends18.743ms
in SUM,7.708ms in rotation,7.447ms in GEMM,9.944ms in outer buffer work,
3.994ms in eager activation,1.365ms in quantization and1.222ms in dequantization.
The tiny MAX collective can include rank-arrival waiting; its elapsed kernel
duration is not pure transfer time.

Thus replacing INT8 GEMM alone leaves most current row cost intact. Copies,
rotation and the collective are larger combined targets. CUDA launch API CPU
time is below1ms per three-call row capture, while GPU work lasts over100ms;
the long `cudaDeviceSynchronize` is waiting for queued GPU work. About99.9% kernel
interval coverage includes NCCL activity and does not mean99.9% useful SM work.
Sampled device memory peaks at2,225MiB on each T4 in this operator session; this
is neither per-operator allocated memory nor a full-model capacity estimate.

Historical36ms SUM measurements came from another allocation/run. Do not call
their difference from this session's20.619ms an optimization gain. Only the
within-session channel comparison supports the6.8%/4.3% row improvements above.

## Next optimization order

The external-max/packing work in items2–3 below has now been implemented and
measured on T4. See `TP2_ROW_FUSION_RESULTS.md` for staged ablations, confirmed
complete-API gains and memory results. The list records the priorities at the
end of this attention/communication pass.

1. Keep these five attention configurations as explicit operator candidates.
   Preserve scalar pooling and the existing sparse policy. Integrate BND stores
   only with matching caller layout changes; do not silently double-transform.
2. For row-linear, implement a quantizer taking the already reduced global row
   maximum. This can remove sentinel allocation/copy, concatenation, redundant
   maximum calculation and the INT8 slice copy while retaining native scale,
   rounding and clamping. Validate output bytes before timing. Keep INT32 SUM
   before dequantization.
3. Fuse owned-slice packing and local absolute-maximum reduction around the
   existing FP32 rotation. Preserve the global rotation GEMM geometry. Replacing
   that rotation with a different reduction order needs separate validation;
   it is not an automatic precision-neutral optimization.
4. For long DiT attention, test a narrow SM75 kernel change that reduces repeated
   K/scale movement or synchronization, with register/shared-memory limits from
   these actual launches. Avoid a broad tile/cache search and prioritize full
   API confirmation over isolated GEMM throughput.
5. Routing fusion and additional finite-output fusion are lower priority given
   their small measured contribution or failed screen. A new full H3 run is not
   required for these operator studies and is not scheduled.

The session records537.55s of subprocess-stage time, including176.41s of shared
package/runtime setup and the capped profiler export. This is a duration ledger,
not Kaggle's GPU billing meter. No extra session was launched merely to recover
the missing Video profile.

## Saved artifacts and reproduction

Version-bound raw artifacts, source hashes,15-sample distributions,
`paired_summary.json`, `verified_candidates.json`, process-safe profiler
attribution and the evidence SHA256 index are under
`D:/attention_eval/operator_optimization_tp2_20261016/operator_pass2/v1/operator_pass2`.

CPU-only analysis:

```
python benchmarks/export_tp2_traces.py <artifact-root>
python benchmarks/attribute_tp2_trace.py <sqlite-files> --output <artifact-root>/profile_attribution.json
python benchmarks/summarize_tp2_optimization.py <artifact-root>
python benchmarks/finalize_tp2_optimization.py <artifact-root>
```
