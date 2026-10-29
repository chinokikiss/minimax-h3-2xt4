# DiT VRAM fusion: bounded T4 operator experiment

Status: corrected Kaggle CPU version 2 passed **179 regression + 5 experiment
tests**. Kaggle acknowledged **T4 operator version 1**; GPU execution outcome and
final measurements remain pending. No default promotion is claimed. This experiment implements the first-stage
candidates from [the ROI analysis](H3_DIT_VRAM_FUSION_ROI_20261029.md).

The ongoing PTD E2E job is a separate immutable submission and is unchanged.
This new experiment runs no sampler, TE, VAE or complete H3 pipeline. It
downloads only two public trained block-0 rank shards, **392,696,256 bytes**
total, from the already validated TP2 bundle revision
`d3e6a4b131056d2c626a01acf6ba48123a885783`. File and Native tensor hashes are
checked before use. No full model, real LoRA or private activation download is
needed. The generated residual fixture is not a real H3 distribution capture.

## Geometry and controls

All large tests use **59,134 tokens**, hidden 5,376, 28 local heads × 128,
local FFN 7,168, current FP16-fix storage and FP32 stability islands. Modulation
spans are text 0–2,392, reference images 2,392–3,892, audio 3,892–4,702, and
video 4,702–59,134. Native modality tags are video=0, text=1, audio=2. Three
conditioning clocks use a nine-row modulation table: pinned text/reference
rows 7/6, audio row 5 and video row 0. Both ranks share the seeded input and
generated eight-dimensional FP32 conditioning coordinates; these do not replay
the checkpoint's interpolated AdaLN curve or a captured denoiser input. Block weights are
real trained base weights, not PTD-baked weights; shapes/operator contracts
match the PTD path, but generation-quality conclusions are out of scope.

Baseline is the unmodified frozen `TPDiTBlockINT8.forward`, current policy and
scaled FP16 transport. Each rank holds one actual trained block throughout the
comparison. Saved activation space cannot silently turn into extra prefetched
weights. This first experiment does not establish a full-model residency-window
peak or implement a new prefetch policy.

| Block variant | Change |
|---|---|
| `baseline` | Current production block |
| `ownership` | Prepare/consume column and row operands; release floating owners at their last consumer; split Kitchen prequantization if supported |
| `norm_gate` | Fused FP32 RMS/AdaLN with FP16 destination, and in-place FP32 gated residual with FP16 storage |
| `fc1_4k` | Full existing activation quantization, then 4,096-row INT8 GEMM/epilogue chunks |
| `fc1_8k` | Same, 8,192 rows |
| `fc1_16k` | Same, 16,384 rows |
| `combined_8k` | Ownership, small fusions, 8K FC1 and scaled-FP16 decode/dequant/gated-residual epilogue |

There are also isolated norm/gate, FC1, row-epilogue and attention sections.
Each section launches a fresh pair of worker processes; one failing CUDA
context cannot contaminate the next section. Failed byte qualification excludes
the corresponding combined block variants instead of silently accepting them.

## Numerical contracts

- Column preparation uses the full original ConvRot/quantization geometry.
  Only already-quantized INT8 rows are chunked. INT32 GEMM stays the existing
  Kitchen cuBLAS primitive. The epilogue first multiplies FP32 row and weight
  scales, then the FP32 accumulator conversion, then stores FP16 directly into
  the final output slice. Scalar and vector scale metadata are explicit.
- Prepared objects own codes, scales and shape metadata, and are consumed once.
  They retain no floating operand. A worker-side copy-stream handoff test uses
  a ready event and `record_stream` before deleting the transferred owner.
- Row fusion retains the existing power-of-two scaled FP16 SUM, INT32
  reconstruction and two FP32 dequantization products. It explicitly rounds
  the branch through FP16 **before** FP32 gated residual computation.
- The gated residual uses FP32 FMA and final FP16 storage. Byte comparison to
  current PyTorch `addcmul` is required, rather than assuming matching algebra
  implies matching CUDA instructions.
- Fused RMS keeps FP32 statistics, gains, folded epsilon, modulation and the
  final FP16 boundary. Reduction order can differ; the isolated RMS tests use
  a declared relative-RMSE budget of 1e-3 and report maximum error/mismatch
  fraction. This is an operator budget, not a generation-quality threshold.
- Kitchen split attention is checked against the pinned installed API and byte
  compared to the normal call. If the wheel lacks this API, that case is marked
  blocked; Kitchen is not upgraded. Its isolated timing includes identical
  template H2D staging on both sides; full-block timing uses resident inputs.

Zero/tiny/max-FP16 norm rows, signed INT8 extremes, non-divisible row tails,
global MAX/SUM call counts and finite outputs are included. Actual per-variant
errors remain in the results even if a candidate fails qualification.

## Measurement and evidence

Compile requests/cache loads are recorded separately. The first call after
validation is labeled accordingly; it is not presented as a cold compilation.
Any compilation observed inside a steady-state timing sample rejects that timing.
Isolated sections use two warmups and five balanced/reversed repeats; attention
and full block use one warmup and three repeats to bound GPU cost. Error
reductions, model loading and profiler capture are outside timed measurements.

Memory runs are separate and record initial, peak and incremental PyTorch
allocation, peak reservation and sampled full-device usage. Full-block passes
also record phase allocation boundaries and CUDA allocator histories, retaining
Python allocation/free stacks without holding extra tensor owners. A sampled
device-use value is not reported as the full-device maximum.

Worker CUDA profiling must first capture and export a minimal GPU kernel. Full
traces are exported, checked for CUDA events and compressed before process exit.
No attribution is claimed for an empty trace. Trace summaries include launch,
kernel and memcpy counts; overlapping kernel durations are not called critical
path time. This batch uses PyTorch worker traces; Nsight availability is recorded
and no Nsight or protected NCU-counter result is implied.

Section limits are 240/300/240/300/480 seconds, under one 1,800-second execution
watchdog. Core setup has its separate 600-second bound; the notebook bound is
2,500 seconds. There are no status/list/live-log polls. Final-output transfer
retries follow AGENTS.md and never trigger duplicate jobs.

## Preservation and local checks

Only five new experiment modules are overlaid on the frozen source capsule.
Every pre-existing production file, Native ComfyUI archive, legacy attention API
and core dependency lock remains byte-exact. Local execution uses conda
`comfyui`; Kaggle retains its native Torch family and exact Kitchen 0.2.37 /
TileLang 0.1.13 pins.

The extracted initial capsule passed **179 existing CPU tests and four new
ownership/partition/chunk-destination tests** locally and on Kaggle CPU version 1.
A pre-GPU Native-source audit corrected the fixture's modality tags and added
three-clock/nine-row coverage, with a fifth CPU contract. A new immutable CPU
qualification is required for that corrected capsule before GPU submission.
SM75 factory compilation
was checked locally without claiming T4 execution. An attempted SM75 launch on
the local SM120 device correctly had no compatible binary; a separate SM120
preview encountered the existing Windows compiler-diagnostic issue. A separate
small SM120 scalar-store check then passed: gate, column store and fused row gate
were byte-exact, and RMS relative RMSE was 3.59e-6. Vectorized stores were disabled
only in that local compiler process, without package/source changes. Neither
preview constitutes T4 numerical or performance evidence.

In-place Q/K RMS-RoPE, the later combined FC1-dequant/SwiGLU producer, and
phase-aware weight-window scheduling remain follow-ups. They are not silently
included in this first batch.

## Immutable submission

- [CPU qualification, version 2](https://www.kaggle.com/code/pamcks/h3-dit-vram-fusion-cpu-20261029): corrected Native modulation fixture; exact frozen core/API gate passed. The installed Linux Kitchen exposes both split-attention functions.
- [T4 operator batch, version 1](https://www.kaggle.com/code/pamcks/h3-dit-vram-fusion-t4-20261029): one accepted GPU submission; no status/list/live-log query.
- Source commit `089a371`, archive `3b24746a0a09a9f87fbf8c39ae0f587fbd2bbabb8a8169be499feac814c21557`, capsule `7286ec2f971f1ec7b096db161eb534a7e2d26ec1a20cbdb517f87bc2fcbb6fc3`.
- Evidence: `docs/data/h3_dit_vram_fusion_20261029/`; local raw working data: `D:\attention_eval\h3_dit_vram_fusion_20261029`.

The initial four-contract CPU version and the corrected five-contract version
each had one successful logical final-artifact collection. The GPU notebook
uses the corrected capsule and both matching CPU qualification receipts. Neither
CPU version fetched model weights or requested GPU acceleration. The initial
capsule was never submitted as a GPU experiment.

The artifact-only analyzer checks incomplete sections, keeps numerical/trace
blockers, and computes the median after taking the slower rank in each aligned
repeat. Its three synthetic CPU tests passed; those fixtures are not GPU
performance measurements. An additional nine-row SM120 scalar-store arithmetic
check passed (RMS relative RMSE 5.47e-6, exact gate/row-gate outputs), again without
claiming T4 qualification or substituting its performance.

## T4 version 1 returned: four sections passed, block validation incomplete

One final collection downloaded all 74 server artifacts and verified capsule
`7286ec2f…` against the submitted `089a371` source. The device receipts identify
two distinct physical Tesla T4s, Torch 2.11.0+cu128 / CUDA 12.8, Kitchen 0.2.37
and TileLang 0.1.13. Four isolated sections exited successfully. The block
section exited with code 1; the batch itself reports **`partial_or_failed`**.
The notebook's green provider status therefore does not establish success of
every experiment. No full-DiT or end-to-end improvement has been measured.

These are 59,134-token generated residual fixtures through the trained base's
block-0 shards, with one resident block. They are not captured PTD activations,
LoRA-baked weights, or a full-model residency-window experiment. The two shards
total 392,696,256 downloaded bytes. Compilation, input loading and numerical
checks were outside warmed timing; the timing samples record zero compilation
requests. Memory was measured in separate passes. Times below take the slower
rank for each aligned trial, then its median.

| Operator / candidate | Baseline → candidate CUDA ms | Speedup | Allocated peak GiB, baseline → candidate | Peak reduction GiB | Qualification |
|---|---:|---:|---:|---:|---|
| RMS + AdaLN | 53.603 → 5.635 | 9.51× | 4.358 → 2.582 | 1.777 | Finite; relative RMSE 4.06e-6; not byte-exact |
| Gated residual | 51.748 → 13.566 | 3.81× | 5.852 → 2.582 | 3.270 | Byte-exact on both ranks |
| Scaled row epilogue + gate | 68.045 → 13.451 | 5.06× | 6.444 → 2.582 | 3.862 | Byte-exact on both ranks |
| FC1 prepared, unchunked | 264.634 → 358.442 | 0.738× | 6.739 → 7.430 | **−0.691** | Slower, more memory; fails byte qualification |
| FC1 4K chunks | 264.634 → 294.614 | 0.898× | 6.739 → 4.786 | 1.952 | Memory tradeoff; fails byte qualification |
| FC1 8K chunks | 264.634 → 290.572 | 0.911× | 6.739 → 5.005 | 1.734 | Memory tradeoff; fails byte qualification |
| FC1 16K chunks | 264.634 → 308.449 | 0.858× | 6.739 → 5.443 | 1.296 | Worse tradeoff than 4K/8K here |
| Kitchen prepare/consume attention | 1613.754 → 1614.074 | 1.000× | 3.775 → 3.381 | 0.395 | Byte-exact; latency essentially unchanged |

Allocation peaks include the fixture's live inputs, weights, reference tensors
and outputs; they are not pure temporary-buffer sizes. The gate timings also
include the harness's input cloning. Attention includes identical host-template
H2D staging on each path, as declared before submission. None of the component
peak reductions can be added together or applied to the full DiT peak.

### What the worker traces establish

Every completed isolated operator has a successful minimal-CUDA capture gate
and an exported, compressed worker trace. In rank 0's captured operator calls:

| Comparison | Baseline kernels → candidate kernels | Evidence / implication |
|---|---:|---|
| RMS/AdaLN | 15 → 1 | Eliminates separate reductions, conversions and segment elementwise passes |
| Gated residual | 16 → 1 | Removes full FP32 residual/branch intermediates and segment passes |
| Row epilogue + gate | 18 → 1 | Fuses transport reconstruction, dequantization, branch FP16 rounding and residual store |
| FC1 production → 8K chunks | 4 → 31 | Replaces an already fused production GEMM epilogue with repeated GEMMs and stores |
| Attention normal → prepared | 5 → 5 | Changes ownership lifetime, without removing attention kernel work |

The FC1 baseline **already selects Kitchen's CUDA INT8 linear**, with a CUTLASS
SM75 GEMM containing a fused FP16 output epilogue. Its trace also contains two
FP16 rotation GEMMs and CUDA row quantization. The candidate uses eager
preparation, an explicit INT32 GEMM result, and a separate TileLang store. In
the 8K trace, eight INT8 GEMMs and eight dequant stores are visible. The
unchunked candidate therefore adds work and materialization to a baseline
that already has fusion; earlier reasoning based solely on the eager fallback
does not describe this actual run.

Column candidates differ from production in about 0.0044% of stored values,
with maximum absolute difference 0.00048828125 and relative RMSE about 4.6e-6.
Their signed-extreme INT32 GEMM check and copy-stream lifetime check pass.
However, this is not byte qualification: preparation/backend and epilogue
arithmetic both changed. The candidate explicitly forms `row_scale*weight_scale`
before multiplying the accumulator; production's CUDA visitor has its own
ordered epilogue. A codes/scales/intermediate ablation is still needed to
isolate the differences. It would be unsupported to call them INT8 overflow,
or to infer generation quality from these operator fixtures.

Fused RMS stress rows remain finite (relative RMSE 5.33e-6), within the declared
1e-3 operator budget. That budget is not a full-generation quality guarantee.
The gate and row epilogue are byte-exact only against their declared isolated
operator references, not a new FP32 model baseline.

Nsight Systems was unavailable. Nsight Compute's binary was present but was
not used for this batch. These are PyTorch/CUPTI worker-trace observations;
there is no measured NCU occupancy/stall attribution or Nsight critical path.

### Block failure and next decision

The candidate's non-prepared attention branch retained the policy result's
`[1,S,D]` batch axis. Production flattens it to `[S,D]` before row projection.
The candidate consequently passed an incompatible branch shape into the
two-dimensional fused gate and raised
`ValueError('Matching contiguous FP16 branch required')` on both ranks. This
occurred during numerical validation, before block timing, memory histories
or block traces. It is a benchmark wiring failure, not an observed CUDA OOM.

The experiment-only forward now restores production's reshape. A CPU
regression executes the mocked block and checks projection/gate boundaries;
all six candidate contracts pass. The job builder now requires the final
batch status to be `complete`, preventing a future green notebook from
concealing failed sections. Neither change modifies production runtime files
or retroactively qualifies the failed block run. **No GPU retry was submitted.**

The highest-return follow-up is to qualify norm/gate and row-gate fusion inside
the corrected block, preserving the actual production CUDA column path.
Prepared attention is a modest memory optimization, with no demonstrated
latency gain. Keep unchunked prepared FC1 out of promotion; consider 4K/8K only
as measured memory-pressure alternatives after resolving byte qualification.
Production defaults remain unchanged.

All original artifacts, including failed block receipts, worker logs and
compressed CUDA traces, are under
`docs/data/h3_dit_vram_fusion_20261029/gpu_v1_final/` (plain logs archived as gzip).
The aligned summary is under `gpu_v1_analysis/`. The original download manifest
and an additional original-file hash manifest are preserved. Collection used
one logical final-artifact request and no status/list/live-log queries.

## User-requested production integration

Following the explicit promotion request, RMS/AdaLN, gated residual, scaled
row epilogue/gate and prepared Kitchen attention are now default-on production
choices, with independent off switches. The production factories match the
measured factories structurally. Online LoRA, Native INT32 transport,
unsupported devices/dtypes/layouts and other attention backends keep their
appropriate existing paths. FC1 chunking/prepared column changes were not
promoted. Runtime code imports no experiment modules.

[Production switches and contracts](../guides/H3_DIT_FUSION.md) describe
rounding, eligibility, label lifetime protection, AOT and fallback behavior.
This user-requested integration supersedes the earlier no-promotion decision;
it does not change the historical T4 evidence or qualify combined T4/full-model
latency, memory or generation quality. CPU validation records are kept in
`docs/data/h3_dit_fusion_production_20261029/`.
