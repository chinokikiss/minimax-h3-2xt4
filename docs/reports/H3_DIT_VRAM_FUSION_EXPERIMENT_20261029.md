# DiT VRAM fusion: bounded T4 operator experiment

Status: implementation and local CPU validation complete; fresh Kaggle CPU
qualification precedes a separate physical two-T4 operator job. No results or
default promotion are claimed yet. This experiment implements the first-stage
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
spans are text/references 0–3,892, audio 3,892–4,702, and video 4,702–59,134.
Both ranks share the seeded input and time-basis fixture. Block weights are
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

First-call time, including any compilation, is separate from steady state.
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

The extracted candidate capsule passed **179 existing CPU tests and four new
ownership/partition/chunk-destination tests** locally. SM75 factory compilation
was checked locally without claiming T4 execution. An attempted SM75 launch on
the local SM120 device correctly had no compatible binary; a separate SM120
preview encountered the existing Windows compiler-diagnostic issue. Neither
constitutes T4 numerical or performance evidence, and no dependency was changed.

In-place Q/K RMS-RoPE, the later combined FC1-dequant/SwiGLU producer, and
phase-aware weight-window scheduling remain follow-ups. They are not silently
included in this first batch.
