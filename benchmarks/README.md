# Benchmark and diagnostic entry points

These modules are explicit developer tools, not runtime imports. The ComfyUI
node imports `patches`; it never imports this package. Numerical reference
loaders are isolated from the normal worker execution path.

The former root-level scripts now live in [`tools/`](../tools/README.md).
Runtime commands use `python -m tools.runtime.NAME`; historical builders and
analysis helpers use `python -m tools.experiments.NAME`. Historical submissions
remain unchanged and are not current setup templates.

`build_te_parallel_job.py`, `run_te_parallel_batch.py` and `te_parallel_matrix.py`
implement the [TE attention/communication comparison](../docs/reports/TE_ATTENTION_TP2_MATRIX_20261027.md).
The resident trained-block matrix measures single-GPU and actual NCCL TP2,
with exact fixed-input collective-result replay as a separate causal ablation.
It uses two public TE shards and the existing CPU-qualified frozen core setup.
`analyze_te_parallel_results.py` merges retained points by allocation and verifies
finite outputs, exact replay, collective inventories and CUDA trace coverage.
It reproduces the archived tables and figure offline, without Kaggle calls.

`te_column_layout.py` and `run_te_column_layout.py` isolate the repeated INT8
weight transpose from activation quantization, scaling, attention and transport.
The [layout ablation](../docs/reports/TE_COLUMN_LAYOUT_20261027.md) compares the
pinned copy, a Torch transpose view and the existing Kitchen direct GEMM.
`build_te_column_layout_job.py` binds the explicit experiment to the frozen
setup; `analyze_te_column_layout.py` reads final evidence offline. Runtime
defaults stay unchanged until the physical T4 results qualify a replacement.

`run_vae_temporal_batch.py` and `run_vae_temporal_compare.py` implement the
[decoder-only Temporal/Query comparison](../docs/reports/VAE_TEMPORAL_QUERY_20261027.md).
They replay the private saved H3 video latent and load only the trained Video
decoder. CPU source/input qualification precedes a bounded physical two-T4 run;
local logical-rank correctness is checked by `qualify_temporal_vae.py`.
`analyze_vae_temporal_results.py` reproduces timing, memory, communication and
per-frame error summaries from the retained numeric traces without Kaggle calls.

`build_review_t4_delivery.py`, `run_review_cpu.py` and `run_review_t4.py`
implement the [bounded T4 review experiment](../docs/reports/H3_REVIEW_T4_EXPERIMENT_20261024.md).
Only four hash-verified public block shards are used, with seeded shape
fixtures. CPU qualification is required before a two-T4 GPU submission.
Unprofiled repeated measurements and a separately coverage-gated Nsight replay
test columns, shared TE preparation and optional online QKV B slicing.

`test_review_optimizations.py` covers shared eager preparation, column rounding,
bounded I/O, reply ownership, staged remote Video planning and loader controls.
The three `preview_review_*.py` tools replay only local operators using actual
TP2 shapes; column/TE previews read trained base weight slices, while the QKV
LoRA preview uses seeded factors. They perform no full H3 generation or Kaggle
request. Their SM120 results do not qualify T4, and numerical candidates remain
opt-in; see the [review follow-up](../docs/reports/H3_REVIEW_OPTIMIZATIONS_20261024.md).

Kaggle interaction policy: do not poll status/list endpoints or live logs.
Batch submissions, allow bounded jobs to finish, and collect final outputs once
per batch. API/browser polling is not a workaround. The latest CPU preparation
attempt and its dependency blocker are recorded in the
[T4 follow-up report](../docs/reports/T4_FOLLOWUP_PREFLIGHT_20261021.md).

Audio VAE parallelism was removed from the runtime on 2026-10-21.
`test_native_audio_decode.py` covers Native-only dispatch, full stereo/tiling
delegation, old-workflow compatibility and rejection of obsolete worker requests.
The old `audio_channel_decode` / `multi_gpu_audio_vae` helper modules and their
standalone experiments are historical provenance, not current inference routes.

`test_video_decode_memory.py` checks separate Video/DiT admission budgets,
coordinated refusal/cleanup, parent planner ownership and Native planning.
`qualify_video_decode_memory.py --output D:/attention_eval/video_decode_memory/local`
uses the trained Video VAE on two logical ranks of the local GPU. It checks
public tiling, Native equality, admission fallback and error recovery. Add
`--decode-inputs <decode_inputs.pt> --long-only` to replay saved sampler Video
latents without running TE, DiT or Audio VAE. This is a local correctness check,
not a physical dual-T4 performance measurement.

`run_h3_tensor_flow.py` runs bounded trained TE/DiT, vision, refiner/packing,
preprocessing-ownership and Video VAE comparisons against actual BF16
execution. It does not run the H3 sampler. `test_h3_tensor_flow.py` checks
gauge/weight-fold/copy/public-conditioning contracts. The source-validated
aggregate is produced by `summarize_h3_tensor_flow.py`; see the
[FP16/BF16 integration report](../docs/reports/H3_FP16_TENSOR_FLOW_BF16_20261021.md).

Use the existing conda environment and pinned local ComfyUI. From the repository
root, the focused regression command is:

```powershell
$env:PYTHONPATH = "$PWD\benchmarks"
& C:\ProgramData\miniconda3\envs\comfyui\python.exe -m unittest `
  benchmarks.test_pipeline_review benchmarks.test_legacy_runtime `
  benchmarks.test_frozen_kaggle_setup benchmarks.test_video_query_fallback `
  benchmarks.test_h3_operator_enhancements benchmarks.test_h3_loading_stages `
  benchmarks.test_idle_window benchmarks.test_idle_lifecycle benchmarks.test_idle_identity `
  benchmarks.test_residency_policy -v
```

This suite checks ownership, cancellation, offline bake publication, dispatch,
frozen setup, diagnostic boundaries and launcher defaults. It does not run the
full H3 model. Avoid blanket test discovery: historical files named `test_*`
include top-level GPU experiments and require separate review.

The component scripts `test_h3_pipeline_wiring.py`,
`test_h3_operator_integration.py --cuda-preview`, `check_legacy_video_query.py`
and `qualify_h3_staged_vae.py` take `--output` paths. They are local component
checks, not SM75/NCCL timing or generation-quality evidence.

Profiling/capture tools such as `causal_trace.py`, `cpu_causal_trace.py`,
`capture_real_attention.py` and `profile_real_attention.py` must be invoked
explicitly. `MINIMAX_H3_PROFILE_DETAILS=1` enables bounded detail metrics only;
it does not start a profiler or dump tensors.

`te_compression_local.py` is a RAM-only lossless codec benchmark on real
production-loaded TP2 TE tensors. It uses a separate `--deps` directory with
`requirements-te-compression.txt`; it never installs packages into the active
runtime. `build_te_compression_codecs.ps1` compiles unchanged upstream ZigZag
and FSE library files for its C ABI. See the
[compression report](../docs/reports/TE_COMPRESSION_LOCAL_20261020.md) for
the results and the ZipNN INT8 blocker. No SSD or full-pipeline speedup is
implied by its decode throughput.

Historical builders/analysis scripts preserve earlier experiments. Their input
artifacts may be local-only. Do not submit an old notebook or run its setup as a
current deployment recipe. Historical wheelhouse setup remains archived; the
current authorized delivery uses immutable GitHub source, private HF inputs and
direct pip with exact approved hashes, retaining Kaggle's native platform family.
Both paths reject unqualified core version substitutions.

`run_fp16_residual_fix.py` is an isolated FP16 residual-stream and weight-folding
experiment. It reconstructs the first DiT block input from retained conditioning
and noise, stops before the main block stack, then tests trained block 0 with
logical TP2 transport. It never installs an inference backend or runs a sampler.
See the [local study and reproduction commands](../docs/reports/FP16_RESIDUAL_FIX_LOCAL_20261020.md).
`test_fp16_residual_fix.py` checks epsilon scaling, opposing overflow/underflow
bounds, lost residual updates, and reversible folding with AdaLN compensation.

`run_h3_native_fp32_flow.py` compares BF16/FP16 activation flows against the
existing legacy FP32 TP2 tensor flow, retaining current INT8 weights/GEMMs,
attention dispatch and native precision islands. Local SM75-only fallbacks are
reported explicitly. `summarize_h3_native_fp32_flow.py` validates the evidence.
The historical `run_h3_tensor_flow.py` reports pairwise BF16 distances only.
`run_h3_fp32_ground_truth.py` and `fp32_ground_truth.py` preserve an out-of-scope
forced-dense-FP32 study; they do not implement the user's Native FP32 reference.

The user-authorized Kaggle native-Torch profile uses
`prepare_kaggle_native_torch.py` and `build_kaggle_native_torch_jobs.py` for
CPU-only core/API preparation. `build_kaggle_component_assets.py` prepares
selected trained components on CPU. `build_t4_native_validation_batch.py`
freezes a bounded physical two-T4 batch only after both receipts pass. Kitchen
0.2.37, TileLang 0.1.13 and frozen ComfyUI/legacy attention are preserved; the
native platform stack is recorded separately. The local conda lock is unchanged.
`summarize_t4_native_validation.py` analyzes downloaded artifacts offline. See
[the profile and evidence report](../docs/reports/T4_NATIVE_TORCH_VALIDATION_20261021.md).
Do not poll Kaggle CLI, APIs or browser status; use a single version-bound final
artifact collection after the bounded run window.

`h3_fp16_fc2_experiment.py` implements benchmark-only FP16 gate/up storage with
FP32 SwiGLU / regular ConvRot / shared row quantization. `run_t4_fc2_fusion.py`
tests control, stored fusion and recompute fusion with original INT8 GEMM and
INT32 or scaled FP16 SUM. `run_t4_video_split_localize.py` locates the 261-token
Video split's first differing stage. `build_t4_fc2_video_experiments.py` freezes
the bounded batch from the qualified CPU/assets/prior receipts; it does not
update dependencies or production defaults. Analyze final artifacts offline
with `summarize_fc2_video_experiments.py`. See the
[completed dual-T4 results](../docs/reports/FP16_FC2_FUSION_VIDEO_SPLIT_20261022.md).

On 2026-10-22, the measured fused kernels were promoted unchanged into
`patches/tilelang_swiglu_quant.py`. The default policy is FP16-fix +
`fused_stored` + `tilelang_scaled_fp16`. `test_h3_default_fused_row.py` checks
measured-kernel identity, LoRA-before-store order, fallback preparation and
preserved policy choices. `preview_h3_default_row.py` executes the production
row dispatch on local SM120 logical owners; it is not a physical T4 benchmark.
Historical Native-FP32 comparison runners explicitly retain packed/owned-FP32
controls so the promoted default cannot silently change their reference.
See [the production follow-up](../docs/reports/H3_FP16_FUSED_DEFAULT_20261022.md).

The subsequent RowPartial/reference-loading integration is covered by
`test_h3_row_partial_combined.py`, `test_h3_reference_loading.py` and
`preview_h3_row_partial_combined.py`. The latter runs the production DiT row
with two rank-128 adapters, fused preparation and scaled base transport on
local logical owners; LoRA hidden SUM stays FP32. `qualify_h3_staged_vae.py`
now checks metadata-only construction before explicit encoder preload and
trained Native encode/decode comparisons. These do not establish T4 latency
or full-pipeline generation quality; see
[the integration record](../docs/reports/H3_ROW_PARTIAL_REFERENCE_LOADING_20261022.md).

`build_h3_row_partial_t4.py` builds a fresh CPU revalidation using archived
hash-bound wheels, then a source-matched bounded real-T4 batch.
`run_t4_row_partial_combined.py` measures production gather/partial rows and
trained blocks in paired reversed rounds; `run_t4_reference_demand_vae.py`
checks actual component reads and Native output on T4 GPU0. Analyze downloaded
artifacts offline with `summarize_h3_row_partial_t4.py`; an empty provider output
is explicitly unavailable, never a hardware pass. The user stopped version 1
after a silent run; initial and post-stop artifact collections were both empty.
`t4_notebook_watchdog.py` now wraps future builds with a setup-inclusive
600-second supervisor and immediate stage-start output. Its two Windows CPU
tests pass; Linux process-group cleanup remains unvalidated. The revised GPU
notebook was accepted as version 2 after the user's explicit retry request;
execution is unverified. Include `.jsonl` progress files
when collecting a future revision's artifacts. See the
[submission status and exact experiment scope](../docs/reports/H3_ROWPARTIAL_REFERENCE_T4_20261022.md).
Do not poll or resubmit the job merely because output acquisition is empty.

`build_h3_l1_current_e2e.py` builds the separately authorized long-generation
retry. `run_h3_l1_current_batch.py` verifies pinned downloads, bakes all LoRA
targets across two GPUs and releases its private original-download cache before
TE/VAE downloads. `run_h3_l1_current_e2e.py` runs one 896×512/362-frame request
with current FP16/fused/scaled defaults, the previous scalar sparse policy,
lazy VAE components and Native single-GPU audio. It saves sampler latents before
decoding. This does not change the existing operator-only batches. See
[the source-bound plan and submission status](../docs/reports/H3_L1_CURRENT_OFFLINE_E2E_20261022.md).

The current zero-mount path uses `build_kaggle_light_delivery.py`,
`build_kaggle_light_job.py`, `prepare_kaggle_direct_pip.py` and
`launch_kaggle_light.py`. CPU v1 passed 23 tests and preserved the native base
inventory; source/API qualification took 120.02 seconds inside the notebook.
The private retained input is now committed on HF. Unchanged operator v3 is
accepted, pending final evidence. See
[the delivery record](../docs/reports/KAGGLE_LIGHT_DELIVERY_20261022.md).

Long E2E v2 ran on dual T4: offline baking and TE completed, then FC1's eager
GPU epilogue ran out of memory at 51,220 tokens. `bounded_column_int8.py`
contains two benchmark-only remedies: a preallocated eager epilogue and bounded
row calls to the pinned Kitchen operator. `test_bounded_column_int8.py` has
three passing local CPU tests. `preview_bounded_column.py` records selected
backend, shapes, library versions, full-output equality, finite values, warmed
samples and allocator increments. Its repeated RMS-normalized input is a shape
proxy, not the failed E2E FC1 input. The local SM120 screen is not T4 evidence.
Production defaults and operator v3 remain unchanged. A source-preserving
archive extension and CPU-only v2 qualification prepare the next T4 gate.

The 2026-10-23 request pauses candidate work for a memory profile first.
`run_dit_memory_51220.py` drives one TP2 DiT evaluation, including refiner and
residency but excluding TE/VAE/sampling. `dit_memory_lines.py` records source
boundaries without retaining tensor locals and saves continuous-history prefix
snapshots for OOM recovery. `analyze_dit_memory_lifetimes.py` exports aliases,
birth/release/reuse times, live sets at peaks/OOM and lifetime plots. Five CPU
analyzer tests and small local observer tests qualify the tooling; only returned
physical T4 evidence qualifies the target workload. See
[the completed dual-T4 profile and FC1 plan](../docs/reports/DIT_MEMORY_51220_20261023.md).
Both ranks completed all 50 blocks. FC1's concatenation overlaps 6.712 GiB of
internal storage; the measured allocator peak is 12.427 GiB. The shape-matched
synthetic/base-weight probe does not reproduce the previous offline-LoRA E2E
OOM. `summarize_dit_memory_profile.py` builds the evidence-bound summary and
allocation/release diagrams offline; no optimization was enabled.

The subsequent lifetime-only change is qualified by `test_dit_temporaries.py`.
`preview_dit_lifetimes.py` compares the pinned eager implementation with the
same full-size column GEMM and scale loop after earlier releases; it records
byte equality and warmed incremental peaks. `check_dit_lifetime_block.py`
checks preserved-source small blocks and a nondefault CUDA stream.
`compare_dit_lifetimes.py` separates actual T4 workspace changes from weight
cache growth. See [the follow-up report](../docs/reports/DIT_TEMPORARY_LIFETIMES_20261023.md).
