# Experiment reports and reviews

[Documentation index](../README.md)

[TE causal prefix invariance and FP32 precision](TE_CAUSAL_PREFIX_PRECISION_20261026.md)
records the current Kitchen default's numerical contract on both physical T4s.

Latest attention results: [Kitchen dense SM75](KITCHEN_DENSE_SM75_20261025.md)
and the [Kitchen / TileLang short and long T4 matrix](KITCHEN_ATTENTION_T4_MATRIX_20261025.md).

Measurements are version-bound. Read each report's hardware, software, shapes,
numerical reference, and timing boundary before comparing results.

- [Video VAE Temporal versus Query Parallel](VAE_TEMPORAL_QUERY_20261027.md) — optional Native clip pairing, real saved H3 latents, 178 CPU checks and local trained equality; private CPU preflight submitted, physical T4 comparison pending

The [zero-mount GitHub/HF/direct-pip migration](KAGGLE_LIGHT_DELIVERY_20261022.md)
records the passed CPU gate, private input publication and replacement jobs.
Current feature selection is described in the [technical guide](../guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf).
The [documentation coverage audit](H3_DOCUMENTATION_COVERAGE_AUDIT.md) reconciles
the live source with historical experiment statuses and records the latest additions.
The [T4 validation backlog — 2026-10-21](T4_VALIDATION_BACKLOG_20261021.md)
separates local-only work, current-stack requalification, completed historical
T4 experiments, and deferred/blocked tests.
The [follow-up CPU preflight](T4_FOLLOWUP_PREFLIGHT_20261021.md) records the
new account's actual resolver blocker and the unapplied Linux CUDA proposal.
The [native-Torch follow-up](T4_NATIVE_TORCH_VALIDATION_20261021.md) records the
subsequent user-authorized Kaggle profile, preserved core APIs and bounded T4
component qualification.

## Attention and numerical methods

- [Attention bottleneck audit](ATTENTION_BOTTLENECK_REPORT.md)
- [Attention compatibility on Turing / Tesla T4](ATTENTION_T4_COMPATIBILITY.md)
- [Quality-constrained attention decisions: interim](FINAL_PARETO_REPORT.md)
- [Group32 audit and optimization](GROUP32_OPTIMIZATION_REPORT.md)
- [Local Kitchen-inspired Group32 preview — 2026-10-17](LOCAL_KITCHEN_GROUP32_PREVIEW.md)
- [MiniMax-H3 Text Encoder (Qwen3-VL-32B): Multi-GPU Strategy Benchmark Report (2× Tesla T4)](TEXT_ENCODER_SP_INT8_REPORT.md)
- [TileLang INT8 sparse attention: physical T4 experiments](TILELANG_T4_ATTENTION_RESULTS.md)
- [TP2 attention and transport candidate experiments](TP2_ATTENTION_TRANSPORT_TRADEOFFS.md)
- [Kitchen-inspired Group32 optimization on T4](TP2_KITCHEN_GROUP32_RESULTS.md)
- [Kitchen-inspired Group32 second pass](TP2_KITCHEN_GROUP32_V2_RESULTS.md)

## Linear operators, LoRA and memory

- [Main synchronization and script layout](REPOSITORY_SYNC_20261025.md) —95 Python scripts moved under tools,169 CPU tests pass; experiment heads retained with archive tags before branch cleanup; frozen dependencies preserved
- [0.5MP Turbo long E2E with persisted TP2 LoRA](H3_L1_TURBO_05MP_E2E_20261024.md) — real2×T4 completes58,102-token/four-step generation,208-target disk bake and parallel Video decode without fallback;25 artifacts verified,19m00s foreground generation and near-full device memory; quality review pending
- [Review T4 partial results and scoped recovery](H3_REVIEW_T4_RECOVERY_20261024.md) — measured byte-exact column gains; isolated recovery timing completes, combined profiler replay fails; all48 recovery artifacts collected and long-sequence online-LoRA retries cancelled
- [Configuration and toggle review](H3_CONFIGURATION_REVIEW_20261024.md) — all55 original controls assessed; supported defaults promoted, redundant/negative runtime controls retired, attention operators and meaningful fallbacks retained; no new GPU speedup claim
- [Bounded review-optimization T4 experiment](H3_REVIEW_T4_EXPERIMENT_20261024.md) — 130-test native-Linux CPU preflight passes; one two-T4 operator batch accepted for columns, TE shared preparation and online QKV B slicing; warmed repeated timings, raw-storage equality and gated Nsight replay, final GPU outputs pending
- [Performance review implementation follow-up](H3_REVIEW_OPTIMIZATIONS_20261024.md) — six remaining prioritized findings addressed; exact allocation/ownership/validation repairs, shared TE preparation and opt-in column/QKV/IO candidates; 125 CPU tests and explicit SM120 operator previews, physical T4 qualification pending
- [Explicit worker trace checkpoint replacement](BUNDLE_SCALAR_EXPLICIT_FLUSH_20261024.md) — real dual-T4 capture/restart passes; one Scalar and one matched Kitchen NFE complete at 51,220 tokens with all 50 blocks traced; 133 artifacts verified; CPU offline-LoRA preparation dominates and Scalar attention GPU time exceeds Kitchen
- [Single replacement bundle/Scalar experiment](BUNDLE_SCALAR_RETRY_20261024.md) — 67 CPU tests pass; real dual-T4 periodic failure-path flush gate fails before weight downloads, explicit/graceful flush passes; zero DiT calls, all 28 returned artifacts recovered and verified
- [TP2 bundle offline LoRA and worker profiling](BUNDLE_SCALAR_WORKER_PROFILE_20261023.md) — 61-test CPU gate and real dual-T4 minimal capture pass; Kitchen NFE hits 600-s deadline after 28 host block scopes, Scalar not started; incomplete worker trace and invalid CPU profiles explicitly rejected
- [Single DiT-call performance profile](DIT_SINGLE_CALL_PROFILE_20261023.md) — real dual-T4, 51,220 tokens, one NFE complete; 161.0-s CUDA command spans, all 36 artifacts downloaded; worker Systems CUDA capture missing and Compute counters permission-blocked
- [Video decode memory policy and weight-free parent](VIDEO_DECODE_MEMORY_POLICY_20261023.md) — separate 2-GiB Video reserve; parent GPU decoder removed; local trained and saved-latent checks, physical T4 follow-up pending
- [DiT temporary lifetime reduction](DIT_TEMPORARY_LIFETIMES_20261023.md) — actual 2×T4, all 50 blocks complete; allocated peak 12.43→10.53 GiB; byte-exact FC1/QKV; attention O quantization peak 11.15→8.93 GiB
- [Long E2E retry with validated lifetimes](H3_L1_LIFETIME_E2E_20261023.md) — actual 2×T4, 51,220 tokens, all four steps complete; 896×512/362-frame video with audio exported; native Video fallback, quality review pending

- [DiT 51,220-token memory lifetime profile](DIT_MEMORY_51220_20261023.md) — actual dual T4, all 50 blocks complete; 12.427 GiB allocator peak, FC1 live-buffer diagnosis and proposed plan; synthetic shape workload/base weights
- [New T4 RowPartial/reference-demand experiment](H3_ROWPARTIAL_REFERENCE_T4_20261022.md) — CPU preflight passed; earlier instances stopped; zero-mount v3 accepted, execution unverified
- [RowParallel LoRA partial SUM and reference-demand VAE loading](H3_ROW_PARTIAL_REFERENCE_LOADING_20261022.md) — current default, FP32 hidden communication and lazy encoder lifecycle
- [FP16-fix + stored fusion + scaled transport default promotion](H3_FP16_FUSED_DEFAULT_20261022.md) — production wiring, preserved controls and validation boundaries
- [FP32 SwiGLU/quantization fusion + scaled transport, and Video 261 localization](FP16_FC2_FUSION_VIDEO_SPLIT_20261022.md) — 18 cases per rank on real dual T4; final Video projection isolated; experimental choices
- [H3 FP16/BF16 against the existing Native FP32 tensor flow](H3_NATIVE_FP32_FLOW_20261021.md) — current INT8 operators retained; local component checks
- [H3 FP16 tensor flow: historical BF16 pairwise checks](H3_FP16_TENSOR_FLOW_BF16_20261021.md) — superseded numerical-reference interpretation; implementation and bounded range evidence retained
- [FP16 residual fix and weight folding: local operator study](FP16_RESIDUAL_FIX_LOCAL_20261020.md) — experimental; current defaults unchanged
- [DiT temporary tensor lifetimes: one unique block sequence](DIT_TEMPORARY_TENSOR_LIFETIMES.md)
- [FC2 Global Quantizer memory optimization](FC2_QUANTIZER_MEMORY.md)
- [Optional SwiGLU reduction and H3 Multi-LoRA](H3_INT8_SWIGLU_LORA.md)
- [Physical dual-T4: optional SwiGLU reduction and DMAD LoRA](H3_PHYSICAL_INT8_LORA_RESULTS.md)
- [Byte-exact online LoRA: accepted storage optimization](ONLINE_LORA_BYTE_EXACT_OPTIMIZATION.md)
- [Online LoRA accuracy repair](ONLINE_LORA_NATIVE_ORDER_REPAIR.md)
- [Online LoRA choices: Native default and optional A schemes](ONLINE_LORA_OPTIONS.md)
- [Row Partial: retained option and memory lifetime audit](ROW_PARTIAL_MEMORY.md)
- [Selective B GEMM feasibility study (benchmark only)](SELECTIVE_B_FEASIBILITY.md)
- [SwiGLU + TP2 padding fusion](SWIGLU_PADDING_FUSION.md)
- [Performance work with sparse behavior frozen](TP2_OFFLINE_OPTIMIZATION.md)
- [Current TP2 operator optimization](TP2_OPERATOR_OPTIMIZATION.md)
- [TP2 operator optimization, second physical-T4 pass](TP2_OPERATOR_PASS2.md)
- [TP2 operator screen: physical T4 results](TP2_OPERATOR_RESULTS.md)
- [TP2 P2 operator investigation](TP2_P2_OPTIMIZATION_REPORT.md)
- [TP2 row-linear fusion on T4](TP2_ROW_FUSION_RESULTS.md)

## Loading, CPU I/O and residency

- [CPU critical-path audit — 2026-10-13](CPU_CRITICAL_PATH_AUDIT.md)
- [Persistent host weight cache](HOST_CACHE_IMPLEMENTATION.md)
- [Rank-local loading and separate preparation/residency](LOADING_OPTIMIZATION.md)
- [H3 full-pipeline GPU memory peak](PIPELINE_GPU_MEMORY_PEAK.md)
- [Full MiniMax-H3 lookahead and cache residency audit](PIPELINE_RESIDENCY_AUDIT.md)
- [Request-wide weight residency: implementation and local evidence](RESIDENCY_IMPLEMENTATION.md)
- [TE blocks 2–49: causal profile](TE_CAUSAL_PROFILE.md)
- [Local TE compression study: ratios and CPU decompression](TE_COMPRESSION_LOCAL_20261020.md)
- [TE critical path — physical evidence and completed ablations](TE_CRITICAL_PATH.md)
- [MiniMax-H3 TE / DiT: actual dual-T4 parallel study](TE_DIT_STUDY_REPORT.md)
- [TE file-read causality experiment](TE_FILE_READ_CAUSALITY.md)
- [TE idle residency: validation status](TE_IDLE_RESIDENCY_RESULTS.md)
- [TE I/O and residency ablation results](TE_IO_ABLATION_RESULTS.md)
- [TE checkpoint I/O causal audit](TE_IO_CAUSAL_AUDIT.md)
- [TE residency optimization: changes and gates](TE_RESIDENCY_OPTIMIZATION.md)
- [Physical dual-T4 residency results — version 1](TE_RESIDENCY_P01_RESULTS.md)

## Video and audio VAEs

- [2-Slab Video VAE](2-Slab%20Video%20VAE.md)
- [Channel Parallel audio VAE: source of the small error](AUDIO_CHANNEL_ERROR_DIAGNOSIS.md)
- [Local Video VAE KV-sharing diagnosis](SLAB_KV_DIAGNOSIS.md)
- [Video VAE: physical dual-T4 KV profiling](VAE_KV_T4_RESULTS.md)
- [MiniMax-H3 Video VAE & Audio VAE Multi-GPU Tile Decode Benchmark Report](VAE_TILE_BENCHMARK.md)
- [Video VAE query-parallel fallback — checked 2026-10-19](VIDEO_VAE_QUERY_FALLBACK_CHECK.md)
- [MiniMax-H3 Video VAE ViT3D INT8 ConvRot Multi-GPU Inference Benchmark](VIDEO_VAE_TP_BENCHMARK.md)

## Pipeline, generation and validation

- [MiniMax-H3 Ref2VA Pruned DiT: Multi-GPU Benchmark Report (2× NVIDIA Tesla T4)](BENCHMARK_REPORT.md)
- [Causal profiling of the actual MiniMax-H3 request path](CAUSAL_PROFILING.md)
- [Dual-T4 causal profiling: completed measurements, partial attribution](CAUSAL_T4_RESULTS.md)
- [Generation quality: existing V4 screen](GENERATION_QUALITY_REPORT.md)
- [H3 pipeline review — 2026-10-20](H3_PIPELINE_REVIEW.md)
- [ComfyUI source provenance correction — 2026-10-20](H3_RUNTIME_PROVENANCE_AUDIT.md)
- [L1 offline LoRA and detailed GPU memory audit](L1_OFFLINE_MEMORY_PROFILE.md)
- [MiniMax H3 P012 Spawn Repaired Audit: completed results](P012_T4_RESULTS.md)
- [Full REF2VA generation benchmark — 2026-10-19](REF2VA_E2E_20261019.md)
- [REF2VA end-to-end results — Kaggle dual T4, version 2](REF2VA_E2E_T4_RESULTS.md)
- [Current-infrastructure long-sequence offline-LoRA retry — 2026-10-22](H3_L1_CURRENT_OFFLINE_E2E_20261022.md) — real dual-T4 v2: bake/TE completed, first FC1 OOM at 51,220 tokens; targeted local memory candidates
- [MiniMax-H3 Ref2VA: physical dual-T4 results, version 4](REF2VA_T4_RESULTS.md)
- [Repaired dual-T4 benchmark findings — 2026-10-08](REPAIRED_T4_REPORT.md)
- [Benchmark findings and actionable repair plan](RETEST_REPORT.md)
- [Actual local dual-rank validation — 2026-10-08](SEMANTIC_VALIDATION.md)
