# Experiment reports and reviews

[Documentation index](../README.md)

Measurements are version-bound. Read each report's hardware, software, shapes,
numerical reference, and timing boundary before comparing results.
Current feature selection is described in the [technical guide](../guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf).
The [documentation coverage audit](H3_DOCUMENTATION_COVERAGE_AUDIT.md) reconciles
the live source with historical experiment statuses and records the latest additions.

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
- [MiniMax-H3 Ref2VA: physical dual-T4 results, version 4](REF2VA_T4_RESULTS.md)
- [Repaired dual-T4 benchmark findings — 2026-10-08](REPAIRED_T4_REPORT.md)
- [Benchmark findings and actionable repair plan](RETEST_REPORT.md)
- [Actual local dual-rank validation — 2026-10-08](SEMANTIC_VALIDATION.md)
