# Technical documentation coverage audit - 2026-10-20

The source audit found supporting optimizations missing or described too briefly in the first technical PDF. The expanded **40-page guide** adds nine explanatory pages and five diagrams. Runtime code, operators, Native ComfyUI, checkpoint bytes and dependency versions are unchanged. No H3 generation, GPU benchmark, environment installation or Kaggle submission was performed.

## Search and decision method

- Searched **3,489** available text/source/compact-record files: 377 benchmark/build/analysis sources, 98 published document/data/authoring files, 56 runtime/node sources, 1,468 retained repository experiment files and 1,490 external experiment files.
- Reconciled **85** non-index guides, designs, reports and archived documents, and **52** implementation items against live loader/runtime selections.
- Search roots include `benchmarks/`, root builders/analysis scripts, `kaggle*`, `semantic_results/`, `slab_results/`, `tilelang_t4_results/`, and the available experiment directories under `D:/attention_eval/`. Local paths, provider capabilities, raw tensors and private traces are not newly published.
- Static text scope is UTF-8 files up to 2 MiB; 82 larger files plus binary captures, raw profiler trees and third-party source/package trees are excluded. Repeated snapshots count as files, not independent experiments. Report references to unavailable older artifacts are not treated as inspected raw evidence.
- Integration requires a current source entry point or guarded runtime branch. A historical result, a function existing in a module, or a notebook winner alone is insufficient. Current source resolves selection; each experiment retains its own correctness, workload, software and timing scope.

## Additions and expanded explanations

1. **Turing row GEMM dispatch:** the installed Kitchen CUDA/cuBLAS path avoids the old per-call transposed-weight and padded-row intermediates. Column projections and Native CUDA dequantization remain separate retained boundaries.
2. **Packed row preparation:** external-global-max quantization, strided owned-slice maxima, positive-zero packing, sentinel/copy removal, exact row-quantizer rounding, and the relationship to later SwiGLU/owned-FC2 work.
3. **Actual attention preparation:** fused stride-aware quantize/cast/pad, per-stream geometry reuse, direct BND stores with retained FP16 rounding, current Q-cache/warp/tile selections, mode-specific pruning and the finite-centroid tau0 precondition.
4. **Native LoRA storage:** activate before gathering, use one rank-major receive buffer, restore Native feature order, scale/add only owned delta views, and release those views before the next adapter. Historical arithmetic root causes and retired modes are explicit.
5. **Startup and auxiliary streaming:** separate vision/embedding CPU groups, on-demand vision GPU lifetime with its synchronization fence, rank-zero text refinement/input projections, canonical CPU backing, window leases and broadcasts.
6. **Cross-phase scheduling:** TE-to-DiT CPU prefetch, DiT reuse between actual evaluations, decode-start startup preparation, lifecycle-controlled early TE prefix, and the difference from queue-idle GPU preload.
7. **Host admission and eviction:** deadline/foreground scheduling, reuse-aware cache policy, optional cgroup/PSI pressure and file-cache safeguards, preparation leases, and missing-metrics fallback.
8. **Evidence bridge:** matched historical row-fusion, NCCL and LoRA results with their original measurement boundaries; the standalone idle pilot is explicitly separated from the integrated ComfyUI prompt-start lifecycle.
9. **Integration exclusions:** INT8 PV, packed-INT8 transport, unselected Group32 variants, retired adapters, slab/temporal paths, experimental I/O readers and CPU codecs are not presented as live defaults.

## Important reconciliations

- **Selective B:** `SELECTIVE_B_FEASIBILITY.md` says benchmark-only because that was its status at measurement time. Current `H3LoRAStore.selective_rows` and the loader enable only the guarded FC1 geometry. The historical report is preserved; pages 16/37 and this matrix describe current integration.
- **Group32 screen winners:** a kernel module still contains many candidate flags. Current TE uses its dedicated masked kernel; optional DiT dense explicitly selects BQ64/128-thread/FullRow/Q-cache. Do not substitute a previous 256-thread or Video cast/pad winner by name.
- **P0/P1/P2:** the loading plan and later residency ablation reuse these labels for different changes. Use `rank_checkpoint_dir`, `prefetch_lifecycle`, `memory_pressure_aware`, `cross_request_te`, `cross_model_cpu` and `idle_residency` to identify behavior.
- **Defaults:** `cross_request_te=True` and `cross_model_cpu=True`; `prefetch_lifecycle=False`, `memory_pressure_aware=False`, `adaptive_lookahead=False`, `idle_residency=False`. The ComfyUI loader sets lifecycle mode from its idle toggle. Idle false therefore does not disable all CPU prefetch.
- **NCCL:** current workers set four channels with P2P disabled and leave protocol automatic. Old reports calling four channels a candidate describe an earlier snapshot. Their one-allocation timing is not a current combined-pipeline measurement.
- **Unavailable profiler counters:** Systems evidence can explain launches and phase intervals. ERR_NVGPUCTRPERM prevents occupancy/stall/issue-rate claims from the retained Compute probes.
- **Scope:** old physical T4 measurements do not qualify current frozen conda Torch 2.12.1 / TileLang 0.1.13 on Linux. Sparse conformance is not a semantic-quality gate. Optional FP16 transport, Row Partial and offline requantization remain separate approximations.

## Integrated feature coverage

12 previously omitted supporting mechanisms now have explicit coverage; 12 existing descriptions are expanded. The remaining 28 mapped items were already covered and were rechecked. Page numbers refer to the 40-page revision.

| Item | Current selection | Implementation | PDF pages | Revision |
|---|---|---|---|---|
| Component-aware row/column ownership | `default` | [`patches/rank_local_int8.py`](../../patches/rank_local_int8.py) (`plan`) | 4 | covered |
| Native Turing cuBLAS INT8 row dispatch | `default` | [`patches/optimized_tp.py`](../../patches/optimized_tp.py) (`native_int_mm`) | 29 | added |
| Kitchen INT8 column projections | `default` | [`patches/tp_dit.py`](../../patches/tp_dit.py) (`run_int8_linear_forward`) | 29 | expanded |
| Fixed-size fallback feature gather | `fallback` | [`patches/optimized_tp.py`](../../patches/optimized_tp.py) (`fixed_gather`) | 11,29 | expanded |
| Native CUDA dequantizer/bias/cast order | `default` | [`patches/parallel_int8.py`](../../patches/parallel_int8.py) (`dequantize`) | 11,29 | expanded |
| Shared full-row MAX before quantization | `default` | [`patches/optimized_tp.py`](../../patches/optimized_tp.py) (`global_row_quantize`) | 11 | covered |
| Quantize directly with external global max | `default` | [`patches/tilelang_row_quant.py`](../../patches/tilelang_row_quant.py) (`quantizer`) | 30 | added |
| Direct owned-slice maximum and reads | `default` | [`patches/tilelang_row_quant.py`](../../patches/tilelang_row_quant.py) (`maximum_kernel`) | 30 | added |
| Pack ownership and positive-zero padding | `default` | [`patches/tilelang_row_quant.py`](../../patches/tilelang_row_quant.py) (`packing_kernel`) | 30 | expanded |
| Separate/packed SwiGLU plus padding fusion | `default` | [`patches/tilelang_swiglu_pad.py`](../../patches/tilelang_swiglu_pad.py) (`swiglu_pad`) | 12 | covered |
| Eligible owned-FP32 FC2 rotation | `default` | [`patches/tilelang_swiglu_pad.py`](../../patches/tilelang_swiglu_pad.py) (`owned_fp32`) | 13-14 | covered |
| INT32 SUM before dequantization | `default` | [`patches/optimized_tp.py`](../../patches/optimized_tp.py) (`projected_reduce`) | 11 | covered |
| Power-of-two scaled FP16 SUM and fused epilogue | `optional_approximation` | [`patches/tilelang_transport_p2.py`](../../patches/tilelang_transport_p2.py) (`row_linear`) | 15 | covered |
| Masked Group32 TE with compact GQA | `default` | [`patches/tilelang_group32_masked.py`](../../patches/tilelang_group32_masked.py) (`attention`) | 6-7 | covered |
| Optional DiT Group32 QK / FP16 PV | `optional_approximation` | [`patches/h3_operator_policy.py`](../../patches/h3_operator_policy.py) (`tilelang_dense`) | 8,31 | covered |
| DiT default Kitchen attention | `default` | [`patches/h3_operator_policy.py`](../../patches/h3_operator_policy.py) (`dit_attention`) | 5,8 | covered |
| Token Refiner Native SDPA and retained norm/RoPE | `default` | [`patches/h3_operator_policy.py`](../../patches/h3_operator_policy.py) (`NativeRefinerAttention`) | 8 | covered |
| Fused stride-aware attention preparation | `selected_paths` | [`patches/tilelang_grouped_tp_kitchen_v2.py`](../../patches/tilelang_grouped_tp_kitchen_v2.py) (`fused_quantizer`) | 31 | expanded |
| Bounded per-stream immutable geometry cache | `selected_paths` | [`patches/tilelang_grouped_tp_optimized.py`](../../patches/tilelang_grouped_tp_optimized.py) (`_geometry_cached`) | 31 | added |
| Direct BND/dtype stores with FP16 rounding boundary | `selected_paths` | [`patches/tilelang_group32_masked.py`](../../patches/tilelang_group32_masked.py) (`execute_prepared`) | 31 | expanded |
| Current BQ64 / Q-only reuse / FullRow dense selection | `optional_dense` | [`patches/h3_operator_policy.py`](../../patches/h3_operator_policy.py) (`cache='q'`) | 8,31 | expanded |
| Mode-specific sparse preparation and tau0 shortcut | `optional_sparse` | [`patches/tilelang_grouped_tp_optimized.py`](../../patches/tilelang_grouped_tp_optimized.py) (`fast_tau0`) | 9-10,31 | expanded |
| PackedLayout-derived protected dense prefix | `optional_sparse` | [`patches/h3_operator_policy.py`](../../patches/h3_operator_policy.py) (`packed_prefix`) | 9 | covered |
| Sparse dropping | `optional_approximation` | [`patches/h3_operator_policy.py`](../../patches/h3_operator_policy.py) (`tilelang_drop`) | 9-10 | covered |
| Scalar pooling with omitted-score workspace cap | `optional_approximation` | [`patches/h3_operator_policy.py`](../../patches/h3_operator_policy.py) (`scalar_score_workspace_mib`) | 10 | covered |
| Four channels, no P2P, automatic protocol | `default` | [`patches/comfy_tp_runtime.py`](../../patches/comfy_tp_runtime.py) (`NCCL_MIN_NCHANNELS`) | 3,36 | expanded |
| Native adapter order, full A, post-B scaling | `default` | [`patches/h3_lora.py`](../../patches/h3_lora.py) (`NativeAlignedLoRA`) | 16,32 | expanded |
| One-buffer activated-input gather and owned-delta lifetimes | `default` | [`patches/h3_lora.py`](../../patches/h3_lora.py) (`all_gather_into_tensor`) | 32 | added |
| Guarded selected-B FC1 loading/GEMM | `default_if_eligible` | [`patches/h3_lora.py`](../../patches/h3_lora.py) (`selective_rows`) | 16 | covered |
| Bounded hidden staging for optional Row Partial | `optional_approximation` | [`patches/h3_lora.py`](../../patches/h3_lora.py) (`HIDDEN_STAGE_BYTES`) | 16,32 | covered |
| Whole-layer up-to-two-GPU offline LoRA baking | `optional_requantization` | [`patches/offline_lora.py`](../../patches/offline_lora.py) (`cached_bake`) | 17 | covered |
| Positional tensor range reads and fallback | `default_platform_guarded` | [`patches/windowed_tp.py`](../../patches/windowed_tp.py) (`read_tensor`) | 18 | covered |
| Verified optional rank-local safetensors exports | `optional` | [`patches/rank_local_int8.py`](../../patches/rank_local_int8.py) (`load_offline_block`) | 18 | covered |
| Separate vision/embedding startup CPU groups | `default` | [`patches/comfy_tp_runtime.py`](../../patches/comfy_tp_runtime.py) (`startup_specs`) | 33 | added |
| On-demand vision GPU copy and phase fence | `default` | [`patches/comfy_tp_models.py`](../../patches/comfy_tp_models.py) (`StagedVision`) | 33 | added |
| Rank-zero leased refiner/input projections and broadcasts | `default` | [`patches/windowed_tp.py`](../../patches/windowed_tp.py) (`DiTPreprocessing`) | 33 | added |
| Process-shared host budget/cache/pin ownership | `default` | [`patches/host_weight_cache.py`](../../patches/host_weight_cache.py) (`HostBudget`) | 19,35 | covered |
| Event-owned bounded H2D residency and use leases | `default` | [`patches/weight_residency.py`](../../patches/weight_residency.py) (`ResidencyWindow`) | 19-20 | covered |
| Retain DiT groups across actual evaluations | `default_when_admitted` | [`patches/comfy_tp_runtime.py`](../../patches/comfy_tp_runtime.py) (`retain_keys`) | 34 | expanded |
| TE-complete DiT CPU preparation | `default` | [`patches/comfy_tp_runtime.py`](../../patches/comfy_tp_runtime.py) (`cross_model_cpu`) | 34 | added |
| Decode-start TE startup and lifecycle prefix prefetch | `mixed_defaults` | [`patches/comfy_tp_runtime.py`](../../patches/comfy_tp_runtime.py) (`decode_te_blocks`) | 34 | added |
| Reuse-aware CPU deadlines and eviction | `optional_lifecycle` | [`patches/host_weight_cache.py`](../../patches/host_weight_cache.py) (`reuse_aware`) | 34-35 | added |
| Optional cgroup/PSI file-cache admission | `optional` | [`patches/host_memory_pressure.py`](../../patches/host_memory_pressure.py) (`CgroupMemory`) | 35 | added |
| Optional one/two-group VRAM adaptive lookahead | `optional` | [`patches/weight_residency.py`](../../patches/weight_residency.py) (`_adapt_gpu_lookahead`) | 19 | covered |
| Attention/MLP caller lifetime repair | `default` | [`patches/tp_qwen3vl.py`](../../patches/tp_qwen3vl.py) (`del normed_x, q, k, v, attn_out`) | 7,20 | covered |
| Encoder-first / decoder-at-output VAE staging | `default_for_recognized_checkpoints` | [`patches/staged_vae.py`](../../patches/staged_vae.py) (`StagedVAE`) | 21 | covered |
| Global-KV aligned video query parallelism | `auto_with_native_fallback` | [`patches/parallel_vae.py`](../../patches/parallel_vae.py) (`AlignedVideoDecode`) | 22-23 | covered |
| FP32 stereo channel-parallel audio | `auto_accepted_rounding_difference` | [`patches/audio_channel_decode.py`](../../patches/audio_channel_decode.py) (`channel_decode`) | 24 | covered |
| Idle parent VAE encoders then worker TE prefix | `optional` | [`patches/comfy_tp_runtime.py`](../../patches/comfy_tp_runtime.py) (`_preload_idle_vaes`) | 25,34 | covered |
| Prompt compatibility, cancellation and pressure ownership | `optional_idle` | [`patches/comfy_idle_lifecycle.py`](../../patches/comfy_idle_lifecycle.py) (`PromptBridge`) | 25,34-35 | covered |
| Persistent serial TP session, Native sampler and diagnostics separation | `default` | [`patches/comfy_sampler.py`](../../patches/comfy_sampler.py) (`RemoteDenoiser`) | 3,27-28 | covered |
| Retained Kitchen RMSNorm/RoPE and Native mixed precision | `default` | [`patches/tp_dit.py`](../../patches/tp_dit.py) (`rms_rope_split_half`) | 8,29,31 | expanded |

## Experiments intentionally not promoted

| Family | Current disposition | Reason |
|---|---|---|
| INT8 PV / alternate attention mapping | `benchmark_only` | Current integrated TileLang attention uses FP16 PV. |
| Packed INT8 SUM and retired FP32 partial SUM | `not_default` | Native INT32 is default; scaled FP16 has its own explicit backend. |
| Concat A and streamed SwiGLU | `retired` | Rejected by current UI/runtime; Row Partial is a separate optional approximation. |
| Register/staged scales, castpad center, full QK cache, finite-output epilogue | `unselected_kernel_options` | Current live callers retain their specified flags; Video remains Kitchen. |
| Cropped slabs / video TP-SP / temporal audio | `historical_or_compatibility` | Current integration selects global-KV query and FP32 channel paths. |
| Preadv / mmap / descriptor advice / additional workers | `storage_study_only` | Current read_tensor uses positional pread with seek/readinto fallback. |
| ZSTD / Blosc2 / ZipNN / ZigZag-FSE | `CPU_benchmark_only` | No production decompression integration. |
| SM120 previews / historical Linux dependencies | `scoped_evidence_only` | Do not qualify current frozen conda Linux stack or claim T4 speed. |

## Search inventory and reproducibility

The [machine-readable coverage record](../data/h3_documentation_coverage_20261020.json) lists every reviewed non-index document with its source hash and role, each scanned experiment family and file count, and each feature with its checked source anchor/hash. The full local per-file search inventory is retained outside the repository; its SHA256 is in the compact record. Counts describe the pre-revision snapshot, so this new audit and its own data are not recursively counted.

Source anchors are checked against the saved implementation; they are navigation/coverage checks, not new numerical validation. Source line numbers describe this audit snapshot. Historical result JSONs are preserved. Rebuild and render the PDF using the separate document environment described in [PDF authoring](../pdf/README.md).

Validation of this revision includes all relative published Markdown/HTML links, Python syntax, PDF source hashes/bookmarks/page layout, explicit source preservation, and the focused existing CPU regression suite. No new operator tests or hardware measurements are necessary for these documentation changes.

## Reviewed document index

| Document | Interpretation |
|---|---|
| [HISTORICAL_README.md](../archive/HISTORICAL_README.md) | `historical_record` |
| [KAGGLE_P012_RECOVERY.md](../archive/KAGGLE_P012_RECOVERY.md) | `historical_record` |
| [KAGGLE_SPAWN_AND_SHARD_SCALES.md](../archive/KAGGLE_SPAWN_AND_SHARD_SCALES.md) | `historical_record` |
| [MANUAL_NATIVE_TIMEOUT.md](../archive/MANUAL_NATIVE_TIMEOUT.md) | `historical_record` |
| [IDLE_RESIDENCY_DESIGN.md](../design/IDLE_RESIDENCY_DESIGN.md) | `design_not_automatic_integration` |
| [INT8_SPARSE_ATTENTION_SM75_DESIGNS.md](../design/INT8_SPARSE_ATTENTION_SM75_DESIGNS.md) | `design_not_automatic_integration` |
| [INTEGRATION_PLAN.md](../design/INTEGRATION_PLAN.md) | `design_not_automatic_integration` |
| [OPERATOR_OPTIMIZATION_PLAN.md](../design/OPERATOR_OPTIMIZATION_PLAN.md) | `design_not_automatic_integration` |
| [QUALITY_FIRST_T4_EXPERIMENT_PLAN.md](../design/QUALITY_FIRST_T4_EXPERIMENT_PLAN.md) | `design_not_automatic_integration` |
| [REF2VA_BENCHMARK_PLAN.md](../design/REF2VA_BENCHMARK_PLAN.md) | `design_not_automatic_integration` |
| [SLAB_EXPERIMENT_PLAN.md](../design/SLAB_EXPERIMENT_PLAN.md) | `design_not_automatic_integration` |
| [TE_RESIDENCY_P01.md](../design/TE_RESIDENCY_P01.md) | `design_not_automatic_integration` |
| [TP2_GROUP32_KITCHEN_DESIGN.md](../design/TP2_GROUP32_KITCHEN_DESIGN.md) | `design_not_automatic_integration` |
| [VAE_KV_PROFILE_PLAN.md](../design/VAE_KV_PROFILE_PLAN.md) | `design_not_automatic_integration` |
| [ARTIFACT_RETENTION.md](../guides/ARTIFACT_RETENTION.md) | `current_reference_with_scoped_evidence` |
| [COMFYUI_SAMPLER_INTEGRATION.md](../guides/COMFYUI_SAMPLER_INTEGRATION.md) | `current_reference_with_scoped_evidence` |
| [H3_LEGACY_RUNTIME.md](../guides/H3_LEGACY_RUNTIME.md) | `current_reference_with_scoped_evidence` |
| [H3_LOADING_STAGES.md](../guides/H3_LOADING_STAGES.md) | `current_reference_with_scoped_evidence` |
| [H3_OPERATOR_ENHANCEMENTS.md](../guides/H3_OPERATOR_ENHANCEMENTS.md) | `current_reference_with_scoped_evidence` |
| [H3_OPERATOR_INTEGRATION.md](../guides/H3_OPERATOR_INTEGRATION.md) | `current_reference_with_scoped_evidence` |
| [OFFLINE_CHECKPOINT_PUBLICATION.md](../guides/OFFLINE_CHECKPOINT_PUBLICATION.md) | `current_reference_with_scoped_evidence` |
| [PARALLEL_VAE_INTEGRATION.md](../guides/PARALLEL_VAE_INTEGRATION.md) | `current_reference_with_scoped_evidence` |
| [2-Slab Video VAE.md](../reports/2-Slab%20Video%20VAE.md) | `version_bound_experiment_or_review` |
| [ATTENTION_BOTTLENECK_REPORT.md](../reports/ATTENTION_BOTTLENECK_REPORT.md) | `version_bound_experiment_or_review` |
| [ATTENTION_T4_COMPATIBILITY.md](../reports/ATTENTION_T4_COMPATIBILITY.md) | `version_bound_experiment_or_review` |
| [AUDIO_CHANNEL_ERROR_DIAGNOSIS.md](../reports/AUDIO_CHANNEL_ERROR_DIAGNOSIS.md) | `version_bound_experiment_or_review` |
| [BENCHMARK_REPORT.md](../reports/BENCHMARK_REPORT.md) | `version_bound_experiment_or_review` |
| [CAUSAL_PROFILING.md](../reports/CAUSAL_PROFILING.md) | `version_bound_experiment_or_review` |
| [CAUSAL_T4_RESULTS.md](../reports/CAUSAL_T4_RESULTS.md) | `version_bound_experiment_or_review` |
| [CPU_CRITICAL_PATH_AUDIT.md](../reports/CPU_CRITICAL_PATH_AUDIT.md) | `version_bound_experiment_or_review` |
| [DIT_TEMPORARY_TENSOR_LIFETIMES.md](../reports/DIT_TEMPORARY_TENSOR_LIFETIMES.md) | `version_bound_experiment_or_review` |
| [FC2_QUANTIZER_MEMORY.md](../reports/FC2_QUANTIZER_MEMORY.md) | `current_reference_with_scoped_evidence` |
| [FINAL_PARETO_REPORT.md](../reports/FINAL_PARETO_REPORT.md) | `version_bound_experiment_or_review` |
| [GENERATION_QUALITY_REPORT.md](../reports/GENERATION_QUALITY_REPORT.md) | `version_bound_experiment_or_review` |
| [GROUP32_OPTIMIZATION_REPORT.md](../reports/GROUP32_OPTIMIZATION_REPORT.md) | `version_bound_experiment_or_review` |
| [H3_INT8_SWIGLU_LORA.md](../reports/H3_INT8_SWIGLU_LORA.md) | `version_bound_experiment_or_review` |
| [H3_PHYSICAL_INT8_LORA_RESULTS.md](../reports/H3_PHYSICAL_INT8_LORA_RESULTS.md) | `version_bound_experiment_or_review` |
| [H3_PIPELINE_REVIEW.md](../reports/H3_PIPELINE_REVIEW.md) | `current_reference_with_scoped_evidence` |
| [H3_RUNTIME_PROVENANCE_AUDIT.md](../reports/H3_RUNTIME_PROVENANCE_AUDIT.md) | `current_reference_with_scoped_evidence` |
| [HOST_CACHE_IMPLEMENTATION.md](../reports/HOST_CACHE_IMPLEMENTATION.md) | `version_bound_experiment_or_review` |
| [L1_OFFLINE_MEMORY_PROFILE.md](../reports/L1_OFFLINE_MEMORY_PROFILE.md) | `version_bound_experiment_or_review` |
| [LOADING_OPTIMIZATION.md](../reports/LOADING_OPTIMIZATION.md) | `version_bound_experiment_or_review` |
| [LOCAL_KITCHEN_GROUP32_PREVIEW.md](../reports/LOCAL_KITCHEN_GROUP32_PREVIEW.md) | `version_bound_experiment_or_review` |
| [ONLINE_LORA_BYTE_EXACT_OPTIMIZATION.md](../reports/ONLINE_LORA_BYTE_EXACT_OPTIMIZATION.md) | `version_bound_experiment_or_review` |
| [ONLINE_LORA_NATIVE_ORDER_REPAIR.md](../reports/ONLINE_LORA_NATIVE_ORDER_REPAIR.md) | `version_bound_experiment_or_review` |
| [ONLINE_LORA_OPTIONS.md](../reports/ONLINE_LORA_OPTIONS.md) | `version_bound_experiment_or_review` |
| [P012_T4_RESULTS.md](../reports/P012_T4_RESULTS.md) | `version_bound_experiment_or_review` |
| [PIPELINE_GPU_MEMORY_PEAK.md](../reports/PIPELINE_GPU_MEMORY_PEAK.md) | `version_bound_experiment_or_review` |
| [PIPELINE_RESIDENCY_AUDIT.md](../reports/PIPELINE_RESIDENCY_AUDIT.md) | `version_bound_experiment_or_review` |
| [REF2VA_E2E_20261019.md](../reports/REF2VA_E2E_20261019.md) | `version_bound_experiment_or_review` |
| [REF2VA_E2E_T4_RESULTS.md](../reports/REF2VA_E2E_T4_RESULTS.md) | `version_bound_experiment_or_review` |
| [REF2VA_T4_RESULTS.md](../reports/REF2VA_T4_RESULTS.md) | `version_bound_experiment_or_review` |
| [REPAIRED_T4_REPORT.md](../reports/REPAIRED_T4_REPORT.md) | `version_bound_experiment_or_review` |
| [RESIDENCY_IMPLEMENTATION.md](../reports/RESIDENCY_IMPLEMENTATION.md) | `version_bound_experiment_or_review` |
| [RETEST_REPORT.md](../reports/RETEST_REPORT.md) | `version_bound_experiment_or_review` |
| [ROW_PARTIAL_MEMORY.md](../reports/ROW_PARTIAL_MEMORY.md) | `current_reference_with_scoped_evidence` |
| [SELECTIVE_B_FEASIBILITY.md](../reports/SELECTIVE_B_FEASIBILITY.md) | `version_bound_experiment_or_review` |
| [SEMANTIC_VALIDATION.md](../reports/SEMANTIC_VALIDATION.md) | `version_bound_experiment_or_review` |
| [SLAB_KV_DIAGNOSIS.md](../reports/SLAB_KV_DIAGNOSIS.md) | `version_bound_experiment_or_review` |
| [SWIGLU_PADDING_FUSION.md](../reports/SWIGLU_PADDING_FUSION.md) | `version_bound_experiment_or_review` |
| [TEXT_ENCODER_SP_INT8_REPORT.md](../reports/TEXT_ENCODER_SP_INT8_REPORT.md) | `version_bound_experiment_or_review` |
| [TE_CAUSAL_PROFILE.md](../reports/TE_CAUSAL_PROFILE.md) | `version_bound_experiment_or_review` |
| [TE_COMPRESSION_LOCAL_20261020.md](../reports/TE_COMPRESSION_LOCAL_20261020.md) | `version_bound_experiment_or_review` |
| [TE_CRITICAL_PATH.md](../reports/TE_CRITICAL_PATH.md) | `version_bound_experiment_or_review` |
| [TE_DIT_STUDY_REPORT.md](../reports/TE_DIT_STUDY_REPORT.md) | `version_bound_experiment_or_review` |
| [TE_FILE_READ_CAUSALITY.md](../reports/TE_FILE_READ_CAUSALITY.md) | `version_bound_experiment_or_review` |
| [TE_IDLE_RESIDENCY_RESULTS.md](../reports/TE_IDLE_RESIDENCY_RESULTS.md) | `version_bound_experiment_or_review` |
| [TE_IO_ABLATION_RESULTS.md](../reports/TE_IO_ABLATION_RESULTS.md) | `version_bound_experiment_or_review` |
| [TE_IO_CAUSAL_AUDIT.md](../reports/TE_IO_CAUSAL_AUDIT.md) | `version_bound_experiment_or_review` |
| [TE_RESIDENCY_OPTIMIZATION.md](../reports/TE_RESIDENCY_OPTIMIZATION.md) | `version_bound_experiment_or_review` |
| [TE_RESIDENCY_P01_RESULTS.md](../reports/TE_RESIDENCY_P01_RESULTS.md) | `version_bound_experiment_or_review` |
| [TILELANG_T4_ATTENTION_RESULTS.md](../reports/TILELANG_T4_ATTENTION_RESULTS.md) | `version_bound_experiment_or_review` |
| [TP2_ATTENTION_TRANSPORT_TRADEOFFS.md](../reports/TP2_ATTENTION_TRANSPORT_TRADEOFFS.md) | `version_bound_experiment_or_review` |
| [TP2_KITCHEN_GROUP32_RESULTS.md](../reports/TP2_KITCHEN_GROUP32_RESULTS.md) | `version_bound_experiment_or_review` |
| [TP2_KITCHEN_GROUP32_V2_RESULTS.md](../reports/TP2_KITCHEN_GROUP32_V2_RESULTS.md) | `version_bound_experiment_or_review` |
| [TP2_OFFLINE_OPTIMIZATION.md](../reports/TP2_OFFLINE_OPTIMIZATION.md) | `version_bound_experiment_or_review` |
| [TP2_OPERATOR_OPTIMIZATION.md](../reports/TP2_OPERATOR_OPTIMIZATION.md) | `version_bound_experiment_or_review` |
| [TP2_OPERATOR_PASS2.md](../reports/TP2_OPERATOR_PASS2.md) | `version_bound_experiment_or_review` |
| [TP2_OPERATOR_RESULTS.md](../reports/TP2_OPERATOR_RESULTS.md) | `version_bound_experiment_or_review` |
| [TP2_P2_OPTIMIZATION_REPORT.md](../reports/TP2_P2_OPTIMIZATION_REPORT.md) | `version_bound_experiment_or_review` |
| [TP2_ROW_FUSION_RESULTS.md](../reports/TP2_ROW_FUSION_RESULTS.md) | `version_bound_experiment_or_review` |
| [VAE_KV_T4_RESULTS.md](../reports/VAE_KV_T4_RESULTS.md) | `version_bound_experiment_or_review` |
| [VAE_TILE_BENCHMARK.md](../reports/VAE_TILE_BENCHMARK.md) | `version_bound_experiment_or_review` |
| [VIDEO_VAE_QUERY_FALLBACK_CHECK.md](../reports/VIDEO_VAE_QUERY_FALLBACK_CHECK.md) | `current_reference_with_scoped_evidence` |
| [VIDEO_VAE_TP_BENCHMARK.md](../reports/VIDEO_VAE_TP_BENCHMARK.md) | `version_bound_experiment_or_review` |
