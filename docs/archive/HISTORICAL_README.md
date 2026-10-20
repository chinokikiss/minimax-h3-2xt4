# MiniMax-H3 Deployment & Maximum Optimization on 2× NVIDIA T4 GPUs

This repository contains the architecture, benchmarks, optimization patches, and Kaggle automation suite for deploying **MiniMax-H3** on **2× NVIDIA T4 GPUs (16 GB VRAM each)** with maximum performance optimization.

**ComfyUI runtime:** preserve the local `ComfyUI` checkout at `d49e888` and the existing conda `comfyui` environment. `launch_h3_legacy.cmd` uses that exact interpreter. Kaggle setup uses the frozen source/package manifest and a CPU-prepared wheelhouse; Linux/T4 qualification of this conda stack remains pending. See [setup, pins and validation](../guides/H3_LEGACY_RUNTIME.md). The legacy attention API is retained.

**Loading stages (2026-10-19):** optional offline LoRA loading now bakes whole layers across two GPU workers and caches a new checkpoint. H3 VAE loading is encoder-only for reference inputs and decoder-only for outputs. Idle residency preloads the VAE encoders before the remaining TE window. Small trained Native encode/decode comparisons are byte-exact locally; no new dual-T4 timing or full H3 generation was run. See [loading options and validation](../guides/H3_LOADING_STAGES.md).

**Operator enhancements (2026-10-19):** optional scaled FP16 SUM and fused epilogue, TE stage buffer release, positional checkpoint reads, opt-in VRAM-aware lookahead, and native SDPA for the Token Refiner are implemented. Packed/native-INT32 row transport remains the default. Local arithmetic and regression gates pass; these changes have no new T4/NCCL performance qualification. See [configuration and checks](../guides/H3_OPERATOR_ENHANCEMENTS.md).

**Operator integration (2026-10-18):** the TP2 loader now defaults to TileLang Group32 dense attention for TE and Kitchen INT8 attention for DiT/Video VAE, with optional TileLang dropping/scalar pooling for DiT. Packed row-linear, SwiGLU padding, qualified FC1 Selective B and persistent-worker audio decoding are integrated. Two-T4 operator smoke gates pass; full-generation quality and integrated performance remain unmeasured. See [configuration, workflow and validation](../guides/H3_OPERATOR_INTEGRATION.md).

**Causal pipeline diagnostic (2026-10-13):** the [continuation](https://www.kaggle.com/code/chinokiki9/minimax-h3-causal-pipeline-resume) is COMPLETE. All six remaining passes and Native correctness checks passed; Offline+P1+P2 averaged 379.20 s versus P1+P2 387.10 s and P1 399.21 s on the same allocation. Measurement results are complete; contention/stall attribution remains partial, and an attention category-classification defect requires trace re-analysis. See [results and audit limits](../reports/CAUSAL_T4_RESULTS.md) and [capture design](../reports/CAUSAL_PROFILING.md).

**TE critical path (2026-10-14):** the [dual-T4 loading ablation job](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-critical-path-ablations) is COMPLETE. All 13 full requests passed Native correctness checks. Coalesced loading reduced warm TE/end-to-end latency by 4.11%/2.71% in the initial comparison, but only 0.91%/1.14% in reverse-order confirmation; it failed the required 3% TE gate, so baseline remains selected. Parallel preparation did not beat coalesced alone initially. Diagnostic inference passed, but report generation hit a bundled analyzer dependency mismatch; saved evidence permits local recovery. Exact GIL wait duration remains unavailable. See [physical evidence and comparisons](../reports/TE_CRITICAL_PATH.md).

**TE file-read investigation (2026-10-14):** [the physical dual-T4 job](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-file-read-causality) is COMPLETE: 35 byte-checked read controls and two Native-correct full requests. Warm block reads take 88.7 s per rank but only 9.3 s CPU; source residency is ~2.4%, with heavy background reclaim/refault and ~250 MB/s backing reads. Small cache-eviction probes hit a lower cached storage layer and did not read backing `sdb`. Direct reclaim and caller major faults are not the main cause. See [verified cause, trace intervals and ranked solutions](../reports/TE_FILE_READ_CAUSALITY.md).

**TE residency P0/P1 (2026-10-14):** [physical A/B version 1](https://www.kaggle.com/code/chinokiki9/minimax-h3-residency-p0-p1) is COMPLETE: all 21 full requests passed Native checks. P0+P1 reduced warm end-to-end latency by 1.98% initially and 2.98% in reverse-order confirmation, with 11.50% less observed whole-device read traffic in confirmation. TE gains are small, Future waits persist and the cgroup still reaches its limit. Defaults remain unchanged. See [measured results and limits](../reports/TE_RESIDENCY_P01_RESULTS.md) and [implementation audit](../design/TE_RESIDENCY_P01.md).

**TE block causality and idle residency (2026-10-15):** the [physical block diagnostic](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-block-critical-path) is COMPLETE with two Native-correct full requests. Warm block 2–49 worker observations show ~80 s off CPU per rank, only 0.30–0.34 s runnable delay, ~87% D-state samples and dominant folio waits. Verified weight dependencies intersect 61.017 s of both-GPU inactivity; this is not a predicted speedup. The first [idle pilot](https://www.kaggle.com/code/chinokiki9/minimax-h3-idle-residency-pilot) stopped because a diagnostic wrapper suppressed idle preparation; four full requests were correct but no idle work ran. The wrapper is repaired/requalified; pilot version 2 completed all 14 requests with Native checks passing. Its 15 s standalone warm median TE/E2E improved 17.81%/4.80%, but prefetch continued into reference encoding unlike the ComfyUI prompt-start pause, so production lifecycle equivalence remains unverified. [Full-working-set storage controls](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-storage-controls) completed all 22 cases and 2,008 byte-correct block loads without OOM. Pread merits follow-up, but its read-union gain is only 3.41%/0.84%; no new inference speedup is established. P1/idle remain off. See [I/O causal audit](../reports/TE_IO_CAUSAL_AUDIT.md), [ablation status](../reports/TE_IO_ABLATION_RESULTS.md) and [implementation/rollback](../reports/TE_RESIDENCY_OPTIMIZATION.md).

**Artifact policy:** detailed inference event histories and benchmark tracing are opt-in. Thirty completed trace files were checksum-verified through compression, freeing 10.72 GiB while preserving references and evidence. See [retention and restore commands](../guides/ARTIFACT_RETENTION.md).

**Default parallel VAE integration (2026-10-11):** the TP2 workflow now uses Native-aligned Query Partitioning for video and FP32 Channel Parallel for audio. Native public chunk/tile/output handling is retained, with per-model `native` overrides and coordinated memory fallback. The full public video checks pass bit-exact on local logical ranks; audio retains its diagnosed small rounding difference. Integrated physical T4 latency is not yet measured. See [implementation, use and qualification](../guides/PARALLEL_VAE_INTEGRATION.md).

**ComfyUI sampler integration (2026-10-09):** standard MODEL/CLIP/VAE adapters now use Native conditioning, vision, actual guider/samplers and audiovisual conversions with persistent TP2 workers. Full 50-layer TE/DiT local qualification passes seven bit-exact comparisons. The physical dual-T4 guided/cancellation/decode gate passes eleven bit-exact comparisons using two trained TE/DiT layers and full Native vision/video VAE. Separate TP2 and Native INT8 workflow variants are available. See [use, evidence and limits](../guides/COMFYUI_SAMPLER_INTEGRATION.md). Complete queued workflow acceptance and integrated latency remain unmeasured.

**Residency implementation:** request-wide fixed/adaptive weight streaming and event-owned handoff are implemented. Earlier local trained CUDA simulations passed the TE/DiT token matrix and a composed 50-layer TE → 50-layer DiT → 36-layer Native video decode. See [implementation and scope](../reports/RESIDENCY_IMPLEMENTATION.md) and the [integration plan](../design/INTEGRATION_PLAN.md).

**Persistent RAM cache:** next-request TE preload after video decode is implemented with one shared 16 GiB managed-host budget for the 32 GB Kaggle host. The earlier physical block/compact-video gate passed ten bit-exact comparisons and verified post-decode preload ordering, TE RAM reuse, and a 12.74 GiB peak managed reservation. The current sampler integration adds the Native VAE completion hook. See [historical host cache evidence and scope](../reports/HOST_CACHE_IMPLEMENTATION.md).

**Latest accepted physical TE/DiT evidence:** optimized TP2 beats Native on the measured single-request full-entry workloads (TE S512/2048/4096 and DiT S8192). Earlier rankings below describe historical implementations. See [the current dual-T4 results and limits](../reports/REF2VA_T4_RESULTS.md).

**Completed benchmark review (2026-10-07):** corrected native 50-block placement passed all 30 comparisons. TE/DiT TP/SP failed numerical correctness, and video TP/SP was slower than native decoding. The original reports retain historical estimates; their strategy rankings and generation speedups are unsupported. See [the findings and repair plan](../reports/RETEST_REPORT.md) and completed measurements (`kaggle_retest_output/final/MEASURED_RESULTS.md`, local-only evidence).

**Actual dual-rank simulation (2026-10-08):** 51 comparisons passed bit-exact on the 5060 Ti, including trained TE/DiT blocks and the full trained 36-block video decoder. Detected mask, rotary, packed-weight, global-index and native-kernel/rounding defects were repaired. Full TE/DiT integration and physical NCCL remain unverified. See [repairs and validation scope](../reports/SEMANTIC_VALIDATION.md).

**Repaired dual-T4 benchmark (2026-10-08):** TE/DiT TP/SP passed all 12 stored comparisons bit-exact, but were slower than native on every tested workload. Video TP failed its first physical correctness check (1.58% L2); video SP was not attempted afterward. The run is stopped. See [physical findings and actionable work](../reports/REPAIRED_T4_REPORT.md).

**Two-slab video results:** physical scheduling and INT8-kernel checks passed. At 768x1344, overlap 64/128 took 974.7/1022.3ms, with 8.11%/7.53% full-decoder L2 error; native raw decoding took 2190.3ms. All slabs fail the original 1% quality gate. Treat them as an optional approximation and validate generated-video quality before deployment. See full measurements (`kaggle_repaired_output/MEASURED_RESULTS.md`, local-only evidence).

---

## 1. Executive Summary & Hardware Topology

### Target Environment: Kaggle 2× T4
- **GPUs:** 2× NVIDIA Tesla T4 (Turing architecture, Compute Capability 7.5)
- **VRAM:** 16 GB GDDR6 per GPU (32 GB total VRAM)
- **Observed peer-access eligibility:** `torch.cuda.can_device_access_peer(0, 1) == True` in both retest allocations.
- **Retest transport settings:** NCCL P2P and IB were explicitly disabled. Those flags do not classify PyTorch's direct cross-device copies used by native layer placement. Inspect each allocation's topology and transport rather than assuming all T4 pairs behave identically.

---

## 2. Measured findings and integration priorities

The MiniMax-H3 conditioning model is based on **Qwen3-VL-32B** (truncated to the first 50 layers), packaged as `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (**~27.14 GB**).

### The Challenge
A 27.14 GB checkpoint cannot remain fully resident on a single T4. Native layer placement across two GPUs is a viable capacity strategy for the tested blocks. A complete deployment must also account for embeddings, vision, heads, activations and component lifecycle.

### Completed rerun

| Experiment | Measured result | Scope and assessment |
| :--- | :--- | :--- |
| Native TE, 50 blocks, S512 | **969.70 ms** | Corrected two-GPU placement; all tested stack outputs report bit equality with a streamed native reference. |
| Native DiT, 50 blocks, S8192 | **50,862.64 ms** | FP32 native computation; all tested stack outputs report bit equality. Excludes refiner, heads and sampling. |
| TE/DiT TP/SP blocks | **60 failed correctness comparisons** | Their faster TP block timings cannot establish equivalent-computation speedups. |
| Full video, 768×1344, two latent frames | Native **14.060 s**, TP **30.592 s**, SP **29.583 s** | Native is faster on all five tested workloads; TP/SP also fail the declared numerical gate. |

### Recommended work order

1. Integrate native placement with explicit CUDA device guards, native conditioning behavior and measured memory headroom. The harness does not validate the complete production PP loaders.
2. Keep native video decoding as the default. The repaired TP/SP paths pass local component checks; measure their revised physical behavior only if performance uncertainty affects a decision. Old timings describe the previous algorithms.
3. Evaluate channel-split audio as an optional approximation: its observed 1.71–1.92× timing benefit comes with small errors that fail the strict zero-like tolerance.
4. Keep tiling explicitly approximate. Parallel tile paths match their sequential equivalents, but differ substantially from full-context outputs and are not faster on every workload.

No CPU-offload latency baseline or full-generation speedup was measured. The prior 30× and PP-versus-TP ranking claims are withdrawn. Use [the detailed repair plan](../reports/RETEST_REPORT.md) for specific causes, fixes and acceptance criteria.

---

## 3. Project Structure

```
.
├── ComfyUI/                         # Upstream ComfyUI repo
├── benchmarks/
│   ├── benchmark_te_performance.py  # Empirically benchmarks PP=2 vs TP=2 vs CPU Offload
│   ├── test_ck_int8.py              # Verifies comfy_kitchen INT8 ConvRot execution
│   └── test_tp_int8_math.py         # Numerically verifies Column/Row parallel slicing
├── patches/
│   ├── pp_qwen3vl.py                # High-speed Layer-Sharded Pipeline Parallelism (PP=2)
│   ├── tp_qwen3vl.py                # Full Tensor Parallelism (TP=2) for Qwen3-VL INT8
│   ├── comfy_sampler.py            # Native MODEL/CLIP/VAE adapters
│   └── comfy_tp_runtime.py         # Persistent TP2 workers and residency
├── custom_nodes/minimax_h3_tp2/     # ComfyUI loader and VAE handoff nodes
├── workflows/                      # Separate TP2 and Native INT8 variants
├── automation/
│   └── kaggle_browser_controller.py # browser-use automated controller for Kaggle 2x T4
├── notebooks/
│   └── 01_minimax_h3_te_tp2_benchmark.ipynb  # Interactive Kaggle benchmark notebook
└── README.md
```

---

## 4. Quick Start

### A. Inspect completed results without GPU work
```bash
python summarize_retest.py kaggle_retest_output/final
python audit_retest.py kaggle_retest_output/cu130 --suites audio,video,video_tiling,te --skip-profile
python audit_retest.py kaggle_retest_output/fp32_pp --suites dit,te_stack,dit_stack --skip-profile
```

The audits above cover downloaded core JSON artifacts and timing arithmetic; they skip full profiler traces. See the report for limitations and the explicit allocation selection.

### B. Historical benchmark and automation entry points

The original benchmark scripts, notebook and browser automation remain in the repository for reference. They are superseded as evidence by the completed rerun and contain the methodological limitations described in the report. Running those entry points starts additional experiments; they are not needed to inspect the saved results.

---

## 5. Next Optimization Stages

1. **Diffusion Model (`ref2va` Pruned INT8 ConvRot ~20.9 GB):**
   - Apply sequential module swapping (TE unloads before DiT loads) or shard DiT transformer blocks across GPU 0 and GPU 1.
   - Preserve supported native precision, conditioning and attention behavior. An attention-backend speedup requires a separate equivalent-computation measurement; no 2× attention speedup is established by this rerun.
2. **Video VAE (INT8 ConvRot ~2.81 GB):**
   - The ~2.81 GB checkpoint size is not total runtime VRAM. Native decoding completed all tested workloads on one T4 and was faster than TP/SP. Use native decoding when it fits; expose spatial tiling as an optional approximation after quality validation.


Current online LoRA: [Native default and optional Row Partial, with memory audit](../reports/ROW_PARTIAL_MEMORY.md). Row Partial defaults off. Concat A is deprecated; streamed mode is removed.
