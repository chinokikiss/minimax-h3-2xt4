# H3 Sol v2 Production E2E — Physical 2×T4 Result and Baseline Comparison

**Date:** 2026-10-30  
**Kaggle kernel:** `pamcks/h3-sol-v2-production-e2e-20261030/1`  
**Status:** COMPLETE (batch + generation); worker exit code 0  
**Source:** `docs/data/h3_sol_v2_production_e2e_20261030/sol_v2_production_e2e/{batch_results.json,generation/result.json,request.json,source_overlay.json}`

## Scope and source boundary

The fixed production run used two physical Tesla T4 (SM75), PyTorch 2.11.0+cu128, CUDA 12.8, Kitchen 0.2.37, TileLang 0.1.13, and the same 1216×704 × 243-frame / 240-export-frame / 10-second regular 8-step Turbo request, seed 20261020, Euler/Simple, CFG 1, video shift 6, audio shift 3 and offline-baked LoRA.

Attention schedule: TE Kitchen causal; DiT NFE1/NFE8 Kitchen, NFE2–7 Sol v2 at tau 1.0; Video VAE Sol v2 native D64; audio Native. The source overlay was submitted **before** the later ABI11/ABI12 native-Q optimization sprint. This E2E cannot qualify those later changes.

## Completion and actual dispatch

- Full E2E subprocess: **1506.675 s**; separately measured offline disk bake **328.539 s**.
- Foreground generation: **1458.364 s**.
- Both ranks: **100 Kitchen + 300 Sol** DiT attention dispatches, precisely 50 calls in each NFE; zero out-of-command dispatch.
- Video VAE: `native_temporal_pairs`, Sol backend, **2016 Sol attention calls per rank**, no fallback.
- Audio VAE: Native, no fallback.
- Eight completed denoiser calls and eight sampler steps.
- Finite final video/audio latents, decoded pixels and waveform.
- Decoded: pixels `[1,243,704,1216,3]`; waveform `[1,324000,2]`. Exported duration: **10.0 s**.
- Final latent SHA256: video `d029275e51e30024760f11d9ba35e4be11cbcacea3099f244fa79faa83786854`; audio `950ebe9fead0fabc45b24478f386388f831cdcd1cf979087f8427d9cd44b769d`.

## Phase-level comparisons (wall seconds)

| Stage | All Kitchen baseline | Old Sol Hybrid v4 | Production Sol v2 |
|---|---:|---:|---:|
| Reference + TE | 171.878 | 181.249 | **145.501** |
| Sampler (all 8 NFE + preparation) | 1348.776 | **1191.284** | 1191.718 |
| Video decode | **92.150** | 100.943 | 93.636 |
| Audio decode | 3.004 | 2.973 | 2.892 |
| File export | 22.194 | 19.845 | 20.013 |
| **Foreground** | 1639.157 | 1497.299 | **1458.364** |
| Full E2E subprocess | 1703.296 | 1549.299 | **1506.675** |
| Offline disk bake (separate) | 365.356 | 340.140 | **328.539** |

Historical baselines are recorded in `docs/reports/H3_TURBO_085MP_10S_E2E_20261030.md` and `docs/reports/H3_SOL_HYBRID_085MP_10S_T4_20261030_RESULTS.md`. Values are descriptive cross-run comparisons, not controlled identical-timestep operator A/B tests.

Derived from observed wall times:

- Sol v2 vs Kitchen: **157.058 s / 11.64%** shorter sampler; **180.793 s / 11.03%** shorter foreground, speedup **1.124×** on the foreground boundary.
- Sol v2 vs old Hybrid v4: sampler **0.434 s slower**, practically unchanged at this granularity; foreground **38.935 s / 2.60%** shorter, largely associated with the **35.748 s** shorter reference/TE interval and a **7.307 s** shorter video decode. Neither can be attributed to ABI11/ABI12 optimizations.
- Existing source changed between runs. External workload identity and matching operator policy do not make the measured differences exclusively causal.

## Numerical and quality boundary

The final latent hashes differ from old Hybrid v4. A different SHA256 proves bitwise nonidentity only; it does not quantify L2 drift or establish better/worse perceived quality. Human visual/audio assessment (reference identity, umbrella persistence, spirit continuity, vertical jitter, audio alignment) remains **pending**.

No dense Native generation-quality equivalence is claimed. Full E2E successful execution is not proof of semantically acceptable long-video quality.

## Current local ABI12 and next qualification

After this submitted E2E, the workspace source has advanced to **native ABI12** with true local-Q / full-KV preparation and compact Q routing, beyond ABI11's main-kernel-only query partition. The following local CPU gates passed in the current conda environment:

- `benchmarks.test_sol_production`: **16/16**.
- `benchmarks.test_h3_configuration_defaults`, `benchmarks.test_source_overlay_closure`, `benchmarks.test_video_query_fallback`: **15/15**.

**These 31 CPU tests do not establish CUDA correctness or physical-T4 performance of ABI12.**

The prior ABI11 v4 T4 microgate was RUNNING at the previous handoff; its completed artifacts have not been qualified in this continuation. Per project policy, do not status-poll or auto-resubmit. Once completion is independently known, collect its final artifact and verify D64 native semantics, direct Q scales, prepared/sync-free consistency, topology, joined-vs-full query partition, latency and temporary memory. Then qualify exact latest ABI12 source on a zero-weight T4 gate, followed by real 65,166-token attention replay. Only then consider another E2E.

## Artifact note

This continuation used one targeted final-output collection for the **already confirmed complete** production E2E. It downloaded the batch, generation, request, source overlay and supervision JSON to `docs/data/h3_sol_v2_production_e2e_20261030/`. No Kaggle status/list/live-log request or new GPU job was made.
