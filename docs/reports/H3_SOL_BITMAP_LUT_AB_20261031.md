# H3 Sol Bitmap LUT A/B — Physical Tesla T4 Result Report

**Experiment:** H3 Sol Bitmap LUT AB 20261031  
**Provider:** Kaggle, `pamcks/h3-sol-bitmap-lut-ab-20261031`, version 1  
**Result:** `complete`; **numerical/route semantics PASS**; **memory improvement PASS**; **long-sequence latency improvement NOT demonstrated**.

## 1. Verified environment and provenance

- GPU: **NVIDIA Tesla T4** (SM75); NVIDIA compiler: **CUDA 12.8**, `cuda_12.8.r12.8/compiler.35583870_0`.
- Runtime: PyTorch `2.11.0+cu128`; SM75 native library ABI **12**.
- Runner exit code: **0**; runtime: **28.246 s** for main ABI12 microgate (not full Bitmap A/B experiment time).
- Default ABI12 microgate: **14/14** acceptance flags true.
- Bitmap comparison: **2/2** `semantic_gate=true`.
- Frozen source: runner source hashes **exactly match all 23 source-binding hashes**.
- Collection: 8 final JSON/log files downloaded from **Kaggle version 1**; **8/8 SHA256 verified**. No status/list/live-log polling or GPU resubmission.
- Original request vs resolved kernel slug differ. The **actual provider URL** is `https://www.kaggle.com/code/pamcks/h3-sol-bitmap-lut-ab-20261031`; final artifact retrieval was bound to this URL's slug and **version 1**.

## 2. Exact-route and numerical verification

| Property | D64 VAE (1×8×1029×64) | D128 long (1×4×65166×128) |
|---|---:|---:|
| ExactMask decoded equality | PASS | PASS |
| RowCount equality | PASS | PASS |
| Logical ordered-LUT SHA256 | PASS | PASS |
| Output relative L2 | **0.0** | **0.0** |
| Semantic gate | **PASS** | **PASS** |
| Routing density | 56.4446% | 34.4093% |

The comparison is between separate processes using the same deterministic seeds. Bitmap entries were decoded into the same **ascending sequence of exact key-block indices** as the original int32 LUT. Equality therefore includes both *membership and execution order*.

The recorded D128 cosine is `1.0040393`, slightly above the mathematical upper bound of 1. This is a **floating-point reduction artifact / metric implementation issue** and should not be represented as a meaningful >1 similarity. The decisive output equality measure for this run is the separately recorded `rel_l2=0` (and the matching route/order checks). The D64 cosine is `0.9999965`.

## 3. Physical GPU latency

Reported values are **medians of three runs**, in milliseconds. Timing of preparation and execution stages is independently sampled; stage medians are not additive.

| Workload | Metric | Original int32 LUT | Maskless Bitmap LUT | Change |
|---|---|---:|---:|---:|
| D64 VAE, L=1029 | Prepare | 0.3542 | 0.3395 | −4.16% |
| D64 VAE, L=1029 | Execute | 0.2131 | 0.2020 | −5.18% |
| D64 VAE, L=1029 | **Complete call** | **0.5120** | **0.4777** | **−6.71% (1.072×)** |
| D128, L=65166 | Prepare | 9.4904 | 9.3636 | −1.34% |
| D128, L=65166 | Execute | 129.7000 | 135.5314 | **+4.50% (slower)** |
| D128, L=65166 | **Complete call** | **139.4679** | **140.0346** | **+0.41% (slower)** |

**Interpretation:** the bitmap route representation reduces memory substantially, and improves short D64 latency in this microgate. For long D128, however, main-kernel `execute` regresses 4.50%, and full attention does **not** speed up. Different independently sampled stage and complete medians must not be summed to infer an exact stage decomposition.

## 4. Physical GPU temporary peak memory

Memory units use decimal MB below (`1 MB = 10^6 bytes`).

| Workload | Original | Bitmap | Reduction |
|---|---:|---:|---:|
| D64 VAE L=1029 | 2,751,488 B (2.751 MB) | 2,740,224 B (2.740 MB) | 11,264 B / **0.41%** |
| D128 L=65166 | **189,895,168 B (189.895 MB)** | **169,445,376 B (169.445 MB)** | **20,449,792 B (19.50 MiB) / 10.77%** |

The D128 logical LUT changes from `[1,4,1019,1019]` (16,613,776 B) to `[1,4,1019,32]` (521,728 B); the main kernel no longer holds the redundant byte ExactMask. The *observed* total temporary peak reduction is **20,449,792 B**, and is not simply equated to the theoretical LUT byte savings.

Earlier Kitchen `~167.2 MB` is from **another Real-H3 benchmark**. The current 169.445 MB result uses **synthetic deterministic Q/K/V shaped like H3**, not captured post-RoPE H3 activations. Accordingly, Kitchen-vs-Bitmap absolute performance comparisons are **not controlled A/B evidence** and are not used to declare pass/fail here.

## 5. Scope, limitations, and recommendation

**Qualified in this run:** native CUDA 12.8 compilation and execution on Tesla T4; ABI12 default gate; *original-vs-bitmap* exact route, ordered LUT, row counts and operator output; D64 and D128 microgate latency and temporary allocated peak.

**Not qualified:** full H3 generation, actual post-RoPE H3 input distribution, two-rank Real-H3 performance, Video VAE end-to-end latency, production-wide peak VRAM.

**Engineering decision:** mark Maskless Bitmap LUT **numerically validated and memory-beneficial**, but **do not promote it as a general-purpose latency optimization** at L=65166 until the long-sequence main-kernel regression is addressed. Keep opt-in `H3_SOL_LUT_BITMAP=1`; retain the default int32 LUT as the performance fallback. The separately built Real-H3 same-input A/B notebook remains unsubmitted and is required for final H3-specific validation.

## 6. Reproducible evidence paths

- Submission receipt: `docs/data/h3_sol_bitmap_maskless_ab_20261031/gpu_submission.json`
- Exact A/B result: `docs/data/h3_sol_bitmap_maskless_ab_20261031/gpu_outputs_v1/sol_v2_microgate/bitmap_ab_result.json`
- Baseline and bitmap detail: `.../sol_v2_microgate/bitmap_baseline/result.json`, `.../bitmap_bitmap/result.json`
- CUDA 12.8/SM75 runner: `.../sol_v2_microgate/runner.json`
- Default ABI12 14/14: `.../sol_v2_microgate/result.json`
- Download hash manifest: `docs/data/h3_sol_bitmap_maskless_ab_20261031/gpu_outputs_v1/download_manifest.json`

**Final verdict: SEMANTICS PASS; MEMORY PASS (−10.77% at 65k); LONG-SEQUENCE SPEEDUP FAIL (complete call +0.41%, execute +4.50%).**
