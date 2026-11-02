# H3 DiT P0-B — Physical Dual-T4 Submission (2026-11-02)

## State: submitted; final results pending

Kaggle accepted exactly one private physical dual-T4 notebook, **version 1**:

- Actual URL: https://www.kaggle.com/code/pamcks/h3-dit-p0-b-t4-20261102
- Requested metadata ID: `pamcks/h3-dit-p0b-t4-20261102` (the title-derived URL slug differs).
- Notebook SHA256: `cfcde131ee002b20c1be8f15ecfcf9d157829796d999a17e683be73974d4e95b`
- Frozen source SHA256: `af147d56c50f2c48dee013a9b2e9325a95c6bf80cf4672158769a98219356aa6`
- Source-binding manifest: **106 files**, including actual production dispatch, the experimental P0-B CUDA kernels and fail-closed T4 guards.
- Local Windows conda `comfyui` CPU preflight: **21 tests passed**; experimental kernels also passed local offline SM75 compilation. Earlier local SM120 precision/memory/performance results remain in `H3_DIT_P0_AC_PROMOTION_P0_B_LOCAL_20261102.md`.
- CPU Kaggle notebook was generated but **not** submitted. Reused pinned source/environment bootstrap and local CPU-first validation, consistent with the previous P0-A1/C dual-T4 qualification.
- Kaggle CLI reported `Kernel version 1 successfully pushed.` with exit status 0. Submission accepted is **not** execution success.

## Frozen two-rank experimental contract

- Real dual Tesla T4 (SM75) required, Torch / Kitchen 0.2.37 / TileLang 0.1.13 identity checked by the frozen setup.
- Two real NCCL ranks with a Gloo barrier group, public trained DiT block-0 rank shards and per-rank SHA256 state checks.
- Same FP16 residual divisor **512** in both A/B variants; P0-A1, P0-C, P0-2 and P0-3 **default on** for both.
- P0-B applies **only** to exact owned contiguous `[S,14336]` FP16 `fused_stored` DiT FC2 preparation; the baseline runs the unchanged production FC2. No production P0-B switch, module or default was changed.
- At S=128, 6,814, 59,134 and 65,166: compare real two-rank global-MAX INT8 codes and FP32 scales against the production baseline using raw-bit comparisons.
- At S=59,134 (Kitchen attention) and S=65,166 (Sol-v2): compare trained-block final output raw bits; interleave **9 warmed CUDA-event measurements** per arm after 3 warmups; record FC1/O-Proj/FC2 phase peaks and separately normalized whole-block allocated-peak samples with their initial live allocation state.
- Bounded watchdog: 2,600 seconds for GPU notebook; bounded two-rank worker process. Frozen source/archive provenance and final per-rank receipts retained.
- This is trained **single-block** validation, not 25-shard full-H3 E2E certification. Any zero-regression or full-H3 claim must be qualified accordingly.

## Next action

Wait until the user reports Kaggle completion, then perform a **single logical final-output collection** and verify:

1. Actual dual-T4 hardware, pinned environment, signed source manifests, trained shard hashes and successful worker exit.
2. Both ranks' raw-bit quantization and trained-block output comparisons.
3. Per-phase and normalized whole-block Allocated Peak at both long lengths; reject contaminated comparisons whose initial allocations differ materially.
4. Interleaved warmed timing distributions, rather than only medians.
5. Dispatch counters proving the P0-B kernel executed and P0-A1/P0-C/P0-2/P0-3 stayed enabled in both arms.

Do **not** infer success from the Kaggle submission receipt, poll status/list/live logs, auto-retry or promote P0-B into production before final results are audited.

Submission receipt: `docs/data/h3_dit_p0b_t4_20261102/gpu_submission.json`.
