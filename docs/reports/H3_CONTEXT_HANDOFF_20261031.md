# MiniMax-H3 Dual-T4 Infrastructure: Historical Context Handoff

> **Snapshot:** 2026-10-31, Windows workspace `C:\Users\Administrator\Documents\project\2xT4 minimax h3`, branch `main`. This record describes **the 2026-10-31 state** and must not supersede the newer production map in `AGENTS.md` (2026-11-02).
>
> **Purpose:** preserve source- and version-bound facts about MiniMax-H3 Ref2VA, Kaggle 2xTesla T4, Kitchen/Sol attention, E2E generation and prior code cleanup. Distinguish verified results, submitted jobs without completed evidence, unverified designs, retired routes and open tasks.
>
> **Kaggle rule:** never automatically poll job status, read live logs, resubmit or retry a GPU job. An accepted notebook is not a completed GPU execution. A user-initiated failure or result collection authorizes only the requested version-bound terminal-evidence inspection. Attention Rel-L2 is not a proxy for subjective video quality.

## 1. Goal and constraints

- Repository: `chinokikiss/minimax-h3-2xt4`; objective: run MiniMax-H3 Ref2VA on **Kaggle 2xTesla T4**, SM75, 16 GiB per GPU, no NVLink, with lower generation latency and VRAM while retaining reference-image identity, motion continuity and numerical control.
- Pipeline: reference image/prompt -> TE with vision -> TP2 DiT joint audio/video denoising -> Video VAE / Audio VAE -> export. Long-sequence attention has **N=65,166, head dimension 128** and potentially 28 heads per rank. Smaller real-QKV experiments typically use one or four heads.
- Existing infrastructure includes sliding model-weight windows, lookahead, CPU/GPU tiered residency and admission budgets. Do not claim host stalls or low GPU utilization without actual wait measurements.
- Core TP2 weights are INT8 with ConvRot. Prefer offline LoRA baking into independent TP2 shards/bundles and account for baking **outside** E2E wall time. Pin training base, inputs, LoRA and experiment source identity as appropriate to the workflow. Later inference policy moved to metadata-only checks; see `AGENTS.md`.
- Compare identical inputs and frozen source bindings, with one controlled variable. Do not compare RTX SM120 and T4 SM75 milliseconds as though they benchmark the same device. Do not upgrade Torch, CUDA, Kitchen, TileLang or the pinned ComfyUI checkout opportunistically.

## 2. Workspace and environment at snapshot date

| Item | 2026-10-31 fact |
| --- | --- |
| Windows repository | `C:\Users\Administrator\Documents\project\2xT4 minimax h3` |
| Local runtime | `C:\ProgramData\miniconda3\envs\comfyui\python.exe` |
| Local GPU | RTX 5060 Ti 16 GiB, SM120 |
| Local packages | PyTorch 2.12.1+cu132; comfy-kitchen 0.2.37; TileLang 0.1.13 |
| Verified Kaggle environment | Physical T4, PyTorch 2.11.0+cu128, CUDA/nvcc 12.8, Kitchen 0.2.37, TileLang 0.1.13, Sol native ABI 12 |
| Kaggle CLI at snapshot | 2.2.4; do not upgrade merely because a newer version is advertised |
| Git | `main` with many pre-existing modifications and untracked files; do not reset, clean, arbitrarily commit or push |

Use the CatDesk connector to operate on the actual Windows workspace. A separate sandbox cannot stand in for the user's local repository. Local command-job polling is permitted for tests; it is **not** permission to poll Kaggle.

## 3. Runtime source map at snapshot date

| Module | Responsibility |
| --- | --- |
| `custom_nodes/minimax_h3_tp2/__init__.py` | ComfyUI Loader, policy inputs, service caching |
| `patches/comfy_tp_runtime.py` | Persistent dual-rank workers, weights/VAE lifecycle, AOT, per-NFE dispatch |
| `patches/h3_operator_policy.py` | TE/DiT operator choices, Sol-v2 routing, prepared attention, row/column dispatcher |
| `patches/attention_schedule.py` | NFE 1-based attention schedule and `sol@tau` parsing |
| `patches/sol_attention.py`, `sol_sm75_backend.py` | SM75 Sol-v2 All-INT8, native ABI, QKV preparation and routing |
| `patches/sol_attention_native/` | CUDA/C++ Sol kernels, including `sla_sm75_sparse.cu`, `sla_sm75_fp16_kernel.cuh`, `sol_sm75_preprocess.cuh` |
| `patches/kitchen_attention.py`, `kitchen_te_attention.py` | Kitchen DiT dense INT8 and TE masked causal attention |
| `patches/windowed_tp.py`, `tp2_bundle.py`, `tp2_disk_bake.py` | TP2 weight windows, rank bundles and offline baking |
| `patches/h3_operator_aot.py`, `h3_tensor_flow.py` | Precompile and FP16 residual / FP32 numeric islands |
| `patches/parallel_vae.py`, `staged_vae.py`, `video_decode_planner.py` | Video VAE temporal or query partitioning and fallbacks |
| `patches/ptd/{bank,runtime,nodes}.py` | Opt-in PTD / RefAsymOPD LoRA-bank workflow |
| `benchmarks/ref2va_e2e_run.py`, `run_h3_l1_turbo_e2e.py` | Full E2E evaluation and per-NFE instrumentation |
| `docs/guides/{SOL_ATTENTION,H3_OPERATOR_INTEGRATION}.md` | Operator/production design references |
| `docs/data/h3_sol_v2_active_source_identity.json` | Five current-at-snapshot Sol source SHA256 identities; not an archive rollback manifest |

Backend distinctions at this snapshot:

- TE defaults to `kitchen_causal`; `native` is selectable. Full causal and additive/padding masks matter.
- DiT defaults to Kitchen INT8; `sol` on physical SM75 and `native` PyTorch SDPA are selectable. `dit_attention_schedule` can switch by NFE, e.g. `1-3:kitchen,4-8:sol@1.0`. The DiT token refiner retains Native SDPA and must not be mistaken for the joint DiT attention.
- Video VAE accepts **only `kitchen` or `native`** after retirement of its Sol branch. Default Kitchen; native temporal pairs are a separate partitioning decision.
- Audio VAE is native, single GPU. Video VAE and Audio VAE do not inherit the DiT Sol choice.
- Sol-v2 boundary example: `[text 0:2528, ref1 2528:3344, ref2 3344:4164, audio 4164:4974, video 4974:65166]`. Old `protected_prefix()` returns **4164** (conditioning only); new `sol_v2_routing()` returns **4974**, protecting conditioning **and** generated audio. Video topology starts at 4974 with 836 tokens/frame, temporal radius 1, local radius 64 and frame-boundary protection. Native Q64/K64 scheduling rounds protection outward. **These functions are not interchangeable.**
- Sol native ABI12 supports D128 and D64, prepared attention, INT8 Q/K/V, K-centroid routing and sync-free LUT. D64 support **does not mean Video VAE Sol is enabled**.
- Optional but not default: `H3_SOL_WORKSPACE_POOL=1` and `H3_SOL_LUT_BITMAP=1`; each needs independent memory and speed qualification.
- Retired: `H3_SOL_FUSED_KC_THRESHOLD` / `compute_kc_stats_threshold_fused`, and attempted math-parity changes. Current-at-snapshot KC statistics remain `reduce_kc_stats` plus `compute_diag_threshold`. Frozen old notebooks may still contain retired code; do not revive it by copying.
- Other retained features include lookahead, offline LoRA, TileLang `tilelang_scaled_fp16` row transport, `cuda_int8_tilelang` policy selection, `tilelang_fp32` RMS/RoPE, and `fused_stored` / optional `fused_recompute` SwiGLU. Confirm **actual runtime backend dispatch** with current `AGENTS.md`, which supersedes this snapshot's broad labels.

## 4. Standard reproducible 0.85 MP / 10-second input

| Field | Frozen comparison |
| --- | --- |
| Model | MiniMax-H3 Ref2VA TP2 INT8 base, `cnmds/minimax-h3-tp2-int8-561a6211` |
| Resolution | 1216 x 704 = 856,064 pixels = 0.856064 MP |
| Frames and export | 243 native decoded frames; 240 frames x 24 FPS = 10.0 s export |
| Sampling | 8 NFE, Euler / simple, CFG 1.0 |
| Shifts | Video 6.0, audio 3.0 |
| References/prompt | Fixed anime shrine scene, purple-haired shrine maiden, white fox, two reference images; read exact request for contents and hashes |
| LoRA | `lightx2v/Minimax-h3-Turbo` / `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` |
| LoRA commit / strength | `30f485ed874aa44c8183348ad361c5eabad318da` / 1.0, 208 targets, offline baked |
| Earlier seed | 20261020 |
| Latest 2026-10-31 seed | 987654321; all other external inputs unchanged |
| Packed layout | 65,166 total tokens; 60,192 video tokens, 72 latent frames x 836 tokens |

Authoritative request: `docs/data/h3_turbo_085mp_10s_20261029/gpu_v2_final/turbo_085mp_e2e/retry_request.json` and matching `generation/result.json`; frozen source environment in `docs/data/h3_turbo_085mp_10s_20261029/provider_v2_source/`. Do not reconstruct exact prompts or reference hashes from this handoff.

## 5. Verified historical E2E timings

These jobs were separate historical runs, **not** single-variable kernel performance experiments. In particular, old 2026-10-30 Sol runs still used Video VAE Sol, which was subsequently retired.

| 2026-10-30 job | DiT NFE | Video VAE | Sampler | Foreground | E2E worker excluding bake |
| --- | --- | --- | ---: | ---: | ---: |
| All Kitchen baseline | All Kitchen | Kitchen | 1348.776 s | 1639.157 s | 1703.296 s |
| Old Sol hybrid v4 | Kitchen 1,8; Sol 2-7 | Sol | 1191.284 s | 1497.299 s | 1549.299 s |
| Sol-v2 Production E2E v1 | Kitchen 1,8; Sol 2-7 | Sol | 1191.718 s | 1458.364 s | 1506.675 s |

The Sol-v2 Production sampler was 157.058 s (11.64%) faster than the all-Kitchen baseline; foreground difference 180.793 s (11.03%). Other TE/I/O/decode job differences cannot be attributed to DiT Sol. Observed per-rank dispatch: 100 Kitchen + 300 Sol, 8/8 denoise steps completed, 2016 historical Video VAE Sol attention calls/rank, decoded pixels `[1,243,704,1216,3]`, audio `[1,324000,2]`, 10-second export. Offline bake was 328.539 s separately.

Sources: `docs/reports/H3_SOL_V2_PRODUCTION_E2E_20261030_RESULTS.md`, `docs/reports/H3_SOL_HYBRID_085MP_10S_T4_20261030_RESULTS.md`. Different latent hashes do not determine which video looks better; visual/temporal/identity reviews are open.

## 6. Submitted 2026-10-31 GPU jobs without retrieved terminal proof

The following states were **provider accepted** as of the snapshot, **not verified GPU complete**. No automatic status checks, artifact downloads or resubmissions:

| Job | Version | Seed | DiT schedule | Video VAE | Proven at snapshot |
| --- | ---: | ---: | --- | --- | --- |
| [Kitchen 1-3 / Sol 4-8 hybrid](https://www.kaggle.com/pamcks/h3-kc1-3-sol4-8-vae-kc-long-e2e-20261031) | v4 | 20261020 | Kitchen 1-3; Sol 4-8, tau=1 | Kitchen | Accepted; older `id_no` may have reused notebook identity |
| [All Kitchen control](https://www.kaggle.com/code/pamcks/h3-kc-all-e2e-20261031) | v1 | 20261020 | All Kitchen | Kitchen | Accepted, independent notebook |
| [Hybrid, new seed](https://www.kaggle.com/code/pamcks/h3-kc-sol-seed987654321-e2e) | v1 | 987654321 | Kitchen 1-3; Sol 4-8, tau=1 | Kitchen | Accepted, independent notebook; priority for user-triggered review |

New-seed submission UTC 2026-10-31T06:18:49Z, 2xT4 requested, notebook SHA256 `a9cff3db6bc213b349aa2bfe7d657a078d6011f8390f0437a25c0eba2d8ca083`. Local pre-submission gates: 35/35 passes. Receipt: `status=accepted`, `version=1`, `final_collection_count=0`. Its frozen notebook changed the seed allowlist to admit 987654321; source/prompt/other inputs were held fixed. Large numerical seed separation does **not** predict large visual separation. Expected but unverified: 150 Kitchen + 250 Sol DiT calls/rank and Kitchen Video VAE.

Relevant files: `docs/reports/H3_KC_SOL_SEED987654321_LONG_E2E_SUBMISSION_20261031.md`; `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/{request.json,gpu_submission.json,preflight.log,gpu_job/experiment.ipynb,gpu_job/source_binding.json}`; builders/tests in `benchmarks/build_h3_kc_sol_seed987654321_long_e2e.py`, `test_h3_kc_sol_seed987654321_job.py`. Other frozen jobs live under `docs/data/h3_kc_sol_8step_long_e2e_20261031` and `h3_kc_all_8step_long_e2e_20261031`.

## 7. Real H3 QKV: T4 Sol-v2 vs ComfyUI low-level Sol

[Real H3 Sol Comfy Capture v1](https://www.kaggle.com/code/pamcks/h3-real-sol-comfy-capture-20261031) was marked failed by the provider **after** both T4 ranks had saved first-layer, post-RMS/post-RoPE real Q/K/V and Sol outputs. The final aggregator incorrectly expected `rank['summary']` and raised `KeyError: 'summary'`; do not classify this as a GPU kernel failure or silently change provider status.

Per rank, captured head0 Q/K/V and T4 output were FP16 `[1,1,65166,128]`. Rank files `rank0_real_qkv.pt` / `rank1_real_qkv.pt` were each 66,732,515 bytes; data and tensor hashes validated. Local SM120 replays used Kitchen low-level `sol_attn` for eight settings total. Since Comfy Sol requires SM80+ while the native T4 Sol is SM75, the replay is **not same-GPU kernel timing**.

| Comfy low-level option: output vs T4 Sol-v2, overall Rel-L2 | Rank0 | Rank1 |
| --- | ---: | ---: |
| tau=1.0, matching protected prefix, token_aug=0 | 1.3055% | 2.6213% |
| tau=1.0, Comfy H3 sink_q, extra=0 | 4.2120% | 3.4228% |
| tau=1.0, Comfy H3 sink_q, extra=256 | 3.4064% | 2.9910% |
| tau=1.3, Comfy H3 sink_q, extra=256 | 5.2645% | 4.7708% |

Against a sampled FP32 dense reference with **85 real queries** and full K/V:

| Output vs sampled FP32 dense, Rel-L2 | Rank0 | Rank1 |
| --- | ---: | ---: |
| T4 Sol-v2, tau=1 | 5.3483% | 2.8443% |
| Comfy Sol, matched sink, tau=1, extra=0 | 6.0101% | 4.0387% |
| Comfy Sol, H3 sink, tau=1, extra=256 | 7.2490% | 8.8262% |
| Comfy Sol, H3 sink, tau=1.3, extra=256 | 9.2438% | 9.8780% |

For these two first-layer heads and the non-exhaustive 85 sampled queries, old T4 Sol-v2 was closer to sampled FP32 dense than the tested low-level Comfy settings. **This is not video quality proof** and does not cover `sol_attn_chunked` projection partitioning or inter-stride kmean/vscale reuse. Evidence: `docs/reports/H3_REAL_SOL_COMFY_V1_FAILURE_RECOVERY_RESULTS_20261031.md`, `docs/data/h3_real_sol_comfy_capture_20261031/{recovered_v1_result.json,local_comfy_replay/result.json}` and `failed_v1_payloads/.../generation/rank_results/`.

## 8. Other Sol experiments and retired ideas

### Optional Bitmap LUT

Physical T4 CUDA12.8/ABI12 D64 and synthetic-shaped 65,166-token D128 routes had identical effective ExactMask, RowCount and LUT sequences with output Rel-L2=0. D128 temporary GPU peak fell **189,895,168 -> 169,445,376 bytes** (about 19.50 MiB, -10.77%), but full-call time rose **139.4679 -> 140.0346 ms** (+0.41% slower), execute phase +4.50% slower. This was not real two-rank E2E; default remains int32 LUT, `H3_SOL_LUT_BITMAP=1` opt-in. Details: `docs/reports/H3_SOL_BITMAP_LUT_AB_20261031.md`.

### Retired partitioned KC threshold fusion

`compute_kc_stats_threshold_fused` preserved routes in T4 tests but showed no dependable D128 speed benefit (approximately -0.011%, +0.258%, +0.024% full-call differences; D64 proxy had a +7.30% smaller-case gain). Native macro, dispatch and cache-key code were removed by user direction. Keep regular `reduce_kc_stats` + `compute_diag_threshold`; do not recreate this branch.

### Retired math-parity patch

Attempted numerical changes did not reliably reduce real-H3 Sol-vs-Kitchen Rel-L2 (around 4.5%) across both ranks and sometimes regressed rank1 performance. The experiment/rollback report and patch backups were removed by user direction. At snapshot, five active Sol files were bound by `docs/data/h3_sol_v2_active_source_identity.json`. Historical notebooks may still carry retired code.

## 9. Verified 2026-10-31 cleanup

1. Removed Video VAE Sol from Loader UI, RuntimeConfig, `parallel_vae`, AOT and E2E assertions; retained Kitchen default and Native choice.
2. Removed KC threshold fusion CUDA kernels, preprocessor macros, Python toggles and compile-cache branches; kept ordinary statistics/threshold kernels.
3. Kept standalone Bitmap experiment `benchmarks/sol_bitmap_route_probe.py` and optional workspace-pool testing.
4. Removed obsolete math-parity / rollback and KC-fusion reports, backups, builders and experimental data; updated docs and affected builder backend choices.
5. Local regressions: first **44/44**, then **50/50** pass, including Sol/VAE/config/E2E source closure; Python compilation passed. **No new full physical-T4 CUDA12.8 E2E was performed after this cleanup.**
6. Git worktree already had substantial mixed changes; **no commit or push** was performed for that cleanup. Do not attribute every historical diff to one task.

Frozen submitted notebooks are not rewritten to follow current source edits; a rebuilt experiment requires a new source identity and test receipt.

## 10. Prioritized open work (as of 2026-10-31)

- **P0, on explicit user request only:** collect the terminal version-1 evidence for `pamcks/h3-kc-sol-seed987654321-e2e`; validate status, GPU identities, rank count, 8/8 steps, expected 150/250 dispatches/rank, Kitchen Video VAE, decode/export, wall times, and offline bake separately. For failures, examine terminal logs before considering any new submission. Use the version-bound download helper in `benchmarks/download_kaggle_evidence.py` and `kaggle_tls12_runner.py` only after checking its actual flags. Intended destination `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/gpu_final_v1/`. Do not assume intended `id_no` equals actual notebook URL.
- **P1:** compare all-Kitchen vs Kitchen 1-3 / Sol 4-8 at the **same seed 20261020** with Kitchen Video VAE. Then compare the mixed schedule between seeds 20261020 and 987654321. Two seeds cannot estimate a universal failure rate.
- **P2:** inspect actual output media for character/fox identity, shrine composition, umbrella/hands/eyes, motion, temporal discontinuity and audio-video alignment. Neither hash differences nor sampled attention errors replace blinded or fixed-criteria review.
- **P3:** on a separately authorized physical-T4 run, qualify the cleaned Sol ABI12 build and hybrid dispatch; distinguish 50/50 local gates from GPU qualification. Measure cold/warm/resident peaks for workspace pool; revisit Bitmap only with timing evidence; profile weight window, prefetch, TE I/O and PCIe collective costs before diagnosing a bottleneck.
- **P4:** explore opt-in PTD / RefAsymOPD multi-LoRA offline bake (`taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs`). Compatibility with the TP2 base's **pruned AdaLN curves** remained unresolved. Independent TP2 datasets/bundles are preferred to repeated online bake, but require correctness and E2E validation.

## 11. Handoff rules

1. Read the current request and the latest `AGENTS.md` before acting; this 2026-10-31 handoff is historical.
2. Never poll, silently retry or auto-resubmit Kaggle. An explicit user request for completed results permits a bounded version-specific collection.
3. Label evidence correctly: submission != successful GPU run; CPU tests != T4 kernel compilation; one attention head != E2E video quality; distinct output SHA != visual collapse; wall time across jobs != isolated kernel speedup.
4. Never revive retired Video VAE Sol, partitioned KC threshold fusion or failed math-parity backups. Keep valid DiT Sol-v2, normal KC routing and opt-in Bitmap/Pool unless separately directed.
5. Keep provider URL, notebook version, input request, source binding, hashes and measured results consistent; preserve frozen notebooks unchanged.
6. Inspect git status and exact targets; never reset/clean unrelated work, and do not commit/push without authorization.
7. Reports should prioritize **measured E2E outputs, exact phase timings and evidence boundaries**, distinguishing completed work from proposals. Write repository files in English.

---

**Historical continuation note:** As of 2026-10-31 the priority was user-triggered terminal collection for seed 987654321 using Kitchen NFE1-3, Sol-v2 NFE4-8 and Kitchen Video VAE, at 0.85 MP / 10 s. The post-cleanup full dual-T4 E2E remained unverified. **For the live 2026-11-02 production P0-2/P0-3 defaults and concrete dispatch audit, refer to the top of `AGENTS.md`.**
