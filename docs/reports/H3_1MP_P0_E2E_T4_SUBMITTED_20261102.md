# H3 1MP FP16/512 Sol-v2 8-NFE Long E2E — Kaggle v1 Submitted (2026-11-02)

## Submission verdict

**ACCEPTED by Kaggle, not yet GPU-execution-qualified.** After the initial `SaveKernel` TLS EOF failure, the user explicitly authorized one retry. The corrected private kernel title/ID was submitted exactly once more; Kaggle CLI exit code 0 returned `Kernel version 1 successfully pushed.`

Provider URL: https://www.kaggle.com/code/pamcks/h3-1mp-fp16-512-sol-v2-8nfe-10s-p0-20261102

The first failed attempt is separately archived in `gpu_push_attempt1.json`; the successful retry in `gpu_push_attempt2.json`; the authoritative acceptance record is `gpu_submission.json`. No Kaggle status/list/live-log polling, execution inference, final-output collection, or further retry was performed. **An accepted notebook is not proof of a completed dual-T4 E2E.**

## Frozen workload

- 2 x Tesla T4 requested; private notebook, no dataset or kernel-output mounts.
- 1344 x 768 = **1,032,192 pixels (1.032192 MP)**; 243 native frames at 24 FPS, 240 exported frames / 10 seconds, fixed original prompt and two reference images.
- MiniMax-H3 Ref2VA TP2 INT8; pinned 8-step regular Turbo LoRA (strength 1) offline baked, no online LoRA.
- Seed **20261020**, 8 Euler/Simple NFE, CFG 1.0, Video Shift 6.0, Audio Shift 3.0.
- FP16 residual exponent **9** (division by 512), Debug False, TE Kitchen causal INT8, DiT **Sol-v2 all 8 NFE at tau=1.0**, Video VAE Kitchen, Audio VAE Native.
- New production P0-A1/P0-2/P0-3/P0-B/P0-C default-on paths and current Sol key-only condition protection are included in the 95-file SHA256-verified source overlay.
- Expected generated-video tokens **72,576**, actual DiT packed length to be collected from a successful GPU execution, not assumed.
- The immutable reference-image mirror and SHA256 checks were incorporated from the previously completed v2 0.85MP E2E.

## Verification and limits

- Notebook 499,160 bytes (below 900,000-byte envelope); 95 current-source files included with exact hashes.
- Local `C:\ProgramData\miniconda3\envs\comfyui\python.exe` 44/44 integration/preflight tests passed; the explicitly repeated delivery gate passed 5/5 before the successful push.
- `docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102/gpu_job/` contains the submitted source-bound notebook, private metadata, and source binding.
- It is not legitimate to report OOM status, runtime, peak GiB, finite output, decoded media, or end-to-end quality before final provider artifacts are actually collected.
- The older **1216 x 704 (0.856MP)** frozen E2E completed successfully but differs in both shape and the later promoted production kernels; it is not an isolated P0 ablation baseline.

## Follow-up boundary

Wait for user-provided completion signal. Then perform one version-bound final artifact collection and review physical device identity, successful completion, 8/8 NFE, actual attention dispatch, allocated peak per rank, warmed/total time, nonfinite checks and exported output. Do not poll Kaggle status or live logs automatically and do not resubmit without new authorization.
