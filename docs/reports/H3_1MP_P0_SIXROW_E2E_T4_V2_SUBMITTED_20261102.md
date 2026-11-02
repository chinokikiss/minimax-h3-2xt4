# H3 1MP FP16/512 Sol-v2 E2E — Six-Row AdaLN Repair, Kaggle v2 Submitted (2026-11-02)

## Submission verdict

**ACCEPTED, kernel version 2. Physical GPU execution is not yet verified.**

- Private Kaggle URL: https://www.kaggle.com/code/pamcks/h3-1mp-fp16-512-sol-v2-8nfe-10s-p0-20261102
- The single user-authorized v2 push used `C:\ProgramData\miniconda3\envs\comfyui\python.exe -m kaggle kernels push -p docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102_v2/gpu_job`.
- CLI exit code: `0`. Provider response: `Kernel version 2 successfully pushed.`
- No status/list/live-log polling, no output collection, and no automatic GPU retry were performed.

## Motivation and immutable v1 history

The preceding physical dual-T4 v1 failed at the first DiT Attention O-Proj, where the non-consuming padded ConvRot attempted 2.09 GiB with only 1.93 GiB free. Read `docs/reports/H3_1MP_P0_E2E_T4_FAILURE_20261102.md` and the immutable `docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102/gpu_final_v1/`.

Source tracing localized the fallback to `patches/tilelang_dit_fusion.py`: the first sigma with two image references uses **six** AdaLN modulation rows; the prior fused gate admitted only nine. The local production repair accepts actual `3*N` modulation rows, preserves segment-index/dtype/SM75/ownership/LoRA/grad guards, and forwards the dynamic row count to the existing unchanged TileLang norm/gate/row-gate kernel factories. See `docs/reports/H3_1MP_P0_ROW_GATE_FALLBACK_FIX_20261102.md`.

## Identical E2E workload

The v2 build asserts exact request-object equality with v1:

- Two requested Tesla T4 (SM75), private kernel, no dataset or kernel mounts.
- 1344 x 768 = 1.032192 MP; 243 native frames; 240 exported frames / 10 s.
- Same two anime-reference images, prompt, seed `20261020`, regular Turbo LoRA revision/file/strength 1 with offline TP2 disk baking.
- Eight Euler/Simple NFE; CFG 1; shifts video/audio 6/3; FP16 residual exponent 9 (=1/512).
- DiT Sol-v2 `1-8:sol@1.0`, TE Kitchen causal, Video VAE Kitchen; debug off.
- Default-on P0-A1/P0-2/P0-3/P0-B/P0-C and selected key-only Sol path.
- The only intentionally new production runtime source change from v1 is the corrected dynamic AdaLN row-count admission in `patches/tilelang_dit_fusion.py`.

## Rebuilt source binding and CPU gates

Isolated v2 artifacts: `docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102_v2/`.

- Reproducible rebuild command: `python -m benchmarks.build_h3_1mp_p0_sixrow_retry` under `conda comfyui`.
- 95-file source-closed overlay; notebook 499,172 bytes.
- Overlay SHA256: `c1080b7da9ccebe6c14ee47a3629a17993e2e0566a812698e1165f4c9cf68193`.
- Bound production six-row repair SHA256: `f3d8a5afa2237692d7ee0f5c418df50348b595529f1ded7d7f194a2ac8c64669`.
- 42 local Python regression tests passed; five version-bound delivery/preflight tests passed; Node loader migration test passed. The delivery gate verifies source bytes against all overlay hashes, exact E2E workload, private T4 metadata, source closure, watchdog, and credential absence.
- No original v1 evidence or submitted notebook was overwritten.

## Qualification boundary

The accepted v2 submission establishes **only upload acceptance**. It does not certify the first NFE now takes P0-2/P0-C, actual allocated peak GiB, attention precision, E2E latency, 8/8 completion, or output fidelity. A completion signal from the user is required before one version-bound final output collection and evidence-based physical-GPU report. Do not poll Kaggle job status or live logs.
