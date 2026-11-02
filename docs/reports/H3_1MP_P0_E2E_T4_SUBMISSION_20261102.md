# H3 1MP FP16/512 Sol-v2 8-NFE Long E2E — Prepared, Submit Blocked (2026-11-02)

## Outcome

**Local source-bound preparation succeeded; the single Kaggle submission attempt FAILED during the HTTPS SaveKernel request.** The CLI exited 1 with `SSLEOFError: UNEXPECTED_EOF_WHILE_READING`. No provider acceptance/version receipt was returned. Server-side acceptance cannot be inferred from a failed transport; **GPU execution has not been verified**. No status/list/live-log polling, automatic retry or final-artifact collection was attempted.

The first failed CLI attempt also warned that the human-readable notebook title did not normalize to the requested kernel id. The local `kernel-metadata.json` and builder were corrected to a matching title *after* the failed attempt; this correction has not been submitted. Any subsequent push must be explicitly authorized, since the remote SaveKernel operation may have been processed despite the lost TLS response.

## Frozen input and intended physical run

- **Private two Tesla T4 / SM75 GPUs**, no dataset/kernel mounts, direct pinned source/core setup reused from the successful earlier 0.85MP v2 E2E.
- **1344 x 768 = 1,032,192 pixels (1.032192 MP)**, divisible by 32 in both axes; 243 native frames at 24 FPS (10.125-second internal duration), 240 frames exported (10 seconds).
- Estimated generated-video tokens: `72 * (768/32) * (1344/32) = 72,576`; exact full DiT packed length will only be known if the GPU runtime executes.
- MiniMax-H3 Ref2VA TP2 INT8 base; pinned `lightx2v/Minimax-h3-Turbo` regular 8-step adapter at strength 1, baked offline with no online LoRA.
- Original frozen two reference images/prompt and seed `20261020`, Euler/Simple **8 NFE**, CFG 1, Video Shift 6, Audio Shift 3.
- FP16 residual stream **exponent 9 (divisor 512)**, Debug False; TE Kitchen causal; **DiT Sol-v2 all NFE 1–8, tau 1.0**; Video VAE Kitchen; Audio VAE Native.
- Actual current-source overlay includes P0-A1, P0-2, P0-3, P0-B, **P0-C on all positive lengths**, and the current Sol condition-key-only source. No legacy notebook backend substitution or benchmark P0-B monkey patch.
- Baseline: `docs/data/h3_t4_fp16_512_sol_all_8nfe_10s_20261101/gpu_report_v2.md` (1216x704, completed). This new run is not single-variable compared with that frozen baseline: it changes canvas, source/kernel defaults and Sol condition-protection semantics.

## Source and local validation

- Builder: `benchmarks/build_h3_t4_fp16_512_sol_all_8nfe_1mp_10s.py`.
- Source-bound notebook, metadata, binding and receipt: `docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102/gpu_job/`.
- Pinned source closure: **95 files**, local SHA256 checks passed including P0-B production in-place kernel, P0-C all-length source, Sol SM75 backend and new Loader.
- Notebook size: **499,160 bytes** (below 900,000-byte envelope), private, GPU enabled, two T4 requested, zero mounts. Immutable public shrine image mirror is staged and SHA256-checked before full E2E.
- Local `conda comfyui`: **44/44** Python preflight/integration/source-contract tests passed.
- Submission attempt record: `docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102/gpu_push_attempt1.json`.
- No full-GPU performance, memory, precision, output-frame, or semantic-quality result exists for this 1MP job.

## Next gate

Do not automatically resubmit or status-poll the Kaggle notebook. Request explicit user direction for one controlled retry of the same private kernel identity after the TLS failure; if a previously accepted version surfaces independently, collect its final artifacts once without treating a new submission as necessary.
