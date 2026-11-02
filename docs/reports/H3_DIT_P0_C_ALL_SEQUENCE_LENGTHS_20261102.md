# H3 DiT P0-C — All Positive Sequence Lengths Default (2026-11-02)

## Production decision

The user explicitly accepts the observed small non-bitwise P0-C differences and requests removal of the sequence-length whitelist. **Implemented** in the live DiT O-projection P0-C dispatch, which remains enabled by default via `dit_p0_c_local_rotation=True`.

In `patches/tilelang_row_quant.py::_local_o_proj_p0c`, the former `S in (128,6814,59134,65166)` restriction was replaced by **`S>0`**. The docstring no longer promises universal byte-identical output. The kernel, row-global MAX collective, TileLang quantization and scaled-FP16 row reduction are otherwise unchanged.

The production path is still:

`ComfyUI Loader -> H3OperatorPolicy -> DiT out_proj -> tilelang_dit_fusion.row_gated -> tilelang_transport_p2.row_linear -> tilelang_row_quant.global_row_quantize -> _local_o_proj_p0c`.

P0-C is attempted only when `tilelang_transport_p2.row_linear` selects the existing `optimized_tp.use_fallback_rotation` packed branch. Removing the P0-C shape list does not force this branch over a qualifying Native fused ConvRot backend.

All other eligibility gates remain: actual SM75/T4, two ranks, ConvRot256, packed branch, owned contiguous exact `torch.Tensor` of FP32 `[S,3584]`, supported weights, no online LoRA, and no gradient/compiler tracing. Invalid or unsupported conditions retain the original P0-2 padded path. The existing Loader flag allows manual opt-out.

## Precision evidence and accepted uncertainty

The earlier **actual physical dual-T4** study found zero raw-bit differences in P0-C rotated FP32 values, row maxima/scales, quantized codes and selected final trained-Block outputs at studied `S=128,6814,59134,65166`.

The later **local RTX 5060 Ti / SM120** comparison of full padded and local-width Kitchen FP32 rotations studied `S=1,127,129,256,1024,4096,6815,8192,32768,59135`. All studied `S>=256` were raw-bit exact in this local proxy. `S=127,129` had at most `3.338e-6` FP32 rotation absolute difference, `2.049e-8` global-scale absolute difference and `11/6` combined-rank INT8 *proxy* code mismatches, each of magnitude at most one code unit; `S=1` had differing rotation bytes but no code mismatch. The earlier local `S=128` also differed at the proxy-code level.

This evidence supports the user's relaxed acceptance of local operator numerical differences. **It does not demonstrate every positive S on physical T4, a bound on trained-block output for untested S, or a full video/audio quality-equivalence guarantee.** The local proxy used PyTorch rounding, not the physical SM75 TileLang quantizer.

Evidence: `docs/data/h3_dit_p0c_shape_local_20261102.json`, `docs/reports/H3_DIT_P0_A1_P0_C_T4_20261102.md`, and archived `docs/reports/H3_DIT_P0_B_PRODUCTION_P0_C_SHAPE_AUDIT_20261102.md`. The earlier report's whitelist-preservation decision is superseded by this explicit user instruction.

## Last completed physical long-sequence H3 E2E configuration

The latest **completed** available physical two-T4 long-sequence run was the frozen 2026-11-01 **FP16/512 Sol-v2 all-8-NFE 10-second** v2 notebook, not the older 2026-10-30 divisor-64 all-Kitchen baseline.

- Immutable receipt: `docs/data/h3_t4_fp16_512_sol_all_8nfe_10s_20261101/gpu_report_v2.md`, `gpu_final_v2/h3_t4_fp16_512_sol_v2_all_8nfe_10s/batch_results.json`; Kaggle: https://www.kaggle.com/code/pamcks/h3-t4-fp16-512-sol-v2-8nfe-10s-debug-off.
- Hardware: 2 x Tesla T4 SM75; Torch 2.11.0+cu128, CUDA 12.8, comfy-kitchen 0.2.37, TileLang 0.1.13.
- Base: `cnmds/minimax-h3-tp2-int8-561a6211`; regular `lightx2v/Minimax-h3-Turbo` 8-step LoRA v1.0 at strength 1.0, 208 targets baked offline; no online LoRA.
- Canvas: 1216 x 704 = 856,064 pixels; 243 native frames at 24 FPS (10.125-second model duration); export 240 frames/10 seconds; actual packed DiT sequence length 65,166.
- Sampler: 8 steps/NFE, Euler/Simple, CFG 1.0, video shift 6, audio shift 3, seed 20261020.
- Inputs: fixed anime shrine maiden, white fox and shrine prompt, two pinned character/environment reference images, empty negative prompt; see immutable request for full prompt and hashes.
- Precision: FP16 residual flow, exponent 9 (1/512), with FP32 numerical islands; Vision and Video VAE gauge exponent 0; Debug False.
- Attention: TE Kitchen causal; **DiT Sol-v2 NFE 1–8, tau=1.0**, bidirectional condition protection in the frozen job; Video VAE Kitchen; Audio VAE Native; token refiner Native.
- Runtime: 8/8 steps completed, 400 DiT Sol attention calls per rank, 48/48 persisted finite stage checks, full E2E command 1,490.008 seconds including sampler 1,181.188 seconds; separate offline bake 337.235 seconds.
- Important difference from present production: the frozen run predates later key-only Sol condition protection and P0-A1/P0-B/P0-C lifetime and kernel promotions. **It does not measure the current integrated all-length P0-C runtime.**
- The 4-NFE / 15-second Sol-v2 job under `docs/data/h3_t4_fp16_512_sol_all_4nfe_15s_20261101` was only provider-accepted in the available receipt, with no collected final execution proof; do not mislabel it as a more recent completed E2E.

## Local verification of the promotion

Local Windows environment: `C:\ProgramData\miniconda3\envs\comfyui\python.exe`.

- 39 Python integration/source-contract tests passed, including the explicit no-whitelist / positive-row / unchanged-ownership-and-SM75 assertions.
- `node benchmarks/test_loader_migration_p0.mjs` passed.
- `git diff --check` passed (only pre-existing Windows CRLF notices).
- No physical T4 run or Kaggle submission was performed for this new whitelist removal. No Git commit or push was requested.
