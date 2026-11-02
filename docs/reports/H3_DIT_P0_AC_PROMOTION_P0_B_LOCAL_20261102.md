# H3 DiT P0-A1 / P0-C Production Promotion and P0-B Local Qualification — 2026-11-02

## Executive status

**P0-A1 and P0-C are now wired into the real ComfyUI TP2 production dispatch and enabled by default, with independent opt-outs.** These changes were requested after the prior physical dual-T4 experimental report. They do not restore the deleted P0-1 code or alter existing P0-2/P0-3 defaults.

**P0-B has passed local experimental bitwise and phase-memory checks on RTX 5060 Ti (SM120) at four tested row counts, including both long-sequence geometries. It is not in production.** Final physical dual-T4, real TP2 collective, trained-block, and E2E qualification is outstanding; do not claim formal no-regression before those tests.

## Production P0-A1 wiring

Actual route:

1. ComfyUI Loader -> `RuntimeConfig.operators=H3OperatorPolicy(dit_p0_a1_turing=True)`.
2. `windowed_tp.prepare_tp_block` -> `configure_block`, installing `_vram_p0_a1` on **DiT QKV and FC1 only**.
3. `tp_dit.ColumnParallelLinearINT8.forward` passes `p0_a1=True` when the input is a regular Tensor, uncompiled inference, without an online LoRA adapter.
4. `dit_column_linear.column_linear` checks that the **actual Kitchen registry implementation** is `comfy_kitchen.backends.cuda.int8_linear` and that the selected backend remains `cuda_int8_tilelang`.
5. `dit_kitchen_turing_a1.turing_column_a1` reproduces the actual pinned Kitchen 0.2.37 ConvRot quantization branch and calls `_int8_linear_turing_quantized` directly, without first allocating its unused full `[S,N]` FP16 placeholder.
6. Any incompatible version, device, dtype, weight geometry, backend, shape, grad/compile state, or Turing preference returns to the original `ck.int8_linear`. No FC1 input is detached or mutated.

Earlier physical dual-T4 experiment (frozen source with exact source-patch injection): both ranks at S=59,134/65,166, trained-block final output **0 bit mismatches**, **22/22** successful Turing GEMM calls per rank/shape, and QKV/FC1 phase-local peak savings of **1.184/1.305 GiB** and **1.579/1.740 GiB** respectively. The current production wrapper is a separately implemented, source-equivalent direct branch; that integration has **not** been re-run on T4.

## Production P0-C wiring

Actual route: DiT out_proj -> `tilelang_dit_fusion.row_gated` -> `tilelang_transport_p2.row_linear(p0_c=True)` -> `tilelang_row_quant.global_row_quantize(local_rotation=True)` -> `_local_o_proj_p0c`.

Eligibility: physical SM75, TP2, 256-element ConvRot groups, local width 3584, packed row preparation, owned contiguous FP32 Tensor, P0-2 input consumption, no online LoRA, no grad/compile, and one of **S=128, 6,814, 59,134, 65,166**. The explicit row guard reflects the actual physical T4 precision sample coverage. All unqualified calls retain original P0-2 padded/packed quantization.

The candidate rotates owned local FP32 rows before global MAX all-reduce and canonical TileLang quantization, avoiding separate zero-padded full-width FP32 matrices. The input is detached only after the local rotation's final read. P0-2 FP16 input-owner release remains unchanged.

Earlier physical T4 study: both ranks, all four lengths, **zero raw-byte differences** for rotated local slices, row scales and INT8 codes. The observed O-Proj phase-local peak improvement was **1.381/1.523 GiB** at the two long sizes. These numbers are **not** full-H3 peak reductions.

## Default and saved workflow compatibility

- `H3OperatorPolicy.dit_p0_a1_turing=True` and `dit_p0_c_local_rotation=True`.
- Loader exposes both optional controls, default `True`, independently disableable.
- Loader positional migration advances from **v5 to v6**, mapping v5 and earlier widget names before inserting the new switches. Old P0-2/P0-3 explicit opt-outs are retained.
- TE, Video VAE, Audio VAE, online LoRA and unsupported CUDA/shape paths are unchanged.
- No Git push was requested or performed in this change.

## P0-B local experimental design

Source: `benchmarks/dit_p0b_inplace_probe.py`, imported by no production module.

Current FC2 `fused_stored` kernel reads a `[S,14336]` FP16 FC1 output and writes `[S,7168]` FP32 rotated activations in a separate equally sized allocation. Each `(row,group)` CTA owns 256 gate and 256 up FP16 samples, all read before its synchronized butterfly transform. The P0-B candidate reinterprets the same FC1 buffer as `float32 [S,7168]` and writes 256 results into its two disjoint CTA-owned locations:

- Values `i < 128` to `R[row,group*128+i]`.
- Values `i >= 128` to `R[row,3584+group*128+(i-128)]`.

A remapped TileLang quantizer then restores canonical `[S,7168]` INT8 code order. Both candidates use the same FP32 SwiGLU + Radix-4 transform, row-maximum math, `__fdividef`, `nearbyintf` and INT8 saturation. The local baseline is the shipped `fused_kernel(...,stored)` plus an architecture-matched reimplementation of the shipped canonical quantizer. The local communicator is a **no-op MAX** (not a physical two-rank NCCL collective). No FC2 GEMM or entire block was benchmarked here.

### Local environment

Windows conda `comfyui`, **RTX 5060 Ti SM120**, Torch **2.12.1+cu132**, Kitchen **0.2.37**, TileLang **0.1.13**. All tests are native CUDA kernels, not CPU emulation. These versions/devices differ from the earlier physical T4 runtime (Torch 2.11.0+cu128, dual SM75).

### Local P0-B results

Allocated-peak values are incremental over memory already live immediately before the preparation call; they are **not** absolute device peaks or full-FC2 maxima. Timings use CUDA events after compilation/warmup, 5 samples for short sizes and 7 for long sizes. Individual samples were not interleaved between A/B arms; timing differences are indicative only.

| S | INT8 mismatches | FP32 scale bit mismatches | Baseline incremental peak (GiB) | In-place incremental peak (GiB) | Reduction (GiB) | Baseline median (ms) | In-place median (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0 | 0 | 0.004273 | 0.000855 | 0.003418 | 0.09690 | 0.08330 |
| 6,814 | 0 | 0 | 0.227493 | 0.045540 | 0.181953 | 2.55862 | 2.57203 |
| 59,134 | 0 | 0 | 1.974358 | 0.395203 | **1.579155** | 20.60954 | 19.63069 |
| 65,166 | 0 | 0 | 2.175636 | 0.435516 | **1.740120** | 22.57136 | 21.72758 |

The two long local sizes measured **~4.75% and ~3.74% faster** preparation medians respectively; the 6,814 case was **~0.52% slower**. The source is not yet production-safe: there is no physical SM75 alias/compiler qualification, no concurrent rank MAX test, no FC2 weight GEMM, no trained-block output check, no E2E allocated-peak measurement and no robust interleaved performance confidence interval. This does **not** validate the original proposed 1.184/1.305 GiB *FC2-stage* reduction, which is a different baseline/peak measure.

## Local production regressions

- `compileall` of all touched runtime Python sources: passed.
- `python -m unittest benchmarks.test_dit_p0_ac_dispatch benchmarks.test_dit_p0_production_defaults benchmarks.test_dit_p0_source_contracts -v`: **17/17 passed**.
- `python -m unittest benchmarks.test_h3_default_fused_row benchmarks.test_h3_dit_fusion benchmarks.test_h3_operator_integration -q`: **18/18 passed**.
- `node benchmarks/test_loader_migration_p0.mjs`: passed, including v4 and v5 migration.
- `python -m benchmarks.dit_p0b_inplace_probe --rows 128 6814 --repeats 5`: passed.
- `python -m benchmarks.dit_p0b_inplace_probe --rows 59134 65166 --repeats 7`: passed.
- Local offline **SM75 code generation** for both experimental kernels at S=128: passed (`P0B_SM75_COMPILE_PASS`). This does not execute them on a T4.

## Final T4 qualification gate for P0-B

Requires a separately authorized frozen-source physical two-T4 job with the exact same trained block, weights, real MAX collective, P0-A1/P0-C and P0-2/P0-3 enabled equally for baseline and candidate. Check raw bytes of intermediate rotated data/scale/codes and final block output, consistent-normalization Allocated Peak for preparation, FC2, block and long E2E, and sufficiently many interleaved warmed CUDA samples. Reject on mismatches or material slowdown; retain original P0-3 fallback and keep P0-B disabled. Per project policy, **no automatic Kaggle submission, status polling, or retry** was made.
