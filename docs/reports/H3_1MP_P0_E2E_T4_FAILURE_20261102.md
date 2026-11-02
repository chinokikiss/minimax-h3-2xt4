# H3 1MP FP16/512 Sol-v2 Dual-T4 E2E Failure — 2026-11-02

## Verdict

**FAILED: CUDA out of memory during the first DiT denoising iteration.** The submission reached two physical Tesla T4 GPUs, completed offline regular Turbo LoRA disk baking and reference/text-encoder conditioning, then failed in the DiT Attention O-Proj row quantization preparation. No denoising step completed; there is no end-to-end timing, output-quality, or full-run memory-success result.

Kernel v1: https://www.kaggle.com/code/pamcks/h3-1mp-fp16-512-sol-v2-8nfe-10s-p0-20261102

## Immutable workload and observed environment

- Canvas: 1344 x 768 (1.032192 MP); 243 native frames; target 240 exported frames / 10 s.
- Two physical Tesla T4 (SM75), Torch 2.11.0+cu128 / CUDA 12.8, Kitchen 0.2.37, TileLang 0.1.13.
- Eight Euler/Simple NFE, CFG 1, FP16 tensor flow and residual exponent 9 (1/512), Sol-v2 tau 1.0 at all NFE1-8.
- Offline baked regular Turbo LoRA, 208 verified native targets, no online inference remerge.
- The saved runtime configuration sets `dit_p0_2_lifetime`, `dit_p0_3_lifetime`, `dit_p0_b_inplace`, `dit_p0_a1_turing` and `dit_p0_c_local_rotation` to `true`. Configuration is **not** executed-branch proof.

## Execution and failure

`batch_results.json`: offline_disk_bake succeeded (exit 0; 336.475 s), full_e2e failed (exit 1; 248.989 s). The `generation/result.json` stages `vae_metadata` and `reference_and_TE` completed; `sampler` failed. `sampler_steps=[]`, `denoiser_outputs=[]`, and the progress bar remained at `0/8`.

The GPU worker traceback identifies:

```text
patches/tp_dit.py:337                     x = self.out_proj(attn_out, ...)
patches/tp_dit.py:150                     out = dispatch_row(...)
patches/h3_operator_policy.py:238         return run(*args)
patches/tilelang_transport_p2.py:118     prepare.global_row_quantize(..., variant='packed')
patches/tilelang_swiglu_pad.py:107       previous.global_row_quantize(...)
patches/tilelang_row_quant.py:136        rotated = backend._rotate_activation(padded, h, groupsize)
comfy_kitchen/tensor/int8_utils.py:72    x_rotated = torch.matmul(x_grouped, h)
torch.OutOfMemoryError: Tried to allocate 2.09 GiB.
GPU 0 capacity: 14.56 GiB; free: 1.93 GiB.
PyTorch allocated: 9.28 GiB; reserved but unallocated: 2.55 GiB.
Process memory including non-PyTorch: 12.07 GiB.
```

The initial PyTorch allocation failure is conclusive. The available evidence does not isolate fragmentation versus aggregate live allocations; `expandable_segments` is a diagnostic suggestion from PyTorch, not a validated fix.

## Critical dispatch finding

The observed path enters `RowParallelLinearINT8.forward`'s ordinary `h3_operator_policy.row_linear` fallback. The qualified `patches.tilelang_dit_fusion.row_gated` fusion did **not** return a fused output for this workload. Consequently the transport call did **not** pass `consume_input=True` or `p0_c=True`. It took `tilelang_transport_p2.row_linear:118` (`else: q,scale=prepare.global_row_quantize(...,variant='packed')`) and the full-width padded/native FP32 ConvRot path.

Thus this failure does **not** demonstrate that P0-C local rotation itself exceeded T4 capacity; it demonstrates that P0-C and P0-2 input consumption did not execute at the failing projection, despite their default-on configuration flags. `row_gated` can decline for policy, ownership, shape, dtype, layout, online-LoRA, gradient/compile, weight scale, fused gate label, or other explicit guards. The current logs have no per-guard decision record, so the **precise failed eligibility predicate remains unproven**. Do not attribute it to a particular guard without instrumented reproduction.

This is also not evidence about P0-3/P0-B FC2 efficacy, since the first O-Proj failed before completion of the DiT block.

## Next qualification before any GPU rerun

1. Locally inspect/reproduce the real block-0 `row_gated` admission inputs (tensor shapes, dtypes, strides, `_base`, `is_contiguous`, `grad_enabled`, `operator_policy`, LoRA state, `segments`, `labels_for` success, `weight_scale`). Add bounded, one-shot branch-reason telemetry only if user authorizes a diagnostic change.
2. Verify the qualified P0-2 + P0-C branch would be selected at this exact 1MP DiT sequence length on SM75, then run local CPU/no-GPU admission checks in the pinned `comfyui` environment.
3. Request explicit authorization for any code modification or subsequent physical dual-T4 replay. Keep the previous v1 outputs immutable and do not infer speed, accuracy, or all-NFE feasibility from this failed run.

## Archived evidence

All final outputs were collected **once** on user-reported failure, without Kaggle status/list/live-log polling:

`docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102/gpu_final_v1/`

Key artifacts:

- `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/generation/exception.txt`
- `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/generation/result.json`
- `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/full_e2e.log`
- `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/batch_results.json`
- `h3_t4_fp16_512_sol_v2_1mp_10s_supervision/supervision.json`

No production source code was changed, and no Kaggle rerun was submitted in this diagnostic collection.
