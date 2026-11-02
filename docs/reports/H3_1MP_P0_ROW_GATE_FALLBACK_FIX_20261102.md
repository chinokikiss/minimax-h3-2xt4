# H3 1MP DiT O-Proj Fallback: Six-Row AdaLN Fix — 2026-11-02

## Status

**Source-level root cause reproduced; targeted production fix implemented and validated locally. Physical dual-T4 rerun remains pending.**

The failed v1 E2E workload is `1344x768 / 243f / 8 Euler NFE / FP16 residual 1:512 / Sol-v2 all NFE / two image references / offline regular Turbo LoRA`. Original immutable evidence: `docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102/gpu_final_v1/`. The OOM at `patches/tilelang_row_quant.py:136` happened at the first O-Proj because the fused row-gate admission fell through and used non-consuming packed FP32 rotation.

## Source-derived fallback predicate

1. `ComfyUI/comfy/ldm/minimax/model.py:669-711` derives the per-stream timestep set. At the first denoiser sigma `1.0` (observed in `generation/result.json`), the video timestamp and shifted audio timestamp are both `0` (`time_shift_sigma(1.0,6,3)==1.0`).
2. This Ref2VA request has **two image references and no audio reference**. Native `PackedLayout` therefore has visual reference rows at `VISUAL_COND_TIMESTEP=0.999` but no `ref_audio/cond_audio` reference rows requiring an extra audio-conditioned timestamp. Its two distinct AdaLN timesteps at NFE1 are `{0, 0.999}`.
3. `model.py:707-710` maps each unique timestep to three modality rows, so AdaLN modulation has **6 rows**, not 9. `TPDiTBlockINT8.forward` reshapes the AdaLN projection into this dynamic `[6,5376]` table.
4. The prior `patches/tilelang_dit_fusion.py::_table` required the literal `(9,5376)`. `row_gated(...)` therefore returned `None` before its P0-2/P0-C lifetime/local-rotation branch. `tp_dit.RowParallelLinearINT8.forward` dispatched to the ordinary scaled-FP16 transport `row_linear` with `consume_input=False`, matching the saved stack in `generation/exception.txt`. The padded FP32 activation then hit the 2.09 GiB allocation failure on GPU 0.

This is a **deterministic source-level explanation** under the recorded image-only, unmasked test contract. The failed run did not record a runtime predicate bitmap; direct physical-GPU post-fix branch and peak-memory proof still require a new authorized run.

At other sigma values, video/audio timesteps generally differ and the table may have 9 or more rows. The hard-coded 9 also excluded legitimate lower/higher row counts, independently of sequence length. The fault was **AdaLN row cardinality, not a P0-C sequence-length whitelist**.

## Minimal production repair

Changed **only** `patches/tilelang_dit_fusion.py` among runtime files:

- Accept `3*N` rows in FP32 modulation tables (positive, matching width/device); enforce equal shift/scale row counts.
- Pass the actual modulation-row count to `norm_kernel`, `gate_kernel`, and `row_gate_kernel`. Their already-qualified factory bodies are unchanged; the generated kernels receive a correct dynamic `T.Tensor((modulation_rows,5376))` signature.
- Validate segment indices against the active modulation-row count. Continue rejecting negative/out-of-bounds values, gaps/overlaps, non-integer or per-token tensor-valued row indices, and all previous device/dtype/layout/LoRA/grad guards. No relaxation of P0 ownership or SM75 constraints.
- `row_gated` can now enter the existing production P0-2 plus P0-C path for six-row eligibility, and P0-3/P0-B for its separate FC2 eligibility. The O-Proj kernel and P0-C quantizer algorithms are unchanged.

The archived v1 notebook/source capsule and every original output file remain unmodified. The AOT plan still precompiles its previous nine-row shape; a six-row kernel may require first-use compilation until a separately qualified AOT change is made.

## Verification

Local Windows conda interpreter: `C:\ProgramData\miniconda3\envs\comfyui\python.exe`.

- 42 Python tests passed: `benchmarks.test_h3_dit_fusion`, `test_dit_p0_production_defaults`, `test_dit_p0_source_contracts`, `test_dit_p0_ac_dispatch`, `test_h3_default_fused_row`, `test_h3_operator_integration`.
- Loader widget migration test passed: `node benchmarks/test_loader_migration_p0.mjs`.
- New tests reproduce the `sigma=1.0` six-row derivation, reject out-of-range and tensor-valued indices, exercise `RowParallelLinearINT8.forward` through six-row P0-2/P0-C O-Proj consumption and P0-3/P0-B FC2 selection, and verify row-count propagation for fused norm/gate plus mismatched-table fallback.
- `git diff --check` passed. Existing unrelated dirty worktree changes were preserved.

These are **local dispatch/source tests**, not physical SM75 kernel numerical comparisons, CUDA memory measurements, throughput benchmarks, or full-E2E output quality validation.

## Remaining validation

Before claiming resolution of the 1MP OOM, a separate authorized physical dual-T4 run must verify that the first O-Proj actually enters `row_gated` and `_local_o_proj_p0c`; compare precision and runtime overhead across NFE1 (six AdaLN rows) and later NFE (normally nine), record full allocator peak including any new dominant FC1 peak, and complete all eight denoising steps plus decoder/export. Do not silently resubmit the failed v1 job.
