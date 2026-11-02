# H3 DiT P0-B Production Promotion and P0-C Shape-Tolerance Audit (2026-11-02)

**Superseded P0-C dispatch decision:** The user subsequently authorized all positive sequence lengths. See `H3_DIT_P0_C_ALL_SEQUENCE_LENGTHS_20261102.md`; the four-shape restriction below is an archival snapshot of the prior guard.

## Decision and scope

The user explicitly requested **default-on production P0-B**, overriding the earlier physical-T4 report's recommendation to hold promotion because warmed S=59,134 block timing had not passed strict non-regression. This change is implemented in the live ComfyUI TP2 DiT pipeline, not as an experimental monkey-patch or benchmark import.

**P0-C remains default-on only for the four previously physically qualified sequence lengths**: 128, 6,814, 59,134 and 65,166. Relaxing bit-for-bit equality does **not** establish an acceptable full-block or image/audio-generation quality bound at other lengths. Its whitelist was not changed.

## Runtime implementation

- Loader: `custom_nodes/minimax_h3_tp2/__init__.py` exposes `dit_p0_b_inplace=True`, passed to `H3OperatorPolicy`.
- Policy: `patches/h3_operator_policy.py::configure_block` installs `_vram_p0_b` on **DiT FC2 only**. TE down_proj, DiT QKV, FC1 and O projection do not run P0-B.
- Dispatch: `patches/tilelang_dit_fusion.py::row_gated` requires the existing scaled-FP16 fused row gate, valid T4 FP16 geometry, TP2/ConvRot256, no online LoRA, no gradients or compilation, and an eligible P0-3 owner. It passes `p0_b=True` only under those conditions.
- Entry: `patches/tilelang_swiglu_quant.py::prepare` first tries `patches/tilelang_swiglu_inplace.py::prepare`. Unsupported conditions return `None`, retaining the original P0-3 fused_stored kernel and its lifetime release. The original fused kernel body is unchanged.
- Eligible in-place input: actual SM75/T4, `mode='fused_stored'`, `consume_input=True`, exact owned `torch.Tensor`, FP16 contiguous `[S,14336]`, `S>0`, `_base is None`, `storage_offset()==0`, and no training/compiler. P0-B uses the equal-size `[S,7168]` FP32 alias of the FC1 owner, a CTA-owned SwiGLU/ConvRot transform, global MAX all-reduce, and the remapped INT8 quantizer. It detaches FC1 only after the final quantizer read.
- Explicit opt-out: `dit_p0_b_inplace=False`; disabling P0-3 also prevents P0-B. Unsupported row counts are not specially whitelisted for P0-B, because CTA groups are disjoint and the kernel has no row-dependent arithmetic reduction; this is a source-level generalization, **not** physical T4 qualification for every S.
- Saved workflow migration: version **7** adds P0-B; saved v1-v6 positional settings are restored by their historical widget names, preserving existing P0-2/P0-3/P0-A1/P0-C opt-outs.

No production module imports `benchmarks/dit_p0b_inplace_probe.py`. The two compiled P0-B kernel factory ASTs are byte-independent source-structure matches of the physical-T4-qualified experimental factories.

## Physical-T4 evidence inherited from the isolated candidate

The prior report `docs/reports/H3_DIT_P0_B_T4_20261102.md` compared the same CTA-owned transformation and remapped quantizer on **both real Tesla T4 GPUs**, using the trained TP2 block 0 and true two-rank all-reduce.

| Gate | S=59,134 | S=65,166 |
| --- | ---: | ---: |
| FC2 phase Allocated Peak reduction | 1.190 GiB | 1.312 GiB |
| Trained whole-block Allocated Peak reduction | 0.597 GiB | 0.503 GiB |
| INT8 code/FP32 scale raw-bit mismatch | 0 / 0 | 0 / 0 |
| Trained-block FP16 final raw-bit mismatch | 0 | 0 |
| Warmed slower-rank block median relative change | **+0.93% (slower)** | **−0.80% (faster)** |

The previously recorded candidate additionally passed raw-bit FC2 codes/scales at S=128 and S=6,814. Its performance comparison always ran baseline before candidate within each timing pair, so timing-order bias remained unresolved. The user now chooses to accept this uncertainty in exchange for memory savings. These figures are **not** full 25-block H3 end-to-end savings, and the integrated production code has **not** been remeasured on physical T4 in this change.

## P0-C: tolerance beyond the four certified sizes

The live gate is `patches/tilelang_row_quant.py::_local_o_proj_p0c`: T4/SM75, TP2, FP32 contiguous owned `[S,3584]`, no LoRA/grad/compile, packed path, and `S in (128,6814,59134,65166)`. Other sizes retain the original packed P0-2 rotation; they do **not** switch to P0-C automatically.

On local **RTX 5060 Ti / SM120**, the Kitchen FP32 padded-versus-local rotation and a **PyTorch INT8 quantization proxy** were compared on two deterministic synthetic ranks. Evidence: `docs/data/h3_dit_p0c_shape_local_20261102.json`, using `benchmarks.verify_dit_p0a_p0c_local.p0c_exact`. This is **not** the SM75 TileLang quantizer or a trained model output test.

| Local S | Max absolute FP32 rotation difference | Max absolute scale difference | INT8 proxy code mismatches, rank 0 / 1 | Max code difference |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.15e-7 | 0 | 0 / 0 | 0 |
| 127 | 3.34e-6 | 1.86e-8 | 2 / 9 | 1 |
| 129 | 3.10e-6 | 2.05e-8 | 6 / 0 | 1 |
| 256 | 0 | 0 | 0 / 0 | 0 |
| 1,024 | 0 | 0 | 0 / 0 | 0 |
| 4,096 | 0 | 0 | 0 / 0 | 0 |
| 6,815 | 0 | 0 | 0 / 0 | 0 |
| 8,192 | 0 | 0 | 0 / 0 | 0 |
| 32,768 | 0 | 0 | 0 / 0 | 0 |
| 59,135 | 0 | 0 | 0 / 0 | 0 |

The earlier local S=128 proxy also had nonzero INT8 differences (10 / 4), whereas the **actual dual T4** studied S=128/6814/59134/65166 was raw-bit exact. This demonstrates architecture/batched-GEMM-sensitive rounding rather than a simple universal row-count threshold.

**Numerical interpretation:** The local nonexact cases show small rotation deltas and at most one INT8 step per differing code, but a 1-LSB code change can propagate through INT8 matmul, dequantization, residual scaling, 25 blocks and sampling. Neither a permissible final-output error budget nor actual dual-T4 non-whitelisted Block/E2E discrepancies has been established. Consequently **the answer is not yet “within allowed range at arbitrary S.”** A defensible relaxed qualification must define final-output relative error/maximum error/nonfinite limits and compare actual two-rank T4 trained-block and generation outputs for representative additional S, alongside memory and warmed latency.

## Verification of this production edit

- Local environment: `C:\ProgramData\miniconda3\envs\comfyui\python.exe`.
- Relevant local test suites (`test_dit_p0_production_defaults`, `test_dit_p0_source_contracts`, `test_dit_p0_ac_dispatch`, `test_h3_dit_fusion`, `test_h3_default_fused_row`, `test_h3_operator_integration`) passed **38 tests** including the promoted-kernel AST parity check.
- ComfyUI migration: `node benchmarks/test_loader_migration_p0.mjs` passed.
- `git diff --check` passed (existing Windows line-ending notices only).
- Local P0-C proxy completed 10 additional row sizes; no extra Kaggle submission, physical T4 run, or full H3 E2E claim.
