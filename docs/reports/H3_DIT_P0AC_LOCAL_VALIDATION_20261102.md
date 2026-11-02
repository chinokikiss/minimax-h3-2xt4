# H3 DiT P0-A / P0-C — Isolated Local Validation (2026-11-02)

## Decision

**Local checks completed; physical dual-T4 qualification is pending. Neither candidate is promoted to production.** P0-2 and P0-3 remain default-enabled, and removed P0-1 is still absent.

Environment: Windows conda comfyui at C:\ProgramData\miniconda3\envs\comfyui\python.exe, PyTorch 2.12.1+cu132, comfy-kitchen 0.2.37, RTX 5060 Ti (SM120). This is **not** SM75/T4. All measurements below are isolated local proxies, not full-H3 end-to-end or T4 non-regression evidence.

Evidence:
- Reusable script: benchmarks/verify_dit_p0a_p0c_local.py
- JSON receipt: docs/data/h3_dit_p0ac_local_20261102.json
- Local command: python -m benchmarks.verify_dit_p0a_p0c_local --rows 128,6814,59134,65166 --include-proxy --json-output docs/data/h3_dit_p0ac_local_20261102.json
- Production source/default contract tests: 12 passed.
- No live production sources or installed Kitchen packages were changed.

## P0-A — Kitchen CUDA double output

In the pinned Kitchen source, cuda.int8_linear allocates an FP16 output before selecting the Turing-fused branch. The successful _int8_linear_turing_quantized branch allocates and returns its own second FP16 output; the first output remains allocated until the outer function returns. Both H3 TP2 QKV and FC1 geometries satisfy the source-level fused-branch shape predicate. Actual physical T4 branch dispatch has **not** been logged in this validation.

**A1 (skip dead output):** An experiment-only clone of Kitchen's Python function moves the first allocation to *after* the Turing early return, retaining the fallback allocation. Direct local QKV and FC1 complete forward outputs were bit-exact at S=128, but the local SM120 used the non-Turing fallback. Kitchen activation quantization codes and scales were byte-identical at S=128, 6814, 59134, 65166.

**A2 (detach input owner):** Real ownership is unresolved. Production ColumnParallelLinearINT8.forward creates x_2d = x.view(...). A view.set_() test retained **all** 688,128 bytes owned by its base Tensor; only owner.set_() released storage. Thus an inner view-only release would not free the actual h/h2. The A2 allocator result below presumes detaching an independently owned tensor and does **not** prove a production fix. A2 must be guarded for aliases, training, compilation and LoRA and separately tested; do not resurrect removed P0-1 blindly.

### P0-A FC1 allocation ledger proxy

True local Kitchen ConvRot activation preparation plus torch.empty allocations; **no GEMM**, no real Turing branch, no block E2E. Incremental PyTorch Allocated Peak:

| S | Baseline | A1: skip dead output | A1+A2: hypothetical owned detach | A1 reduction | Further A2 reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 59,134 | 3,538.116 MiB | 1,921.171 MiB | 1,314.816 MiB | 1,616.945 MiB | 606.355 MiB |
| 65,166 | 3,898.352 MiB | 2,116.352 MiB | 1,448.146 MiB | 1,782.000 MiB | 668.206 MiB |

The A1 numbers equal one FP16 FC1 output (approximately 1.579/1.740 GiB). No claim is made that a physical T4 stage peak will fall by the same amount. The A2 reduction is **conditional on real owner detachment**, which the current view-only stack does not provide.

## P0-C — Local-width O-Proj rotation

Qualified production route: tilelang_transport_p2.row_linear -> tilelang_row_quant.global_row_quantize -> packed padding [S,3584] to [S,7168] -> Kitchen _rotate_activation -> TileLang rank-local maximum -> global MAX all-reduce -> SM75 quantizer. The local experiment ran Kitchen rotation with padded and unpadded inputs and simulated both ranks/global maxima.

**Blanket bitwise identity is disproved.** At S=128, rank 0 had 408,799/458,752 differing FP32 rotated values (maximum absolute difference 3.3378601e-6); rank 1 had 409,309 differences. Global maxima and scales differed on 104 rows. The PyTorch proxy quantizer had 10/4 differing codes for ranks 0/1, maximum code delta 1. Changing the batched cuBLAS matmul geometry can change rounding despite group independence.

| S | Both ranks' rotated FP32 bytes | Global maximum and scale bytes | Proxy INT8 code bytes | Actual SM75 TileLang |
| ---: | --- | --- | --- | --- |
| 128 | **Different** | **Different** | **Different** | Not run |
| 6,814 | Exact | Exact | Exact | Not run |
| 59,134 | Exact | Exact | Exact | Not run |
| 65,166 | Exact | Exact | Exact | Not run |

Floating results were compared by raw 16/32-bit integer views, not by tolerance. The production-sized *local synthetic* cases passed, but this does not prove T4 cuBLAS or SM75 TileLang quantizer bitwise identity. The proxy uses PyTorch rounding, not TileLang __fdividef/nearbyintf. P0-C cannot be labeled universally 100% bit-exact.

### P0-C rotation-only local proxy

The padded reference here uses PyTorch zero-fill and copy, **not** the production SM75 packing kernel. Timings are therefore not production-speed evidence. Measured incremental Allocated Peak:

| S | Padded peak | Local peak | Reduction | Padded median | Local median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 59,134 | 2,425.418 MiB | 808.473 MiB | 1,616.945 MiB | 25.2971 ms | 8.0456 ms |
| 65,166 | 2,673.059 MiB | 890.941 MiB | 1,782.118 MiB | 25.3513 ms | 8.0907 ms |

These are rotation-phase proxies, not real O-Proj stage reduction and not true full-H3 Allocated Peak reduction. The saved JSON contains four sizes, individual rank comparisons and all measurements.

## Required physical dual-T4 gates — not yet run

1. Freeze the proven trained TP2 block0 and exact source/core pins; retain residual exponent 9, default P0-2/P0-3, S=59134 Kitchen and S=65166 Sol-v2. CPU-preflight any new T4 notebook. Do not replay frozen V3 notebooks against the P0-1-removed live tree.
2. A1: instrument actual Kitchen registry/Turing CUTLASS dispatch; compare byte-exact QKV/FC1 outputs and final trained block outputs. Measure real per-stage and full-block PyTorch Allocated Peak and interleaved warmed latency.
3. A2: separately prove actual h/h2 Tensor ownership, last-use safety, alias and LoRA safeguards; only then compare its incremental performance, bytes and peak against A1.
4. C: compare *actual* rank0/rank1 padded versus local SM75 paths, raw rotated bytes, row maxima, scales, TileLang INT8 codes, O-Proj output, and final block bytes; include nonfinite and signed-zero checks. Record real packing-kernel baseline latency, real stage peaks and E2E block peak.
5. Reject any candidate with a byte mismatch, performance degradation, no real peak reduction, changed backend or failed ownership guard. Never convert the proxy table into a full-H3 E2E claim.

**Kaggle follow-up (2026-11-02):** One independent dual-T4 GPU notebook (v1) was accepted at https://www.kaggle.com/code/pamcks/h3-dit-p0-a1-p0-c-t4-20261102. The metadata-requested ID and title-derived URL slug differ; the actual provider URL is recorded in docs/data/h3_dit_p0ac_t4_20261102/gpu_submission.json. Submission is **not** execution or validation success. Final results are pending. No Kaggle status/live-log polling or automatic retries. Production flags/default dispatch remain unchanged.
