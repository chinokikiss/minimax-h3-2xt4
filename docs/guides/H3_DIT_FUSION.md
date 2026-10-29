# Production DiT fusion

The four 2026-10-29 operator candidates are integrated into the production
DiT block. Each has a default-on boolean in `H3OperatorPolicy` and the
`MiniMaxH3TP2Loader` optional inputs:

| Switch | Production behavior |
|---|---|
| `dit_fused_norm` | TileLang RMS + AdaLN, FP32 statistics/gains/modulation, FP16 output |
| `dit_fused_gate` | In-place FP32 gated residual FMA with FP16 output |
| `dit_fused_row_gate` | Scaled-FP16 SUM followed by fused INT32 reconstruction, FP32 dequantization, FP16 branch rounding and gated residual store |
| `dit_prepared_attention` | Pinned Kitchen prepare/execute API with floating QKV released between the two stages |

Set all four switches to `False` to restore the previous operator selection.
Disabling only `dit_fused_gate` leaves the independent row-epilogue/gate fusion
enabled; disable `dit_fused_row_gate` as well to restore separate residual gates.
Policy values participate in normal session and residency identity, so a changed
selection does not silently reuse a previously configured block.

The fused elementwise kernels retain the tested SM75 factories without changing
their arithmetic. They target contiguous FP16 residual matrices with width 5376
and the usual nine-row modulation table. Other devices, BF16/FP32 residual flow,
other modulation layouts, gradient-enabled execution and compilation through
Torch retain existing operators. Segment labels must cover the packed sequence
without gaps or overlaps. A bounded four-entry label cache shares immutable
labels across blocks, with CUDA ready events and `record_stream` lifetime
protection for stream handoff/eviction.

Row fusion is restricted to the current TP2 O/FC2 geometries, ConvRot256,
vector FP32 weight scales and `tilelang_scaled_fp16`. FC2 retains the selected
`fused_stored` or `fused_recompute` quantizer and its shared MAX. The existing
INT8 GEMM, power-of-two encoding and FP16 SUM are unchanged. Its epilogue retains
the FP16 branch rounding before FP32 residual FMA even though that branch is
not stored as a full tensor.

**Online LoRA bypasses row-gate fusion**. Its contribution is still added in
FP32 before the branch storage cast, followed by the separate residual gate.
Offline-baked and PTD-bank weights use the fused path when otherwise eligible.
Selecting native/INT32 row transport also keeps the separate epilogue. Ordinary
RowParallel calls without residual arguments retain their existing return API.

Prepared attention applies only to Kitchen FP16 self-attention on the target
T4 geometry. Dense TileLang, sparse drop/scalar and Native SDPA remain selectable
and keep their existing implementations. The installed Kitchen package and
legacy attention API are unchanged; unavailable prepare/execute entrypoints
fall back to the prior call. TE and both VAEs are unaffected by these switches.

AOT preparation includes the exact norm, gate and two row-gate factory keys.
The worker passes the norm epsilon corresponding to its configured residual
divisor. Built-in Kitchen attention kernels remain precompiled. Diagnostics
remain under `benchmarks/`; production does not import experiment code.
E2E/profile observers recognize the prepared execute entrypoint and fused row
epilogue without double-counting ordinary Kitchen calls.

The prior physical T4 run qualified the isolated kernels, including byte-exact
gate/row-gate/attention outputs against their declared operator references.
Fused RMS has small FP32 reduction-order differences; it is not byte-exact and
can be independently disabled. No full-generation quality equivalence is
claimed. The production combination has CPU wiring, fallback, ordering and
lifetime coverage; **a fresh combined-production T4 run has not been performed**.
The earlier component speedups and memory reductions are not a full-DiT result.

See [the T4 experiment report](../reports/H3_DIT_VRAM_FUSION_EXPERIMENT_20261029.md)
for actual latency, allocation peaks, traces and the original block-harness
failure. No dependency, Native source, FC1 backend or attention operator was
removed or upgraded for this integration.
