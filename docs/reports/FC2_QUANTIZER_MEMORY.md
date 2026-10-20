# FC2 Global Quantizer memory optimization

The current work targets the DiT TP2 FP32 fallback-rotation quantizer. It does
not change per-rank/global scale semantics, INT8 rounding or INT32 reduction.
Physical dual-T4 version1 passed correctness and performance validation.

## Root cause and candidates

At S8192 each rank's packed FC1 output occupies 448 MiB. The original quantizer
overlaps that input with 448 MiB zero padding and 448 MiB full-width rotation.
The caller also retains padding through MAX and output quantization, adding a
56 MiB INT8 payload and two 32 KiB row vectors to the three large buffers.

1. **Lifetime fix:** split rotation from MAX/quantization and release padding in
   the actual caller after rotation. Original FP32 GEMM geometry and all
   arithmetic are unchanged. The helper's inner `del` alone was insufficient.
   Both packed DiT and separate TE gate/up callers are repaired. Stream order
   is unchanged; same-stream last reads precede storage reuse.
2. **Owned-only FP32 rotation:** validated as a benchmark candidate, then added
   as the explicit `owned_fp32` session option. Materialize
   the owned SwiGLU output (224 MiB), then rotate only its complete256-channel
   groups into another 224 MiB buffer. Reduce the same one FP32 maximum per
   complete row across ranks, then use the existing external-global-max INT8
   quantizer. No FP16 activation, per-shard scale, approximate sigmoid or
   compressed AllReduce is introduced. GEMM flattened M changes, so byte
   equivalence is tested instead of inferred from algebraic locality.

Theoretical live payload including the retained 448 MiB packed input, excluding
AdaLN/weights/library state: original 1400.0625 MiB, lifetime 1344 MiB,
owned rotation 896 MiB. Whole-FFN peak can move to base FC1, whose unchanged
duplicate-output INT8 dispatch previously measured 938.03 MiB at S8192.

## Local qualification

RTX5060Ti, real block0 checkpoint projections, S513/8192/16384, no adapter and
two separate DMAD entries (+0.6/-0.2), Row Partial and selected FC1 B. Both
logical owners execute real CUDA arithmetic; collectives are substituted only
for local correctness. This is not physical dual-T4 performance evidence.

All18 variant/configuration cases (36 rank cases) pass. Lifetime and owned
rotation both match original INT8 payloads, scale representations and complete
FC2 outputs byte for byte: max absolute/relative L2 zero, cosine1. The two-entry
stack is two strengths of the same supplied checkpoint, not two independently
trained adapters. Separate TE353x12800 chain tests pass on both logical owners
after the shared helper change, including quantized payload, INT32 partial/SUM,
bias and residual output.

Local isolated quantizer extra allocation above resident input/weights:

| S | Original | Lifetime fix | Owned-only FP32 rotation |
|---:|---:|---:|---:|
| 513 | 59.621 MiB | 56.109 MiB | 28.055–28.121 MiB |
| 8192 | 952.0625 MiB | 896 MiB | 448 MiB |
| 16384 | 1904.125 MiB | 1792 MiB | 896 MiB |

The retained packed input is outside these deltas. Local memory gates use a
precomputed global maximum from the validated logical run; mock-transport
temporary allocations are excluded. Local architecture/private-helper
adaptations are recorded and confined to the benchmark process.

## Physical protocol and evidence

Kaggle `ocnsai/h3-fc2-quantizer-memory`, version1; actual 2xT4 required by the
harness. Only sources/plans are uploaded. About425 MB of selected checkpoint
ranges download directly on Kaggle and are removed after the bounded job.

Identical trained inputs/weights and three ablations, S513/8192/16384, with and
without the two-entry adapter stack. Seven samples per round, three operator
warmups, two reversed rounds, synchronized CUDA Events and wall time, fixed
NCCL configuration, separate peak-only passes after timing. Quantizer, FC2 and
whole FFN are measured separately. Every correctness tensor is released before
timing. Comparator packed input remains resident equally in all variants;
full-FFN temporary deltas additionally include its freshly computed FC1 output.
Absolute allocated/reserved values are retained separately; PyTorch values
exclude NCCL's internal allocations. No full generation speedup is implied.

## Physical dual-T4 results, version1

All18 variant/configuration cases passed on both physical ranks. For each
variant, all12 rank cases have identical INT8 payload, scale bytes and FC2
outputs against the frozen original: max absolute/relative L2 zero, cosine1.
This is byte equality against the **same existing Row Partial configuration**;
it does not erase Row Partial's previously accepted difference from Native A
gathering. No new observed numerical loss is introduced by the quantizer.

Median synchronized wall milliseconds, slower rank per iteration,14 samples
across two reversed rounds. Temporary MiB is above the resident entry and is
identical on both ranks at S8192/16384:

| S | Stack | FFN original / lifetime / owned, ms | FFN temporary original / lifetime / owned, MiB |
|---:|---|---:|---:|
| 513 | None | 6.803 / 6.805 / 6.416 | 87.676 / 84.164 / 58.742 |
| 513 | Two entries | 11.993 / 11.960 / 11.542 | 88.419 / 84.164 / 58.742 |
| 8192 | None | 101.548 / 101.988 / 92.853 | 1400.063 / 1344 / 938.031 |
| 8192 | Two entries | 182.948 / 182.530 / 172.999 | 1400.063 / 1344 / 938.031 |
| 16384 | None | 237.077 / 237.947 / 220.291 | 2800.125 / 2688 / 1876.063 |
| 16384 | Two entries | 380.450 / 380.721 / 364.989 | 2800.125 / 2688 / 1876.063 |

At S8192 with the adapter stack:

| Scope | Original | Lifetime | Owned FP32 |
|---|---:|---:|---:|
| Quantizer wall time, ms | 21.513 | 21.239 | 13.362 |
| Quantizer extra peak, MiB | 952.063 | 896 | 448 |
| Complete FC2 wall time, ms | 96.548 | 96.476 | 89.019 |
| Whole FFN CUDA Event median, ms | 182.840 | 182.434 | 172.897 |
| Whole FFN absolute allocated peak, MiB | 2338.486 | 2282.423 | 1876.455 |
| Whole FFN reserved peak, MiB | 2504 | 2504 | 2280 |

S8192 whole-FFN temporary peak falls33.00%, latency5.44% with the stack and8.56%
without it. Quantizer extra allocation falls52.94%. The ownership-only fix
removes56.0625 MiB/rank but does not establish a meaningful latency change.
Owned FFN gains repeat in both orderings:182.948 pooled baseline vs owned
round medians172.965/173.032 ms with adapters. This is one physical allocation,
not a statistically established cross-machine or end-to-end speedup.

The new FFN peak is the unchanged **base FC1 INT8 dispatch**: two448 MiB output
buffers +42 MiB activation INT8 +32 KiB scale =938.03125 MiB. Owned FC2 itself
keeps packed448 + owned activation224 + owned rotation224 =896 MiB at its
largest explicit boundary; at quantization the activation is already released.
No new simultaneous full-width padding or full-width rotation is allocated.

This does not prove a33% reduction in whole-block or pipeline VRAM. The updated
analytical block ledger with Row Partial/selected B moves its named-storage
maximum to the attention O projection (1289.107 MiB including U3 AdaLN), where
earlier attention tensors remain retained. Weights, allocator cache, NCCL and
other contexts also remain. No unrelated attention lifetime change is made.

Version 1 completed its physical audit in 274.96 s; setup/download took about
142 s. Version 2 then validated S32768 through the integrated session-policy
dispatch and production row-linear function, including real MAX, INT32 SUM
and adapter addition. It completed in 217.90 s, excluding setup.

### S32768 integrated confirmation, version 2

All six variant/configuration cases pass on both ranks. INT8 payload changes,
changed scale rows, output maximum absolute error and relative L2 are all zero;
cosine is 1. Source hashes for the tested quantizer/dispatcher match current
files when the evidence is collected. No OOM or collective timeout occurred.

| Stack / scope | Original | Lifetime | Owned FP32 |
|---|---:|---:|---:|
| Quantizer extra peak, either stack, MiB/rank | 3808.250 | 3584 | 1792 |
| FFN temporary peak, either stack, MiB/rank | 5600.250 | 5376 | 3752.125 |
| No adapter, FC2 latency, ms | 302.180 | 302.205 | 279.218 |
| No adapter, FFN latency, ms | 470.033 | 466.985 | 438.848 |
| Two entries, FC2 latency, ms | 427.425 | 427.138 | 405.868 |
| Two entries, FFN latency, ms | 797.886 | 804.084 | 781.633 |

The 1848.125 MiB/rank whole-FFN reduction is 33.00%. FC2 improves 5.04% with
adapters and 7.60% without them. **Do not treat the two-entry whole-FFN pooled
2.04% gain as stable:** its reversed round medians are 785.883 -> 790.297 ms
(0.56% slower), then 811.004 -> 770.304 ms (5.02% faster). FC2 itself wins both
orderings. This run sampled neither GPU clocks nor power, so the source of the
FFN variation is unproven. No-adapter FFN improves in both rounds, with a
descriptive pooled 6.63% reduction.

## Enable and compatibility

The loader and runtime policy default to `fc2_quantizer='owned_fp32'`;
`native` remains selectable for full-width rotation.
Use owned mode with `row_linear_backend=tilelang_packed` and
`swiglu_padding=True`. Other backend combinations retain their existing dispatch;
the quantizer preference does not override them. The production branch is restricted to FP32 H3 packed
FC2 width14336, TP2 and256-channel groups; unsupported operator shapes use the
original path. TE down projection keeps full-width rotation, with the lifetime
fix. INT32 SUM is retained. The separate scaled-FP16 transport option is not
modified or selected by this feature.

For Python configuration use `H3OperatorPolicy(fc2_quantizer='owned_fp32')`.
The new field is appended after existing positional fields; old loader calls
retain their argument meanings. Changing it rebuilds the session through the
existing configuration equality check. UI/default/session-switch/dispatch
smoke checks pass, as do12 existing CPU integration regressions.

With online LoRA, the full-input-gather row mode can peak at1512 MiB/rank at
S8192 even after quantizer optimization. The whole-FFN numbers above use the
existing optional Row Partial path to avoid that separate gather peak.

Full-width Native geometry remains selectable. Changing flattened GEMM M
can select another FP32 reduction algorithm on other inputs/backends.
Observed byte equality for the qualified matrix is not a universal promise.
No FP16 accuracy tradeoff is necessary for the measured memory/latency gain.

Reproducible evidence:

- `semantic_results/fc2_quant_memory/baseline/tilelang_swiglu_pad.py`: frozen
  pre-change baseline.
- `semantic_results/fc2_quant_memory/local/results.json`: local comparisons,
  local-only memory deltas and source hashes.
- `semantic_results/fc2_quant_memory/te_parts_gate/te_chain_gates.json`: shared
  helper safety qualification.
- `semantic_results/fc2_quant_memory/ledger_lifetime_fixed.json` / `.csv`: updated
  per-T analytical storage ledger. Older ledgers remain preserved.
- `semantic_results/fc2_quant_memory/ledger_owned_fp32.json` / `.csv`: optional
  owned-rotation block ledger, including the new O-projection maximum.
- `semantic_results/fc2_quant_memory/t4_v1/`: version-bound physical raw results,
  source hashes/download receipts and derived summary.json. Performance and
  accuracy are scoped to those captured sources, before dispatch integration.
- `semantic_results/fc2_quant_memory/t4_v2/`: integrated physical S32768
  results, source/receipt binding and derived summary.json.
- `semantic_results/fc2_quant_memory/local_final/results.json`: all four
  sequence lengths on local logical ranks (24 variant cases), including
  S32768 multi-adapter completion without OOM. Its imported source snapshot
  precedes the final signature/dispatch changes; current-source S513/8192
  binding is separately recorded in `local_binding/results.json`.
- `semantic_results/fc2_quant_memory/ui_dispatch_gate.log` and
  `control_regressions.log`: option/default/session-switch smoke checks and
  12 existing CPU regression tests.
- `benchmarks/fc2_quant_memory.py`, `fc2_quant_candidates.py`,
  `summarize_fc2_quant_memory.py`, `build_fc2_quant_memory_job.py`.

The final source defaults to the lifetime fix and owned FP32 path;
the original full-width numerical geometry remains available as `native`.
