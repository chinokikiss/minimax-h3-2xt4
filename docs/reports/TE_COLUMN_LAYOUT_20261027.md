# TE INT8 column layout — 2026-10-27

Status: local checks passed; one bounded physical-T4 job
[version 1 submitted](https://www.kaggle.com/code/pamcks/h3-te-column-layout-t4-20261027).
**T4 performance qualification is pending. The production default is unchanged.**

## Cause and proposed change

The [previous physical-T4 trace](TE_ATTENTION_TP2_MATRIX_20261027.md) contained
five large INT8 layout copies per TE block: Q/K/V and Gate/Up. Their summed
durations were 7.318/6.792 ms per owner in the separately instrumented S353
Kitchen pass. These sums are evidence of work, not guaranteed removable latency.

`shared_column_linears` shares ConvRot/quantization within QKV and Gate/Up, but
still calls `weight.T.contiguous()` before every individual GEMM. The stored
weight already has contiguous `[N,K]` layout. Its transpose view is `[K,N]`
with column-major strides and shares the same storage. Materializing that view
as another contiguous tensor is avoidable if the GEMM accepts those strides.

The experiment compares three exact-accumulator routes:

| Route | Weight supplied to GEMM | Other differences |
|---|---|---|
| `legacy_copy` | `weight.T.contiguous()` | Current pinned eager helper, including its row padding |
| `torch_view` | `weight.T` | Same pinned eager helper and padding; removes explicit transpose materialization |
| `kitchen_direct` | Original contiguous `[N,K]` | Existing `native_int_mm` Kitchen cuBLASLt wrapper; also avoids Torch row padding |

All routes retain stored INT8 weights, input rotation, row quantization,
FP32 scale multiplication order and FP16 output conversion. The direct route
may select different GEMM geometry; separate INT32 comparisons prevent later
dequantization from hiding a difference. For K5120, even the all-`-128` case
accumulates to 83,886,080, within INT32 range.

The runtime change only extracts `_column_accumulate` as a private column-only
boundary. Its default still calls the original contiguous-transpose expression.
The explicit benchmark swaps this boundary in a scoped context and restores it
on exit. Row projections and Kitchen's global helpers are never patched.
Checks require exactly five column calls per block, so a different dispatch
cannot silently produce a no-op experiment.

No transposed weights are cached. A full cached TP2 column copy would add
`(4096+512+512+12800+12800)*5120 = 157,286,400` bytes, **150 MiB per resident
TE block per rank**. TE's one-use streaming cannot assume that cost is
amortized. A view avoids that storage and introduces no stale-cache or LoRA
invalidation problem; the direct route uses the existing Kitchen workspace.

## Local preview

Hardware/software: RTX 5060 Ti, SM120, local conda Torch 2.12.1+cu132,
Kitchen 0.2.37 and TileLang 0.1.13. The local runtime normally chooses a
different column backend. This preview explicitly forces the pinned eager
columns and uses **Native attention in an unsharded block**. It validates the
candidate paths, but is not a current-default local pipeline speedup and cannot
predict TP2/T4 performance.

Trained block0 weights reconstruct the unsharded block from the two immutable
public rank shards. Inputs are fixed seeded FP16 residuals through that trained
block. Five warm-ups precede nine balanced, interleaved measurements per route.
Compilation/loading, validation, context setup and profiling are excluded from
timing. Every three repetitions balance the routes' measurement positions.

| Tokens | Legacy block, ms | Torch view, ms | Kitchen direct, ms |
|---:|---:|---:|---:|
| 128 | 12.348 | 4.816 | 4.447 |
| 353 | 16.568 | 8.177 | 8.188 |
| 1306 | 39.728 | 30.210 | 29.949 |
| 4096 | 115.087 | 105.979 | 105.514 |

These are wall-time medians of the explicitly forced local control. Native
attention and row projections retain their own local paths, so these values
must not be compared directly with the earlier physical-T4 block matrix.

All three routes produced byte-exact FP16 block and grouped projection outputs
at all four lengths. INT32 accumulators were exact on trained TP2 projection
weights at N512/N4096/N12800, including the non-multiple-of-32 row counts, and
on a separate full-range signed-extreme input. A preliminary seeded raw-GEMM
probe found the TP2 N12800 Torch-view call about 2.4–5× faster than the copy
route locally; that is an operator-only result.

The warmed local S353 trace confirms removal of the intended work:

| Profiled route | Total kernels | INT8 copy kernels | INT8 copy duration sum |
|---|---:|---:|---:|
| Legacy copy | 123 | 7 | 8.836 ms |
| Torch view | 118 | 2 | 0.0248 ms |
| Kitchen direct | 108 | 2 | 0.0252 ms |

Exactly five large transpose copies disappear. The remaining two INT8 copies
belong to other preparation work. These trace sums include overlap and
instrumentation; the paired unprofiled timings determine actual benefit.
The Windows trace contains mangled symbols; the offline analyzer recognizes
the signed-char copy lambda encoding as well as Linux demangled names.

Nineteen CPU tests passed: six new layout contracts, existing shared-column
hook/quantization and output-destination checks, bounded-column tests, and the
eight TE matrix contracts. They cover scope restoration, tensor immutability,
signed extremes, FP16/FP32 projection equality and balanced measurement order.

## Physical-T4 qualification contract

Only trained TE block0 and its operators run. There is no full TE, vision tower,
embedding pass, weight-streaming test, DiT sampling or H3 generation. Both
physical T4 owners execute the actual current TP2 implementation with:

- Kitchen causal attention, FP16 fix and divisor64;
- FP32 SwiGLU/quantization, `fused_stored` and Scaled FP16 row SUM;
- the same Native-style finite additive mask;
- the current NCCL configuration and four AllReduce calls per block;
- S128/S353/S1306/S4096, five warm-ups and nine balanced interleaved samples.

The job measures raw INT8 GEMMs, grouped QKV/Gate-Up calls and the complete
resident block separately. Every candidate must retain exact INT32 accumulators,
byte-exact grouped/full-block outputs and agreement between both TP owners.
Both owners export a minimal CUDA probe and a separate three-route S353 trace;
kernel attribution requires complete CUDA coverage. The analyzer also verifies
four NCCL kernels in every physical block profile. Compilation/loading and
profiling remain outside latency measurements. Peak-memory deltas are measured
relative to this harness's resident tensors, not a complete TE deployment.

The passed frozen bootstrap is reused without upstream or dependency updates.
Kitchen0.2.37 and TileLang0.1.13 stay pinned; Kaggle's matching native Torch/CUDA
family is retained and recorded. A hash-bound supplemental overlay contains
the column-boundary refactor and explicit experiment files. The original
capsule stays unchanged; the other nine relevant runtime/operator hashes must
still match it. New CPU layout contracts run before the operator stage.

Only two public hash-verified rank shards are downloaded, with `token=False`.
There are no dataset/output mounts or credentials. Setup is bounded to 300 s,
the experiment to 900 s and the notebook to 1500 s. The job was submitted once;
no status/list/live-log polling or automatic retries are configured. Final
artifacts have not been collected yet.

The preferred candidate is the simplest route that wins on actual T4 while
passing all exactness/dispatch checks. No default promotion is based on the
local preview. Current findings justify testing a view/direct GEMM before any
persistent weight-layout cache or CPU prepacking scheme.

Evidence: [status and bindings](../data/te_column_layout_20261027.json),
[local raw records and source manifest](../data/te_column_layout_20261027/manifest.json).
Offline reproduction:

```powershell
python -m benchmarks.analyze_te_column_layout --input D:/attention_eval/h3_te_layout_20261027/local_preview_final --output D:/attention_eval/h3_te_layout_20261027/local_summary.json
```

The first local attempt stopped when the production SM120 dispatcher correctly
declined the eager shared path. It was an experiment-selection issue, not an
operator failure; its failed record is retained locally. The successful preview
explicitly labels its forced eager columns/Native attention. Historical evidence
and submitted source bytes are preserved.
