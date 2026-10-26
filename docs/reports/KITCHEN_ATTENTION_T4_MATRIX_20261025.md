# Kitchen native attention / TileLang T4 matrix — 2026-10-25

**Both physical T4s passed the native correctness gate and completed every timing
case. Sparse native attention has not beaten Kitchen INT8.** Causal masking and
the short D128 query64 dense candidate improved latency in this allocation.
Production operators and policy are unchanged.

## Measured complete-API latency

Milliseconds, excluding compilation and warmup. Each value is the slower GPU of
the two GPUs' medians across the two measurement rounds. All APIs return BND in
the original input dtype. These are seeded synthetic operator inputs.

DiT uses B1/H28/D128 FP16 on each rank, with packed/strided V:

| Case | Kitchen INT8 | TL Group32 dense | TL Drop | TL Scalar Sparse | Native drop | Native pooled |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 0.786 | 1.288 | 1.880 | 2.173 | 1.826 | 3.956 |
| 6,814 | 20.393 | 61.906 | 43.147 | 48.076 | 20.757 | 39.835 |
| 51,220 | 1105.324 | 3726.062 | 2372.244 | 2716.576 | 1416.855 | 2075.353 |

The native route policy retained **76.17%, 63.47% and 51.92% of token pairs** at
512, 6,814 and 51,220 tokens respectively. Dense prefixes were 64, 874 and 874
tokens. TileLang Scalar used its existing split path at 512/6,814 and inline path
at 51,220 under the unchanged 96 MiB score-workspace cap.

| Additional case | Kitchen INT8 | TL Group32 | Native candidate |
|---|---:|---:|---:|
| TE causal, N353/H32/Hkv4/D128 FP32 | 0.838 | 1.131 | 0.522 |
| Video, H32/D64 FP16, Q1024/773 per rank, K1797 | 1.400 | 3.254 | 1.311 |

Use the paired ratios in the receipt for speed comparisons. Ratios from the
rounded slower-GPU table can differ because each backend's slower GPU can differ.

## What the results support

- **Causal masking:** paired speedup over masked Kitchen was 1.41–1.76×, median
  1.46×; over TileLang Group32 it was 2.11–2.20×, median 2.17×. This path avoids
  constructing a full token-pair mask inside the native API.
- **Short D128 dense:** query64 took 0.727 ms,
  versus 0.786 ms for Kitchen. Paired speedup was
  1.06–1.12×, median 1.10×. Against the native original schedule, the median was
  1.045×, with all four paired comparisons faster.
- **Long sparse:** native drop beat TileLang Drop by 1.64–1.74×, median 1.70×.
  Native pooled beat TileLang Scalar by 1.28–1.38×, median 1.34×. Both still lost
  to Kitchen: Kitchen/native ratios were 0.77–0.88× for drop and 0.52–0.55× for
  pooled. These are performance comparisons between different approximation
  contracts, not a quality ranking.
- **Short sparse:** native pooled was slower than TileLang Scalar and Kitchen.
  Sparse preparation and pooled processing did not pay for themselves with
  76% of token pairs retained.
- **Dense cached Q:** it improved the native original schedule by a median 1.8%
  at N6,814, but was a median 1.2% slower at N51,220. It is not a general long
  sequence improvement. The D64 query64 candidate also had no consistent gain
  over the native original schedule, despite fewer compiled registers.

The unmodified-schedule native rebuild itself differs from the installed
Kitchen binary in timing:

| Case | Kitchen INT8 | Native original schedule |
|---|---:|---:|
| DiT 512 | 0.786 | 0.791 |
| DiT 6,814 | 20.393 | 20.582 |
| DiT 51,220 | 1105.324 | 1048.580 |
| Video D64 | 1.400 | 1.301 |

Consequently the entire native-versus-package difference cannot be attributed
to tile selection or cached Q. No source-level cause for the rebuild delta is
established by this benchmark.

## Why Scalar Sparse remains slower

The measured long Group32 dense baseline was 3726.062
ms versus 1105.324 ms for Kitchen, already about 3.4×
slower in matched comparisons. The retained TileLang Group32 implementation
performs separate 32-channel INT8 QK products with FP32 scale application, and
uses FP16 P×V. Kitchen uses its per-thread scale packing and INT8/U8 P×V path.
Dropping roughly half the work does not close that initial execution-cost gap.

Scalar also processes omitted centroids and updates normalization/output state.
At N51,220 its matched complete-call penalty over TileLang Drop had a median
of 442.9 ms. The inline path avoids a multi-GiB precomputed score buffer but
retains serial per-leaf centroid work. These comparisons support the diagnosis
that both the fine Group32 path and pooled processing matter.

The native routed path currently uses query64/key64 tiles with sequential
selected K/V loading and synchronization. Long dense Kitchen uses key128 and
the original staging schedule. The native pooled path adds fine LSE writes,
centroid preparation and a one-warp centroid/merge kernel; its matched penalty
over native drop was a median 676.9 ms at N51,220. Schedule/reuse loss and pooled
processing are therefore the next optimization targets. This is a source-based
interpretation of complete-call timings, not a hardware profiler attribution.

Peak extra allocated memory at N51,220 was 877.5 MiB for Kitchen, 1,423.0 MiB
for native drop/pooled, 1,812.4 MiB for TileLang Drop and 2,162.3 MiB for
TileLang Scalar. Native pooled reduced measured temporary allocation versus
TileLang Scalar by about 34%, while remaining above Kitchen. Both long pooled
implementations avoid the uncapped token-by-centroid score matrix. These are
operator allocations without model residency; reserved memory is not included.

## Validation and numerical limits

The Kaggle CPU preflight passed 70 preserved checks plus 21 experiment checks,
then built every native SM75 specialization with nvcc 12.8.93 on CPU. Both T4s
passed 31 recorded numerical comparisons each, plus compact-list integrity,
empty-row, density-one identity and fixed-packed future-V isolation assertions.
Dense candidates matched the tested installed-Kitchen baseline bytes. Pooled
reference relative L2 errors on the three small gate fixtures were about
0.00080, 0.00451 and 0.00203. Long timed outputs were checked for layout, dtype
and finiteness; no long dense mathematical oracle or generation test ran.

Native pooled is a new approximation in Kitchen's quantized operand space;
it is not numerically equivalent to TileLang Scalar. Its fine probabilities/V
use Kitchen's INT8 contract and its coarse path uses FP16 with FP32 state.
Sparse causal pooling is rejected. Causal score masking passes the tested
offset/GQA/ragged cases, but the preserved quantizer can share future-dependent
scales/anchors: strict floating-input streaming prefix invariance is not claimed.

## Method and provenance

- Physical devices: two Tesla T4s, SM75, distinct UUIDs recorded in the receipt.
- Kaggle native Torch 2.11.0+cu128 / CUDA 12.8 / NumPy 2.1.3; exact
  comfy-kitchen 0.2.37 and TileLang 0.1.13. Frozen ComfyUI legacy API retained.
- Five samples after two warmups per round, two reversed-order rounds per GPU.
  First-call/JIT timings, all samples, CUDA-event and host times, pruning and
  allocated-memory baselines are preserved. CUDA events bracket the full API
  and can include stream gaps caused by host dispatch. No fixed clocks assumed.
- Identical raw seeded Q/K/V across APIs, with seed 20261025. Each sparse API
  executes its own preparation using the same tau0/local/global/prefix policy.
  No real activations, weights, model pipeline or TP communication were used.
- Native dense query64/key128 was withheld after the exact build showed spills;
  the other candidates passed the physical gate before timing.
- Immutable source commit:
  [`f8fffbfd0e3cff2138e1b2ebd4cae02c248d969f`](https://github.com/chinokikiss/minimax-h3-2xt4/commit/f8fffbfd0e3cff2138e1b2ebd4cae02c248d969f).
  Distribution SHA256: `3e9947c02cf3452fccc90297c3018d6d37cf8a6e186b85a0d88759e79b985273`.
  Capsule SHA256: `f85e6eed35f96ba00328a5f96ebbe10c0381981f29b35a41699418e26bcde275`.
- Private Kaggle CPU version 3 and GPU matrix version 2. GPU execution completed
  under its 1,100-second bound in 427.55 seconds. Final GPU outputs were
  collected once after the fixed window; no status or live-log polling.
- The first GPU attempt failed at CUDA compilation before any timings; its
  artifacts are preserved. The fix supplies standard assert declarations before
  the frozen headers. Source/header/library pins were not upgraded. A CPU final
  artifact download had one TLS handshake failure with zero artifacts; one
  bounded transport retry retrieved the final output, with both records retained.

Evidence root: `D:/attention_eval/kitchen_native_matrix_20261025_r4`.
Raw batch SHA256: `52aefa08a6ac0b52d0b9f1b946f539f99effce0fea4cdeaba155ec6cedaa00f2`.
Machine-readable [receipt](../data/kitchen_attention_t4_matrix_20261025.json).
Implementation [README](../../experiments/kitchen_attention/README.md).
The earlier [prototype report](KITCHEN_NATIVE_ATTENTION_20261025.md) and its local
receipt remain historical; this report supplies physical T4 execution evidence.

## Full timing table

| Shape | Backend | GPU ms | Host ms | Peak allocated GiB |
|---|---|---:|---:|---:|
| dit_6814 | kitchen_int8 | 20.393 | 20.476 | 0.576 |
| dit_6814 | native_dense_baseline | 20.582 | 20.673 | 0.576 |
| dit_6814 | native_dense_cache_q | 20.143 | 20.235 | 0.576 |
| dit_6814 | native_drop | 20.757 | 20.857 | 0.615 |
| dit_6814 | native_pooled | 39.835 | 39.909 | 0.646 |
| dit_6814 | tilelang_drop | 43.147 | 43.255 | 0.667 |
| dit_6814 | tilelang_group32_dense | 61.906 | 62.048 | 0.620 |
| dit_6814 | tilelang_scalar | 48.076 | 48.186 | 0.744 |
| dit_long_51220 | kitchen_int8 | 1105.324 | 1105.454 | 4.065 |
| dit_long_51220 | native_dense_baseline | 1048.580 | 1048.722 | 4.065 |
| dit_long_51220 | native_dense_cache_q | 1058.460 | 1058.582 | 4.065 |
| dit_long_51220 | native_drop | 1416.855 | 1416.985 | 4.593 |
| dit_long_51220 | native_pooled | 2075.353 | 2075.455 | 4.598 |
| dit_long_51220 | tilelang_drop | 2372.244 | 2372.359 | 4.973 |
| dit_long_51220 | tilelang_group32_dense | 3726.062 | 3726.142 | 4.625 |
| dit_long_51220 | tilelang_scalar | 2716.576 | 2716.699 | 5.320 |
| dit_short_512 | kitchen_int8 | 0.786 | 0.831 | 0.049 |
| dit_short_512 | native_dense_baseline | 0.791 | 0.846 | 0.049 |
| dit_short_512 | native_dense_q64 | 0.727 | 0.769 | 0.049 |
| dit_short_512 | native_drop | 1.826 | 1.864 | 0.054 |
| dit_short_512 | native_pooled | 3.956 | 4.050 | 0.054 |
| dit_short_512 | tilelang_drop | 1.880 | 1.930 | 0.058 |
| dit_short_512 | tilelang_group32_dense | 1.288 | 1.339 | 0.054 |
| dit_short_512 | tilelang_scalar | 2.173 | 2.222 | 0.061 |
| te_causal_353 | kitchen_int8 | 0.838 | 0.897 | 0.029 |
| te_causal_353 | native_causal | 0.522 | 0.561 | 0.029 |
| te_causal_353 | tilelang_group32_dense | 1.131 | 1.170 | 0.026 |
| video_short_d64 | kitchen_int8 | 1.400 | 1.439 | 0.058 |
| video_short_d64 | native_dense_baseline | 1.301 | 1.345 | 0.058 |
| video_short_d64 | native_dense_q64 | 1.311 | 1.356 | 0.058 |
| video_short_d64 | tilelang_group32_dense | 3.254 | 3.308 | 0.073 |

Native qualification: qualified.
These backends have different quantization/approximation contracts; timings do not rank generation quality.
