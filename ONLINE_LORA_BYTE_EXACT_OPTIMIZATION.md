# Byte-exact online LoRA: accepted storage optimization

> Historical benchmark record: streamed mode has since been removed from inference. Current optional Row Partial is documented in [ROW_PARTIAL_MEMORY.md](ROW_PARTIAL_MEMORY.md); Concat A is deprecated.

Retained after local qualification and physical **2 x Tesla T4** A/B testing.
The default `scalemax` and optional `scalemax_streamed` both retain Native INT32
reduction. FP32-partial and INT8 reduction are not reintroduced.

## Changes and numerical invariants

- Replace tensor-list AllGather plus concatenation with one rank-major
  `all_gather_into_tensor` buffer, then restore contiguous Native feature order.
  Release the local activated tensor before that layout conversion.
- Keep the full Native B GEMM. Scale and add only the owned halves of packed
  QKV/gate-up through one strided view, avoiding discarded-row writes and
  separate additions per packed component. Release that view before the next
  adapter allocates its B output.
- Preserve full A/B GEMM shapes, per-adapter source order and scaling after B.
  No A concatenation, scale folding, partial-A SUM, sharded-B geometry, or GEMM
  tiling is used. Adapter cache identity advances to v3.

Current-stream NCCL dependencies protect buffer reads before layout conversion;
no device-wide synchronization or persistent activation pool is added. Factors
remain complete and replicated. Host RAM and checkpoint reads do not decrease.

## Correctness and measurement

**106/106 local checks** passed before each physical submission, viewing tensors
as bytes (including signed zero). These cover single/multi-LoRA, base/adapted
blocks 0/25/49 at S33/257/512/1024, real projections and copied refiner modules,
learned FFNs through S8192, unequal ranks, nontrivial alpha and negative/zero
strengths. Logical-rank timings are not performance evidence.

**104/104 physical checks** passed in the accepted version 2: 72 complete-block
comparisons and 32 FFN A/B comparisons. Both ranks agreed. Runtime source hashes
match the current files. No OOM, deadlock or runtime failure occurred. Runtime:
torch 2.11.0+cu130, comfy-kitchen 0.2.37. Physical driver: 124.66 seconds.

Job: [H3 Native Order LoRA Storage](https://www.kaggle.com/code/chinokiki9/h3-native-order-lora-storage), **version 2**.

Performance compares the previous **correct** implementation with the candidate
in the same process, using identical inputs, adapter stacks and resident weights.
Ten synchronized samples per cell span two forward/reverse ordering rounds.
Numbers below are medians of the slower rank per sample. Warmup, byte validation
and profiler passes are outside the timing boundary. CUDA Event and wall times
are both retained. Temporary peaks are allocations above resident inputs/weights,
**not total pipeline VRAM**; absolute per-rank peaks remain in the raw results.

## Accepted same-run results

| S | Stack | Mode | Exact baseline ms | Optimized ms | Latency reduction | Temporary MiB/rank before to after |
|---:|---|---|---:|---:|---:|---:|
| 513 | multi | scalemax | 16.412 | 15.690 | 4.40% | 112.22 to 112.22 |
| 513 | multi | scalemax_streamed | 18.287 | 18.260 | 0.15% | 108.71 to 94.68 |
| 513 | single | scalemax | 11.978 | 11.443 | 4.47% | 112.22 to 112.22 |
| 513 | single | scalemax_streamed | 14.476 | 13.971 | 3.48% | 108.71 to 94.68 |
| 8192 | multi | scalemax | 274.594 | 262.163 | 4.53% | 1792.00 to 1792.00 |
| 8192 | multi | scalemax_streamed | 301.089 | 290.231 | 3.61% | 1736.00 to 1512.00 |
| 8192 | single | scalemax | 204.041 | 195.045 | 4.41% | 1792.00 to 1792.00 |
| 8192 | single | scalemax_streamed | 232.329 | 223.495 | 3.80% | 1736.00 to 1512.00 |

At S8192, FFN median latency decreases **3.6-4.5%**. Streamed temporary memory
falls **224 MiB/rank (12.9%)** for both stacks. Default-mode peak is unchanged:
base quantization/rotation still dominates its peak. Optimized streamed mode
saves 280 MiB/rank versus optimized default but remains slower. Keep default
for latency and select streamed/Native INT32 when that headroom is needed.

S513 multi-LoRA streamed improves only 0.15%: treat this as **no proven latency
benefit**. This is a bounded two-round study, not a confidence interval. At
S8192, all four mode/stack configurations improve their medians in both execution
orders; individual sample ranges can overlap. Absolute cloud timings varied
across versions. No cross-job latency ratio is used as evidence of speedup.

## Rejected candidate and lifetime fix

Version 1 passed all 104 physical byte checks and improved S8192 latency about
3%, but was rejected for multi-adapter memory regression. The owned output view
retained the preceding full B buffer into the next adapter's B GEMM:
448 MiB base output + 2 x 896 MiB B outputs + 4 MiB hidden = **2244 MiB**.
Explicitly deleting the view before the next adapter restored the intended
lifetime. Local checks passed again before version 2. The accepted physical
multi-adapter streamed peak is **1512 MiB**, with no memory regression in any
of the eight measured mode/stack/sequence combinations. Both versions' raw
results remain available; the rejected candidate is not presented as retained.

## Trace evidence and remaining cost

Eight compressed per-rank Chrome traces and compact summaries are preserved.
For S8192 multi-LoRA streamed rank 0, baseline versus accepted candidate:

- `aten::add_` calls: **6 to 4**; `aten::copy_`: **258 to 257**; `aten::cat`:
  **33 to 32**. These totals include the streamed quantizer's repeated work.
- GPU GEMM count: **75 to 75**; NCCL count: **3 to 3**. Other kernels:
  **559 to 557**. The Native linear computation remains intact.
- Other-kernel activity: **77.812 to 70.531 ms**; NCCL kernel residence:
  **62.975 to 62.912 ms**. The evidence supports less pointwise/copy work,
  rather than less communication, as the improvement.
- Baseline AllGather: trace timestamp **1434445958840.292 us**, duration
  **37.542 ms**. Candidate: **1434451002948.356 us**, **37.490 ms**. These
  identify events in separate trace passes, not elapsed time between variants.
  All per-rank collective intervals are in the compact summary.

Kernel activity can overlap and is not a critical-path sum. NCCL residence
includes rank-arrival waits and is not pure transfer-bandwidth evidence. The
S8192 input gather still has a **224 MiB/rank** FP32 payload; the INT32 SUM
payload remains **168 MiB/rank**. Payloads are not measured PCIe link traffic.

Exact online LoRA still pays for full Native A inputs and full-shape B GEMMs.
Do not restore rank concatenation, folded scales or partial-A reductions to
chase larger speedups: earlier ablations demonstrated downstream INT8 threshold
changes. Further memory work should first shorten base quantization lifetimes
while preserving its GEMM geometry, under a separate byte gate and physical A/B.

## Scope, evidence and reproduction

This validates selected blocks and learned FFNs, **not end-to-end generation**,
all 50 blocks, or original DMAD quality on the different public Ref2VA backbone.
No full-pipeline speedup is claimed.

- Local receipts: `semantic_results/h3_lora_storage_{single,multi,long}.json`.
- Physical evidence: `semantic_results/h3_lora_storage_t4_v{1,2}/`.
- Accepted summary, per-rank peaks and trace intervals:
  `semantic_results/h3_lora_storage_t4_v2/summary.json`.
- `benchmarks/native_order_lora_baseline.py`: frozen previous exact update;
  AST identity verified against the previously completed repair notebook.
- `benchmarks/physical_h3_lora_storage.py`: identical-workload physical A/B.
- `build_h3_reduction_job.py --storage`: source-only submission; rejects stale
  or failing local proof hashes. Selected checkpoint ranges download on Kaggle.
- `benchmarks/summarize_h3_lora_storage.py`: paired result/trace summary.

The two evidence directories together occupy about 2 MiB. Profiling is confined
to explicit benchmark passes; no hooks, tensor dumps or new logging are added
to normal inference. The default inference strategy remains unchanged.
