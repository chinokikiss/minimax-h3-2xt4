# Row Partial: retained option and memory lifetime audit

At the historical measurement date, inference kept the Native exact path as default and exposed only
`lora_row_partial_reduce` as an approximate LoRA option, disabled by default.
As of 2026-10-22, the combined FP16-fix/fused-stored/scaled-transport pipeline
defaults to Row A partial reduction with FP32 hidden SUM; explicit False keeps
the Native input gather. See the [current integration follow-up](H3_ROW_PARTIAL_REFERENCE_LOADING_20261022.md).
The measurements below retain their original sources and configuration.
Concat A is deprecated: removed from the UI, and legacy True values are
rejected by the loader/runtime/store. False legacy arguments remain compatible.
Streamed SwiGLU remains removed.

## Allocation audit and implementation

The previous physical S8192 FFN peak was 1792 MiB/rank. Row Partial already
removed the large LoRA input gather, but FP32 base quantization dominated the
remaining peak. It retained local SwiGLU activation, full-width padding and
full rotation storage together while materializing the owned rotated slice.

Two independently benchmarked changes:

1. **Quantizer lifetimes:** release the activated input after copying it into
   full-width padding; perform the same full-shape Native rotation; release
   padding before copying the owned slice; release the full rotation afterward.
   No GEMM geometry, quantization, number of passes or collective changes.
2. **Row Partial lifetimes:** prepare each adapter's A partial and its original
   SUM, then release the owned SwiGLU activation before sequential B/scale/add.
   Hidden staging is capped at 32 MiB and used only when smaller than the
   activation it replaces. Otherwise use the previous per-adapter path.
   A GEMM dimensions, SUM order, B dimensions and scale/add order are preserved.

Temporary buffers remain local to the call. All last reads are queued on the
current CUDA stream before allocator reuse; PyTorch NCCL stream dependencies
are retained. No forced device barrier, persistent buffer pool, extra rotation
pass, rank concatenation or new approximate arithmetic is introduced.

Resident LoRA factors are unchanged from previous Row Partial: 133 MiB/rank
for the complete block's two-adapter factors in the tested stack. Memory work
focuses on activation lifetimes, not another weight precision change. Cache
identity advances to v5; buffer-based CPU/GPU accounting remains intact.

## Local gates completed before submission

- 78 required Native checks passed, including actual blocks 0/25/49 through
  S1024, projections/refiner copies, mixed rank/alpha/strength and S8192 FFN.
- 44 memory-audit cases passed. Native controls match Native bytes; memory
  variants match previous Row Partial bytes, not merely a loose tolerance.
- Eight INT8 payload/row-scale checks match Native exactly at S33/257/513/8192.
- Mixed ranks 7/13, signed strengths and runtime module copying pass. Both
  staged execution and a forced bounded fallback match previous output bytes.
- Existing UI widget order is preserved, Row Partial defaults off, and Concat
  True is rejected at loader, RuntimeConfig and checkpoint-store boundaries.

## Physical validation

Completed [H3 Row Partial Memory](https://www.kaggle.com/code/chinokiki9/h3-row-partial-memory), version 1,
on two Tesla T4 GPUs, torch 2.11.0+cu130 and comfy-kitchen 0.2.37. All runtime
source hashes match the working tree. Physical driver duration: 138.77 seconds;
no OOM, deadlock or runtime failure.

All **60/60** required comparisons passed: **14 Native controls** matched Native
bytes, and **46 Row Partial controls/variants** matched previous Row Partial
bytes, including the adapter-only tests. Thus the memory changes add no observed
numerical drift to the previously accepted approximation. The frozen before
methods were AST-verified against the preceding completed T4 job.

The identical-workload ablation compares previous Row Partial, quantizer-only
lifetime changes, their combined implementation, and a Native exact control.
Whole learned FFNs include single/multi-LoRA at S513/8192 with reversed ordering
rounds, warmup, synchronized CUDA Events/wall times, peak allocation and output
checks. An isolated row-LoRA pass distinguishes its own activation savings
from the base quantizer's full-FFN peak. Profiling uses separate bounded passes.

### Measured S8192 results

Whole learned FFN, identical inputs and single/two-adapter DMAD stacks. Ten
synchronized samples per whole-FFN entry over two reversed-order rounds;
medians use the slower rank per sample. Temporary peaks are above resident
weights and inputs, **not total pipeline VRAM**. Profiler passes and numerical
checks are outside timing. No cross-job latency ratio is used.

| Stack | Configuration | FFN ms | Temporary MiB/rank |
|---|---|---:|---:|
| single | before | 154.733 | 1792.00 |
| single | quant_only | 154.649 | 1348.00 |
| single | row_partial | 154.977 | 1348.00 |
| multi | before | 198.928 | 1792.00 |
| multi | quant_only | 198.662 | 1348.00 |
| multi | row_partial | 198.788 | 1348.00 |

Retained combined changes save **444 MiB/rank (24.8%)** of whole-FFN temporary
peak for both stacks. Single-adapter median changes +0.16%, multi-adapter
-0.07%; treat these as **unchanged latency**, not a speedup. Quantizer-only
changes deliver the full FFN peak reduction. The additional adapter lifetime
change helps its substage but does not reduce the overall FFN peak further.

| Stack | Isolated row LoRA before ms | After ms | Temporary MiB/rank before to after |
|---|---:|---:|---:|
| single | 20.440 | 20.435 | 564.00 to 396.00 |
| multi | 35.115 | 34.904 | 564.00 to 400.00 |

The isolated adapter pass has five samples per entry and includes allocating
its output, while its input and weights are resident. It is not full-FFN timing.
Adapter-only memory falls 29.8% single / 29.1% multi without an observed latency
penalty. After release, its peak occurs during A preparation: 168 MiB output +
224 MiB local activation + 4/8 MiB hidden tensors = 396/400 MiB.

The full-FFN peak now occurs in column-LoRA FC1: 448 MiB packed output +
896 MiB full Native B output + 4 MiB hidden = **1348 MiB**. This explains why
adapter-only savings do not further lower the full peak. Reducing B geometry
or adding tiling could change rounding; those are not part of this change.
The exact Native control peaks at 1512 MiB because it retains the full row-input
gather; Row Partial continues to avoid that gather.

At the quantization boundary, source-level full-FFN live-buffer accounting
drops from 1792 MiB to a predicted 1344 MiB (448 MiB packed input + 448 MiB
padding + 448 MiB full rotation output). This explains the quantizer-only FFN
ablation. No standalone physical quantizer peak was measured, so 1344 MiB is
not presented as a separate benchmark.

Resident adapter bytes remain unchanged: 66.5 MiB/rank/block single and 133 MiB
multi. No additional RAM, weight-memory, PCIe-traffic or throughput gain is claimed.
Eight bounded per-rank traces, raw wall/CUDA Event samples and per-rank absolute
and temporary peaks remain in the downloaded evidence. Collectives, GEMM shapes
and counts are preserved; no new quantization or streamed mode is introduced.

This is selected-block/FFN qualification. Full pipeline and generative quality
are outside the scope of the memory test. Existing Row Partial drift relative
to Native is retained only if every before/after output byte matches.

## Evidence and reproduction

- `semantic_results/h3_row_memory_exact_{single,multi,long}.json`
- `semantic_results/h3_row_memory_local/results.json`
- `semantic_results/h3_row_memory_payload.json`
- `semantic_results/h3_row_memory_controls.json`
- `semantic_results/h3_row_memory_baseline_identity.json`
- `semantic_results/h3_row_memory_t4_v1/`: raw physical results, source/local
  receipts, bounded logs and compressed traces; `summary.json` contains the
  complete A/B data including S513 and per-round medians.
- `benchmarks/row_partial_memory_baseline.py`: frozen previous arithmetic and
  allocation lifetimes, confined to the benchmark.
- `benchmarks/row_partial_memory_audit.py`: local and physical A/B.
- `build_h3_reduction_job.py --row-memory`: source-only upload, local hash gates;
  selected checkpoint ranges download on Kaggle.
- `benchmarks/summarize_row_partial_memory.py`: accepted-result summary.

Previous Concat A and streamed results remain archived, with historical notices.
