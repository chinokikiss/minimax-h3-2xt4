# Physical dual-T4: optional SwiGLU reduction and DMAD LoRA

Run status: **complete**. Torch 2.11.0+cu130, CUDA 13.0, Kitchen 0.2.37.
Two physical Tesla T4 GPUs, one NCCL rank per GPU. Pinned trained H3 block-0 weights and DMAD adapter ranges downloaded by Kaggle.

**Scope:** learned FFN/FC2 latency and compact complete DiT blocks; no full-model, TE, sampling or generation-quality result.

## Learned FFN / output reduction

All variants use the same inputs, Native ConvRot activation scales and GEMM workload. Two fixed forward/reverse rounds; two warmups and five synchronized timing samples per round. Profiler passes are separate.

| S | Strategy | FC2 critical wall ms | FFN critical wall ms | Temporary peak MiB, max rank | Relative L2 | Strict gate |
|---:|---|---:|---:|---:|---:|---|
| 257 | existing/native_int32 | 3.348 | 5.170 | 42.16 | 0 | PASS |
| 257 | streamed/fp32 | 5.150 | 6.576 | 43.76 | 5.58e-08 | PASS |
| 257 | streamed/int8 | 4.467 | 5.797 | 43.76 | 0.0687 | FAIL — diagnostic timing only |
| 257 | streamed/native_int32 | 5.473 | 6.524 | 43.76 | 0 | PASS |
| 512 | existing/native_int32 | 5.246 | 7.133 | 84.00 | 0 | PASS |
| 512 | streamed/fp32 | 6.849 | 8.691 | 45.50 | 5.57e-08 | PASS |
| 512 | streamed/int8 | 5.553 | 7.531 | 45.50 | 0.0691 | FAIL — diagnostic timing only |
| 512 | streamed/native_int32 | 6.622 | 8.557 | 45.50 | 0 | PASS |
| 8192 | existing/native_int32 | 80.962 | 114.493 | 1344.00 | 0 | PASS |
| 8192 | streamed/fp32 | 110.734 | 146.138 | 239.84 | 5.57e-08 | PASS |
| 8192 | streamed/int8 | 95.420 | 134.864 | 239.84 | 0.0691 | FAIL — diagnostic timing only |
| 8192 | streamed/native_int32 | 106.583 | 141.563 | 392.03 | 0 | PASS |
| 16384 | existing/native_int32 | 173.309 | 257.325 | 2688.00 | 0 | PASS |
| 16384 | streamed/fp32 | 230.300 | 318.636 | 463.87 | 5.57e-08 | PASS |
| 16384 | streamed/int8 | 199.332 | 292.825 | 463.88 | 0.0691 | FAIL — diagnostic timing only |
| 16384 | streamed/native_int32 | 225.539 | 312.512 | 784.06 | 0 | PASS |

Temporary deltas are measured above resident inputs/full reference weights/TP shards. Absolute per-rank allocation/reservation peaks and CUDA-event timings are in results.json. They do not represent full-pipeline VRAM residency.

### Original single-GPU Native FFN

| S | Critical wall ms | Native gate |
|---:|---:|---|
| 257 | 7.976 | PASS |
| 512 | 6.339 | PASS |
| 8192 | 132.794 | PASS |
| 16384 | 286.041 | PASS |

Memory guard skipped: S32768. No out-of-memory trial was intentionally attempted.

## Complete block correctness

| Adapter mode | Reduction | Cases passed / total | Worst relative L2 | Worst absolute error |
|---|---|---:|---:|---:|
| base | scalemax/native_int32 | 3/3 | 0 | 0 |
| base | scalemax_streamed/fp32 | 3/3 | 6.29e-08 | 0.000976562 |
| base | scalemax_streamed/int8 | 0/3 | 0.0213 | 32.2256 |
| base | scalemax_streamed/native_int32 | 3/3 | 0 | 0 |
| offline | scalemax/native_int32 | 3/3 | 0 | 0 |
| offline | scalemax_streamed/fp32 | 3/3 | 5.7e-08 | 0.000976562 |
| offline | scalemax_streamed/int8 | 0/3 | 0.0213 | 34.4248 |
| offline | scalemax_streamed/native_int32 | 3/3 | 0 | 0 |
| online_multi | scalemax/native_int32 | 0/3 | 0.000379 | 8.95703 |
| online_multi | scalemax_streamed/fp32 | 0/3 | 0.000379 | 8.95703 |
| online_multi | scalemax_streamed/int8 | 0/3 | 0.0212 | 33.6289 |
| online_multi | scalemax_streamed/native_int32 | 0/3 | 0.000379 | 8.95703 |
| online_single | scalemax/native_int32 | 2/3 | 4.84e-06 | 0.115723 |
| online_single | scalemax_streamed/fp32 | 2/3 | 4.84e-06 | 0.115723 |
| online_single | scalemax_streamed/int8 | 0/3 | 0.0213 | 29.5684 |
| online_single | scalemax_streamed/native_int32 | 2/3 | 4.84e-06 | 0.115723 |

Offline independently merged tensor bytes: 4/4.
Online uses upstream sequential bypass as the reference. Offline uses the independently verified merged-model reference; these are different numerical models.

## Targeted communication evidence

| Strategy | MAX calls | SUM calls | SUM payload dtype | Tensor payload MiB / FC2 invocation |
|---|---:|---:|---|---:|
| scalemax/native_int32 | 1 | 1 | torch.int32 | 168.031 |
| scalemax_streamed/native_int32 | 1 | 1 | torch.int32 | 168.031 |
| scalemax_streamed/fp32 | 1 | 32 | torch.float32 | 168.031 |
| scalemax_streamed/int8 | 33 | 32 | torch.int8 | 42.062 |

### Confirmed cost of the memory-saving path (S8192, rank 0)

| Trace / strategy | GPU kernels | Rotation kernels / activity ms | INT8 GEMM kernels / activity ms | NCCL kernel activity ms |
|---|---:|---:|---:|---:|
| S8192_scalemax_native_int32_rank0.json.gz | 17 | 1 / 8.668 | 1 / 7.947 | 44.485 |
| S8192_scalemax_streamed_fp32_rank0.json.gz | 705 | 64 / 19.640 | 32 / 12.869 | 44.208 |
| S8192_scalemax_streamed_int8_rank0.json.gz | 1121 | 64 / 21.214 | 32 / 12.584 | 17.776 |
| S8192_scalemax_streamed_native_int32_rank0.json.gz | 612 | 64 / 20.311 | 1 / 9.414 | 46.501 |

The existing route uses one large rotation GEMM; the streamed two-pass route repeats rotation across 64 tiles. FP32 additionally splits one INT8 GEMM and one output SUM into 32 each. INT8 reduces SUM payload, but adds 32 scale MAX calls and quantization kernels. These observed extra operations explain the memory/latency tradeoff; smaller communication does not imply lower latency than the original path.
Exact per-trace GPU timestamp intervals are stored in `trace_causality.json`. Activity totals include NCCL waiting and can overlap; they are not isolated link-transfer times. Profiler passes are separate from timing samples.

Payload sums count collective tensor arguments, not physical PCIe/NCCL link traffic. Per-rank compressed Chrome traces and top kernel/CPU summaries are included. CPU/GPU self-time sums can overlap and are not added to critical latency.

## Interpretation limits

INT8 is approximate even when the overflow safety tests pass. Failed gates cannot support a speedup or production recommendation. Online concatenation likewise remains unqualified wherever its strict gate fails.
A single bounded physical run with two ordering rounds screens numerical/memory/performance behavior; full-model multi-request throughput, loading/caching, long-sequence attention and audiovisual quality still need separate validation.

Evidence directory: `C:\Users\Administrator\Documents\project\2xT4 minimax h3\semantic_results\h3_physical_t4_v1`. Source identities, tensor-range SHA256 plans, and per-rank raw samples accompany the run.

## Recommendation

Keep existing exact INT32 for latency when it fits. Use streamed exact INT32 when byte equivalence is mandatory and memory is tight; use streamed FP32 for a smaller temporary peak when the measured tiny numerical difference is acceptable. Both streamed variants are slower in this run.
Keep INT8 manual and experimental: approximately 6.9% FC2 relative error, no material VRAM reduction versus streamed FP32, and no latency win versus existing INT32.
Offline LoRA passed the tested Native byte/block checks. Online single LoRA failed one of three compact shapes and online multi-LoRA failed all three under the strict gate; neither is promoted as universally Native equivalent.
The next targeted optimization to test is a larger, headroom-bounded row tile and fewer launches/collectives. This is a proposal, not a measured improvement. No inference policy or numerical gate was changed during this diagnostic run.
