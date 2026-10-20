# MiniMax-H3 TE / DiT: actual dual-T4 parallel study

> Superseded for the prescribed Ref2VA matrix by [version 4 results](REF2VA_T4_RESULTS.md). The tables below describe earlier allocations.

Status: optimized block tests complete; full-model comparisons pending. SP and VAE are excluded. Performance evidence comes exclusively from physical Tesla T4 pairs; local 5060 Ti executions establish correctness only.

[Kaggle run](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-dit-parallel-study). [Measured JSON](kaggle_study_output/optimized/analysis.json), [trace index](kaggle_study_output/optimized/TRACE_INDEX.md), [frozen executed source](kaggle_study_output/optimized/study_source.tar.gz).

## Why legacy TP lost

The baseline trace demonstrates communication and layout overhead, rather than a failure to split attention. TE S512 attention falls from 0.689 to 0.364 ms, but TP adds 5.382 ms of AllGather kernels, 5.046 ms of AllReduce kernels and 3.705 ms of layout/copy kernels. DiT S1024 attention falls from 6.853 to 3.477 ms; its AllGather and AllReduce kernels sum to 20.373 and 10.757 ms. These are durations from one profiled rank, not components to subtract from unprofiled wall latency.

Each row projection gathers lengths, synchronizes four scalar reads per block, pads/copies activations, gathers full feature rows, repeats full-row quantization on both GPUs, transposes weights for `torch._int_mm`, and reduces INT32 accumulators. Native uses different optimized Turing GEMM kernels. At TE S2048 the combined profiled INT8 GEMM time is actually higher in TP (31.123 vs 28.346 ms), despite sharding.

NCCL initialization logs show `NCCL_P2P_DISABLE=1` and `SHM/direct/direct` channels. Peer eligibility is true, but the baseline explicitly disabled NCCL peer transport. This explains the observed transport; it does not establish that every T4 pair lacks P2P.

## Targeted changes

| Change | Purpose and semantic constraint |
|---|---|
| Fixed-size gathers | Remove the length collective, `.item()` synchronization and zero-padding for equal feature shards. |
| Native CUDA INT32 GEMM | Remove weight transposition and padded activation copies. Preserve integer reduction before native dequantization. |
| Global row maximum | Replace full activation AllGather with one FP32 row-maximum AllReduce on native fallback rotation routes. Use native row quantization and discard sentinel columns before GEMM. |
| FP32 rotation geometry | Preserve the full Native rotation GEMM geometry. Local-only FP32 rotation changed accumulation order and failed correctness; zero unowned groups instead. This retains some redundant rotation work. |
| P2P transport experiment | Change the NCCL transport flag independently of the arithmetic; accept only native-equivalent outputs and inspect actual channel logs. |
| Balanced PP scheduling | Use native layers and complete native entry points, two stage streams, independent microrequests, CUDA Events, measured split costs and stream-safe tensor ownership. |

Two local defects were fixed before submission: PP tensors/metadata needed ownership recorded for each consuming stream; FP32 partial rotation required Native GEMM geometry. Full 50-layer TE/DiT TP and eight PP microrequests repeated three times passed bit-for-bit at S512 in the local two-rank simulation. PP simulation used two distinct trained layers with the complete embedding/refiner/packing/head entry points; physical full-50 PP has its own pre/post Native gates.

## Block measurements

Trained checkpoint block 0. TE uses an actual batch dimension; DiT batch 2 means two independent sequential requests because native DiT supports batch 1. No block result estimates full-model latency. PP partitions layers and therefore has no single-block entry in this comparison.

### TE

| Strategy | S | B/requests | Latency ms | Requests/s | Peak allocated GiB/GPU | Correctness |
| --- | --- | --- | --- | --- | --- | --- |
| native | 37 | 1 | 5.792 | 172.67 | 0.473 | bit-exact |
| tp_fixed | 37 | 1 | 4.830 | 207.02 | 0.249/0.249 | bit-exact |
| tp_legacy | 37 | 1 | 14.960 | 66.85 | 0.306/0.306 | bit-exact |
| tp_p2p | 37 | 1 | 4.506 | 221.91 | 0.247/0.247 | bit-exact |
| tp_scalemax | 37 | 1 | 4.624 | 216.25 | 0.247/0.247 | bit-exact |
| native | 512 | 1 | 22.960 | 43.55 | 0.574 | bit-exact |
| tp_fixed | 512 | 1 | 23.460 | 42.63 | 0.374/0.374 | bit-exact |
| tp_legacy | 512 | 1 | 27.767 | 36.01 | 0.426/0.426 | bit-exact |
| tp_p2p | 512 | 1 | 19.340 | 51.71 | 0.344/0.344 | bit-exact |
| tp_scalemax | 512 | 1 | 18.762 | 53.30 | 0.344/0.344 | bit-exact |
| native | 512 | 2 | 29.356 | 68.13 | 0.672 | bit-exact |
| tp_fixed | 512 | 2 | 32.826 | 60.93 | 0.508/0.508 | bit-exact |
| tp_legacy | 512 | 2 | 48.514 | 41.23 | 0.551/0.551 | bit-exact |
| tp_p2p | 512 | 2 | 23.149 | 86.40 | 0.449/0.449 | bit-exact |
| tp_scalemax | 512 | 2 | 23.172 | 86.31 | 0.449/0.449 | bit-exact |
| native | 2048 | 1 | 68.962 | 14.50 | 0.881 | bit-exact |
| tp_fixed | 2048 | 1 | 68.043 | 14.70 | 0.778/0.778 | bit-exact |
| tp_legacy | 2048 | 1 | 100.161 | 9.98 | 0.804/0.804 | bit-exact |
| tp_p2p | 2048 | 1 | 51.361 | 19.47 | 0.659/0.659 | bit-exact |
| tp_scalemax | 2048 | 1 | 51.284 | 19.50 | 0.659/0.659 | bit-exact |

### DIT

| Strategy | S | B/requests | Latency ms | Requests/s | Peak allocated GiB/GPU | Correctness |
| --- | --- | --- | --- | --- | --- | --- |
| native | 37 | 1 | 4.521 | 221.17 | 0.386 | bit-exact |
| tp_fixed | 37 | 1 | 4.532 | 220.63 | 0.219/0.219 | bit-exact |
| tp_legacy | 37 | 1 | 11.123 | 89.91 | 0.249/0.249 | bit-exact |
| tp_p2p | 37 | 1 | 3.918 | 255.20 | 0.213/0.213 | bit-exact |
| tp_scalemax | 37 | 1 | 4.209 | 237.57 | 0.213/0.213 | bit-exact |
| native | 1024 | 1 | 38.550 | 25.94 | 0.695 | bit-exact |
| tp_fixed | 1024 | 1 | 51.930 | 19.26 | 0.789/0.789 | bit-exact |
| tp_legacy | 1024 | 1 | 62.799 | 15.92 | 0.785/0.785 | bit-exact |
| tp_p2p | 1024 | 1 | 35.579 | 28.11 | 0.611/0.611 | bit-exact |
| tp_scalemax | 1024 | 1 | 35.840 | 27.90 | 0.611/0.611 | bit-exact |
| native | 1024 | 2 | 76.856 | 26.02 | 0.759 | bit-exact |
| tp_fixed | 1024 | 2 | 105.445 | 18.97 | 0.852/0.852 | bit-exact |
| tp_legacy | 1024 | 2 | 124.049 | 16.12 | 0.849/0.849 | bit-exact |
| tp_p2p | 1024 | 2 | 65.109 | 30.72 | 0.675/0.675 | bit-exact |
| tp_scalemax | 1024 | 2 | 64.924 | 30.81 | 0.675/0.675 | bit-exact |
| native | 4096 | 1 | 294.404 | 3.40 | 1.638 | bit-exact |
| tp_fixed | 4096 | 1 | 272.154 | 3.67 | 2.553/2.553 | bit-exact |
| tp_legacy | 4096 | 1 | 310.376 | 3.22 | 2.548/2.548 | bit-exact |
| tp_p2p | 4096 | 1 | 197.380 | 5.07 | 1.842/1.842 | bit-exact |
| tp_scalemax | 4096 | 1 | 194.996 | 5.13 | 1.842/1.842 | bit-exact |

The optimized row-maximum variant reduces TE S2048 block latency from 100.161 to 51.284 ms and DiT S4096 from 310.376 to 194.996 ms versus legacy TP in the same allocation. These are block improvements, not full-model speedups. TE S512 Native samples shift from about 23 to 17 ms within the sample window; its 22.960 ms median is not a stable baseline for claiming an advantage over optimized TP. Baseline allocation Native S512 was 16.971 ms.

The P2P-enabled experiment still initialized SHM/direct/direct channels. Its near-equal timings do not demonstrate direct peer transport. The selected full-model variant is tp_scalemax for both models.

Both Native full-model processes reached their 350-second orchestration limit after M4, with no correctness exception. Their completed measurements and traces are preserved below; TP/PP were skipped. The following completion allocation was canceled by the user during setup. A replacement uses the prescribed TE S512/2048/4096 and DiT S8192/16384/32768 matrix with native image/video reference packing; these historical measurements are not results for that matrix.

## Full TE

| Strategy | S | Requests | Purpose / execution | Wall ms | Requests/s | Peak GiB/GPU | Correctness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 37 | 1 | latency / TE native batch | 192.211 | 5.20 | 12.897/11.448 | bit-exact |
| native | 512 | 1 | latency / TE native batch | 1019.453 | 0.98 | 12.989/11.545 | bit-exact |
| native | 512 | 2 | latency / TE native batch | 1881.008 | 1.06 | 13.077/11.637 | bit-exact |
| native | 1024 | 1 | latency / TE native batch | 1976.170 | 0.51 | 13.080/11.640 | bit-exact |
| native | 512 | 4 | throughput / M independent batch-1 requests, microbatch size 1 | 2649.131 | 1.51 | 12.990/11.574 | bit-exact |

## Full DIT

| Strategy | S | Requests | Purpose / execution | Wall ms | Requests/s | Peak GiB/GPU | Correctness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 512 | 1 | latency / independent requests | 1206.347 | 0.83 | 12.230/9.251 | bit-exact |
| native | 512 | 2 | latency / independent requests | 2519.566 | 0.79 | 12.232/9.251 | bit-exact |
| native | 1024 | 1 | latency / independent requests | 2214.767 | 0.45 | 12.369/9.388 | bit-exact |
| native | 512 | 4 | throughput / M independent batch-1 requests, microbatch size 1 | 4627.794 | 0.86 | 12.234/9.252 | bit-exact |

## Timing, memory and limits

Block timings use 3 warmups and 9 synchronized samples. Full latency uses 1 warmup and 5 samples; throughput uses 1 warmup and 3 complete cohorts. Distributed wall latency is the maximum rank, with barriers outside the timer; CUDA Events are archived separately. Profiling is a separate diagnostic pass. Inputs and Native references are identical for each accepted comparison; strict pre/post gates require relative L2 ≤ 1e-5, cosine ≥ 0.999999 and max absolute error ≤ 0.005. Both ranks agree before timing.

Native block timing includes the input clone, with masks/RoPE precomputed equally for all strategies. Full timing includes TE embedding/50 distinct layers and the complete DiT refiner, modality packing, curve interpolation, 50 distinct blocks and both output heads. Outputs are GPU resident; CPU copying, checkpoint loading and sampler/VAE work are excluded. Full weights require two-GPU Native placement; the streamed single-GPU reference validates placement and is never treated as resident latency.

Throughput cohorts execute the same M independent batch-1 requests for Native, TP and PP. TE true-batch-2 latency is shown separately from PP microbatch-1 execution and is not used as its speedup baseline. Full TE is text-only prefill, without vision/KV decoding; full DiT uses synthetic text conditioning and one audiovisual denoising step without reference conditions or denoise masks. The results do not validate arbitrary multimodal payloads, generation quality or end-to-end sampling.

Allocated/reserved peaks, resident allocation and peak temporary increments are in the measured JSON. Kernel duration sums, memcpy times, launches and per-device kernel-free gaps are in the trace analysis. Those gaps are activity-timeline evidence, not SM occupancy; NCCL busy kernels can include waiting. Bootstrap intervals describe within-session samples, not independent allocation uncertainty.

The final TE/DiT recommendations remain pending physical optimized/full-model results. Baseline evidence supports replacing the measured overhead, not claiming an optimization speedup.
