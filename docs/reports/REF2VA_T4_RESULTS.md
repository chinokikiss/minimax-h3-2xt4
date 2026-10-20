# MiniMax-H3 Ref2VA: physical dual-T4 results, version 4

**Stopped, incomplete:** the allocation exhausted its 5,400-second benchmark ceiling and ended with Kaggle status ERROR. Successful rows remain valid; failures are excluded. Recovered and audited 38 accepted measurements and 37 traces. The executed source archive matches the frozen submission. No replacement run has been launched.

Audited measurements (`kaggle_study_output/ref2va_v4/analysis.json`, local-only evidence) · Profiler trace index (`kaggle_study_output/ref2va_v4/TRACE_INDEX.md`, local-only evidence) · Manifest and failures (`kaggle_study_output/ref2va_v4/study_results/manifest.json`, local-only evidence) · Executed source (`kaggle_study_output/ref2va_v4/study_source.tar.gz`, local-only evidence)

## Complete-model latency

Medians of three synchronized samples. Every listed Native/TP/PP single-request output passed pre/post validation **bit-for-bit**. These are complete trained 50-layer executions, not block extrapolations.

| Model | S | Native s | TP2 s | PP2 s | TP speedup |
| --- | --- | --- | --- | --- | --- |
| TE | 512 | 0.848 | 0.659 | 0.877 | 1.29× |
| TE | 2048 | 3.923 | 2.722 | 3.942 | 1.44× |
| TE | 4096 | 9.632 | 6.555 | 9.912 | 1.47× |
| DIT | 8192 | 52.249 | 31.616 | 52.606 | 1.65× |

**TE recommendation:** use optimized TP for the measured single-request prefill workloads. For independent-request throughput, use validated Native placement/dispatch: it is faster than TP in both measured cohorts.

**DiT recommendation:** optimized TP reduces complete S8192 denoiser-step latency from 52.25 s to 31.62 s (1.65×). Full-model S16384/S32768 and DiT throughput were not completed; there is no verified full-model recommendation for those cases.

**PP recommendation:** the single-request scheduler adds overhead and cannot overlap one request across stages. Multi-request TE PP is currently incorrect and must not be used or credited with a speedup.

## Independent-request TE throughput

All requests use S512, microbatch 1. Native and TP execute identical independent requests; Native has no per-request GPU fence.

| Requests | Native cohort s | TP cohort s | Native req/s | TP req/s | PP |
| --- | --- | --- | --- | --- | --- |
| 4 | 2.523 | 2.694 | 1.586 | 1.485 | Rejected: correctness failure |
| 8 | 4.524 | 5.498 | 1.768 | 1.455 | Rejected: correctness failure |

Native already overlaps GPU0 stage work with GPU1 work for other requests. The Native M4 diagnostic trace contains **1,176.6 ms of simultaneous CUDA kernel activity across the two GPUs**. This is evidence of implicit pipeline overlap, not SM occupancy or a component to subtract from ordinary latency. An explicitly threaded PP scheduler must beat this valid asynchronous comparator, not an artificially synchronized serial loop.

## Why legacy TP lost, and what changed

At TE S512, legacy TP takes 28.160 ms/block; optimized TP takes 13.453 ms/block. One profiled rank shows AllGather 5.048 → 0 ms, layout/copy kernels 3.744 → 0.403 ms, and INT8 GEMM 8.128 → 3.882 ms. AllReduce rises from 5.192 to 6.549 ms because the optimized path adds global row maxima. CUDA stream synchronizations fall from six to zero; runtime launch count rises from 59 to 63. The gain comes from cheaper communication/layout/GEMM, not fewer launches.

At DiT S16384, legacy → optimized block latency is 2,958.609 → 2,371.983 ms. Rank-zero profiler kernel sums show AllGather 306.192 → 0 ms and INT8 GEMM 294.849 → 157.863 ms. Attention still dominates (1,956.450 ms in the optimized diagnostic pass). Long DiT sequences therefore benefit from splitting attention even with the remaining communication cost.

The optimized implementation removes length exchanges, scalar synchronization, padded gathers and weight transposition; uses native CUDA INT32 GEMM; and replaces full feature gathers with row-max reductions. FP32 rotation retains Native GEMM geometry to preserve rounding, so some redundant rotation and temporary storage remain. NCCL logs confirm SHM/direct/direct transport; direct peer transport is not claimed. Profile sums can overlap and NCCL kernels include waiting.

## Memory and stage evidence

Peak **PyTorch allocated** memory in GiB, GPU0/GPU1; reserved memory and temporary increments are in the JSON. These are not total driver/NCCL memory.

| Model | S | Native GiB | TP GiB | PP GiB |
| --- | --- | --- | --- | --- |
| TE | 512 | 12.99/11.55 | 12.92/12.92 | 12.99/11.55 |
| TE | 2048 | 13.27/11.84 | 13.22/13.22 | 13.27/11.84 |
| TE | 4096 | 13.65/12.24 | 13.63/13.63 | 13.65/12.24 |
| DIT | 8192 | 12.69/11.31 | 12.80/12.21 | 12.69/11.31 |

PP single-request S512 TE median stage events are 458.5 / 416.9 ms; DiT S8192 stages are 28,402.4 / 24,203.8 ms. Stage 1 includes handoff and heads. Fill/drain are included in wall time. No valid multi-request PP cadence or steady-state throughput was obtained. The ideal bubble fraction is a scheduling reference, not a measured gain.

## Failures and actionable next steps

1. **TE PP throughput failed before timing:** relative L2 0.57455, cosine 0.82242, max absolute error 16,745.75. Single-request placement passed, so the failure is specific to overlap. The local same-device simulation missed this physical behavior. Source ownership during asynchronous transfers is a leading hypothesis, not a confirmed root cause. Reproduce it locally with real copies and delayed consumption between logical streams; verify source activations/metadata stay alive through destination use. Use Native asynchronous placement for TE cohorts while PP is repaired. Before accepting a PP fix, run a short physical correctness check rather than repeating the full benchmark.

2. **One optimized DiT S8192 block job never initialized NCCL:** TCPStore returned EADDRINUSE. Its timing is absent. The complete S8192 optimized TP model and two-request block case passed. Avoid releasing a randomly selected port before spawning: let the parent own a TCPStore and have both workers connect to it.

3. **Legacy DiT S32768 ran out of GPU memory.** Native recorded a valid 17,255.480 ms block median before its job hit the profiling/time limit; optimized TP recorded 9,025.815 ms and passed. These are block results only. No S32768 full-model latency was measured.

4. **DiT Native M4 throughput exhausted the remaining 1,369-second job budget.** No complete, post-validated throughput row was saved; TP/PP throughput and larger DiT full-model jobs were not reached. A follow-up should run only missing cases, reserve their budget before starting, preserve sample progress, and avoid repeating successful TE/DiT latency or block ablations.

## Scope and credibility

TE is text-only decoder prefill on the specified INT8 checkpoint. DiT is one complete native audiovisual Ref2VA step with precomputed conditioning, target video/audio latents, reference image/video packing, native modality tags, AdaLN, RoPE and both output heads. S is the requested total packed token count. Fixture geometry is not an actual workflow token count or a pixel-resolution speed estimate; vision/VAE encoders and sampling loops are excluded. Optional stress cases, SP and VAE were excluded.

Native complete weights require validated two-GPU layer placement. DiT auxiliary weights are staged from immutable CPU tensors; transfer, synchronization and TP preprocessing broadcast are inside the timed forward. Temporary increments include those staged weights. Strict gates are relative L2 ≤ 1e-5, cosine ≥ 0.999999 and max absolute error ≤ 0.005; both TP ranks agree. All 38 accepted rows are bit-exact. Block timings use nine samples; full timings/cohorts use three. These are within-allocation results, not confidence across independent allocations.

Profiler CPU call markers can end before asynchronous GPU work completes. Analysis excludes GPU user annotations from CPU markers and derives kernel-free intervals from each device's own kernel span. Neither CPU submission scopes nor kernel-free intervals establish SM occupancy.