# TP2 P2 operator investigation

Status: **operator investigation complete, 2026-10-18**. All nine submitted Kaggle versions are terminal. Eight benchmark/source snapshots, 28 exported Nsight Systems reports, 2,400 measured event/wall sample pairs across both ranks, 70 fresh-process output-identity checks, and all rejected/failed records are retained in `TP2_P2_RESULTS.json`. A Kaggle notebook marked COMPLETE can contain a failed experiment stage; the internal status and logs take precedence.

The strongest new attention result is omitted-leaf scalar-score preparation with FP16 PV: **9.74% / 11.25% lower complete-API latency**, with bitwise-identical outputs on the two real captured rank fixtures, at **78.3MiB additional workspace per rank**. Dense FP16 PV with a 128-thread mapping gains 2.97% / 3.98%; dense INT8 PV with that mapping gains 9.62% / 11.47% while introducing another approximation. Compressed row transport wins, but real cancellation and variable FC2 timings prevent a production-quality claim. No new approximation is enabled by default.

The experiment keeps native INT32 SUM and Group32 QK / FP16 PV as the numerical controls. Production dispatch is not changed by these patches. The target is two physical Tesla T4 GPUs, SM75, P2P disabled, four NCCL channels, default NCCL protocol selection.

## Inputs and reproducibility

The current TP2 schema was audited again: DiT O projection is M=6814, N=5376, local K=3584; FC2 is M=6814, N=5376, local K=7168, with local packed gate/up width 14336. Attention uses 28 heads per rank, N=6814, D=128, leaf64 and the existing prefix874/tau0 policy. Video remains sequence-sharded; this investigation does not substitute the old head-sharded Video layout.

Two real row workloads are prepared: layer25 O projection at denoising call9 and layer0 FC2 at the first denoising call. O input comes from the saved full native attention output. FC2 capture reuses the original prompt conditioning, noise, sigmas and complete native preprocessing, and stops before the first DiT FC2. A shape guard excludes the token refiner, whose FFN has the same width but only 353 rows.

The raw local real fixtures also contain O calls0/19. These endpoints are not in the previously published TP2 Kaggle fixture subset and are excluded from target accuracy claims. There is no claim of all-layer/all-step coverage.

To avoid uploading 1.56GB of duplicated captures, the new private dataset contains a 36.7MB real FC1 INT8 operand and activation scales, plus manifests. Selected learned FC1/FC2/O weights are fetched as pinned public checkpoint ranges and SHA256 checked. Native cuBLAS INT32 GEMM plus native dequantization must reproduce the original FC2-input SHA256 on T4 before measurements. The O capture has its original SHA256 checked independently. This is lossless fixture reconstruction, not a random substitute or an inference that two hardware distributions are equivalent.

The CPU input audit uses exhaustive maxima/RMS and explicitly sampled percentiles. Native O inputs have row max/RMS ratios up to 15.64 across the local trajectory endpoints; the measured call9 fixture reaches 13.84. The raw first FC2 gate/up input reaches 32.41. These values describe inputs, not output-error tolerances.

The shared checkout changed concurrently during preparation: retired streaming code and allocation lifetimes in `optimized_tp.py` changed. Experiment dependencies were therefore frozen from the completed previous T4 attention-repair submission. The current shape audit matches the historical GEMM/attention schemas. New experimental sources are version-bound separately. This suite does not silently benchmark a moving production implementation.

Pinned target software: Torch2.11.0+cu130, comfy-kitchen0.2.37, TileLang0.1.15, nvcc12.8.93. Provider notebook/image identities and dataset versions are recorded with each collected version. Local SM120 runs are arithmetic previews only. Windows CUDA13.3 / TileLang0.1.13 offline SM75 cubins are static code-generation evidence only.

Each comparable run uses one second of local clock prewarming, ten distributed warmups and fifteen CUDA-event plus synchronized-wall samples. A fresh process reverses candidate order. The primary metric is the slower rank's complete-API median. Both ranks, every sample and dispersion are retained. The extra FC2 confirmation uses 30 interleaved blocks with rotating order after ten warmups per candidate; its fresh process reverses the base order. This is a separate scheduling experiment, not an interchangeable absolute timing sample. JIT, dependency setup, fixture materialization and correctness work are outside steady-state timing.

## Transport and reconstruction contract

The immutable controls are native INT32 SUM, scaled FP16 SUM and packed INT8 AllGather followed by INT32 reconstruction and native dequantization. New fused epilogues remove the reconstructed INT32 allocation. The fused caller also releases the original INT32 partial after encoding and before allocating receive/output buffers.

FP16 uses the existing power-of-two shift: 11 for O and 12 for FC2. The fused epilogue scales the reduced FP16 payload back, performs the specified INT32 cast, then casts to FP32 and applies activation scale followed by per-output weight scale. Packed INT8 keeps group128 absmax scales in the same little-endian payload, reconstructs rank0 and rank1 products with round-to-nearest FP32 operations, rounds the sum to INT32 with ties-to-even, and then performs the same two FP32 scale products. Skipping the INT32 reconstruction rounding would be a different method.

The supported fused contract is the current no-bias FP32 O/FC2 boundary with vector weight scales. It does not claim scalar-scale, bias, residual, FP16-output or arbitrary-shape support. Kitchen0.2.37's actual no-bias vector dequantization source was inspected at tag `v0.2.37`, commit `be003b7c23c5b01328657955b8bc5d3f073d868e`.

The reconstruction allocation is 146,528,256 bytes (139.74MiB) per rank. Avoiding its write and read removes 293,056,512 bytes of nominal epilogue traffic. This is a byte-accounting statement, not measured DRAM traffic or a guarantee of the same reduction in whole-operator peak VRAM. The full-width FP32 rotation can determine the API peak before transport starts.

Independent gates exercise random values, zeros, worst-case bounds, cancellation, tie-adjacent integers and outliers. The fused result must match independent reconstruction followed by native dequantization bit-for-bit. Real diagnostics include full-output RMSE/max/cosine, absolute-error percentiles, endpoint/clamp/overflow counts, near-zero absolute error and error conditioned on `abs(p0+p1)/(abs(p0)+abs(p1))`. The near-zero rule is global reference RMS times 1e-4; individual tiny outputs are not denominators for a misleading pointwise relative-error score.

Peak allocated/reserved memory is measured separately from latency, after releasing GPU correctness references and emptying the allocator cache. The measurement includes the live input/weights and output; PyTorch counters exclude NCCL's internal allocations. Single-session `nvidia-smi` memory telemetry is a separate whole-process observation. NCCL payload size is not peak VRAM or physical PCIe traffic.

## Attention causal matrix

Group32 QK, prepared operands, routes, output layout and leaf geometry remain fixed. Dense candidates are the FP16-PV control, existing block-local INT8 PV, a 128-thread mapping of the same 64-query tile, a running-maximum probability normalization variant and a unit-restoration diagnostic. Prepared-main timings exclude V quantization; complete API timings include it.

The unit-restoration diagnostic deliberately removes V-scale restoration from the numerator. It is not a usable attention method or a quality candidate. Its timing difference includes changed register lifetimes, loads and instruction scheduling; it cannot assign a scalar multiply an independent hardware duration. Running-maximum normalization is a separate approximate numerical method, with an independent Torch reference, rather than an equivalence optimization.

The scalar candidates retain the existing FP16-PV and INT8-PV paths and add batched FP32 centroid-score precomputation. The split path leaves routing, log-count weighting, online state updates and FP16 pooled-value probabilities unchanged. It removes raw-query centroid reduction state from the main kernel, at the cost of another kernel and 82,066,432 bytes of score workspace per rank. FP32 reduction layout can change rounding, so small independent score tests and same-method full-output gates are required. Sparse-versus-dense output L1/L2 is not a quality gate.

Sorted Compact Drop is not retried. Group64/128, V-quantizer tuning and broad tile searches are not included without evidence of a bottleneck.

## Profiling interpretation

Nsight Systems captures selected complete APIs with NVTX scopes for quantization, GEMM, codec, NCCL, reconstruction/dequantization, routing, V quantization, centroid scores and the fused attention main kernel. CPU enqueue ranges are asynchronous and must not be read as synchronized API latency. CUDA runtime allocation/synchronization records and warmed wall samples provide complementary CPU-overhead evidence.

Existing Kaggle Nsight Compute probes returned `ERR_NVGPUCTRPERM`. This suite does not pretend that registers or shared memory measure achieved occupancy, spills, memory stalls or Tensor Core utilization. Offline SASS confirms SM75 INT8 MMA mapping and inventories conversions/barriers, but its compiler/version differs from target execution. Target launch resources come from Systems. Static resource limits and performance hypotheses are labeled separately from measured facts.

## Qualification and dispatch

All approximate transports remain unqualified for H3 generation. Full weights exist locally, but the local machine has one SM120 GPU. The bounded T4 operator jobs attach operator fixtures and selected learned weights, not the full TP2 generation runtime/checkpoint set. Prior native/attention PP clips do not qualify changed TP2 transports. No new end-to-end latency, visual, temporal, semantic or audio equivalence is inferred from row errors, arithmetic identity or previous videos.

Proposed policy: retain native INT32 as the production default and Group32/FP16 PV as the attention baseline. If a transport mode later passes full TP2 generation qualification, use its exact-contract fused epilogue when it wins repeated complete-API measurements. Evaluate FP16 FC2 and FP16 versus INT8 O as explicit per-site choices. Scalar pooling remains optional. Do not automatically select INT8 PV or approximate transport from an isolated best run.

## Real transport results

The following are **complete APIs**, including global activation quantization/rotation, the local learned-weight INT8 GEMM, actual NCCL and the final epilogue. Times are slower-rank CUDA-event medians in milliseconds. Two columns are independent processes, with reversed candidate order; they are not best-run selections.

| Site | Mode | Screen | Reverse repeat | Repeat reduction vs native | Peak allocated MiB, repeat | Output relative RMSE |
|---|---|---:|---:|---:|---:|---:|
| O layer25/call9 | Native INT32 | 29.6356 | 29.8958 | — | 497.4 | 0 |
| O | FP16 unfused | 23.3799 | 23.7675 | 20.50% | 544.3 | 0.03056% |
| O | FP16 fused | 22.3827 | 22.1199 | 26.01% | 497.4 | 0.03056% |
| O | Packed INT8 unfused | 20.1144 | 20.1789 | 32.50% | 544.3 | 0.68419% |
| O | Packed INT8 fused | 20.0447 | 20.1953 | 32.45% | 497.4 | 0.68419% |
| FC2 layer0/call0 | Native INT32 | 42.2337 | 43.6552 | — | 1213.7 | 0 |
| FC2 | FP16 unfused | 39.8774 | 40.7470 | 6.66% | 1213.7 | 0.02922% |
| FC2 | FP16 fused | 38.0116 | 40.5544 | 7.10% | 1213.7 | 0.02922% |
| FC2 | Packed INT8 unfused | 36.8157 | 40.1592 | 8.01% | 1213.7 | 0.66749% |
| FC2 | Packed INT8 fused | 36.0028 | 38.1339 | 12.65% | 1213.7 | 0.66749% |

Every fused full-shape result is bitwise equal to its independently reconstructed INT32/native-dequantization reference. Both rank replicas and fresh-process output hashes match. Exact agreement with an approximate codec reference does **not** make that codec equivalent to native INT32.

O FP16 fusion improves over its unfused control by 4.27% and 6.93%. FC2 packed-INT8 fusion improves over its control by 2.21% and 5.04% in this original matrix, but fails to win consistently in the later paired confirmation below. O packed fusion has no reproducible latency advantage: +0.35% then -0.08%, while both repeats reduce peak allocation by approximately 46.9MiB. FC2 FP16 fusion has a much smaller second-process benefit, 0.47%; the later interleaved confirmation also shows a varying effect size. Neither FC2 fusion has a robust isolated latency gain established by this suite.

All FC2 variants peak at approximately 1213.7MiB because the full-width FP32 rotation dominates earlier. The removed 139.7MiB reconstruction buffer therefore does not translate into a lower full-API FC2 peak. O fused/native peaks are approximately 497.4MiB versus 544.3MiB for unfused compressed paths. Reserved peaks and post-output-release allocations are retained per rank in JSON. These are PyTorch allocated/reserved measurements, not total driver/NCCL VRAM.

The earlier synthetic speedup percentages are not reproduced automatically: the primary real FC2 result is 7.10% for fused FP16 and 12.65% for fused INT8 versus native. The later interleaved confirmation spans 20.39–22.96% for fused FP16 and 23.36–29.66% for fused INT8. Report all of these rather than choosing the result closest to the earlier 25–29% claim. Cross-job timing differences are not a controlled distribution ablation; no causal attribution to learned weights, topology or clock differences is asserted without matching evidence.

### Cancellation and outliers

| Site / codec | Global relative RMSE | Max absolute error | Absolute-error p99.9 | Most severe cancellation bin: error RMS / reference RMS |
|---|---:|---:|---:|---:|
| O FP16 | 0.03056% | 1.9695 | 0.2011 | 0.01826 / 0.008454 = **2.16×** |
| O INT8 | 0.68419% | 8.7654 | 2.7800 | 0.74243 / 0.008454 = **87.82×** |
| FC2 FP16 | 0.02922% | 10.0801 | 0.3391 | 0.007710 / 0.003552 = **2.17×** |
| FC2 INT8 | 0.66749% | 91.9651 | 7.1307 | 0.47872 / 0.003552 = **134.77×** |

The severe bin is `abs(p0+p1)/(abs(p0)+abs(p1)) < 1e-4`: 2445 O entries and 2291 FC2 entries out of 36,632,064. Its large relative errors are accompanied by absolute RMS values above; they are not unstable pointwise divisions. The separate near-zero-output test finds 67,550 FC2 outputs below 0.00810 in magnitude, with worst absolute error 0.00834 for FP16 versus 8.739 for INT8. O near-zero outputs have worst errors 0.1355 versus 2.828.

No FP16 overflow or INT8 clamp occurs. INT8 endpoint use is approximately one value per group, as expected from absmax scaling, and is not counted as clipping. Thus the observed degradation comes from lossy rounding of independently represented partials and cancellation after transport, rather than saturation. Actual FC2 partial absmax is 4,230,846 versus p99.9 566,790; learned scale/input outliers also make absolute output tails much larger than the synthetic report suggested. A high cosine or low aggregate RMSE cannot qualify full-model safety.

### Transport profiling and CPU behavior

Rank0 mean kernel intervals over three profiled calls show native INT32 SUM around 18.9ms, FP16 SUM around 9.6–10.4ms and packed AllGather around 6.1–6.6ms. These NCCL intervals include peer waiting; they are not physical-link-bandwidth measurements.

For O, unfused FP16 decode plus dequantization takes approximately 1.016+1.217ms; the fused epilogue takes 1.023ms. Packed reconstruction plus dequantization takes approximately 1.075+1.224ms versus 1.088ms fused. FC2 packed reconstruction/dequantization similarly changes from 1.256+1.209ms to 1.236ms. One epilogue launch is removed in each compressed fused path.

The unchanged FC2 work remains substantial: SwiGLU/padding ~3.3ms, full-width rotation ~8–10.7ms, GEMM ~8–9.3ms, plus maxima/quantization and synchronization. This explains why halving or quartering the transport payload cannot produce a corresponding full-API speedup.

O packed fusion illustrates why the epilogue cannot be selected alone. In its rank1 profiles, reconstruction+dequantization falls from about 2.222ms to 1.079ms, but MAX-collective time rises from 0.275ms to 0.945ms, rotation from 3.510ms to 3.875ms and GEMM from 3.572ms to 3.763ms. This offsets the local saving. The physical cause of common-phase variation is not identified by Systems; the complete-API repeats correctly refuse a latency win.

O CPU enqueue ranges average roughly 1.1–1.4ms while GPU API latency is 20–30ms. Scalar attention enqueue ranges are roughly 2–3ms. Captured GPU kernel-interval coverage is around 99% or higher; that measures timeline coverage, not SM utilization. Large `cudaDeviceSynchronize` durations drain the queued calls at capture completion and must not be added to kernel times as extra compute. Warmed event/wall samples, CUDA runtime summaries and individual launch counts are retained in the attribution JSON.

## Attention results and controlled follow-ups

V2 dense FP16-PV complete API is 64.8670 / 71.0798ms (screen/reverse). Existing 256-thread INT8 PV regresses to 71.917 / 76.009ms. Changing only that candidate's mapping to 128 threads gives 59.0195 / 62.2242ms: 9.01% / 12.46% faster than the current FP16-PV control. Additional arithmetic RMSE relative to Group32/FP16 PV is 0.666–0.708% across ranks. This is an operator arithmetic diagnostic, not generation quality.

The running-maximum approximation is rejected: 72.477 / 74.692ms and approximately 3.03–3.43% additional arithmetic RMSE. It loses the block-local probability resolution while failing to accelerate the API. No claim of observed semantic failure is made.

V2 scalar APIs: current 53.157 / 53.656ms; existing INT8 PV 51.237 / 51.821ms; full precomputed FP16-PV scores 51.621 / 52.886ms; full precomputed INT8-PV scores 51.556 / 53.012ms. Full precomputation therefore offers limited complete-API ROI and an extra 78.3MiB per rank.

Measured rank0 profiling: dense main 60.334ms (current), 66.943ms (256-thread PV8), 52.868ms (128-thread PV8); V quantization ~0.322ms. The regression is in the main kernel, not the standalone V quantizer. Scalar main changes from 41.569ms (current) to 39.791ms (PV8), 33.370ms (FP16-PV precomputed scores), or 34.346ms (PV8 precomputed scores), but new score preparation costs 5.7–6.1ms. Existing scalar preparation costs ~8.9ms across 23 kernels, and routing ~1.2ms across 21 kernels.

Target launch resources are 221 registers / 37,888 shared bytes / 256 threads for dense current, 186 / 25,600 / 256 for dense PV8, and 255 / 23,552 / 128 for PV8-128. Scalar current uses 255 registers and 46,336 shared bytes; splitting scores reduces shared memory to 29,952 bytes for FP16 PV or 17,664 bytes for INT8 PV, with registers still 255. Fewer registers alone did not predict speed.

Static capacity reasoning: Turing provides 64k registers, 64KiB shared memory and at most 32 resident warps per SM. These resources permit two 128-thread PV8 CTAs where the 256-thread PV8 register footprint permits one. Both arrangements can still have an eight-warp static ceiling; this is not evidence of higher achieved occupancy. Likewise, scalar shared-memory reduction can remove a one-CTA resource constraint. Actual dynamic occupancy and stall causality remain unmeasured. [NVIDIA Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#occupancy).

V3 adds the missing FP16-PV 128-thread control and route-aware omitted-leaf score precomputation. The following matched complete APIs use the same real DiT fixture, Group32 QK, outputs and route policy, with the slower rank reported.

| Path | Screen ms | Reverse ms | API reduction, screen / reverse | Numerical interpretation |
|---|---:|---:|---:|---|
| Dense current, FP16 PV / 256 threads | 63.8792 | 65.9335 | — | Immutable control |
| Dense FP16 PV / 128 threads | 61.9825 | 63.3078 | 2.97% / 3.98% | Same method; floating reduction mapping changes rounding |
| Dense INT8 PV / 128 threads | 57.7346 | 58.3700 | 9.62% / 11.47% | New PV quantization error |
| Scalar current, FP16 PV | 51.7094 | 53.0980 | — | Existing optional scalar control |
| Scalar INT8 PV | 49.8954 | 50.4932 | 3.51% / 4.91% | New PV quantization error |
| Scalar full score split, FP16 PV | 49.2783 | 50.7126 | 4.70% / 4.49% | Same-method, bitwise full-fixture pass |
| **Scalar omitted-only score split, FP16 PV** | **46.6736** | **47.1232** | **9.74% / 11.25%** | **Same-method, bitwise full-fixture pass** |
| Scalar omitted-only score split, INT8 PV | 46.4206 | 47.0157 | 10.23% / 11.45% | Bitwise to its existing INT8-PV scalar control |

The dense FP16-128 result has additional relative RMSE 4.60e-6 / 4.93e-6 across the ranks (0.00046–0.00049%) and maximum absolute difference 0.03125 from the FP16-256 control. It is not bitwise-equivalent. INT8-PV128 adds relative RMSE 0.006659 / 0.007081, with maximum differences 1.34375 / 1.375. These compare implementation/numerical methods; they do not establish semantic or generation distribution acceptance. Sparse-versus-dense L1/L2 is deliberately not used to select a sparse policy.

INT8-PV128 is 6.85% / 7.80% faster than the matched FP16-128 control. Thus the mapping improvement and the extra INT8-PV arithmetic each contribute; the whole 9–11% gain cannot be attributed to INT8 GEMM alone. Original 256-thread PV8 remains slower despite lower register/shared-memory counts. V quantization remains a small preparation phase rather than the regression's location.

### Scale restoration diagnostic

The separate restoration job matches current FP16-256, normal INT8-PV256 and a diagnostic replacing real V-scale/beta numerator restoration with a safe constant 1/127. Complete APIs are 66.7648 / 73.6883 / 71.7725ms in screen order and 71.3052 / 76.0832 / 72.9600ms in reverse order. Rank0 reverse prepared-main times are 63.5884 / 67.7226 / 65.1590ms.

Removing restoration-related loads/arithmetic/state recovers 2.56ms of the 4.13ms rank0 main-kernel regression, while remaining 1.57ms slower than FP16 PV. This is a controlled whole-kernel effect including compiler scheduling and live state, not the independent duration of multiplication instructions or an exact percentage of hardware stalls. The diagnostic changes the result and is excluded from dispatch. Target PTX confirms SM75 `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`; the compiler source/PTX and offline SASS inventories are retained. No measured effective Tensor Core utilization is available.

### Scalar critical path and targeted improvement

The final profile-only repair captures all three FP16-PV scalar paths on two physical T4s. Mean per-call kernel intervals, over three calls per rank:

| Phase | Existing scalar, rank0 / rank1 ms | Full score split | Omitted-only split |
|---|---:|---:|---:|
| Existing preparation, 23 launches | 8.869 / 8.867 | 8.921 / 8.966 | 8.892 / 8.908 |
| Routing, 21 launches | 1.225 / 1.224 | 1.242 / 1.257 | 1.239 / 1.246 |
| Extra centroid scores, one launch | — | 5.493 / 5.686 | 3.588 / 3.532 |
| QK / online softmax / PV main | 39.605 / 40.299 | 31.464 / 32.373 | 31.330 / 31.746 |

The split removes centroid reduction live state from the main kernel. Skipping scores for retained leaves then reduces its new preparation cost by 1.91 / 2.15ms. The retained path, log-count weighting, online updates and pooled probability/value behavior are preserved. The score tensor keeps its full 82,066,432-byte shape: entries for retained leaves are intentionally not computed and are never read under the matching route mask. This optimization does **not** claim a smaller workspace. Candidate-specific whole-API attention peak allocation was not measured; the table accounts for the explicit extra tensor only.

The omitted-only INT8-PV variant buys just 0.2530 / 0.1075ms over omitted-only FP16 PV in the two unprofiled matrices. Prefer the FP16 version for the next integration gate: it captures almost all the measured gain without introducing PV quantization. Scalar pooling stays optional. Seven independent units per rank and the actual full-fixture same-method gates pass, including both full and omitted-only score paths.

This also explains why block sparsity percentages cannot be read directly as API acceleration: existing preparation/routing still costs about 10.1ms, retained attention still runs, and omitted summaries have their own score and online-update work. The transformation removes measured repeated centroid work; merely dropping more blocks does not remove those fixed costs. These traces are for this specific route/shape and do not re-establish a universal relationship between a historical 37.6% block reduction and speed.

## FC2 stability confirmation

After the primary FC2 spread proved material, a final job measured 30 rotating blocks per process. Both ranks execute the same candidate order, with a fresh process reversing the base order. Times below are complete-API medians for the slower rank; no memory measurement was repeated in this job.

| FC2 mode | Interleaved screen ms | Reverse process ms | Reduction vs native, screen / reverse |
|---|---:|---:|---:|
| Native INT32 | 47.2044 | 47.9499 | — |
| FP16 | 36.8844 | 41.0663 | 21.86% / 14.36% |
| FP16 fused | 36.3684 | 38.1736 | 22.96% / 20.39% |
| Packed INT8 | 35.2316 | 36.2483 | 25.36% / 24.40% |
| Packed INT8 fused | 36.1793 | 33.7256 | 23.36% / 29.66% |

FP16 fusion's within-block unfused-minus-fused rank0 median is 0.619ms in screen order (p10/p90 -0.589 / 3.851ms, wins 18/30 blocks), versus 2.573ms in reverse order (-1.730 / 4.809ms, wins 23/30). For packed INT8 it is -0.304ms (-3.253 / 3.016ms, wins 14/30), versus +2.900ms (+0.774 / 4.252ms, wins 29/30). Rank1 shows the same broad pattern; its complete samples and paired differences are retained. These are paired blocks, not simultaneous counterfactual calls. Compressed transport is a consistent win against native; a stable incremental FC2 fusion effect is not isolated.

The synchronized 300-call Systems capture matches this per-call event/synchronization behavior. It records **zero CPU_GC callback ranges and zero CUDA allocation/free calls** in the warmed measured interval. Rank0 CPU enqueue medians are 1.14–1.31ms; summed internal GPU kernel-interval gaps have medians 0.021–0.023ms per API. This capture does not support a garbage-collection or device-allocation explanation for multi-millisecond variation.

Instead, unchanged rotation and GEMM intervals vary materially: rank0 unfused-FP16 rotation p10/p90 is 10.995 / 14.285ms and GEMM 8.528 / 10.541ms. Their event-latency correlations are 0.956 and 0.895, respectively; these correlations are observational, not independent causal estimates. FP16 SUM stays near 9.35–9.53ms. Fused FP16 removes an epilogue from 1.008+1.197ms to 1.009ms, but scheduling-sensitive common work changes too. Packed fusion similarly removes about 1.36ms of rank0 epilogue time, while its rotation/GEMM means increase by about 1.09 / 0.65ms relative to its unfused control. Counter access is required to distinguish clock, cache, memory or other GPU-level causes. No such cause is claimed here.

## Pareto comparison and deployment recommendation

![Real H3 transport latency, arithmetic error and allocator memory](D:/attention_eval/tp2_p2_20261018/transport_pareto.png)

The plotted Pareto set uses the original reversed-order transport matrix, with p10–p90 on the slower-median rank. Native INT32 remains the zero-codec-error endpoint; FP16 fused is the low-error compressed endpoint; packed INT8 provides lower latency at higher cancellation error. O packed fused trades a statistically unestablished 0.016ms latency difference for approximately 46.9MiB less allocator peak. FC2 has no measured full-API peak improvement. Formal point-median dominance in JSON ignores uncertainty and can be affected by allocator granularity; it is not a deployment acceptance test. The later FC2 confirmation must accompany the plot rather than being omitted.

Proposed dispatch, without enabling it: keep native INT32 transport and Group32/FP16 PV defaults; retain scalar pooling as an explicit choice. First integrate/test omitted-only FP16-PV scalar splitting at this exact route and shape when workspace fits. Next evaluate the small dense FP16-128 mapping gain. If full TP2 generation later qualifies FP16 transport, use the exact fused O epilogue; choose FC2 fusion only with further deployment-matched evidence. INT8 transport and INT8 PV require explicit opt-in and independent full-generation qualification. Do not extrapolate this DiT kernel schedule to causal/GQA TE or sequence-sharded Video without their own matched operator gates. Drop attention remains unchanged.

## Artifacts and reproduction

Main machine-readable result: [TP2_P2_RESULTS.json](../data/TP2_P2_RESULTS.json). Raw version-bound evidence: [D:/attention_eval/tp2_p2_20261018](D:/attention_eval/tp2_p2_20261018). Each provider directory archives the exact submitted notebook, embedded sources, image identity, dataset versions and SHA256 inventory. Runtime hashes equal provider hashes for all eight benchmark versions. Native and previous Group32/FP16-PV controls come from frozen historical dependencies; production defaults are untouched by these experimental files.

| Deliverable | Files |
|---|---|
| Fused codec epilogues and scoped caller functions | `patches/tilelang_transport_p2.py` |
| Controlled attention and omitted-score kernels | `patches/tilelang_attention_p2.py`, generator `benchmarks/build_tp2_p2_attention_sources.py` |
| Independent arithmetic gates | `benchmarks/test_tp2_p2_epilogue.py`, `benchmarks/tp2_p2_attention.py`; raw `unit_rank*.json` |
| Real capture, pinned reconstruction, upload | `prepare_tp2_p2_real.py`, `stage_tp2_p2_seed.py`, `materialize_tp2_p2_seed.py`, `upload_tp2_p2_real.py` under `benchmarks/` |
| Target harness and balanced confirmation | `benchmarks/tp2_p2_transport.py`, `benchmarks/tp2_p2_transport_confirm.py`, `build_tp2_p2_job.py` |
| Profiler analysis | `benchmarks/export_tp2_traces.py`, `attribute_tp2_p2.py`, `analyze_tp2_p2_jitter.py` |
| Evidence verification and summaries | `benchmarks/verify_tp2_p2_evidence.py`, `summarize_tp2_p2.py`, `plot_tp2_p2_pareto.py` |

There are 28 exported Systems reports: 10 transport, 8 attention-v2, 3 attention-v3 before the export timeout, 3 matched restoration, 1 synchronized FC2 confirmation and 3 scalar repair. All captured kernels are attributed to process-qualified scopes. Separate `trace_attribution.json` files retain both ranks, launches, CPU CUDA APIs, timeline intervals and resource fields. The final scalar repair supplies the originally missing decisive scalar traces; its profiled values do not replace the unprofiled 15-sample performance matrix.

Setup is explicitly outside latency claims: main dependency installation takes 11–14s and runtime/environment staging 106–129s per job; real-fixture materialization takes 38–44s. Screen stage totals include compilation, gates, warmups and measurements (about 78–258s in the attention/primary suites), while repeat stages reuse the kernel cache in a fresh process. Per-candidate first-call-with-JIT-and-gate seconds are retained; these are not pure compiler timers. All compilation/gate/export failures are retained separately from warmed latency.

For CPU-only regeneration after downloading the archived artifacts:

```powershell
python benchmarks/analyze_tp2_p2_jitter.py
python benchmarks/verify_tp2_p2_evidence.py
python benchmarks/summarize_tp2_p2.py
python benchmarks/plot_tp2_p2_pareto.py
```

The source patches are callable experiment implementations, not production dispatch integration. Using the full source snapshots when reproducing is required: the shared checkout can continue changing independently.

## Failure retention

Attention v1 stopped before timing because an unused centroid dummy had the wrong tensor rank. The fix changes the dummy view to one FP32 element; independent local and T4 checks are retained separately. Transport v1 could not find the originally named, unavailable capture dataset; v2 uses the verified published TP2 subset and explicitly records missing trajectory endpoints. V2's unit-restoration diagnostic passes small arithmetic tests but produces nonfinite full-shape FP16 outputs and is not timed or treated as a candidate. The follow-up retains 1/127 scaling to keep that diagnostic in range. Attention v3 completes both measurement processes but its report export hits the 120-second cap after three reports; unpersisted `/tmp` streams could not be recovered. The separate scalar-only job repairs the required scalar profiles. All original logs and rejected records remain in versioned evidence.

A fresh physical two-T4 counter probe confirms Nsight Compute 2025.1.1 still fails with **`ERR_NVGPUCTRPERM` on GPU0**. The tool exists; hardware-counter access is denied. This is the specific blocker to measured stalls, achieved occupancy, memory throughput and Tensor Core utilization, not a reason to replace T4 measurements with SM120 results.

## Prioritized decision table

| Decision | Candidate / scope | Evidence and next gate |
|---|---|---|
| **Promote** | No new production default | Full TP2 approximation-quality gates are not complete. Preserve existing native INT32 / FP16-PV defaults. |
| **Experimental — priority 1** | Optional scalar omitted-only scores + FP16 PV | 9.74% / 11.25% API gain, full-fixture bitwise pass on both ranks. Integrate behind an explicit shape/route/workspace gate; broader route/shape identity and deployment peak-memory validation remain. |
| **Experimental — priority 2** | Dense FP16-PV128 mapping | 2.97% / 3.98% gain, tiny floating mapping difference; validate other target DiT shapes and integration before dispatch. |
| **Experimental — priority 3** | FP16 transport, especially fused O | Fused O 24.47% / 26.01% faster than native, exact codec contract, lower peak than unfused. Real cancellation still needs full TP2 generation qualification. |
| **Experimental** | Packed INT8 transport / fused epilogues | Larger transport gain, global error ~0.67–0.68% and large cancellation-conditioned errors. O fusion is a memory option, not a proven latency win; FC2 incremental fusion benefit varies. |
| **Experimental** | Dense INT8-PV128 | 9.62% / 11.47% gain; new quantization approximation. No semantic/distribution quality claim. |
| **Reject for selection** | Dense INT8-PV256; running-max variant | Regressions in repeated complete APIs. Running-max also introduces a larger numerical approximation without acceleration. |
| **Reject as preferred scalar path** | Full scores or omitted scores + INT8 PV | Full scores spend unnecessary prep time; added PV8 buys only ~0.1–0.25ms over the omitted FP16 variant. Retain as controls, not preferred dispatch. |
| **Reject as deployable output** | Unit-restoration diagnostic | Deliberately changes the result; first version also fails full-shape finite gate. Causal timing evidence only. |
| **Blocked** | Hardware-counter root cause | Current target `ERR_NVGPUCTRPERM`; distinguish static resource hypotheses from measured dynamic execution. |
| **Blocked for promotion** | Full TP2 quality/end-to-end acceptance; all-layer/step generalization | Full TP2 generation was not staged/run in the bounded operator scope. No new video/audio/temporal or pipeline latency/throughput/VRAM qualification. Additional O endpoints remain local only. |
