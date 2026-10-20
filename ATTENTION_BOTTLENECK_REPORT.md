# Attention bottleneck audit

P0 controlled T4 single-head and native56-head experiments and Systems attribution are complete. The generation timeline needed for full pipeline attribution is pending. Historical hypotheses and newer measured results are distinguished below.

## Measured existing observations

Native DiT captures:15 tensors, five layers×three denoising steps, each Q/K/V[1,56,6814,128] FP32. Kernel tiles query32/key64,128threads,107key leaves. Packed prefix874tokens:353text+221reference+300audio. Fourteen64-token leaves are forced dense for both prefix keys and prefix queries, conservatively retaining22additional video tokens in the boundary leaf.
| V4 run | Mean block density | Min layer/step density | Max layer/step density |
|---|---:|---:|---:|
| drop_tau0_dit_seed20261013 | 0.6242 | 0.5474 | 0.7605 |
| drop_tau0_dit_seed20261014 | 0.6238 | 0.5527 | 0.7604 |
| scalar_tau0_dit_seed20261013 | 0.6242 | 0.5484 | 0.7605 |
| scalar_tau0_dit_seed20261014 | 0.6237 | 0.5527 | 0.7604 |

V4 route statistics cover all50layers×20steps averaged across heads. Native-capture CPU route analysis additionally resolves all56heads at the15captured layer/step points; these are native trajectories, not the altered V4 trajectories. Block density and token-pair-weighted density are recorded separately.
V4 second-seed sampling: Sage569.548s, dropping545.443s, scalar547.317s. The approximately4.2% drop-versus-Sage difference is descriptive cold fixed-order data. It cannot isolate kernel speed, compilation, loading or clocks. First native sampling also includes extra placement qualification. No controlled Amdahl fraction is measured from these runs.

## Source-confirmed differences and testable hypotheses

The V4 sparse implementation computes unused V INT8 operands/scales and LSE shift in generic prepare, whereas fused Sage32 avoids them. Sparse routing computes centroids, small matrix multiplication and selection in PyTorch. Public dropping execute uses a Python bool on a GPU reduction to validate nonempty rows, introducing a device-to-host synchronization; the Sage internal route path avoids it. Sparse QK is per-token channelwise scale with Sol centroid centering; Sage uses Q32/K64 scales with token-mean centering. Consequently the original APIs differ in approximation and overhead.
At each query CTA, dropping still scans all107key-block route entries. It skips the selected-block QK/softmax/PV branch for omitted leaves, not the route scan. Scalar pooling performs a centroid dot and value outer-product for every omitted leaf. Quantization/preparation, routing and non-attention model work are not reduced by the selected-block fraction. Their actual contributions require the new traces; these facts do not establish memory-stall or occupancy explanations.
Earlier real-T4 sparse diagnostic head at density0.415: raw drop1.299ms, scalar1.545ms, original padded pool2.847ms; API drop2.932ms and scalar3.315ms. That single deliberately chosen head is not representative of the V4 head-average density0.624. These traces motivate measuring common inputs/precision instead of extrapolating a37.6% sparsity reduction to pipeline speed.

## Controlled replay and feasibility

New matrix: native diagnostic DiT head, lengths2048/4096/native6814, requested effective densities25/33/50/62/100%, dense fused Sage32/drop/scalar. Common Sage Q32/K64 INT8 operands and FP16 PV; API boundary includes original FP32-to-FP16 casts, preparation, necessary summaries and routing. Three independent processes, randomized order, at least1s/5warmupcalls,15event/wall samples per timing. Independent streaming math distinguishes implementation from approximation.
Prefix-only density floors:68.36% at2048,38.96% at4096,24.46% at6814; local neighbors can increase the achievable minimum. Report actual density rather than claiming infeasible budgets were reached.2048/4096 are contiguous native truncations with the same prefix, not independent deployment distributions.8192/16384 are skipped because no captured native sequence of those lengths exists; tokens are not tiled or synthesized to call them real distributions.
Current matrix source is single-head GPU0, not full56-head API/TP2 throughput. Systems NVTX/OSRT traces and CPU cProfile are requested. Existing Compute probe gives ERR_NVGPUCTRPERM: no measured occupancy, spills, stalls, Tensor Core utilization, or instruction-level QK/softmax/PV apportionment. Fused QK/softmax/PV timing remains a single GPU interval.
Nsight capture-range repeat semantics follow NVIDIA: https://docs.nvidia.com/nsight-systems/UserGuide/index.html .
Native per-head audit: D:\attention_eval\quality_constrained_20261015\native_capture_head_routes.json; V4 per-layer/step audit: D:\attention_eval\quality_constrained_20261015\v4_route_audit.json.

## Collected controlled results

process0: complete, 33 cases collected.
process1: complete, 33 cases collected.
process2: complete, 33 cases collected.

| N | Kernel | Actual density | Median raw ms across processes | Median API ms across processes |
|---:|---|---:|---:|---:|
| 2048 | sage32 requested1.0 | 1.0000 | 0.3400 | 0.6348 |
| 2048 | drop requested0.62 | 0.7344 | 0.2700 | 1.1908 |
| 2048 | scalar_pool requested0.62 | 0.7344 | 0.2587 | 1.2719 |
| 2048 | drop requested0.33 | 0.7344 | 0.2798 | 1.2129 |
| 2048 | drop requested0.25 | 0.7344 | 0.2763 | 1.2303 |
| 2048 | drop requested1.0 | 1.0000 | 0.3486 | 1.2800 |
| 4096 | sage32 requested1.0 | 1.0000 | 1.3207 | 1.6072 |
| 4096 | drop requested0.62 | 0.6216 | 0.7548 | 1.7469 |
| 4096 | scalar_pool requested0.62 | 0.6216 | 0.7625 | 1.7148 |
| 4096 | drop requested0.33 | 0.4258 | 0.5673 | 1.5172 |
| 4096 | drop requested0.25 | 0.4258 | 0.5717 | 1.5277 |
| 4096 | drop requested1.0 | 1.0000 | 1.3007 | 2.3470 |
| 6814 | sage32 requested1.0 | 1.0000 | 3.0860 | 3.4447 |
| 6814 | drop requested0.62 | 0.6263 | 2.0052 | 2.9921 |
| 6814 | scalar_pool requested0.62 | 0.6263 | 2.1187 | 3.1232 |
| 6814 | drop requested0.33 | 0.3339 | 1.1800 | 2.1791 |
| 6814 | drop requested0.25 | 0.2688 | 0.9945 | 1.9983 |
| 6814 | drop requested1.0 | 1.0000 | 3.0989 | 4.1144 |

All three independent timing processes completed33cases each with implementation checks in process0. The notebook later reported ERROR because nsys stats refused an older SQLite export; all three raw traces were preserved. Missing SQLite exports were recovered locally with Nsight Systems2026.1.3, performing CPU-only analysis of actual-T4 traces. No benchmark was rerun on SM120.
At native length/density0.6263, raw dropping saves35.0% versus dense while API saves13.1%; at2K/density0.7344 the sparse API is slower despite a faster raw kernel. At100% routes, the raw drop and dense kernels coincide closely, while routing/summary preparation remains additional API work. This isolates preparation/routing overhead from sparse compute; the V4 sampling fraction still requires full-H56 pipeline traces, so no complete Amdahl attribution is claimed.
Duplicate infeasible density requests at2K and4K produced identical routes and redundant measurements. Retain the original evidence; future extensions should deduplicate route-equivalent cases rather than repeat them.

## Actual Systems phase attribution

Three API calls per trace. Kernel durations are correlated to the innermost NVTX CPU enqueue scope using runtime/driver correlation IDs; zero kernels are unattributed. CPU scope durations measure enqueue, not GPU execution. Summed intervals can overlap host activity and must not be added as serialized pipeline costs.
| NVTX phase | GPU kernels / API | GPU duration / API ms | CPU enqueue / API ms |
|---|---:|---:|---:|
| drop/input_cast | 3 | 0.0638 | 0.1249 |
| drop/QK_quantization | 6 | 0.0874 | 0.4212 |
| drop/centroid_packing | 11 | 0.1519 | 0.4203 |
| drop/routing | 20 | 0.1059 | 0.7328 |
| drop/QK_softmax_PV | 1 | 1.6345 | 0.1134 |
| scalar_pool/input_cast | 3 | 0.0639 | 0.1124 |
| scalar_pool/QK_quantization | 6 | 0.0864 | 0.3469 |
| scalar_pool/centroid_packing | 11 | 0.1530 | 0.4027 |
| scalar_pool/routing | 20 | 0.1068 | 0.6514 |
| scalar_pool/QK_softmax_PV | 1 | 1.7209 | 0.1159 |
| sage32/input_cast | 3 | 0.0671 | 0.1911 |
| sage32/QK_quantization | 6 | 0.0876 | 0.3534 |
| sage32/QK_softmax_PV | 1 | 2.5193 | 0.0929 |

Matched dropping/scalar APIs launch41GPU kernels each versus10for dense Sage32, including3input casts and6quantization/mean kernels. Sparse-only centroid preparation adds11kernels/API and routing20kernels/API. Routing CPU enqueue is about0.65–0.73ms/API for only0.106–0.107ms summed GPU kernel work. This directly identifies launch/enqueue and extra preparation as optimization targets for this single-head shape.
Trace kernel-interval busy fractions are75.1% dropping,80.6% scalar and93.8% dense across first GPU start to final GPU end. These are timeline duty fractions, not SM utilization or achieved occupancy. Profiled absolute kernel times differ from unprofiled event timing; do not mix these as one latency sample.
Raw attribution: D:\attention_eval\quality_constrained_20261015\p0_trace_attribution.json; legacy CPU cProfile summary: D:\attention_eval\quality_constrained_20261015\p0_cpu_profile_summary.json. The latter includes first-call TileLang IR specialization (about438ms over the three legacy calls), so it is a cold initialization diagnostic, not steady-state CPU overhead. The legacy public validation synchronization is source-confirmed; its exact contribution is not isolated by the current matched traces.

## Completed native H56 controlled replay

These are content-based top-k budget routes with mandatory prefix/local blocks, using matched fused-Sage QK precision. They isolate density/overhead effects; they are not the Sol tau0 routes used in V4 generation. No native dense-attention timing is inferred from this INT8 comparison.

| Layer | Mode | Requested density | Actual density | Raw ms | Cast/preparation ms | API ms |
|---:|---|---:|---:|---:|---:|---:|
| 0 | sage32 | 1.00 | 1.0000 | 161.149 | 10.248 | 172.458 |
| 0 | drop | 0.25 | 0.2688 | 51.400 | 19.440 | 76.500 |
| 0 | drop | 0.33 | 0.3339 | 60.698 | 19.416 | 85.426 |
| 0 | drop | 0.50 | 0.5045 | 87.041 | 19.432 | 112.205 |
| 0 | drop | 0.62 | 0.6263 | 104.867 | 19.644 | 132.096 |
| 0 | drop | 1.00 | 1.0000 | 161.552 | 19.397 | 186.530 |
| 0 | scalar_pool | 0.62 | 0.6263 | 109.664 | 19.550 | 134.229 |
| 10 | sage32 | 1.00 | 1.0000 | 161.280 | 10.297 | 173.337 |
| 10 | drop | 0.25 | 0.2688 | 51.433 | 19.511 | 76.422 |
| 10 | drop | 0.33 | 0.3339 | 60.754 | 19.533 | 85.917 |
| 10 | drop | 0.50 | 0.5045 | 87.034 | 19.566 | 112.319 |
| 10 | drop | 0.62 | 0.6263 | 105.158 | 19.665 | 131.967 |
| 10 | drop | 1.00 | 1.0000 | 161.971 | 19.520 | 186.744 |
| 10 | scalar_pool | 0.62 | 0.6263 | 108.863 | 19.532 | 135.307 |
| 25 | sage32 | 1.00 | 1.0000 | 160.841 | 10.296 | 172.153 |
| 25 | drop | 0.25 | 0.2688 | 51.266 | 19.456 | 76.188 |
| 25 | drop | 0.33 | 0.3339 | 60.537 | 19.430 | 85.718 |
| 25 | drop | 0.50 | 0.5045 | 86.972 | 19.504 | 112.014 |
| 25 | drop | 0.62 | 0.6263 | 104.798 | 19.526 | 130.830 |
| 25 | drop | 1.00 | 1.0000 | 161.335 | 19.522 | 186.193 |
| 25 | scalar_pool | 0.62 | 0.6263 | 108.644 | 19.472 | 134.316 |

At density0.6263, full-H56 dropping reduces raw compute about35% and complete matched API latency about24% across these layers. It is not intrinsically speed-neutral. At density1 the extra routing/preparation makes the sparse API slower than Sage despite similar raw compute. Lower density improves the measured kernel/API further; quality has not been established for these budget routes.

| Layer | Original Sol tau0 actual density | Original raw ms | Original preparation ms | Original API ms |
|---:|---:|---:|---:|---:|
| 0 | 0.7465 | 123.494 | 51.246 | 180.628 |
| 10 | 0.6586 | 109.789 | 51.255 | 167.324 |
| 25 | 0.5793 | 97.338 | 51.205 | 155.586 |

The actual V4-style Sol approximation has different INT8 operands and heterogeneous native densities: layer0.7465, layer10.6586, layer25.5793 here. Its preparation costs about51.2ms/API, and complete API is180.6/167.3/155.6ms. Layer0 is slower than fused Sage despite faster raw sparse compute. This directly demonstrates substantial legacy preparation cost and a density-dependent break-even point. These native inputs are not the evolving V4 candidate inputs; no pipeline gain is back-calculated as a measured result.
Full-H56 matched Systems traces attribute all kernels. Representative layer0 dropping:3casts(3.548ms),8QK preparation kernels(6.399ms),12centroid/packing kernels(9.233ms),20routing kernels(2.845ms),1fused QK/softmax/PV kernel(101.330ms), per API under profiling. Dense Sage uses3casts+8preparation+1attention. Fused QK/softmax/PV remains one measured interval, not independently apportioned.
Full-H56 timeline kernel-union/span duty fractions are about99.9–100%; the earlier H1 launch-gap fraction is not representative of H56. This is timeline activity, not SM utilization. Warm legacy CPU profiles now exclude initial JIT: most host time is inside public execute and final device synchronization. The bool(GPU reduction) validation blocks the host on preceding device work; cProfile does not measure pure CPU computation or an independently removable134ms penalty.
The actionable source targets remain unnecessary V8/LSE work, centroid/mean preparation and host validation synchronization. Group32 fused preparation is implemented and measured; a mathematically faithful optimized Sol path still requires its own conformance and T4 benchmarks. Pipeline Amdahl attribution is pending the generation timeline, so the original small generation gain is only partly explained.
Matched reference maximum relative L20.00002853; faithful implementation check only. Nsight Compute remains blocked by ERR_NVGPUCTRPERM; memory stalls, achieved occupancy, spills and Tensor Core utilization are unavailable.
Full-head CPU/trace attribution and repetition ranges: D:\attention_eval\quality_constrained_20261015\native_full_summary.json.

## Targeted Sol preparation improvement

Implemented patches/tilelang_sol_drop_fused.py with fused per-token INT8 quantization. It retains the legacy Sol leaf-centroid mean, nearest-even rounding, prefix/local routes and unchanged dropping QK/softmax/FP16-PV kernel. It omits unused V8/value centroids/LSE correction and constructs mandatory nonempty routes internally to avoid host bool validation. No approximation change is intended; conformance is not assumed.

### Completed physical-T4 rebenchmark

| Layer | Candidate | Density | Raw device ms | Preparation ms | API GPU ms | API wall ms |
|---:|---|---:|---:|---:|---:|---:|
| 0 | fused_sage32 | 1.0000 | 162.464 | 10.317 | 174.054 | 174.131 |
| 0 | legacy_sol_drop | 0.7465 | 121.946 | 51.032 | 177.847 | 177.914 |
| 0 | legacy_sol_unchecked | 0.7465 | 121.999 | 51.067 | 179.171 | 179.244 |
| 0 | fused_sol_drop | 0.7465 | 123.855 | 12.547 | 139.732 | 139.822 |
| 10 | fused_sage32 | 1.0000 | 160.907 | 10.308 | 172.245 | 172.350 |
| 10 | legacy_sol_drop | 0.6586 | 108.733 | 51.104 | 166.038 | 166.119 |
| 10 | legacy_sol_unchecked | 0.6586 | 108.766 | 51.122 | 165.048 | 165.150 |
| 10 | fused_sol_drop | 0.6586 | 108.855 | 12.556 | 125.305 | 125.367 |
| 25 | fused_sage32 | 1.0000 | 161.515 | 10.306 | 172.367 | 172.455 |
| 25 | legacy_sol_drop | 0.5793 | 96.932 | 51.036 | 153.037 | 153.112 |
| 25 | legacy_sol_unchecked | 0.5793 | 96.564 | 51.075 | 153.070 | 153.161 |
| 25 | fused_sol_drop | 0.5793 | 97.229 | 12.608 | 114.109 | 114.166 |

All36cases completed after exact operands/routes gates. Maximum full-head output relative difference from legacy: 0; faithful reference maximum: 3.03542438e-05. These establish tested-input conformance, not generation-quality or deployment acceptance.
Fused preparation reduces approximately51ms to12.6ms (about75%); complete API latency falls21–25% versus the legacy public Sol path. Unchecked execution alone changes API by less than1% in either direction across the three captured layers, providing no consistent observed benefit. The measured preparation treatment is the main successful optimization in this experiment.
The unchecked ablation removes public host validation and unused LSE correction together. Fused-versus-unchecked additionally changes quantization/preparation and omits unused V8/VC/shift work. These differences must not be attributed to any single deleted operation without a further ablation.
Raw process results and Systems traces: D:\attention_eval\quality_constrained_20261015\sol_fused_v1\sol_fused; machine summary: D:\attention_eval\quality_constrained_20261015\sol_fused_summary.json.

Latest saved job state: quality: evidence_collected, KernelWorkerStatus.COMPLETE; sol: evidence_collected, KernelWorkerStatus.COMPLETE.
