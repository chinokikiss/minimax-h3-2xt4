# H3 SM75 operator optimization: analysis and execution plan

**Current TP2 execution targets and status are in [TP2_OPERATOR_OPTIMIZATION.md](../reports/TP2_OPERATOR_OPTIMIZATION.md). The native/full-shape matrix below is historical analysis, not the active per-rank benchmark configuration.**

Date: 2026-10-16, Asia/Shanghai. This analysis used checkpoint headers, source code and CPU analysis of five existing physical-T4 traces. **Zero new GPU jobs were submitted. No full H3 pipeline runs are authorized in this phase.**

## Fixed product choices and implementation gaps

| Component | Choices retained | Actual native attention shape |
|---|---|---|
| TE language transformer | Group32 dense, causal, native GQA | Q `[1,64,353,128]`; K/V `[1,8,353,128]`, FP32 boundary |
| DiT | Group32 dense; Group32 sparse dropping; Group32 sparse scalar pooling | `[1,56,6814,128]`, FP32 boundary |
| Video VAE decoder | Group32 dense | `[1,32,1797,64]`, FP16 boundary |

Scalar pooling remains an optional, separately benchmarked mode. It is not removed or silently replaced by dropping. Attention INT8 means grouped Q/K INT8 with FP32 scales/softmax/accumulation and FP16 P/V in the present contract. INT8 PV is a separate approximation, outside this first optimization pass.

Group32 means contiguous groups of **32 channels within each token**, not 32 tokens. D128 uses four scales; D64 uses two. The current centering is the unweighted mean of true-count 64-token leaf centroids, including the ragged leaf. Rounding is nearest-even; clipping and the `1e-8` scale floor remain fixed.

Important gaps:

1. Existing tested Sol dropping and scalar pooling use one D-wide scale per token. Even the faster `tilelang_sol_drop_fused.py` calls the quantizer with `group=d`. Their measured wins are **not Group32 sparse performance results**. Add grouped retained-leaf QK to both sparse modes before optimizing/promoting them.
2. `prepare_grouped_fused` currently requires matching Q/K/V shapes. Legacy Group32 expands GQA K/V with `repeat_interleave`. Implement native Hq64/Hkv8 indexing and quantize K/V once per KV head. Mapping is `kv_head = query_head // 8`.
3. The TE visual frontend has source-derived hidden1152/head-count16, hence D72. Existing D64/D128 kernels do not cover it. Complete TE coverage requires a separately qualified tail group `[32,32,8]`, padded MMA operands and the original `1/sqrt(72)` scaling; do not relabel it as D96. Its token/window shapes have not been inventoried from real visual attention captures. Preserve the native fallback until covered.
4. Checkpoint INT8 GEMM is a different contract: ConvRot groups256, activation **one scale per complete row**, weight scales per output channel (actual stored `[N,1]`). Do not change it to Group32 activation scales as a performance optimization.

The newer motion profile additionally observes DiT N7895 with prefix983 (16 mandatory 64-token leaves), versus N6814/prefix874 (14 leaves). Full real QKV exists for6814;7895 has shape/profile evidence, not a saved real QKV distribution. Do not synthesize or tile6814 activations and describe them as real7895 activations.

## What the existing T4 evidence says

Current best tested dense Group32 combines fused quantization with the original Group32 MMA kernel. Across three native layers and three independent processes its median complete capture-boundary API is161.796ms, versus171.123ms for the larger cached-QK variant and174.360ms for fused Sage32. More shared-memory caching was not automatically faster.

The faithful **D-wide** Sol improvement reduced preparation from about51ms to12.6ms and API latency by21–25% with identical operands/routes and identical tested outputs. The unchecked-execution ablation alone showed no consistent improvement. This establishes preparation as a successful target; it does not establish a large removable host-synchronization penalty or a Group32 sparse speedup.

CPU reanalysis of existing generation Systems traces gives the following descriptive sums:

| Trace | Group32 QK/softmax/PV scope | INT8 GEMMs, both backend paths | Outside-attention FP32 GEMMs |
|---|---:|---:|---:|
| Fixed N6814, call9, 50 DiT blocks | 7.802s /39.3% | 6.292s /31.7% | 1.768s /8.9% |
| Motion N7895, call9, 50 DiT blocks | 10.339s /42.1% | 7.498s /30.5% | 2.068s /8.4% |

These are **fractions of summed GPU kernel duration**, across serial two-device placement. They are not exclusive wall-time bottleneck fractions, achieved SM utilization, or a deployment speedup prediction. The native and optimized traces have different trajectories and timings; do not subtract them to claim isolated instruction costs.

In the motion Group32 trace,200 `volta_sgemm_128x64_nn` kernels account for2.063s of the outside-attention FP32 GEMM total. The count matches four ConvRot activation-rotation calls per DiT block and the pinned Python dispatch path. This is a source/count-based inference; confirm with NVTX around each standalone linear's rotation before attributing every SGEMM to it. This is not evidence that the model's four INT8 GEMMs secretly ran FP32.

The same trace has100 SM75 CUTLASS visitor/fused-epilogue GEMMs and100 other SM75 INT8 GEMMs. The separate dequantization kernels total only about139ms, around0.57% of summed kernel time. Merely fusing an already-small dequantization pass is a weaker target than the entire linear-preparation path. A different GEMM backend can still help, but its full API must win.

Local Kitchen is0.2.33; the T4 experiments used0.2.37. The audit downloaded and SHA256-verified the published0.2.37 Python wheel without installing it. Its Turing GEMM heuristic uses a fused path when `M<=128 or N>=2*K`; the remaining large contraction shapes can take cuBLAS plus separate dequantization. Actual trace kernel names confirm both backend families. Published Python source is not a substitute for the binary trace evidence.

The0.2.37 fused ConvRot quantizer estimates `(K + scratch)*4` shared bytes and uses the Turing default block shared-memory limit. For ordinary multi-row configurations the estimate is54,272bytes at K5376,61,440bytes at K7168,90,112bytes at K14336 and135,168bytes at K25600. These exceed the48KiB default; the last two exceed even64KiB. K25600 also exceeds the implementation's16384-wide fused cap. Enabling a flag cannot make every shape fit. Source `input_act='swiglu'` support is conditional on that fused route, so the presence of the API does not prove activation fusion happened on T4.

DiT source calls a fused RMSNorm/RoPE API, but the pinned CUDA constraints accept FP16/BF16 Q/K while the real DiT boundary is FP32. The saved trace has200 normalization kernels over50blocks and no kernel name containing `rms_rope`. Together these motivate checking the FP32 fallback; the name absence alone is not proof of all internal dispatch details.

## ROI ranking

| Priority | Candidate | ROI judgment | First change and evidence needed |
|---|---|---|---|
| P0 | Correct Group32 sparse drop **and scalar pooling** | Required capability, high reuse | Share dense grouped QK machinery; retain route/summary semantics; density1 must agree with dense Group32 |
| P1 | ConvRot/activation preprocessing for exact K values | High potential, strong source/trace motivation | Benchmark the existing staged quantizer and a T4-sized tiled path before writing a new GEMM |
| P1 | Native GQA and strided attention preparation | High for short TE/VAE; moderate for long DiT | Avoid K/V expansion and FP32→FP16→contiguous→padded copy chains |
| P1 | Dense/grouped-sparse long-sequence attention core | Material opportunity, moderate engineering risk | Start from winning original-MMA layout; test a few query tiles and selective Q/scale reuse |
| P2 | GEMM shape dispatch/tile selection | Moderate/uncertain | Compare current cuBLAS and existing SM75 fused CUTLASS on exact MNK; then a small TileLang shortlist |
| P2 | GEMM output scale/bias/gated residual epilogues | Conditional | Extend only paths actually materializing intermediates; preserve cast/FMA ordering |
| P3 | Broad custom GEMM rewrite, split-K, large persistent/mega kernels | Low initial ROI | Require a measured gap; extra reduction, register use and compile cost must be recovered |

For N6814 dense Group32, measured preparation is roughly11.6ms in a161.8ms API. Even deleting all of it would reduce that API by only about7.2%; practical savings are smaller. Long-DiT improvements therefore need attention-core work as well. Earlier H1 TE/VAE measurements are launch-heavy, but their preparation fractions must not be extrapolated to native H64/H32 shapes.

## Attention optimization plan

**Shared dense/drop/pool implementation.** Use INT32 MMA per channel group, then accumulate `sum_g(dot_int32[g] * q_scale[g] * k_scale[g])` in FP32 before online softmax. A single D-wide integer dot followed by one scale is not equivalent. QK/softmax/PV are already fused; do not claim that fusion as a new optimization.

**Preparation.** Read native strides directly, cast at the established FP16 boundary, produce group integers/scales and padded values without multiple materializations. Cache counts, masks and geometry by shape. A staged design is necessary for K's sequence-wide center: leaf statistics, then the global mean, then centered K quantization. There is no safe implicit global barrier between arbitrary CTAs inside an ordinary kernel. Reuse those leaf statistics for routing and optional value centroids.

**Core shortlist.** Keep routing/softmax leaf64 initially. Compare query tiles16/32/64 and at most two viable warp/load policies, pruned by static register/shared-memory limits. Hoist/reuse Q and scales selectively; do not copy the previously slower full-D cached strategy wholesale. Preserve FP32 scale storage first. FP16 scales or INT8 PV are precision experiments, not free performance wins.

**Sparse handling.** Keep both modes explicit. Dropping can optionally compact retained leaf indices in ascending order; measure index construction plus kernel execution. At current moderate sparsity, scanning107route entries may be cheaper than packing, so this is not the first assumed win. Scalar pooling still needs work for omitted leaves: true-count centroid logits and FP16 mean-V outer products. Preserve `log(count)` in each omitted leaf's logit. Do not collapse all omitted leaves into one centroid or process pooled leaves in a new order without treating it as a different numerical contract. Pool mode computes VC; drop mode does not.

**Causal/GQA/ragged correctness.** TE quantizes8KV heads and64Q heads independently, with causal block skipping and token masking. Ragged final key leaves have33valid tokens for TE353,30for DiT6814 and5for VAE1797. Sol's unweighted leaf mean must retain these true counts. In particular, the VAE's5-token final leaf cannot be treated as64tokens during fused centering.

**Output.** Store the native BND layout directly where practical. For FP32 TE/DiT consumers, retain the old FP16 attention-output rounding before widening to FP32. Preserve the VAE's FP16 and `nan_to_num` behavior. Returning raw FP32 accumulators would change the operator.

## INT8 GEMM target matrix

Notation is `(M,N,K)` for `[M,K] @ [N,K].T`. These are real weight shapes plus native captured token counts; they are not per-GEMM runtime input captures.

| Component | Projections | Target MNK |
|---|---|---|
| TE | Q; K/V | `(353,8192,5120)`; `(353,1024,5120)` |
| TE | O | `(353,5120,8192)` |
| TE | Gate/Up; Down | `(353,25600,5120)`; `(353,5120,25600)` |
| DiT | QKV; O | `(6814,21504,5376)`; `(6814,5376,7168)` |
| DiT | FC1; FC2 | `(6814,28672,5376)`; `(6814,5376,14336)` |
| Video VAE | QKV; O | `(1797,6144,2048)`; `(1797,2048,2048)` |
| Video VAE | W1; W2 | `(1797,16384,2048)`; `(1797,2048,8192)` |

There are15logical linears but13unique MNK shapes. Do not confuse DiT hidden5376 with attention inner width7168. M7895 is an optional timing-only extension after the native matrix; exact real activation coverage is currently missing for it.

**First GEMM experiments:** isolate prequantized INT8 MMA, quantization/rotation, output epilogue and complete linear API. Test current dispatch, forced existing fused SM75 backend and current cuBLAS backend with identical operands. Prepack weights once outside timing, but include per-call packing if deployment cannot cache it. Only try custom TileLang tiles on shapes where existing dispatch loses materially. Keep the current backend as the fallback.

**ConvRot strategies:** a smaller-scratch fused CTA for K5120/5376/7168/8192; staged rotate/partial-amax then full-row-amax/quantization for K14336/25600. The published0.2.37 package already has `quantize_int8_convrot_staged`; qualify and benchmark it before reimplementing it. For wide K, never substitute per-chunk activation scales: the checkpoint expects one activation scale per complete row. FWHT versus dense-Hadamard evaluation can change FP32 reduction order and INT8 codes; compare codes/scales against the actual reference rather than assuming bit identity.

**TE input reuse:** where actual execution uses separate Q/K/V projections, share one ConvRot activation quantization across all three; similarly share the gate/up input quantization. Packed QKV/gate-up GEMMs are a second option, preserving per-output weight scales and channel order. Check the loaded merged/unmerged configuration first. Pack at weight load time, not every invocation.

GEMM QKV captures do not contain the corresponding pre-GEMM hidden states. The first timing/core-equivalence screen can use explicitly labeled seeded fixtures, with actual checkpoint weights where practical. Such tests cannot establish real-distribution activation-quantization fidelity. Reuse any verified existing operator input fixtures, or obtain only a bounded isolated operator/block fixture from saved inputs; never run the complete pipeline to fill this gap.

## Fusion opportunities, ordered by expected payoff

| Fusion chain | Priority | Constraints |
|---|---|---|
| Native strided QKV loads → cast/pad → Group32 Q quant; K statistics; V packing | High | Q/K group scaling and old FP16 cast boundary stay exact |
| FP32 Q/K RMSNorm + RoPE → Group32 Q quant / staged K statistics | High | Existing API fusion may fall back for FP32; preserve partial/split-half rotation and K global-mean stage |
| RMSNorm + AdaLN → ConvRot → row quantization before QKV/FC1 | High conditional | DiT per-segment scale/shift; full-row reductions; avoid recomputing normalization for every GEMM N tile |
| SwiGLU → ConvRot → row quantization before FC2 | High conditional | The DiT API already requests it but shape dispatch can prevent fusion; wide-K staged path needed |
| TE shared input quantization → separate or packed QKV/gate/up GEMMs | High conditional | Confirm unmerged route; unchanged activation scale and independently retained weight scales |
| GEMM dequant/bias/output cast → gated residual | Medium | DiT modality/segment gates and VAE channel scales; preserve cast-before-residual semantics |
| Routing statistics → threshold + mandatory mask + optional indices | Medium | Keep FP32 route-score math and FP16 KC rounding; compare masks exactly before timing |
| Attention store → BND cast + VAE finite-value handling | Medium, low implementation cost | Keep original FP16 rounding and nonfinite semantics |
| Whole attention → output GEMM, or both MLP GEMMs in one kernel | Low initial priority | Attention CTAs own separate heads; next ConvRot group256 and row amax span heads. Global dependencies and duplicated computation can erase savings |

Video VAE source still materializes `silu(gate)*up` before W2; it is a specific fusion target. Its FP16 intermediate rounding differs from a naive FP32 SwiGLU implementation, so reuse of a DiT fused path needs a faithful reference. DiT already combines QKV projection, uses a SwiGLU input-activation API and applies segment residuals in place; optimization must address remaining dispatch/layout costs, not merely rename existing fusion.

## GPU-budget plan and promotion rules

1. **This step:** finish CPU/source/trace analysis and freeze the manifest. No new GPU work.
2. **First proposed T4 allocation:** hard20-minute wall cap, including JIT and qualification; actual provider quota accounting recorded separately. Pin the already qualified mixed-operator environment: Torch2.11/cu130, Kitchen0.2.37, TileLang0.1.15, CUDA12.8 compiler and explicit SM75 targets. Load only operator inputs and selected weight matrices. No TE50-layer pass, denoising loop, decoder36-layer pass, or model/media generation.
3. Qualify native TE GQA, VAE D64 and true Group32 dense/drop/pool on N6814. Initially use one representative real capture per component; reserve existing layers0/10/25 and steps0/9/19 for the final shortlisted attention variants. Do not upload additional giant dumps.
4. Screen at most three core variants per family and two/three existing GEMM backend choices per target. Warm at least1s/5calls; randomize order; retain per-call event/wall samples. Cache compilation and report it separately. Shortlist first, then confirm winners in independent processes on the same device. Do not spend three-process timing on every losing tile.
5. Capture one Systems/NVTX operator chain for each winning family, recording rotation, activation, quantization, packing, GEMM, epilogue and output-layout work. Existing Systems traces are reused for ranking. Do not retry Nsight Compute counter collection repeatedly while `ERR_NVGPUCTRPERM` remains unchanged.
6. Promotion requires an unchanged contract, exact masks/indices and integer operands where intended, faithful per-head output-reference checks, finite outputs, correct GQA/causal/ragged behavior and a repeatable complete-API improvement. A raw kernel win with a slower API is rejected. A practical initial threshold is ≥5% API improvement beyond observed variation; this is an engineering screening rule, not a promised gain or statistical confidence interval. Tiny, low-cost changes can still be retained if they improve frequent native shapes without regressions.

No new generation-quality conclusion, full-pipeline latency, TP2/NCCL speedup or concurrent-request throughput will be claimed from these operator tests. Preserve precision-changing variants behind separate options and keep the known backend fallback.

## Reproduction and sources

CPU audit: `C:/ProgramData/miniconda3/envs/comfyui/python.exe benchmarks/audit_operator_optimization.py` from the workspace. It reads headers and existing traces; it does not import Torch, allocate CUDA or contact Kaggle.

Machine evidence is under `D:/attention_eval/operator_optimization_20261016/`: `operator_policy.json`, `operator_shape_inventory.json`, `gemm_target_shapes.json`, `existing_t4_operator_attribution.json`, source hashes and the verified pinned Kitchen source. Previous measured event samples remain under `D:/attention_eval/quality_constrained_20261015/`.

Architecture constraints are from NVIDIA's [Turing tuning guide](https://docs.nvidia.com/cuda/archive/12.9.0/turing-tuning-guide/index.html): INT8 Tensor Core accumulation is INT32, and dynamic shared memory above48KiB requires opt-in with a64KiB architectural block limit. [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async) requires SM80+ for `cp.async`; it is not a T4 optimization path. Use SM75 MMA and software pipelining, without importing Hopper/Blackwell assumptions. NVIDIA's [CUTLASS GEMM discussion](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) describes epilogues and the additional reduction cost of split-K; support in newer tutorials does not imply an SM75 implementation exists for every fusion.
