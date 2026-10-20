"""Physical diagnostics, profiler evidence, limits and pending full-study stages."""
import json
from pathlib import Path

OUT=Path('D:/attention_eval/t4_redo_20261013')
improvement=OUT/'improvements_v2/improvement_results'
result=json.loads((improvement/'benchmarks/targeted_improvements.json').read_text())
sparse=OUT/'sparse_failure_v1/h3_gate_results'
diagnostics=json.loads((sparse/'accuracy/accuracy.json').read_text())
chunk=diagnostics['records'][0]['head_chunks'][0]
root=chunk['root_causes']
lines=['# Actual T4 attention diagnostics — 2026-10-14','',
    'Status: diagnostic experiments completed on physical T4 SM75; the complete 170-capture upload and deployment study are pending. This is not the requested final optimization report.',
    '', '## Evidence and scope','',
    'Private Kaggle datasets contain exact native captures, preserving full sequence lengths and original FP32/FP16 values. Each slice records its original capture SHA-256 and original head index; each transferred .pt file has its own verified SHA-256. No model checkpoints were uploaded.',
    'The completed gates cover TE layer 10/head 40 (N=353), DiT layer 49/call 9/head 10 and layer 25/call 9/head 14 (N=6814), and Video VAE layer 0/pass 0/head 5 (N=1797). These are intentionally selected diagnostic heads. Their timings must not be extrapolated to full heads, TP2 throughput, or the complete pipeline.',
    'Both allocated GPUs were Tesla T4, capability 7.5, with Torch 2.11.0+cu128 and TileLang 0.1.15. Operator gates use GPU 0; allocation of two GPUs is not evidence of a two-GPU pipeline measurement.',
    '', '## Confirmed profiler blocker','',
    'Nsight Systems 2025.1.1 successfully produced CUDA/NVTX timelines, SQLite exports and kernel/API summaries. Nsight Compute 2025.1.1 returned ERR_NVGPUCTRPERM on an actual CUDA kernel. The provider denies performance-counter access even though the container process is root. Achieved occupancy, warp stalls, DRAM throughput counters and instruction-level QK/softmax/PV time attribution are unmeasured.',
    f"Evidence: [probe](<{OUT/'probe_v2/t4_profiler_probe/probe.json'}>), [counter error](<{OUT/'probe_v2/t4_profiler_probe/ncu_capture.json'}>). NVIDIA documents the permission requirement at https://developer.nvidia.com/ERR_NVGPUCTRPERM.",
    '', '## Why selected-block reduction is not equivalent to saved work','',
    'The current pooled path still iterates every leaf. An omitted leaf runs a padded 32-column FP16 QK GEMM for one centroid, a padded 64-column FP16 PV GEMM for one value summary, and the same 64-column softmax/reduction machinery. Source inspection confirms that pooling replaces the operands but does not skip the PV tile. INT8 PV continues to use FP16 PV for pooled leaves. Routing and preparation add operations before this fused kernel.',
    'On the real layer-49/head-10 gate, Sol retained 66.59% of blocks (33.41% fewer selected blocks). Nsight Systems measured approximately 6.73 ms dense and 6.54 ms pooled attention kernels, with 231 versus 270 total GPU kernel launches over three API calls. This is a separate diagnostic observation, not a claim that the earlier whole-pipeline 37.6% reduction has now been benchmarked on T4.',
    'The launch records show 54,272 bytes shared memory/block for dense FP16 PV and pooled FP16 PV, and 65,024 bytes for dense INT8 PV, with 128 threads/block. Turing has 64 KiB shared memory per SM, so these allocations permit at most one resident CTA from shared-memory capacity alone. This is a theoretical resource bound, not measured achieved occupancy. See [NVIDIA Turing guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html).',
    f"Raw launch records: [baseline and sparse](<{OUT/'replay_gate_dataset_v2/h3_gate_results/nsys_launch_evidence.json'}>).",
    '', '## Accuracy failure mechanisms','',
    'TE layer-10/head-40: full-output dense INT8 error is 26.91%, while FP16 cast error is 0.061%. Controlled 64-query ablations give Q-only 8.13%, K-only 22.15%, P-only 2.48%, V-only 2.80% and P+V 3.77% relative output error. QK quantization, particularly K, is the dominant mechanism on this selected head. QK perturbations change softmax probabilities; the sampled QK total variation is 3.78% and top-1 key changes occur in 4.69% of sampled query rows. The slightly lower combined INT8-PV error is cancellation, not evidence that PV quantization is intrinsically more accurate.',
    'DiT layer-49/call-9/head-10: QK quantization dominates; dense error is 22.84%. Pooling adds almost no error on this head because its important keys remain selected. This head cannot explain the severe pooling failures on other heads.',
    f"DiT layer-25/call-9/head-14: dense INT8 error is {chunk['outputs']['dense_int8']['relative_l2'][0]*100:.2f}%, but Sol pooling increases it to {chunk['outputs']['sol_tau0']['relative_l2'][0]*100:.2f}%.",
    f"Sampled omitted native attention mass: mean {root['omitted_native_mass']['mean'][0]*100:.2f}%, p95 {root['omitted_native_mass']['p95'][0]*100:.2f}%, maximum {root['omitted_native_mass']['max'][0]*100:.2f}%.",
    f"Replacing only omitted-block logits with centroid logits causes {root['ablations']['centroid_key_only']['output']['relative_l2'][0]*100:.2f}% sampled output error. Preserving exact block mass but replacing its conditional attention-weighted value with the unweighted V mean causes {root['ablations']['mean_value_only']['output']['relative_l2'][0]*100:.2f}%. Both centroids give {root['ablations']['both_centroids']['output']['relative_l2'][0]*100:.2f}%; dropping blocks gives {root['ablations']['drop_without_pooling']['output']['relative_l2'][0]*100:.2f}%.",
    'An oracle retaining exact block log-sum-exp and the exact conditional weighted value reconstructs the native result within floating-point rounding. This isolates failures in both coarse block mass and unweighted pooled values. The ablation errors are interacting counterfactuals, not additive percentages.',
    '', '## Implemented and re-tested candidates','',
    'Dense preparation removes unused V8 quantization, pooled V preparation, dense router scores, repeated route validation, and unused LSE reconstruction. It retains the same K centering and Q/K quantization. Outputs match the original operator bit-for-bit on all three gate heads. Nsight Systems records 76 → 32 cudaLaunchKernel calls/API invocation, excluding the one driver attention launch.',
    'The grouped-QK TileLang kernel uses independent token scales over groups of 32 channels, retaining INT8 tensor-core QK and FP16 PV. This reduces the effect of one large channel setting the scale for all 128 features. Independent streaming dequantized references validate its implementation; maximum relative kernel/reference error across the three gate heads is 0.0213%. Its TE launch uses 24,576 bytes shared memory and 202 registers/thread versus 54,272 bytes and 243 registers/thread for the original causal kernel.',
    '', '| Diagnostic head | Original API ms | Same math, reduced preparation ms | Group32 API ms | Original error | Group32 error |',
    '|---|---:|---:|---:|---:|---:|']
for case in result['cases']:
    timing=case['timings'];label=f"{case['capture']['component']}, head {case['capture']['source_head']}"
    lines.append(f"| {label} | {timing['baseline_attention_API']['gpu_median_ms']:.3f} | {timing['fast_dense_attention_API']['gpu_median_ms']:.3f} | {timing['group32_attention_API']['gpu_median_ms']:.3f} | {case['baseline_error']['relative_l2'][0]*100:.3f}% | {case['group32_error']['relative_l2'][0]*100:.3f}% |")
lines+=['', 'Each operation is warmed for at least one second and five calls, then measured 15 times. Compilation, loading and first execution are separate. Raw GPU-event samples, CPU wall times, and GPU clock/utilization/memory telemetry are retained. GPU-event API times include host enqueue gaps and are not pure CPU-free kernel durations. Nsight capture timing retains profiler overhead and is separate from these unprofiled medians.',
    f"Results: [re-benchmark](<{improvement/'benchmarks/targeted_improvements.json'}>), [launch evidence](<{improvement/'nsys_launch_evidence.json'}>), [telemetry](<{improvement/'hardware_telemetry.csv'}>).",
    'Group32 changes quantization error on these selected heads. Its generation quality has not been evaluated; operator error alone does not accept or reject production use. No full-pipeline improvement is claimed.']
scalar_path=OUT/'scalar_pool_v1/scalar_pool_results/benchmarks/scalar_pool.json'
if scalar_path.exists():
    scalar=json.loads(scalar_path.read_text());timing=scalar['timings']
    lines+=['', '## Pooled-kernel bottleneck: controlled implementation change','',
        'The scalar-pool TileLang variant replaces only the omitted-leaf calculation: a direct centroid dot, scalar softmax update and value outer product replace both padded GEMMs and the 64-column pooled softmax. Quantization, selected leaves, routing and the mathematical pooling approximation remain the same.',
        f"On the real layer-25/call-9/head-14 gate, the raw kernel improves {timing['original_raw']['gpu_median_ms']:.3f} → {timing['scalar_raw']['gpu_median_ms']:.3f} ms ({timing['original_raw']['gpu_median_ms']/timing['scalar_raw']['gpu_median_ms']:.2f}×), and the complete attention API improves {timing['original_API']['gpu_median_ms']:.3f} → {timing['scalar_API']['gpu_median_ms']:.3f} ms. Native-output error remains {scalar['scalar_error']['relative_l2'][0]*100:.2f}%. Relative change from the original pooled output is {scalar['vs_original']['relative_l2'][0]:.3g}; independent quantized-reference error is {scalar['implementation_error']['relative_l2'][0]*100:.4f}%.",
        'Nsight Systems independently measures approximately 2.56 → 1.32 ms for the attention kernel over three profiled calls. Both APIs launch 270 total GPU kernels across those calls; shared memory drops 54,272 → 42,112 bytes/block and registers/thread drop 184 → 168. This controlled result identifies the padded pooled computation as a performance bottleneck. The remaining approximation changes this head output substantially. Whether it preserves useful contributions and generation quality requires paired downstream evaluation.',
        f"Evidence: [re-benchmark](<{scalar_path}>), [Nsight launch records](<{OUT/'scalar_pool_v1/scalar_pool_results/nsys_launch_evidence.json'}>)."]
lines+=[
    '', '## Pending work and limits','',
    'The authorized private upload was found stopped with resumable state and is being recovered. The revised quality-first plan supersedes numerical-only acceptance language; stage status and source hashes are required before further claims.',
    'The full 170-capture per-head/layer/step study, warmed deployment-head-count comparisons, native and candidate T4 sampling trajectories, visual/audio comparisons, and real 2×T4 single/concurrent pipeline timelines/PCIe/NCCL/VRAM/throughput have not yet been completed. The existing SM120 pipeline results do not satisfy those requirements.',
    'Nsight Compute counter profiling requires a provider allocation with counter permissions enabled. Systems launch-resource bounds are not a substitute for the blocked counter measurements.']
(OUT/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n')
print(OUT/'DIAGNOSTIC_REPORT.md')
