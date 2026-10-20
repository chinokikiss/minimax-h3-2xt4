# Kaggle failure and activation shard scales

## Confirmed failure

The physical dual-T4 Native run completed: request end-to-end 608.781 s,
sampler 455.030 s, reference encoding plus TE 103.050 s, video decode 30.598 s.
The enclosing Native process took 622.536 s. Those boundaries differ and
must not be interchanged. This is one request, not a statistical latency estimate.

TP failed after 5.63 s, before GPU workers could start. Its exact exception was:

`PicklingError: Can't pickle <class 'pipeline_residency_audit.AuditConfig'>: it's not the same object as pipeline_residency_audit.AuditConfig`

`run_loading_audit.py` replaced the module's dataclass symbol with a configuration
factory. Thread-based logical ranks accepted that; multiprocessing spawn requires
the original class to remain importable under its declared module/name. The repair
applies configuration changes when constructing services and retains the class.
Model operators, checkpoint shards, and quantization are unchanged.

Validation: all four policies passed the actual wrapper on the local 5060 Ti,
two logical CUDA ranks, two layers, four sampling evaluations, cold/warm requests.
All 40 output comparisons were Native bit-exact. This checks semantics, not T4
performance. A separate CPU spawn test sent the actual configuration to eight
child processes across the four policies and checked its complete round trip.
The new physical bootstrap performs this spawn check before original model downloads.
Previous full-depth math qualifications remain applicable because their sources
are unchanged. Proofs: `semantic_results/loading_launcher_repair.json` and
`semantic_results/loading_spawn_qualification.json`.

The saved failed-run evidence was retrieved. No TP performance row from this
run is accepted. Kaggle refuses to mount a failed notebook as a kernel input;
the continuation therefore uses private saved-output download capabilities,
with byte counts and SHA-256 identities, to reuse the completed Native numerical
reference. The next run will verify unchanged model sources before accepting it.
Native timing from the prior session will not be treated as a paired baseline.
All model-weight downloads remain on Kaggle; no model weights are uploaded locally.

Continuation: [P012 Spawn Repaired Audit](https://www.kaggle.com/code/chinokiki9/minimax-h3-p012-spawn-repaired-audit),
version 2, accepted with the `NvidiaTeslaT4` accelerator and an 8,400 s total
ceiling. At 2026-10-11 04:25 UTC it was RUNNING. Its live log confirmed all six
saved Native artifacts' integrity and verified unchanged operator sources before
dependency installation. Four TP cases have 1,200 s each; Native is excluded.
Physical results are still pending. Version 1 of this continuation failed its
early evidence guard because Kaggle refused the prior failed kernel input;
version 2 uses saved-output transport instead.

## What separate activation scales would change

For feature-sharded row-parallel projections, let `A_r` be rank r's INT32 GEMM
partial, `s_r` its activation row scale, and `s_w` the existing Native per-output
weight scale. Independent scales require:

`y = sum_r(FP32(A_r) * s_r * s_w) + bias`

Thus the proposed order is valid: local INT8 activation quantization, local INT8
GEMM, local FP32 dequantization, FP32 SUM AllReduce, then final cast and
bias/residual handling. Do not dequantize the input before GEMM, add bias on both
ranks, or round each partial to FP16 before reduction. Norm/AdaLN statistics and
256-channel ConvRot groups must retain their original scope and arithmetic.
Packed SwiGLU gate/up ordering also remains unchanged.

The existing optimized fallback uses one maximum per complete activation row:

`global max = MAX_AllReduce(local absmax)`

`y = NativeDequantize(SUM_AllReduce(INT32 GEMM partials), shared scale, weight scale)`

It already avoids gathering the full activation on that route. The fused Native
quantizer route still falls back to the validated gather implementation to
preserve dispatch and rounding. Column-parallel projections have replicated
inputs and do not present this same feature-shard scaling problem.
Independent scales could eliminate these fused-route activation gathers too,
which makes their potential communication benefit larger than removing only
the small MAX collective. That also changes the quantizer's dispatch/geometry
and must be measured separately.

| Property | Current global row scale | Independent feature-shard scales |
|---|---|---|
| Native quantization policy | Preserved | Changed |
| Scale communication on optimized fallback | One FP32 value per row, MAX | None |
| GEMM-output reduction | INT32 SUM before dequantization | FP32 SUM after local dequantization |
| Output-reduction payload with FP32 partials | Four bytes per output element | Four bytes per output element |
| Rounding | Native dequantization after integer sum | Each partial scaled before floating sum |
| Accuracy claim | Native equivalence already qualified | Must be measured |
| T4 latency claim | Requires accepted physical results | Requires a separate physical comparison |

A smaller local absolute maximum gives that shard a smaller quantization step
and a smaller worst-case nearest-rounding bound, assuming the same unclipped
absmax quantizer. Actual coordinate errors need not decrease monotonically,
and output error also depends on weights and error cancellation. If both shards'
maxima are similar, the potential gain is small. Native output agreement and
better numerical accuracy are different criteria.

The scaling algebra follows the documented INT8 quantize/dequantize equations:
[NVIDIA quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html).
The project's dispatch and reduction ordering were checked directly in
`patches/optimized_tp.py` and `patches/parallel_int8.py`.

## Recommended experiment

Treat independent scales as a separate quantization variant, not an invisible
replacement in the Native-equivalent P0/P1/P2 audit. First capture representative
TE/DiT row-projection inputs and measure shard-max imbalance, activation error,
and GEMM-output error against FP32 activations with the same dequantized INT8
weights. That reference isolates activation quantization; it is not an original
full-precision checkpoint. Include equal-range and strongly unequal-range cases.

Then validate full sampling drift and quality across held-out Ref2VA requests.
Only after the accuracy criteria are explicit should physical T4 measurements
compare the removed MAX collective, local dequantization, FP32 reduction,
temporary buffers, and end-to-end latency. Do not label this variant Native
bit-exact or claim an accuracy/speed improvement from the algebra alone.
