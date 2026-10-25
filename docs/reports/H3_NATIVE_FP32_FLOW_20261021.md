# BF16 and FP16 against the existing Native FP32 tensor flow

**2026-10-21 — corrected reference definition; local component results only.**

**Subsequent default change (2026-10-21):** at the user's request, the loader
and `RuntimeConfig` now default to FP16-fix with TE/DiT divisor 64. `legacy`
remains an explicit option and the reference for the measurements below.
The default change does not extend their qualification scope.

FP32 ground truth means the existing Native FP32 **activation flow** in the TP2 runtime. It retains the validated checkpoint weights, INT8 activation quantization/GEMMs, attention dispatch, integer reductions and native mixed-precision islands. It does not mean converting every operator, weight, attention calculation or VAE stage to FP32. BF16 and FP16 are both candidates against this reference.

The earlier forced-dense-FP32 experiment changed operators and was outside the requested scope. Its results are excluded here and preserved separately. The [older BF16 report](H3_FP16_TENSOR_FLOW_BF16_20261021.md) contains pairwise distances to BF16, not ground-truth errors. Neither study changed the production runtime to dense FP32.

## Exact reference and scope

The block reference uses `TensorFlowPolicy()` / `tensor_flow=legacy`, FP32 input activations and the same `H3OperatorPolicy()` as every candidate. Main TE/DiT checkpoint weights remain INT8; their bytes are hashed per rank and checked across configurations. Row outputs still use the existing quantization-preserving INT32 SUM route. No `_full_precision_mm` override, forced dequantized GEMM, or attention replacement is installed to create the reference. Output errors are measured after restoring each candidate's residual gauge.

DiT uses the existing Kitchen INT8 attention. On this local RTX 5060 Ti, SM75-only TE Group32, TileLang packed rows and fused SwiGLU take their existing native fallbacks. These fallbacks apply to reference and candidates under the same policy. Dtype-dependent kernel dispatch and explicit FP32 stability islands can still differ between flows. This is a comparison of the implemented flows, not a pure storage-rounding experiment or validation of physical T4 kernels.

The existing condition projection, Token Refiner and latent input projections use their checkpoint's dense BF16/F32 weights through the native FP32 factory. They remain native dense projections in the reference; Token Refiner attention remains native SDPA. They are not artificially changed to INT8. Qwen vision and Video VAE retain their existing Native FP16 stages; Audio VAE retains Native FP32. These stages are not converted to FP32 merely to name a reference.

All experiments use the pinned `ComfyUI` source `d49e888586dd8ae012c0667b33466b815fee07f7` and the conda `comfyui` environment: Torch 2.12.1+cu132, Kitchen 0.2.37, TileLang 0.1.13. No dependency or upstream update was performed. No full H3 pipeline, sampler or Kaggle job was run.

## Corrected numerical results

All percentages below are relative L2 **against the existing FP32 flow**, not against BF16. They diagnose local numerical changes; they do not establish video/audio quality or semantic fidelity.

| Probe | BF16 candidate | FP16 candidate, divisor 64 | Range observation |
|---|---:|---:|---|
| DiT block 0, retained S7895 input | 0.7611% | 0.2653% | Unscaled FP16 has one nonfinite output; divisor 64 is finite |
| DiT block 0, S513 subset | 0.7302% | 0.2622% | Unscaled FP16 has one nonfinite output |
| DiT layer 24, reused S513 stress input | 4.6689% | 3.1988% | Both candidates finite; this is not the layer's actual input distribution |
| DiT layer 49, reused S513 stress input | 2.8513% | 1.6703% | Unscaled FP16 has 37,900 nonfinite outputs; divisor 16 still has one; divisor 64 is finite |
| DiT block 0 + online DMAD LoRA, S513 | 0.7395% | 0.2866% | Divisor 64 is finite; no new LoRA policy is introduced |
| TE layer 0, synthetic S257 | 0.3289% | 0.1065% | Both candidates finite |
| TE layer 49, reused synthetic S257 | 3.9905% | 2.4685% | Both candidates finite; not a captured late-layer distribution |
| Real retained conditioning: projection + refiner | 0.7193% | 0.0901% | Native SDPA retained |
| Same conditioning/reference/noise: packed DiT input | 0.2806% | 0.0351% | After gauge restoration; no sampler executed |

The DiT fixture reconstructs the first block input of a retained **512 × 512, 90-frame** run, seed 20261013: `[7895, 5376]`, comprising 417 text, 266 reference-image, 300 audio and 6,912 generated-video tokens. It is not a fresh capture of the original 640 × 352 request. Late-layer probes reuse its 513-token subset and are stress tests only. TE uses seeded synthetic, half-rounded inputs, not real captured TE activations.

Within these bounded probes, scaled FP16 is closer to the existing FP32 flow than BF16. This does not prove better generation quality or full-trajectory safety. Power-of-two scaling adds range headroom; it does not improve FP16 mantissa precision. The previous measured underflow observations remain relevant, but old BF16 pairwise distances must not be relabeled as FP32 errors.

## Reproducibility and retained implementation

Each block configuration executes one warm-up and two output runs. All FP32 references, BF16 candidates and selected divisor-64 FP16 candidates have byte-equal rank outputs and repeats. Instrumentation records leaf outputs, pre-sanitization attention outputs and activation casts, and counts calls to the installed INT8 GEMM without replacing its arithmetic. INT8 weight bytes match across all block configurations. The separate frontend probe uses its native dense checkpoint projections and makes no repeat/performance claim.

Two logical ranks execute on one physical GPU using simulated transport. This cannot measure NCCL/PCIe behavior, two-device memory pressure or T4 speed. The instrumented runs are numerical checks, not performance benchmarks. Smaller local L2 values are not a reason to change deployment defaults.

The FP16 runtime implementation, divisor-64 TE/DiT folding, explicit FP32 stability boundaries, Native FP16 vision/Video defaults and unchanged FP32 Audio VAE remain as described in the [implementation report](H3_FP16_TENSOR_FLOW_BF16_20261021.md). At measurement time, `tensor_flow=legacy` was the default; the subsequent default change is noted above. No production source was changed while correcting the numerical reference itself.

Evidence: [38 configurations](../data/h3_native_fp32_flow_20261021.json). Raw outputs, diagnostic hooks and source snapshots: `D:\attention_eval\h3_native_fp32_flow_20261021`. Earlier studies remain in their original artifact directories. Preliminary frontend tracing records are excluded by an explicit file list; the checkpoint's native dense frontend is recorded explicitly in the accepted result.

```powershell
& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_h3_native_fp32_flow.py dit --tokens 0 --output D:/attention_eval/new_native_flow/dit_7895.json
& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_h3_native_fp32_flow.py te --tokens 257 --output D:/attention_eval/new_native_flow/te_layer0.json
& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_h3_native_fp32_flow.py frontend --output D:/attention_eval/new_native_flow/frontend_native_components.json
```

Full-trajectory overflow behavior, semantic/video/audio quality and physical SM75 behavior on the frozen Linux stack remain unqualified. No current result establishes a speedup or production acceptance.
