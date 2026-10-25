<p align="center">
  <img src="docs/assets/h3-banner.svg" alt="MiniMax-H3 - TP2 inference for two NVIDIA T4 GPUs" width="100%">
</p>

<p align="center">
  <a href="docs/guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf"><b>Technical guide (PDF)</b></a> &nbsp; / &nbsp;
  <a href="docs/README.md">Documentation</a> &nbsp; / &nbsp;
  <a href="docs/guides/H3_LEGACY_RUNTIME.md">Setup</a> &nbsp; / &nbsp;
  <a href="docs/reports/H3_PIPELINE_REVIEW.md">Validation status</a>
</p>

MiniMax-H3 Ref2VA inference with persistent tensor-parallel workers, INT8 operators,
bounded weight residency, and parallel VAE decoding. Native ComfyUI retains
conditioning, sampling, and public VAE planning.

> **Deployment status:** the validated local source and dependency versions are
> frozen. The [960×544, 362-frame E2E](docs/reports/H3_L1_TURBO_05MP_E2E_20261024.md)
> completed on two T4s with native Torch2.11.0+cu128, Kitchen0.2.37 and
> TileLang0.1.13. Generation quality review remains pending; this is not a
> controlled speedup comparison or qualification of another dependency stack.

## Start here

| Resource | What you will find |
|---|---|
| [Technical guide (PDF)](docs/guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf) | Architecture diagrams, numerical contracts, current defaults and physical T4 results |
| [Runtime setup](docs/guides/H3_LEGACY_RUNTIME.md) | Pinned ComfyUI, conda environment, launch commands, and platform limits |
| [Operator configuration](docs/guides/H3_OPERATOR_INTEGRATION.md) | Backend choices, workflow integration, and qualification scope |
| [Loading and LoRA](docs/guides/H3_LOADING_STAGES.md) | Layer-parallel baking, staged VAEs, and idle preloading |
| [All documentation](docs/README.md) | Current guides, experiment reports, design notes, and historical records |

## Local preview

Run from the repository root with the existing `comfyui` conda environment:

```powershell
.\launch_h3_legacy.cmd --preview --check
.\launch_h3_legacy.cmd --preview
```

The preview supports a single-GPU host. Physical TP2 execution requires two
matching CUDA GPUs and NCCL. Keep the pinned attention API and package versions;
see the [frozen runtime record](runtime/README.md).

## Current defaults

| Component | Selection |
|---|---|
| Tensor flow | FP16-fix by default; TE/DiT residual divisor 64, explicit FP32 stability islands; `legacy` FP32 flow optional |
| TE / DiT / Token Refiner | TileLang Group32 dense / Kitchen INT8 / Native SDPA |
| Row projections | Scaled FP16 SUM with fused FP32 dequantization; explicit INT32 SUM alternative |
| TE down / DiT FC2 | `fused_stored`: FP32 SwiGLU + regular ConvRot256, global row scales, original INT8 GEMM |
| DiT QKV / FC1 columns | Direct INT8 GEMM + ordered TileLang epilogue on supported SM75 shapes; preallocated fallback |
| Video / Audio VAE | Aligned query parallelism with Kitchen INT8 / Native single-GPU FP32 decoding |
| LoRA | Offline by default; prefer bounded TP2 bundle merges for long sequences. Optional short/medium online uses FP32 RowParallel A partial SUM and guarded FC1/QKV owned B |
| Streaming | Lifecycle prefetch, host-pressure admission and adaptive depth 1–2 enabled; idle speculation off |
| Reference VAE loading | Metadata-only shells; load Video/Audio encoders only on actual reference encoding, decoders at output |

Sparse dropping/scalar pooling, recompute preparation and idle residency remain
explicit options. Removed controls and saved-workflow migration are documented
in the [configuration review](docs/reports/H3_CONFIGURATION_REVIEW_20261024.md).
Numerical differences and Native fallbacks
are explained in the [technical guide](docs/guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf).
The [2026-10-22 promotion record](docs/reports/H3_FP16_FUSED_DEFAULT_20261022.md)
separates physical operator evidence from pending full-pipeline quality checks.

## Repository map

| Directory | Purpose |
|---|---|
| [`patches/`](patches/) | Runtime, operators, loaders, and residency control |
| [`custom_nodes/`](custom_nodes/) · [`workflows/`](workflows/) | ComfyUI integration and workflow graphs |
| [`benchmarks/`](benchmarks/README.md) | Explicit tests, profiling, and operator experiments |
| [`tools/`](tools/README.md) | Runtime commands and historical scripts moved out of the root |
| [`runtime/`](runtime/README.md) | Frozen source/dependency identity |
| [`docs/`](docs/README.md) | PDF guide, current documentation, reports, and plans |

Models, private upload capabilities, and large traces stay outside the published
source. See [artifact retention](docs/guides/ARTIFACT_RETENTION.md).
