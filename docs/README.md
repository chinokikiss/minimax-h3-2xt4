# Documentation

[Repository home](../README.md) / Documentation

Start with the **[technical guide](guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf)**
for the implementation technique report, also available as
[Markdown](guides/H3_OPTIMIZATION_TECHNIQUES.md). Its 130 techniques explain
algorithms, work removal, numerical contracts, TP2 ownership, fusion, LoRA,
residency and VAE methods. The PDF has bookmarks and an embedded source-hash
manifest; measurements remain in separate reports.

## Current usage

| Guide | Use it for |
|---|---|
| [TE causal prefix and precision](reports/TE_CAUSAL_PREFIX_PRECISION_20261026.md) | Both T4s: packed masking passes, floating-input prefix dependence and INT8/BF16 precision measured against FP32 |
| [Kitchen dense on SM75](reports/KITCHEN_DENSE_SM75_20261025.md) | Kitchen causal TE default, paired T4 dense candidates and complete raw receipts |
| [Performance engineering follow-up](reports/H3_REVIEW_OPTIMIZATIONS_20261024.md) | Allocation/ownership/integrity repairs and opt-in column/QKV/IO candidates; local operator evidence and pending physical T4 qualification |
| [TP2 bundle LoRA and profiling experiment](reports/BUNDLE_SCALAR_WORKER_PROFILE_20261023.md) | Bounded offline LoRA without baked-model uploads; matched Kitchen/Scalar, worker trace capture and observer-overhead investigation |
| [FP16-fix / stored fusion / scaled transport default](reports/H3_FP16_FUSED_DEFAULT_20261022.md) | Current production dispatch, preserved alternatives, real T4 evidence and remaining quality limits |
| [Runtime setup](guides/H3_LEGACY_RUNTIME.md) | Preserved local ComfyUI and environment, launch commands, Linux qualification limits |
| [Operator integration](guides/H3_OPERATOR_INTEGRATION.md) | Node options, attention policies, row operators, and workflow wiring |
| [Operator enhancements](guides/H3_OPERATOR_ENHANCEMENTS.md) | FP16 transport, positional reads, lifetimes, adaptive lookahead, Native refiner |
| [Loading stages](guides/H3_LOADING_STAGES.md) | Online/offline LoRA, layer-parallel baking, encoder/decoder stages, idle preloading |
| [VAE integration](guides/PARALLEL_VAE_INTEGRATION.md) | Video query partitioning, Native single-GPU FP32 audio |
| [Sampler integration](guides/COMFYUI_SAMPLER_INTEGRATION.md) | Native conditioning/sampling interfaces and the scoped sampler gates |
| [Pipeline review](reports/H3_PIPELINE_REVIEW.md) | Current code boundaries, cleanup, regression evidence, remaining blockers |
| [Runtime provenance audit](reports/H3_RUNTIME_PROVENANCE_AUDIT.md) | Which source/dependency versions were selected and preserved |
| [Technique coverage matrix](data/h3_optimization_coverage_20261026.json) | 130 techniques mapped to all64 runtime modules, all52 prior audited items and retained native attention kernels |

## Explore by topic

| Topic | Key reports |
|---|---|
| Attention | [Kitchen dense SM75](reports/KITCHEN_DENSE_SM75_20261025.md), [T4 Kitchen/TileLang matrix](reports/KITCHEN_ATTENTION_T4_MATRIX_20261025.md), [Bottlenecks](reports/ATTENTION_BOTTLENECK_REPORT.md), [Group32](reports/GROUP32_OPTIMIZATION_REPORT.md), [TP2 P2 investigation](reports/TP2_P2_OPTIMIZATION_REPORT.md) |
| Linear operators and LoRA | [FC2 memory](reports/FC2_QUANTIZER_MEMORY.md), [tensor lifetimes](reports/DIT_TEMPORARY_TENSOR_LIFETIMES.md), [Selective B](reports/SELECTIVE_B_FEASIBILITY.md), [row partials](reports/ROW_PARTIAL_MEMORY.md) |
| Loading and residency | [Host cache](reports/HOST_CACHE_IMPLEMENTATION.md), [residency](reports/RESIDENCY_IMPLEMENTATION.md), [TE I/O](reports/TE_IO_CAUSAL_AUDIT.md), [CPU compression](reports/TE_COMPRESSION_LOCAL_20261020.md) |
| Video and audio | [Video fallback](reports/VIDEO_VAE_QUERY_FALLBACK_CHECK.md), [physical video KV results](reports/VAE_KV_T4_RESULTS.md), [audio rounding](reports/AUDIO_CHANNEL_ERROR_DIAGNOSIS.md) |

## Documentation layout

```text
docs/
  guides/    Current usage guides and the technical PDF
  reports/   Operator measurements, correctness reviews, and investigations
  design/    Proposed architectures and experiment plans
  archive/   Earlier overview and troubleshooting notes
  data/      Small, version-bound aggregate result records
  assets/    Documentation artwork
  pdf/       Reproducible PDF authoring source
```

[All reports](reports/README.md) · [Design notes](design/README.md) ·
[Historical notes](archive/README.md) · [Result data](data/README.md) ·
[PDF source](pdf/README.md) · [Old-to-new path index](path-index.json)

Reports describe their own source and software snapshots. A historical T4
result does not qualify the current frozen conda stack on Linux, and operator
speedups must not be added into a pipeline speedup. Large traces, captures,
model files, and private upload/download capabilities remain local; references
marked **local-only evidence** are intentionally not published artifacts.
