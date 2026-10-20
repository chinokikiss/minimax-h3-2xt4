# Documentation

[Repository home](../README.md) / Documentation

Start with the **[40-page technical guide](guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf)**
for an illustrated explanation of the integrated H3 pipeline. It covers operator
contracts, TP2 ownership, fusion, LoRA, memory residency, and staged parallel VAEs.
The PDF includes bookmarks and an embedded source-hash manifest.

## Current usage

| Guide | Use it for |
|---|---|
| [Runtime setup](guides/H3_LEGACY_RUNTIME.md) | Preserved local ComfyUI and environment, launch commands, Linux qualification limits |
| [Operator integration](guides/H3_OPERATOR_INTEGRATION.md) | Node options, attention policies, row operators, and workflow wiring |
| [Operator enhancements](guides/H3_OPERATOR_ENHANCEMENTS.md) | FP16 transport, positional reads, lifetimes, adaptive lookahead, Native refiner |
| [Loading stages](guides/H3_LOADING_STAGES.md) | Online/offline LoRA, layer-parallel baking, encoder/decoder stages, idle preloading |
| [Parallel VAE integration](guides/PARALLEL_VAE_INTEGRATION.md) | Video query partitioning, FP32 audio channels, Native fallbacks |
| [Sampler integration](guides/COMFYUI_SAMPLER_INTEGRATION.md) | Native conditioning/sampling interfaces and the scoped sampler gates |
| [Pipeline review](reports/H3_PIPELINE_REVIEW.md) | Current code boundaries, cleanup, regression evidence, remaining blockers |
| [Runtime provenance audit](reports/H3_RUNTIME_PROVENANCE_AUDIT.md) | Which source/dependency versions were selected and preserved |
| [Documentation coverage audit](reports/H3_DOCUMENTATION_COVERAGE_AUDIT.md) | 52 implementation items checked against 85 documents and retained experiment families; omissions and historical status reconciled |

## Explore by topic

| Topic | Key reports |
|---|---|
| Attention | [Bottlenecks](reports/ATTENTION_BOTTLENECK_REPORT.md), [Group32](reports/GROUP32_OPTIMIZATION_REPORT.md), [TP2 P2 investigation](reports/TP2_P2_OPTIMIZATION_REPORT.md) |
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
