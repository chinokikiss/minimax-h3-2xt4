# Design notes and experiment plans

[Documentation index](../README.md)

These documents record proposed mechanisms and planned experiments. Inclusion here does not mean a feature is enabled or qualified.

- [Optional PTD RefAsymOPD TP2 adaptation](H3_PTD_TP2_ADAPTATION.md) — eight per-hop offline INT8 expert sets, Native adapters, explicit pruned-base approximation and bounded T4 qualification

- [Bounded streaming and stage overlap](ZERO_BUBBLE_STREAMING_ARCHITECTURE.md) — reviewed 2026-10-21; replaces unsupported zero-bubble guarantees with an ownership-safe, phased plan
- [TE compressed-weight streaming proposal](H3_TE_COMPRESSED_WEIGHT_STREAMING_PROPOSAL.md) — reviewed 2026-10-21; pre-fold artifacts, equivalent RAW controls, bounded CPU replay and frozen-stack gates

- [Idle-time TE VRAM residency](IDLE_RESIDENCY_DESIGN.md)
- [INT8 sparse attention designs for NVIDIA T4 / SM75](INT8_SPARSE_ATTENTION_SM75_DESIGNS.md)
- [MiniMax-H3 dual-T4 integration plan](INTEGRATION_PLAN.md)
- [H3 SM75 operator optimization: analysis and execution plan](OPERATOR_OPTIMIZATION_PLAN.md)
- [H3 quality-first attention experiments — 2026-10-14](QUALITY_FIRST_T4_EXPERIMENT_PLAN.md)
- [Prescribed MiniMax-H3 TE / DiT benchmark](REF2VA_BENCHMARK_PLAN.md)
- [Two-slab Video VAE: findings and new experiment](SLAB_EXPERIMENT_PLAN.md)
- [TE residency P0/P1 audit](TE_RESIDENCY_P01.md)
- [What Group32 can borrow from Kitchen INT8 attention on T4](TP2_GROUP32_KITCHEN_DESIGN.md)
- [Physical dual-T4 VAE profiling](VAE_KV_PROFILE_PLAN.md)
