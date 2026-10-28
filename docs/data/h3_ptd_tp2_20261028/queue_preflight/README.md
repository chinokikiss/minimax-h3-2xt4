# Persistent PTD expert queue qualification

Source commit: `b6d094d391ecaec4dc89feacb32abe23b912f0ad`.
Native ComfyUI and frozen core dependency locks are preserved.

- Local extracted-source CPU contracts: 19 passed, with GPUs hidden and no trained LoRA download.
- Local extracted-source review regression suite: 179 passed.
- Kaggle CPU version 3: accepted; final results have not been collected.
- Revised physical T4 queue/rebuild comparison: not yet submitted; requires matching CPU receipts.

The T4 plan is a bounded two-layer, eight-hop fixture in rebuild/queue/queue/rebuild order, using one baked bank and identical seeded inputs. It checks byte-identical output hashes and records selection/denoiser RPC timings, worker CPU/default-stream intervals, window wait/H2D events and memory peaks. Default-stream intervals include waits; they are not summed kernel times or Nsight evidence. This fixture cannot establish full-model throughput or generation quality.

The previous failure and the successful 128 merge-target checks are recorded in `../t4_v1_final/`. No status/list/live-log polling was used.
