# PTD CPU scheduling and runtime-ready refiner experiment

P0 consumption-aware reclamation, P1 cross-hop CPU refiner preparation and optional offline FP32 refiner sidecars are implemented on `codex/h3-ptd-refasymopd`. They remain opt-in. The validated queue and all Native defaults remain available. The physical fixture has now completed; [measured results and limitations](H3_PTD_CPU_SCHEDULING_T4_RESULTS_20261029.md) qualify memory/average-latency effects for the bounded fixture only.

## Consumption-aware cache

`future_cache=True` installs the entire eight-hop weight-use sequence before the first refiner runs. This changes metadata only; it does not materialize eight models. Every CPU group has known next-use and remaining-use metadata. Repeated keys stay reusable until their final planned consumption.

- Zero-use entries are released when no CPU reader or owner-rank CUDA protection remains.
- Other eviction candidates are ordered by distance to next use, prioritizing groups outside the current CPU prefetch prefix. Speculation cannot displace protected/nearer groups. Foreground admission may evict other unprotected future entries to make progress. This is a future-use heuristic for variable-size groups, not a claim of globally optimal Belady caching.
- A group is protected before the GPU lease begins. The cache receives a release only when the rank owner observes the completion event; CPU workers never query CUDA or infer another rank's completion.
- Pinned copies have independent storage. Pin borrowers remain protected while copying; the residency window owns pinned DMA sources until copy-ready. Pageable `take()` transfers both physical ownership and its reservation to the window until DMA-ready. GPU-consumer holds conservatively protect any remaining CPU cache copy.
- Caches are rank-local; only HostBudget reservations/slots are process-shared. One rank's release cannot free the other rank's tensors.
- Final teardown drains consumer protections before joining CPU preparation, avoiding a cache-space/producer-wait cycle. No per-hop global synchronization is added. The ordinary GPU event-owned window remains bounded.

The complete plan is immutable for a request and cannot rewind. At sample end, preparation drains before the plan resets; a subsequent request can reuse the service. This mode assumes a known sequence. It does not change the open-ended Native sampler cache policy.

## Next-hop refiner lane

`prefetch_refiner=True` requires `future_cache=True` and per-hop refiners. One additional CPU lane handles refiner groups. Ordinary preparation keeps an imminent two-group prefix; the next refiner is considered separately, even when the next hop lies beyond the ordinary layer lookahead. Only the upcoming refiner is speculated, rather than all future refiners.

Both lanes share the existing per-rank cache capacity and process-shared byte, physical-memory, cgroup/file-cache and preparation-slot guards. In-flight final entry sizes count toward the cache bound; concurrent producers cannot each claim the same free capacity. This does not expand the GPU window or pin an additional multi-GiB refiner copy. GPU consumption waits on the existing ready event. Actual overlap may be limited by cache capacity, slot contention, CPU bandwidth or insufficient GPU work in a two-layer fixture.

## Offline refiner sidecar

Bake `runtime_layout=True` produces an additional `refiner_runtime.safetensors` per hop. Adapter `baked_refiner=True` selects it. The original merged auxiliary overlay remains unchanged and usable by the baseline. Sidecars contain dense FP32 parameter matrices in the exact existing runtime representation; this is not a new packed SM75 GEMM layout.

For each refiner projection, baking performs Native's load-to-FP32 conversion and, for attention output / MLP FC2 weights, FP32 division by the existing power-of-two residual divisor. An independent destination-FP32-copy-then-fold computation must compare byte-exact before publication. Norms, epsilon, attention, activation casts and forward operation ordering remain unchanged. A tiny complete Native-refiner state comparison also passes locally.

The sidecar is bound to the source overlay digest and full tensor-flow policy. The initial implementation bakes the existing FP16/e6 policy. A mismatched policy or missing sidecar fails explicitly. Runtime verifies each loaded FP32 tensor against a hash/shape proof in the hash-bound hop receipt, plus file size/signature; it does not silently trust a cast-free tensor. Original overlay validation remains intact for the original heads. Checkpoint integrity checks are not disabled.

FP32 sidecars increase disk space and read bandwidth. Disk admission accounts for them before baking. They can lose to compact original overlays despite avoiding runtime conversion/folding. The benchmark measures their additional bytes and one-time bake cost separately. Existing banks are not overwritten or upgraded in place.

## Bounded T4 ablation

Use the same two trained blocks, both refiners, all heads, eight hops, 24-token seeded inputs and FP16_fix / fused_stored / Scaled FP16 operator policy as the qualified fixture. Bake one bank with optional sidecars once; six fresh-service cases use identical inputs:

| Case | Future cache | Refiner lane | Runtime sidecar |
|---|---|---|---|
| Queue baseline | off | off | off |
| P0 | on | off | off |
| P0 + P1 | on | on | off |
| P0 + P1 + sidecar | on | on | on |
| P0 + P1 repeat | on | on | off |
| Queue baseline repeat | off | off | off |

Every video/audio hop output and final FP32 carried state must hash identically to the baseline. Record selection plus denoiser RPC totals, AOT separately, complete worker CPU/default-stream spans, host-budget/cache peaks, CUDA peaks, H2D bytes, consumed releases, future evictions and memory-pressure snapshots. The benchmark-only worker probe adds nested wall/thread-CPU spans for refiner preparation, raw auxiliary reads/validation and sidecar read/checksum. Nested spans overlap and cannot be summed indiscriminately. Default-stream intervals include waits; they do not provide Nsight kernel attribution.

All candidates use the same 6 GiB GPU window, 4 GiB per-rank CPU cache, 4 GiB pin bound, 20 GiB process-shared host limit and 4 GiB GPU activation reserve. No full H3 generation or local trained-LoRA download is involved. CPU/import qualification precedes any new GPU submission; core pins and frozen Native source are preserved. The batch uses public trained inputs, no credentials or dataset mounts, and no status/list/live-log polling.

## Current validation

The rebuilt frozen source passed **35/35** local PTD/queue/scheduling/residency contracts and **179/179** existing review regressions, with GPUs hidden and no trained weight download. Tests cover repeated group use, farthest-future eviction, active/pin ownership, owner-event release, teardown ordering, two-lane capacity, shared cross-rank preparation slots, repeated requests, full-plan installation and exact refiner state/folding. These are CPU contracts, not physical T4 acceptance.

Archive SHA256: `18bff57659f4259f30df2db5d58f90954843463074753b67c3a0a68566914bdc`. Capsule SHA256: `835c47888ca5210eb99d9e979632d7845c49c5e826a07384077b175fc537b42d`. The delivery preserves the frozen ComfyUI archive and exact core-lock hashes and excludes unrelated dirty-worktree changes. Kaggle submission/results will be recorded with their exact source binding.

The final scheduling review also excludes groups already in the window preparation/transfer pool from CPU speculation. This prevents a taken pageable source from being prepared again while ownership is moving to the GPU window. The additional contract passes. The source-bound CPU gate must match this final capsule before T4 submission. Earlier CPU-only submissions and their source are preserved.

The final CPU ownership/scheduling tests additionally passed 600 repeated checks across 100 concurrency cycles. Kaggle CPU version 1 passed its base/review/API gates and 12 scheduling contracts, using the earlier capsule. It did not fetch weights. The final-source [Kaggle CPU version 2](https://www.kaggle.com/code/pamcks/h3-ptd-cpu-schedule-cpu-20261029) passed: 111 base + 179 review tests and 22 explicit checks (2 registration/identity, 7 queue, 13 scheduling). Both receipts match the final capsule, and no weights were fetched. Its artifact transfer stopped after 20 files; the bounded recovery was denied by a kernels.get permission error. Passing receipts were recovered in full; remaining tail artifacts were not. See [CPU evidence and transfer limitation](../data/h3_ptd_cpu_schedule_20261029/cpu_v2/README.md).

The first GPU upload was blocked by CLI authentication. After the user logged in and authorized retry, one byte-identical submission was accepted as **[H3 PTD CPU Schedule T4 20261029, version 1](https://www.kaggle.com/code/pamcks/h3-ptd-cpu-schedule-t4-20261029)**. Notebook SHA256: `3b49bbb46d451c49bf18356728f1b1bed7be412ecdc56f7bfa062dd205580524`. The six-case matrix, source, limits and matching CPU gates are unchanged. No status/list/live-log polling was performed. At submission acceptance, allocation, execution and performance were still pending. The final physical results below do not promote a new default. [Submission evidence](../data/h3_ptd_cpu_schedule_20261029/gpu_v1/) preserves the previous auth block separately.

On 2026-10-29, the user-requested final artifact collection failed with SSL `UNEXPECTED_EOF_WHILE_READING` at `api.kaggle.com` before receiving any output file. One bounded retry of the same download failed identically. No status/list/live-log call or new GPU submission was made. No new authentication error was observed. Actual experiment completion/failure and all candidate metrics remain unverified; this is a transport blocker, not a reported experiment failure. Further automatic requests were stopped. [Transfer evidence](../data/h3_ptd_cpu_schedule_20261029/gpu_v1/collection_20261029/transfer_failure.json).

The user-authorized subsequent download recovered all 40 final artifact files and verified the exact source binding. Actual two-T4 execution completed all six eight-hop cases, 32 contracts, 128 merge-target checks and 64 static layout checks with byte-identical per-hop/final outputs. P0 lowers shared managed peaks about 18.8% without a clear latency gain. P0+P1 lowers mean RPC time 15.5% but increases the maximum later-hop time. The FP32 sidecar is 93.2% slower than P0+P1 and remains disabled. See the [physical results and root-cause evidence](H3_PTD_CPU_SCHEDULING_T4_RESULTS_20261029.md) and [final artifacts](../data/h3_ptd_cpu_schedule_20261029/t4_v1_final/README.md). Earlier failed download attempts remain preserved as transport history.
