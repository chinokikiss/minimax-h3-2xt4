# Persistent host weight cache

Implemented 2026-10-09 for a 32 GB Kaggle host and single-request latency. **Physical qualification passed:** `chinokiki9/minimax-h3-host-cache-gate` version 2 completed, with ten bit-exact comparisons and verified post-decode preloading on both ranks. This qualifies the cache mechanism at the tested block/compact-video scope; production ComfyUI sampler integration remains.

This document records the earlier gate's settings and source version. The subsequent [ComfyUI sampler integration](../guides/COMFYUI_SAMPLER_INTEGRATION.md) adds the Native VAE completion hook, actual sampler scheduling, 3 GiB per-rank pageable caches, allocator-rounded pinned reservations and stage-boundary cleanup. Its new qualification is recorded separately; the earlier results do not qualify changed source automatically.

## Runtime

`patches/host_weight_cache.py` adds a persistent pageable cache behind `ResidencyWindow`. Each rank has a CPU preparation worker and a cache. Both ranks must receive the **same process-shared `HostBudget`** from their parent. The budget is not multiplied by the number of GPUs.

| Limit | Initial setting | Meaning |
| --- | --- | --- |
| Managed host reservations | 16 GiB total | Pageable cache, admitted pinned staging, and estimated construction peaks across both ranks |
| Retained pageable cache | 4 GiB per rank | At most 8 GiB retained; entries can be evicted for preparation/staging |
| System available-RAM guard | 3 GiB reserve | Additional admission check against current available RAM |
| Pinned staging in the gate | 3 GiB per rank | Included in the shared total; permits a complete compact-test video decoder group |

These limits do not imply 16 GiB total process RAM: model shells, activations, framework libraries, allocator caches, output buffers and external applications also consume memory. Construction bounds are estimates supplied by each loader, not operating-system allocation enforcement. The available-RAM guard is an additional safeguard; it does not reserve RAM against unrelated processes.

The generic preparation reservation is four times a group's maximum final bytes. The full video loader uses an explicit three-times bound after inspecting its Native mixed-precision loading path: INT8 Linear creates no initial dense weight and CPU quantized payloads wrap checkpoint storage. The first local run conservatively refused the second video preparation under the generic bound. The specialized bound passed without changing the total budget or available-memory reserve.

The disk cache and GPU window have separate lifetimes. GPU-window completion drains its copies and releases pinned ownership; it leaves the pageable cache available to the next request. Closing the cache releases its remaining reservations. Prepared modules retain Native INT8 payloads and layout metadata.

## Request lifecycle

1. Call `cache.begin_request(specs)` with the actual model/configuration keys, then pass `host_cache=cache` into each rank's `ResidencyWindow`.
2. Execute the existing Native/TP interfaces. Required CPU preparation takes priority over idle tasks at module boundaries. Cache hits bypass disk reads and module construction; H2D still occurs when GPU weights are absent.
3. After final VAE consumers complete, call `cache.prefetch(te_startup_specs)` for the next request. The call enqueues bounded disk/CPU work and returns; result delivery must not wait for it. Only an execution-order prefix fitting the cache's conservative bounds is queued, preventing a long prefetch list from evicting its own startup layers.
4. The next request reuses matching cached weights. Configuration changes cancel unrelated queued idle work; already-running preparation is discarded at its module boundary if it is no longer wanted. Cancellation does not interrupt arbitrary Native loading midway through a module.

The API is implemented and exercised through the benchmark driver. Actual ComfyUI completion hooks, embeddings/vision startup scheduling, sampler workers, and optional idle GPU promotion remain integration work. Two rank caches can still independently read the same checkpoint block on a miss; shared raw checkpoint reads are a later optimization. Both preparations are charged to the common budget.

## Executed local validation

`semantic_results/host_cache_ordered/result.json` records the current passing run using the exact TE/DiT block fixtures selected for the physical job, with their original checkpoint ranges and hashes verified by the builder. `host_cache_final/result.json` retains the preceding driver version.

- Two logical CUDA ranks execute trained TE block zero at S512 and trained DiT block zero at S8192. Each DiT block is used twice while its GPU weights remain resident.
- Two requests use different activation/timestep-conditioning fixtures. Both ranks match Native bit-exact.
- Rank zero executes the complete trained 36-layer public Native video decoder at compact geometry after each request. Both outputs match Native bit-exact.
- After the first decode, the host cache is deliberately trimmed to force checkpoint reads into the pageable cache; the OS file cache is not flushed. TE preload completes during a separately recorded diagnostic idle interval. The second request performs **no additional TE preparation** on either rank.
- Ten trained output comparisons pass. Additional actual execution checks verify changed checkpoint keys select changed weights, stale queued/in-flight preparation is cancelled/discarded, a spawned child and parent enforce one shared byte limit, and all reservations drain to zero.

Peak shared reservation: **12,393,347,456 bytes (11.54 GiB)**, below the 16 GiB cap. Final shared reservation: zero. This is reservation accounting, not a measured peak RSS. The diagnostic idle preload intervals are retained in the artifact; no dual-T4 speedup is inferred from local timings.

The initial refused-allocation run remains in `semantic_results/host_cache/`; the successful full-checkpoint run is `semantic_results/host_cache_v2/`; the final exact-fixture run is `semantic_results/host_cache_final/`. Each has source fingerprints. The video fixture is compact, and TE/DiT coverage here is block-level: this does not validate a complete production generation or an entire TE checkpoint retained in RAM.

## Physical gate

`build_host_cache_gate.py` requires the final local pass and matching source hashes before packaging. The job downloads approximately 0.88 GB of exact TE/DiT checkpoint ranges plus the 2.81 GB video checkpoint, instead of downloading the entire TE/DiT models. It runs the same rank function in two GPU processes with NCCL and the shared host budget. A parent-owned TCPStore retains its listener throughout worker startup.

Limits: 720 seconds setup, 480 seconds execution, 1300 seconds overall Kaggle timeout; no automatic restart. Failures stop the job and preserve logs/partial JSON. This is a physical correctness and host-admission gate, not a latency benchmark. Existing broad benchmark matrices are not repeated.

Version 1 completed in 256.83 seconds including 198.87 seconds setup and 57.96 seconds gate execution. All ten physical comparisons were bit-exact, with a shared reservation peak of 13,184,843,712 bytes and zero reservations remaining at shutdown. Evidence is retained in `kaggle_host_cache_output/v1/cache_results/`.

Log review exposed a lifecycle defect despite numerical success: rank 1 could enqueue NCCL completion and then start CPU prefetch before rank 0 finished the VAE stage. NCCL stream ordering alone does not block CPU disk workers. The driver now observes the reduced completion flag on the host once at this request boundary. It records monotonic timestamps and asserts that both ranks' prefetch starts follow rank zero's VAE completion. The corrected local run and physical version 2 both pass. Version 2 was a targeted validation after a diagnosed defect, not an automatic retry of the completed job.

### Accepted physical version 2

Evidence: `kaggle_host_cache_output/v2/cache_results/result.json`, per-rank JSON, manifest and NCCL logs. Every recorded source hash matches the current source. `passed`, `physical` and `post_decode_ordering_passed` are all true.

| Check | Result |
| --- | --- |
| Two requests, trained TE S512 / DiT S8192 blocks and complete compact public video decode | Ten comparisons, all bit-exact against Native |
| TE preload begins after first video decode completes | Rank 0: +79.78 ms; rank 1: +43.24 ms |
| Second TE request uses preloaded CPU weights | One RAM hit per rank; no new TE preparation on that request |
| Repeated DiT calls | One GPU weight load per rank per request |
| Peak shared managed reservation | 13,681,128,376 bytes (12.74 GiB), below 16 GiB |
| Final shared reservation / rejected admissions | Zero / zero |
| Gate execution / setup / total | 57.46 s / 186.34 s / 243.80 s |

The observed host exposed 33,658,318,848 physical bytes, consistent with the stated 32 GB class. The reservation peak is not a measured process-RSS peak. TE checkpoint preparation during the diagnostic idle interval took 818.19/886.52 ms for ranks 0/1, with the OS page cache unflushed. These are single diagnostic samples, not a benchmark distribution or a demonstrated end-to-end latency saving.

NCCL logs show two physical GPU ranks communicating via `SHM/direct/direct`, with direct NCCL P2P disabled. The gate does not test the separate `ActivationHandoff` cross-device copy implementation. It also does not establish full-model host-cache capacity or five-second generation performance.

Both submitted jobs completed; no further job was launched. The next work is to connect this qualified cache API and completion boundary to long-lived ComfyUI conditioning/sampler workers, then measure integrated cold/warm request latency against the actual Native cache behavior.

Use `host_cache_control.py` to inspect the current session. During a running version it reads only live logs, avoiding previous-version completed artifacts. Preserve the versioned evidence folders when collecting later runs.
