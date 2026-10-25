# Bounded Streaming and Stage Overlap for MiniMax-H3 TP2

**Review date: 2026-10-21. Status: revised design, not a newly implemented scheduler.**

The original filename is retained for links. “Zero bubble” is an optimization goal, not a latency guarantee. The useful idea is to prepare the next required weights while current work runs, then retire old storage safely. Unbounded cascading, fixed layer/step thresholds and unconditional device-first placement are not suitable for this runtime.

This review changes documentation only. It does not change operators, enable new prefetching, run H3 generation, consume Kaggle GPU time or qualify a new T4 environment.

## 1. Assessment and corrections

| Original proposal | Assessment | Revised rule |
|---|---|---|
| Overlap preparation with useful compute | Keep | Measure the exposed dependency and protect foreground deadlines |
| Reclaim weights after their final use | Keep with ownership fences | Extend existing cache/window lifetimes; do not delete entries or release budgets from an unrelated hook |
| TE has 48 layers; evict layers 0–35 at layer 40 | Incorrect for this checkpoint and too rigid | Both TE and DiT have 50 layers, indices 0–49; use actual invocation uses, readers and events |
| Step 19 is the final NFE for a 20-step sampler | Unsafe assumption | Require an explicit final-forward contract, or keep the existing end-of-sample boundary |
| `priority=-1` is low priority and lets foreground copies preempt it | Incorrect | Lower numbers mean higher priority; stream priority does not provide H2D/D2H bandwidth preemption |
| Greedily fill every free host/device slot | Likely to amplify contention and refaults | Admit a bounded current-stage horizon and one known successor head |
| Audio-first creates a free 3–6 second window | Unproven; graph-dependent | Compare decode orders only through an explicit workflow dependency, with identical Native decoding |
| Audio on both TP ranks | Obsolete | Audio VAE is Native single-GPU FP32 only |
| Video decoder is approximately 1.8 GB | Underestimates the current checkpoint | Decoder-stage payload is 2,450,218,528 bytes before construction/workspace overhead |
| Handoffs of exactly 0 ms and 27+ seconds saved | Unsupported | Report readiness, exposed waits, foreground slowdown and request latency separately |

### Preserved runtime contract

- Use the pinned local `ComfyUI` source, revision `d49e888586dd8ae012c0667b33466b815fee07f7`, and the frozen dependency versions. Local Python is `C:\ProgramData\miniconda3\envs\comfyui\python.exe`.
- FP16-fix is the pipeline default: TE/DiT residual divisor 64, explicit FP32 numerical islands, existing FP16 vision/Video VAE, and FP32 Audio VAE. The legacy FP32 tensor flow remains an explicit option and the numerical reference when assessing flow changes.
- Preserve current attention, INT8 weights, quantization, transport and LoRA order. This scheduler proposal must not introduce a new numerical approximation.
- Audio is decoded by the parent Native VAE on its configured single device. Do not restore channel, temporal or hybrid audio parallelism.
- Video query partitioning remains conditional on its existing compatibility/admission gates. Keep Native public normalization, tiling, chunking and fallback behavior.
- The current session serializes sampling. Concurrent H3 requests and a process-global multi-service memory broker are outside this design.

See [current operator choices](../guides/H3_OPERATOR_INTEGRATION.md) and [current VAE behavior](../guides/PARALLEL_VAE_INTEGRATION.md).

## 2. What the implementation already does

The proposed optimization must start from these existing mechanisms, rather than treating the runtime as an entirely serial loader.

| Mechanism | Existing implementation | Consequence for this proposal |
|---|---|---|
| Foreground-first CPU preparation and deadline queue | [HostWeightCache](../../patches/host_weight_cache.py), `_queue`, `_run` | Reuse its admission and cancellation; avoid another competing loader |
| Ordered GPU weight uses and asynchronous copies | [ResidencyWindow](../../patches/weight_residency.py), `prefetch`, `use` | Existing last-use retirement is already present |
| Copy/consumer lifetime fences | `_Entry.ready`, `_Entry.last_use`, `record_stream`, `_reap` | A Python forward return is not GPU completion |
| GPU lookahead 1–2 and separate CPU horizon | `RuntimeConfig`, `ResidencyWindow._adapt_gpu_lookahead` | Adaptive behavior is optional; lookahead bounds are not an additional free budget |
| Rank-local weight reads | [rank_local_int8](../../patches/rank_local_int8.py), [CheckpointReader](../../patches/windowed_tp.py) | Count physical per-rank reads and replicated state, not ideal full-tensor traffic |
| CPU preparation of next-request TE state during decode | [RankRuntime.dispatch](../../patches/comfy_tp_runtime.py), `decode_start` | This already begins at a decode boundary; it is not universally blocked by pending audio |
| Successful queue-idle TE/VAE encoder preload | [PromptBridge](../../patches/comfy_idle_lifecycle.py), `PhysicalTPService.request_idle` | Optional, cancellable and separate from within-request overlap |
| Encoder-first / decoder-last VAE loading | [StagedVAE](../../patches/staged_vae.py) | Preserve stage ownership instead of preloading a second complete VAE |

Important limitation: `HostWeightCache.prefetch()` **replaces** `wanted` and cancels queued background tasks outside the new set. `ResidencyWindow.prefetch()` calls it repeatedly. A separate cascade hook cannot safely submit a successor-only list: the two planners would overwrite each other's requirements. A combined plan with one owner is a prerequisite for cross-stage cache scheduling.

`remaining[key] == 0` means the key's declared uses are exhausted in that window. It is not proof that no subsequent TE batch or denoiser invocation will append another use. GPU retirement and retaining a useful pageable CPU copy are distinct policies.

## 3. Evidence and capacity facts

### 3.1 Do not turn historical gaps into promised savings

The original document provides no trace identity establishing its “27+ seconds of dead-time handoff” as one removable critical path. Historical [causal T4 analysis](../reports/CAUSAL_T4_RESULTS.md) explicitly leaves several large inactivity gaps only partially attributed. The [CPU audit](../reports/CPU_CRITICAL_PATH_AUDIT.md) distinguishes wait-associated exposure from removable latency; overlapping waits on two ranks must not be summed as request savings.

The [idle-residency report](../reports/TE_IDLE_RESIDENCY_RESULTS.md) also documents a lifecycle mismatch between a standalone runner and integrated ComfyUI. Its next-request overlap must not be silently treated as evidence for this proposed schedule. Older T4 source/dependency snapshots do not qualify the current frozen Linux stack.

### 3.2 Checked against current checkpoint headers / TP2 metadata

These are storage facts, not throughput or live-memory measurements. VAE payloads include the shared buffers selected by the staged loader.

| Object | Current bytes | Scope |
|---|---:|---|
| TE transformer layers | 50 layers, 0–49 | Current checkpoint |
| DiT transformer layers | 50 layers, 0–49 | Current checkpoint |
| TE block 0 per TP rank | 243,978,752 | Prepared TP2 parameter-size accounting, excluding preparation peaks/adapters |
| DiT block 0 per TP rank | 196,346,880 | Same scope |
| Video encoder stage | 360,747,328 | Checkpoint payload |
| Video decoder stage | 2,450,218,528 | Approximately 2.282 GiB, per complete decoder payload |
| Audio encoder stage | 345,477,248 | Checkpoint payload |
| Audio decoder stage | 259,672,288 | Approximately 247.64 MiB, FP32 payload; not peak VRAM |

The default retained block cache is 3 GiB per rank. It cannot retain every TE or DiT block across invocations. LRU alone therefore does not eliminate cross-step disk rereads. Nor can evicting a chosen layer range guarantee a fixed number of freed GiB: only currently live entries can release storage.

FP16-fix reduces activation storage at selected boundaries; it does not halve the INT8 checkpoint weights. Video query decoding also retains the parent decoder while workers load their replicas. Count all replicas and their workspaces on the physical devices.

Local review facts and the original document are preserved under `D:\attention_eval\streaming_architecture_review_20261021`. Only checkpoint headers/meta shells were inspected; no GPU benchmark was run for this review.

## 4. Optimize the dependency graph, not a fixed stage list

A simplified dependency graph is:

~~~mermaid
flowchart LR
    Inputs[Prompt and references] --> TE[Native vision and TE frontend]
    Inputs --> Enc[Required Native reference encoders]
    TE --> Ref[Token Refiner on rank 0]
    Ref --> Pack[Pack conditioning and latents]
    Enc --> Pack
    Noise[Identical initial noise] --> Pack
    Pack --> DiT[TP2 denoiser calls]
    DiT --> Sampler[Native sampler completes]
    Sampler --> Audio[Native single-GPU FP32 audio decode]
    Sampler --> Video[Video decode with existing query option]
    Audio --> Output[Native output and AV assembly]
    Video --> Output
~~~

Actual ComfyUI graph dependencies, caching and request options determine which nodes execute and their order. The diagram does not impose reference-encoder-before-TE ordering or concurrent audio/video execution.

The decoders consume separate latent branches, but this is computational separability after sampling, not statistical independence of audio and video. The public stereo audio latent is `[B,32,2,T]`; do not omit its stereo dimension or change its batching. The video latent is `[B,24,T,H,W]` for this model. Keep Native sample rate, FPS, length trimming, AV alignment and metadata.

### Optional audio-first order

Audio-first is a valid candidate for a workflow that already requires both decoded outputs. It can finish the smaller decoder first and leave video as the final decode stage. It does not create new compute time or guarantee a free prefetch window: video still uses memory/bandwidth, and CPU decode-time lookahead already exists.

Implement a future order experiment through a combined decode node or an explicit execution dependency in the workflow. Independent graph branches are not ordered by their visual placement. Do not intercept sampler completion and eagerly call raw VAE decoders: that can bypass public planning, duplicate decode work, change previews/cancellation and run an output that the graph did not request.

Keep the existing order as the reference. Consider optional audio-first only if total request latency improves without harming the intended output/preview latency. Measure both orders; do not assume audio takes 0.3 seconds or video takes 3–6 seconds on the target shapes.

## 5. Bounded admission and scheduling

### 5.1 Choose the next useful head, not the largest resident set

Start with **CPU-first successor preparation**. The pageable/pinned/GPU path remains necessary; “device-first” is a placement preference, not direct disk-to-VRAM loading.

Use three classes under one planner:

1. **Required now:** current operator, active quantization/communication scratch, current outputs and admitted H2D sources.
2. **Required soon:** the bounded current-stage CPU/GPU horizon and the earliest dependency-ready successor head.
3. **Speculative:** compatible next-request immutable weights, only when foreground slack and budgets permit.

Within a class, prefer the earliest actual consumption deadline. Do not evict a nearer-use weight to stage a later one. Promote a matching speculative entry to foreground without rereading/rebuilding it. Stop background submission on foreground delay, admission denial, pressure or request cancellation.

Initially allow only the current horizon plus **one known successor stage**. Cascading to more stages needs measured avoided waits, not merely spare RAM. A later weight may be cheaper to read when needed than to retain through page-cache eviction and reconstruction.

### 5.2 A cost model for experiments

For a successor head, a useful approximate comparison is:

~~~text
exposed handoff ≈ max(0, required-head preparation time - usable overlap)
                  + control/teardown work that remains serial
net benefit     = avoided exposed waiting
                  - added foreground delay, rereads and reclamation work
~~~

This is an experiment model, not an equality for arbitrary overlapping resources. Reconstruct the actual dependency/resource critical path from timelines. A full-model-ready condition is unnecessary if the next operator's required head is ready; an assembled CPU module is not GPU-ready.

Maintain separate stage estimates for read, optional decompress, assemble/fold, pin, H2D, compute and communication. Increasing queue depth cannot fix insufficient sustained SSD, CPU, host-memory or PCIe bandwidth. Both ranks must reach foreground collectives in the same order; one ready rank does not remove the other rank's exposed wait.

### 5.3 Candidate placement by active work

| Active work | First candidate | Limits / prerequisites |
|---|---|---|
| Required reference encoding | TE startup or first needed blocks on CPU | Follow actual graph dependencies; do not evict an encoder still in use |
| TE tail | Rank-0 refiner/projection CPU state and a bounded DiT head | Merge with current TE deadlines; no hard-coded layer 40 trigger |
| Refiner / packing | DiT block head, then ordinary current-stage horizon | Rank-0 ownership and existing broadcasts retained |
| DiT before a proven terminal forward | Normally retain current policy | Repeated weight use dominates; deep decoder speculation can cause rereads |
| Proven terminal DiT forward | Immediate decode successor on CPU | Must not displace required current blocks or output-head workspace |
| Native audio decode | Video decoder CPU preparation, if video is actually next | Audio stays on one Native device; no decoder replicas in TP workers |
| Video decode | Compatible next-request startup CPU weights | No speculative GPU TE weights until coexistence and cancellation are qualified |
| Successful queue idle | Existing optional encoder/TE residency | Preserve request identity, queue checks, cancellation and admission |

This table proposes scheduling opportunities. The new within-forward cross-stage planner is not implemented by the existing `decode_start` callback.

## 6. Memory accounting across processes and devices

### Host memory

Keep one managed host ledger shared by parent and the two rank workers. Charge physical storage ownership for pageable prepared modules, pinned copies, decoder staging, preparation peaks and any new compressed buffers. A parent-owned VAE allocation and a worker-owned tensor are not the same object.

Use separate checks for managed reservations and physical/container pressure:

~~~text
managed live + new managed reservation <= managed host limit
observed cgroup current + new unmaterialized charge + safety margin
    <= effective cgroup limit
~~~

These are conservative admission estimates, not OOM guarantees. Account for construction peaks before launching work. Avoid adding charges already represented in observed usage twice. Include file cache, unmanaged Python/native allocations and cached pinned storage in pressure analysis. Do not add every process RSS as if shared mappings were independent physical copies.

Read the effective cgroup limit rather than assuming “30 GB”, and distinguish decimal GB from GiB. `psutil` host availability alone is insufficient in a container. Existing cgroup-aware behavior is optional; enabling it is a separate tested change. The kernel defines `memory.current` as usage by the cgroup and its descendants. [Linux cgroup v2 memory interface](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files).

### Device memory

Admission is per physical device and aware of the process/context owning each allocation. Include active/prefetched/retired weights, Native parent VAE, worker video replicas, activations, INT32/FP32 operator scratch, NCCL buffers, outputs and allocator slack.

A proposed successor allocation must fit its byte budget **and** conservatively estimated physical headroom after pending allocations and the remaining foreground peak. Calibrate that peak by stage, tensor shape and current FP16-fix/operator policy, with an additional margin. A fixed 1.5 GiB reserve and a delayed `mem_get_info()` poll cannot certify safety.

The current `_room()` considers this process's reusable inactive allocator storage. That is not transferable credit for another process on the same GPU. For example, rank-0 allocator reservations cannot directly fund the parent Audio VAE. Releasing that memory to the driver requires an owner-controlled, fenced transition. Do not independently count the same driver-free bytes in multiple process admissions; a future cross-stage GPU broker must reserve them coherently.

Speculative OOM must cancel/retire speculative work and preserve foreground progress. Do not retry indefinitely or turn a collective failure into an independent per-rank fallback.

## 7. Retirement means ownership completion

Reuse the existing window/cache lifecycle. Do not mutate `entries` or call `HostBudget.release()` from a new layer hook.

~~~mermaid
flowchart LR
    Planned[Planned and budgeted] --> CPU[CPU prepared]
    CPU --> Pin[Pinned source lease]
    Pin --> Copy[H2D in flight]
    Copy --> Ready[Ready event]
    Ready --> Use[Consumer use]
    Use --> Retired[Logically retired]
    Retired --> Done[All consumer events complete]
    Done --> Reusable[Owner storage released or reusable]
~~~

The pageable cache can retain its own immutable copy after the pin-copy reader exits. The pinned source lease must survive H2D completion. The GPU module must survive every consumer stream/event. Shared CPU backing and replicated tensors require their own owner counts; do not free them when only one borrow ends.

`ResidencyWindow.use()` already records ready/last-use events, tracks ordered uses and retires exhausted entries. `_reap()` waits for completion before retiring ownership. Extend that mechanism rather than promising immediate physical memory release when Python returns. Preserve `record_stream` or an equivalent complete allocator-lifetime protocol. [PyTorch 2.12 CUDA stream semantics](https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-streams).

CUDA allocations can remain reserved after tensor references disappear. CPU and pinned allocator caching can likewise leave physical pressure after logical leases are released. Reclamation is not a microsecond slot swap. Keep the current explicit stage-boundary fences and cache cleanup until an incremental replacement is proved correct; removing `_host_emptyCache()` or global synchronization is a separate ablation, not a prerequisite.

### TE and DiT last-use rules

- TE eviction uses the actual invocation plan, including multiple encodes, prompt branches and any repeated calls. Do not use fixed layer ranges or assume an autoregressive/KV-cache path that the current prefill-only wrapper does not support.
- Ordinary DiT calls may need the same weights again. Do not purge the CPU cache on each local `remaining == 0` merely because a GPU entry is retired.
- Sampling steps, progress callbacks and denoiser NFEs are different counts. A sampler may invoke the denoiser multiple times, use partial schedules or have extra evaluations. `step == total_steps - 1` is insufficient.
- Early terminal eviction requires a request-bound final-forward lease from a sampler integration that can prove no later use. This contract does not exist merely by adding `step_notify`. If unavailable, use the current `end_sample` teardown.
- Video decoder blocks and latent tiles can be reused throughout one Native public decode. Do not apply DiT-style last-layer retirement to the first video chunk/tile. Retain the exact public invocation's reuse plan.
- Audio/video latents and output buffers remain live until their actual graph consumers finish; executing audio first does not free the video latent.

## 8. Copy scheduling, rank agreement and cancellation

### Streams are not a bandwidth scheduler

CUDA uses lower numbers for higher stream priority. `priority=-1` therefore does not create a low-priority stream. Priority is a scheduling hint, does not preempt already-running work and does not affect host/device memory-copy priority in the documented runtime. Query the supported priority range; a positive value can be clamped back to the default. [CUDA 13.2 stream priority API](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__STREAM.html).

Use ordinary streams and explicit admission instead. Bound speculative transfers in flight and pause new submissions while foreground H2D/NCCL has a deadline. Larger transfers already in flight are not cancellable bandwidth reservations. Chunking may reduce interference but adds launches/copies; choose sizes from measurements. Two copy engines do not imply two independent H2D paths. Validate the actual T4 topology rather than assuming independent PCIe bandwidth per GPU.

Pinned sources and appropriate independent streams enable overlap; pageable transfers may stage or block. Pinning itself consumes CPU bandwidth and memory. Measure it separately. [CUDA transfer and overlap guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation).

### Owner-thread and distributed rules

- `HostWeightCache` remains CPU-only. Decoder construction/preparation workers must not launch speculative CUDA work behind the runtime's owner thread.
- A rank busy inside one denoiser RPC cannot service another parent RPC at each layer. Within-call planning needs a nonblocking local tick at a defined window boundary; a callback must not synchronously call back into the parent service lock.
- Rank-local background preparation performs no collectives. Foreground phase commit preserves the existing communicator and identical collective ordering on both ranks. Never insert a rank-0-only prefetch collective into the ordinary TP sequence.
- Token Refiner/input projections are rank-0 preprocessing consumers. Rank 1 follows existing receive/broadcast contracts; it does not request refiner slices. Native audio belongs to the parent, not rank 0's worker process.
- Checkpoint/adapters, offline bake identity, rank, operator policy, tensor flow/gauges, codec identity and preparation schema must participate in compatible keys. Prevent double folding when a prepared object is promoted.
- Cancellation marks a generation epoch, stops new jobs and discards stale completions. Native read/decode calls already running may finish; retain their leases until they do. Retire in-flight copies by events, not by erasing their references.
- Define a consistent lock order. Do not wait on a Future or CUDA event while holding a planner lock needed by its completion path. Shutdown drains ownership once and returns every reservation once.
- Keep the existing queue-success and identity checks for idle work. Before an unknown next request, prefetch only compatible immutable weights; never speculative prompt activations, reference latents, sampler state or RNG draws.

## 9. Compression is an independent optional producer stage

The [local TE codec study](../reports/TE_COMPRESSION_LOCAL_20261020.md) supports ZSTD as an initial candidate, not as an integrated loader. Its ZSTD-1 result is approximately 1.124× compression and 5.648 decimal GB/s raw decode with six workers on the tested local corpus. Those are RAM-only CPU results; they do not predict target SSD or end-to-end T4 throughput. The experiment excluded embeddings and vision.

If evaluated later, use existing libraries and independent rank-local frames; 1/4/16 MiB are initial experimental chunk sizes, not selected production constants. Bound compressed input, decompressed output, assembly and pin queues under the same host accounting. Give foreground reads/decode capacity before speculative frames; avoid nested codec threads competing with two ranks, LoRA assembly and the main thread.

Keep safetensors authoritative. An optional sidecar must bind source identity, tensor/rank ranges, offsets, dtype/layout, codec version and integrity checks, and reproduce bytes before existing folding/casts. A corrupt or incompatible frame must fail validation before execution; an uncompressed fallback must be explicit and identical. Do not change dependency versions or import the isolated codec experiment's packages into production implicitly.

`os.pread` avoids shared seek-position mutation; it is not Linux `O_DIRECT` or zero-copy. The current reader copies returned bytes into destination storage and retains a Windows/nonstandard-descriptor fallback. Compression integration must not claim to remove assembly, allocation or pinning costs.

## 10. Implementation sequence and boundaries

All additions below are proposed. Keep each independently selectable and initially disabled; current FP16-fix/audio defaults are unaffected.

| Phase | Concrete work | Entry/exit condition |
|---|---|---|
| P0: observe | Add bounded epoch/spec IDs and timing for read, prepare, pin, ready, wait, compute, retire and reclaim in the existing diagnostic layer | Identify a foreground dependency and a feasible successor overlap; no new GPU work |
| P1: CPU successor head | One combined deadline plan in `HostWeightCache`; retain current required lookahead, add a bounded known successor head, cancel/reprioritize coherently | CPU fault/pressure/cancellation tests pass; no nearer-use evictions or duplicate preparation |
| P2: incremental retirement | Extend `ResidencyWindow` owner APIs where a real last-use gap remains; reuse current `_evict`/`_reap` fences | Delayed-copy/consumer and repeated-use tests pass; no early budget credit |
| P3: GPU stage overlap | Add a shared physical-device admission/ownership plan and an explicit successor-window promotion path | Same-process vs parent/worker allocator ownership, remaining activation peak and coordinated phase commit are qualified |
| P4: optional order/codec experiments | Audio-first workflow dependency and lossless compressed input, evaluated separately | Isolate each contribution; no full H3/Kaggle generation under current task constraints |

Do not build GPU co-residency first. `_phase()` currently drops the old model/window, constructs a new shell and may clear allocator caches. A prefetched successor attached to the old window could be closed accidentally. Likewise, a rank-local replica cannot be handed to the parent without a new ownership/IPC design. P3 must specify that promotion explicitly before weakening teardown.

The first practical optimization is **one CPU deadline plan for the immediate successor plus accurate physical-memory accounting**. It has lower implementation risk than final-NFE inference, global decode interception or broad GPU co-residency.

## 11. Validation and performance decision

Use the current TP2 shapes, FP16-fix policy and frozen operators. Keep identical source weights, adapters, tensors, random state and Native public plans. A scheduling-only change should preserve the tested output bytes; if concurrency changes a kernel path/output, investigate rather than quietly accepting a wider tolerance.

Under the current no-full-H3 constraint, start with CPU fixtures, delayed-Future/event tests and component/trace replay. Do not synthesize an end-to-end speedup from those results. Do not consume Kaggle GPU time for setup checks. Frozen-stack Linux/T4 ABI/tool availability remains an explicit blocker until independently resolved.

| Gate | Required cases |
|---|---|
| Lifetime and accounting | Delayed H2D, delayed consumer, repeated key, aliases, cancellation during read/pin/copy, failure, shutdown, exactly-once lease release |
| Planner correctness | Successor merge without overwriting current wanted keys; foreground promotion; both ranks; rank-0-only state; multiple encodes/NFEs; no starvation |
| Native contracts | Audio single-GPU/no audio-worker commands; video short/native fallback and repeated tiles; FP16 folding once; unchanged latent/conditioning metadata |
| Resource pressure | Managed host limit and cgroup pressure; pinned cache; parent/worker allocator ownership; speculative OOM; foreground output-head peak |
| Performance ablations | Existing schedule, CPU-only successor head, retirement alone, optional GPU promotion, each decode order, each optional codec |

For a future qualified T4 component run, collect Nsight Systems CPU/CUDA/NCCL timelines and correlate spec IDs with producer readiness. Use Nsight Compute only on representative affected kernels if copy overlap changes their behavior. Profile only bounded windows because instrumentation can change the schedule.

Report warmed repeated measurements with compilation/loading separated, cold vs warm file cache stated, and median/tail latency. Include:

- Exposed foreground wait and successor head readiness at the actual transition.
- Active-stage compute/communication slowdown, H2D in flight and all-rank readiness.
- Disk bytes, duplicate reads, useful prefetch hits, cancelled/evicted-before-use bytes and page-cache refaults.
- Managed reservations, process allocator allocated/reserved/active bytes, device-wide use, peak pinned/RSS and cgroup pressure/OOM events.
- Reclaim/synchronization cost and wait associated with the slower rank; do not sum rank-overlapping intervals into savings.
- Output correctness and cancellation/cleanup results, alongside any latency benefit.

Accept a change only when it reduces the chosen critical-path metric without an unacceptable foreground regression or memory increase. A ready-to-use successor is a measured state, not proof of a 0 ms transition. Whole-request throughput, preview latency and end-to-end savings remain unclaimed until an explicitly authorized full-pipeline evaluation is available.
