# MiniMax-H3 dual-T4 integration plan

**2026-10-11 default update:** Native-aligned video Query Partitioning and FP32 audio Channel Parallel are integrated into the TP2 ComfyUI workflow. Native public planning and encoding are retained. The focused local public-decode suite passes video bit-exact checks; audio deliberately retains its diagnosed rounding difference and still fails the old 1e-5 strict gate. See [current implementation and remaining physical qualification](../guides/PARALLEL_VAE_INTEGRATION.md). This supersedes the earlier Native-only VAE decisions below.

Date: 2026-10-09. Updated for the user's explicit single-request latency objective and cross-model asynchronous sliding window. The residency core and persistent RAM-cache mechanism are implemented and qualified within the scopes below; complete production integration remains in progress.

Sampler milestone: Native MODEL/CLIP/VAE adapters, persistent TP2 workers and actual sampler-use scheduling are implemented. Full-depth local trained conditioning/sampling/decode/next-request comparisons pass bit-exact. The physical dual-T4 process/NCCL gate passes eleven bit-exact comparisons in the two-layer TE/DiT scope, including guided sampling, interruption recovery, full Native vision/video VAE and next-request TE reuse. See [current integration use and qualification](../guides/COMFYUI_SAMPLER_INTEGRATION.md). Complete queued workflow acceptance, integrated latency, cross-model predictive GPU loading and asynchronous activation transport remain future measured work.

## Decision

Optimize complete single-request latency, including model transitions. Use optimized TE TP2 and DiT TP2 as the compute core, with one residency scheduler spanning layers, repeated invocations and model boundaries. Treat two-slab video decoding as a separate quality/performance option; qualify audio channel parallelism before enabling it. The complete seven-component proposal is a candidate, not a measured optimum. Multi-request throughput and PP cohort tuning are outside the current implementation scope.

| Component | Evidence | Implementation decision |
| --- | --- | --- |
| TE optimized TP2 | Full text decoder: 0.848 → 0.659 s at S512, 3.923 → 2.722 s at S2048, 9.632 → 6.555 s at S4096; bit-exact | Default for the current single-request objective. Native vision/DeepStack conditioning is now integrated and locally qualified; integrated physical latency remains unmeasured. |
| DiT optimized TP2 | Complete audiovisual S8192 step: 52.249 → 31.616 s; bit-exact | Default candidate at S8192. Actual compact Native sampling is locally qualified; full-model S16384/S32768 and integrated loop latency remain unmeasured. |
| Video VAE two slabs | Physical slab scheduling passes; full raw decoder L2 error 7.526–11.395%, failing the original 1% quality gate | Optional approximation. Preserve configured native public decoding as the initial default; assess generated-video quality before selecting slabs. |
| Audio VAE channel parallel | Native code processes stereo independently. Historical 5 s: 517.98 → 302.82 ms; L2 1.94788e-5. All nine strict comparisons failed; maximum absolute error ≤1.59e-4 | Promising mathematically valid partition with unresolved floating-point differences. Diagnose batch-dependent kernels; do not silently relax the strict gate. |
| Adaptive sliding window | No integrated two-GPU result | Schedule weight preparation, H2D loading and event-safe eviction across both layers and model boundaries. Size the window in bytes and compute lead time, using actual reuse and memory constraints. Preserve attention domains and native VAE temporal rules. |
| Persistent GPU weight cache | TP peaks approach 13 GiB per GPU; visible T4 capacity was 14.56 GiB per GPU | Keep reusable DiT weights across invocations where capacity permits. Reclaim weights after their last use to prefetch the next model before the current one finishes. Budget CPU backing and pinned buffers too. |
| Asynchronous stage handoff | No integrated gain measured; physical TE PP overlapping requests corrupted outputs despite local simulation passing | Begin with synchronized ownership. Add explicit producer/consumer events and buffer lifetimes, then prove transport correctness before measuring overlap. |

Sources: [latest TE/DiT results](../reports/REF2VA_T4_RESULTS.md), physical VAE results (`kaggle_repaired_output/MEASURED_RESULTS.md`, local-only evidence), [audio retest findings](../reports/RETEST_REPORT.md), [slab limitations](SLAB_EXPERIMENT_PLAN.md).

These are component results with different timing scopes. They cannot be multiplied into a full-generation speedup. Raw video measurements cover only one/two latent frames, not five-second public decoding. Slabs include CPU input transfers, while their native comparator starts with resident GPU input.

## Disk-to-RAM scheduling and preparation for the next request

The user explicitly proposes preloading TE after VAE decoding finishes. Adopt this as the default idle preparation policy for repeated single requests using the same model configuration. It reduces the next request's startup latency; foreground generation remains the scheduling priority.

The intended path is checkpoint storage → bounded pageable RAM cache → small pinned staging pool → per-GPU weight windows. Disk read-ahead can run further ahead than H2D lookahead because their capacities and deadlines differ. Keep the host cache and I/O coordinator alive across request-window teardown. Pinned staging is part of the total host budget, with additional headroom for checkpoint construction, output frames, the OS and ComfyUI. Do not preload or pin the entire approximately 27 GB TE checkpoint by default on the 32 GB local machine.

| State | Host scheduling policy |
| --- | --- |
| Active TE/DiT/VAE work | Required weights and imminent misses have priority. Retain reusable DiT host shards when GPU capacity forces recurrent reloads. Avoid duplicate reads/full CPU block construction in the two rank workers; use a shared immutable source or prepare rank shards from one admitted read. |
| VAE decoding complete | Start bounded TE preparation in execution order: required startup/embedding weights and the first block groups. Release completed VAE backing only after remaining audio, temporal chunks, output transfers and other consumers relinquish ownership. |
| Idle before the next request | Grow the TE RAM window within available-memory limits. Optionally promote its first groups to idle GPUs once current GPU consumers finish and capacity permits. Neither operation delays returning the completed result. |
| Next request arrives | Promote matching cached groups to foreground ownership and immediately prioritize its actual graph. Cancel remaining speculative reads at bounded chunk boundaries if checkpoint, patches, dtype/backend or shard configuration differs. A cache miss falls back to normal loading. |

Disk prefetch during the VAE tail is a later optimization: enable it only if measured storage/CPU/RAM contention does not delay decoding or output encoding. Post-decode TE preload is the conservative starting point. When reference encoding/vision must run before TE in the next graph, those actual dependencies take priority over speculative TE GPU residency. If unchanged conditioning is cached and TE is skipped, its weights should not displace the next required stage.

A session-level cache must distinguish immutable weight payloads from request-dependent activations and conditioning. Cache identities include checkpoint identity, effective patches, dtype, backend and rank layout. Bound raw and prepared copies together, and release unused speculative entries under memory pressure. Retaining a CPU shard avoids another disk read and preparation pass; it does not avoid H2D unless that shard also remains on GPU.

Validation should execute two consecutive requests with actual checkpoint reads and changed inputs, compare both against Native, and exercise a configuration change between requests. Report cold storage/startup, warm OS page-cache startup, explicit RAM-cache startup and GPU-resident startup separately. Measure read bytes/time, preparation time, host peak, residual load stalls and current/next request completion latency. Record the idle prefetch interval so setup work is visible rather than silently excluded from latency claims.

Current status (2026-10-09): `HostWeightCache` now persists prepared pageable weights across `ResidencyWindow` lifetimes and queues bounded idle TE preparation independently of H2D lookahead. Both GPU workers share a 16 GiB host reservation budget on the 32 GB machine; retained cache capacity is 4 GiB per rank. Two consecutive trained CUDA requests passed locally and physically on dual T4, including post-decode TE reload/reuse and process-shared budget enforcement. Physical log review found and repaired a missing CPU observation of NCCL completion; the targeted follow-up verifies both ranks preload only after VAE completion. Its peak managed reservation was 12.74 GiB. See [implementation and qualification scope](../reports/HOST_CACHE_IMPLEMENTATION.md). Actual ComfyUI completion hooks, shared raw checkpoint reads and idle GPU promotion remain integration work.

## Runtime and memory lifecycle

The actual graph is conditioning → denoising → audiovisual decoding → output assembly. Conditioning includes native tokenization, reference vision embeddings, video/audio reference encoding where applicable, tags and packing; it is not just the benchmark's text-only TE.

1. Run reference encoders and vision preprocessing under bounded residency. Preserve native reference order, masks, positions, normalization and seed-dependent augmentation.
2. Load TE startup shards, run optimized TP2 and prefetch its remaining layers as needed. After each shard's final use completes, reclaim its storage and prefetch DiT startup modules into the freed space while the TE tail runs. Retain conditioning outputs. Vision and decoder residency must be scheduled rather than assumed to fit simultaneously.
3. Start DiT once conditioning and its first required modules are ready; overlap later loads with earlier computation if profitable. Use native once-per-sampling text refinement and layout preparation. Retain transformer weights and reusable quantization/layout state throughout the sampler loop where capacity permits. Preserve the audio/video sigma schedules, guidance behavior, AdaLN, complete reference packing and both heads.
4. During the final DiT invocation, reclaim each shard after its last use and prefetch the selected VAE startup modules. Keep final heads and shared parameters until their actual final consumers complete. After final latents are ready, begin decoding with available modules and continue prefetching as necessary. Compare video/audio decoding sequentially with both GPUs against independent native video/audio branches on one GPU each; minimize time until both outputs are ready.
5. Stream finalized video frames to host storage/output encoding using bounded buffers. Preserve audio sample rate/count, video frame count and mux alignment.

The GPU cache has an active-phase budget, not an obligation to keep every model resident. TE and DiT checkpoints alone total approximately 48 GB on disk; keeping all canonical tensors in host RAM may also exceed the machine's RAM. Use bounded CPU/pinned caches and checkpoint-backed storage rather than duplicating whole checkpoints. GPU eviction of a read-only shard need not copy it back to CPU when a canonical backing copy already exists.

Cache keys include checkpoint identity, effective model patches, dtype, backend, device and shard layout. Conditioning caches additionally depend on reference content, prompt, tags, layout and augmentation settings. Do not cache changing latent projections, timestep-dependent modulation or attention results.

Start with fully resident DiT shards between repeated invocations when measured capacity permits. Use windowed startup and cross-model transitions to reduce load stalls even when a whole model eventually fits. Stream recurrent layers only when required by capacity or when an integrated measurement demonstrates a lower request latency. An adaptive prefetch window should use actual free memory and observed transfer stalls, reserve activation/quantization/NCCL workspace, and retain evicted tensors until queued work is complete. Allocated memory is contained within reserved memory; do not add both as disjoint usage. Measure driver memory separately.

All proposed TP/slab/channel stages can occupy both GPUs. Dependencies and weight capacity limit cross-stage compute overlap. Initial overlap targets are CPU preparation, bounded weight loading after memory becomes available, activation transfers and output encoding. Additional streams are not evidence of useful overlap; H2D traffic can compete with NCCL and computation.

## Unified sliding window for single-request latency

The window follows the ordered weight uses of the entire request. Model boundaries do not force a GPU fence or require the next model to be loaded in full. A separate retained set holds weights with valuable upcoming reuse; a bounded lookahead set holds weights being prepared/transferred for imminent use. Full DiT residency is one valid operating point of this same scheduler.

| Point in the request | Eviction and prefetch policy |
| --- | --- |
| TE startup | Fetch embeddings and first required blocks first, then overlap later block loads with execution where measured compute permits. |
| TE tail | Reclaim completed TE shards after their final use. Prioritize DiT conditioning/refinement, input projections, first blocks and required metadata rather than loading the checkpoint in file order. |
| Repeated DiT invocations | Retain reusable shards when they fit. A simple FIFO window would reload the model on every invocation and can increase latency substantially. |
| Final DiT invocation | Gradually replace completed DiT shards with VAE startup weights. The scheduler must know the actual final invocation, accounting for guidance branches, multistage samplers and any extra evaluations. |
| Repeated VAE temporal chunks | Retain decoder weights for the next chunk when feasible. Spatial slabs do not eliminate this reuse. |
| Final decode/output | Release weights after final use and overlap output transfer/encoding with remaining GPU work, subject to output dependencies. |

For an upcoming weight bundle, choose enough lookahead that the compute available before its use can cover remaining CPU preparation plus H2D transfer. Estimate these times under real two-GPU TP traffic. A deeper window cannot fix a sustained transfer-bandwidth deficit; it may only consume activation headroom and delay current work. Adjust at block-group or phase boundaries using observed transfer waits and compute slowdown, with conservative bounds to avoid oscillation.

The objective is lower complete request latency: hidden transfers are useful only if they save more stall time than the contention they introduce. At measured S512, TE computation lasts only about 0.66 s, with a shorter tail available for cross-model prefetch. It is therefore unjustified to assume that all DiT loading can be hidden. The final S8192 DiT invocation offers a much longer potential overlap interval, but VAE loading still needs physical measurement.

Implement bounded checkpoint → CPU preparation → pinned staging → GPU transfer queues. Prepare the exact native INT8 payload, scales and layout without a transient full FP32 model. Hold the pinned source until its H2D completion event; hold GPU weights and quantization metadata until all consuming kernels/collectives finish. Release immutable GPU copies without D2H when a valid canonical backing copy exists. Reserve D2H for state or outputs that actually need preservation.

Use per-rank compute/copy streams and event dependencies. Both ranks execute the same model/collective order; each waits on its own required weight readiness. Avoid a new CPU or global GPU barrier at every layer. Memory accounting and slot reuse must remain in the owning GPU process so a model handoff can reuse freed allocator storage without process teardown. Existing native prefetch hooks are the starting mechanism; the new capability is request-wide scheduling across model boundaries and repeated uses.

The minimal additional benchmark compares synchronous stage loading, intra-model streaming, and intra-model plus cross-model streaming with identical TP math. Measure full request latency, startup and transition stalls, recurrent reload bytes, NCCL/compute slowdown under prefetch, and peak GPU/host memory. Report cold local-checkpoint execution and warm execution separately. Network downloads are setup. No throughput cohorts are needed for this objective.

Pinned asynchronous copies and cross-stream ownership follow the [PyTorch 2.11 CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-streams) and [pinned-memory guidance](https://docs.pytorch.org/docs/2.11/notes/cuda.html#use-pinned-memory-buffers). These mechanisms permit overlap; they do not establish a speedup on the target T4 topology.

## Reuse and implementation boundaries

| Area | Starting point | Required work |
| --- | --- | --- |
| TP math | `patches/optimized_tp.py`, `patches/tp_qwen3vl.py`, `patches/tp_dit.py` | Reuse validated native INT8 GEMM, row-max reductions and rounding-preserving rotation. Keep native fallbacks; do not introduce new sharding semantics. |
| Model integration | `benchmarks/full_models.py`, native `comfy/model_base.py` and MiniMax model/TE code | Move proven loading/placement behavior into a production adapter. The old `custom_te_loader.py` TP branch returns `None` and is not a working integration. Preserve native vision/deepstack and conditioning metadata. |
| Residency | Native model-management/prefetch hooks | Extend existing mechanisms to two rank-owned model instances. Add a bounded residency policy and explicit phase transitions; avoid a competing global cache manager. |
| Workers | Existing TP rank entry points | Keep one long-lived worker per GPU, device-private scratch and deterministic collective order. Use a parent-owned TCPStore listener to remove the observed port allocation race. |
| Transfers | `benchmarks/full_models.py:move` / `PipelineRunner` as diagnostic examples | Implement explicit source ownership through copy completion, destination ownership through consumption, and producer-event → copy-event → consumer dependencies. Propagate worker failures and drain queued work. The current overlapping PP scheduler is not a validated handoff implementation. |
| Video | `patches/slab_video_vae.py`, native `comfy/ldm/minimax/vae.py` | Insert optional slabs inside the native temporal decode loop. Preserve denormalization, padding, dropped frames, overlap blending, finalization and output-buffer behavior. |
| Audio | `patches/multi_gpu_audio_vae.py`, native audio VAE | Load trained replicas independently; compare mono/stereo batch execution at intermediate convolutions. Diagnose kernel/accumulation changes and transfer ownership before enabling channel parallelism. |

Native integration already refines text once in `MiniMaxH3.extra_conds`; the benchmark direct model entry stages preprocessing and may refine text each timed call. Preserve the production behavior for both Native and TP in the integration comparison. Native also already provides dynamic prefetch hooks. Attribute improvements against this real baseline, not an artificially uncached version.

The native video VAE already has temporal chunking and spatial tiling. “Matches Native” therefore means matching the same configured public path, with its exact padding/tiling settings. Separately report differences from full-width raw decoding; do not confuse the two references.

## Implementation sequence and acceptance

### 1. Establish a complete synchronous integration

Connect optimized TP to the real ComfyUI Ref2VA model interfaces, using the validated INT8 checkpoints. Keep native vision/reference encoding and public VAE APIs. Match prompt/reference preprocessing, seed, sampler, guidance, schedules, lengths and dtype. The supplied workflow uses other checkpoint formats; do not transfer INT8 results to that stack without qualification.

Deliverable: one complete synchronous Native/TP path with stage load/unload accounting and a working fallback. No new parallel strategy or large performance sweep.

### 2. Execute targeted local simulations on the 5060 Ti

Run actual trained CUDA computation on two logical rank streams. Retain full conditioning/attention semantics. Stream trained layers where the single GPU cannot hold both logical rank models; never present local times as dual-T4 performance.

For handoff, force separate destination storage and actual copies. The previous same-device `.to()` could alias the input and concealed a transport hazard. Exercise delayed consumers, rapid buffer reuse, alternating shapes, distinct metadata and repeated requests with event-owned buffers. The suspected source-lifetime fault remains a hypothesis until reproduced and isolated.

For audio, compare native stereo versus separate mono execution, locate the first differing operator and try a native-equivalent kernel configuration. Do not remove the historical failed gate. If the fixed configuration cannot satisfy the retained gate, keep native audio as default and document any proposed numerical policy prospectively.

For video, run the public temporal/padding path on generated latents. In optional slab mode, assess seam-crossing motion, thin structures, textures, temporal flicker and final pixel output against the configured native path and a feasible full-width control. Changing chunk sizes or spatial context is a quality decision, not a transparent cache optimization.

Deliverable: executable simulation evidence and fixes for found defects, outside the production hot path. A local pass cannot guarantee physical NCCL/cross-device correctness.

### 3. Add active-model caching and bounded prefetch

Keep DiT shards resident across repeated invocations and reuse existing native once-per-run preprocessing where memory permits. Add windowed model startup, layer-level last-use eviction and cross-model lookahead. Track cache misses, bytes transferred and stalls. Stream recurrent layers only when needed or measurably beneficial. Avoid recurring whole-model copies, repeated hot-path weight conversions and unconditional per-step `empty_cache()`.

Deliverable: residency transitions stay within actual device/host budgets and preserve synchronous outputs. The benefit must be measured; native model management already performs some caching.

### 4. Add asynchronous handoff incrementally

Replace broad fences only where explicit events and ownership establish a dependency. Keep a synchronized fallback. Overlap host preparation/output copying first; enable weight prefetch only when there is safe headroom. Do not reuse the failed multi-request PP path unchanged or assume TE and DiT can coexist in VRAM.

Deliverable: delayed-copy/reuse simulations preserve synchronous results, with bounded live buffers and identical collective order.

### 5. Run a bounded physical correctness check, then missing integration measurements

After local fixes pass, perform a short physical dual-T4 correctness check of the integrated paths and transfer stress cases. Validate both TP ranks and Native outputs before accepting timings. Start with S8192 and a short sampler smoke; then run one representative five-second generation. A short sampler smoke is not a production-quality generation comparison.

Reuse all accepted historical component timings. Do not repeat the broad Native/TP/PP sweeps or optional stress cases. Keep prescribed TE S512/2048/4096 and DiT S8192/16384/32768; references remain inside packed DiT S. Missing full DiT 16k/32k cases follow only after capacity probes and an explicit remaining-time estimate.

Compare incremental configurations: synchronous TP integration → active-model cache → bounded async transfers. Add slab/audio options only after their respective qualification. Capture cold load versus warm execution, complete sampler latency, video/audio completion and mux time, GPU allocated/reserved/driver peaks, host peaks, transfer stalls, communication and per-stage profiler events. Use identical inputs, sampling work and timing boundaries; retain pre/post correctness and individual samples.

Estimate each job's validation/warmup/sample/profiler cost before starting it. Reserve budget for artifact collection, save progress immediately, stop on correctness failure and do not automatically restart or launch replacement instances. No integrated speedup or five-second quality claim is accepted from block extrapolations or local simulation.

## Initial production policy

- Single request: optimized TE TP2 + optimized DiT TP2 within validated scope; native conditioning and VAE semantics.
- Video slabs: explicit optional quality/performance mode after generated-output qualification.
- Audio channel parallel: enable only after retained numerical acceptance or an explicitly selected prospective numerical policy.
- Memory: one request-wide residency window, bounded host backing, DiT persistence across repeated invocations, and early next-model prefetch into storage freed after last use.
- Transfers: synchronized integration first, then event-owned asynchronous transfers proven locally and physically.

## Implementation progress (2026-10-09)

The request-wide fixed/adaptive residency core, explicit event-owned activation handoff, and Native TE/DiT entry-point adapters are implemented. Real 5060 Ti executions passed the prescribed trained block matrix and a composed 50-layer TE → 50-layer DiT → 36-layer public video decode on both logical ranks, with bit-exact results. Native text refinement is performed once per request; per-call input projection has a separate weight group.

See [implementation, fixes and executable evidence](../reports/RESIDENCY_IMPLEMENTATION.md). The full-entry test uses compact fixtures and a deterministic latent update, not the production sampler or vision tower. The remaining milestone is long-lived ComfyUI worker/sampler integration, followed by a bounded physical cross-device/NCCL correctness gate and fair integrated latency measurements. Slab/audio qualifications remain unresolved; Native VAEs remain the defaults.

The persistent host-cache extension has now passed a physical dual-T4 block/compact-video gate, including the VAE-to-next-TE CPU scheduling boundary. This is distinct from full sampler integration and from qualification of the separate cross-device activation handoff API.
