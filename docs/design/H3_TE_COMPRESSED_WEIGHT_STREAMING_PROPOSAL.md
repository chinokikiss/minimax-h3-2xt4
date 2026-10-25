# H3 TE Compressed Weight Residency and Streaming

**Engineering Proposal | Reviewed v1.1 | 21 October 2026 — not implemented**  
**Target:** MiniMax H3 Text Encoder (TE), INT8 rank-local weight loading, tensor parallelism (TP=2)  
**Target deployment:** Kaggle, 2 × NVIDIA T4; 30 GiB was the historical cgroup limit, not a guaranteed current allocation  
**First codec candidate:** ZSTD level 1, independent frames with at most 4 MiB raw output

**Review scope:** documentation only. No packing job, dependency installation, runtime change, GPU/Kaggle experiment or full H3 generation was performed. The original draft and review facts are preserved under `D:\attention_eval\te_compressed_proposal_review_20261021`.

## 1. Executive Summary

Historical H3 TE profiling identified repeated file-page refaults, background `kswapd` reclamation, producer waits in `folio_wait_bit_common`, and exposed weight-readiness waits. In one bounded full-request interval, the backing device supplied approximately 251 MB/s **across both ranks**. This is not a per-rank rate, a proven provider cap, or a measurement of the current FP16-fix runtime. Read-layout improvements may still help and must be separated from compression.

This proposal introduces an **opt-in, lossless compressed-weight path** that (1) stores precompressed, rank-local INT8 weights on disk, (2) leaves their compressed representation eligible for Linux page-cache residency, (3) decompresses independent chunks using a bounded CPU worker pool, and (4) overlaps read, decode, staging, and host-to-device (H2D) transfer with ongoing TE computation. It retains the existing rank ownership, quantization semantics, loader behavior, and pinned runtime dependency set.

**Recommendation:** keep **ZSTD-1 / 4 MiB** as the first lossless candidate, alongside an identically packed **RAW control**. First define a pre-fold artifact contract and bounded CPU-only replay. Then evaluate target read/decode behavior without changing the active loader. Only a successful frozen-stack qualification can justify an opt-in runtime adapter. Treat partial cache/refault relief as a hypothesis: the measured 11% size reduction does not make the entire TE block working set fit the historical cache budget. Do not develop a new codec or promise an end-to-end gain from RAM-only decode throughput.

### Review decisions

| Issue in v1.0 | Revision |
|---|---|
| Compression mixed with rank-local packing/coalescing | Compare the current loader, equivalent RAW packs, and ZSTD packs separately |
| “Final production tensor” could include FP16-fix folds | Pack `load_rank_block` base state **before** policy/folding; apply the existing preparation chain once |
| Compressed page cache assumed capable of retaining the full TE | Quantify the remaining capacity gap; investigate reuse distance and partial hits |
| 4 MiB tasks assumed to imply small streaming memory | Budget complete block destinations, frame buffers, contexts, pin rounding and cached backing |
| Shared pool described as if process topology were unknown | Two spawned rank processes are known; enforce a shared byte/worker budget with local decode contexts |
| New codec assumed compatible with frozen dependencies | ZSTD is absent from the production snapshot; optional backend qualification is an explicit gate |
| New files / inode eviction suggested as cold-cache controls | Require backing-device evidence; layered caches can remain warm |
| Full-generation timing included in near-term phases | Keep current work component-only; defer full H3 and production promotion |

## 2. Evidence and Limits

### 2.1 Prior TE bottleneck

The previous H3 investigation found:

- Host-memory limits of roughly 30 GiB and recurrent Safetensors file-page eviction.
- Approximately 24.810 GB of backing reads over a 98.983-second warm read interval, about 251 MB/s aggregate.
- Most observed page reclaim attributable to background `kswapd`, with worker time frequently blocked in the kernel rather than spent computing.
- TE block weight-staging delays that left the GPU idle on parts of the block 2–49 critical path.

These observations motivate a smaller **file working set**, not merely faster decompression. The [file-read causality report](../reports/TE_FILE_READ_CAUSALITY.md) supplies the workload, intervals, counters and limitations; [TE loading ablations](../reports/TE_CRITICAL_PATH.md) show that a coalesced-loading candidate did not meet its confirmation threshold. These are historical source/workload results, not current performance guarantees.

### 2.2 Local codec screening: measured, not a T4 result

The 20 October 2026 local benchmark prepared **108 final module tensors** using production `rank_local_int8.load_rank_block` for TE layers **0, 24, and 49**, across both TP ranks. The corpus was **1,463,872,512 bytes (1.363 GiB)** and predominantly INT8. Embeddings and the vision tower were not included; no compression ratio is established for the entire **27.14 GB checkpoint**.

Environment: AMD Ryzen 7 7800X3D, 8 exposed CPUs, 31.1 GiB RAM, frozen Python runtime. All tested codecs used independently decodable frames restricted to one tensor and one TP rank. Decode inputs were **already RAM-resident**; timings excluded disk reads, output tensor assembly, pinned-memory registration, H2D, and GPU compute.

Worker counts describe configured concurrency, not effective CPU utilization.
The local Windows CPU-time calibration failed; wall-clock scaling is retained,
but it cannot establish CPU utilization or predict Linux quota behavior.

| Scheme (4 MiB raw frames) | Compression ratio: raw/stored | Stored bytes saved | Decode throughput, 1 worker | Decode throughput, 6 workers |
|---|---:|---:|---:|---:|
| ZSTD-1 | 1.1240× | 11.03% | 1.057 GB/s | 5.648 GB/s |
| ZSTD-3 | 1.1240× | 11.03% | 1.107 GB/s | 5.400 GB/s |
| Direct FSE | 1.1317× | 11.64% | 0.884 GB/s | 5.072 GB/s |
| ZigZag + FSE | 1.1317× | 11.64% | 0.601 GB/s | 3.577 GB/s |
| Blosc2 BitShuffle + ZSTD-1 | 1.0072× | 0.71% | 1.799 GB/s | 9.497 GB/s |

The fast Blosc2 numbers must not be mistaken for useful decompression: its compression ratio was negligible, and the level-3 variant largely stored raw payloads. ZigZag was not useful ahead of FSE for these INT8 symbol frequencies. ZipNN 0.5.4 did not expose a functioning INT8 compression route in the examined implementation and was **not benchmarked**.

Changing ZSTD-1 chunk size from 4 MiB to 16 MiB improved local six-worker decode throughput by only about **3.6%**, with effectively unchanged compression. Four MiB offers more scheduling flexibility. Every tested configuration passed byte-exact round trips, including verification against production-loaded tensors.

### 2.3 Order-of-magnitude latency bound

For the measured 1.464 GB corpus, a **hypothetical** 250 MB/s cold disk reads approximately **5.855 s** uncompressed. At the observed ZSTD-1 ratio, it would read approximately **5.209 s** of compressed bytes, a gross I/O saving of **0.646 s**. Using the *local* six-worker decode time of approximately **0.259 s** gives **5.468 s** for a non-overlapped read-plus-decode estimate, or roughly **6.6% less than raw read time**. This is **not** a measured TE or T4 speedup, and excludes H2D, assembly, cache effects, and scheduling. Full overlap may reduce exposed decode cost; CPU contention or buffer assembly may increase it.

A warm-cache run can regress because raw cached weights require no entropy decoding. A non-linear benefit is possible if compression changes reuse-distance/cache-hit behavior; it must be measured rather than inferred from a smaller file.

### 2.4 Capacity and CPU bounds that change the decision

The current TP2 TE block-size accounting is 243,978,752 bytes per rank per layer, for 50 layers. Across both ranks this is approximately **24.398 GB / 22.722 GiB**, excluding vision and embeddings. Applying the three-layer ratio of 1.1240× would leave approximately **21.706 GB / 20.216 GiB** and save about **2.507 GiB**. This is an extrapolation, not a measured all-layer pack ratio; framing, alignment and RAW exceptions must be counted in actual artifacts.

The historical warm cgroup snapshot had 20.50 GiB anonymous memory and 8.98 GiB `file`, including 1.11 GiB shmem: approximately 7.87 GiB of regular-file cache for all files. A projected 20.216 GiB block pack would still not fit. Partial cache savings may help, but “compression makes TE resident” is not supported by that snapshot. The FP16-fix flow and Native-only audio have changed the live working set, so remeasure current pressure before predicting a threshold crossing.

For serial read+decode, ignoring other costs, improvement over raw reading requires:

```text
B_dec > B_io × r / (r - 1)
```

At 250 decimal MB/s and `r=1.124`, that is approximately **2.27 GB/s aggregate raw decode throughput**. The local single-worker 1.057 GB/s result does not meet this serial bound. Conversely, ideal overlap needs only enough decode supply to keep up with compressed reads (about 281 MB/s raw at this disk rate). These are different conditions; neither proves Linux layer readiness under contention. The 6.6% arithmetic example above is valid only for its stated local decode rate and omitted costs.

## 3. Objectives, Hypotheses, and Non-Goals

### Objectives

1. Reduce **physical file bytes read and file-page refaults** on repeated TE executions under RAM pressure.
2. Reduce **foreground block weight-ready latency**, TE GPU idle time, and ultimately user-visible end-to-end latency.
3. Keep decoding overhead off the GPU critical path through **bounded asynchronous prefetch and CPU decompression**.
4. Preserve bit-exact prepared tensors, TP=2 shard ownership, metadata validation, and established runtime compatibility.
5. Limit extra resident and pinned host memory; avoid converting file-cache pressure into anonymous-memory pressure.

### Testable hypotheses

- **H1 — Cold I/O:** Smaller physical files reduce bytes fetched from a slow disk, provided decompression is sufficiently overlapped.
- **H2 — Cache residency:** Compressed file pages remain resident more often under a fixed cgroup limit and reduce repeated file-page refaults.
- **H3 — Scheduling:** A shared, bounded CPU decode budget services both TP ranks without extending block readiness or starving loader/control threads.

None has yet been demonstrated on the 2×T4 host.

### Non-goals for this iteration

- No lossy INT8→INT7/INT6 requantization or change to numerical semantics.
- No replacement of TE attention, GEMM, quantization operators, or the current model.
- No modification or upgrade of ComfyUI or the frozen production environment. Compression is an optional proposed storage backend; **FP16-fix remains the pipeline default**, and Audio VAE remains Native single-GPU FP32.
- No custom entropy codec development or mandatory FSE integration.
- No claim about embeddings, vision tower, or whole-checkpoint compressibility based on the three-layer sample.

## 4. Proposed Architecture

```text
OFFLINE, ONE-TIME PACKING
Validated load_rank_block base state (pre-fold, no operator/session policy)
  -> exact rank-local tensors, metadata, ownership and complete hashes
  -> independent ZSTD-1 frames (<=4 MiB raw each)
  -> immutable rank-indexed compressed pack + manifest

RUNTIME, OPT-IN
Compressed pack files (Linux page cache)
  -> bounded asynchronous extent reads / cache hits
  -> compressed-chunk queue
  -> shared-budget CPU decode workers (one codec thread/job)
  -> complete contiguous CPU block/module, byte-exact with base loader
  -> existing policy/folding chain once, then bounded HostWeightCache
  -> existing pinned H2D stage and CUDA copy completion event
  -> resident GPU tensor / native TE execution
  -> release staging only after dependent H2D event completes
```

The compressed pack is a **derived, immutable TE-block artifact**. Original Safetensors weights remain authoritative. A packed **block hit** must not read the original block payload first. Small identity/header checks are allowed, and excluded embeddings/vision still use the original source path. Therefore original and packed pages can coexist; the proposal does not eliminate the original checkpoint working set altogether. Do not purge shared caches to manufacture a benefit.

Keep the integration boundary at `prepare_tp_block -> base loader -> existing configure/prepare_tp chain`. Replace only the base-loader choice. The first version preserves complete-module readiness and current GPU transfer behavior; it does not introduce region-wise kernels, a second GPU-residency owner or new numerical code. See [bounded stage-overlap design](ZERO_BUBBLE_STREAMING_ARCHITECTURE.md).

### 4.1 Offline packer and manifest

Build packs from the validated **`rank_local_int8.load_rank_block` output before `h3_tensor_flow.prepare_tp`**, not arbitrary checkpoint ranges or a policy-folded runtime module. This preserves existing TP row/column slices, packed components, replicated scales/norms and Native load-time casts.

Declare `representation=tp2_te_base_v1` and `fold_state=none`. On load, reconstruct the base module, restore its validated quantization metadata, and let the existing preparation chain apply operator policy and FP16-fix once. Never pack divisor-64 scales and fold them again on reload. If policy-prepared artifacts are explored later, give them a distinct schema and explicitly bypass duplicate preparation; do not infer fold state from tensor dtype.

Packing must retain:

- Rank ID, TE layer, canonical tensor/module name, TP row/column partition and replicated-tensor ownership.
- Exact dtype, shape, strides/contiguity assumptions, byte order, and original raw byte count.
- Quantization scales, conversion metadata, and validation rules as required by the existing native runtime.
- For every frame: tensor-local raw offset, raw length, compressed pack offset, stored length, codec ID/version, and integrity metadata.
- Manifest format version, source checkpoint content identity, shard-schema/preparation-code identity and **all tensor hashes**, including small scales and norms. A few representative hashes are insufficient for a byte-exact weight artifact.
- Codec implementation/backend, package and native-library versions, compression parameters, checksums, and explicit limits for frame input/output/window sizes.
- Separate artifact identity from session cache identity. The latter also includes effective operator/tensor-flow policy, gauges and request epoch. Relocating Windows artifacts to Linux needs an explicit verified path mapping; do not weaken the current reader's mutation checks.

The initial schema uses independent ZSTD frames without dictionary chaining.
Preparation keys must also identify the backend, artifact generation and
preparation schema. For controlled comparisons, use matched fresh decoded-cache
state; an existing RAW prepared-cache hit is not a measured ZSTD decode hit.
Any future cross-backend cache sharing must verify identical prepared state and
report such reuse separately. Apply folds only to privately owned modules;
never mutate a shared immutable base artifact or apply a new policy in place.

Each frame must be independently decodable, fully contained within one tensor and one rank; the final frame may be short. Store small or incompressible tensors as **RAW** where doing so beats compression after framing overhead. Pack rank-local data so the two ranks can request their own shards without repeatedly materializing the full checkpoint. Publish immutable generation-named pack files, then atomically publish a manifest/commit pointer referencing them. Two separate renames do not make a pack+manifest pair atomic. Flush required files before publication and keep old generations alive while readers own them. Reject incomplete, mixed-generation or mismatched artifacts; never silently substitute incompatible weights.

Verify every decoded tensor against the base loader during packing. Preserve exact JSON quantization metadata as well as tensor bytes; it is not all present in `state_dict`. Frames must cover each tensor exactly, without gaps/overlaps, and offsets must stay within pack/tensor bounds. Validate exact decoded lengths and frame completion, including checksum/truncation/trailing-data failures, before publishing a module. Record checksumming CPU cost.

Full source/pack hashing is a real read pass that can warm the filesystem. Charge it to artifact creation or startup validation and state that cache effect; do not perform a whole-checkpoint hash before each packed hit or label the immediately following trial cold.

**Deployment caveat:** Packing itself requires reading and writing a large checkpoint. Prefer distributing/reusing prebuilt compressed artifacts. If repacking per ephemeral Kaggle session is unavoidable, include its one-time cost and number of repeated requests in the ROI calculation.

Report pack plus manifest bytes, RAW exceptions, padding and duplicated
rank-owned metadata. Distinguish logical file size from allocated filesystem
blocks and observed backing read traffic; none is a substitute for the others.

### 4.2 Read and decode path

1. The TE layer scheduler requests the next weight-ready deadline by `(layer, rank, tensor)`.
2. The I/O stage coalesces physically adjacent compressed frames into reasonably large reads while preserving 4 MiB **decode** boundaries. A 16 MiB read extent is a candidate, not a fixed rule.
3. Independently decoded frames are placed at their **correct offsets** within a pre-reserved contiguous block destination. The initial correctness path may use library-allocated frame output plus a bounded copy; count that allocation and memory traffic.
4. Final CPU tensor views must match the native loader's output exactly; do not assume a decoded frame is automatically a valid contiguous tensor.
5. In the first version, publish a block only after all its tensors and metadata are validated. Reuse current pinning, H2D and module-ready events; GPU consumers must not see partially decoded parameters.
6. Host staging/pinned buffers and compressed input references are recycled only after no worker or DMA transfer can access them.

The current loader publishes a whole module; partial-region decode/upload is a separate future optimization with a new publication/ownership protocol. Do not assume existing `ResidencyWindow` can consume each 4 MiB frame independently.

An optional copy-reduction experiment can use `ZstdDecompressionReader.readinto()` with a writable destination. Loop over short reads and check frame completion; `readinto` does not establish zero-copy or zero scratch usage. The one-shot `decompress()` API returns a separately allocated `bytes` object. Bound frame content size and decoder window before allocating; do not assume `max_output_size` alone overrides an oversized embedded content size. In python-zstandard 0.25.0, `max_window_size` uses KiB. Pin the exact API/backend and test rejection behavior. [Decompression API](https://python-zstandard.readthedocs.io/en/0.25.0/decompressor.html).

Use one independent decompression context per simultaneously active decode operation. Reuse it only after that frame is complete. The library does not promise overlapping use of a shared context is safe. [Thread and object reuse](https://python-zstandard.readthedocs.io/en/0.25.0/api_usage.html#thread-and-object-reuse-safety).

### 4.3 TP2 coordination and deadlines

The current `PhysicalTPService` uses **two spawned rank processes**. A single host-wide worker/byte admission budget must govern both, while decoder contexts and tensor destinations stay local to their owning rank. Start with at most one decode operation per rank (two total); compare one total token and, only if useful, four total workers in isolated replay. Determine usable CPUs from quota, cpuset and affinity, not host CPU count. The historical target had a four-core quota; the local six-worker result is not its expected decode rate.

Use shared admission tokens/counters or a broker for control metadata. Do not pickle/multiprocess-queue decompressed weights between processes, and do not create another service. Foreground promotion, rank fairness and cancellation need explicit token ownership, including worker failure/shutdown. Reserve memory before dispatching a job; do not hold the broker/cache lock while waiting on a Future, decoder or CUDA completion.

Initial scheduling order:

1. Missing frames for the current foreground layer, balancing readiness across both ranks.
2. Frames needed for the immediately following layer.
3. Limited lookahead only when memory capacity and worker headroom permit.

The controller should prevent one rank from prefetching far ahead while the other rank blocks the same TE layer. Apply bounded queues, fairness/aging and epoch-based cancellation with backpressure in **bytes**, not just task counts. Leave a foreground preparation/pinning reserve for both ranks so speculative block destinations cannot consume all memory while foreground jobs await tokens. An already-running native decode may finish after cancellation; discard it only after its destination leases are safe.

Reuse the current cache/window deadline owner. `HostWeightCache.prefetch()` replaces its wanted set, and ordinary window lookahead calls it repeatedly. A new decompression scheduler must not overwrite it with a frame-only or successor-only plan. Frame jobs inherit their block's priority; no new CPU task may launch CUDA or insert collectives.

### 4.4 Memory residency and safety

Track the following separately:

- **File cache:** compressed pack pages accounted to the host cgroup.
- **Anonymous RAM:** decoded tensor staging and allocator overhead.
- **Pinned RAM:** H2D staging buffers, which are not freely reclaimable.
- **GPU VRAM:** current TE tensors and any competing DiT/VAE/idle-residency consumers.

Use byte caps for compressed extents, block destinations, temporary frame output, decoder contexts, pinned storage and lookahead. A 4 MiB frame is not a 4 MiB total working set: one TE block is approximately 244 MB per rank, and two foreground destinations already approach 488 MB before cache/pinning/context overhead.

The existing `tp_specs()` preparation allowance includes only a small additional margin (8 MiB beyond its current bound/adapter allowance). Do not reuse it unchanged for a parallel decoder. Define a conservative `prepare_peak_bytes` covering the module destination, all simultaneously live input extents, temporary output, context/window memory, assembly/folding transients and allocator overhead. Keep `ResidencyWindow`'s separately admitted pin-copy bounds, including tensor allocation rounding. Avoid charging the same storage twice, but do not treat a logical reservation release as physical RSS reclamation.

For the first adapter, **retain the current bounded pageable HostWeightCache policy**. It currently borrows a CPU module, creates a pinned copy and retains the cached module. Disabling that retention changes reuse behavior and requires its own ablation/ownership API. Direct decode into pinned memory or transferring the sole CPU owner is a later candidate, not an unaccounted shortcut.

| Storage | Safe release/reuse boundary |
|---|---|
| Compressed extent / decoder context | All decode users have completed or acknowledged cancellation |
| Temporary raw frame | Its bytes are copied/validated in the owned final destination |
| Cached CPU module | Cache borrow count is zero and the selected cache policy evicts it |
| Pinned H2D source | Its copy-completion event is complete |
| GPU weights | Existing last-consumer event/retirement rules allow reclamation |

Use the shared `HostBudget` for managed allocations **and** observed cgroup pressure for untracked/native/pinned-cache/file-cache costs. Read `memory.max/current/stat`, memory/IO pressure and effective CPU quota. `file` includes shmem; do not count it twice or equate all file bytes with reusable TE cache. Pressure metrics can lag; admission must reserve pending allocations before launch. No global `drop_caches`, `mlock` of the entire pack, or unmanaged RAM-resident compressed cache is part of this design.

## 5. Performance Model and Selection Policy

For one frame of raw size `S`, compression ratio `r` (raw/stored), disk bandwidth `B_io`, and effective decode throughput `B_dec`, a simple *non-overlapped* bound is:

\[
T_{serial} \approx \frac{S}{r B_{io}} + \frac{S}{B_{dec}} + T_{assembly} + T_{H2D} + T_{queue}.
\]

After pipeline warmup, idealized steady-state throughput is limited by the slowest stage, approximately:

\[
T_{steady} \gtrsim \max\!\left(\frac{S}{rB_{io}},\frac{S}{B_{dec}},\frac{S}{B_{H2D}}\right),
\]

plus startup/tail latency, stalls, contention, and any unoverlappable dependencies. CPU assembly/pinning/checksum work also consumes shared memory bandwidth and CPU capacity; add their resource demand when they share a bottleneck with decode. This is a throughput lower-bound model under independent-resource assumptions, **not a complete TE latency prediction**. A layer deadline is met only when both TP ranks have the required tensors ready. Do not give each rank the full measured aggregate disk rate.

**Initial codec policy:** ZSTD-1 for material compressible INT8 tensors; RAW for very small or incompressible tensors. Direct FSE is reserved for an ablation after actual disk and decode measurements; its additional approximately **0.60 percentage points** of saved disk bytes did not create a compelling local latency advantage. Do not select a codec solely from compression ratio or peak GB/s.

**Initial worker policy:** two total local decoder operations, constrained by the shared admission budget; compare 1/2/4 total operations in CPU replay before expansion. Ordinary ZSTD frame decompression has no automatic compressor-style `threads=N` setting that makes a single frame parallel; use independent frames/contexts, and disable any optional nested batch pool. The 281 MB/s supply comparison applies only to an overlapped steady-state model, not serial read+decode. Record CPU throttling and control-thread delay as well as frame throughput.

### 5.1 Frozen dependency and fallback contract

`zstandard` is **not** in `runtime/local_conda.json`. The local experiment uses `zstandard==0.25.0` from an isolated directory; that is not production qualification. Keep the frozen base environment untouched. An optional backend needs an explicit pinned dependency/wheel/native-library manifest, including backend and hashes, and separate Windows/Linux CPU validation. No automatic pip install, latest-version substitution or unqualified Linux ABI workaround is allowed. A missing compatible codec is a reported blocker for compressed loading, not permission to replace the runtime stack. No GPU allocation is needed for these setup checks.

Use separate concepts:

- **RAW frame** is a declared storage mode inside a validated pack, selected offline when compression is not worthwhile.
- **Source-loader fallback** is an explicit invocation-level policy for a missing/unavailable pack. Validate the same source/schema and log the reason, affected blocks, extra reads and backend selection. Do not silently mix original reads into a compressed benchmark.
- **Integrity/provenance failure** fails the selected artifact path. Do not expose partial modules, resume H2D with mixed generations or silently repair corruption by decoding only some tensors from another backend.

A workload-aware bypass must be selected at a request/component boundary with recorded inputs and hysteresis; do not oscillate per frame or infer physical cache residency from “warm request” alone.

## 6. Evaluation Plan — component-only under current constraints

### Phase A — Artifact correctness and cache feasibility (offline)

- Inventory all 50 layers/rank schemas from metadata first. Any complete targeted-block ratio claim requires packing **all 50 layers on both ranks**, with per-class strata and separate excluded-component accounting; another three-layer sample is not full coverage.
- Emit a manifest with per-layer and per-rank stored/raw bytes, expansion exceptions, and pack-generation cost.
- Verify every packed tensor's exact bytes/hash, shape, dtype, quantization metadata, rank ownership, and successful native load semantics.
- Validate the pre-fold representation, all tensor/metadata hashes, atomic publication and explicit source fallback; leave the production storage path unchanged. Exercise short reads, truncated/extra frames, malformed lengths, stale source identity, cancellation and exactly-once lease release.
- Compare current base modules with decoded base modules, then apply the same existing FP16-fix preparation once to independent copies and compare prepared state/output bytes. Preserve the optional legacy flow too; compression does not redefine FP32 ground truth.

**Exit gate:** All rank-local tensors are bit-exact with the production loader; no metadata or weight-ownership change; total compression savings are established for the actual targeted TE working set.

### Phase B — target-filesystem read-plus-decode replay (CPU-only first)

Build a CPU-only harness using the existing prepared artifacts, not a new TE/GPU run. Local runs can establish correctness and bounded-memory behavior, but cannot decide target SSD benefit. Use a CPU-only Kaggle instance when its storage path, CPU quota and cgroup conditions are representative; otherwise label the mismatch and leave the target claim blocked. Do not consume a T4 session to install/test environments.

Compare the current raw shard reader, an equivalent contiguous RAW pack, and ZSTD-1/4 MiB with the **same tensor order, read-coalescing policy and destination assembly**. A 16 MiB frame variant is secondary; FSE follows only if needed. Include actual I/O, decode, checksums, allocations and assembly in the timer. Use the two rank-process pattern and the same total worker/memory caps.

Measure frame and complete-block readiness distributions, logical reads, process `read_bytes`, backing-device counters, CPU quota/throttling, memory/IO pressure and bounded live buffers. Label three distinct states: application-cache warm, filesystem-cache warm and **backing-I/O observed**. They are not interchangeable.

Historical overlay/loop/device-mapper experiments showed that inode eviction could produce virtual reads with **zero backing-device reads**. Fresh/reflinked files, `DONTNEED`, a new process or an empty application cache do not prove a cold backend; artifact creation/checksumming can warm lower caches too. Do not sum counters from stacked devices or overlapping rank intervals. If the backing mapping/counters cannot be observed, report logical/virtual I/O only and leave physical-cold attribution unresolved. Use bounded relevant-file pressure/reuse traces, not global cache eviction or a stressor that changes the CPU quota.

**Exit gate:** Demonstrate that decode does not simply replace the disk stall with a larger CPU/assembly stall, and characterize the relevant cache conditions.

### Phase C — Loader pipeline and TP2 integration

Instrument every stage with monotonic timestamps and consistent layer/rank/frame identifiers:

`request -> read submitted -> read complete -> decode start -> decode complete -> destination ready -> H2D queued -> H2D complete -> GPU layer starts`.

Only after frozen Linux/T4 operator/ABI preflight has passed, consider a bounded component experiment on target hardware. Until then, CPU replay and local component correctness do not qualify T4 performance. Preserve foreground deadlines and existing pin/H2D ownership; profile representative TE blocks/current TP2 tensor shapes with compilation excluded. A complete TE forward, if later authorized, is a separate component scope; no full H3 generation is part of this proposal's current execution plan.

Collect block-ready exposed waits, useful GPU compute, H2D/event waits, rank imbalance, physical/logical read attribution where available, refault/reclaim/pressure deltas and peak managed/unmanaged memory. A zero caller `pgmajfault` count does not rule out file-read stalls. Reuse Nsight Systems for bounded CPU/CUDA correlation when available; report its absence instead of replacing profiler evidence with GPU-utilization averages.

**Exit gate:** Same-input component output/state bytes and cleanup pass; the measured current-shape critical path shows any loading/overlap gain and warm-path cost. End-to-end generation benefit remains unqualified.

### Phase D — Repeated-request and memory-pressure ablation

Use the same request/model parameters for all configurations and repeat at least enough times to report stable median and tail behavior. Compare:

1. **Current unmodified raw shard loader**: operational reference.
2. **Equivalent RAW pack**: isolates layout/coalescing and artifact-load effects.
3. **ZSTD pack, synchronous decode**: isolates compression/entropy-decode cost versus the same RAW pack.
4. **ZSTD pack, bounded asynchronous decode**: isolates overlap with unchanged retention.
5. **Separate cache/pin/scheduling changes**, each ablated against step 4: avoid crediting them to the codec.

Replay the single-user repeated-request **weight access sequence**, including TE startup/vision reads and the intervening DiT/VAE working-set footprint where captured. A synthetic anonymous allocator can sweep capacity, but cannot alone reproduce that file-cache access pattern or prove H3 behavior. Read the actual cgroup cap instead of assuming it is still 30 GiB.

Use counterbalanced orders, identical caps/worker counts, warmups and repeated samples with uncertainty. Repeat with idle residency disabled/enabled only as a separate factor. Do not simultaneously change cache retention, pin destinations, codec, lookahead and worker count. A warmed checksum pass belongs in the cache-state record.

## 7. Success Criteria and Decision Gates

**Mandatory safety criteria**

- Bit-exact tensor reconstruction and unchanged model outputs under identical deterministic conditions.
- No OOM, leaked pinned buffers, use-after-free on async H2D, incorrect TP rank ownership, or cross-request corruption.
- No new mandatory dependencies or unsolicited changes to ComfyUI/production versions. The optional codec is separately pinned and validated before enabling it; Linux wheel/ABI availability is an explicit gate.
- Original path remains selectable and is the default until performance is established.

**Proposed engineering targets (not claimed outcomes)**

- Measurably fewer storage bytes and file-cache refaults under memory-pressure conditions, with appropriately controlled comparisons.
- A provisional **5% median TE/component critical-path improvement** under a clearly defined representative I/O/pressure condition, beyond measurement variation. First require a win against the equivalent RAW pack, not just against scattered source reads. These are planning targets, not statistically established thresholds or a promise that 11% size savings yield 5% inference speed.
- A provisional **<=2% median warm-component regression** with enough samples to resolve a difference that small. If noise exceeds the threshold, the result is inconclusive. Report small-sample tails as such instead of treating a sparse p95 estimate as a reliable bound.
- No observed correctness, cancellation or memory-budget regression in the tested component cases. Generation p95, repeated full-request latency and user-visible benefit are **deferred** until a separately authorized full-pipeline evaluation; they are not claimed from replay.

If gains are below measurement noise, limited to unrealistic cold-cache setups, or offset by repacking overhead, keep the path experimental and stop further codec-level optimization. These thresholds are proposed decision rules and can be revised based on benchmark variance.

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Only ~11% compression on tested INT8 data | First measure full targeted TE corpus; allow RAW per tensor; avoid custom codec work without entropy evidence |
| Warm-cache decompression overhead | Preserve original loader and opt-in bypass; measure warm request sequences independently |
| Compressed + decoded copies exceed RAM budget | Admit the complete block destination, frames, contexts, cache copies and separately rounded pin copies; retain existing bounded cache policy in the first ablation |
| Two TP rank processes oversubscribe CPUs | Process-aware shared budget or explicit per-rank caps; observe effective CPU quota |
| H2D staging buffer freed too early | CUDA event-based lifetime tracking and explicit ownership |
| Compression introduces fragmentation or extra copies | Decode directly into final offsets where supported; time assembly separately |
| Page-cache savings do not prevent refaults | Verify cgroup file-cache occupancy and `workingset_refault_file` under controlled pressure |
| Per-session pack generation negates ROI | Prebuild immutable artifacts; include amortization over requests |
| Load-time semantics diverge from native loader | Pack validated pre-fold rank-local base tensors/metadata; verify all hashes and apply the current policy/folding chain exactly once |
| RAW-pack layout gain is misattributed to compression | Identical layout/read/assembly controls and counterbalanced RAW-pack vs ZSTD measurements |
| A small probe is mistaken for cold physical I/O | Observe backing-device traffic and layered-cache state or explicitly leave attribution unresolved |
| Optional codec cannot run on the frozen Linux stack | Report the package/wheel/ABI blocker; no dependency substitution or T4 setup job |

## 9. Suggested Deliverables

- `te_pack_manifest.json` schema specifying pre-fold base state, all tensor/metadata hashes, frame limits, source identity and committed immutable generations.
- Feature-flagged ZSTD-1 read/decode adapter wired at the **rank-local weight staging boundary**.
- Process-aware bounded task scheduler; per-rank priority, buffer pools, CUDA event ownership, and safe cancellation.
- CPU-only two-rank I/O replay with RAW-pack controls and explicit cache-state labels; bounded target GPU component harness only after frozen-stack qualification.
- Reproducible baseline/RAW-pack/ZSTD report with readiness distributions, backing-I/O attribution limits, full memory accounting and separate startup/packing costs. No E2E timing is implied by component results.
- A pass/fail decision against Section 7 before changing the default execution path.

## 10. Recommended Next Action

**First define Phase A's artifact/schema and build Phase B's bounded CPU replay.** The decisive comparison is current raw loading vs equivalent RAW pack vs ZSTD-1/4 MiB under the same ownership, memory caps and verified cache conditions. No test is launched by this document review.

Defer production integration and any T4 component job until the codec's optional frozen dependency contract and target Linux stack are qualified. Keep FP16-fix, Native-only audio and the original selectable storage path intact. The research question is whether **partial file-cache/refault relief plus overlapped decode** improves the measured critical path after accounting for layout, memory peaks and CPU competition. A full H3 speedup remains a later, separately authorized question.

---

### Source artifacts

- [Local codec study](../reports/TE_COMPRESSION_LOCAL_20261020.md) — methodology, sample strata, throughput and limitations. Raw `screen.json`, `corpus.json`, provenance and isolated codec dependencies are local-only under `D:\attention_eval\te_compression_20261020`.
- [TE file-read causality](../reports/TE_FILE_READ_CAUSALITY.md) — backing-device/folio/refault evidence and lower-cache confounders; historical source/workload.
- [TE critical path and read-layout ablations](../reports/TE_CRITICAL_PATH.md) — why RAW-layout controls and confirmation are necessary.
- [Bounded streaming architecture](ZERO_BUBBLE_STREAMING_ARCHITECTURE.md) — deadline ownership, cross-process memory accounting and lifecycle constraints.
- [Rank-local base loader](../../patches/rank_local_int8.py), [policy/fold boundary](../../patches/windowed_tp.py), [host cache](../../patches/host_weight_cache.py), [residency window](../../patches/weight_residency.py), [FP16-fix preparation](../../patches/h3_tensor_flow.py).
- [Frozen runtime snapshot](../../runtime/local_conda.json) and [benchmark-only codec pins](../../benchmarks/requirements-te-compression.txt); the latter do not qualify production imports.
