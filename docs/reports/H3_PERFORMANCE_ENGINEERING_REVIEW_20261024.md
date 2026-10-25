# H3 performance engineering review — 2026-10-24

Implementation follow-up: [remaining prioritized optimizations](H3_REVIEW_OPTIMIZATIONS_20261024.md).
It preserves findings 1–2 as requested, records active allocation/ownership
repairs and opt-in candidates, and separates local validation from pending T4
qualification. The review below remains the original pre-change evidence.

**Verdict: yes, several avoidable engineering costs remain. The most serious is duplicated, on-demand CPU LoRA baking in the TP2 bundle path. Scalar Sparse is also a poor performance choice for the measured 51,220-token workload. The normal Kitchen attention default is appropriate.**

The active pipeline is reviewed here from source delivery and reference encoding through TE, conditioning/refinement, denoising, weight preparation/residency, communication, Video/Audio decoding and export. This reviews the current working tree, including uncommitted changes; it is not restricted to the last commit. Historical PP/SP/slab/audio implementations are evidence and compatibility code, not presumed current execution routes.

No runtime, operator, model, dependency, original environment snapshot or historical experiment was changed. No GPU workload, model download, installation, external publication or Kaggle interaction was performed. The only new files are this review and its [source/evidence receipt](../data/h3_performance_engineering_review_20261024.json).

## Evidence boundaries

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| [Completed long E2E](H3_L1_LIFETIME_E2E_20261023.md) | Actual dual T4; four NFEs; 896×512/362 frames; pre-baked LoRA; Scalar override; media export succeeds | Current Kitchen-default E2E latency, online/bundle LoRA viability at that length, or generation quality |
| [Pre-baked Kitchen single call](DIT_SINGLE_CALL_PROFILE_20261023.md) | One complete 51,220-token/50-block NFE, about 161 s | Worker kernel attribution, warmed distribution or full sampler latency |
| [Bundle Scalar/Kitchen capture](BUNDLE_SCALAR_EXPLICIT_FLUSH_20261024.md) and its retained analysis JSONs | Both real T4 workers traced through all 50 blocks; actual CPU preparation and correlated GPU kernels | An uninstrumented steady-state benchmark or a matched pre-baked-versus-bundle speedup |
| Current source and tiny CPU fixtures | Mechanisms, ownership, dispatch, exact memory-size calculations and repeated work | New T4 timing or full-generation numerical acceptance |

The current T4 evidence uses native Torch 2.11.0+cu128, Kitchen 0.2.37 and TileLang 0.1.13. Older cu130/TileLang 0.1.15 measurements remain separate. Local checks used the prescribed conda Python, Torch 2.12.1+cu132, Kitchen 0.2.37 and TileLang 0.1.13. All **94** snapshot package versions match, and the frozen ComfyUI source check passes.

The latest raw bundle records correct one detail in the existing prose report: **rank 0 merged 216 matrices and rank 1 merged 200**, per backend. The difference includes rank-0 auxiliary/refiner work. Both prepared all 50 DiT blocks.

## Prioritized findings

### 1. [P1] Bundle offline LoRA duplicates full-matrix CPU work across TP ranks

Source: [BundleReader.load_rank](../../patches/tp2_bundle.py#L186), [_raw_tensor](../../patches/tp2_bundle.py#L137), [merged_weight](../../patches/offline_lora.py#L38), [bounded host preparation](../../patches/host_weight_cache.py#L335).

For each affected matrix, each rank reads/reconstructs the full Native matrix from both base shards, dequantizes it, applies the ordered FP32 LoRA update, requantizes the full width, then selects its own half. The other rank repeats that same full-width work. A cache miss repeats it again; the 3-GiB per-rank cache cannot hold all 50 DiT blocks. A per-reader lock and one preparation thread serialize each rank's producer.

This is a measured bottleneck. In the matched Kitchen bundle NFE, preparation medians were approximately **20.18/20.20 s per block**, while consumer-wait interval sums were **957.18/953.63 s**. The command span was about **1,000 s**. H2D activity was only about **0.9 s per GPU**. These overlapping quantities are not an additive latency decomposition. They do establish that faster H2D or another few layers of lookahead cannot repair this producer rate.

A tiny CPU reproduction loaded one fixture block for two ranks, twice. It recorded **20 full merges for five targets**, four merges per target, and zero persistent baked-checkpoint bytes. Full-width merge semantics are necessary under the current contract; doing the same merge separately for both owners is avoidable.

**Recommended change:** a byte-bounded layer/matrix preparation owner that computes the Native full-width result once and hands both owned shards to their consumers, with cancellation, identity checks and explicit shared-memory/IPC ownership. Measure CPU versus staged GPU preparation only within the existing memory envelope. Preserve full-width scales, adapter order, hashes and the no-complete-baked-checkpoint constraint. Do not substitute independent local-shard requantization. Multi-NFE reuse must be measured separately; this capture contains one NFE per backend.

### 2. [P1 when selected] Scalar Sparse spends more GPU time than Kitchen at the actual long shape

Source: [attention dispatch](../../patches/h3_operator_policy.py#L82); [physical capture](BUNDLE_SCALAR_EXPLICIT_FLUSH_20261024.md#physical-t4-results).

At 51,220 tokens, complete attention-scoped kernel sums were **113.00/118.11 s** for Scalar and **48.08/50.80 s** for Kitchen: approximately **2.35×/2.33×**. Most Scalar time is its execution kernel; routing contributes only about 0.6 s per GPU. The Scalar whole-call span was slightly shorter, 989 versus 1,000 s, because cold preparation dominates and varies. That difference does not establish an attention acceleration.

Selecting this approximation as a speed setting for this workload is unsupported. Keep Kitchen as the normal default, as the current source already does. Preserve Scalar as an explicit experimental or separately motivated choice. Its small-shape improvements over older Group32 implementations do not establish a win over Kitchen at this shape. No semantic-quality judgment is made from these timings or from attention-relative L2.

### 3. [P2] Column linears still pay for an expensive eager path and output assembly

Source: [scoped column adapter](../../patches/dit_column_linear.py#L12), [pinned dispatch guard](../../ComfyUI/comfy/quant_ops.py#L20), [backend diagnosis](DIT_MEMORY_51220_20261023.md#why-fc1-is-using-the-eager-path).

The frozen Native guard disables Kitchen's CUDA registry backend for CUDA builds below 13. The approved Kaggle cu128 environment therefore uses eager generic INT8 linears even though explicit CUDA primitives are available. Row linears already have a scoped accelerated route. DiT column linears preserve eager rotation/quantization, transpose/copy the weights for the integer GEMM, scale into a list of FP16 chunks, then concatenate them into another output.

The recent lifetime repair is valuable and must remain: it releases the INT32 accumulator before final concatenation and already enabled the long E2E. Remaining duplicated FP16 output storage is still **1.368 GiB** at FC1's 51,220×14,336 shape. Current correlated Kitchen-bundle traces also show about **44.96 s** of column-GEMM kernel activity on GPU0, versus about **5.98 s** for row GEMMs. These different workloads and kernel paths are not a controlled algorithm comparison; they justify focused column work, not a promised speedup.

**Recommended change:** first write the existing dequantization sequence into one allocated output, eliminating chunk-list concatenation and broadcast scale matrices. Preserve eager's exact order: row scale times weight scale, then accumulator times that product, then output rounding. Reusing the row epilogue without checking its different multiplication order would be incorrect. Qualify direct integer-GEMM dispatch separately. Retain the global guard and frozen packages; a scoped implementation needs its own exact checks and paired T4 timings. Use GEMM row chunking only if the remaining peak requires it, because it can reduce throughput.

### 4. [P2] Default online LoRA retains costly full QKV B outputs

Source: [NativeAlignedLoRA.add_to](../../patches/h3_lora.py#L70), [owned-row selection](../../patches/h3_lora.py#L21), [Selective B guard](../../patches/h3_lora.py#L218), [loader defaults](../../custom_nodes/minimax_h3_tp2/__init__.py#L56).

Selective B is restricted to the qualified rank-128 FC1 geometry. QKV adapters still calculate the full B output on each rank and discard the unowned packed rows. Under the current FP32 LoRA arithmetic, a full QKV delta at 51,220 tokens is **4.103 GiB per rank per adapter**, versus 2.052 GiB for owned FP32 rows. Input promotion, the base output, its FP32 accumulation copy, residuals and weights coexist at parts of this path. Stacked adapters execute sequentially, so their delta sizes should not simply be summed as simultaneous peaks.

This is confirmed duplicated compute/storage, with a substantial long-shape memory risk. **An online-LoRA OOM at that shape was not measured in this review.** The successful long E2E had no online adapters, and does not validate the default online route. Historical full-geometry evaluation has a valid numerical rationale: changing GEMM geometry can alter downstream INT8 decisions.

**Recommended change:** extend owned-B qualification to actual QKV adapter/shape families, preserving post-B scaling and adapter order. Treat tiling or selected-output GEMMs as numerical candidates, not universally byte-exact algebraic rewrites. Until qualified, document measured online request limits and keep explicit full-geometry fallback. Online bypass and offline weight merging remain different numerical models.

### 5. [P2] Video's fast path still constructs an unused trained parent CPU decoder

Source: [prepare-before-admission handoff](../../patches/comfy_sampler.py#L196), [staged component lease](../../patches/staged_vae.py#L149), [worker staging](../../patches/parallel_vae.py#L62), [weight-free planner](../../patches/video_decode_planner.py#L47).

The previous third **GPU** decoder and inherited 9-GiB DiT reserve were already removed from current code. Reporting those as unfixed defects would be wrong. However, the handoff still calls `prepare_decode()` before selecting remote execution. This reads/builds the complete trained parent CPU decoder, then both workers read/build their own decoders while the parent executes a weight-free planner.

The parent lease plus both simultaneous worker staging reservations total approximately **14.223 GiB** of the 16-GiB managed host budget, before retained caches and other leases. This is conservative admission accounting, not measured simultaneous RSS. It can deny the desired parallel path even when VRAM is adequate. The parent's decoder payload is about 2.282 GiB; that read/construction is unnecessary when remote decoding succeeds.

**Recommended change:** construct the public planner and compatibility metadata without loading parent learned weights; load the trained parent only after remote admission declines or when Native mode is explicitly selected. Preserve coordinated admission, fallback-before-collectives and exception cleanup. The revised 2-GiB policy and weight-free GPU planner have local trained/saved-latent checks, but physical T4 public-decode latency remains pending. The earlier **392.28-s** Native fallback is an opportunity measurement from the previous source, not evidence that current admission still fails.

The physical 261-token final-projection geometry mismatch also remains a qualification limit. Local exact results on SM120 do not override that T4 result. Any projection repair must retain the public Native tiling/chunk plan.

### 6. [P2] Ordinary checkpoint loading retains a syscall-heavy row-shard path

Source: [CheckpointReader.read_tensor](../../patches/windowed_tp.py#L42), [rank ownership plan](../../patches/rank_local_int8.py#L44), [optional contiguous loading](../../patches/rank_local_int8.py#L152).

The ordinary loader defaults to a monolithic safetensors file and no `rank_checkpoint_dir`. Column-sliced row-projection weights are read one source row at a time. For the two TE row matrices this means roughly **512,000 small range reads per rank across 50 blocks**; for DiT, roughly **537,600**. These are source-derived logical call counts for complete uncached traversals, not measured physical disk I/O or present-job latency. OS page cache does not remove the Python/syscall work.

Contiguous immutable base shards already exist as an optional representation. Prefer them for a measured deployment when their storage/delivery tradeoff is favorable, especially TE, which needs no LoRA merge. Keep full hashes/schema checks on first admission. Historical coalesced/parallel TE ablations failed their reverse-order promotion threshold; that is a reason to retain the baseline until a current-stack paired result passes, not a reason to promote more preparation threads unconditionally.

### 7. [P2] Bundle validation repeats whole-payload hashing after files are verified

Source: [bundle identity checks](../../patches/tp2_bundle.py#L97), [per-preparation state hashing](../../patches/tp2_bundle.py#L194), [ordinary offline once-per-signature validation](../../patches/rank_local_int8.py#L167).

`_shard()` remembers verified file/manifest signatures, but `load_rank()` still hashes every loaded tensor on every preparation. `check()` also stats the entire growing verified-file set during repeated matrix reads. The CPU fixture confirmed one complete state-hash pass for each of four preparations, including repeated unchanged blocks. This is avoidable warm CPU work; its exact contribution to the approximately 20-s real preparation is not isolated.

**Recommended change:** retain full file/schema/state proof on first load and after identity changes, and reuse that proof under unchanged signatures, following the existing ordinary offline-reader contract. Keep integrity work in reported cold setup latency. Do not solve this by removing validation entirely.

### 8. [P3] Worker result handling copies rank-1 tensors that are discarded

Source: [packet copies](../../patches/comfy_tp_runtime.py#L101), [worker reply](../../patches/comfy_tp_runtime.py#L638), [parent CPU request staging](../../patches/comfy_tp_runtime.py#L793).

Both workers call `packet(result, 'cpu')` before rank 1's non-diagnostic result is replaced by `None`. `packet()` always requests a copy, even on the same device. TE and DiT therefore perform unused rank-1 result transfers; normal requests also bounce through parent CPU transport. Video rank 1 already returns `None`, so this finding does not imply duplicate full-video pixel transfers.

Discard unneeded rank-1 output before host conversion, retaining its GPU-completion acknowledgement and error/cancellation boundary. Consider explicitly owned reusable staging after measurement. Whole-session asynchronous transport or CUDA IPC is a larger change and has no demonstrated priority over the preparation bottleneck. No request-level saving is claimed from these copies.

## Further opportunities and choices that are justified

TE Q/K/V separately rotate and quantize the same normalized input; gate/up repeat preparation of their common input as well ([source](../../patches/tp_qwen3vl.py#L178)). Both ranks also execute the full Native vision frontend. Sharing identical prepared operands, or computing vision/embedding preparation once and broadcasting the result, can remove redundant work. Preserve original quantization/GEMM rounding boundaries and Native image/DeepStack ordering. These are unmeasured opportunities; historical TE readiness waits dominate short useful GPU compute, so they follow loading work in priority.

Attention O's FP32 row quantizer also rotates a full-width padded tensor whose unowned half is zero ([source](../../patches/tilelang_row_quant.py#L111)). That is an expensive geometry-preservation choice, and the current fused FC2 improvement does not remove it. A group-local or fused O preparation candidate could reduce temporary storage and work. It needs a separate numerical/latency comparison: preserving ConvRot's real-arithmetic formula alone does not preserve Native FP32 GEMM rounding.

The following choices are reasonable under the present contract:

- **TP2 for batch-one transformer latency:** replacing it wholesale with layer PP would introduce serial execution between stages. Historical SP wins from different implementations do not qualify a replacement for current Native semantics.
- **Frozen source/core pins and the legacy attention API:** stability and reproducibility are valid requirements. A new Torch, TileLang or ComfyUI is not a permissible shortcut around a scoped performance issue.
- **Immutable source delivery and hashed direct pip:** the current bootstrap reuses exact matching packages, installs only approved differences into an overlay, and protects the native Torch/NumPy/CUDA family. CPU preflight precedes GPU work. Source/model delivery, installation, baking and JIT are cold-start costs that must remain visible and separate from warmed inference; removing their accounting would not optimize the pipeline.
- **FP16 residual storage with FP32 stability islands:** prior controlled overflow failures justify the islands. Fuse their storage traffic while preserving their arithmetic if measurements support it; simply deleting promotions is not a safe optimization. The latest worker probes do not reproduce the older half-second-per-Norm CPU anomaly.
- **Shared global row scales:** a MAX collective preserves the existing quantization operator. Independent per-rank activation scales would change it.
- **Scaled FP16 SUM and fused-stored preparation:** these are user-approved approximate performance choices with current-stack component evidence. Stored versus recompute is a legitimate memory/latency tradeoff; recompute is not universally faster. INT32 remains available. Full-trajectory quality is still pending.
- **Bounded CPU/pinned/GPU caches and last-use events:** these prevent real memory/lifetime failures. Increasing lookahead cannot make a slow serial producer meet a sustained faster consumer rate. A shape-aware temporary envelope and coordinated cache placement are more useful than blindly maximizing residency.
- **Single-GPU Native FP32 Audio:** only **3.33 s**, about **0.22%** of the recorded 1,515.38-s foreground request. Audio parallelism is low priority and would add complexity and numerical obligations.
- **Native public sampling and Video planning:** preserve reference/layout construction, temporal overlap, tiling, normalization, trimming and export. The faster incorrect independent-slab results do not justify bypassing them.
- **Automatic NCCL protocol with four channels and P2P disabled in the qualified topology:** historical paired evidence supports this narrow deployment. It does not prove the best setting for every future message size/topology, but forcing Simple or treating NCCL wait time as pure PCIe bandwidth is unsupported.

## Deployment and validation gaps

The successful large-request runners and ordinary loader have different resource policies:

| Setting | Ordinary runtime/loader | Retained large-request runner |
|---|---|---|
| DiT reserve | 1 GiB | 9 GiB |
| Reuse-aware lifecycle | Off unless enabled with idle residency by the UI | On |
| Cgroup-aware host admission | Off | On |
| Command timeout | 300 s | 600 s E2E; 1,500 s bundle profile |
| LoRA execution | Online by default | Pre-baked E2E; separate CPU-bundle profile |

Sources: [RuntimeConfig](../../patches/comfy_tp_runtime.py#L37), [loader wiring](../../custom_nodes/minimax_h3_tp2/__init__.py#L77), [E2E configuration](../../benchmarks/run_h3_l1_current_e2e.py#L85), [profile configuration](../../benchmarks/profile_dit_single.py#L26).

Consequently, current whole-pipeline performance and large-shape admission for the ordinary UI defaults are **not established** by those jobs. In particular, the roughly 1,000-s cold bundle NFE exceeds the ordinary 300-s command deadline on that measured environment. Lifecycle scheduling should be configurable independently of queue-idle speculation. Use a qualified workload profile and shape-dependent admission; do not advertise a single default as covering every measured path.

There is no matched, warmed full-pipeline comparison covering the current Kitchen default, online LoRA, CPU bundle mode and revised Video admission. Kernel sums, rank waits, NVML utilization and command spans are different measurements. They must not be added together or used to promise a total speedup. The 161-s pre-baked call and 1,000-s bundle call are different jobs with different loading/instrumentation; their ratio is not a controlled performance claim.

## Checks performed and disposition

**106 selected CPU tests ran: 105 passed, one errored.** The first suite ran 78 tests, with 77 passes; the second ran 28 and passed all. CUDA was hidden for both. Coverage includes frozen source/setup/direct-pip contracts, tiny offline baking and bundle merges, tensor flow, fused dispatch, LoRA partial reduction, lazy reference loading, Video ownership/admission, Native-only Audio, residency and lifecycle behavior.

The error is [the diagnostic snapshot fixture](../../benchmarks/test_pipeline_review.py#L58): its `SimpleNamespace` lacks `readers`, which [rank_snapshot now uses](../../patches/runtime_diagnostics.py#L24). Real `RankRuntime` initializes `readers`. This is a stale regression fixture, not evidence of an inference failure; the suite must nevertheless be reported as failing. It was left unchanged during this review. No tests were skipped to manufacture a pass.

The additional small CPU call-counter reproduction confirms duplicated full merges and repeated state hashes. The receipt records those counts, 30 selected source hashes, the unchanged 94-package inventory, exact memory calculations, retained evidence hashes and per-rank preparation counts. No new performance measurement is claimed.

Recommended work order: **deduplicate bounded bundle LoRA preparation; keep Kitchen for the measured long workload; qualify the column epilogue and online QKV allocation reductions; remove parent learned-weight loading from successful remote Video decode; then address TE loading/shared preparation, repeated validation and unused transfers.** Keep precision, source and dependency changes out of those experiments, and require complete-API latency plus memory and numerical checks before promotion.
