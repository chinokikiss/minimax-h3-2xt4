# Rank-local loading and separate preparation/residency

P0/P1/P2 implementation and local qualification. This report does not treat
5060 Ti measurements as dual-T4 performance evidence. The previous physical
baseline remains in `PIPELINE_RESIDENCY_AUDIT.md`; these changes need a new
physical comparison before any end-to-end latency claim.

## P0 — direct rank-local INT8 loading

`patches/rank_local_int8.py` constructs TP module shells on the meta device,
then reads owned checkpoint ranges directly into CPU tensor storage. It never
constructs a full Native block. Packed DiT QKV takes half of each Q/K/V
component; packed FC1 takes half of each gate/up component. Input-sharded
projections read owned columns, replicate their output scales, and preserve
256-channel ConvRot boundaries. Norms and AdaLN are replicated with the same
FP16/FP32 casts as Native. Unsupported shapes, additional tensor keys, and
quantization metadata fail explicitly. Attention, RoPE, quantization, AdaLN
execution and TP collectives are unchanged.

The optional offline converter compares every exported tensor, its shape and
dtype to a genuinely Native-loaded shard before saving. Per-block manifests
record source identity, configuration, quantization metadata and tensor
checksums. First load verifies checksums; subsequent loads retain verification
only while artifact and manifest file signatures remain unchanged. Verification
cost is part of cold/startup timing. RuntimeConfig.rank_checkpoint_dir selects
offline loading; the default uses direct runtime ranges. Current artifacts bind
to the exact source path/size/mtime and require conversion/binding on the target
host. Checkpoint files are not committed to the repository.

Commands (with COMFY_ROOT set to this repository's pinned `ComfyUI` checkout,
commit `d49e888586dd8ae012c0667b33466b815fee07f7`):

```powershell
python benchmarks/qualify_rank_local_loader.py --convert-all
```

Conversion writes bounded individual blocks to `D:/minimaxH3/rank_local_tp2`.
It costs additional disk space and work once; this cost must be reported
separately from subsequent online requests. Runtime column reads have about
10,000 small range operations per block. Logical bytes mean requested payload,
not actual disk reads, filesystem readahead, or PCIe traffic.

## P1 — CPU preparation independent of GPU residency

The pageable CPU queue starts with a 16-group horizon and orders tasks by
predicted next-use deadlines. Foreground demand takes priority at module
boundaries. Host preparation/copy time and compute observations adapt this
horizon independently. Cache capacity and the process-shared HostBudget bound
admission; expired entries are evicted before upcoming entries.

Pinned staging remains bounded, separately charged, and owned until its copy
completion event. The short H2D horizon starts at two groups and adapts using
copy/compute time only. CPU preparation does not allocate CUDA tensors. This
is a bounded pool of in-flight staging modules; it does not add an allocator
that reuses arbitrary tensor-shaped pin buffers.

TE releases each layer after its final use. DiT retains eligible weights across
actual denoising evaluations, subject to byte/activation-headroom guards and
next-use eviction. The JIT horizon bounds speculative transfers; retained DiT
weights are a separate residency decision. Known TE completion can queue DiT
CPU block preparation before sampler startup. No cross-model speculative GPU
allocation is introduced.

Background preparation leaves an additional 2 GiB of available RAM beyond its
own construction reservation and the shared budget's RAM reserve. Foreground
pinning and model startup can reclaim speculative cache entries. This margin
is separate from the managed byte cap.

Deadlines use observed compute time and planned layer order; unknown next NFE
or request arrival times are not predicted. This policy does not yet optimize
against measured H2D/NCCL contention.

## P2 — TE startup during video decode

DecodeHandoff sends decode_start before Native VAE decoding. Ranks prepare the
model-invariant TE startup tensors on CPU in a separate bounded cache under
the same HostBudget. Vision and embedding/metadata groups are independently
admitted, with vision first. Native module construction waits for actual TE
demand. Transformer-tail entries are relinquished to
give this working set headroom. No prompt, reference, or conditioning result is
guessed. Later decode notifications do not restart the same work.

The next TE call consumes those raw groups and transfers reservation ownership,
without rereading the checkpoint. Native embedding construction receives its
own transient reservation. It releases CPU embedding ownership
after moving the frontend and retains only the canonical vision backing until
TE teardown. Sampling cancels unused startup speculation; changing checkpoints
replaces/closes the service and its caches. Admission may defer preparation
when the shared budget or available RAM guard cannot safely accommodate it.

This targets next-request readiness and throughput. It does not shorten the
current video decode. Actual contention with decode and benefit under zero-gap
requests need physical measurement. Saved-Native correctness validation between
local requests gives speculation extra time and must not be called a zero-gap
throughput result.

## Qualification status

All 200 trained loader cases passed Native tensor byte/shape/dtype equality,
including both runtime and offline paths. Conversion verified every offline
block before writing it.

Local CPU medians (50 layers × both ranks, four CPU threads, uncontrolled OS
page cache and fixed execution order):

| Loader | TE ms/block | DiT ms/block |
|---|---:|---:|
| Legacy full Native block then shard | 172.55 | 168.36 |
| Direct runtime shard ranges | 84.03 | 77.14 |
| Offline first load including verification | 190.54 | 159.41 |
| Offline subsequent verified load | 55.55 | 45.84 |

Requested payload per block/rank falls from 465.29 to 232.68 MiB for TE,
and from 369.41 to 185.57 MiB for DiT. Offline DiT stores the already-cast
FP32 AdaLN/norm values and reads 187.25 MiB. Median sampled RSS increases were
about 639→233 MiB for TE and 548→187 MiB for DiT. RSS deltas are affected by
allocator history; these are local diagnostics, not isolated-process peaks.

The table's direct reads immediately followed Legacy and benefited from its
page-cache warming. A later verification without that ordering observed roughly
400–600 ms direct reads versus 50–60 ms warm offline reads. Neither run controls
storage cache state. The small-range runtime reader therefore reduces logical
bytes and construction memory, but does not establish a general CPU latency
improvement. Offline contiguous storage is the stronger performance candidate;
conversion, storage cost (41.01 GiB) and first-load verification must be paid.

Final local qualification passed after repairing three failures: non-finite
telemetry deadlines now serialize as null; speculative CPU work leaves RAM
headroom and prepares TE startup in smaller groups; offline checksums use the
tensor's existing byte storage instead of an unbudgeted large byte copy.

| Final local check | Result | Evidence |
|---|---|---|
| All 200 trained TE/DiT rank blocks, direct and offline | PASS, identical Native tensor bytes/dtypes/shapes | `semantic_results/rank_local_loader_final.json` |
| Direct synchronous, full 50 layers, cold/warm | PASS, all five outputs bit-exact, no OOM | `semantic_results/loading_pipeline_direct/sync_result.json` |
| Direct adaptive, full 50 layers, cold/warm | PASS, all five outputs bit-exact, no OOM | `semantic_results/loading_pipeline_direct/adaptive_result.json` |
| Offline adaptive, full 50 layers, cold/warm | PASS, all five outputs bit-exact, no OOM | `semantic_results/loading_pipeline_offline/adaptive_result.json` |
| Repeated sampler evaluations, CFG, masks, image/video references and cancellation recovery | PASS, nine bit-exact cases, two trained layers | `semantic_results/loading_sampler_2/result.json` |
| Fresh streamed versus placed Native, four evaluations | PASS, all five outputs bit-exact, two trained layers | `semantic_results/loading_native_placement/result.json` |
| Physical runner ablations, four evaluations, cold/warm | PASS locally for all three policies, two trained layers | `semantic_results/loading_wrapper/` |

The 200-case refresh reuses the original genuinely Native-validated conversion
hashes and records its parent evidence SHA; it does not claim another Native
execution. Every result above binds the executed sources. All pipeline services
returned the shared reservation count to zero on close.

Full local requests used 512×512, 124 frames and one denoising evaluation.
Direct adaptive peaked at 22.37 GiB process-tree RSS with 1.26 GiB minimum
available system RAM; offline adaptive peaked at 21.83 GiB with 2.40 GiB
available. These are measured local headroom values, not safety guarantees for
other shapes or a different host. No pinned-allocator statistic is used as a
physical RAM measurement.

Both ranks completed TE startup preparation inside the first video decode and
reported two startup cache hits on the warm request. For direct rank 0,
preparation ran from monotonic 3874.234 to 3876.453 seconds. The decode handoff
events in the same artifact delimit the actual overlap. This proves CPU work
was initiated in that stage; physical decode slowdown and zero-gap throughput
remain unmeasured.

## Physical comparison and recommendation

Kaggle version 1 failed during the 600-second offline conversion limit after
all TE layers completed. No GPU benchmark ran. Version 2 removed this blocking
conversion; the user stopped it during checkpoint download. These sessions
provide no performance results.

The independent [Direct Loading Audit](https://www.kaggle.com/code/chinokiki9/minimax-h3-direct-loading-audit)
is stopped (`CANCEL_ACKNOWLEDGED`); it has not been relaunched.
The original [Rank-local Loading Audit](https://www.kaggle.com/code/chinokiki9/minimax-h3-rank-local-loading-audit)
is repurposed as a CPU-only offline publisher, using the user-specified
`HF_token` Kaggle Secret in revisions 3–5. Revision 6 uses the scoped upload
transport described below. HF publication is complete; new physical benchmark
results remain pending.

The bounded direct Kaggle job uses a fresh physical Native reference, then direct
synchronous and direct adaptive execution on the same 2×T4 instance. Each TP
strategy uses identical inputs, four evaluations and cold/warm
requests; profiling is a separate evaluation outside request timings. The
synchronous baseline disables CPU lookahead across models and requests as well
as asynchronous H2D. Both adaptive strategies enable the same scheduling, so
their difference measures checkpoint layout/loading rather than a policy change.

The CPU job reads each owned shard directly and checks it against frozen Native
tensor hashes from the complete local 200-case proof. Full source-file SHA256
checks tie those hashes to the target checkpoints. It explicitly records
`native_execution_this_conversion=false`; it avoids repeatedly constructing
full Native blocks just to reproduce already-verified tensor hashes. Saved
offline files are reloaded and checked before publication. The Secret-based
publisher supports groups of ten and fixed-revision resume. The successful
revision-6 transport uploads each verified object from Kaggle, then commits
all pointers, manifests and the complete bundle marker together. The immutable
HF commit is saved in `publication.json`. Neither path embeds the HF token in
source or exported logs.

`benchmarks/offline_checkpoint_store.py --reuse-revision <HF commit SHA>` downloads
a completed bundle, validates its source identities, file checksums, configuration
and Native tensor hashes, then rebinds the manifests to the current machine.
The production loader keeps its strict path/mtime check. Four local relocation
cases (TE/DiT × both ranks) passed Native tensor verification and rejected a
corrupted Native proof: `semantic_results/offline_portability.json`.

HF authentication is checked before model download/conversion. CPU revisions
3–5 failed when Kaggle's Secret service returned HTTP 400 for `HF_token`, before
HF was contacted; no conversion, repository creation or upload was performed.
Kaggle's client wraps HTTP errors as `ConnectionError`, so the diagnostic checks
the underlying HTTP status and avoids retrying configuration/authorization
errors. The user was asked to check the Secret's attachment to that notebook.
No token value is logged. This is independent of the direct dual-T4 benchmark.

On October 10, the user requested that weight uploads run on Kaggle. The local
uploader was stopped and its nine Xet cache/staging files (35,418,876 bytes)
and task upload log were removed. Downloaded original models and validated
offline checkpoints remain intact.

CPU publisher revision 6 completed on Kaggle. It avoids
the Secret-service failure using HF's LFS multipart protocol: the coordinator
authorizes the exact SHA256/size of each of the 200 existing Native-validated
weight artifacts, then gives the private Kaggle notebook temporary, object-scoped
upload capabilities (10-hour validity). The broad HF token is absent from the
notebook, transport plan and exported results. Transport files are ignored by
Git and must not be published publicly before the capabilities expire.

Kaggle regenerates each shard using the qualified direct loader, verifies its
Native tensor hashes, reloads it with the strict offline loader, and checks the
serialized file SHA256/size before uploading directly to HF storage. A local
reserialization check passed exact file-byte equality for DiT rank 0 block 0.
This is a transport check, not new physical GPU correctness/performance evidence.
Receipts and manifests are exported, while weight bytes remain under `/tmp`.
All 200 upload receipts and Native manifests matched the plan. The coordinator
confirmed HF object availability, committed small JSON evidence and LFS pointers,
and verified every final weight file's SHA256/size at immutable commit
`d3e6a4b131056d2c626a01acf6ba48123a885783` in private repo
`cnmds/minimax-h3-tp2-int8-561a6211`. No weight payload was uploaded from the
user's machine after the switch to Kaggle.
`benchmarks/finish_kaggle_publication.py` downloaded the 203 JSON records
(2,938,193 bytes), then exited after a connection timeout during finalization.
The metadata-only finalizer was retried with bounded HTTP timeouts and passed;
conversion and weight uploads were not repeated. Both local processes have
exited, and their temporary HF credential environment was cleared. Final proof:
`semantic_results/kaggle_hf_publication_v6/publication.json`; successful log:
`semantic_results/kaggle_hf_finalize_v6.log`. The LFS pointer serialization path
was checked to contain no weight content.

The CPU job took 828.083 seconds for setup and 2,443.866 seconds for conversion
plus uploads, totaling 3,271.953 seconds (54.53 minutes). The 200 artifacts total
41.009 GiB. Per-object upload intervals sum to 1,088.093 seconds, with serial
conversion/validation and transfer; this is not a GPU latency measurement.
Native tensor equality and safetensors file-byte equality passed for every
published shard. Target-GPU pipeline numerical correctness and latency still
require the fresh physical Native gate before accepting any speedup.

The first revision-6 submission exceeded Kaggle's 1 MB source limit and was
rejected before execution. Removing irrelevant local profiler/result histories
from the CPU archive reduced source to 771,850 bytes; the next submission was
accepted. No GPU benchmark was retried.

Checkpoints stay under `/tmp`, while small evidence is exported. The direct job
checks disk headroom, checkpoint identities and two physical T4 devices, stops
on the first benchmark failure, and has no automatic benchmark retry. Its setup
limit is 20 minutes and comparison limit is 50 minutes. The independent CPU
conversion/upload limit is two hours; these are limits, not estimated durations.
One cold/warm pair per strategy can diagnose
bottlenecks but cannot establish statistically reliable speedups.

P0 is locally PASS for semantics and payload reduction; latency is PARTIAL until
physical results. P1 is locally PASS for independent queues, bounded admission
and lifecycle checks; deadline quality and transfer contention remain PARTIAL.
P2 is locally PASS for CPU preparation during decode and warm cache reuse;
its physical throughput benefit remains PARTIAL. Prefer offline shards if the
physical latency benefit covers their one-time conversion/storage cost. Keep
direct ranges as the storage-saving fallback. Do not increase GPU lookahead
until profiler evidence shows exposed copies can be hidden without displacing
compute buffers or retained DiT weights.
