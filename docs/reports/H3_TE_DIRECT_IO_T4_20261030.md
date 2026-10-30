# TE page-cache bypass and hash profiling — 2026-10-30

**Kaggle accepted version 1; no T4 performance result yet.** The user requests one additional
GPU instance to test whether bypassing the page cache with `O_DIRECT` reduces
reclaim/refault activity and improves TE time. Hash verification is explicitly
included in the profile and remains enabled in both arms.

## Controlled workload

Use two physical Tesla T4s, the existing 50-layer TP2 TE, native vision frontend,
and the previous regular Turbo workload's prompt and two exact reference images.
Native REF2VA resizes the images using the **1216×704** canvas before tokenization.
The reference node is called with `vae=None`: only TE conditioning is produced.
There is **no DiT, VAE encoding/decoding, sampling, LoRA download or baking**.
The unused latent scaffold produced by the reference node is discarded.

The CPU-qualified and physically exercised runtime/core capsule is reused:
`21abe7dd91f284eed9626bb018c6abf66adf1c066b2f76662d72ce53d46d41a1`.
Kitchen stays **0.2.37**, TileLang **0.1.13**, frozen ComfyUI and the legacy
attention API stay unchanged, and Kaggle retains its native Torch family.
Actual versions are recorded. The effective runtime source is the qualified
capsule, not unrelated uncommitted local production edits.

Both arms retain the capsule's Kitchen causal TE attention, FP16 tensor flow,
column/row operators, shared host budget, CPU cache, GPU residency window,
lookahead and memory-pressure rules. Cross-model prefetch is disabled because
this experiment admits no DiT. An inert benchmark-only DiT reader satisfies the
runtime constructor without loading or verifying unrelated weights. Idle
residency is disabled in both arms so the starting boundary is explicit.

The TE bank contains **100 rank shards, 24,398,037,600 bytes**, plus the exact
**2,746,396,392-byte auxiliary checkpoint**. Download and immutable-source
validation happen once, outside TE timing. No dataset or notebook output is
mounted. Private image bytes remain in the private notebook and are excluded
from public artifacts; weights are read from immutable public HF revisions.

## Arms and timing boundaries

| Stage | Measurement |
|---|---|
| CPU adapter gate | Portable range/hash tests, then actual Linux direct-I/O flags, unaligned-range byte equality, SHA256 streaming and mincore residency |
| Raw I/O | Four real shards, two total readers, buffered/direct/direct/buffered order; separate read/copy throughput without hashing |
| Compilation warmup | One full buffered TE call, excluded from comparisons; persistent compilation caches reused by later fresh workers |
| Main order | Fresh worker pairs: **buffered, direct, direct, buffered** |
| First call in each pair | First verified TE pass; two unprofiled observations per mode, plus startup timing recorded separately |
| Second call | Repeated same-input TE with immutable verification proofs and ordinary CPU/GPU caches retained |
| Detailed traces | Second call of the first buffered and second direct pair; excluded from unprofiled repeat-call timing |

Each main pair contains two TE calls. This yields four unprofiled first-use
calls, two unprofiled repeat calls, two diagnostic profiler calls, and the one
excluded warmup. It is a bounded mechanistic experiment, not a large-sample
P95 benchmark. Fresh process startup and CUDA/module initialization are recorded
separately; residual first-call costs remain visible instead of being labeled
pure disk time.

Before each pair, clean weight files receive per-inode DONTNEED advice and their
residency is measured. There is **no global cache drop, sysctl or cgroup-limit
change**, and no artificial anonymous-memory pressure. DONTNEED does not prove
physically cold storage; remote and device caches remain uncontrolled. If the
buffered arm shows no meaningful reclaim/refault activity, the experiment must
say that page-cache thrashing was not demonstrated under this workload.

## Direct-I/O implementation

The benchmark intercepts only its TE safetensors files. It covers both
`CheckpointReader` payload reads and inference-time file SHA256 reads, including
the auxiliary checkpoint. JSON metadata and runtime library reads remain normal
filesystem I/O. This bypasses weight-data caching rather than disabling the
machine's entire page cache.

Safetensors headers and payload offsets need not satisfy direct-I/O alignment.
The adapter uses `statx(STATX_DIOALIGN)` where available, aligned anonymous
buffers and `preadv`, rounding ranges outward and copying only the requested
bytes into the original destination. Bounce buffers are bounded at 8 MiB per
open file; no full weight matrix or bank is staged by the adapter. Record
rounded syscall bytes separately from logical tensor bytes, including reread
boundary sectors. Kernel syscall byte counts are not physical disk traffic.

An unsupported direct read fails explicitly. The application does not reopen
the weights in buffered mode. Before model loading, a small same-filesystem
probe verifies descriptor flags, unaligned reads, EOF handling and SHA256;
mincore must show zero resident data pages before and after direct reads.
Representative real-shard ranges are checked again before the comparison.
The saved mount table and alignment results identify the actual scratch path.

Linux documents alignment restrictions and possible filesystem-specific fallback
behavior in [open(2)](https://man7.org/linux/man-pages/man2/open.2.html).
The descriptor and residency observations are therefore required alongside
byte equality; merely seeing `O_DIRECT` in source is insufficient. This is
ordinary CPU-mediated direct I/O, independent of the earlier missing GDS host
driver. It does not prove bypass of a storage server's cache.

## Hash verification and TE attribution

No TE verification is removed. Record these distinct costs:

1. Download-time file SHA256: identity validation before any TE trial, with
   per-file CPU wall work recorded separately from download elapsed time.
2. Inference-time file SHA256: auxiliary verification during startup and each
   first-used rank shard's file proof, including bytes, wall/thread CPU time,
   underlying read intervals and the current layer.
3. Tensor SHA256: native rank-state proof after loading, with bytes and
   wall/thread CPU time. This reads CPU tensor storage rather than the file.
4. Cached verification: repeat calls retain immutable-file signatures and proof
   caches. Actual hash call counts establish which proofs were reused.

The diagnostic records per-layer CPU preparation, requested tensor reads,
direct/buffered read calls, hash intervals, host consumer waits, GPU-event wait
and execution spans, maximum allocated/reserved VRAM and runtime cache statistics.
GPU-event spans can include enqueue gaps; they are not pure kernel durations.
Nested spans and cross-thread work totals must **not** be summed as an additive
TE critical-path breakdown. Use the interval timeline to inspect overlap.

Each worker first tests a minimal CUDA profiler capture/export. Detailed TE
CPU/CUDA traces are separate from timing samples; trace export occurs before
reply, then the parent checks actual kernel events after worker exit. Failed
capture remains an explicit profiling limitation. Exported traces and raw I/O
events are compressed after their measured stages.

At 0.2-second intervals, record cgroup memory/current/stat, file versus anonymous
memory, reclaim scans/steals, workingset refaults, memory/I/O PSI, CPU and I/O
counters. Global/device counters contain unrelated host activity and stacked
devices must not be added. One-second native GPU telemetry complements the
kernel timelines. The observer's own CPU sampling time is retained.

All conditioning outputs must be finite and byte-identical across both modes
and repeated calls. This guards the I/O adaptation, not perceptual video quality.

The test can distinguish several outcomes without assuming a win: bypass may
reduce reclaim but increase read traffic when verification and tensor loading
reread the same bytes; hashing CPU cost may dominate; or TE may be waiting on
preparation/H2D rather than the page cache. A speedup claim requires paired TE
wall improvement and corresponding profiler/counter evidence.

## Reproduction and submission

Benchmark sources are isolated in
`benchmarks/te_direct_io.py`, `benchmarks/run_te_direct_io.py`,
`benchmarks/test_te_direct_io.py`, and `benchmarks/build_te_direct_io_job.py`.
Five portable CPU tests passed locally, including exact checkpoint slices and
unchanged file/tensor hashes with injected syscalls. Local Windows tests do not
qualify Linux O_DIRECT support. The new Linux-specific gate runs before TE on
the requested job; the existing matching source/core CPU receipt is not
relabeled as qualification of the new benchmark code.

The builder binds supplemental source hashes, immutable publication, CPU
receipt, prompt/reference hashes and submitted notebook hash. It submits once
with an exclusive durable receipt. Setup is bounded to 600 seconds, the entire
TE experiment to 4,200 seconds and each fresh worker pair to 720 seconds;
the provider cap is 5,100 seconds. Those are upper bounds, not predicted runtime.
No status/list/live-log polling, duplicate submission or completion monitor is
configured. Final artifacts are collected after the user reports completion.

Kaggle accepted version 1 at
[H3 TE Direct IO T4 20261030](https://www.kaggle.com/code/pamcks/h3-te-direct-io-t4-20261030).
The submitted benchmark sources are bound to commit
`cae62cbcfcc645a7f61d14e5b19ea138fa14313a`. The private notebook SHA256 is
`453f73f79fb41ad9dcdcdf7dd321dc4667dec729f8400305a6b3942a0199946f`.
[Receipts and CPU checks](../data/h3_te_direct_io_20261030/) preserve the exact
source binding, metadata, reused core qualification and submission acknowledgement.
The full notebook is kept locally at
`D:\attention_eval\h3_te_direct_io_20261030\gpu_job\experiment.ipynb` and is
excluded from public Git because it contains the private reference image.
There has been no status/list/live-log query or final collection. Acceptance
confirms submission, not successful execution, direct-I/O support or a speedup.
