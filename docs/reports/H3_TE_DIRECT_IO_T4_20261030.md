# TE page-cache bypass and hash profiling — 2026-10-30

**Version 1 failed before model download or TE execution. The small Linux
O_DIRECT gate passed; TE performance remains unmeasured.** The user requests one additional
GPU instance to test whether bypassing the page cache with `O_DIRECT` reduces
reclaim/refault activity and improves TE time. Hash verification is explicitly
included in the profile and remains enabled in both arms.

## Version 1 failure, validated scope and correction

The setup worker completed successfully in **99.93 s**, retaining native
Torch **2.11.0+cu128**, CUDA **12.8**, Kitchen **0.2.37**, TileLang **0.1.13**
and the frozen legacy API. The experiment worker then exited after **5.08 s**:

```text
ImportError: cannot import name 'download_with_retry'
from 'benchmarks.run_h3_l1_turbo_batch'
```

The new harness mistakenly imported a helper present in the current local
checkout but absent from the exact frozen capsule. This is a benchmark
integration bug, not a CUDA, kernel, precision, O_DIRECT or model failure.
The state contains **zero TE cases, zero raw-I/O trials and zero model
payload downloads**. There is no evidence yet about TE speed, hash overhead
or page-cache thrashing.

Before the failure, five adapter tests passed on the Linux allocation. The
actual same-filesystem probe reported 512-byte memory/offset alignment through
statx, an O_DIRECT descriptor, byte-exact range/SHA256 reads and **0/1,042
resident data pages both before and after**. The host exposed two distinct
Tesla T4 devices, SM75. This establishes the small CPU-mediated direct-I/O
probe on `/tmp`'s overlay filesystem; no TE kernels were executed. It does not
establish direct-read throughput or model-loading behavior.

The fix puts the bounded transport retry helper inside the separately bound
benchmark adapter. No frozen production runner, runtime, attention operator or
dependency was updated. Retry remains limited to three transient transport/HTTP
attempts, with 5/15-second backoff; integrity and authorization failures are not
retried, and notices omit exception URLs.

Seven local CPU tests now pass. A stronger builder gate extracts the exact
hash-verified capsule into an isolated temporary source tree, overlays only the
new benchmark files, disables CUDA visibility, blocks network calls and
exercises the lazy asset-preparation imports through the auxiliary-assembly
boundary. The submitted v1 runner reproduces the same ImportError in that
rehearsal; the corrected runner passes, as do all seven tests. This is local
CPU import/adapter qualification, not new Linux or physical-T4 performance
acceptance.

All **18 final files** passed size/SHA256 validation. The three executed
supplemental modules were decoded from the returned supervisor launcher and
matched the submission's hashes; the setup receipt matches the qualified
capsule. Initial final-artifact collection hit TLS EOF, and the first retry
obtained one file before another TLS EOF. A second bounded transfer retry using
the existing certificate-verifying TLS 1.2 helper obtained all files. The
original errors, partial manifest and retry receipts are retained. Signed
artifact URLs remain private; public records contain only a sanitized transport
error summary. No status/list/live-log polling or GPU resubmission occurred.

Evidence:

- [Failed result](../data/h3_te_direct_io_20261030/final_v1/te_direct_io/result.json)
- [Exact failure](../data/h3_te_direct_io_20261030/final_v1/te_direct_io/exception.txt)
- [Source and probe validation](../data/h3_te_direct_io_20261030/final_v1_validation.json)
- [Corrected frozen-source rehearsal](../data/h3_te_direct_io_20261030/fixed_harness_cpu/rehearsal.json)
- [Submitted-v1 negative control](../data/h3_te_direct_io_20261030/v1_frozen_regression_cpu/rehearsal.log)

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
Final artifacts have now been collected and validated as described above.
Version 1 failed before the throughput/TE trials; it must not be reported as
a completed performance comparison.

### Corrected retry prepared, not submitted

A separate corrected notebook is ready at
`D:\attention_eval\h3_te_direct_io_20261030\retry2\gpu_job\experiment.ipynb`.
Its benchmark sources are bound to commit `5e7cebb199d78f3d468c53255bd9dc09d8532803` and notebook SHA256
`9648f491e69fca5e154f5403cd60b25db3d5b5a5315fba3d31785c678f8af1d5`. The builder's frozen-capsule rehearsal and seven CPU tests passed.
The workload, A/B order, TE verification, operators, core pins and time bounds
are unchanged. [Preparation receipt](../data/h3_te_direct_io_20261030/retry2/preparation.json)
records **prepared_not_submitted**. No second GPU job has been submitted;
the repository's one-job/no-automatic-resubmission rule remains in force.
