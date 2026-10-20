# TE checkpoint I/O causal audit

Status: source audit and 22 large-working-set physical controls complete.
All 2,008 trained block loads preserve Native tensor hashes, with zero OOM
events. No fixed storage-bandwidth cap or inference speedup is established.

The new controls show poor full-set source residency even with only ~1.2 GiB
anonymous memory, while a small immediate reread is 100% resident. Aggregate
file-cache headroom therefore does not guarantee useful full-set inode residency
in this layered filesystem. Pread has the best whole-test timings, but actual
read unions improve only 3.41% initially and 0.84% in reverse order. Mmap loses.
See `TE_IO_ABLATION_RESULTS.md` for numbers, drift and attribution limits.

## Established physical evidence

`TE_CAUSAL_PROFILE.md` and `TE_RESIDENCY_P01_RESULTS.md` were reviewed with the
current TE→DiT→video/audio lifecycle. P0's early two-block CPU prefix and retention
across TE startup are preserved. P1 remains experimental and disabled by default.

The targeted dual-T4 capture contains two Native-correct full requests. In warm
TE blocks 2–49, read wall/CPU is 88.824/9.007 s on rank 0 and 89.167/9.039 s on
rank 1. Serial producer observations contain ~80 s off CPU/rank, only 0.30–0.34 s
runnable delay and ~87% D-state samples, predominantly `folio_wait_bit_common`.
Producer storage accounting is 11.649/11.656 GB versus 11.711 GB logical payload
per rank. Caller major faults are zero. Shared cgroup counters show extensive
file refaults and background reclaim near 30 GiB. These are compatible parts of
one chain: missing/unready pages → backing/layered I/O → producer wait → delayed
staging and H2D submission.

Both-GPU inactivity is 68.443 s in the captured warm TE stage; 61.017 s intersects
verified required-weight Future dependencies. Ranks are unioned. Kineto kernels
and memcpy use CPU-fitted clocks; initial CUDA-event anchor uncertainty does not
bound later absolute drift. These numbers are opportunity bounds, not recoverable
latency or idle-cache speedups. Pinning totals only 1.87/1.74 s per rank.

The previous small isolated probes are insufficient to identify physical storage
bandwidth: inode-specific DONTNEED produced virtual-device reads while backing
device reads stayed zero. The full pipeline reached backing storage at roughly
250 MB/s aggregate in that allocation. ~131 MB/s per-reader effective rate is
not a measured physical bandwidth limit. Buffered read_bytes, source residency,
and whole backing-device counters must be interpreted together.

## Audited read path and layout

`patches/rank_local_int8.py:load_offline_block` checks source/manifest identity,
constructs only the rank-local shell, iterates its state_dict and calls
`CheckpointReader.read_tensor` for each item. Each call allocates a final CPU
tensor, opens the shard with unbuffered Python I/O, seeks to the exact safetensors
offset and readinto's the destination. Tensor hashes are verified once per
reader lifetime, not silently omitted on cold startup. The model is assigned
these unchanged tensor bytes, pinned by the existing cache/window and copied
through its event-owned H2D stream. The baseline does not mmap or pread.

A representative rank-local TE shard is 243,980,376 bytes, containing 243,978,752
payload bytes and 18 tensor entries. Physical layout places scales first, then
norms and alphabetically ordered matrices. Model-key order alternates large
matrices and their early scale offsets. It makes **eight backwards seeks** and
about **2.049 GB of logical seek distance** while returning only 244 MB payload.
Seek distance is not extra bytes read, physical head movement or proof of an
8× penalty on virtual/SSD storage. Repeated descriptor opens also reset the
per-open readahead context. Their actual cost needs the controls below.

Norms and scales are intentionally replicated where TP semantics require them.
Each rank reads its own immutable shard; the two large matrix halves are not
the same payload. There is no evidence that both ranks read the complete Native
block. Repeated cross-request reads are a residency issue; per-load read ordering
is a distinct potential inefficiency. Startup vision/embedding reads still use
the Native checkpoint and must not be confused with transformer shard traffic.

## New physical controls

[TE Storage Controls](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-storage-controls),
version 1, completed using all 100 immutable Native-qualified TE shards (~24.4 GB)
on Kaggle. No model weights are uploaded from the local machine. GPU checks
require actual dual T4; this experiment measures that host's I/O, not GPU inference.

Controls include: small inode-dropped/immediately reread subsets; full natural
and repeated scans with low anonymous memory; the same scans with a bounded,
guarded 16 GiB anonymous-memory retention control; reversed-order repetitions
of current reads, offset-sorted persistent-descriptor reads, 8 MiB chunks,
SEQUENTIAL advice, bounded next-8-MiB WILLNEED, preadv and materialized mmap;
and serial-vs-overlapped current rank readers. There is at most one reader per
rank, never speculative extra I/O workers. Releasing the anonymous control gives
another cache-friendly full scan. It is diagnostic memory, not duplicated
checkpoint storage or an inference policy.

Every tensor is checked against the preserved Native-qualified hash outside
read timing. mmap also materializes independent final CPU storage; a zero-copy
view cannot be compared unfairly with a materialized baseline. No checkpoint
format, state shape, scale or quantization metadata changes. Sorted reads bundle
ordering and a persistent descriptor; a win would require a subsequent control
to separate those two mechanisms before making a root-cause claim about either.

Record each file offset/order/size, whole-block wall/thread CPU, scheduler counters,
source mincore residency, producer I/O/faults and sampled wait states. Sample shared
cgroup/device counters once per common phase, including file refaults, reclaim,
PSI and memory events. Mount and block/slave topology are recorded. Device sectors,
read-time/operation and weighted-time/phase allow aggregate throughput, completion
latency and mean outstanding-I/O estimates, not latency distributions or exclusive
TE device attribution. Do not sum physical devices with their partitions/loop/dm
layers. Phase wall includes verification/controls outside read timing and is not
pure media service time.

No global cache drops, sysctl changes, privileged tracing, mlock or full-model
host preload is used. DONTNEED affects only this job's immutable, fsynced shard
files; it is labelled inode-cold, never physical-cold. Provider cache coldness
and other tenants' storage activity remain uncontrolled. Exact folio lock owner,
storage-vs-filesystem wait duration and GIL ownership remain unavailable.

## Idle-pilot validity defect

The first idle pilot's two baseline and two idle-labelled full requests passed
Native checks, but idle preparation failed immediately on both ranks. The
benchmark's `_prepare` wrapper did not forward the cancellation keyword arguments.
Both ranks report zero idle loads/H2D/hits and idle_errors=1. Its latency difference
must not be attributed to prefetch. The wrapper is repaired and a real-CUDA test
now runs idle preparation with that disabled observer installed. Compact Native
requests and trained TE reuse pass again. The initial repaired submission was
capacity-rejected; the continuation subsequently submitted version 2 successfully.
Version 2 completed all 14 full requests with Native/ownership gates passing.
Observed standalone median E2E improves 4.80% at 15 s, but the runner allows
prefetch into reference encoding whereas ComfyUI pauses at prompt start.
Lifecycle equivalence is therefore partial; no production default is promoted.

Compact provenance: `kaggle_te_io_storage/source_binding.json`,
`semantic_results/te_shard_io_qualification.json`, selected previous trace archives,
and the failed pilot's preserved result/logs. Normal inference imports none of
the new I/O-control instrumentation.
