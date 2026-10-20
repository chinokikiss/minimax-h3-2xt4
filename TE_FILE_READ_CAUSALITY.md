# TE file-read causality experiment

## Verified finding — 2026-10-14

The physical dual-T4 job is **COMPLETE**. All 35 isolated cases preserved
Native-qualified shard bytes, and both full requests passed Native output
checks (exact conditioning/latents/video, accepted mono waveform rounding).
Inference policy and numerical operations were unchanged.

The primary cause is **natural file-cache eviction/refault followed by backing
storage reads**, during which the producer sleeps waiting for folios. This is
not primarily CPU tensor work, caller major page faults, or direct reclaim
inside the reader. Those distinctions are supported by separate counters;
file-cache loss and storage readiness form one causal chain.

| Real reference + TE stage | Cold request | Warm request |
|---|---:|---:|
| Stage latency | 127.423 s | 99.854 s |
| Full request latency | 418.701 s | 368.651 s |
| Rank 0 block-read wall / CPU | 56.004 / 9.441 s | 88.728 / 9.322 s |
| Rank 1 block-read wall / CPU | 55.044 / 9.442 s | 88.687 / 9.356 s |
| Rank 0 median aggregate read time per block | 1.091 s | 1.766 s |
| Rank 1 median aggregate read time per block | 1.080 s | 1.771 s |
| Inspected source-page residency, rank 0 / 1 | 12.37 / 12.37% | 2.43 / 2.36% |
| Kernel-accounted read bytes, rank 0 / 1 | 10.901 / 10.892 GB | 12.494 / 12.508 GB |
| Caller major faults in block reads | 0 | 0 |
| Sampled read D states, rank 0 / 1 | 948 / 910 | 1579 / 1544 |
| Of those: `folio_wait_bit_common` samples | 935 / 896 | 1564 / 1533 |

These diagnostic latencies retain observer overhead; they are not a matched
optimization comparison. Instrumentation outside read-call windows totals
about 5.9–6.8 s per rank and can overlap between ranks. The two rank wall times
must never be added to estimate end-to-end savings. CPU counters are bracketed
by read observers, and sample counts do not yield exact D-state durations.

### Reclaim is mostly background, and warm does not mean disk-resident

Across one common, bounded block-read interval per request, cgroup counters
increase as follows. These are endpoint deltas measured once, not sums of
overlapping read windows or rank counters:

| Counter | Cold read interval (98.288 s) | Warm read interval (98.983 s) |
|---|---:|---:|
| `workingset_refault_file` | 10,551,564 pages | 12,012,174 pages |
| `pgscan_kswapd` | 12,681,920 pages | 14,940,992 pages |
| `pgsteal_kswapd` | 12,178,573 pages | 14,926,359 pages |
| `pgscan_direct` / `pgsteal_direct` | 5,956 / 5,956 pages | 9,970 / 9,970 pages |
| CPU quota throttle events | 0 | 0 |

More than 99.9% of these reclaimed pages are attributed to `kswapd`, not direct
reclaim. They cover the cgroup, including other model/file activity; they do
not identify an individual victim inode. The TE's own extent observations
and fresh read traffic independently show that its pages are largely missing
when reused. No experimental eviction/advice occurs inside these requests.
Caller major faults and observed swap-in/out are zero. Exact GIL/runnable
delay remains unavailable, but the overwhelmingly D-state folio waits and
fresh I/O support a storage/page-readiness wait as the dominant explanation.

The actual cgroup limit is **30 GiB**, with `memory.high=max`, no configured
`io.max`, and a four-core CPU quota. In the warm example below, cgroup usage is
29.80 GiB: anonymous memory is 20.50 GiB, `file` 8.98 GiB (including 1.11 GiB
shmem), and kernel charges about 0.32 GiB. System MemAvailable still says
8.42 GiB, because it includes reclaimable cache and is not a useful-cache
reservation. The complete TE shard working set is ~24.4 GB across ranks.
It cannot remain cached alongside the current anonymous working set. The
managed HostBudget peaks at 15.48 GiB and reports zero admission denials;
staying within that budget therefore does not prevent kernel cache churn.

### Concrete dependency and backing-I/O evidence

Warm request, **76.608–77.125 s after reference + TE starts**, rank 0,
`te.block38.rank0.safetensors`, one 65.536 MB tensor read:

- Wall 516.375 ms; calling-thread CPU 46.944 ms.
- Only 65 of 16,001 inspected source pages resident before reading (0.406%).
- Calling-thread `read_bytes` +65.274 MB, major faults 0, minor faults 16,000.
- All 11 sampled states are D, in `folio_wait_bit_common`.
- Cgroup file refaults +62,387 pages; background steals +91,331;
  direct scans/steals 0; CPU throttling 0.
- Backing `sdb` reads ~130.18 MB during that window, about **252 MB/s**
  aggregate, busy for 487 ms. The paired rank is reading too. The several
  loop/device-mapper entries represent the same layered I/O, not additive
  independent disk traffic. This does not prove a provider-enforced rate cap.

Warm request, **33.087–35.105 s**, rank 0's foreground Future for TE block 16:
one matching preparation thread and exact Future publication bounds link
1.843 s of read wall time to that wait. The next same-thread H2D enqueue is at
35.105 s. Across the warm stage, rank 0 / 1 foreground Future wait unions are
87.123 / 85.234 s; linked producer reads overlap 78.957 / 77.602 s. There are
46 unambiguous links per rank; two waits remain unresolved. This establishes
an execution dependency, not a complete GPU DAG or equally removable latency.

### Why the small cold probes were deceptively fast

The notebook's `/tmp` files live on the root **overlay**; device counters show
loop/device-mapper layers and a backing `sdb`. File-specific `DONTNEED` drops
the checkpoint inode's pages, but does not make all lower storage layers cold.
In each isolated dual-rank cold repetition, the virtual layer accounts for
~975.93 MB of reads, while backing **8:16 (`sdb`) reads zero bytes**. The
read returns through a cached lower layer. Immediate rereads cause no virtual
or backing reads. Thus caller `read_bytes` is not proof of physical-device
traffic in this environment.

In the full warm interval, `sdb` reads **24.810 GB over 98.983 s (~251 MB/s)**,
while the virtual layer accounts for ~25.003 GB. It is now reaching the backing
storage. This supports layered-cache loss plus backend supply latency as the
explanation for the small-probe/full-pipeline difference. Exactly which lower
cache satisfies the isolated reads is an inference from the layered counters;
the complete loop/slave graph and provider service limit were not captured.
The first slow isolated dual cold case also has ~488 MB of backing writes,
so it is confounded by writeback and cannot prove sustained read contention.

## Targeted solutions, ranked

1. **Protect useful prepared weights and schedule next use explicitly.**
   Current `decode_start` prepares startup/vision only; transformer prefetch
   starts at `decode_complete`. TE startup construction then unconditionally
   calls `self.cache.trim()`, discarding prepared transformer entries. Test
   a bounded early TE prefix during video decode and conditional trimming
   based on actual admission pressure. Preserve that prefix through startup;
   retain DiT reuse within NFEs. This can reduce exposed reloads, but next-
   request lookahead does not shorten the current request's decode. No new
   latency reduction is claimed until a controlled full-request ablation.
2. **Optimize the storage/memory path, not just CPU thread count.** Account
   for the 30 GiB cgroup limit, anonymous/pinned/unmanaged memory, and useful
   cache headroom. More speculative staging can replace useful cached pages.
   Compare real large-working-set reads on alternative checkpoint mounts and
   aligned direct-I/O into a bounded pool where supported, verifying every
   tensor and buffer lifetime. `/kaggle/working` is another loop-backed mount;
   relocating there alone is not demonstrated to improve throughput. A simple
   larger file-cache allowance may still lose to cyclic scans, so protected
   reuse and a measured placement policy matter. The ~250 MB/s backing supply
   is a real limit in this run; its exact infrastructure cause remains open.
3. **Reuse touched final/staging buffers as a secondary improvement.** In
   isolated, identical-byte controls, median summed read CPU falls from
   0.651 s to 0.298 s per ~976 MB. This reduces destination minor-fault/first-
   touch costs, not the ~79 s per-rank nonrunning component. Any pool must
   remain charged to HostBudget and honor CPU/GPU consumer lifetimes. Do not
   count the isolated result as an end-to-end speedup.

Do not promote extra CPU workers, sequential advice, or `WILLNEED` as a fix.
Two-second `WILLNEED` increased inspected residency only from about 2.05% to
2.56% in these probes, and includes extra lead time. The earlier physical
parallel-preparation candidate did not pass the TE acceptance threshold.

Evidence: `kaggle_te_io_output/te_io_results/io_analysis.json`,
`io_dependencies.json`, `pipeline/io_reads_r0.jsonl`, `io_reads_r1.jsonl`,
per-rank `batch_r*_causal.json`, `probe/io_probe.json`, capabilities and
Native `pipeline/result.json`. Reproduce postprocessing with:

```powershell
python analyze_te_io.py kaggle_te_io_output/te_io_results
python analyze_te_io_dependencies.py kaggle_te_io_output/te_io_results
```

The following sections retain the experiment design and early observations
for provenance; the completed findings above supersede their pending status.

The existing dual-T4 trace shows the TE reader sleeping in
`folio_wait_bit_common` during buffered `readinto` calls. This is evidence of
a kernel wait involving a file folio; it does not identify the page's lock
owner, underlying device latency, or why the page was unavailable. The
possible chain is **file-cache eviction → reread → page readiness wait**.
Those causes can coexist rather than form mutually exclusive explanations.

Original populated block-read medians, reused physical CPU capture:

| Request | Rank 0 median | Rank 1 median |
|---|---:|---:|
| Cold managed caches | 1.130 s | 1.135 s |
| Warm managed caches | 1.873 s | 1.868 s |

Evidence: `kaggle_cpu_audit_output/cpu_audit_results/continuous/offline_p1_p2/kineto/te_critical_path.json`.
These are aggregate file-read wall times per populated block, not every tensor
call, exclusive CPU time, or physical disk service times.

## Physical experiment

[MiniMax H3 TE File Read Causality](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-file-read-causality),
version 1, submitted 2026-10-14 to physical dual T4. The existing inference
policy and numerical operations remain unchanged. There are no additional
denoising evaluations inside the measured requests.

1. Before installers or large-model downloads, download only four immutable,
   Native-qualified shards: TE blocks 10 and 25 for each rank (~976 MB total).
   Verify object SHA256 and every resulting tensor's dtype, shape, and SHA256.
2. Run three repetitions of single/dual rank reads, file-specific best-effort
   guest page-cache eviction, immediate rereads, reused/touched destination buffers, sequential
   read advice, and two-second best-effort `WILLNEED` advice. Observe actual
   residency rather than assuming that hints worked. Counterbalance the
   single/dual order. File-specific advice does not flush hypervisor, device,
   or remote storage caches. Anonymous-retention controls target at most 12 GiB with
   conservative RAM headroom; record the achieved retention and whether any
   reclaim actually occurred. A small hot working set need not be evicted.
3. Restore the same complete checkpoint set and saved Native references, then
   run two continuous full 50-layer Ref2VA requests (512×512, 124 frames,
   four actual NFEs). Apply only narrow TE read observers. Both requests must
   pass the existing Native output contracts. No Kineto or GIL profiler is
   added; existing GPU traces remain the GPU evidence.
4. Correlate each actual read with its rank, source extent, block, request,
   CPU counter, page residency, faults, context switches, storage-accounted
   traffic, memory scan/refault counters, PSI, CPU quota counters, sampled
   states/wait channels, and kernel stacks where permissions allow them.

The full pipeline uses the existing pinned dependencies. Early isolation
uses Kaggle's preinstalled PyTorch; `early_source_identity.json` records its
version. Byte checks guard the actual reader results. CPU allocator timings
across these two dependency environments are not treated as an ablation.

## Preliminary live results

All **35 isolated cases completed**, with exact Native-qualified shard byte
checks. These early live-log observations preceded the full-pipeline capture;
the completed finding above explains their layered-cache limitation.

| First repetition | Sum of read wall times | Sum of calling-thread CPU | Storage-accounted bytes | Residency before reads |
|---|---:|---:|---:|---:|
| Single rank after eviction (~488 MB payload) | 0.3304 s | 0.3257 s | 487.93 MB | 2.05% |
| Two ranks after eviction (~976 MB payload) | 1.5430 s | 0.6903 s | 975.86 MB | 2.05% |
| Two ranks, immediate reread | 0.5289 s | 0.5278 s | 0 | 100% |

These sums combine independent rank read intervals; **they are not dual-rank
elapsed latency**. Other cold dual-rank repetitions were much faster, so the
first slow sample cannot prove persistent bandwidth saturation. Buffered
reads after eviction incurred new I/O despite **zero caller major faults**.
Immediate rereads had no sampled D-state waits and essentially no nonrunning
time. The bounded anonymous-retention controls also retained this small hot
working set; that does not rule out eviction in the full model's larger
working set. Full-request evidence is required before attributing the old
1–2-second block stalls to storage, reclaim, or their combination.

## How conclusions will be accepted

| Explanation | Evidence needed |
|---|---|
| Source-cache misses with storage/page-readiness waits | Low extent residency before reads, new calling-thread storage-accounted bytes, D-state folio/read wait stacks, CPU much smaller than read wall time; immediate cached rereads remove the exposed wait. |
| Repeated reclaim/refault upstream of I/O | Native request cache residency falls between uses; scan/steal/refault counters rise over that interval; available/headroom/anon/file charges support pressure. Separate explicit experimental eviction from natural reclaim. |
| Direct reclaim during destination allocation/copy | Allocation/minor-fault or scan/direct-reclaim activity at the reader, memory PSI or reclaim stacks; reused/touched buffers reduce that component. Source residency and bytes alone cannot prove direct reclaim. |
| Shared storage contention | Both ranks reading together consistently increase per-rank elapsed reads versus counterbalanced serial reads, with fresh I/O and relevant device/cgroup evidence. Aggregate kernel counters may include other jobs. |
| CPU quota, descheduling, or GIL reacquisition | Slow reads even with resident data and no fresh I/O, supported by runnable/quota/futex evidence. No exact GIL duration is claimed when that coverage is unavailable. |

`mincore` measures residency, not folio readiness/lock ownership. Caller major
faults measure a different mechanism from buffered file-cache misses.
Memory PSI alone is not proof of direct allocation reclaim: Linux can account
waiting for a non-uptodate, working-set folio as a memory stall while it waits
on the folio lock. Readiness I/O and memory-thrashing attribution can overlap.
Direct-reclaim claims therefore require scan/steal counters and relevant
stacks, rather than adding memory PSI and I/O PSI totals. See the
[kernel folio-wait implementation](https://code.googlesource.com/linux/torvalds/linux/+/4bf54e47525dcacd4c6cdd97fb5902592414dd7a/mm/filemap.c).

`/proc/.../io` distinguishes returned characters from storage-accounted bytes,
but does not identify device service-time ownership. Global/cgroup counters
include other activity and must not be summed across overlapping read windows.
See the [Linux I/O accounting documentation](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html),
[mincore documentation](https://man7.org/linux/man-pages/man2/mincore.2.html),
and [cgroup memory accounting](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).

## Reproducibility and output bounds

`benchmarks/te_io_pipeline.py` and `benchmarks/te_io_telemetry.py` are opt-in.
The local logical-rank full-request qualification matches all five preserved
uninstrumented outputs bit-for-bit; this qualifies semantics only. The Linux
smoke verifies actual mincore residency, forwarded bytes, and I/O counters.
Frozen source hashes are in `kaggle_te_io_audit/source_binding.json`.

`benchmarks/te_io_probe.py` applies hints only to its own verified checkpoint
files, without deleting or rewriting them or changing global cache/VM policy.
The pipeline workers close before any further read-probe policy can apply.
Each rank's detailed read capture is capped at 20,000 rows with an explicit
drop counter; analysis rejects truncated evidence. There are no tensor dumps
or full Kineto traces. `analyze_te_io.py` emits a compact report, separates
boundary-crossing reads, and clips union exposure rather than adding rank
waits. Profiling overhead is retained and cannot establish speedup by itself.
