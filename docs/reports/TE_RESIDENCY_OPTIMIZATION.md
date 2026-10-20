# TE residency optimization: changes and gates

No inference performance winner is selected. Preserve P0 and keep P1/idle optional.
All 22 physical storage controls completed with 2,008 Native byte-correct block
loads and zero OOM events. Pread merits a full-pipeline ablation; its read-union
benefit is only 3.41%/0.84% despite larger whole-test timing differences. Mmap
loses, aggregate file-cache headroom alone does not solve residency, and extra
workers are not justified. The repaired idle pilot version 2 is now running.

## Implemented now

- Repair `benchmarks/causal_trace.py` to forward all `_prepare` arguments.
  This restores cancellation-aware idle preparation when the same diagnostic
  wrapper is installed with capture disabled. The production load algorithm
  and all model operators are unchanged.
- Add a bounded `idle_last_error` string alongside existing error counts.
  A background failure is visible without a traceback/tensor dump or unbounded
  logging. Candidate acceptance also rejects recovered idle-window errors and
  cgroup OOM events, rather than accepting a silent fallback as a clean run.
- Keep idle weights in the existing event-owned ResidencyWindow, under per-rank
  VRAM caps, shared HostBudget and immediate command priority. No second cache
  manager, permanent TE allocation or extra preparation concurrency was added.
- Add I/O-only experiment readers in `benchmarks/te_shard_io.py`. They are not
  installed in inference. The physical study tests the same immutable bytes
  with offset ordering, descriptor advice, preadv and materialized mmap.

Local checks pass: eight actual trained-shard comparisons for modes supported
on Windows, CUDA idle reuse/cancellation/capacity/restart/disabled-observer
regressions, two trained TE idle/forward cycles, Native checkpoint/manifest
identity invalidation, and compact full Native requests for P0+Idle and
P0+P1+Idle. Full local video can use the existing admission fallback; this is
not physical capacity or speed evidence. Linux-specific modes are byte-checked
on Kaggle before their measurements are accepted.

## Candidate selection, after physical evidence

If low-anonymous/full-set controls reduce backing reads, test a narrow TE-only
anonymous/cache retention bound while preserving near-use CPU weights and DiT
NFE reuse. Do not assume a larger page-cache floor protects particular inodes,
and do not silently enable or expand P1. If offset/advice candidates improve
readiness at equal memory/bytes, integrate only the winning bounded TE-shard
path, preserving strict provenance/hash checks and all dtypes/metadata. If
neither improves full-pipeline exposure, retain the current loader.

Repair/validate bounded idle residency independently. Selection should account
for next-use order and verified exposed cost, available gap and per-GPU startup
headroom. A valid GPU hit must reuse the same physical tensors without a second
file read or H2D. Record prefetched/pending/copy-in-flight/ready/used/evicted
separately. Submitted reads or DMA cannot be instantly preempted; measure residual
foreground interference rather than claiming cancellation eliminates it.

Combine only independently accepted improvements. The success gate is repeated
physical reduction in TE or E2E latency with confidence/limits reported, Native
outputs preserved, no OOM or leaked ownership, and no unexplained transfer/I/O
increase. Bigger speculative depth or additional workers is not the default fix.

## Rollback and artifacts

`idle_residency=False` disables idle GPU prefetch. `memory_pressure_aware=False`
keeps P1 off. Retain `prefetch_lifecycle=True` for the P0 comparison. The I/O
candidates are standalone benchmark entry points and do not affect an ordinary
request. Keep the transparent observer repair even if idle performance fails.

The storage job uploads source and provenance only, downloads its TE shards
on Kaggle, and makes no global VM/cache setting changes. Its bounded anonymous
lease is diagnostic only and released after controls. Preserve baseline Native
references, source bindings and compact verdicts; keep full traces compressed
and limited to selected diagnostics. No user checkpoint/data is deleted.

Reproduction entry points: `build_te_io_storage_study.py`,
`benchmarks/te_io_storage_study.py`, `build_idle_residency.py`, and
`benchmarks/idle_residency_matrix.py`. A capacity-rejected Kaggle submission is
not a running experiment; the continuation subsequently submitted the repaired
pilot as version 2 after the storage job released its slot.

`benchmarks/continue_te_io.py` is a bounded, one-shot local continuation (8 h
deadline, compact state in `semantic_results/te_io_continuation.json`). It waits
for the authorized storage job, downloads/analyzes its compact artifacts, then
submits the already frozen repaired idle pilot after capacity is available.
It validates and collects the pilot without changing optimization policy or
automatically launching combined experiments. This is a running local helper,
not a recurring automation; it requires the local machine to remain awake.
