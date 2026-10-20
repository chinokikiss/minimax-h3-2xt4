# TE residency P0/P1 audit

Status (2026-10-14): implemented, independently opt-in, locally qualified.
After the initial GPU-slot rejection, version 1 was successfully submitted once
a slot became available. Kaggle reports COMPLETE for
[the physical benchmark](https://www.kaggle.com/code/chinokiki9/minimax-h3-residency-p0-p1).
All 21 full requests passed correctness checks. P0+P1 reduced warm end-to-end
latency by 1.98% initially and 2.98% in confirmation. TE gains are small and
memory pressure remains; defaults remain unchanged. See
[physical results and attribution limits](TE_RESIDENCY_P01_RESULTS.md).

The physical evidence in `TE_FILE_READ_CAUSALITY.md` establishes backing reads
after file-cache eviction/refault, rather than useful CPU execution or direct
reader reclaim, as the dominant TE preparation delay. Warm requests read about
24.8 GB from the backing device. The actual cgroup limit is 30 GiB; the managed
HostBudget peak of 15.48 GiB does not describe total memory pressure.

## Audited lifecycle before changes

- `decode_start` clears the old transformer tail and prepares only startup/vision.
- `decode_complete` starts transformer preparation only if startup has no pending
  work. This misses the video decoding interval and can skip the handoff entirely.
- `_phase('te')` preserves matching keys with `begin_request`, then unconditionally
  calls `trim`, deleting every unborrowed ready transformer entry. Running tasks
  are not cancelled by trim and can repopulate the cache, making outcomes timing dependent.
- Startup construction pressure also calls whole-cache trim. Entries have only
  wanted/not-wanted and LRU priority, without distance or preparation cost.
- Each rank has separate startup and transformer workers sharing a byte budget.
  Their admission uses system MemAvailable, without cgroup current/stat/events/PSI.
- DiT's GPU window already retains weights across actual NFEs. GPU eviction uses
  next-use distance, and pinned sources survive until copy events complete.
  These mechanisms must be preserved. CPU horizon updates currently ignore
  kernel memory pressure and repeatedly replace the wanted set.

## Implemented controlled changes

P0: an opt-in two-block CPU prefix during decode; retain it across startup;
pressure eviction removes far/unwanted entries first, with observed reload cost
as a tie-breaker; do not let speculation displace a nearer prepared block.

P1: independent opt-in cgroup-v2 admission, explicit file-cache reserve and
unmanaged/kernel/shmem accounting, bounded shared preparation slots, conservative
in-flight construction accounting, and pressure-capped CPU/staging lookahead.
Foreground allocations retain a smaller page-cache floor to permit progress.
Missing cgroup telemetry is reported and falls back to the existing guard.

Both switches default off until physical results justify promotion. Local logical
ranks establish semantics only. Physical timing and targeted profiling are separate;
all cases use the same full Native Ref2VA workload and Native reference artifacts.

`RuntimeConfig.prefetch_lifecycle` enables P0; `memory_pressure_aware` enables
P1. The decode prefix defaults to two blocks per rank (configurable 1–4).
Eviction prioritizes unwanted/farther entries, then cheaper reloads; speculative
admission cannot evict a nearer wanted block. Required startup allocations
release bounded local ownership under pressure instead of whole-cache trimming.

P1 projects `memory.current - (file - shmem)` plus pending construction/staging,
requested bytes, protected existing file cache and the 3 GiB host reserve.
This includes actual anonymous, pinned/unmanaged, shmem and kernel charges.
Speculation protects up to 6 GiB of existing file cache; foreground demand
protects up to 2 GiB to permit progress. Two preparation slots are shared across
ranks and startup/transformer caches. Staging reserves bytes separately, so
consumers cannot occupy all producer slots. Pressure caps CPU depth and denies
speculative staging. The original managed-byte and system-RAM guards remain.

These bounds are conservative while producers allocate. They reserve aggregate
file-cache capacity, not particular kernel pages, and cannot guarantee zero
refaults. PSI is a pressure signal, not proof of direct reclaim or a separate
stall duration. Missing/unlimited cgroup telemetry is reported and falls back
to the original host guard. GPU event dependencies and DiT NFE reuse remain
the existing implementation; no mathematical operators were changed.

## Validation and A/B status

| Check | P0 | P0+P1 |
|---|---|---|
| Compact trained full Ref2VA on local logical ranks | PASS | PASS |
| Five outputs vs saved uninstrumented outputs | Bit exact | Bit exact |
| Native output checks, including accepted audio contract | PASS | PASS |
| Physical timing / backing I/O / NFE reuse | Complete; see results | Complete; see results |
| Full-pipeline GPU idle / pressure elimination | Partial | Partial |

Three new tests cover prefix retention/eviction order, 30 GiB cgroup/shmem/file
accounting and slot bounds, and failed-producer lease cleanup. Spawned config
and shared-budget qualification pass, as do two legacy loader tests, four
causal analyzer tests and two CPU observer tests.

Local numerical qualification uses two trained TE/DiT layers, full Native
front ends/VAEs, 128px/five frames and four real NFEs. A two-request Windows
attempt reached the host admission guard at second TE startup and was stopped
without relaxing it. Physical continuous requests now complete successfully
for the stated workload; broader capacity/race guarantees
remain unverified.

The frozen physical matrix uses all 50 layers, 512×512/124 frames, four actual
NFEs, identical seed/reference/prompt and restored Native outputs. Checkpoints
are restored anonymously on Kaggle. Short passes run Baseline → P0 → P0+P1
(two requests each); continuous passes reverse the order (three requests each).
Managed caches start fresh per policy; OS/provider cache coldness is uncontrolled.

Timing uses identical compact counters and bounded memory samples. Separate
two-request profiles capture warm TE kernels/Futures and cross-request TE CPU
preparation. GPU idle outside the TE capture is not directly measured; full
DiT/VAE stage timings and window transfer/reuse counters remain available.
Counter boundaries are outside recorded request latency and contribute to the
recorded handoff gap; no deliberate sleep/check/export occurs between requests.
Inclusive rank waits and layered device traffic must not be summed. Detailed
Kineto traces are compressed with a SHA256 round trip; normal inference enables
neither observers nor tensor dumps. Eviction-before-use means before CPU borrow;
the Future/kernel timeline is needed to establish GPU exposure.

The initial launch returned `Maximum batch GPU session count of 2 reached`. Both
`h3-quality-first-t4-pooling` and `h3-quality-first-t4-quantization` were RUNNING;
neither was terminated. A subsequent submission created version 1, now COMPLETE.

Rebuild to refresh private Native-reference read capabilities and submit when
a GPU slot becomes free:

```powershell
& 'C:\ProgramData\miniconda3\envs\comfyui\python.exe' build_residency_p01.py
& 'C:\ProgramData\miniconda3\envs\comfyui\Scripts\kaggle.exe' kernels push -p kaggle_residency_p01 --accelerator NvidiaTeslaT4 -t 19800
```

Promote defaults only after repeatable end-to-end gains, Native correctness,
backing-I/O counters, memory events and zero leaked leases support the change.
