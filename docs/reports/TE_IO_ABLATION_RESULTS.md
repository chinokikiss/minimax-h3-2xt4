# TE I/O and residency ablation results

Status: physical storage controls and repaired idle pilot version 2 COMPLETE.
No new inference optimization accepted.
P0 retained. P1 and idle residency remain opt-in. This file distinguishes
completed evidence from pending measurements.

## Idle pilot v2: measured standalone improvement, production boundary unverified

All 14 actual full requests pass Native contracts and cleanup with no observed
OOM/idle errors or decoder fallback. At a 15 s gap, warm medians are TE
98.685→81.108 s (−17.81%) and E2E 348.464→331.742 s (−4.80%), two warm
observations/policy with reversed order. Zero-gap median E2E is
352.770→350.523 s (−0.64%), with no idle copies submitted.

The standalone benchmark allows idle prefetch for 2–4 s into the next request's
reference encoding; the actual ComfyUI prompt-start hook pauses it earlier.
Consequently these numbers do not establish a ComfyUI speedup or an idle-only
15 s benefit. Keep the feature opt-in and run matching lifecycle boundaries
before promotion. Small samples do not establish significance or stable p95.
See `TE_IDLE_RESIDENCY_RESULTS.md` and the source-bound measured summary.

## Physical storage controls v1: complete

[MiniMax H3 TE Storage Controls](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-storage-controls)
completed 22 controls and **2,008 actual trained block loads**, all Native tensor
hash correct. The anonymous-pressure control reached its 16 GiB target; the
actual cgroup limit is 30 GiB and CPU quota four cores. OOM/kill/max-event
counters remain zero across the study. No global cache settings changed.

All full sweeps return the same 24.398 GB of tensor payload. The following
numbers are **whole I/O-test phase wall**, including hashes, observation and GC
outside timed reads. They are neither TE latency nor media service time.

| Reader | Initial order | Reversed order | Phase change vs respective baseline |
|---|---:|---:|---:|
| Current baseline | 94.616 s | 85.234 s | — |
| Sorted, persistent descriptor | 87.907 s | 81.835 s | −7.09% / −3.99% |
| Sorted, 8 MiB chunks | 88.036 s | 81.824 s | −6.95% / −4.00% |
| Sorted + SEQUENTIAL | 87.383 s | 81.385 s | −7.64% / −4.52% |
| Sorted + bounded WILLNEED | 89.464 s | 84.279 s | −5.45% / −1.12% |
| Sorted + preadv | 85.296 s | 80.398 s | −9.85% / −5.67% |
| Materialized mmap | 114.670 s | 107.838 s | +21.20% / +26.52% |

**Pread is a candidate, not a validated inference winner.** The union of actual
read intervals across ranks changes from 82.104 to 79.305 s (−3.41%) initially,
and 67.877 to 67.305 s (−0.84%) in reverse order. Sorted alone slightly increases
this union in both passes. This is why the larger whole-phase differences must
not be claimed as exposed GPU wait savings. Pread also uses more reader CPU
than baseline. Two passes do not establish significance, and no policy is promoted.

**File residency remains poor even with low anonymous memory.** Immediate reread
of the small subset has 100% source residency and a 0.299 s read union versus
2.418 s after inode-specific eviction. Full-set low-anonymous scans have only
4–5% residency and 23.388/22.358 GB reads on whole device `sda`, despite only
~1.2 GiB anonymous memory and ~25–26 GiB charged file cache. Under 16 GiB added
anonymous retention, residency is still ~5% initially and rises to ~8–11% later.
Releasing that retention does not immediately restore the full working set.
More aggregate file-cache headroom alone did not solve cyclic/layered residency
in this allocation. Double caching/layer-specific reclaim remains a hypothesis;
the experiment does not identify each inode's lower-layer ownership.

**Storage supply is not a proven fixed cap.** Initial baseline `sda` traffic is
20.978 GB at 221.7 MB/s across the full phase; reversed baseline is 17.721 GB at
207.9 MB/s. These average rates include verification gaps. Cache residency,
physical reads and supply change over time. The inode-dropped small subset
returns 0.976 GB but reads only 0.488 GB on `sda`, confirming that inode eviction
does not establish physically cold storage. Do not sum `sda` with partitions,
loop/dm layers, or unrelated `sdb` activity.

Serial rank readers have a 134.223 s phase versus 85.234 s for the preceding
overlapped baseline, but read unions are 65.444 versus 67.877 s. Overlapping
post-read verification and other phase work explains part of the phase gap;
it does not prove unused storage concurrency capacity. Extra workers remain
unsupported. The substantial forward/reverse drift also prevents causal
attribution of all observed differences to reader policy.

Evidence: `kaggle_te_io_storage_output/te_storage_results/storage_controls.json`,
`storage_summary.json/.md`, `storage_blocks.csv`, `capabilities.json`, `layout.json`
and immutable source identities. Reproduce with:

```powershell
python analyze_te_storage_controls.py kaggle_te_io_storage_output/te_storage_results
```

## Completed idle pilot v1: invalid performance experiment

| Physical pass | Warm reference+TE | Warm E2E | Idle GPU loads, rank 0/1 | Verdict |
|---|---:|---:|---:|---|
| P0, 15 s gap | 101.557 s | 372.933 s | — | Two full Native-correct requests |
| Idle-labelled P0, 15 s gap | 100.437 s | 367.603 s | 0 / 0 | Reject idle-performance attribution |

Each idle window records one background preparation failure. The diagnostic
wrapper dropped cancellation options; no idle H2D or reuse occurred. All four
requests pass numerical checks and finish with zero managed/pending preparation
balances. No OOM/kill was observed in their request/gap counters. The matrix
stopped at its activation gate before reversed/zero-gap confirmation. The
single small latency difference is uncontrolled run variation, not an accepted
idle speedup or an estimate of attainable improvement.

## New controlled experiments

| Ablation | Physical status | End-to-end claim |
|---|---|---|
| A: existing P0, unchanged operators | Historical measured reference | Preserve as baseline |
| B: useful file-cache residency | Full-set low-anonymous/pressure controls complete | Aggregate headroom alone insufficient in this allocation |
| C: read ordering/advice/pread/mmap | Full-set repeated/reversed controls complete | Pread follow-up candidate; mmap loses; no E2E winner |
| D: repaired bounded idle GPU residency | V2 complete, 14 Native-correct requests | Standalone −4.80% E2E; ComfyUI boundary differs |
| E: independently qualified combination | Not launched | No combined gain claimed |

Live continuation state: `semantic_results/te_io_continuation.json`. The bounded
helper collects storage outputs and submits the frozen repaired idle pilot once
a slot is free; successful submission is recorded with its actual version.
Generated measured summaries are `storage_summary.json/.md` and
`idle_validated_summary.json`. These may complete after this status document;
their verdicts and source bindings must be reviewed before promoting a policy.

The storage study uses every TE shard and reports logical, thread-accounted,
virtual-layer and actual whole-device reads separately. File-specific eviction
does not establish provider-cold storage. Small-subset and full-set controls have
different roles; their latencies are not compared as equal-work speedups.

## Acceptance protocol for full-pipeline comparisons

Keep 50 TE/DiT layers, identical prompts/reference pixels, seed1729, four real
NFEs and existing Native contracts. Test 512×512/124 frames and a smaller
384×384/124-frame full workload; the smaller case requires its own freshly
generated Native references. Do not reuse 512px tensors as its reference.
Separate fresh-managed-cache first requests, repeated warm requests, steady
zero-gap sequences and actual 5/15/60 s think-time gaps. Cold provider cache
cannot be promised. Keep P1 independent and disabled in primary P0 comparisons.

First require a short activation/correctness/safety pilot. Then run matched,
counterbalanced policy sequences and independently confirm any candidate across
allocations. Predeclare primary outcomes as reference+TE and output-ready E2E;
collect at least ten matched warm observations per selected comparison where
allocation budgets permit. Publish every observation, paired differences and
paired-bootstrap confidence intervals, with allocation/order strata. A confidence
interval excluding zero is evidence only for the sampled workload; samples
sharing an allocation are not independent host replications. With insufficient
data report descriptive median/range and sample counts, without significance.
Do not present stable p95 estimates from two observations; retain raw samples
and uncertainty when enough observations exist for a descriptive tail estimate.

Timing runs use identical bounded counters, without Kineto/mincore/OS observers.
Separate diagnostic runs capture actual Future producer/consumer, H2D and both-
GPU inactivity unions inside the same full request, not extra evaluations.
CPU/Kineto clock alignment is verified. Do not sum ranks, queue residence,
read wall or inclusive Future time into a latency-savings claim.

Report background and foreground bytes/time together, avoided actual file/H2D
on valid cache hits, producer stalls, total request+gap wall, GPU/RAM peaks,
refaults/reclaim, cache evictions and cancellation/model-switch behavior.
Every numerical mismatch, decoder fallback, recovered idle allocation error,
OOM/kill, leaked reservation or incomplete request disqualifies the matched
performance comparison. No optimization is promoted solely for faster raw reads
or a higher cache-hit rate.
