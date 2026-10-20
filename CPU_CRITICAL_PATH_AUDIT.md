# CPU critical-path audit — 2026-10-13

**Update:** the physical probe described below is COMPLETE. Its measured
producer/read/scheduler findings and TE loading ablation status are in
[TE_CRITICAL_PATH.md](TE_CRITICAL_PATH.md). The setup/pending statements below
describe the original capture plan. The py-spy failure was an incompatible
`--native --nonblocking` combination, not proven access denial; the new
targeted capture removes native unwinding and retains nonblocking GIL ownership
samples. Exact runnable/GIL wait durations remain unavailable in this probe.

The existing physical traces are reused first. The 150.64 s CPU-future wall
union does **not** identify 150.64 s of blocked CPU execution or removable
request latency. It mixes foreground and preparation threads, producer work,
and waits concurrent with GPU execution. Old captures have no scheduler
running/runnable counters or GIL state, and lack stable Future identities.

## Reused evidence and current conclusions

`analyze_cpu_path.py` correlates the original per-rank custom spans with
Kineto CPU APIs, kernels and copies using the existing clock fit. It takes
the intersection of inactivity on both physical GPUs, then intersects that
with non-ready foreground `Future.result` spans whose next CUDA submission
is on the same dispatch thread. Nested/thread/rank waits are merged into
interval unions before calculating exposure.

| Existing capture | Foreground-Future / both-GPUs-inactive intersection, request 0/1/2/3 |
|---|---|
| P1 | 122.44 / 121.93 / 121.15 / 122.04 s |
| P1+P2 | 103.00 / 103.57 / 104.54 / 103.42 s |
| Offline+P1+P2 | 85.87 / 79.77 / 73.54 / 27.49 s |

This proves a foreground control-flow dependency and simultaneously observed
inactivity. It is a **conservative wait-associated exposure measure**, not a
complete longest critical path or a promise that all those seconds can be
removed. A spinning/waiting NCCL kernel counts as GPU activity, so it can hide
CPU-induced lack of useful GPU progress from this conservative criterion.
The request-3 difference in the offline capture requires that inspection;
it is not automatically evidence that preparation became four times faster.

The largest mapped offline foreground waits are for TE block 0 (approximately
2.3–3.7 s individually). Those are separately observed from the 13–22 s
D2H/H2D gaps and TE embedding waits. Preparation events overlapping a gap are
recorded as candidates. The current evidence **does not establish GIL
contention, disk stalls, or speculative DiT preparation as their sole cause**.
Old traces cannot prove that arbitrary adjacent D2H and H2D copies move the
same tensor; the analyzer explicitly labels those chains unproven.

Per-case artifacts are under
`kaggle_causal_resume_output/causal_audit_results/continuous/<case>/kineto/`:
`cpu_path_analysis.json` contains every observed Future wait, producer
candidates, foreground dependencies, per-thread unions, and D2H/H2D gaps.
`cpu_gpu_timeline.html` shows the correlated CPU spans and actual GPU kernels,
with clickable intervals. CPU bars are inclusive wall time, not scheduler
running time. Original captures/results remain preserved.

## Targeted observers, with no inference changes

The CPU extension wraps original methods and calls each once. It changes no
weights, masks, RoPE, quantization, AdaLN, precision, cache policy, loading
order, request input, collective, or denoising evaluation count.

- Inclusive `thread_time_ns` counters for preparation, staging, state
  assignment, shard planning, module construction, hashing, allocation,
  pinning, eviction, and transfer submission. Counter unions avoid summing
  nested spans. Wall minus thread CPU is **nonrunning**, not automatically
  blocked time.
- Stable Future IDs and condition links, producer publication/consumer waits,
  ready-on-entry state, and readiness brackets. A publication transition lies
  between wrapper entry and exit; exact nanosecond readiness is not invented.
  Post-readiness return delay is therefore reported as a range.
- CUDA Event constructor IDs, record generations, producer streams, consumer
  wait/synchronize calls, and observed false/true query brackets. Async
  stream-wait submission duration is distinct from GPU dependency duration.
  Python Event waits/sets/clears and Condition waits/notifications are recorded.
- Checkpoint file read/seek calls aggregate by open file, avoiding one trace
  event per row. Read wall/CPU time, bytes and counts are separate from
  deserialization/assignment, allocations and pinning. These are logical read
  bytes; physical storage traffic and page-cache misses are not assumed.
- Tensor transfer submissions and results include source/destination, shape,
  size, request/block context and observed object/implementation tokens.
  Tokens do not prove lifetime/alias relationships through unobserved CPU
  processing. Incomplete lineage remains explicitly unmapped.

Native CUDA/Tensor objects are not held by weak-reference registries: the
first observer prototype exited uncleanly while clearing native-object weak
references on Windows. Constructor-based CUDA Event identities and primitive
Tensor metadata eliminated the exit failure. No prototype was submitted to
the cluster. The final observer completes cleanly and matches all five outputs
bit-for-bit against the uninstrumented logical-rank request; Native/accepted
audio contracts and zero shared-budget reservations after cleanup also pass.
This local test qualifies observer semantics, not T4 performance.

## OS scheduling, GIL and resource contention

An external supervisor reads `/proc/<pid>/task/<tid>/schedstat`, task state,
CPU ticks, wait channel, names, affinity, effective CPU quota and cgroup
throttling. It samples only the request process/ranks. Scheduling counters
cover bounded sample intervals; observed R/S/D states are samples. Runnable
time is usable only if Linux scheduler accounting is enabled. When available,
running + runnable + sleeping/blocked residual are kept separate. Cgroup
counters are shared-container evidence, not an attribution to one thread.

`perf` scheduler tracepoints are probed before the request. If denied or
absent, that limitation is saved; no global scheduler settings are changed.
Target-process inheritance can miss externally generated wakeups/switch-in
events, so trace coverage must be checked before claiming exact scheduling.

External `py-spy dump --json --native --nonblocking` snapshots record
`owns_gil`, OS TIDs, and Python/native frames once per second, without locals.
This exposes sampled GIL holders and resolved `take_gil` waits. It does not
give exact GIL waiting duration, and absence from samples does not prove no
contention. Nonblocking sampling can return inaccurate/incomplete snapshots.
The process permits tracing only by its supervisor/descendants; tool errors
are recorded. See [py-spy's native StackTrace fields](https://github.com/benfred/py-spy/blob/master/src/stack_trace.rs).

The report will test whether a TE consumer's exact producer is runnable while
DiT preparation uses CPU resources, or blocked on an observed native/GIL/I/O
dependency. Merely overlapping the intervals is insufficient. Without
adequate scheduler/native-stack evidence, speculative interference remains a
hypothesis. GILs are process-local; one rank's GIL does not directly lock the
other rank.

## Physical probe and delivery status

[MiniMax H3 CPU Critical Path Audit](https://www.kaggle.com/code/chinokiki9/minimax-h3-cpu-critical-path-audit)
version 1 is RUNNING/setup on requested dual T4. It executes only the existing
Offline+P1+P2 configuration: a matched observer-disabled timing pass and a
CPU/Kineto/scheduler/stack diagnostic pass, two consecutive cold/warm requests
each, no separate warmup or extra evaluations. Both outputs are validated
against the saved physical Native reference. Cold includes request startup;
the second request exposes cross-request TE lookahead. This is an attribution
probe, not an optimization or a broad benchmark matrix.

Weights and the original Native tensors are downloaded/restored on Kaggle.
No local weight upload, repeat conversion, or repeat Native generation is
performed. The inference source closure is frozen to the previous physical
run; unrelated new helper files are excluded and preserved on disk.
`semantic_results/cpu_qualification/result.json` binds the new observer
sources and passing local checks. Two observer fixtures plus the existing
analyzer fixtures validate identity/publication and interval accounting.

Minute-level parent heartbeats report actual observed process CPU deltas and
stack sample counts, in addition to stage elapsed time. Sampling/profiler
perturbation is measured against the matched pass; it is not subtracted
arbitrarily from individual stalls. Expected total duration is roughly one
hour, based on previous setup and request timings, with bounded stage limits.

The final ranking will separate: observed foreground blocking, producer CPU
execution, runnable delay, resolved GIL/native waits, GPU dependencies and
hidden asynchronous preparation. Event/Future readiness bounds, unmapped
lineage, denied tools and unresolved single-blocker attribution will remain
visible. New scheduler/GIL findings are pending the physical probe.
