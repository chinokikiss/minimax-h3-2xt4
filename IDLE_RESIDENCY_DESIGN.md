# Idle-time TE VRAM residency

Status: optional implementation and local trained correctness gates complete;
physical dual-T4 pilot completed 14 Native-correct requests. Standalone timing improves, but it permits prefetch into reference encoding while ComfyUI pauses at prompt start; production performance remains unverified. No default promotion.
P0 remains available. P1 stays experimental and off by default. Idle residency
also defaults off; disabled inference retains its existing lifecycle.

## Ownership and lifecycle

Reuse the existing TE `ResidencyWindow`, physical tensor entries, copy stream,
host cache and shared HostBudget. Do not create a duplicate weight cache.
An idle TE window is handed back to normal TE execution when compatible.
Full TE→DiT→VAE execution may require relinquishing all TE GPU weights; retention
is opportunistic, never a reservation against active DiT/VAE demand. TE-only
workflows can retain already loaded compatible entries.

Use an explicit prompt/request-complete boundary, not `decode_complete` alone:
video completion precedes audio decoding. ComfyUI's prompt lifecycle provider
API at the pinned commit supplies start/end callbacks. Start cancels speculation;
end considers prefetch only after actual execution finishes and no prompt is
queued. The provider does not cache node outputs. Explicit lifecycle methods
support the benchmark and non-server callers.

Idle ticks run between rank RPCs and check pending commands first. Each tick
admits at most one bounded preparation/copy; foreground execution never waits
for an idle plan to finish. A short debounce prevents work during zero-gap
submissions. Pending CPU tasks are cooperatively cancelled; already submitted
DMA keeps its source and destination until its CUDA event completes. No global
GPU fence is introduced into normal prompt start solely to drain speculation.
Physical in-flight I/O/DMA cannot be preempted instantaneously; rapid submission
and switching must measure that residual interference rather than assume zero.

## Budgets and selection

Start with configurable 4 GiB/rank maximum and conservative free-memory margin
for TE startup, reference encoding, activations and other ComfyUI models.
Compute admission independently from each GPU's actual driver/allocator state;
global free VRAM across two cards is not a usable budget. Required active model
loading may evict idle entries. Pinned/pageable staging uses existing accounting.
P1 headroom guards apply only when P1 is explicitly enabled.

Prefer an actually known next request's compatible model. If queued work is
known but not safely resolvable, cancel speculative reuse of the previous TE
and let foreground execution decide. With no known next request, bounded
speculation may reuse the last validated TE. Selection uses expected next-use
order and, where bound to the matching checkpoint/configuration, measured
critical-path costs. It must not rank solely by inclusive Future wait.

Host copies remain when affordable, subject to normal host admission/eviction.
Never evict GPU entries merely to manufacture duplicate RAM copies. No full
model preload or automatic expansion of CPU preparation concurrency.

## Compatibility, cancellation and pressure

Reuse requires Native checkpoint path/version, offline-shard signatures,
trained layer count, quantization/operator mode and rank match. Changed source
or shard metadata invalidates idle entries. The existing unsupported-LoRA/hook
guard remains; modified weights must never silently share the old cache.
CUDA Graph capture is not supported for this dynamic streaming path; reject
idle mutation during graph capture rather than invalidate pointers.

All active leases and queued consumer events remain authoritative. Evicted
entries retire through existing ready/last-use fences. Cancelled futures release
reservations exactly once, including queued tasks that never start. Shutdown
reaps processes and reconciles all managed/pending preparation ledgers.
ComfyUI memory-pressure/unload integration must yield idle ownership before
another model allocation, while preserving a compatible TE handoff during the
proxy's own parent-model reclamation. Model switch/cancel paths are tested.

## Validation sequence

1. Reuse and merge saved physical parent/rank traces; add per-block OS snapshots
   and bounded enqueue timestamps only where evidence is missing.
2. Qualify idle reuse, cancellation, event lifetimes, pressure denial, pointer
   retention and disabled behavior locally. No local timing is T4 evidence.
3. Run physical full-model pilot before a larger matrix. Benchmark Baseline,
   P0, P0+P1, P0+Idle and P0+P1+Idle with 0/5/15/60-second gaps; reverse order
   and repeat useful comparisons. Include sustained zero-gap and switching cases.
4. Separate post-submission latency from background time/I/O and output delivery
   from request-end hooks. Report per-sample timings, median/P90 with sample
   counts, cache/H2D avoidance, pressure, correctness and profiling overhead.
   Two samples cannot establish a stable P90 or statistical speedup.

Compact metrics are the default. Full Kineto is selected diagnostic evidence,
compressed with integrity receipts. Defaults change only after physical safety
and repeated user-visible latency results support promotion.

## Implemented controls and review boundaries

`RuntimeConfig.idle_residency=False`, `idle_weight_bytes=4 GiB`,
`idle_reserve_bytes=4 GiB`, `idle_delay_seconds=0.25` control the feature.
Enabling idle requires P0; P1 remains independently false. The ComfyUI loader
exposes `idle_residency` and `idle_weight_gib` as optional inputs. Unchanged
workflows retain their existing behavior. The bridge is registered only when an
enabled service is created. A standalone runner explicitly calls `request_idle`
after output delivery; it does not pretend to execute the UI server lifecycle.

The rank checks the real command pipe every 50 ms and runs a nonblocking idle
tick only when no command is pending. One CPU staging Future and at most one
new H2D copy are admitted per tick. GPU budgets use local free/allocator memory;
the shared HostBudget owns CPU/pin reservations. A cancelled producer can finish
its current physical read; its reservation remains charged until completion.
Foreground may still encounter that read's resource contention. Physical testing
must quantify it; cooperative cancellation cannot promise instant preemption.

Ready TE entries are handed directly to the ordinary window. Normal TE forward
retains its existing release-after-use policy; idle does not keep the entire
TE window alive through DiT or VAE. Reference vision yields only enough idle
weights for its admitted working set. Pressure release waits for the **copy
stream**, reaps event-owned buffers and permits normal ComfyUI admission; it
does not release tensor storage still owned by in-flight work.

Selection currently follows near next-use order for the matching TE. The measured
early blocks have substantial exposure, but the implementation does not claim an
optimal cost-ranking policy. Known queued work suppresses unknown-next speculation;
prompt start preserves compatible weights and discards them for a model switch.
Native and offline shard signatures are checked before reuse and periodically
during idle. LoRA/custom weight hooks and CUDA Graph capture are not qualified
for dynamic streaming and must not silently reuse stale weights.

Local gates: real CUDA pointer/event reuse, cancellation of a blocked reader,
capacity denial with normal fallback, cancelled-plan restart and active-lease
rejection; ComfyUI queue/history fixtures for completed/failed/rapid/switched
prompts; trained rank-local INT8 TE, two idle/forward cycles on both logical
ranks; compact full Native request with both P0 and P0+P1. Local full-request
video uses the existing admission fallback. Sustained full-pipeline capacity,
physical NCCL contention and real UI concurrency remain physical validation items.
