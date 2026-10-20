# TE idle residency: validation status

Physical pilot version 2 is COMPLETE: all 14 full requests pass Native contracts,
source/workload checks and managed-memory cleanup. Its standalone timing is
promising, but its idle cancellation boundary differs from ComfyUI's prompt
lifecycle. No production default is promoted. P1/idle stay opt-in; P0 is retained.

## Version 2 measured results and lifecycle limit

Same physical dual T4, 50 TE/DiT layers, 512×512/124 frames, four actual NFEs.
Conditioning, latents and video are bit exact; waveform passes the existing CP
rounding contract. All requests use aligned video/channel audio without fallback.
No observed OOM/kill, idle errors or outstanding used/pending/preparing balances.
Peak cgroup memory still reaches 30 GiB; allocated GPU peaks are ~11.48/11.50 GiB.

Warm medians below use two observations/configuration. Fifteen-second comparisons
reverse order; zero-gap observations are the two warm requests of one continuous
three-request pass. This is descriptive, not a significance or stable p95 claim.

| Gap | Policy | Reference+TE | Output-ready E2E |
|---|---|---:|---:|
| 15 s | P0 | 98.685 s | 348.464 s |
| 15 s | P0+Idle | 81.108 s | 331.742 s |
| 0 s | P0 | 98.149 s | 352.770 s |
| 0 s | P0+Idle enabled | 97.455 s | 350.523 s |

At 15 s this is an observed 17.81% TE / 4.80% E2E reduction in the standalone
runner. Actual handoff gaps are ~15.003 s. At that boundary ranks have submitted
9/9 copies in one pass and 9/10 in the other; final idle submissions are 11/11
and 12/12. Warm TE consumes those same cached entries (11 or 12 GPU hits/rank),
moving 2.684–2.928 GB/rank of H2D outside the ordinary TE block-loading path.
Window totals include idle copies: this moves transfers rather than eliminating
request-cycle traffic. Submission counters do not directly certify ready state.

**Lifecycle equivalence is PARTIAL.** The standalone runner waits the gap, reads
idle statistics and starts the next request without invoking `pause_idle`.
It permits more idle copies during Native reference encoding until the TE RPC
stops speculation. Last submissions occur 2.07/2.78 s and 3.84/3.23 s after
the next request starts. In contrast, `PromptBridge.start` pauses idle at prompt
start. Therefore the observed 4.80% is not established for the integrated ComfyUI
policy, nor attributable solely to a 15 s idle window. A matching-boundary A/B
must separate strict idle-only preparation from optional reference-stage overlap.

Zero-gap runs submit no idle copies and show no slowdown in these samples;
their ~0.64% E2E difference is too small and under-replicated to establish a gain.
The 5/60 s gaps, smaller workload, full policy matrix and physical model switches
remain untested. Full kernel-level GPU idle reduction is not measured by this
counters-only timing pilot.

Evidence: `kaggle_idle_residency_v2_output/idle_residency_results/`, per-case
`result.json`, source binding, correctness files and `idle_validated_summary.json`.
The latter explicitly scopes its PASS to correctness/ownership, excludes default
promotion and records per-rank post-submission idle work. Reproduce:

```powershell
python analyze_idle_pilot.py kaggle_idle_residency_v2_output/idle_residency_results
```

## Completed physical diagnosis

[TE block critical-path diagnostic](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-block-critical-path),
version 1, COMPLETE: two actual full requests, 50 TE/DiT layers, 512×512,
124 frames, four real NFEs. Both pass exact conditioning/latent/video validation
and the accepted CP audio contract, without VAE fallback. No managed leases
remain after close.

Warm reference+TE is 103.452 s. Required block 2–49 Future dependencies intersect
61.017 s of inactivity on both physical GPUs, counted as a union. Per-rank
workers spend about 80 s off CPU versus only 0.30–0.34 s runnable delay;
~87% of samples are D-state, predominantly `folio_wait_bit_common`. Producer
storage accounting is 11.649/11.656 GB against 11.711 GB logical payload/rank.
These establish file-read readiness as the dominant producer stall. They do
not establish that an idle cache removes 61 s. See `TE_CAUSAL_PROFILE.md`.

## Implementation and local safety

| Gate | Result | Scope |
|---|---|---|
| Same-storage reuse and copy-event dependency | PASS | Actual CUDA, one 5060 Ti |
| Cancel blocked producer without waiting for full plan | PASS | Producer remains charged until completion |
| Insufficient idle cap → normal staging | PASS | Actual CUDA |
| Cancelled idle plan restart; reject unfinished/active invocation | PASS | Actual CUDA |
| Native checkpoint replacement / offline manifest mutation | PASS | Stale identity rejected before reuse |
| Completed/failed/queued/rapid/model-switch callbacks | PASS | ComfyUI queue/history fixtures |
| Trained INT8 TE reuse, two idle/forward cycles | PASS | Two trained blocks, S512, logical ranks, Native comparison |
| P0+Idle compact full request | PASS | All outputs equal uninstrumented; Native contracts pass |
| P0+P1+Idle compact full request | PASS | Same scope; P1 independently enabled only for this gate |
| Forced worker termination and poisoned budget lock | PASS | All workers reaped; used/pending/preparing balances zero |

These local runs are semantics/lifetime evidence, not dual-T4 timing. Repeated
full requests on the 32 GB Windows host reached the existing admission guard;
it was not relaxed. Compact full-request video uses Native admission fallback.
The trained TE repeat isolates actual cached INT8 storage reuse without claiming
sustained full-pipeline capacity.

Source-bound receipts: `semantic_results/residency_qualification/{p0_idle,p0_p1_idle}/result.json`,
`semantic_results/idle_trained_te_reuse.json`, `semantic_results/abort_budget_ownership.json`.

## Historical pilot v1: failed activation

[Idle Residency Pilot](https://www.kaggle.com/code/chinokiki9/minimax-h3-idle-residency-pilot),
version 1. Frozen source: `kaggle_idle_residency/source_binding_v1.json`.
Only source was uploaded; Native checkpoints, public offline shards and preserved
Native outputs are restored on Kaggle.

Version 1 completed two baseline and two idle-labelled full requests, all Native
correct. Both idle windows failed their first preparation because the benchmark
wrapper dropped cancellation keyword arguments. Idle loads/H2D/hits are zero;
no observed timing difference is attributed to prefetch. The gate stopped the
matrix. The wrapper now forwards arguments, a real-CUDA disabled-observer
regression passes, and compact full/trained reuse gates pass again. The repaired
source is frozen in `source_binding.json`. The first launch attempt was capacity-
rejected; the continuation subsequently submitted version 2 when a slot freed.
Version 2 is now complete; measured results and limits are above.

The pilot uses the same full workload, operators, capacities, lightweight
instrumentation and correctness contracts. It runs P0 vs P0+Idle at a 15 s gap,
then reverses their order (two warm observations per policy), followed by
three-request zero-gap runs for each policy. Every accepted performance sample
must use aligned video and channel-parallel audio without fallback and finish
with zero outstanding leases. Stop on correctness, admission or lifetime failure.

Idle load time, bytes, completed blocks and per-rank VRAM are reported separately
from output-ready request latency. Both idle and foreground I/O are retained;
moving a read outside the request is not eliminating it. Profiler/OS observer
traces are disabled in timing. Small samples do not establish stable P90 or a
statistically significant speedup.

| Comparison | TE/E2E | Correctness / memory | Recommendation |
|---|---|---|---|
| P0 vs P0+Idle, 15 s, repeated/reversed | V2 standalone TE −17.81%, E2E −4.80% | PASS, lifecycle equivalence PARTIAL | Keep idle opt-in; match prompt boundary |
| P0 vs P0+Idle, continuous zero-gap | V2 E2E −0.64%, small sample | PASS for six requests | No general non-regression claim |
| Baseline / P0 / P0+P1 / P0+Idle / P0+P1+Idle, 0/5/15/60 s | Not launched; pilot gate first | Pending | No default promotion |
| Physical model switch / interruption | Pending | Local fixtures only so far | No universal race-safety claim |

The larger matrix is implemented as `benchmarks/idle_residency_matrix.py --phase matrix`
but is not automatically launched before reviewing the pilot. It needs allocation
planning: eighty full requests exceed a short Kaggle session. Run bounded groups
with saved source/workload identities rather than silently truncating the matrix.
Post-download acceptance also inspects recovered idle-window errors and cgroup
OOM events; a caught allocation failure is not a clean no-OOM result. This
additional analysis does not alter the submitted production source.

## Current recommendation

Keep P0 and conservative near-use scheduling. Measure bounded idle GPU residency
as the first ablation, because it can move known critical-path preparation into
real user think time. Preserve useful host/file pages as the second opportunity;
evaluate total I/O and refaults, not cache-hit totals alone. Extra CPU workers or
larger speculative depth are unsupported by the new scheduler/folio evidence.
