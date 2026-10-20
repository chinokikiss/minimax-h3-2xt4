# Physical dual-T4 residency results — version 1

[Kaggle](https://www.kaggle.com/code/chinokiki9/minimax-h3-residency-p0-p1):
**COMPLETE**, no failed cases. All 21 full requests passed Native checks:
15 timing and six separate diagnostic requests. Workload: 50 TE/DiT layers,
512×512/124 frames, four real NFEs, identical inputs/references. Conditioning,
latents and video are bit exact; audio uses the accepted CP rounding contract.
All timing requests use the same aligned-video/channel-audio paths, without fallback.

## Timing

Warm means the second short request or the mean of continuous requests 2–3.
Profiling is excluded. Continuous order reverses initial order; managed caches
start fresh, OS/provider caches are uncontrolled. One allocation and small
samples do not establish a general or statistically significant speedup.

| Pass | Policy | Warm reference+TE | Warm E2E | E2E reduction |
|---|---|---:|---:|---:|
| Short | Baseline | 99.144 s | 357.741 s | — |
| Short | P0 | 95.528 s | 345.319 s | 3.47% |
| Short | P0+P1 | 98.338 s | 350.674 s | 1.98% |
| Continuous | Baseline | 95.411 s | 355.480 s | — |
| Continuous | P0 | 97.180 s | 350.759 s | 1.33% |
| Continuous | P0+P1 | 94.608 s | 344.897 s | 2.98% |

P0 TE improves initially but regresses 1.85% in confirmation. P0+P1 TE improves
only 0.81%/0.84%. Its continuous 10.583 s E2E reduction comprises 0.803 s
reference+TE, 5.203 s sampler, 2.514 s video and 2.063 s audio. Most of the
observed gain is outside TE; its long read stalls have not been eliminated.

## Causal evidence and limits

**P0 lifecycle: PASS.** During warm diagnostic TE, blocks 0–1 have no preparation
or read calls under P0/P0+P1; baseline rereads them. Rank 0/block 0 staging Future
wait: baseline request-relative 12.510–15.305 s (2.794 s), versus P0
9.299–9.593 s (0.294 s). Block 1 falls from 0.796 s to zero. Evidence:
`diagnostic/{policy}/kineto/te_critical_path.json`, request 1/blocks 0–1.
During continuous timing, baseline evicts 15/13 entries before CPU borrowing
on ranks 0/1; P0/P0+P1 record **0/0**, measured before shutdown. This counts
cache outcomes, not GPU stalls.

**P1 pressure mitigation: PARTIAL.** The dominant observed physical device is
**sda in this allocation**, rather than the earlier investigation's sdb. Warm
full-request reads average 50.019 GB baseline, 46.655 GB P0 and 44.266 GB P0+P1
in confirmation (P0+P1: 11.50% less). File refaults fall from 22.128 million
pages to 20.267 million (8.41% less). Initial comparisons also show lower reads.
These are shared whole-device endpoint counters, not exclusive model I/O.
Do not sum partitions or add loop/dm layers to the same traffic.

The cgroup still reaches roughly **30 GiB**. Continuous P0+P1 records 653,408
`memory.events.max` increments across the batch versus baseline 745,995.
Millions of refaults remain. The cache floor does not protect specific pages or
eliminate reclaim; it is a mitigation, not a completed pressure solution.

**Memory/lifetime safety: PASS for this workload.** All six timing batches have
zero OOM/OOM-kill deltas and finish with zero reserved bytes, preparation slots
and pending staging bytes. No observed deadlock or fallback. This does not
establish race freedom or safe capacity for every input.

**DiT reuse: PASS, without an improvement claim.** All four NFEs share the
existing GPU window. Warm P0+P1 windows report 61/56 loads and about 152/152 GPU
hits on ranks 0/1, including preprocessing. H2D is 14.200/10.995 GB per request
versus baseline 14.001/10.799 GB. H2D is slightly higher; the gain cannot be
attributed to fewer GPU weight copies.

**Critical-path attribution: PARTIAL.** Warm diagnostic foreground Future wait
unions are 83.526/81.412 s baseline, 86.792/86.962 s P0 and 84.872/84.379 s
P0+P1 per rank. Do not add ranks or treat inclusive waits as removable latency.
P0+P1 buffered reads still take 83.8/83.6 s wall with about 9.1 s thread CPU.
There is no confirmed reduction of aggregate TE Future exposure. Kineto covers
warm TE/reference; exact physical GPU idle requires merging parent/worker
kernel intervals. Idle outside this capture remains unmeasured.

## Recommendation and evidence

Keep both policies available, with defaults unchanged. P0 fixes the demonstrated
prefix lifecycle defect. P0+P1 has the best continuous E2E time and lower backing
reads, but small TE gains and persistent pressure need further validation.
Prioritize remaining block-2–49 read waits, then tune pressure/staging bounds
with controlled ablations. Do not increase prefix depth solely because decode
provides a long interval.

Compact evidence: `kaggle_residency_p01_output/residency_results/residency_summary.json`.
Reproduce: `python analyze_residency_p01.py kaggle_residency_p01_output/residency_results`.
Per-pass results retain exact boundaries, device/cgroup counters, cache totals,
stage timings and lease cleanup. Compact block profiles and trace receipts are
downloaded. Complete parent/rank Kineto traces remain compressed in Kaggle
outputs with SHA256 receipts; no output tensors were downloaded.

Version 1's summary duplicates memory samples already present in runner results
(~48 MB aggregate). The future summary writer now emits aggregate RAM metrics
only. This output-only cleanup changes neither inference nor completed results;
version 1's frozen source binding is preserved.
The completed duplicated summary was archived locally with a SHA256 round trip
(48.30 → 4.06 MB, freeing 44.25 MB). Re-analysis from the archive produces the
identical compact summary; the compression receipt permits restoration.
