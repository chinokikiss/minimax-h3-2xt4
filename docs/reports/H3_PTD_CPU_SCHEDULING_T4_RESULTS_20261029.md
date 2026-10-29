# PTD CPU scheduling: physical two-T4 results

> Historical experiment record: the FP32 sidecar and P2 Split implementation and switches have been removed. P1 remains configurable; see [current defaults and VRAM analysis](H3_DIT_VRAM_FUSION_ROI_20261029.md). Reproduce removed cases with their recorded immutable source snapshot. See [current cleanup](H3_PTD_RUNTIME_CLEANUP_20261029.md).

All six cases passed on two distinct Tesla T4 GPUs (SM75), preserving byte-identical video/audio outputs and final FP32 carried state. **Consumption-aware reclamation lowers managed host peaks. Refiner prefetch improves average latency but increases per-hop variability. The current FP32 sidecar is a negative optimization.** No candidate defaults are changed by this report.

## Experiment and qualification

[H3 PTD CPU Schedule T4 20261029](https://www.kaggle.com/code/pamcks/h3-ptd-cpu-schedule-t4-20261029), version 1. Source commit `c9bad18ca3ed9a51818b4e321b9339c2fb7e7548`, capsule `835c47888ca5210eb99d9e979632d7845c49c5e826a07384077b175fc537b42d`. Actual runtime: Torch 2.11.0+cu128, CUDA 12.8, Kitchen 0.2.37 and TileLang 0.1.13. Frozen Native ComfyUI and core pins are preserved.

The fixture uses two real trained blocks, both refiners, full output heads and eight hops, with identical seeded 24-token inputs. One bank is baked once and reused by six fresh-service cases. There is no TE/VAE execution, full H3 generation or local LoRA download. All cases use a 6 GiB GPU weight cap, 4 GiB per-rank CPU cache, 4 GiB pin cap, 20 GiB shared host limit and 4 GiB GPU activation reserve.

- All **32 PTD/queue/scheduling contracts** passed on the GPU host.
- All **48 denoiser calls** completed. Each hop's video/audio hashes and final FP32 state hashes match the baseline across all six cases. All values are finite.
- All **128 independently Native-verified merge targets** and **64 runtime-layout static checks** passed. These establish folding/handoff equivalence, not PTD generation quality.
- Every case retains one shell/window for eight hops. Every rank has all 16 completed RPC spans. Host ownership is fully drained after teardown. Every CPU group was prepared exactly once; the duplicate-transfer speculation fix is exercised without duplicate preparation.
- The supervised experiment exited normally in **1,511.14 s**. Baking with sidecars took **545.08 s**. A bake-without-sidecars control was not run on this instance, so that complete bake time is not an isolated sidecar overhead.

## Latency and memory

The execution metric sums parent hop-selection and denoiser RPC durations. It excludes downloads, baking, AOT preparation, service construction/configuration, diagnostic queries and teardown. It is not full-pipeline or sampler wall-clock latency.

| Method | Repetitions | Eight-hop RPC mean, s | Hops 1–7 RPC mean, s | Shared host-budget peak, GiB | Rank-0 peak CUDA allocation, GiB |
|---|---:|---:|---:|---:|---:|
| Existing queue | 2 | 111.97 | 94.12 | 11.53–11.90 | 3.282 |
| P0 future-use cache | 1 | 113.59 | 98.13 | **9.50** | 3.282 |
| P0 + P1 refiner prefetch | 2 | **94.62** | **78.28** | 11.91–12.10 | 3.464 |
| P0 + P1 + FP32 sidecar | 1 | 182.80 | 165.91 | 12.46 | 3.282 |

Case order and individual totals were queue 117.80 s, P0 113.59 s, P0+P1 95.36 s, sidecar 182.80 s, P0+P1 repeat 93.89 s, queue repeat 106.15 s. First AOT preparation took 30.84 s; subsequent AOT preparation was 1.46–1.85 s and is excluded above. Page-cache, CPU frequency/load and ordering effects remain; there is only one P0-only and one sidecar case.

Shared host-budget peaks are admission reservations, not measured peak RSS. CUDA peaks are PyTorch allocations, excluding context/external allocators. Worker RSS values are snapshots. The extra 0.181 GiB CUDA peak with prefetch stays within the unchanged bound, but it is a real overlap tradeoff.

### P0: demonstrated reclamation benefit, no demonstrated latency gain

P0 reduces the shared managed peak **18.8% against the mean baseline peak**, from approximately 11.71 to 9.50 GiB. Rank-0 CPU cache peak falls from 3.891 to 2.974 GiB; rank-1 cache peak falls from 2.946 to 1.102 GiB. At the last active-sample snapshot, the two rank caches retain about 0.554 GiB combined rather than 4.046 GiB for the baseline. All cases drain to zero after teardown.

Each P0 rank records **24 consumed-entry releases**. Transferred pageable refiner sources are separately owned by the GPU window until DMA readiness, so they are not included in that cache-release count. The experiment records **zero competitive farthest-future evictions**. It therefore demonstrates consumed-entry reclamation, not hardware superiority of a Belady eviction policy over LRU. Repeated-use and farthest-future safety are covered by CPU contracts.

P0's single measured total is about 1.4% above the baseline mean; excluding initial hop 0 it is 4.3% above. Given repetition/order limits, this is not sufficient evidence of an intrinsic slowdown either. Retain it for memory control; do not claim a latency win. Cgroup snapshots are not a controlled reclaim-latency ablation, so lower managed peaks do not prove a measured reduction in reclaim time.

### P1: average gain with uneven hop latency

P0+P1 reduces eight-hop RPC mean **15.5%**, or **16.8%** over hops 1–7. Rank-0 window host waits fall to 67.22–70.72 s from baseline 83.22–86.89 s, but they still dominate the execution. GPU H2D copy-event intervals remain around 6.4–6.5 s; transfer work has not disappeared.

This is not smooth or stall-free execution: later-hop denoiser times include 23.39 s followed by 3.64 s, and 21.99 s followed by 1.97 s. Maximum later-hop denoiser time increases from **15.84 s baseline to 23.39 s with prefetch**. The prefetched whole-refiner group can arrive early for one hop and late for another. Exact per-stall attribution still needs aligned CPU/GPU scheduling traces; these timings alone do not isolate disk latency, scheduling slots or a particular kernel.

Managed host peak returns to approximately 12 GiB with concurrent preparation. Rank-0 scheduler-yield counts are 493 and 526, with zero HostBudget admission denials. That yield counter covers shared preparation slots as well as memory conditions; it must not be interpreted as a count of RAM-reclaim stalls. Keep prefetch optional when consistent per-hop latency or lower memory is more important than the measured average.

### Sidecar: extra bytes and validation dominate the saved cast/fold

The FP32 sidecar takes **182.80 s**, **93.2% longer** than P0+P1's mean. It adds **22.97 GiB** of persisted data to the 17.36 GiB reduced bank, for 40.33 GiB total. Correctness passes; arithmetic degradation is not the failure mechanism.

In rank 0, the seven sidecar refiner preparations contain:

| CPU stage | Count | Sum wall, s | Sum thread CPU, s |
|---|---:|---:|---:|
| Refiner preparation total | 7 | 150.87 | 90.00 |
| Sidecar tensor read + validation | 56 | 150.82 | 89.96 |
| Sidecar tensor checksum (nested) | 56 | 66.15 | 65.87 |
| Original-overlay validation, all eight hops | 8 | 47.96 | 36.05 |

These spans nest and can overlap across CPU lanes. They must not be added as independent critical-path times. The tensor read/validation span excludes its nested checksum only conceptually; the residual includes I/O, allocation, signature/shape checks and other overhead, not a measured pure disk time.

Original output heads still trigger verification of the full original auxiliary overlay. Rank 1 also validates all eight large original overlays despite using only small output heads: 43.45–51.91 s of wall time across cases, mostly thread CPU. In the sidecar case this remains 50.95 s wall / 40.26 s thread CPU. Thus the sidecar adds a large FP32 read/checksum stream while retaining original-overlay verification. Removing casts/folds did not remove the dominant work.

## Decisions and next targets

- **Keep P0 as the memory-oriented option.** It demonstrates safe consumption-based release and lower managed peaks, with no accuracy change.
- **Keep P0+P1 optional for average latency.** It wins this fixture, but the increased maximum hop time argues against calling it a general smoothness improvement.
- **Leave the current FP32 sidecar disabled.** Its storage and validation costs exceed its benefit in this implementation. Preserve the experiment for reproducibility.
- **Next, split compact refiner weights and small head banks into separately hash-bound artifacts.** Rank 1 should validate/read its head bank rather than the unrelated large refiner payload. Keep immutable receipt/tensor proofs and mutation checks; do not disable integrity verification.
- **Prefer compact runtime preparation and verified chunks over another full FP32 copy.** Investigate chunked original-weight validation/read/cast/fold and scheduling smaller refiner tasks. That targets CPU head-of-line work and permits measuring whether lower maximum hop time can accompany average throughput gains. A new controlled ablation is required before claiming a speedup.

Defaults and dependency versions remain unchanged. No new GPU experiment was submitted while collecting these results. This two-layer 24-token fixture does not establish full-model throughput, long-sequence performance or generation quality. No Nsight trace was produced; CPU nested spans and CUDA event intervals are the available evidence.

[Final raw evidence and independently checked summary](../data/h3_ptd_cpu_schedule_20261029/t4_v1_final/README.md) include a lossless compressed original result, complete worker reports, AOT receipts and supervision. The user-authorized third download succeeded after two earlier SSL failures. No status/list/live-log polling occurred.
