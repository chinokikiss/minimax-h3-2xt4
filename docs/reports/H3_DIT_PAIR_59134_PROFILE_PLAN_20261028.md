# Two-call DiT profile at 59,134 tokens

Status: implementation and local CPU qualification; physical T4 results pending.
The VAE checkpoint/bake experiment is a separate job and is not queried or changed.

## Experiment contract

The user confirmed **two DiT denoiser calls**, rather than two complete sampling
trajectories. Execute exactly two calls in one persistent TP2 service and one
`begin_sample` session. Reuse identical inputs, timestep, weights, operator policy,
model and residency caches. Do not warm operators, clear caches or recreate
workers between calls. Native conditioning refinement happens once before the
pair; report its cost separately, along with service creation and session setup.
No TE, Video VAE, Audio VAE or multi-step sampler runs.

Use the same frozen production source as the CPU-qualified 2026-10-28 VAE job.
All production patch bytes and dependencies are preserved. The policy remains
FP16-fix, Kitchen DiT attention, fused-stored FC2 quantization, Scaled FP16 SUM
transport and the promoted INT8 column path. Only benchmark code changes.

| Setting | Value |
|---|---|
| Hardware gate | Two distinct physical SM75 Tesla T4 UUIDs |
| Layers / hidden width | 50 / 5,376 |
| Per-rank attention | 28 heads, head dimension 128 |
| CPU lookahead / initial GPU lookahead | Native defaults: 16 / 2, adaptive enabled |
| Per-rank GPU weight / pinned / host-cache budgets | 6 / 6 / 3 GiB |
| Shared host budget / activation reserve | 16 / 9 GiB |
| Weights | Public TP2 slices, persisted offline 8-step Turbo LoRA bake |
| Denoiser calls / operator warmups | 2 / 0 |

The new instance performs one fresh production bake in scratch, outside every
DiT timing range. Require all 100 shard tensor states, all 208 independently
verified targets and the auxiliary hash to match the earlier physical T4 bake.
The running VAE job's scratch is not shared with another instance. Download,
bake, service creation, refinement and profiling are distinct timed stages.

## Inputs and representativeness

Match the completed 1152×672, 243-frame request's Native packed geometry:

| Segment | Start | Stop | Rows |
|---|---:|---:|---:|
| Text / vision presentation | 0 | 2,392 | 2,392 |
| Reference image 1 | 2,392 | 3,151 | 759 |
| Reference image 2 | 3,151 | 3,892 | 741 |
| Audio | 3,892 | 4,702 | 810 |
| Video | 4,702 | 59,134 | 54,432 |

Video input is `[1,24,72,42,72]`; audio is `[1,32,2,405]`. Reference latent grids
are `[1,24,1,46,66]` and `[1,24,1,38,78]`. Native tokenizer checks yield 6 tokens
for each picture label, 759/741 expanded vision rows, two boundary tokens per
image and 876 prompt tokens. Reconstruct the original presentation tag spans
with the frozen `token_tags_from_embeds_info` helper: vision `[6:767]` and
`[773:1516]`; remaining presentation rows are text. This preserves the Native
modulation segmentation without executing TE.

The earlier generation retained input shapes/hashes and decoder latents, but
not its raw TE conditioning or reference latents. This experiment therefore
uses seeded synthetic conditioning/reference values. It regenerates initial
noise with the original seed; **actual hashes are logged**, because the local
Windows Torch build did not reproduce the prior Linux T4 RNG bytes. Both calls
receive identical values and report unchanged-input/output proofs. This is a
performance fixture, not a real-distribution accuracy or generation replay.

The first call is process-cold after setup, not a cold-filesystem or cold-machine
measurement: download and baking already touch weight files, service bootstrap
can compile shared components, and refinement initializes the Native front end.
Preserve these distinctions when explaining the first-to-second difference.

## Timing and prefetch evidence

Nsight Systems captures CUDA, NVTX and OS-runtime activity in both workers.
First qualify the existing minimal capture/stop/restart/flush/process-exit probe.
If Nsight is missing or that probe fails, stop before model downloads. No tool
or dependency upgrades are attempted to make the job pass.

Instrument CPU/NVTX and CUDA stream spans for modulation, QKV/FC1 projections,
ConvRot/quantization, INT8 GEMM, dequantization epilogues, QK norm/RoPE, attention,
row projections, FC2 fused preparation, transport and residual updates. Record
TileLang compilation and Native runtime construction separately. CPU spans for
checkpoint reads, file hashing, pin/preparation, admission and H2D submission
retain process/thread identity. Per-block acquisition records contain actual
GPU/CPU lookahead, pending groups, resident bytes and host-wait/copy counters.

For each call and GPU, require correlated kernels in all 50 trained blocks.
Compute the union of useful compute kernels, NCCL kernels, copies and memsets.
Report GPU idle intervals and CPU scopes overlapping the longest gaps. Include
background-thread kernels in activity totals while leaving their phase
unattributed if their launch lacks a main-thread correlation.

Kernel sums and nested CPU/CUDA spans are not additive critical-path components.
NCCL kernel presence does not establish useful compute utilization. Timeline
busy fractions do not measure occupancy, Tensor Core utilization or memory
stalls. This two-call comparison observes whether the current lookahead keeps
work available; it does not establish causality against a disabled-lookahead
control or prove that the current depth is optimal.

## Temporary tensor lifetime and VRAM evidence

Enable identical allocator history in both calls, with a 200,000-event bound.
Retain allocation and release source stacks across all layers, plus semantic
phase markers and storage/shape aliases in representative layers 0, 24 and 49.
Do not read tensor frame locals or use a global Python line tracer. This avoids
retaining temporary tensors or disproportionately tracing cold compiler code.

Reconstruct each storage epoch, distinguish alias views, and track
`free_requested` separately from `free_completed`. Report active, allocated,
reserved, inactive-split and NVML device memory, with resident/retired weights
separated from activations and scratch. Identify the largest tensors overlapping
the peak, especially FC1 accumulators/output/quantization and attention output
preparation. Keep both allocation and GPU-reuse times rather than interpreting
a Python `del` as immediate device-memory reuse.

Allocator export occurs after each completed forward; CUPTI flush occurs at
call boundaries and clean worker exit. No mid-block device synchronizations or
snapshot compression are inserted into the compute stream. Record export wall
time separately from worker enqueue time, stream spans and parent RPC latency.
Both calls are instrumented equally; their timings remain diagnostic timings,
not an uninstrumented production latency distribution.

Reject complete-profile status if either call/rank lacks a trained block,
allocator history is truncated, frees cannot be reconciled, outputs are
nonfinite, worker exit is unclean or trace flush evidence is absent. Preserve
partial artifacts on failure and report the blocker explicitly.

## Artifacts and submission

`profile_dit_pair.py` runs the pair; `run_dit_pair_batch.py` owns hardware/tool
qualification, public downloads and bake setup. `analyze_dit_pair.py` produces
per-call phase comparisons, cache/wait deltas, idle-gap evidence and full storage
lifetime files. Retain the raw `.nsys-rep`, compressed SQLite export, allocator
histories, NVML samples, worker events, input hashes and clean-exit receipts.
Detailed timeline/VRAM charts can be rendered locally after collection.

The source distribution is public and immutable; notebooks/results remain
private. No datasets, prior kernel outputs, captured tensors or credentials are
mounted. Submit one CPU-only frozen setup/fixture preflight, then one bounded
2×T4 job after both matching CPU receipts pass. The GPU job allows 600 seconds
for setup and 4,800 seconds for the execution batch, within a 5,400-second outer
limit. Do not poll Kaggle status/list/live-log endpoints. Collect final outputs
once when the user reports completion.
