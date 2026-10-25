# Explicit worker trace checkpoints — 2026-10-24

This is the user-authorized replacement for the
[version-2 periodic-flush failure](BUNDLE_SCALAR_RETRY_20261024.md). Version 2's
final outputs were collected and verified before preparing this submission.
The attention experiment itself remains one Scalar Sparse NFE followed by one
matched Kitchen NFE at 51,220 tokens. No full H3 generation is included.

**Current status: both attention runs completed on two physical Tesla T4s.**
Version 3 of
[`pamcks/h3-dit-scalar-single-profile`](https://www.kaggle.com/code/pamcks/h3-dit-scalar-single-profile),
submitted at 2026-10-24 12:42:24 Beijing time, passed both explicit-checkpoint
capture cases and completed exactly one 51,220-token Scalar Sparse NFE and one
matched Kitchen NFE. Both worker GPUs have correlated kernels in all 50 trained
blocks, clean exits and all 12 checkpoint receipts. All 133 final artifacts were
recovered and hashed; a disconnected download preserved 54 complete files and
resumed the remaining 79. No status/list/live-log polling or extra submission
was used. The
[evidence record](../data/bundle_scalar_explicit_flush_20261024.json) separates
execution, capture qualification and the limitations of performance attribution.

The important finding is **CPU offline-LoRA preparation dominates this cold
bundle call**, while Scalar Sparse's attention kernels are slower than Kitchen's
on this workload. The small difference in whole-call duration does not establish
an attention speedup. This run is not a full H3 generation or a steady-state
latency benchmark.

## Physical T4 results

The preserved environment was Torch 2.11.0+cu128 / CUDA 12.8, Kitchen 0.2.37 and
TileLang 0.1.13 on two distinct Tesla T4 / SM75 devices. Nsight Systems was
2025.1.1. Frozen ComfyUI, legacy attention API and core pins were unchanged.

Each backend executed its requested attention 50 times per rank, the fused
SwiGLU quantizer 50 times and scaled-FP16 epilogue 100 times. No attention
fallback was counted. Conditioning and both noise hashes match each other and
the previous long-sequence input hashes. Both returned video/audio denoiser
tensors are finite. This establishes execution, not generation quality.

| Measurement | Scalar Sparse | Kitchen control |
|---|---:|---:|
| DiT calls per rank | 1 | 1 |
| Denoise CUDA command span, rank 0 / 1 (s) | 989.412 / 989.408 | 999.977 / 999.980 |
| Sampling wrapper including refiner/bookkeeping (s) | 1,010.334 | 1,020.601 |
| Correlated NFE kernels, rank 0 / 1 | 16,750 / 16,720 | 14,800 / 14,770 |
| Trained blocks with correlated kernels, each rank | 50 / 50 | 50 / 50 |
| Attention kernel duration sum, GPU 0 / 1 (s) | 113.000 / 118.109 | 48.085 / 50.796 |
| GPU kernel-active union within NFE, GPU 0 / 1 (s) | 239.919 / 246.899 | 172.447 / 186.892 |
| Kernel timeline busy fraction, GPU 0 / 1 | 24.25% / 24.96% | 17.25% / 18.69% |
| Consumer-wait interval sum, rank 0 / 1 (s) | 929.332 / 929.101 | 957.184 / 953.632 |
| Peak allocated, rank 0 / 1 (GiB) | 10.175 / 10.510 | 10.175 / 10.510 |
| Peak reserved, rank 0 / 1 (GiB) | 13.877 / 14.283 | 13.877 / 14.283 |

The attention row includes Sparse quantization/centroids, routing and execution;
Kitchen includes the kernels within its complete attention API. QK Norm/RoPE
is excluded from both. Kernel-active union measures timeline coverage, not SM
utilization or occupancy. Kernel sums, CPU wait intervals and command spans
overlap and must not be added into an exclusive latency decomposition.

Real-input preparation took 222.023 s outside either NFE. Base bundle delivery
took 591.449 s. The parent one-block warmup receipts report 56.230 s for Scalar
and 16.804 s for Kitchen; model loading, compilation and warmup are outside the
denoise command spans. The complete model/input/two-case batch took 3,281.882 s.
Weights are still CPU-baked on demand during each NFE; that work is deliberately
identified rather than presented as kernel-only inference time.

## What limits this run

Both ranks prepared all 50 DiT blocks. Rank 0 merged 216 matrices, including
auxiliary/refiner work, and rank 1 merged 200, with zero persistent
baked-checkpoint bytes. Preparation intervals have medians of about
19.9–20.2 s per block. The dominant retained owner-thread stack is
`weight_residency.use → Future.result → threading.wait`: 93.9–95.6% of retained
main-thread samples. Background preparation stacks repeatedly show
`offline_lora.merged_weight`, `lora.calculate_weight`, and ConvRot weight
dequantization/requantization through `int8_utils._rotate_weight`. This locates
the current bottleneck in CPU bundle preparation; it does not establish an SSD
bandwidth bottleneck. Each rank reconstructs and merges full matrices before
selecting its shard, so duplicated work remains an optimization target.

Correlated H2D transfers take about 0.88–0.94 s per GPU, far less than the
consumer-wait intervals. NCCL kernels sum to 33.3–42.9 s per GPU across these
cases, including 100 FP32 and 100 FP16 AllReduce kernels plus one broadcast.
These durations include rank skew/waiting; they are not pure PCIe bandwidth
measurements. `cudaFree` API sums of 5.1–11.4 s are a secondary allocation/sync
target, not the main cause of the ~1,000-second call.

Sparse's execution kernel alone sums to 109.292 / 114.260 s across 50 layers;
quantization/centroids add 3.127 / 3.239 s and routing adds 0.581 / 0.610 s.
The total is about 2.35× / 2.33× Kitchen's attention GPU duration. Routing is a
small part of this regression; changing only CPU routing cannot close it.
This run does not capture kept/dropped mass or hardware stalls, so it does not
establish the instruction, cache or occupancy mechanism within the sparse
execution kernel. No precision or semantic-quality conclusion is inferred.

The next useful changes are bounded, layer-at-a-time LoRA preparation that
avoids repeating full-width work on both ranks, followed by an isolated trained
attention replay for the Scalar execution kernel. Retain the current numerical
contract and memory bound; do not materialize a full baked checkpoint. A warm
repeat would then distinguish cold preparation from reusable inference. These
are follow-up directions, not improvements claimed by this collection.

## Norm/Residual anomaly and observer checks

All four live worker probes reached `final: true`, with 100 Norm and 100 Residual
calls per rank per backend. Norm median owner-thread CPU time is 1.26–1.54 ms;
Residual is 1.85–2.05 ms. The largest Norm owner-thread CPU sample is 8.31 ms.
Some Norm wall outliers reach 127–162 ms, without comparable owner-thread CPU
time. The earlier approximately 0.5-second-per-call CPU anomaly is **not
reproduced** here; its historical cause is not proven by this changed loading
path. Live traces use direct thread timers and stack sampling, with no
deterministic profiler.

Both ranks completed all six observer replay modes, three repetitions each,
with byte-exact outputs: 72 replay pairs across both backends. Median host time
for the combined resident Norm/Residual pair is 2.22–2.59 ms with full observer
instrumentation, 2.47–2.89 ms with sampling, and 4.80–5.55 ms with the additional
thread-local deterministic profiler. The off modes are 2.22–3.13 ms. These small,
noisy samples do not support subtracting a fixed observer overhead from a live
NFE. All 24 returned resident `.pstats` files pass timing-validity checks.

## Capture validity and remaining profiler limits

Both minimal capture cases retain eight kernels per rank, four in each resumed
chunk, with valid stop/restart receipts. Full NFEs have all expected per-five-
block checkpoints, denoiser-exit and worker-close records; all four worker exit
codes are zero. SQLite uncompressed hashes match the producer receipts. No
dropped-record/overflow diagnostic was found. Approximately 0.06 s per GPU of
matched NFE kernels lacks a narrower semantic phase label and remains explicitly
unattributed.

Nsight warns that driver 13.0 is newer than its supported 12.8 tracing libraries.
This warning is retained for both actual workers, even though the concrete
restart, per-layer and kernel correlation checks pass. CPU scheduling/context-
switch sampling was disabled; Python stacks cannot establish native CPU
instruction costs. Compilation/helper subprocesses produce no-CUDA/no-NVTX
warnings, and are distinguished from the measured workers. Nsight Compute
hardware counters remain blocked by the previously observed
`ERR_NVGPUCTRPERM`; no new counter measurement is claimed. There is no measured
memory-stall, achieved-occupancy or Tensor Core utilization result.

CPU-bundle baking is not byte-identical to the earlier GPU-baked checkpoint.
Therefore the old 161-second Kitchen result is not a matched baseline for this
experiment. Only the current Scalar/Kitchen pair shares the same preparation
contract and inputs. Checkpoint synchronizations cost 32.6–50.4 s per rank in
wall intervals, including waits for preceding GPU work; they cannot be
subtracted to claim uninstrumented steady-state latency.

## The change

`benchmarks/worker_cuda_checkpoint.py` implements one worker-owned checkpoint:

1. Synchronize that worker's GPU, completing preceding work.
2. Call `torch.cuda.profiler.stop()` with Nsight's
   `--flush-on-cudaprofilerstop=true` enabled.
3. Restart CUDA profiling with `torch.cuda.profiler.start()` when more measured
   work remains.
4. Atomically save a per-rank receipt with label, stop/restart status, timestamps,
   wall duration and owner-thread CPU duration.

The periodic-save option is removed from the new profiling launch. There is no
assumption that it protects a force-killed process. CPU-only preparation threads
cannot invoke a CUDA checkpoint; an owner-thread assertion rejects that path.
Failed checkpoint calls write an explicit failure receipt and propagate failure.

During the DiT call, checkpoints occur after blocks 4, 9, …, 49 (zero-based).
There is another checkpoint at denoiser exit, and a final stop without restart
before orderly worker closure. This preserves completed groups before a later
timeout can force termination. A partial failed trace still cannot be treated
as a complete 50-block performance profile.

The model, weights, LoRA preparation, tensor flow, kernel dispatch, residual
stream and numerical operations are unchanged. These are benchmark changes;
production service cancellation/termination behavior remains unchanged.

## Restart must be demonstrated, not assumed

The new minimal gate uses the **same checkpoint helper** as the DiT worker. Each
rank runs two chunks of four additions. It flushes and restarts between chunks.
The final value is checked exactly. The gate then tests two lifecycle endings:

| Case | Final checkpoint | Worker exit |
|---|---|---|
| Checkpoint then forced termination | Flush and restart after chunk 1 | Parent terminates the already acknowledged worker |
| Checkpoint then graceful termination | Flush without restart after chunk 1 | Parent acknowledges orderly exit |

Both cases require eight correlated kernels per rank across two distinct GPUs,
and **at least four kernels inside each chunk's NVTX range**. A trace containing
only the first chunk fails. The outer NVTX scope must also retain all eight
kernels across stop/restart. Both checkpoint receipts per rank must show the
expected stop/restart sequence. The helper's SHA256 is bound into the gate and
checked again by the DiT launcher.

The gate ran before weight downloads. It tested the changed checkpoint protocol
only; it did not rerun the historical four-case matrix or the failed periodic
save experiment. The new stop/restart sequence passed on both physical T4s.

## Timing boundaries and interpretation

Checkpoint work has a separate `H3_FLUSH` NVTX range and durable JSON records.
The block's operator spans end before this range. The full instrumented NFE
wall time nevertheless includes the checkpoints and their file writes.

Synchronization can wait for preceding GPU operations. Therefore checkpoint
wall duration is **not exclusive profiler overhead**, and subtracting it from
NFE wall time would not establish uninstrumented inference latency. Actual
kernel correlations, CPU preparation/wait records and the existing observer
replay must be interpreted with these boundaries. No steady-state/p95 or full
pipeline speedup is claimed by this experiment.

Complete attribution still requires all 50 block scopes on both worker GPUs,
the ten expected block checkpoints plus denoiser-exit/final-stop records, clean
worker exits and the requested backend counters. Otherwise the launcher exports
partial coverage diagnostics and withholds timing summaries. Existing Nsight
Compute counter permissions remain an explicit limitation.

The 1,500-second benchmark command deadline, 15-second CPU evidence checkpoints,
resident wall-clock profiler validation, input sharing, one-attempt backend
claims and Scalar-first failure isolation from version 2 are retained. The
concurrent CPU-profiler test now explicitly yields the GIL while the disk thread
runs. No cProfile is used in the live NFE.

## CPU preflight repair

CPU version 4 ran 70 tests: 69 passed and the strengthened concurrent profiler
test failed. The test deliberately waited while a disk thread executed. Native
Python 3.13.15 still returned invalid cProfile timing statistics with its default
wall clock. The guard correctly rejected them; the validity requirement was not
relaxed. This failed capsule and its receipts are preserved, and it was never
used for a GPU submission.

Resident replay now uses the standard library's `profile.Profile` with
`time.perf_counter`. Its `runcall()` installs `sys.setprofile` on the calling
thread. The test requires valid timing arithmetic and excludes the concurrent
disk thread's functions from the owner profile. This deliberately slower
deterministic profiler is confined to the `profiled_full` replay mode, whose
overhead is measured. Direct owner-thread CPU timers and Python stack sampling
remain the live-NFE instruments. No Python/core-library version is changed.

## Delivery and validation

The corrected capsule contains 327 files and has SHA256
`df0d8fe5c5ebf6c9ed78edd4a42bacca94c87c6aa4e51b92f82d59846f373b1a`.
It is published at immutable Git commit
`f9d8588556ed271e825a6938b27d01aa39436c10`.
Its nine changed/added source files are under `benchmarks/`; frozen ComfyUI,
production patches/node, the exact core wheel lock and base shard proofs are
unchanged. Kitchen remains 0.2.37, TileLang 0.1.13, and Kaggle retains its native
Torch family.

The 70-test isolated suite covers the new checkpoint call order, durable
failure state, owner-thread restriction and rejection of missing resumed-chunk
kernels, alongside the existing runtime/bundle/CPU profiling checks. This is
CPU/mock evidence, not a substitute for the in-job T4 capture gate.

One exclusive submission receipt permits one GPU replacement of the existing
notebook. Each backend has its own exclusive execution claim and no automatic
NFE retry. Source publication preserves the user's current Git checkout and
index. No new weight/LoRA upload or Kaggle Dataset mount is created.

Evidence root: `D:/attention_eval/bundle_scalar_explicit_flush_20261024/`.
There is no status/list/live-log polling and no timed final collector. GPU
outputs were collected on the user's request, with the disconnected transfer
resumed to completion at 2026-10-24 13:48:07 Beijing time. The original transfer
failure receipt is preserved alongside the successful resume. Local analysis
and its assertions are reproducible with `analysis_v1/analyze.py`; it makes no
Kaggle request and uses no GPU. `analysis_v1/verified_results.json` and the two
`*_attribution.json` files retain the full derived evidence.
