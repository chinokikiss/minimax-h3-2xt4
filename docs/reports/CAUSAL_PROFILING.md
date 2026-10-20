# Causal profiling of the actual MiniMax-H3 request path

Status: local numerical/instrumentation qualification passed. Physical traces
and conclusions are pending the dual-T4 diagnostic. No inference optimization
is made by this diagnostic.

## Measured execution

`benchmarks/causal_pipeline_audit.py` executes Native Ref2VA reference encoding,
the remote TE, the original ComfyUI CFGGuider/res_multistep sampler, every
actual DiT evaluation, and the current public video/audio VAE adapters. It
uses full 50-layer TE/DiT models, 512×512/124-frame output, four steps and a
fixed gradient reference/prompt/seed. The reference is a declared fixture;
the packed sequence and inference operators come from the real pipeline.
`auto` is the current default VAE policy; any memory-driven Native fallback is
recorded per request. The benchmark forces each request through inference;
it does not benchmark HTTP delivery or ComfyUI graph-output caching.

The former audit profiled an unrelated extra denoising evaluation. This
runner never does that. Warmup is an explicitly named **complete request**,
captured to preserve the producer side of the first handoff, and excluded
from measured latency comparisons. There are no validation, logging, stats,
export, sleep or added stage fences between the measured requests. Actual
handoff gaps are reported. CPU outputs are retained until the batch finishes
for Native comparisons; four video outputs require approximately 1.5 GiB of
additional host storage. That diagnostic storage is identical between cases
and protected by the unchanged available-RAM admission guard.

## Causality and clocks

Each process records request, stage, command and actual NFE IDs. Block keys,
load IDs, collective ordinals, CPU-thread IDs and CUDA-stream IDs connect
the weight queue, checkpoint slices, CPU preparation, pinning, H2D enqueue,
ready-event wait, block compute, eviction and acknowledgement.

CPU spans use the common host monotonic clock. CPU/NVTX ranges include the
same IDs. Kineto traces are aligned by matched CPU ranges; the median fit
residual is reported. CUDA-event intervals use a calibration outside the
request, with an explicit host-alignment uncertainty. They measure stream
elapsed time, not kernel busy time. Background preparation records both its
request of origin and the request active when preparation begins.

CUDA runtime launch correlations link kernels back to block enqueue spans.
A CPU weight future is identified as a **confirmed launch dependency** only
when the following kernel maps to the same block, its launch follows the
future's completion, and the physical GPU is inactive during the intersected
wait. Other overlapping operations are reported as candidates. H2D/NCCL
concurrency alone is not claimed to prove interference.

Physical GPU idle gaps combine parent and worker contexts: a worker's empty
timeline during parent reference encoding/VAE work is not GPU inactivity.
Kernel categories and operator groups use interval unions. Category unions
can overlap; they must not be added together. Dominant kernel names and raw
operator/launch flows remain available to check heuristic classification.
Logical checkpoint/H2D payload counts are distinct from estimated NVML bus
traffic. GPU activity fraction is distinct from SM occupancy.

## Run plan and safeguards

1. Download/hash-check four original checkpoints on Kaggle. The reusable
   repository `cnmds/minimax-h3-tp2-int8-561a6211` is public and ungated as
   of 2026-10-12; anonymous API and shard HEAD access were verified. The
   default loader explicitly disables authentication (`token=False`) and
   obtains fresh object URLs on Kaggle. Shard downloads overlap dependency
   installation, then immutable identities and Native-qualified manifests
   are bound before inference. No HF Secret, local weight upload, or
   repeat conversion is required. Legacy Secret/ticket/regeneration modes
   remain explicit alternatives.
2. Generate fresh full-model Native outputs on that allocation, outside
   profile/timing comparisons.
3. Run a two-request P1+P2 pilot, once with observers/profiler disabled and
   once with causal capture. Check Native outputs, four actual NFEs/request,
   both rank kernel traces, loading records and NCCL records.
4. Only after those checks pass, compare Direct Sync, P1, P1+P2 and
   Offline+P1+P2. Each fresh process runs one warmup plus four consecutive
   measured requests, with matched timing and Kineto passes. Case order and
   timing/profile order are counterbalanced, but this remains a bounded
   diagnostic rather than a statistically powered policy ranking.
5. If Nsight Systems is installed, collect a separate matched real-request
   pass using its supported CLI options. It does not compete with Kineto for
   CUPTI ownership. Record the tool/version/runtime limitation if unavailable.

Unprofiled passes supply performance latency. Diagnostic wall times retain
profiler perturbation; a matched-pass ratio is reported, never an invented
subtraction of overhead from individual stalls. Profiler setup, calibration,
trace export and output comparisons occur outside reported request latency.
The job has an eight-hour hard budget and stops on failed correctness,
missing trace coverage, memory admission failure or stage timeout. It makes
no inference-policy adjustments or retries during diagnosis.

## Evidence delivered

- Per-parent/per-rank `*_kineto.json` traces and `*_causal.json` records.
- `timeline.html`: self-contained, zoomable CPU/GPU lanes with clickable IDs.
- `causal_analysis.json`: per-request/stage/rank kernel unions, memory/bus
  telemetry, cache outcomes, block loading overlap and interval-cited stalls.
- `ROOT_CAUSES.md`: readable stall evidence with confirmed dependencies
  separated from hypotheses.
- `comparison.json`: matched unprofiled latencies and capture perturbation.
- Native output checks, hardware/source bindings and NCCL logs.

Recommendations will be based on physical intervals, not local simulation.
The final audit will report PASS/PARTIAL/FAIL by area and rank the top three
opportunities using measured exposed time as an upper bound, distinguishing
that bound from a verified speedup.

## Local qualification and limits

`semantic_results/causal_qualification/result.json` contains nine passing
checks: spawn transport plus uninstrumented/traced Native and default VAE
requests for synchronous and adaptive policies. All instrumented outputs
match their uninstrumented counterparts bit-for-bit. Conditioning, sampled
latents and video pixels match fresh Native; Channel Parallel audio uses
the already accepted rounding contract and is explicitly not labeled
Native bit-exact.

The 5060 Ti's TF32 convolution differs from T4 semantics. Local Native
references were regenerated with TF32 disabled. Two consecutive requests
triggered the Windows available-RAM guard even without Kineto. No guard was
relaxed; local qualification therefore establishes single-request numerical
and instrumentation correctness, not continuous dual-T4 memory safety.
The physical pilot independently gates that behavior.

Three analyzer fixtures passed: interval unions avoid double counting, nested
interval indexes preserve all overlapping dependencies, parent
GPU work prevents false worker-idle attribution, and a known block future
maps to its correlated launch and exact exposed idle intersection.

## Physical run recovery

Version 1 completed both two-request P1+P2 passes and both continuous Direct
Sync passes (one warmup plus four requests). Their Native output checks pass;
video uses Native-aligned query partitioning and audio Channel Parallel,
without fallback. Analysis of the continuous trace exceeded its 300-second
stage budget, stopping subsequent cases. This was an analysis timeout after
the inference outputs and raw traces were saved, not an inference failure.
The analyzer now indexes temporal dependencies and pre-merges compute spans
instead of scanning every kernel for every loading event and idle gap. Its
source hash invalidates stale analysis caches. Physical comparison and
root-cause conclusions remain incomplete until all cases are recovered/run.

The version-1 raw captures have now been recovered using resumable range
downloads. Both captures were reanalyzed successfully with the indexed
analyzer; per-request kernel counts are approximately 49,000 on each rank,
loading/NCCL records are present, and the maximum clock-error bound is
33.793 microseconds. The original source-qualified inference/capture files
are unchanged. All three analyzer fixtures and compilation checks pass.

[MiniMax H3 Causal Pipeline Resume](https://www.kaggle.com/code/chinokiki9/minimax-h3-causal-pipeline-resume)
version 1 was submitted for dual T4 on 2026-10-12 and entered RUNNING/setup.
It executes only the remaining P1, P1+P2, and Offline+P1+P2 continuous cases,
each with the original one-warmup/four-measured-request workload and matched
timing/Kineto passes. Original counterbalanced capture order is retained.
Analysis now indexes event intervals and has a 900-second per-case ceiling;
GPU inference and cache policies remain unchanged.

Native tensors are restored directly from the original Kaggle output onto
the new Kaggle allocation. No weights/reference tensors are uploaded from
this machine, no Native generation/pilot/Direct Sync execution is repeated,
and every new output is checked against that saved physical Native reference.
The continuation manifest records inherited result hashes, identical source
bindings, reference transfer sizes/hashes, and the original checkpoint
identities. Reference restore validates lengths, shapes and finite values;
its download hashes are recorded checksums, not independently known upstream
checksums. This reuse does not claim a fresh Native execution. Comparisons
with Direct Sync span two allocations and must disclose that uncontrolled
hardware/runtime variability; the remaining three cases share one allocation.
Offline shards download anonymously on Kaggle and retain their established
SHA256 and Native manifest verification. Full rankings remain pending.

## Completion check, 2026-10-13

The continuation is COMPLETE: six actual full-model passes, four measured
requests each, output validation and combined analysis all finished. Total
notebook duration was 3.97 hours, including setup and exports. Local result,
correctness, analysis and NCCL evidence has been recovered. See
[CAUSAL_T4_RESULTS.md](CAUSAL_T4_RESULTS.md) for validated request latency,
per-stage/rank evidence, audit statuses and remaining attribution limits.

A post-run inspection found the category heuristic classified CUTLASS FMHA
as GEMM because of match order. Attention now takes precedence; four local
analyzer fixtures pass. Already exported heuristic category totals require
raw-trace re-analysis. Named kernels, operator attribution, output correctness
and unprofiled request latency are unchanged. No new inference benchmark was
started to repair this reporting issue. Nsight Systems remains unavailable.
