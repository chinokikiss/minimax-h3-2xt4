# T4 experiment: RowPartial LoRA and reference-demand VAE loading

**Current status: zero-mount GPU version 3 accepted; execution unverified.**
After the user stopped both earlier instances, the
[CPU-qualified direct-pip/source delivery](KAGGLE_LIGHT_DELIVERY_20261022.md)
replaced the mounted wheel/component outputs. The retained real input is
committed privately on HF at an immutable revision. Version 3 keeps the same
operator matrix and exact core pins, uses no dataset/kernel-output mounts,
and has separate 900-second setup and 1,200-second execution watchdogs with
a 2,400-second CLI limit. The explicitly authorized temporary authentication
is confined to its private notebook. No status/live-log polling or final
collection has been performed for version 3. The
[delivery record](../data/kaggle_light_delivery_20261022.json) binds its source,
qualification, private-input revision and submission hashes.

**Version 2 history: accepted, then stopped by the user.** The user
explicitly requested another GPU submission after cancelling version 1. Kaggle
accepted version 2 with the same operator matrix and frozen dependencies, the
600-second outer worker deadline, immediate setup/stage-start output and a
requested 900-second CLI runtime limit. No status/live-log polling or artifact
collection has been performed for version 2. Its
[submission receipt](../data/h3_rowpartial_t4_submission_20261022_v2.json) binds
the exact notebook and source hashes. This is submission acceptance, not a
physical T4 result. The existing source-bound CPU preflight and six local
watchdog/evidence tests pass; Linux process-group cleanup remains unqualified.

**Version 1 history: stopped by the user; no physical results available.** Kaggle
accepted GPU version 1 at approximately 19:18 China time on 2026-10-22. The
initial version-bound retrieval returned **zero files**. The user subsequently
confirmed **Running**, then manually stopped the notebook after observing about
25 minutes with no output. One final post-stop artifact collection also returned
**zero files**, in a new directory preserving the first attempt. No device
receipt, timing, numerical result, worker log or profiler artifact is available.
The UI observations do not establish the stalled stage or actual GPU compute
time. No automatic status/list/live-log polling or retry submission was used.
The CPU preflight below passed;
**no new T4 qualification or speedup is claimed**.

The [submission/acquisition record](../data/h3_rowpartial_t4_submission_20261022_v1.json)
preserves source/receipt hashes and the empty artifact manifest.
The [cancellation and harness audit](../data/h3_rowpartial_t4_cancelled_20261022_v1.json)
records the second empty collection and the unsubmitted local remediation.

This is a new source-bound hardware experiment for the production combination
FP16-fix + fused-stored preparation + scaled FP16 base transport + FP32 LoRA
A-partial SUM. It does not run the H3 sampler or full generation. Numerical
operators, Kitchen 0.2.37, TileLang 0.1.13 and the frozen legacy ComfyUI API are
preserved. Production defaults are not changed by the measurement harness.

## CPU preflight and submission

The fresh private [CPU preflight](https://www.kaggle.com/code/pamcks/h3-rowpartial-reference-cpu-preflight),
version 1, reuses the prior hash-bound wheel archives rather than resolving or
upgrading packages. Current production-source imports, legacy signatures,
Linux positional reads and **17 current-source tests** passed. The actual CPU
image is Python 3.13.15 / Torch 2.11.0+cpu; this CPU result is not GPU qualification.
The prepared ABI archives for Python 3.11/3.12/3.13 retain their prior hashes.

The private [GPU batch](https://www.kaggle.com/code/pamcks/h3-rowpartial-reference-t4),
version 1, requests `NvidiaTeslaT4`, has a **600-second internal budget** and a
900-second requested CLI runtime limit, disables internet, and mounts the frozen source,
reviewed component assets, new CPU receipt and prior physical gate result.
It refuses changed core-source hashes, missing receipts, a changed native GPU
Torch version or hardware other than two distinct SM75 T4s. Offline core
installation/import validation is separated from measurement. No dependency
resolver or environment installer runs as a GPU experiment.

Local root: `D:/attention_eval/h3_row_partial_t4_20261022/`. `cpu_v1` and `gpu_v1`
preserve submitted notebook/metadata/source bindings. Final artifacts are
collected after its fixed window; the additional final attempt followed the
user's manual UI update and cancellation. No status/list/live-log polling is
used. Physical results are added below only from returned artifacts.

## Cancellation audit and local remediation

Static review of the immutable submitted notebook identifies two harness gaps:

- The internal 600-second deadline caps `run(...)` experiment subprocesses, but
  does not interrupt source verification/extraction, native package inventory,
  wheel hashing/extraction or offline installation. The install helper has its
  own independent 600-second pip timeout; it does not enforce the remaining
  whole-batch deadline.
- Visible progress is emitted only after a test subprocess completes. Setup
  phases have no immediate stage-start output, and subprocess output goes to
  files. A notebook that has started can therefore appear silent during setup.

These are confirmed observability/deadline defects, **not proof that setup was
the cause of this particular cancellation**. Provider startup/input mounting,
setup and device tests cannot be distinguished without returned artifacts.
The requested CLI limit is also not evidence of an enforced notebook wall-time
limit; the user's reported duration exceeded it.

The builder now prepares a separate worker under an outer 600-second supervisor.
It starts timing before the worker's source/setup code, streams output to the
notebook and `notebook_worker.log`, records setup/stage starts in
`progress.jsonl`, writes `supervision.json`, and kills the worker plus active
experiment process groups on deadline. Provider boot/input mounting before the
first notebook cell remains outside this deadline. Completion of the worker is
explicitly not an experiment pass; original device/source/correctness gates
still determine acceptance. Core source, frozen dependencies and the operator
matrix are unchanged.

`gpu_v2_prepared_not_submitted/` contains this revision; its historical directory
name is retained, but it **has now been submitted as version 2 at the user's
explicit request**. Two local CPU watchdog tests cover normal exit and a simulated
pre-experiment setup hang; four evidence-analysis tests also pass. Linux process
group termination and provider behavior still require CPU-host validation
before relying on the revised harness for another GPU run. Submitted version 1
and both empty acquisition manifests remain unchanged.

## Controlled row/block matrix

The paired comparison changes only the row LoRA implementation: Native full
activation gathering versus FP32 A-partial reduction. Both use current
production dispatch, FP16 residual divisor 64, fused-stored FC2 preparation,
original INT8 GEMM, scaled FP16 base SUM, FP32 epilogue/LoRA and one FP16 store.
Column-parallel Selective B and unsharded adapters remain common to both sides.

| Input / layer | Projection | Current TP2 local K -> N | Ordered adapter stack |
|---|---|---|---|
| S6814 / layer 0 | attention O | 3584 -> 5376 | one trained rank128 adapter; two-entry stack |
| S6814 / layer 0 | FC2 | 7168 -> 5376 | same |
| S513 / layer 49 | attention O | 3584 -> 5376 | same |
| S513 / layer 49 | FC2 | 7168 -> 5376 | same |

Each input/stack also measures the complete trained block with both row
adapters switched together. Two modes × four input/stack combinations × three
scopes gives **24 records per rank**. The two-entry stack uses the same learned
DMAD factors with strengths 0.7 and -0.25, preserving separate adapter operations;
it is a controlled stack, not two distinct LoRA training runs.

At S6814, the FP32 local activation buffers are 93.2 MiB for O and 186.3 MiB
for FC2; each rank128 hidden buffer is 3.33 MiB. The one-adapter payload-size
ratios are therefore 28 and 56 respectively. These are logical buffer sizes,
not measured PCIe traffic or expected latency speedups. A two-entry stack
shares one activation gather in the control but performs two hidden SUMs in
the partial implementation, so launch cost and local GEMM cost still need
measurement.

Inputs select 6814 or 513 rows from the retained S7895 first-block tensor.
Layer 0 uses that real first-block input subset; layer 49 reuses it as stress,
not its own denoising trajectory. These are deployment-relevant TP2 shapes,
not a new complete trajectory capture. The Native FP32-flow block reference
retains original INT8 operators, prior packed/owned-FP32 preparation and full
input-gather LoRA; it does not inherit the new lossy default as its reference.

The isolated rows use identical captured production row inputs and learned
weights for both sides. Their independent LoRA reference holds the common,
previously qualified FP32 base output fixed. It computes A projections, gathers
partials and adds the two FP32 hidden results independently of the production
AllReduce, then evaluates B, post-B scale/add and final FP16 storage. This is a
LoRA implementation check with a common base, not a new independent proof of
the entire base codec. Whole-block diagnostics compare with the gather control
and Native FP32 flow; they do not imply generation-quality acceptance.

## Timing, allocation and profiler boundaries

First calls are recorded separately, then both modes receive two warmups.
Two reversed rounds interleave five samples per mode, giving ten paired samples.
Each sample barriers outside its timer, synchronizes CUDA and measures the
complete Python API. Aggregate latency takes the slower corresponding rank.
Both ratio-of-medians and median paired speedup are retained; clocks are not
locked. Compilation, loading, numerical comparisons and profiler passes are
outside repeated steady-state timing.

Allocation measurements record the resident baseline and incremental peak for
each timed call. Both adapter implementations and fixed captured inputs remain
resident, consistently, across candidates. Allocator reservations, external
NCCL allocations and full-pipeline VRAM are not represented by this increment.

Post-warm PyTorch CPU/CUDA traces record both row projections and modes at
S6814 for both stack sizes. Explicit adapter transport observations distinguish
the large FP32 activation AllGather from one FP32 `[M,R]` SUM per adapter. A
separate short Nsight Systems pass, when available within budget, marks warmed
production rows with NVTX regions and exports its raw report plus CUDA kernel,
API and copy statistics. Whole-process statistics also include load/compile
activity; the named row regions identify the intended execution boundary.
Nsight Compute counters are not re-probed under the unchanged historical
permission blocker, so no new hardware memory-stall/counter attribution is made.

## Reference-demand VAE component test

This separate process runs the actual trained Native Video/Audio components on
T4 GPU0 with real checkpoint reads and host-budget leases. A lightweight parent
supplies the same StagedVAE and reference-filter methods; it does not create
persistent TP2 workers or validate their complete lifecycle concurrently.

It checks metadata-only constructors with zero component payload/lease,
image-only, audio-only and mixed encoding, reference-specific idle selection,
encoder release before sampling, and decoder-first-after-release execution
without encoder reads. Video input is one 32×32 frame; audio is 1,600 stereo
samples. Output shapes and deterministic learned-model outputs are compared
with ordinary Native loading. First load+execution and warmed three-sample
execution are separated. Small shapes verify lifecycle/arithmetic, not real
video/audio decode throughput. Disk state is uncontrolled, so cold-call numbers
must not be interpreted as a storage-algorithm comparison.

The known Video 261-token query-parallel final-projection mismatch is outside
this test and is not claimed repaired.
