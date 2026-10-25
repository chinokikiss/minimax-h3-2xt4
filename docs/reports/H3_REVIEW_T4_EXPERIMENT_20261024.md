# Bounded T4 qualification of the review optimizations — 2026-10-24

This experiment qualifies the new GPU operator paths from the
[review implementation follow-up](H3_REVIEW_OPTIMIZATIONS_20261024.md).
It does not run H3 sampling, construct a complete H3 model or decode media.
The requested exclusions remain: cross-rank deduplication of offline LoRA CPU
merges and Scalar Sparse kernels are unchanged.

The new 130-test local CPU suite passes with zero failures, errors or skips.
Kaggle CPU qualification version 2 **passes all130 tests**, with zero failures,
errors or skips, at
[H3 Review Operators CPU](https://www.kaggle.com/code/pamcks/h3-review-operators-cpu-20261024).
Version1 passed imports, signatures and128 of130 tests; two tests exposed
missing launcher/workflow fixture files in the source capsule. The corrected
archive includes them and independently passes all130 tests after extraction
into an isolated local directory. No GPU job was submitted for version1.
The matching GPU notebook version1 was accepted at
[H3 Review Operators T4](https://www.kaggle.com/code/pamcks/h3-review-operators-t4-20261024),
bound to that corrected source capsule and successful CPU receipt.
Submission acceptance does not establish completion or physical T4 results.
Final GPU artifacts are pending the user's completion reminder.

**Update:** version1 final artifacts were collected after the user reported
failure. Ten cases per owner completed with byte-exact outputs; the last large
LoRA validation repeat OOMed while retaining its GPU oracle. Profiling had not
started. See the [partial results and scoped repair](H3_REVIEW_T4_RECOVERY_20261024.md)
for the failure, preserved measurements and replacement scope. The preceding
submission text records the original state, not the current completion status.

The CPU host retained native Python3.13.15 / Torch2.11.0+cpu; Kitchen0.2.37,
TileLang0.1.13 and the legacy attention signatures pass. Setup took119.12s,
including17.23s for the additional130-test suite. These are CPU setup timings,
not T4 operator performance. GPU versions and compiler ABI are checked again
on the actual T4 host before the fixture downloads.

## Measurements

Each TP2 owner runs on a separate physical T4. The device gate requires two
distinct GPU UUIDs, Tesla T4 names and SM75 capability. Every selected column
route is checked through its public dispatcher so silent backend fallback
cannot masquerade as an accelerated measurement.

| Case | Shapes and variants | Validation |
|---|---|---|
| DiT QKV column | `M × 5376 → M × 10752`; M = 261, 4096, 51220. Exact pre-edit implementation, preallocated eager destination, eager INT8 GEMM + ordered TileLang epilogue, direct native INT8 GEMM + ordered TileLang epilogue. | Final FP16 raw-storage equality against the preserved pre-edit source; nonfinite fixture references fail. |
| DiT FC1 column | `M × 5376 → M × 14336`, same M and four variants. | Same checks, preserving the complete INT8 GEMM geometry. |
| TE shared preparation | `1182 × 5120`; Q/K/V owned output widths 4096/512/512, gate/up 12800/12800. Independent pinned eager calls versus shared rotation/quantization. | Raw-storage equality for every projection on each owner. |
| Optional QKV LoRA B slicing | Same three M values; A rank128, full B output21504 versus component-aware owned10752. Two ordered seeded adapters, FP32 LoRA compute, FP16 residual/output. | Raw-storage equality against full B plus owner selection. Base clone is included equally in both timing boundaries. |
| Ordered dequantization | Adversarial INT32 values near conversion boundaries, non-power-of-two row/channel or scalar scales, FP16 and FP32 outputs. | Raw-storage equality, including signed zero and FP16 overflow rounding behavior. BF16 is not presented as native T4 execution. |

There are 11 timed cases and 34 variants per owner. Every variant has a first
invocation record and two additional warm-ups, then five measurements with
alternating variant order. CUDA events measure the device-stream interval;
synchronized host wall time, allocated peak and reserved memory are recorded
separately. The large correctness reference is released before peak-memory
measurement. First dispatch, compilation/warm-up, download and loading time
are excluded from steady-state measurements and retained in separate fields.

Both owners execute concurrently. Their results are reported separately;
this is not a distributed collective benchmark or a full-pipeline speedup.
No automatic default promotion follows from the result.

## Data and source boundary

Only four public trained block-zero shards are downloaded: TE and DiT for
rank0/rank1, totalling **880,657,008 bytes**. The plan rejects additional
blocks, mutable revisions, authentication or an asset payload above 1GiB.
The source repository is `cnmds/minimax-h3-tp2-int8-561a6211`, immutable revision
`d3e6a4b131056d2c626a01acf6ba48123a885783`. Each complete shard's byte count
and SHA256 are checked before use. No baked checkpoints are produced.

Activations and online LoRA factors are seeded fixtures. Trained base weights
are real; these inputs are **not captured H3 activation distributions** and
the results cannot establish generation-distribution or semantic acceptance.
Private activations and credentials are absent from the source and notebook.

The frozen ComfyUI archive, legacy attention API, original environment
snapshot and direct-pip core lock are byte-preserved. Kitchen0.2.37 and
TileLang0.1.13 remain exact. Kaggle keeps its native platform package family;
the actual environment is recorded. A mismatched CUDA compiler/extension ABI
blocks the experiment instead of changing dependencies.

Published source archive:
`077fee305cf8cd0a538682bcfbd31d0f9e50e6e2cf60fb3b3d4e228d2578f669`,
source capsule:
`e56aeebd92c6a2884e1442290fff89eedb59d9ea617ef2ef0627fd469c4057d2`,
Git commit `1064ab91f2f9f253ea7b78dfbba5475109b3f7fa`.
The pre-edit column baseline is independently bound to SHA256
`437741a15f536925563453eb7463c472a297f3f2c8c7983deed1bd78f1125ebf`.

## Profiler evidence and remaining boundaries

Steady-state measurement workers run without Nsight. A separate warmed
long-shape replay records CUDA, NVTX and OS-runtime activity. First, the
minimal two-worker capture/stop/restart/exit test must pass. The replay then
requires all14 expected operator labels, correlated kernels on both distinct
GPUs, explicit worker stop/flush receipts and clean process exits. Summary
CSV attribution is withheld when coverage is incomplete. An unavailable
Nsight tool is recorded as a blocker while keeping independent timing data.
The historical Nsight Compute counter-permission blocker is not reprobed.

Loading/integrity, bounded read amplification, rank1 reply ownership, loader
controls and metadata-only Video parent behavior receive native-Linux CPU
contract coverage before submission. Their tests are not physical SSD
throughput measurements or a learned Video decoder qualification. In
particular, two-T4 Video admission/public decoding and the historical
261-token final-projection discrepancy remain separate pending work.

The CPU setup has a480-second cell-execution ceiling; GPU execution has a
1200-second ceiling, with additional bounded worker/profiler subprocesses.
Provider queue/startup time is outside the notebook cell limit. Submission
receipts use exclusive claims. No status/list/live-log polling or automatic
GPU collection is scheduled; final artifacts are collected once when the
user reports completion.

Evidence root: `D:/attention_eval/h3_review_t4_20261024/`.
The [aggregate evidence record](../data/h3_review_t4_experiment_20261024.json)
retains both CPU attempts, source hashes, publication and the accepted GPU
receipt. There were three submissions and two final CPU artifact collections,
zero status/list/live-log polls and zero whole-H3 runs in this task.
Entry points: `benchmarks/build_review_t4_delivery.py`,
`benchmarks/run_review_cpu.py`, `benchmarks/run_review_t4.py`.
