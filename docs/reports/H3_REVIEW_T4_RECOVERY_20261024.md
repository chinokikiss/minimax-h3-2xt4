# T4 review operators: partial results and scoped recovery — 2026-10-24

**User disposition:** long-sequence online-LoRA tests and profiling retries are
cancelled. No further GPU submission is scheduled. Version2's operator timing
case completed byte-exactly, but its combined profiling replay OOMed at LoRA
after the column/TE ranges. All48 final artifacts were retrieved, resuming the
interrupted transfer without rerunning the experiment. Historical results are
retained; the earlier plans below are superseded by this cancellation.

GPU version1 failed during a benchmark validation/warm-up repeat of the
51,220-row full-B QKV LoRA control on both actual T4 GPUs. **Ten cases on each
owner completed before the failure**, including all column/TE cases and the
two smaller LoRA cases. Their outputs are byte-exact in these fixtures.
The [retained partial results](../data/h3_review_t4_partial_20261024.json)
include all aligned samples, independent rank records, hardware/package
identities, transfer hashes and the exact errors. All25 final artifacts were
collected once and verified. No status/list/live-log polling was used.

## Failure and repair

The failed allocation was the full B output `[51220,21504]` in FP32, about
4.10GiB. Each T4 exposed14.56GiB total memory. The exception reported7.43GiB
allocated by PyTorch and3.59GiB reserved but unused, with3.41GiB driver free.
The **first full-B call had succeeded** and its FP16 output
`[51220,10752]` remained on GPU as a correctness oracle. Repeating the same
control then needed that extra1,101,434,880bytes, about1.026GiB, to coexist
with the candidate's own temporaries. Fragmentation may also have contributed;
the exception alone does not identify exclusive allocator root causes.

This is a benchmark lifetime defect. It does not establish that the owned-B
candidate failed, nor that the operator fits a complete online-LoRA H3 request.

The repair copies the correctness oracle to CPU and releases its GPU storage
before any repeated candidate call. Candidate output comparisons are also
performed on CPU, outside timing. Cache cleanup is confined to phase boundaries;
there is no per-sample `empty_cache`, allocator-environment change, chunked LoRA
GEMM or numerical shortcut. Original full-M GEMM geometry is retained.

The recovery timing scope contains **only the unfinished M51220 LoRA pair**.
The earlier ten timed cases are not repeated. Discarded RNG draws preserve the
original scripted draw order without running their GEMMs or loading their
weights. V1 did not export input hashes, so cross-version input identity is
not claimed as independently hash-verified; the repair now records hashes.
The separate14-label long-shape trace replay is still required because v1
failed before any profiling stage started.

Only benchmark source changes. Production patches, native source, core lock,
weights, geometry, factor strengths, seed, five-repetition timing count and
the complete profile matrix are preserved from the submitted archive.

## Completed actual-T4 measurements

Each table entry is the median of five samples, taking the slower of the
corresponding rank samples. Both owners' complete outputs match their original
full-geometry reference as raw storage bytes. Compilation, loading and checks
are outside these steady-state timings. These are seeded shape fixtures with
real trained base shards, not a real H3 trajectory-distribution evaluation.

| Column,51,220rows | Pre-edit baseline ms | Preallocated ms | TileLang epilogue ms | Direct INT8 + TileLang ms |
|---|---:|---:|---:|---:|
| QKV, K5376/N10752 | 533.289 | 520.692 | 466.791 | 232.461 |
| FC1, K5376/N14336 | 742.564 | 736.233 | 668.215 | 338.258 |

| Incremental operator peak,GiB | Original | Preallocated | TileLang epilogue | Direct + epilogue |
|---|---:|---:|---:|---:|
| QKV | 3.803 | 3.579 | 3.079 | 3.078 |
| FC1 | 4.811 | 4.604 | 4.104 | 4.104 |

These peaks subtract the already-live input/weight allocation, and take the
larger owner value. They are not complete-block or whole-request VRAM.

The direct path is approximately2.29×/2.20× faster than the original QKV/FC1
adapters in these measurements. This is not a whole-DiT speedup, and no
Nsight-based attribution is available yet. The five samples and alternating
variant order do not establish p95 or locked-clock repeatability.

| Completed helper comparison | Original ms | Candidate ms |
|---|---:|---:|
| TE Q/K/V,1182tokens | 9.626 | 7.619 shared preparation |
| TE gate/up,1182tokens | 38.553 | 36.578 shared preparation |
| Online QKV LoRA,261rows | 1.701 | 1.374 owned B |
| Online QKV LoRA,4096rows | 27.148 | 20.842 owned B |

V1 recorded160 aligned sample pairs for the completed variants, or320
individual rank samples. It provides no complete M51220 LoRA comparison and
no valid profiler trace. The original version1 and its failure remain retained.

## Recovery validation and submission boundary

The updated132-test CPU suite passes with zero failures/errors/skips both in
the working source and an isolated extraction of the delivery archive.
Regression checks ensure the oracle does not retain source tensor storage and
that recovery preserves the scripted RNG schedule. A local SM120 lifetime
preview also completes the M51220 LoRA pair with a12GiB PyTorch allocator quota;
its outputs are byte-exact. This is harness validation, **not T4 qualification**.

The new source archive must pass its own CPU-only Kaggle setup/API/contracts
gate before one replacement GPU version is submitted. The replacement keeps
the original test items and complete trace coverage requirement while avoiding
duplicate timed cases. No full-H3 generation, dependency update, upstream
source update or production-default promotion is part of this repair.

**Submission update:** CPU notebook version3 passes all132 tests, with zero
failures/errors/skips, in the unchanged pinned-core/native-platform setup.
Setup took109.99s. Replacement GPU **version2 was accepted** at
[H3 Review Operators T4](https://www.kaggle.com/code/pamcks/h3-review-operators-t4-20261024),
with `review_recovery=True`, zero mounts and a1200-second execution ceiling.
It is bound to source capsule
`e51b6a7d7b3af344a79eb7cd05e0913cdbb73ed2315fa8a51b8bbb1d428196e2`,
published at Git commit `c5a59a4e9a2b79840a2444b9808ca0eb6c23834b`.
Accepted submission is not a completed T4 recovery result. Final retrieval is
left to the user's completion reminder; no polling or automatic collection is
scheduled. The [recovery evidence index](../data/h3_review_t4_recovery_20261024.json)
retains the source-only diff, CPU receipt, local lifetime preview and submission.

Evidence roots: `D:/attention_eval/h3_review_t4_20261024/gpu_outputs_v1/` and
`D:/attention_eval/h3_review_t4_recovery_20261024/`.
