# Selective B GEMM feasibility study (benchmark only)

## Data-flow verdict

**Feasible on the tested H3/DMAD geometry; physical dual-T4 validation passed.**
Selective B is byte-exact against existing Row Partial in all tested cases.
At S8192 it reduces column-FC1 temporary allocation from 1348 to 938.03 MiB/rank
and whole-FFN latency by 6.17% (single) / 9.10% (two-entry stack). Whole-FFN
temporary allocation only falls from 1348 to 1344 MiB/rank: approximately
900 MiB for the whole FFN is **not achieved**.

**Recommendation: investigate further before adoption.** The latency benefit
is promising, but the requested large whole-FFN memory saving does not occur,
short-sequence timing is unstable, and independent multi-adapter/rank-width
coverage remains open. The implementation stays benchmark-only. No runtime/UI,
checkpoint, cache-key, Row Partial or quantizer implementation is changed.

The actual producer is `ColumnParallelLinearINT8.forward` in `patches/tp_dit.py`.
The base column projection returns rank-local packed gate/up features. Its
`NativeAlignedLoRA.add_to` computes every adapter's A, then full B, and
`_add_delta` consumes only the owned B rows. No consumer observes discarded rows.
SwiGLU and FC2 read the updated rank-local packed output; FC2's global row maxima
and reductions already combine the two feature owners. They do not require
redundant unowned B values on each rank.

Actual FC1 shapes for the supplied DMAD checkpoint (FP32 runtime factors):

- A `[128, 5376]`; input `[S, 5376]`; hidden `[S, 128]`.
- Full B `[28672, 128]`; full delta `[S, 28672]`.
- Rank 0 rows: `[0,7168)` then `[14336,21504)`.
- Rank 1 rows: `[7168,14336)` then `[21504,28672)`.
- Packed selected B `[14336,128]`; selected delta `[S,14336]`, gate then up.

For an output-row selector P, `P(H B^T) = H (P B)^T` over real arithmetic.
No normalization, reduction or scaling across B output rows precedes selection.
Changing B GEMM N can still change FP32 kernel selection / accumulation, so this
identity is not assumed to imply byte-exactness on every backend or rank width.

## Isolated prototype

`benchmarks/selective_b_prototype.py` is used only by the study harness. It
packs each adapter's two B intervals once with `torch.cat`. The prototype
shares the original A buffer and original scalar strength; each A GEMM stays
separate. It preserves alpha normalization, adapter order and separate post-B
multiply/add. It performs one whole selected-B GEMM per adapter, with no B
output tiling, fusion, extra collective, new quantization or A approximation.

The two separated intervals require one contiguous packed weight allocation:
7 MiB per adapter/rank (14 MiB for the tested two-entry stack). Packing is outside
steady-state timing, and its CUDA/wall time is reported separately. No slicing
or concatenation is performed in forward. Full B remains in the comparison
harness for the baseline, but is not retained by the prototype itself.

The benchmark toggles only its private block's FC1 adapter. Production source
hashes and checkpoint signatures are protected before/after. Prototype factors
are not written to a checkpoint or registered in a residency/cache identity.

## Numerical protocol

Both implementations use the existing, validated Row Partial and Quantizer
Lifetime path. Compare against that baseline rather than only Native:

- Real learned blocks 0/25/49, S513/8192.
- Single DMAD at strength 1 and two separate entries at +0.6/-0.2.
  Only one real LoRA checkpoint is available; the multi test is not a second
  independently trained LoRA or a different-rank qualification.
- Per-owned-rank FC1 output, same-hidden B-only output, whole FFN output:
  byte equality, max absolute error, relative L2 and cosine.
- Actual unchanged quantizer on each packed FC1: changed INT8 elements,
  changed row-scale representations and scale numerical differences.
- Replicated whole-FFN outputs must agree across TP ranks. FC1 outputs belong
  to different feature intervals, so cross-rank FC1 equality is inappropriate;
  each rank is compared to its own baseline ownership.

Local initial checks passed all 12 configurations with byte-exact FC1/FFN and
unchanged INT8 payload/scales. The final harness separately releases numerical
diagnostic tensors before timing/residency measurements and is hash-gated
before physical submission. Logical-rank timings are not performance evidence.

## Physical A/B protocol

Kaggle `chinokiki9/h3-selective-b-feasibility`, version 1, completed successfully
on physical 2x Tesla T4, NCCL 2.28.9, PyTorch 2.11.0+cu130, comfy-kitchen 0.2.37,
TF32 disabled. ComfyUI is pinned to
`d49e888586dd8ae012c0667b33466b815fee07f7`.
Timing covers block0, S513/8192 and both stacks. Other blocks qualify numerics.
Inputs are seeded FP32 fixtures; projections use actual checkpoint factors.
These are isolated projection/FFN measurements, not complete generation latency.

Each measured scope (column FC1, whole FFN, and a same-hidden B-only GEMM) uses
two reversed-order rounds, two warmups and five synchronized CUDA Event/wall
samples per round. Transient allocator cache is cleared before each scope;
peak allocated, peak reserved and resident baseline are reported per rank.
Both comparison adapters remain resident, with shared A and a separate selected
B buffer. Absolute peaks therefore include this fixed comparator state;
temporary allocated deltas are the primary activation comparison. Reserved
memory is PyTorch allocator residency, not complete driver/NVML memory.

Separate diagnostic passes measure FC1 and FC2 peak allocation relative to the
same entry, and capture bounded shape-aware profiler traces for B-only, FC1 and
FFN. Profiling is excluded from accepted latency and peak measurements because
shape recording can hold references. Kernel names and CPU/GPU correlations are
retained to distinguish B execution from copies and unrelated GEMMs.

## Physical numerical results

| Comparison | Rank comparisons | Byte-exact | Max absolute / relative L2 | Minimum cosine |
|---|---:|---:|---:|---:|
| Packed column FC1 | 24 | 24 | 0 / 0 | 1 |
| Whole FFN | 24 | 24 | 0 / 0 | 1 |
| Same-hidden first-adapter B GEMM | 24 | 24 | 0 / 0 | 1 |

All 12 configurations have matching replicated whole-FFN outputs across ranks.
Downstream INT8 changed elements: **0**; changed scale rows: **0**. No numerical
drift was observed. This proves equality for the tested input/shape/backend
matrix, not a universal guarantee across different GEMM algorithms or adapters.
The two-entry FC1/FFN test checks both sequential adapters; isolated B-only
timing and direct-B comparison use the first adapter to isolate one GEMM.

## Physical latency

Milliseconds below are medians of ten samples, each using the slower rank for
that iteration. B-only is one adapter; FC1/FFN apply the named complete stack.
CUDA Events and synchronized wall time agree closely.

| S | Stack | B GEMM wall baseline -> selected | FC1 wall baseline -> selected | Whole FFN wall baseline -> selected | Whole FFN CUDA baseline -> selected |
|---:|---|---:|---:|---:|---:|
| 513 | Single | 1.406 -> 0.695 | 4.744 -> 4.250 | 11.487 -> 9.897 | 11.477 -> 9.882 |
| 513 | Two entries | 1.043 -> 0.547 | 5.175 -> 4.425 | 12.147 -> 10.861 | 12.132 -> 10.851 |
| 8192 | Single | 15.767 -> 5.929 | 65.925 -> 57.692 | 141.782 -> 133.034 | 141.766 -> 133.022 |
| 8192 | Two entries | 16.180 -> 5.926 | 95.192 -> 77.614 | 183.736 -> 167.023 | 183.720 -> 167.007 |

S8192 FC1 latency decreases 12.49% / 18.47%; whole FFN decreases 6.17% / 9.10%
for single / two-entry stacks. Both reversed rounds reproduce the S8192 FFN
direction: single 142.058 -> 132.826 and 141.738 -> 133.384 ms; two entries
183.781 -> 167.004 and 183.690 -> 167.042 ms.

**S513 is exploratory.** Single-stack baseline FFN round medians move from
13.652 to 9.306 ms, while selected moves from 10.933 to 8.739 ms. Single-stack
FC1 even reverses direction in round 2 (3.424 -> 3.634 ms). Two-entry FC1 round 1
only changes 4.555 -> 4.431 ms. Consequently the pooled short-sequence medians
are not stable estimates of the gain. Clock/thermal state is a possible cause,
not established by this run; clocks/power were not sampled. No fixed S513
speedup is recommended from these data.

## Physical memory and actual new peak

Both ranks report identical maxima. MiB values are PyTorch allocator data;
reserved peaks are maxima across both timing rounds after per-scope cache reset.
Resident baseline is equal between A/B within each stack: 447.00 MiB single,
522.88 MiB two entries at S8192. Both comparator adapters are resident, so
absolute allocated/reserved totals include the benchmark's extra packed B.

| S8192 scope / stack | Temporary allocated baseline -> selected | Absolute allocated baseline -> selected | Absolute reserved baseline -> selected |
|---|---:|---:|---:|
| B-only / single or two entries | 896 -> 448 | See raw per-scope data | See raw per-scope data |
| FC1 / single | 1348 -> 938.03 | 1795.00 -> 1385.03 | 2466 -> 1570 |
| FC1 / two entries | 1348 -> 938.03 | 1870.88 -> 1460.91 | 2552 -> 1656 |
| Whole FFN / single | 1348 -> 1344 | 1795.00 -> 1791.00 | 2466 -> 2018 |
| Whole FFN / two entries | 1348 -> 1344 | 1870.88 -> 1866.88 | 2552 -> 2104 |

At S513, baseline FC1/FFN temporary peak is 84.415 MiB/rank. Selected FC1 is
58.987 MiB single / 59.487 MiB two entries, and selected FFN is 84.164 MiB.
All per-rank allocator fields, including B-only reserved values, are in summary.json.

The separate S8192 stage passes locate the peaks identically on both ranks:

- Baseline FC1: `448 packed + 896 full B + 4 hidden = 1348 MiB`.
- Selected LoRA arithmetic would require `448 packed + 448 selected B + 4 hidden
  = 900 MiB`. However the unchanged **base FC1 INT8 operator** already peaks at
  **938.03125 MiB**. comfy-kitchen 0.2.37 `int8_linear` allocates a 448 MiB `out`
  before its Turing helper allocates another 448 MiB output; 42 MiB input INT8
  payload and 0.03125 MiB row scales remain live. Their sum matches the measured
  peak exactly. The trace confirms the Turing fused INT8/dequant kernel route.
- FC2/quantization stage: **1344 MiB** with either variant. The unchanged global
  FP32 quantizer overlaps 448 MiB packed gate/up input, 448 MiB zero-padded
  full-width activation and 448 MiB Native-geometry rotation output. The source
  lifetime accounting and independent stage peak agree. Allocation-level memory
  profiling was intentionally disabled; this is source/route/peak attribution.

Thus Selective B removes 448 MiB from each B output, reduces the measured FC1
peak by 409.97 MiB, but saves only **4 MiB (0.30%)** at the whole-FFN peak. Lower
reserved memory does not establish equivalent live-memory savings or driver VRAM
savings. No changes to those other allocation sites were made in this study.

## Kernel selection and extra work

S8192 GEMMs: `[8192,128] x [128,28672]` becomes
`[8192,128] x [128,14336]`. Both use `volta_sgemm_128x128_tn`; grid changes from
`[224,64,1]` to `[112,64,1]`. The selected output halves B FLOPs/output writes
while preserving the observed kernel family. B-only CUDA Event latency is
15.758/16.171 -> 5.922/5.917 ms for the single/two-entry measurement contexts.
Differences beyond a factor of two are not attributed solely to FLOP reduction.

The bounded profiler captures S8192 two-entry work only. In rank0's FC1 traces,
CPU `aten::mm` external IDs 1071/1091 (baseline FFN trace) and 2607/2623
(selected FFN trace) identify the two B launches and correlate to CUDA kernels.
FC1-only trace examples: baseline external IDs 558/578 and selected 2094/2110;
use summary.json for the actual IDs/intervals on each rank. B-only profile
kernel activity is 10.620/10.307 ms baseline vs 6.045/5.511 ms selected on
ranks0/1; these diagnostic durations are not acceptance timings.

FC1 and B-only traces contain no `aten::cat`, `clone`, `contiguous`, or `copy_`.
Whole-FFN copy counts are unchanged: one cat, two contiguous, two clone and four
copy_ CPU events per rank/variant, all belonging to the unchanged surrounding
path. No additional collective is introduced. Base quantization, Row Partial
A reductions, scalar strengths and adapter accumulation order remain unchanged.

Packing requires one 7 MiB contiguous weight copy per adapter/rank, outside
forward; two entries require 14 MiB. First packing takes 3.90/9.43 ms on ranks0/1;
later packing observations are 0.34-0.72 ms (different stack/block counts).
These are setup measurements, not a forecast for checkpoint loading. The
prototype itself shares A and retains only selected B; the comparison harness
also retains baseline full B for fair toggling. No production residency saving
or checkpoint-format improvement is claimed.

## Decision and limits

Selective B is mathematically valid, observed byte-exact, and measurably faster
for the tested S8192 projection/FFN cases. It fails the proposed approximately
900 MiB **whole-FFN** target because other unchanged allocations dominate.
Before a separate adoption decision: qualify independent LoRA checkpoints and
rank widths, repeat short-sequence timing under stable measured GPU clocks,
and establish complete-request latency. None of that is silently substituted
with the isolated results here. No production integration was performed.

## Evidence / reproduction

- `semantic_results/selective_b_integrity_before.json`: protected source hashes
  and checkpoint signatures from before the study.
- `semantic_results/selective_b_local/`: full initial numerical matrix.
- `semantic_results/selective_b_local_final/`: final-source submission gate.
- `benchmarks/selective_b_prototype.py`, `selective_b_audit.py`.
- `build_selective_b_job.py`: standalone source-only Kaggle job builder; selected
  checkpoint ranges download on Kaggle, with no local weight upload.
- `semantic_results/selective_b_t4_v1/selective_b_results/results.json`: complete
  numerical/timing/stage/packing records; status.json and dual_t4.log confirm
  process success (physical audit 51.04 s, excluding setup).
- `semantic_results/selective_b_t4_v1/summary.json`: derived per-rank memory,
  sample/round timing ranges, numerical aggregate and correlated GEMMs/kernels.
- Twelve compressed `SelectiveB_*.json.gz` traces in that evidence directory;
  profiler was opt-in and excluded from accepted timings/peaks.
- `semantic_results/selective_b_t4_v1/kitchen_0_2_37_cuda.py`: source from upstream
  tag v0.2.37; SHA256 `4b32e00e230d451e8bdc9c1fbd6917d8e7d2cc0d439dec13e3107e0539252ffd`.
  Relevant allocation sites: lines 2190/2200 and helper line 1065.
- `benchmarks/summarize_selective_b.py`: reproducible bounded evidence summary.
  Run with the selective_b_results directory as its single argument.

Physical worker source hashes match local sources; seven protected production
source hashes and original checkpoint size/mtime signatures remain unchanged.
No checkpoint/tensor dumps were downloaded. Version-bound evidence collection
has SHA256 receipts in download_manifest.json; only compact logs/results and
compressed traces are retained. The complete evidence directory, summary and
upstream allocation-source copy total approximately 568 KiB.
