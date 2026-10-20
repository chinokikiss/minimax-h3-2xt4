# Kitchen-inspired Group32 second pass

2026-10-17. Operator experiments only; current TP2 shapes and captured inputs.
No H3 generation, sparse-policy change, new capture upload or production hook.

Private job `chinokiki9/h3-tp2-kitchen-group32-v2`, version1, is complete.
Hardware is confirmed as two Tesla T4 SM75 devices, with nvcc12.8 and
Torch2.11.0+cu130. Screening, a fresh-process repeat, Video Systems capture and
DiT Systems capture all succeeded. Ten submitted source hashes match the
version-bound outputs. The local preview below remains separate evidence.

## Actual T4 outcome

The strongest result is the **Video cast/pad preparation variant**, which
combines register-held quantization scales with K cast/pad fusion. It reduces
complete API latency7.42% in screening and9.45% in the repeat relative to the
previous T4 winner. The DiT main-kernel changes do not establish an additional
gain; retain the previous BQ64/FullRow/256-thread configuration. TE's apparent
gain shrinks substantially in confirmation and should not be advertised as a
stable14% improvement.

All values below are slower-rank median API CUDA-event intervals in ms,
including preparation and the output boundary; JIT/loading are excluded.
These are paired comparisons within the same allocation and round.

| Component / variant | Screen ms | Screen reduction | Fresh-process repeat ms | Repeat reduction |
|---|---:|---:|---:|---:|
| TE previous winner | 0.618496 | — | 0.589824 | — |
| TE register scales + cast/pad | 0.529152 | 14.45% | 0.579424 | 1.76% |
| DiT previous winner | 71.304932 | — | 71.325699 | — |
| DiT staged K scales | 69.803551 | 2.11% | 72.886978 | -2.19% |
| Video previous winner | 2.269664 | — | 2.239456 | — |
| Video register scales + cast/pad | 2.101248 | 7.42% | 2.027840 | 9.45% |
| Video native Kitchen comparator | 1.321568 | different numerical method | 1.263648 | different numerical method |

Other screen-only ablations:

| Component / variant | API ms | Reduction relative to same-round previous winner |
|---|---:|---:|
| TE register scales only | 0.898944 | -45.34% |
| DiT register scales only | 70.345215 | 1.35% |
| DiT register scales + cast/pad | 70.376289 | 1.30% |
| DiT row-local softmax with explicit exchanges | 90.929153 | -27.52% |
| Video register scales only | 2.239616 | 1.32% |
| Video finite-output epilogue only | 2.420608 | -6.65% |
| Video cast/pad + finite-output epilogue | 2.185984 | 3.69% |

Removing a barrier or output launch does not guarantee a complete-API win.
The standalone register-scale and Video finite-epilogue variants did not
justify selection. The winning Video variant still uses separate finite
handling. Kitchen remains faster: selected Group32/Kitchen API ratios are
1.590× in screening and1.605× in confirmation. These ratios are matched in the
same rounds and are not extrapolated from local SM120 numbers.

### Runtime gates

All42 timed rank/configuration records complete:28 screening and14 repeat.
Four are native Kitchen comparators; all38 Group32 records have exact prepared
QI/KI/QS/KS/V/count identity and pass full-output rtol1e-3/atol2e-4.36/38 are
bitwise baseline-identical. Only the two rank records for the row-softmax
exchange variant change output rounding. Both ranks also pass all seven
synthetic ragged/GQA/packed-V/finite-output unit tests.

This is implementation conformance against the frozen Group32 approximation.
It is not a new dense-reference accuracy survey, sparse-quality result or
generation-distribution/semantics check. The separate SwiGLU/padding fusion
was implemented after this notebook was submitted and is **not** included in
these T4 results.

### Profiler evidence and limits

All five Systems reports export to SQLite and correlate to intended NVTX
scopes with zero unattributed kernels. The added completion barrier preserves
all three API iterations on both ranks in every capture. Its NCCL wait kernels
are outside the attention API and must not be reported as attention latency or
deployment communication cost.

Video preparation falls from7 to5 GPU launches per API. The replacement scope
contains a cast/pad kernel, a register-scale QKV preparation kernel, the same
two reductions and the leaf-count division. Current scope contains four
elementwise/conversion calls, two reductions and one shared-scale preparer.
The selected preparer has32/34 registers per thread and zero dynamic shared
memory; the current one has38/37 registers and128B dynamic shared memory. The
attention main-kernel resources are unchanged. Thus the observed gain is a
preparation optimization rather than a faster QK/softmax/PV schedule.

The separately timed Video preparation event medians fall from0.648/0.645ms
to0.444/0.440ms for ranks0/1. These are separate event intervals, not additive
components to subtract from complete-API medians. Profiled main-kernel times
vary strongly between captures despite identical main code; they are not
treated as controlled steady-state comparisons or evidence of an MMA gain.

DiT K-scale staging reduces actual launch register count221→215 but increases
dynamic shared memory37,888→38,912B at256 threads. It does not improve the
fresh-process repeat. On the complete three-call device0 capture, the main
kernel averages64.694ms current versus66.425ms staged; on device1 it is
60.615 versus61.923ms. The explicit layout-exchange softmax version regresses
27.52% in the unprofiled screen and is rejected. These measurements do not
support optimizing register count alone or adding shared-memory copies to
the default fragment layout.

TE has noticeable host/event variation. The register-only screen has samples
spanning0.559–1.178ms on rank0 and0.561–1.110ms on rank1. The cast/pad repeat
has0.526–0.717ms samples on rank0 versus current0.553–0.639ms; the distributions
overlap. No TE Systems capture was scheduled, so the cause is not established.
Keep TE cast/pad as a tentative candidate, not a stable quantified speedup.

No Nsight Compute counters were collected under the previously observed
Kaggle permission restriction. No memory-stall, instruction-throughput or
achieved-occupancy diagnosis is claimed from Systems durations/resources.

### Decisions

- Retain Video register-scale + cast/pad preparation as the useful second-pass
  candidate; keep its BQ64 main kernel and separate finite-output handling.
- Retain the first-pass DiT winner; do not select staged scales or row-softmax
  exchange on this evidence.
- Keep TE cast/pad tentative because its confirmed gain is small and variable.
- Preserve sparse drop/scalar semantics. These experiments were dense-only.
- Production hooks remain unchanged. No full H3 or concurrent-request run was
  performed, and no new GPU job was needed for evidence collection/analysis.

Version-bound evidence:
`D:/attention_eval/operator_optimization_tp2_20261017/kitchen_secondpass/v1/kitchen_group32_v2`.
It contains download/source hashes, raw timings and phase intervals, unit
gates, paired summary, telemetry, five Systems reports, SQLite exports and
process-qualified attribution. Recorded command durations total about7.52min;
this is not a provider billing/quota measurement.

## What changed

All experiments are isolated in `patches/tilelang_grouped_tp_kitchen_v2.py`.
The eight frozen first-pass source hashes remain unchanged.

- **Register scales:** keep the Group32 quantizer's reduced scales in fragment
  registers rather than staging them through shared memory. The existing
  generated code already had warp max reductions, but then crossed shared
  memory and CTA barriers to distribute scales. The second-pass preparation
  CUDA sources have zero explicit `__syncthreads()` sites. Division remains
  `__fdiv_rn`, rounding `__float2int_rn`, saturation[-127,127], scale floor1e-8.
- **K cast/pad fusion:** one TileLang kernel produces exactly the padded FP32
  representation of FP16-rounded K. The subsequent PyTorch64-key leaf sums,
  division by true counts and unweighted mean of leaf centroids are retained.
  This saves conversion/padding work without attempting a new reduction tree
  or changing centering. This is not a fully fused mean/quantization kernel.
- **Video finite epilogue:** exercise the existing FP16 finite-output store
  instead of a subsequent `torch.nan_to_num`. Test the epilogue independently
  and combined with register scales and cast/pad fusion.
- **K-scale staging:** load the current64-key leaf's Group32 scales into shared
  memory once rather than leaving their uses as global accesses in each group
  epilogue. The extra storage/synchronization may outweigh its benefit.
- **Row-local softmax experiment:** exchange completed score fragments into a
  row-owned fragment, reduce within warps, then exchange row reductions back.
  A direct annotation conflicted with the fixed MMA accumulator layout during
  CPU lowering. The explicit exchange version compiles, but pays extra shared
  memory work and changes the FP32 sum grouping; it is not assumed faster or
  bitwise identical. This does not yet implement a completely custom MMA/PV
  fragment layout.

The baseline is the **previous T4 winner**, not the original pre-optimization
operator: TE fused preparation/BQ32, DiT fused preparation/BQ64/Q reuse/
FullRow/256 threads, Video fused preparation/BQ64. Main-kernel shape and
schedule are held fixed for preparation ablations.

## Static and local checks

All12 target SM75 sources lower and compile on the CPU with no reported stack
or spills. Windows nvcc13.3 estimates are not Linux nvcc12.8 launch resources:
register preparation44 registers/thread on TE/DiT and32–34 on Video; cast/pad
12–16; DiT row-softmax189; DiT staged scales168. Full resource logs are retained.

The isolated SM120 preview uses Torch2.12.1+cu132, TileLang0.1.13 and
Kitchen0.2.33. SM75 guards/targets are retargeted only in generated copies.
The same process-local16-bit packing compatibility helper as the first local
preview preserves vectorized stores; the shared environment is not upgraded.

Logical rank0 completed14 screen cases, including the native Video comparator.
All13 Group32 cases have identical prepared operands and pass full-output
rtol1e-3/atol2e-4;12/13 are bitwise identical to the baseline. Only the
row-softmax exchange variant changes output rounding. Seven unit gates cover
ragged keys/queries, GQA, packed V, extended query padding and NaN/+Inf/-Inf
Video finite handling. The latter cases match `nan_to_num` byte for byte.

An independent local process repeated the selected candidates. These are
**local CUDA-event API medians, logical rank0**, with compilation outside
timing, and include preparation/output boundary work:

| Component | Screen current → selected (ms) | Fresh-process repeat (ms) |
|---|---:|---:|
| TE, register scales + cast/pad | 0.2920 → 0.2433 | 0.3241 → 0.2365 |
| DiT | 21.4559; no candidate selected over current | 21.1566 current |
| Video, register scales + cast/pad + finite store | 0.6192 → 0.5748 | 0.6094 → 0.5821 |

The local row-softmax candidate takes30.6802ms versus21.4559ms current; staged
scales takes22.4480ms. They remain in the small T4 screen because the first
pass demonstrated that an SM120 regression can become an SM75 improvement.
Separately measured phase intervals are retained in JSON and must not simply
be added: host launch gaps, allocation/cache state and drift differ between
phase and complete-API measurements.

The773-query Video logical rank1 partition also completes all six local
screen cases. All five Group32 outputs there are bitwise baseline-identical;
its combined candidate is0.5037ms versus0.5453ms current. These sequential
single-GPU partitions do not measure two-GPU latency or communication.

## Actual T4 experiment

Fixed screen: TE3, DiT5 and Video6 configurations per rank. No autotuning
sweep. Both ranks run their actual owned inputs: TE Q32/KV4,N353,D128 causal
GQA; DiT H28,N6814,D128 with packed V; Video Q1024/773 and global K1797,H32,D64.
Fixture SHA256 is verified before loading each component.

The private notebook requests720s provider runtime and has an internal660s
deadline, bounded dependency/screen/repeat/profile subprocesses and cleanup.
It pins Torch2.11.0+cu130, TileLang0.1.15, Kitchen0.2.37 and requires nvcc12.8.
The screen includes independent preparation and attention/output intervals,
plus complete APIs. T4 uses10 warm-ups/15 samples for complete API timing;
separate phase intervals use5/10. Each timed phase prewarms device clocks.
Finalists repeat in fresh processes against the same-round baseline.

Nsight Systems profiles the selected Video and DiT cases if budget permits.
The profiler adds a labelled rank-completion barrier after all three API calls
and before stopping capture, addressing the truncated peer-rank iterations
seen in the previous pass. This extra barrier is outside the attention API.
No Nsight Compute run is requested under the previously observed counter
permission restriction. No measured stall/occupancy claim is made.

Production hooks and sparse drop/scalar behavior remain unchanged. Sparse
schedule transfer is deferred until dense T4 results identify a useful layout;
this pass cannot establish generation semantics or distribution fidelity.

## Reproduction and evidence

- `benchmarks/build_kitchen_v2_sources.py`: checked generation from frozen source.
- `benchmarks/lower_kitchen_v2.py`: CPU target lowering.
- `benchmarks/kitchen_v2_matrix.py`: fixed matrix and resource pruning.
- `benchmarks/kitchen_v2_attention.py`: local/T4 numerical gates and timings.
- `build_tp2_kitchen_v2_job.py`: bounded source-hashed T4 notebook.
- `benchmarks/summarize_kitchen_v2.py`: paired hardware analysis after download.

Evidence root:
`D:/attention_eval/operator_optimization_tp2_20261017/kitchen_secondpass`.
It contains offline sources/cubins/resource reports, local preview and repeat
JSON/logs, submitted source hashes and `submission_audit.json`.
