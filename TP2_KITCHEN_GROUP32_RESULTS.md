# Kitchen-inspired Group32 optimization on T4

2026-10-17. Implementation and bounded operator experiment following
`TP2_GROUP32_KITCHEN_DESIGN.md`. No full H3 pipeline or model-weight loading.
Sparse routes, drop behavior and optional scalar pooling are unchanged.

## Implemented candidates

`patches/tilelang_grouped_tp_kitchen.py` adds two independent ideas:

- Dense preprocessing combines Q quantization, centered K quantization and
  V conversion/padding in one TileLang launch. Its linear block grid accounts
  for actual Q/KV head counts, avoiding idle K/V head blocks for TE GQA. It
  reads packed DiT V strides directly. Group32 scales remain FP32, divisions
  use `__fdiv_rn`, saturation remains[-127,127], and input conversion still
  rounds through FP16. The existing eager leaf-centroid/mean calculation is
  retained unchanged; this is not a new centering approximation or mean kernel.
- The attention kernel adds128/256-thread and row/square GEMM warp partition
  choices. QK remains INT8 with32-channel scales, softmax/accumulators remain
  FP32 and P/V operands remain FP16. Query-tile options retain64-key online
  softmax order, GQA, causal masks, query offsets and output rounding.

The baseline files and production hooks are unchanged. The new main-kernel
body was statically compared to `tilelang_grouped_tp_optimized.py`: after
removing thread/GEMM-policy options, its math/control-flow AST is identical.
This is a CPU source audit, not runtime numerical or performance evidence.

The baseline named `current` means the current previously tested Group32
operator configuration, not an assertion that the H3 production service has
already switched all its attention calls to Group32.

## Offline pruning

Twelve target sources lower and compile to SM75 cubins on the CPU. The128-query
Video variant at128 threads reaches255 registers/thread on both ranks and is
excluded from GPU testing. The256-thread Video variants use roughly160–168
registers/thread; DiT64-query variants use167/187. No spills/stack frames are
reported by the local compiler. These local Windows resource figures are a
conservative screen and cannot replace Linux T4 launch resource measurements.

The surviving matrix contains11 component/configuration cases per rank:

| Component, actual TP2 boundary | Configurations |
|---|---|
| TE Q32/KV4,N353,D128, causal GQA | Current BQ32/group loads; fused preparation with that same main kernel |
| DiT H28,N6814,D128, dense | Current BQ32/Q reuse; fused preparation; fused preparation+BQ64/256 threads with row or square warp distribution |
| Video H32,Q1024/773,K1797,D64 | Current BQ64/group loads; fused preparation; fused preparation+BQ128/256 threads with group loads or Q reuse; Kitchen INT8 comparator |

No128-query sparse tile is introduced. No INT8 PV, anchor centering, whole-head
rotation, changed scale floor or faster quantizer division is introduced into
Group32. Kitchen is kept as a separate numerical backend and is excluded from
Group32 winner selection.

## Hardware measurement contract

Private Kaggle job `chinokiki9/h3-tp2-kitchen-group32`, version1. Provider cap
720s; internal deadline660s, bounded subprocess stages and process-group cleanup.
The job explicitly requires two TeslaT4 SM75 devices and pinned nvcc12.8,
Torch2.11.0+cu130, Kitchen0.2.37 and TileLang0.1.15. It reuses the existing
private real-activation dataset; no new capture upload or model input is needed.

The real native captures are sliced to the current TP2 head/query partitions;
they are not a fresh TP2 generation trajectory. Current TP source byte hashes
are checked against the verified shape manifest before submission. DiT V
preserves its packed local-QKV stride. Video inputs already contain global
K/V: this study excludes KV exchange, QKV production, RoPE, output linear and
the remainder of the decoder.

The unit gate includes FP32→FP16 conversion, ragged keys/queries, GQA, packed V
strides and padding beyond the original64-query alignment. Each candidate then
checks full prepared QI/KI/scales/V/count identity against current preparation at
the matching tile size. Complete outputs must match the current operator within
rtol1e-3/atol2e-4. Output-byte equality and SHA256 are recorded separately: a
warp-mapping rewrite is not assumed bitwise identical merely because its AST
has the same arithmetic. Kitchen has different numerics and is checked for
finite output, not Group32 output equivalence.

Each timed API prewarms clocks locally for at least1s, aligns the two ranks,
performs10 fixed warm-ups and records15 event/wall samples. Complete API
includes preparation and BND/native-dtype conversion/finite handling. Loading,
JIT and gates occur outside timing. Candidates run in a randomized order;
finalists repeat in fresh processes with another order and identical inputs.
Selection uses the slower rank's median. Both Group32 and Kitchen are measured
in the same Video batch and repeated, avoiding cross-allocation speed ratios.

For dense fidelity context, up to32 evenly spaced real query positions per
head attend to all keys using FP32 matmul/softmax with TF32 disabled. Causal
reference masking uses original query positions. Per-head relative RMSE and
max-absolute errors are recorded outside timing. Video ranks share heads but
partition queries, so combined statistics describe head/rank partitions. This
is a sampled dense operator diagnostic, not full-output accuracy, generation
semantics or a sparse approximation quality metric.

Nsight Systems prioritizes the Video current/finalist/Kitchen comparison, then
DiT current/finalist if budget remains. Hardware counters remain unavailable
from the earlier Kaggle restriction; no Nsight Compute retry is scheduled.

## Results

Both version1 jobs are now complete, with hardware-confirmed **2×Tesla T4,
SM75**, nvcc12.8 and Torch2.11.0+cu130. The second job is
`chinokiki9/h3-tp2-kitchen-group32-retry`; its notebook is byte-identical to the
original, SHA256
`bdc10ec954bcaf190a6eb594f3e44ab097c980cea70b5bdc63c861f26d2ab4cb`.
All source hashes match the submitted manifest. Screen, fresh-process repeat,
Video Systems capture and DiT Systems capture stages succeeded in both jobs.
No candidate failed. No new GPU job was needed to collect or analyze evidence.

Each value below is the slower rank's median API CUDA-event interval, in ms.
Ranges of savings describe paired rounds, not cross-instance ratios. They
include preparation and output-boundary work, with JIT/loading excluded.

| Job / round | TE current → fused | DiT current → q64_row256 | Video current → fused | Kitchen Video |
|---|---:|---:|---:|---:|
| Original screen | 0.7090 → 0.6110 | 70.4819 → 65.7477 | 2.0987 → 2.0290 | 1.2079 |
| Original repeat | 1.1244 → 0.6172* | 74.0705 → 68.1498 | 2.1678 → 2.0931 | 1.2039 |
| Retry screen | 0.6723 → 0.5858 | 70.2648 → 65.9990 | 2.1121 → 2.0441 | 1.1639 |
| Retry repeat | 0.6641 → 0.5728 | 73.2620 → 67.3995 | 2.1977 → 2.1135 | 1.1947 |

TE consistently benefits from preparation fusion: three paired rounds give
12.87–13.83% reduction. The starred original-repeat baseline is unstable:
rank0 samples span0.659–1.288ms and rank1 spans0.866–1.153ms. Its apparent
45.11% saving is retained in raw evidence but is not used as the expected
gain; no TE profiler trace establishes the cause of that variation.

DiT's fused preparation alone improves the screen by2.01–2.53%. Combining it
with BQ64, cached Q,256 threads and FullRow GEMM partition improves all four
paired rounds by6.07–8.00%. The square-warp version gains only0.44–0.47% in
screening. **The useful row-warp variant regressed on the local SM120 preview
but wins on T4**, demonstrating that local candidate rankings are not a safe
SM75 selection criterion.

Video preparation fusion improves all four rounds by3.22–3.83%. Both BQ128
variants regress by6.65–10.39% on T4 and should not replace BQ64. Kitchen is
still substantially faster: the best Group32 API is1.68–1.77× the matched
Kitchen latency across these rounds. No claim of Kitchen-level speed is
supported by this pass.

### Runtime numerical gates

Across both jobs there are72 successful timed rank/candidate records, including
8 native Kitchen records and64 Group32 records. Every Group32 record passes
full prepared QI/KI/QS/KS/V/count identity at matching tile geometry and the
full-output tolerance gate.52/64 Group32 records are bitwise identical to the
current baseline. The three selected configurations are bitwise identical on
both ranks in both rounds of both jobs. Square-warp / Video BQ128 alternatives
can change output rounding; they are not mislabeled as bitwise equivalent.
Both two-case ragged/GQA/packed-V unit gates also pass in each job.

Video's sampled median head/query-partition relative RMSE is0.09365% for
Group32 versus0.13584% for Kitchen, while the worst is0.68224% versus0.68114%.
This is not evidence that Group32 is uniformly more accurate. No new
generation distribution, temporal, semantic, audio or sparse-quality claim is
made; all selected changes preserve the existing Group32 output on these
tested cases.

### Profiler evidence and remaining bottlenecks

Ten Systems reports were downloaded, exported to SQLite on the CPU and
attributed using process-qualified CUDA correlation IDs. All captured kernels
map to the intended NVTX scopes, with zero unattributed kernels. Profiling
does not replace the unprofiled API timing table above.

The retry DiT device1 capture contains all three attention calls for each
variant. Its preparation GPU launch count falls from13 to8 per API;
preparation kernel-duration sum falls from5.837 to4.983ms per API. The
QK/softmax/PV/output kernel mean falls from64.380 to60.150ms. The original
device1 capture independently gives64.673→61.263ms for the main kernel and
5.847→4.985ms for preparation. This identifies both preparation traffic/launch
removal and the attention schedule as useful changes; the main attention
kernel remains the dominant GPU work.

The new fused preparation kernel itself takes about2.054ms per API in retry
DiT, compared with about0.966ms for the two old quantizer launches combined.
Those scopes perform different work: the new kernel also casts/pads packed V
and absorbs external input conversions. The total preparation phase improves;
timing only the named quantizer kernel would give a misleading conclusion.
Eager K-centering reductions still remain and are a future fusion opportunity.

Actual Linux T4 launch resources are220 registers/thread,128 threads and
28,160B dynamic shared memory for current DiT, versus221 registers/thread,
256 threads and37,888B for the row-warp candidate. These differ from local
offline compiler estimates. High register demand is observable, but these
resources alone do not establish measured occupancy or a register-stall cause.

Video's complete current scopes have10 preparation launches plus attention
and finite-output handling per API; complete fused scopes have7 preparation
launches. Some multi-rank captures contain only2 of3 final API iterations
because the process-wide capture ends while the peer rank is still finishing.
Kernel counts are checked before normalization; incomplete scopes are not
divided blindly by3 or treated as matched steady-state latency evidence.
Video traces support launch removal, but do not isolate QK, softmax and PV
inside the fused kernel or establish the full cause of the gap to Kitchen.

Nsight Compute counters remain unavailable under the previously observed
Kaggle permission restriction. No new NCU run was attempted in these jobs.
There is no measured memory-stall, instruction-pipeline or achieved-SM-
occupancy conclusion. Systems timeline coverage is not SM utilization.

### Decisions and evidence

Retain TE fused preparation, DiT q64_row256 plus fused preparation, and Video
BQ64 plus fused preparation as exact-shape T4 operator candidates. Production
hooks and existing sparse drop/scalar behavior remain unchanged. This pass
does not measure KV exchange, PCIe/NCCL, linear/GEMM fusion, concurrent request
throughput or full-pipeline latency. DiT sparse variants require their own
operator-level scheduling measurements before borrowing the dense winner.

Version-bound evidence folders below each contain download manifests,
paired summaries, selected candidates, input/source hashes, raw sample
timings, telemetry, Systems reports, SQLite exports and attribution:

- `D:/attention_eval/operator_optimization_tp2_20261017/kitchen_inspired/v1/kitchen_group32`
- `D:/attention_eval/operator_optimization_tp2_20261017/kitchen_inspired/retry_v1/kitchen_group32`

No whole-pipeline generation was performed. Recorded stage-command wall time
is about6.62min for the original and6.19min for the retry; this is not a
provider quota/billing measurement.

## Reproduction and local evidence

- `benchmarks/lower_kitchen_attention.py`: CPU code generation.
- `benchmarks/compile_grouped_tp_offline.py --input <offline-directory>`: CPU cubin/resource checks.
- `benchmarks/gate_kitchen_attention.py`: conservative fixed-shortlist pruning.
- `benchmarks/audit_kitchen_precision.py`: normalized main-kernel AST check.
- `build_tp2_kitchen_attention_job.py`: source-hashed private notebook builder.
- `benchmarks/tp2_kitchen_attention.py`: hardware conformance/timing harness.
- `benchmarks/summarize_kitchen_attention.py`: CPU paired analysis.

Evidence root:
`D:/attention_eval/operator_optimization_tp2_20261017/kitchen_inspired`.
