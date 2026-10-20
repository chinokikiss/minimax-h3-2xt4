# TP2 attention and transport candidate experiments

2026-10-18. Operator-only suite. No full H3 model or generation is run.
Kaggle kernel: `chinokiki9/h3-tp2-attention-transport-tradeoffs`, version 1.
Provider cap 840 s; internal deadline 780 s. Communication completed; attention
required the reference-runtime repair described below. Both sequential jobs
are finished with hardware-confirmed two-T4 allocations. The combined operator
experiment and evidence verification are complete.

Evidence root: `D:/attention_eval/operator_optimization_tp2_20261018/tradeoffs`.
The refreshed CPU/meta manifest verifies current TP2 shapes and hashes seven
production sources. Sixteen normalized UTF-8 source snapshots and the exact
notebook hash are archived. Existing kernels and production dispatch are unchanged.

## Fixed experiment matrix

- Communication: DiT O projection `(6814,5376,3584)` and FC2
  `(6814,5376,7168)`, with actual TP2 quantization/global FP32 MAX and native
  INT8 GEMM. Deterministic synthetic activations and weights; no real linear
  distribution claim. Current best packed quantization and fused SwiGLU/padding
  are common to all transport candidates. Full API uses native post-SUM
  dequantization, without adding optional bias/residual absent from these call sites.
- Dense attention: existing real native H3 captures, sliced into current TP2
  partitions: TE Q32/KV4, N353/D128 causal GQA; DiT H28/N6814/D128 with packed
  V stride; Video H32, Q1024/773 and global K1797/D64. Capture SHA256 and shapes
  must match. These are native captures, not fresh TP2 generation trajectories.
- Sparse attention: same DiT inputs, current tau0/leaf64 routing and prefix874.
  Both drop and scalar pooling remain available. No density/policy change.

Every candidate receives 10 identical distributed warm-up calls and 15 CUDA-event
and synchronized wall samples. Device clocks are prewarmed separately; compilation,
loading and gates precede timing. Fresh processes repeat the fixed matrix in reverse
order. Both physical GPUs must identify as T4/SM75. P2P remains disabled, four NCCL
channels are fixed and protocol selection is automatic. No architecture spoofing,
mock transport or forced rotation dispatch exists in the T4 harness.

## Candidate numerical contracts

1. **Scaled FP16 SUM.** Compute the current INT32 partial. Divide by a fixed
   power of two derived from `2*K*127*127/60000`, cast to FP16, execute one real
   SUM, restore the power of two and reconstruct INT32 before native dequantization.
   The bound leaves room for both ranks' same-sign maxima. No additional MAX
   collective is introduced. This mode loses accumulator precision, especially
   when large partials cancel; passing implementation gates is not quality approval.
2. **Packed INT8 exchange.** Each rank quantizes groups of 128 output channels
   independently using full signed INT8 range. A group packs 128 payload bytes
   and four FP32-scale bytes. One UINT8 all-gather exchanges the packed matrix.
   Both ranks reconstruct quantized rank0 plus quantized rank1 in that order,
   round to INT32, then run native dequantization. The local contribution is also
   quantized, preventing different replicas from mixing exact/approximate terms.
   This avoids INT8 SUM overflow/headroom loss and shared-scale MAX collectives.
3. **Group32 QK + INT8 PV.** Keep current QI/KI/scales and key centering. Quantize
   FP16-rounded V by key-leaf/channel into INT8 with FP32 scales. For each retained
   leaf, use its own logit maximum to quantize nonnegative probabilities to 0..127.
   Compute INT8×INT8→INT32 PV, restore V/P scales and the online-max correction.
   The denominator uses unquantized FP32 probabilities. Sparse scalar summaries
   retain their current FP16 probability/value path. V quantization remains a
   separate kernel in this initial ablation, and its cost is included in the API.
4. **Sorted compact drop.** Prefix-scan current binary routes into ascending
   retained-key-leaf indices. Traverse that list instead of scanning all leaves.
   This preserves original leaf order and must pass full output-byte equality
   against the same scheduled drop baseline. Packing time is included. Scalar
   pooling cannot skip omitted-leaf contributions and does not use this shortcut.

The INT8 packed transport payload is 25.78125% of the INT32 tensor size, including
scale metadata; FP16 payload is 50%. These are tensor-argument byte ratios, not
physical PCIe traffic or predicted speed ratios.

## Gates and profiling

Transport candidates must match independent PyTorch codec/reconstruction references
after real NCCL collectives. Dense INT8-PV units compare with an independent online
attention reference, including GQA/ragged/causal cases; V integers/scales are checked
against independent quantization. Empty/full/alternating/random routing patterns
verify compaction, including sorted tails and lengths. Full real-input runs verify
unchanged Group32 operands and routes. Approximate-mode arithmetic diagnostics are
recorded separately from performance and cannot establish generation quality.
Sparse-versus-dense L1/L2 is not a quality gate.

Offline Windows nvcc13.3 compilation passed all 12 SM75 sources with no reported
spills. The explicit local SM120 preview passed nine arithmetic/compaction cases;
it provides no T4 timing or hardware-performance claim. The prefix scatter triggers
a conservative static race warning because the analyzer does not infer prefix-sum
injectivity. For nonzero routes the inclusive prefix is strictly increasing, so
retained entries have unique destinations; pattern gates independently check this.
The checker has not been disabled.

Nsight Systems profiles FC2 transport alternatives, dense DiT/Video PV alternatives
and drop compaction. Captures contain three calls per rank, with a separately labelled
completion barrier outside API ranges. NVTX distinguishes codecs, collectives,
quantization, GEMM, V preparation and attention. Nsight Compute hardware counters
were previously restricted on Kaggle and are not retried. No achieved-occupancy,
memory-stall or generation-quality conclusion is made without supporting evidence.

Builder: `build_tp2_tradeoffs_job.py`; T4 harness: `benchmarks/tp2_tradeoffs.py`;
operators: `patches/tilelang_attention_tradeoffs.py` and
`patches/tilelang_transport_tradeoffs.py`. Status/results will be appended after
version-bound artifact collection and independent-process verification.

## Completed communication experiment

The initial job completed on two physical Tesla T4/SM75 GPUs. Communication
screen, reverse-order fresh-process repeat and three Nsight Systems captures
all succeeded. The overall initial job is correctly marked partial: the
attention unit's independent PyTorch `exp2` reference could not load
`libnvrtc-builtins.so.13.0` on this image. The failure is in the reference's
Jiterator runtime, not a candidate-kernel numerical gate. An attention-only
repair uses precompiled `exp(x*ln2)` in the independent reference, with identical
candidate kernels and tolerances. It does not repeat communication experiments.
Repair job: `chinokiki9/h3-tp2-attention-repair`, version 1, provider cap 600 s,
internal deadline 560 s. Only the harness source hash changes between jobs.

All communication implementation gates passed in both rounds and profiling;
all post-reduction output hashes agree across TP ranks and fresh processes.
Each approximate mode exactly matches its independently specified codec
reference, rather than the native INT32 numerical model.

Slower-rank complete-API CUDA-event medians, ms:

| DiT operation / transport | Screen | Repeat | Repeat reduction vs INT32 |
|---|---:|---:|---:|
| O projection / INT32 SUM | 46.2489 | 46.3120 | — |
| O projection / scaled FP16 SUM | 30.8490 | 30.9723 | 33.12% |
| O projection / packed INT8 exchange | 23.5024 | 23.7015 | 48.82% |
| FC2 / INT32 SUM | 56.9856 | 57.1989 | — |
| FC2 / scaled FP16 SUM | 42.5267 | 42.9045 | 24.99% |
| FC2 / packed INT8 exchange | 40.3932 | 40.6907 | 28.86% |

Synchronized wall reductions corroborate these: repeat O 33.12%/48.77%,
FC2 25.02%/28.84% for FP16/INT8 respectively. Comparisons are within this
allocation, not against timings from earlier sessions. Each API includes
unchanged global-row ConvRot INT8 activation quantization, native integer
GEMM, the chosen transport and native dequantization. No bias/residual is added.

Standalone transport also repeats consistently. FC2 medians including a
preserving input clone for **all** modes are 36.999 ms INT32, 21.305 ms FP16,
14.422 ms packed INT8. These separate intervals are not additive components
to subtract from the complete API. Actual full API has no preserving clone.

Arithmetic diagnostics on these synthetic A/W inputs are approximately
0.03064% output relative RMSE for FP16 and 0.6462% for packed INT8. These
figures are codec diagnostics, not real learned-linear or generation quality.
In particular they cannot be compared as an accuracy improvement over the
older learned-weight streamed INT8 experiment, which used different inputs,
shapes, scaling and reduction implementation. Cancellation and real outliers
remain unqualified. No automatic inference-policy change is made.

### Systems evidence

All three communication reports export to SQLite with zero unattributed kernels.
For rank 0, mean FC2 kernel durations per captured API call are:

| Phase | INT32 ms | FP16 ms | Packed INT8 ms |
|---|---:|---:|---:|
| Data collective | 35.388 | 17.853 | 9.494 |
| Encode/pack | — | 0.887 | 1.286 |
| Decode/reconstruct | — | 1.022 | 1.355 |
| FP32 activation MAX | 0.032 | 2.000 | 0.036 |
| Full-width FP32 rotation | 7.553 | 7.638 | 9.794 |
| INT8 GEMM | 7.300 | 7.587 | 9.008 |
| Native dequantization | 1.229 | 1.226 | 1.204 |

Rank 1 confirms data-collective durations 34.733 / 17.205 / 10.186 ms.
Each alternative replaces the one output-data collective with one smaller
collective; no additional scaling MAX is added. The original activation MAX
remains. MAX spans can include peer arrival waits. Rotation/GEMM variability
in the separate captures is retained; no improvement or regression of their
code is claimed. Profiler phase averages do not replace warmed API medians.

The 146,528,256-byte INT32 tensor becomes 73,264,128 FP16 bytes or 37,776,816
packed INT8 bytes per rank. The latter includes scales and uses all-gather
rather than a risky narrow integer SUM. These are collective tensor-argument
sizes, not measured physical link traffic. NCCL durations include waiting.

Communication evidence:
`D:/attention_eval/operator_optimization_tp2_20261018/tradeoffs/v1/tradeoffs`.
It contains raw sample distributions, both-rank records, independent gate
results, source hashes, telemetry, three Systems/SQLite pairs,
`trace_attribution.json` and `validated_summary.json`. The initial attention
failure log is retained rather than omitted from the record.

## Completed attention experiment

The attention-only repair completed on two physical Tesla T4/SM75 GPUs.
Setup stages took 14.61 s and 159.67 s; attention screen 181.34 s,
fresh-process reverse-order repeat 39.52 s and Systems profiling 75.76 s.
All three attention stages succeeded within the 560 s internal deadline.
The repair changed only the independent reference's exponential evaluation;
all candidate kernels, input shapes, routes and tolerances remained identical.

All 12 independent attention/compaction unit cases passed across both GPUs.
All 56 full-size timed/profiled rank/configuration records passed their
implementation gates, with zero failed candidates. Group32 prepared operand
and route identity checks pass. All six compact-drop records are bitwise
identical to the corresponding current drop operator. Every successful
configuration has stable output hashes across the two fresh processes for
each owned rank partition. Different attention TP ranks own different heads
or query positions, so their output hashes are not required to match each other.

Slower-rank complete API CUDA-event medians, milliseconds. Positive reduction
means faster; negative means slower. V quantization and route compaction are
included in the candidate's API.

| Family / candidate | Screen current → candidate | Screen reduction | Repeat current → candidate | Repeat reduction |
|---|---:|---:|---:|---:|
| TE dense / INT8 PV | 0.8801 → 0.6399 | 27.29% | 0.5696 → 0.6615 | -16.14% |
| DiT dense / INT8 PV | 63.6217 → 69.6470 | -9.47% | 68.1043 → 71.7988 | -5.42% |
| DiT drop / INT8 PV | 47.7696 → 48.1880 | -0.88% | 51.2939 → 51.2267 | 0.13% |
| DiT scalar / INT8 PV | 51.7137 → 49.9964 | 3.32% | 55.2733 → 52.6928 | 4.67% |
| Video dense / INT8 PV | 1.8638 → 2.0867 | -11.96% | 1.9149 → 2.0788 | -8.56% |
| DiT drop / compact sorted leaves | 47.7696 → 52.0827 | -9.03% | 51.2939 → 54.3534 | -5.96% |

Synchronized wall medians corroborate the repeat: scalar 55.3292 → 52.7465 ms;
dense DiT 68.1638 → 71.8539 ms; Video 1.9488 → 2.1107 ms; compact drop
51.3517 → 54.4131 ms. TE's apparent first-round gain disappears when its current
baseline changes 0.8801 → 0.5696 ms. INT8 PV stays near 0.64–0.66 ms and is slower
than the repeated current baseline; do not promote the first-round TE result.
No TE Systems capture establishes the jitter's cause.

The additional dense-output deviations versus current Group32/FP16-PV arithmetic
are 0.841–0.928% relative RMSE for the TE rank partitions, 0.666–0.708% for DiT
and 0.307–0.310% for Video. These compare numerical modes at one captured
boundary, not against native dense attention or generated outputs. They do not
qualify generation semantics or total model accuracy. Sparse quality is not
evaluated with dense-relative L1/L2; no such sparse-quality table is produced.

### Why the proposed attention shortcuts did not generally win

Six Systems reports contain all three intended calls on both ranks. The five
fully attributed reports and the valid API scopes in the sixth are usable.
One 102.493 us kernel in the Video-current report is outside every API scope,
after the measured calls, and is excluded from phase attribution. No captured
API kernel is missing its intended scope.

DiT rank 0 main-kernel means increase **60.509 → 65.686 ms** with INT8 PV;
rank 1 independently increases **63.113 → 68.310 ms**. V quantization is only
0.323 ms per call, while the common preparation remains about 4.99–5.00 ms.
Therefore fusing away V preparation alone cannot explain or fix this regression:
the combined QK/softmax/PV kernel itself is slower. Actual T4 registers fall
221 → 186 per thread and shared memory 37,888 → 25,600 bytes at 256 threads,
yet latency increases. Resource counts and narrower arithmetic are insufficient
performance predictors.

The new main loop performs block-local probability normalization/rounding and
per-block/channel scale restoration in addition to INT8 MMA. Those extra
operations are visible in source, but Systems cannot isolate their cycle cost
from MMA/layout/synchronization effects inside the fused kernel. No unavailable
stall or achieved-occupancy counter is inferred. This is an initial INT8-PV
implementation result, not a proof that an optimized INT8-PV kernel cannot win.

For compact drop, compaction itself takes only **0.0155/0.0163 ms** on ranks0/1.
The attention main kernel rises **40.037 → 43.596 ms** and **44.308 → 48.030 ms**.
Common preparation/routing is effectively unchanged. Thus packing overhead is
not the main regression: the indirect/dynamic traversal's compiled main kernel
is slower despite identical retained leaves and output bytes. Registers fall
199 → 157, while shared memory remains 24,576 bytes. Dependent index addressing
and changed loop code generation are plausible mechanisms; no measured memory-
stall attribution is available to distinguish them.

Video rank0 attention plus unchanged finite handling increases 0.984 → 1.063 ms,
plus 0.055 ms V quantization. Rank1's core behaves differently in this short
capture, so the repeated complete API, rather than one rank's profiled core,
determines the decision. The scalar candidate's modest gain is supported by
both unprofiled rounds; scalar was excluded from the fixed profiler shortlist,
so no scalar phase-level causal or occupancy claim is made.

## Decisions and next optimization targets

1. **Prioritize scaled FP16 communication for later quality validation.** It has
   stable 25–33% complete row-operator gains with smaller synthetic arithmetic
   deviations than packed INT8. It remains approximate and opt-in; real learned
   linear activations, cancellation/outliers and generation trajectories are
   not qualified by this experiment.
2. **Keep packed INT8 exchange experimental.** Its stable 29–49% row gains show
   substantial transport opportunity. It is a different, more approximate
   operator. Do not restore the retired INT8-SUM implementation or silently
   install this exchange into default production dispatch.
3. **Retain FP16 PV for dense TE, dense DiT, Video and drop.** This INT8-PV
   implementation does not demonstrate useful repeatable gains there. The
   scalar INT8-PV option remains available as a modest 3–5% numerical tradeoff;
   full-generation distribution/semantic validation is deferred.
4. **Reject compact traversal as the default.** It preserves tested arithmetic
   but regresses 6–9%. Keep the original route scan and existing drop/scalar
   choices. Higher sparsity and Group64/128 were not part of this fixed suite;
   their policies remain unchanged.
5. **Next fusion target is codec reconstruction plus dequantization.** Current
   FP16/INT8 codecs write a reconstructed INT32 tensor and native dequantization
   reads it again. A separately gated fused epilogue could remove that traffic
   and one launch. It has not been benchmarked here. A further INT8-PV pass
   needs main-loop/layout ablations rather than preparation fusion alone.

Smaller collective arguments do not establish lower peak VRAM. Current codecs
retain INT32 partials while allocating payload/gather/reconstruction buffers.
Whole-session communication telemetry peaks at 3,083 MiB per GPU, including
references and diagnostic temporaries; per-candidate peak memory and full-model
capacity were not measured. Nsight Compute remains unavailable under the prior
Kaggle counter restriction. No full-pipeline latency, concurrent-request,
generation-distribution, temporal or audio-quality result is claimed.

Repair evidence:
`D:/attention_eval/operator_optimization_tp2_20261018/tradeoffs/attention_repair/v1/tradeoffs`.
Both jobs have version-bound download manifests, source hashes, raw GPU/wall
samples, telemetry and original logs. Nine Systems reports and CPU SQLite exports
cover communication and attention. `benchmarks/summarize_tp2_tradeoffs.py` checks
both-rank records, independent-process output hashes and paired timings;
`benchmarks/attribute_tp2_tradeoffs.py` performs process-qualified NVTX attribution.
Current numerical modes remain explicit operator entry points; production
attention/reduction policy is unchanged.

The fixture dataset is version 1, last updated 2026-10-16, predating both jobs;
provider version information and the expected capture manifest are archived.
The downloaded repair metadata also records the provider container image digest.
`final_summary.json` combines qualified implementation/performance evidence
without treating approximate modes as generation-approved.
`analysis_manifest.json` hashes CPU analyses, source files and SQLite exports;
per-job download manifests independently hash the original provider artifacts.
