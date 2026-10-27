# TE attention and TP2 communication — 2026-10-27

Status: [private T4 job](https://www.kaggle.com/code/pamcks/h3-te-attention-tp2-matrix-20261027)
version 1 failed in an extra mask control after completing all S128 points on
both physical T4s. Its final artifacts were collected once. The corrected
version 2 has been accepted; its final artifacts have not been collected.
**The full matrix is pending; no overall TE speedup conclusion is claimed.**

## Version 1 failure and bounded recovery

The S353 padding diagnostic constructed an FP32 additive bias for FP16 Q/K/V.
The pinned T4 Native SDPA backend rejected it with
`invalid dtype for bias - should match query's dtype`. The primary matrix
already used the Native FP16 finite causal mask; the mismatch was introduced
by the extra benchmark control. Production code and library versions are
unchanged.

The correction constructs additive padding masks in the query's dtype, with
the Native finite sentinel `finfo.min/4`. New CPU tests verify the dtype and
resume validation. All eight harness tests pass, together with four local
Native mask smoke cases covering Q64/KV8 and Q32/KV4, boolean and FP16 additive
masks. The local smoke is an API check, not T4 performance validation.

All six S128 configurations per owner were complete, finite and matched the
required exact collective-replay contract. Their TP2 output hashes also agree
between owners. Version 2 therefore executes **only S353, S1306 and S4096**:
18 configurations per owner, with no duplicate S128 measurements. Existing
points retain the original allocation's GPU identities; the retry records its
own allocation. Within-shape speedup comparisons stay paired on the same
hardware allocation. Cross-allocation differences are not attributed to this
mask fix. Performance checkpoints are now saved before extra mask diagnostics.

The [attempt evidence](../data/te_attention_tp2_matrix_20261027/manifest.json)
retains version 1 sources, per-rank results, traceback, collection receipt and
the version 2 submission/qualification. There was one final collection and one
new version submission in response to the reported failure, with no polling.

The question is whether the current TE's smaller per-rank compute is outweighed
by real NCCL communication. Historical scripts estimated transport time; those
estimates are not evidence for the current operators. Older profiles also show
NCCL waiting for weight-readiness skew. This experiment removes weight I/O from
the timed region to test compute and communication directly.

## Matrix

| Item | Contract |
|---|---|
| Attention backends | Every current TE policy choice: Native SDPA (`native`), TileLang Group32 Masked (`tilelang_group32`), `kitchen_causal` |
| Unsharded geometry | B1, hidden5120, Q64/KV8, D128; intermediate25600 |
| TP2 geometry per rank | Q32/KV4, intermediate12800; replicated residual5120 |
| Tokens | 128, 353, 1306, 4096 |
| Masks | Native FP16 finite additive causal `[S,S]`, sentinel `finfo.min/4`; separate boolean/additive padding controls at S353 |
| Weights | Trained TE block0, two public immutable rank shards; no DiT/LoRA/VAE weights |
| Tensor flow | Current FP16 fix, divisor64; FP32 SwiGLU/quantization and current fused preparation / Scaled FP16 row transport |
| Inputs | Identical seeded FP16 residuals, normalized through the trained block; these are not captured H3 activation distributions |
| Sampling | Three warm-ups and seven repeated measurements per operation/configuration |
| Scope | Attention operators and one resident language-transformer block; no full TE, vision tower, embedding, weight-streaming or H3 generation |

The length sweep covers short and longer prefill. S1306 also matches the text
conditioning span recorded by the last H3 E2E; that span does not independently
establish the original Qwen multimodal input length. The sweep is a shape study,
not a claim that every length was captured at a TE input boundary.

Each rank runs an unsharded single-GPU control using weights reconstructed from
the two trained shards. Column weights/scales concatenate output rows; row
weights concatenate input columns; replicated row scales/norms must be equal.
The single-GPU controls execute one at a time, coordinated with CPU Gloo
barriers, so two concurrent full-block requests do not contaminate that baseline.
Both single and TP2 models remain resident in the harness; allocator peaks
therefore include the harness's two control models, not a complete TE deployment.

TP2 uses the current production block implementation and real NCCL collectives,
with `NCCL_P2P_DISABLE=1`, four channels and the default protocol selection,
matching the runtime. Loading, compilation/warm-up, output validation and
diagnostic capture are outside steady-state timing. Wall time and CUDA-event
spans include launch gaps and synchronization at each sample; they are not
exclusive kernel times or a 50-layer latency estimate.

## Causal communication control

An untimed TP2 pass records each collective's exact post-operation GPU result,
shape, dtype, reduction and order. A separate replay substitutes GPU copies of
those recorded results while keeping the block input fixed. It must reproduce
the actual TP2 output byte-for-byte and consume every recorded collective.
Real NCCL is absent from that replay; the copies, replay cache and Python
dispatch remain. This is an attribution control, not a deployable communication
replacement or a quality approximation.

The batch records seven real-TP2 samples and seven replay samples independently.
It reports both physical single-GPU baselines, the slower TP2 owner's median,
the slower replay owner's median, and the actual collective inventory. A speedup
below one against both single-GPU controls indicates a resident-block TP2 loss
for that backend and shape. A replay that remains slower than the single-GPU
control points to costs beyond communication alone, such as duplicate work,
different GEMM geometry and launches. Between-device differences are retained
rather than hidden in an average.

No sum of per-rank wait times, event spans or nested CPU durations is treated as
removable request latency. Collective operand bytes are not physical PCIe
traffic. Results cannot decide the full encoder's offload/residency policy:
27.14GB of TE weights cannot all reside on one T4, and streaming/readiness costs
remain a separate deployment question.

## Numerical and profiler controls

Attention outputs are compared with a bounded-query FP32 QK/softmax/PV oracle
using the same Q/K/V and TF32 disabled. INT8 weights retain their stored types.
Full-block comparisons use the same-backend unsharded output and a separate
Native-SDPA FP16-flow control. Those block controls are not labeled native FP32
tensor-flow ground truth. No generation/semantic quality result is inferred.

Actual and replayed TP2 outputs must be byte-equal. Both TP2 owners must also
return identical full-block output hashes. The padding controls preserve the
complete mask, and all numerical differences are retained.

A separate warmed S353 Kitchen TP2 profile first captures a minimal CUDA add,
exports/flushes the trace and verifies CUDA kernel events on both owners.
Only after that shared coverage gate does it capture actual collectives and
exact-result replay. Missing CUDA/NCCL coverage is recorded as a blocker.
Steady-state timings do not include profiler instrumentation. Nsight Systems
and Compute availability is recorded; a Torch profiler trace is not presented
as Nsight evidence. The parent checks both completed rank records and the
distributed launcher's exit code, together with the bounded supervisor's
termination result.

## Preserved setup and delivery

The passing frozen CPU environment/API receipt and immutable source bundle from
the VAE comparison are reused. All ten relevant TE runtime/operator source
hashes are checked against that bundle; none changed. The new three-module
experiment is embedded as separately hashed supplemental source, with eight CPU
contract tests passing on those exact bytes before submission. The Temporal
default promotion is separate: this TE-only job never loads a VAE.

The current runtime/default regression suite passed 179 tests. The local
unsharded Native/Kitchen smoke used a 5060 Ti only to check execution/finite
outputs; it does not qualify T4 timing or the SM75 Group32 path. The two local
trained shard files match the public files byte-for-byte. Kaggle fetches just
those 487,960,752 bytes from `cnmds/minimax-h3-tp2-int8-561a6211`, revision
`d3e6a4b131056d2c626a01acf6ba48123a885783`, with `token=False`.

Core pins remain Kitchen0.2.37 and TileLang0.1.13; the native Kaggle Torch/CUDA
family is retained and recorded. There are zero dataset/output mounts and no
private credentials. One bounded submission uses a 300-second setup cap,
900-second experiment cap and 1500-second notebook limit. No status/list/log
polling or automatic replacement is configured.

Sources: `benchmarks/build_te_parallel_job.py`, `run_te_parallel_batch.py`,
`te_parallel_matrix.py`, `test_te_parallel_matrix.py`.
Evidence: `D:/attention_eval/h3_te_parallel_20261027/`.
Machine-readable [status and source bindings](../data/te_attention_tp2_matrix_20261027.json).
