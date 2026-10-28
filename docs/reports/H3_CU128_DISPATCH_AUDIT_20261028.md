# H3 cu128 fallback audit and isolated repair candidates

The physical cu128 probe has returned: **27 executed candidates, one expected
FP32 constraint rejection, and one convolution adapter failure**. All 27 saved
candidate kernel traces were verified offline. CUDA13 is unnecessary for the
specific INT8 linear, FP16 RoPE and encoder norm/pad paths tested here. This is
an operator compatibility/performance result, not full-pipeline quality approval.

## Answer to the proposed one-line fix

`ck.registry.enable('cuda')` removes the frozen ComfyUI version guard in the
current process. It does not establish binary compatibility, force unsupported
dtypes/layouts through CUDA, eliminate CUDA implementations' internal fallbacks,
activate separate precision opt-ins, or propagate registry state into spawned
TP workers. It therefore cannot fully solve the listed issues.

Keep native Kaggle Torch 2.11.0+cu128, the frozen ComfyUI legacy API, Kitchen
0.2.37 and TileLang 0.1.13. No global enable is added to production startup or
the attention adapter. No dependency, source revision or default is upgraded.

## Verified source and evidence

The audit uses the exact Linux CP312-ABI3 Kitchen wheel approved for Kaggle
Python 3.13: SHA-256
`03cca0d10c314aaa13bff8f9422d03974221a344162b23d4f10d27eb2a35d0e8`.
It was downloaded for offline inspection, not installed locally. Its CUDA
Python backend matches the local wheel's normalized source byte-for-byte.
The production/frozen ComfyUI guard disables the CUDA registry when Torch
reports a CUDA runtime older than 13; Triton is independently off by default.

The `_C.abi3.so` SHA-256 is
`d19debb2fbc197aa26f73496ca30513386fae424a03e28ee2697afde1fb8f988`.
Its embedded PTX includes `.version 9.0`. ELF DT_NEEDED has no direct cudart
or cuBLAS dependency, while runtime lookup strings include `libcublasLt.so`
and `libcublasLt.so.13`. These strings are not proof that cuBLAS13 is loaded.
The new physical probe records actual CUDA library mappings, extension calls,
selected implementations and actual CUDA kernel events. Do not equate
`torch.version.cuda`, system nvcc and every loaded extension's runtime.

Earlier physical cu128 DiT runs already executed the private
`_C.cublas_gemm_int8` entry point successfully. This establishes that this
particular entry point works on the tested stack; it does not qualify the
entire public CUDA backend. The separate CUDA13 VAE results support testing
fused linears but do not prove that a cu128 registry override is safe.

## Pre-probe source audit and corrections to the proposed table

This table records the source audit that motivated the experiment. Physical
results and the remaining qualification limits are reported below.

| Stage | Verified cu128 behavior | Does global enable fully fix it? | Scoped candidate |
| --- | --- | --- | --- |
| Video decoder, 36 blocks x 4 INT8 linears | Public registry selects eager under the version guard. Eager still executes GPU INT8 GEMM and tensor operations. | It can change public dispatch, provided the CUDA extension, dtype, shape and library branches work. cu128 correctness/performance still needs physical qualification. | Explicit CUDA `int8_linear` alias, actual norm/SwiGLU/residual API, warm/error/kernel evidence. |
| Video decoder RMSNorm + partial RoPE | FP16 Q/K and partial dimension 48/head64 are eligible in the CUDA API; version guard prevents selection. | Potentially, for accepted layouts. In-place aliases and actual extension execution need testing. | Functional and in-place calls at the packed QKV stride, without changing attention. |
| DiT RMSNorm + RoPE | The tested model has **50 layers**, not 36. FP16_fix explicitly promotes Q/K to FP32 before RMS-RoPE. The pinned CUDA RMS-RoPE constraints accept only FP16/BF16. | **No.** It remains eager with FP32 inputs, including under cu130. | Preserve the FP32 island; a future dedicated fused FP32 implementation needs separate numerical validation. Never downcast just to pass dispatch constraints. |
| TE RoPE, 64 Qwen blocks | `llama.apply_rope` calls Kitchen split-half RoPE; TP2 Q/K have 32/4 local heads. | Potentially for FP16 and supported GQA strides. Parent registry mutation alone does not reach spawned ranks. | Explicit split-half alias with actual GQA geometry. |
| TE QKV / Gate / Up columns | `project_columns` intentionally shares eager rotation/quantization. `_dequantize_eager` uses one output destination and FP32 scale-first rounding. | It bypasses this shared preparation rather than simply fusing dequantization. Existing DiT column selection can also be bypassed by the same global change. | Reuse the existing eager-ordered TileLang epilogue independently, retaining shared quantization and GEMM. Require byte-exact outputs. |
| Video encoder GroupNorm + SiLU + padding | The public fused pad/norm path is reachable, but CUDA registry disable selects eager. Even the CUDA implementation can decline a layout/shape and use its internal eager helper. | Only for admitted shapes. | Probe C128/C1024, actual channels-last output, extension calls and numerical deltas. |
| Video encoder convolution | Native `_fp16_accum_conv` requires `_fp16_linear_wanted(x)`, including the separate FP16-accumulation opt-in. This opt-in is off by default. Ordinary convolution remains GPU PyTorch/cuDNN. | **No.** Registry enable does not activate the opt-in; CUDA `fp16_conv3d` can itself return to Torch convolution. | Diagnostic opt-in probe only. Record whether CUTLASS actually accepts the shape. Preserve the precision default. |

At 2392 TE tokens, even the widest 12800-output epilogue fits into the existing
256 MiB FP32 chunk budget in one chunk. The TE implementation has no final
chunk concatenation. Describing this path as an unavoidable many-iteration loop
plus concatenation confuses it with Kitchen's generic eager linear.

The selected DiT column route already executes private CUDA INT8 GEMM plus
TileLang dequantization for `[10752,5376]` / `[14336,5376]` weights. A blanket
enable changes the preceding implementation lookup and can route those calls
through public Kitchen CUDA linears instead. This is a policy/precision change,
not merely removal of the RoPE fallback.

## Placement and process ownership

`PhysicalTPService` uses spawn. An enable call in its parent constructor does
not transfer that registry state to rank0/rank1. A worker-side candidate must
be installed after selecting the rank's CUDA device and after ComfyUI imports
that apply the guard. The parent-side reference encoder also needs its own
explicit scope if it is later qualified.

`patches/kitchen_attention.native_scope()` currently swaps a proxy for
`comfy.quant_ops.ck` to adapt attention. TE and DiT also call the real
`comfy_kitchen` module directly. Adding a process-global enable inside this
scope couples unrelated operators, precision policies and background threads.
Nested exit restoration also cannot make a global registry flag thread-local.

## Implemented candidate and qualification plan

`patches/kitchen_cu128_candidate.py` registers an explicit candidate backend
with a narrow operation allowlist and the original pinned constraints. Its
alias is **not added to automatic priority**, and the original CUDA disable
flag remains intact. Only an explicit `Candidate.call()` selects it. There is
no runtime startup hook or production default change.

Five CPU contracts pass: normal automatic dispatch remains eager, an explicit
alias can select a permitted function, CUDA RMS-RoPE rejects FP32, and Native
convolution does not call the FP16 kernel without its independent opt-in. The
public linear wrapper's default BF16 output behavior is also preserved.

The bounded probe uses synthetic operands at actual deployment shapes, not
real-conditioning accuracy data. It downloads no models, runs no full H3
pipeline and uses no private credentials or Dataset mounts. Each unvalidated
case gets a fresh process; failures do not reuse a poisoned CUDA context.

It covers:

- VAE QKV/O/FC1/FC2 linears at M=1797 and 7188, including norm/SwiGLU API.
- TE K=5120 linears, especially M=96/97/127/128/129 and sensitive larger N.
- TE GQA split-half RoPE, VAE partial/in-place RMS-RoPE, and an expected
  explicit rejection of DiT FP32 RMS-RoPE even after a temporary diagnostic
  enable in the isolated child.
- Byte-exact TE dequantization with the existing TileLang epilogue at
  N=512/4096/12800, without changing activation quantization or GEMM.
- Encoder group norm/padding and the separately labeled FP16-convolution opt-in.

Results include CUDA extension calls/acceptance flags, actual profiler kernel
events, loaded library paths, five warmed measurements, peak memory, and error
against the existing eager operator. Finiteness is required and dequantization
must be byte-exact. Other numerical deltas are reported; they do not constitute
full-pipeline accuracy or generation-quality qualification.

CPU setup/API qualification precedes any T4 submission. Existing native-cu128
source and core pins are retained. No status/list/live-log polling is scheduled.
The physical override results are reported in the final section below.

CPU preflight version 1 passed at
[H3 cu128 Dispatch CPU 20261028](https://www.kaggle.com/code/pamcks/h3-cu128-dispatch-cpu-20261028).
It binds capsule `e91e0aba28007e0b996b7107a474d7a840eba683b0e9db5889c2fd44bb175b5d`.
The five operator contracts and all 179 CPU review tests passed. The native
CPU host used Python 3.13.15 and Torch 2.11.0+cpu, with Kitchen 0.2.37 and
TileLang 0.1.13. This qualifies imports, signatures and contracts; it does not
qualify the CUDA implementation. Returned source hashes match the published
capsule and the four added experiment files.

The one final-output collection transferred eight artifacts before an SSL EOF.
The remaining qualification receipts were recovered from the same signed final
artifact CDN URL using Windows curl. No second Kaggle API collection, status,
list or live-log request was made. The original interrupted collection receipt
is preserved alongside a separate final-verification receipt.

The GPU matrix contains 29 isolated cases. Its VAE cases retain the
Native per-output-channel scale shape, bias, and O/FC2 residual epilogue inputs;
encoder norm/padding inputs use channels-last storage. After both matching CPU
gates passed, version 1 was accepted at
[H3 cu128 Dispatch T4 20261028](https://www.kaggle.com/code/pamcks/h3-cu128-dispatch-t4-20261028)
on 2026-10-28 at 07:09:16 UTC (15:09:16 Beijing time). Acceptance is not evidence
that a T4 has started or that a candidate passed. The subsequently collected
physical results are reported below; production defaults remain unchanged.

Before submission, a local notebook guard incorrectly rejected historical
launcher filenames embedded in the CPU receipt's source-hash map. The guard
now examines AST string constants, allowing that JSON provenance while rejecting
executable launcher payloads. Direct and nested launcher rejection and receipt
acceptance were checked locally. The failed draft is preserved; that failure
occurred before the Kaggle push, so no duplicate GPU submission was created.

Sanitized qualification and submission evidence is recorded in
[`progress.json`](../data/h3_cu128_dispatch_20261028/progress.json).

Evidence root: `D:/attention_eval/h3_cu128_dispatch_20261028`.

## Physical T4 cu128 results

The returned hardware receipt confirms two distinct Tesla T4 devices, SM75,
driver 580.178.04, Torch 2.11.0+cu128 and CUDA 12.8, with Kitchen 0.2.37 and
TileLang 0.1.13. The operator supervisor exited cleanly after **79.84 seconds**;
setup took 85.61 seconds separately. The batch status is `operator_blockers_found`
because the convolution diagnostic failed before reaching its backend kernel.
The other 28 cases terminated as intended, including the FP32 rejection.

All 122 expected evidence files were recovered. The Kaggle CLI's one collection
again hit SSL EOF; remaining files were fetched from its existing final CDN URL.
No status/list/live-log polling or second Kaggle output API call was made.
All 27 successful cases have actual Chrome `cat=kernel` events, matching stored
kernel counts, names and durations. GPU annotations are counted separately.
The global CUDA registry disable flag remained unchanged.

The following are medians of five warmed CUDA-event measurements, in milliseconds:

| Operator / shape | Eager reference | Explicit candidate | Speedup |
| --- | ---: | ---: | ---: |
| VAE B1 QKV, M1797 / K2048 / N6144 | 8.239 | 1.821 | 4.53x |
| VAE B1 FC1, M1797 / K2048 / N16384 | 20.468 | 4.044 | 5.06x |
| VAE B4 QKV, M7188 / K2048 / N6144 | 24.996 | 6.597 | 3.79x |
| VAE B4 FC1, M7188 / K2048 / N16384 | 48.438 | 15.210 | 3.18x |
| VAE B4 output projection, M7188 / K2048 / N2048 | 11.617 | 4.146 | 2.80x |
| VAE B4 FC2, M7188 / K8192 / N2048 | 30.741 | 15.950 | 1.93x |
| TE split-half RoPE, 2392 tokens, Q/K heads32/4 | 1.850 | 0.714 | 2.59x |
| VAE RMSNorm + partial RoPE, 1797 tokens | 2.674 | 0.702 | 3.81x |
| TE dequantization, M2392 / N512 | 0.236 | 0.096 | 2.46x |
| TE dequantization, M2392 / N4096 | 1.383 | 0.305 | 4.53x |
| TE dequantization, M2392 / N12800 | 4.094 | 0.805 | 5.09x |
| Encoder norm + SiLU + padding, C128 | 1.221 | 0.346 | 3.53x |
| Encoder norm + SiLU + padding, C1024 | 8.734 | 1.494 | 5.84x |

All 20 tested public INT8 linears executed, including the TE M97/M127 boundary
cases at N4096/N12800. VAE linears span 1.93–5.06x and TE standalone linears
2.12–7.16x against generic Kitchen eager. These TE results do **not** compare
against H3's shared QKV/Gate/Up quantization and existing private GEMM policy.
They do not justify replacing that policy wholesale.

Profiler and extension evidence establish which paths actually ran:

- VAE QKV/FC1 each use two candidate kernels: fused ConvRot/norm/quantization
  followed by the SM75 INT8 TensorOp GEMM visitor. The extension reported
  `cutlass_turing_int8_dequant/returned_True` on all nine observed calls per case.
- Residual-bearing VAE output/FC2 use the cuBLAS INT8 path and separate epilogue
  operations: four/seven kernels respectively. Selecting CUDA does not fuse
  every residual/SwiGLU branch into one GEMM kernel.
- TE GQA RoPE executes two kernels; VAE functional RMS-RoPE executes one, with
  the in-place packed-QKV variant checked separately. Encoder norm/pad executes
  the three statistics/apply kernels through `_C.group_norm_silu_pad3d`.
- Each TileLang TE dequantization executes one kernel. At N12800, isolated
  peak allocated memory is 245,001,728 bytes versus 489,942,528 eager; these
  values include operands and the retained reference, not full TE residency.
- Library maps contain cuBLAS/cuBLASLt12 and cudart12, with no mapped
  cuBLAS/cuBLASLt13. This demonstrates the tested paths on cu128; it does not
  qualify every operation exported by the wheel.

All executed outputs were finite. The three TE dequantization cases are
**byte-exact** against the existing eager scale-first multiplication order.
Other candidates change rounding: VAE QKV/FC1 relative RMSE is about 0.275–0.276%,
VAE output projection about 0.165–0.169%, and FC2 about 0.020%. TE RoPE and VAE
RMS-RoPE have relative RMSE below 3.1e-6 on this fixture; norm/pad is about 0.046%.
These synthetic measurements identify numerical differences, not their impact
on real H3 video quality. The fused quantization/norm and GEMM/epilogue branches
were changed together, so their individual error contributions remain unisolated.

Two limitations are now established rather than inferred:

1. **DiT FP32 RMS-RoPE still selects eager after global CUDA enable.** Explicit
   dispatch rejects FP32 Q. Preserve the FP32 numerical island; this needs a
   separately validated FP32 fused implementation rather than a registry flag.
2. **Convolution remains unqualified.** The probe adapter incorrectly forwarded
   the public wrapper's `out=None` to the five-argument backend function. This
   was a diagnostic adapter bug, not evidence of a failing cu128 convolution
   kernel or a production ComfyUI API regression. The adapter now consumes
   `out=None`, normalizes stride exactly as the public wrapper does, and rejects
   the separate output-buffer variant. Six local CPU contracts pass, including
   binding to the pinned backend signature. The corrected adapter has not run
   on T4. Native's independent FP16 accumulation opt-in remains disabled.

The next integration priority is the byte-exact TE dequantization epilogue,
followed by scoped TE/VAE RoPE and encoder norm/pad dispatch with real-input
checks. VAE fused linears are promising but need captured-input/block accuracy
validation before production promotion. Keep the existing DiT linear route and
FP32 RMS-RoPE. A blanket `ck.registry.enable('cuda')` is still inappropriate.

This is an initial compatibility probe: two warmups, five samples, fixed
candidate-then-eager order, and both GPUs running separate cases. CUDA-event
elapsed times include possible host enqueue gaps; GPU clocks were not locked.
The separate profiler capture has different timing overhead. No Nsight Systems
or Compute capture, full decode, pipeline throughput or visual-quality result
is implied. Compilation and profiler/export work are outside warmed timings.

Full case samples, numerical deltas, extension counts and trace hashes are in
[`analysis.json`](../data/h3_cu128_dispatch_20261028/analysis.json), with the
29-case timing table in [`timings.csv`](../data/h3_cu128_dispatch_20261028/timings.csv).
`benchmarks/analyze_cu128_dispatch.py` regenerates these offline from the
preserved raw artifacts, without contacting Kaggle.
