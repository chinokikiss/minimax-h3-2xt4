# H3 cu128 fallback audit and isolated repair candidates

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

## Corrections to the proposed table

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
Physical override results are pending; no new cu128 override speedup is claimed.

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
that a T4 has started or that a candidate passed. No physical results have been
collected yet, and production defaults remain unchanged.

Before submission, a local notebook guard incorrectly rejected historical
launcher filenames embedded in the CPU receipt's source-hash map. The guard
now examines AST string constants, allowing that JSON provenance while rejecting
executable launcher payloads. Direct and nested launcher rejection and receipt
acceptance were checked locally. The failed draft is preserved; that failure
occurred before the Kaggle push, so no duplicate GPU submission was created.

Sanitized qualification and submission evidence is recorded in
[`progress.json`](../data/h3_cu128_dispatch_20261028/progress.json).

Evidence root: `D:/attention_eval/h3_cu128_dispatch_20261028`.
