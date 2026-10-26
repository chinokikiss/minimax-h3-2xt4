# Native Kitchen attention extension — 2026-10-25

**Implemented as an isolated candidate; physical T4 execution and performance
are not yet qualified.** Attention code generation uses CUDA C++ and nvcc,
without TileLang. The installed Kitchen package, source/core pins, frozen
ComfyUI legacy API and production operator policy are unchanged.

Implementation and usage: [experiment](../../experiments/kitchen_attention/README.md).
Evidence: [local receipt](../data/kitchen_native_attention_20261025.json).

## Causal and sparse execution

Kitchen's retained header contains `MaskMode::kCausal`, but its public Python
entry does not expose an `is_causal` flag. The new extension implements a routed
leaf64 iterator with token masking, GQA and explicit nonnegative query offsets.
This avoids requiring a token-pair mask, and allows sparse lists and causal
boundaries to be combined. Dense causal mode uses an implicit contiguous list.
Unselected leaves perform no QK or PV work. Lists are compacted and kept sorted
by a GPU warp-ballot kernel, without a CPU readback of route counts.

The existing noncausal tau=0/local/global/dense-prefix routing policy is ported
to ordinary PyTorch preparation. CPU comparisons establish exact route identity
on the tested ragged lengths and prefixes. INT8 Q/K/V, U8 probabilities, pinned
Hadamard/anchor quantization and FP32 softmax state remain the fine operator's
contract. This is not the old Group32 operator under another compiler.

Pooled mode uses a second native kernel with FP16 Tensor Core tiles of sixteen
centroids and FP32 softmax/output state. Means are formed from Kitchen's
dequantized packed operands, so fine/coarse logits share the same key-centering
shift. Count weighting handles the ragged final leaf. The kernel dequantizes Q
into shared memory and merges each coarse tile directly into the fine output;
it does not allocate the previous multi-GiB N-by-centroid score workspace, a
global Q-half copy or a full coarse-output tensor. Extra persistent pool state
is K/V centroids plus the fine LSE. Actual peak model-resident memory is pending.

This is an explicitly new pooled numerical approximation. It changes the
quantization basis, P/V precision and evaluation/rounding order relative to the
old Group32 Scalar path. Causal pooling remains rejected, since partial causal
leaves need a separate approximation. No generation-quality claim is made.

`is_causal` currently means score masking on packed operands. The original
quantizer's Q/K scale sharing, anchor selection and V scales can depend on
future floating-point inputs. Thus **strict end-to-end prefix invariance is not
established or claimed**. Fixing that requires a separate quantization contract.
The physical gate distinguishes this limitation from future-token isolation
with fixed packed operands.

## Dense T4 opportunities

Two independent candidates retain Kitchen's numerical contract and input
packing: a 64-query tile and an SM75 cached-Q specialization for D128/key128.
The default 128-query schedule remains a comparison baseline. Turing's register
and shared-memory constraints make tile size and accumulator lifetime credible
targets, but do not guarantee a latency improvement. See NVIDIA's
[Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html).

Local Windows CUDA 13.3.33/MSVC compilation gives:

| Specialization | Registers/thread | Spill store/load bytes |
|---|---:|---:|
| D128, FP16 output, query128/key128 baseline | 255 | 4 / 4 |
| Same, cached Q | 255 | 0 / 0 |
| D128, FP16 output, query64/key128 | 255 | 12 / 12 |
| D64, FP16 output, query128/key64 baseline | 255 | 0 / 0 |
| D64, FP16 output, query64/key64 | 192 | 0 / 0 |

These are exact-extension compiler results, not measurements of the installed
Kitchen binary or actual T4 occupancy. Cached-Q eliminates a small spill in the
local D128 FP16 compilation; smaller query tiles are not universally beneficial.
The benchmark excludes new spilling dense specializations from performance
selection using the exact library's build receipt. The original baseline is
retained even when it spills. Physical Linux compilation may change all of
these resource counts.

Other possible dense targets include reducing quantization launch/allocation
cost, balancing K/V staging with accumulator pressure and specializing the
measured TE/DiT/Video geometries. None is promoted based on source inspection
alone. SM75's retained copy helper uses synchronous vector loads, not Ampere's
asynchronous-copy hardware.

## Validation performed and remaining gates

**17 CPU tests pass**, with CUDA hidden. They cover route equivalence, sorted
compaction, ragged and offset causal masks, GQA scale layouts, empty rows,
pooled count normalization, common-key-shift invariance, fine/coarse state
merging, source identity, deterministic derivation, ctypes ABI layout and the
exact Kitchen API imports. Mathematical CPU oracles do not validate the native
GPU implementation. The native shared library and all SM75 specializations
compile successfully on the CPU. Compilation exposes two spilling dense FP16
specializations; it is not reported as a spill-free suite.

The local environment is prescribed conda Python, Torch 2.12.1+cu132, Kitchen
0.2.37 and TileLang 0.1.13. No package installation or environment change was
performed. The explicit build uses an already installed nvcc 13.3.33 and MSVC.
The Windows library must not be copied to Linux or described as qualification
of the approved native-Torch/cu128 Linux stack.

The new physical-only harness requires a T4, compares the unchanged Kitchen
baseline and exact dense candidates, validates masked/drop/pooled behavior,
then optionally times complete APIs in randomized and reversed order. It records
memory and both event/host distributions, supports private real captures and
both TP2 head partitions, and writes a failure receipt on rejection. Synthetic
inputs and operator L2 checks are not full-generation evidence. An external
process deadline is still needed for a GPU driver hang.

Linux CPU/bootstrap qualification and physical two-T4 correctness/performance
remain required before integration. **No GPU run, Kaggle submission/polling,
checkpoint download or full H3 pipeline was performed in this work.**

The retained upstream headers are hash-bound to official archive
`94d06941b4bb8e523008d29157a9c0f0522cd3e37472ad651fc13ffc6a55669c`.
They are copied unchanged with LICENSE/NOTICE; a guarded derivative is generated
only into a separate build directory. Earlier failed local build diagnostics
and successful compile artifacts remain under
`D:/attention_eval/kitchen_native_20261025/`.
