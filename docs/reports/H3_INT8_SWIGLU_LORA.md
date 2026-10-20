# Optional SwiGLU reduction and H3 Multi-LoRA

**Historical v1 qualification.** Streamed SwiGLU and FP32/INT8 base reduction
modes have been removed. The current default is Native INT32 with exact online
LoRA; RowParallel A partial reduction remains optional and Concat A is deprecated.
See [ROW_PARTIAL_MEMORY.md](ROW_PARTIAL_MEMORY.md).
The historical folded-scale/fused-B combination below is not restored.

Physical follow-up completed on 2026-10-18: see
[H3_PHYSICAL_INT8_LORA_RESULTS.md](H3_PHYSICAL_INT8_LORA_RESULTS.md).
Streaming saved temporary VRAM but increased latency; INT8 failed the strict
accuracy gate and provided no material extra VRAM saving over streamed FP32.
The tested offline merge passed; online concatenation still failed some gates.
The remaining measurements below describe the preceding **local** qualification.

Updated 2026-10-18. All new measurements below use one physical RTX 5060 Ti
with two logical CUDA ranks. **No new dual-T4 performance result is claimed.**

## User controls

`MiniMaxH3TP2Loader` now exposes two independent choices:

| Control | Choices | Default |
|---|---|---|
| `tp_quantization` | `scalemax`, `scalemax_streamed` | `scalemax` |
| `swiglu_reduction` | `native_int32`, `fp32`, `int8` | `native_int32` |

INT8 reduction is manual and opt-in. There is no automatic accuracy-changing
switch when memory pressure increases. FP32 remains available. The existing
exact INT32 GEMM-partial reduction remains the unchanged default.

To try lower-memory execution, select `scalemax_streamed` plus `fp32` first.
`int8` selects approximate communication for the SwiGLU **output partials**.
Other row projections retain their Native INT32 reduction. LoRA's rank-sized
hidden reduction remains FP32.

The tensor-parallel activation shards contain different features: summing
those INT8 activation shards directly would change the model. The correct
order is Native global row maximum → INT8 activation → local INT8 GEMM →
reduction of output partials. The optional communication mode introduces a
second quantization after that GEMM.

## Memory and accuracy

The streaming path uses two bounded FP32 SwiGLU/rotation passes, retaining INT8
activations and row scales between passes. It preserves global feature-width
rotation geometry and pads tail tiles to avoid the one-row kernel-route error
found during qualification. Optional FP32/INT8 output reductions also tile the
GEMM partials instead of retaining the entire INT32 partial matrix.

An ordinary full-range INT8 SUM can overflow. The opt-in implementation first
reduces shared per-row maxima, then clamps each rank to
`±floor(127 / world_size)`, and performs a real INT8 SUM collective. TP2 uses
±63 per rank, followed by FP32 dequantization. This prevents overflow at the
cost of about six effective bits per rank and one extra MAX collective per
tile. It does not offer independent full-range INT8 scales per rank.

Learned H3 block-0 FC1 → SwiGLU → FC2, S8192:

| Quantization / reduction | Temporary peak delta, MiB | Relative L2 vs Native FC2 | Native bytes exact |
|---|---:|---:|---|
| Existing scalemax / INT32 | 2752.00 | 0 | Yes |
| Streamed scalemax / INT32 | 1080.06 | 0 | Yes |
| Streamed scalemax / FP32 | 607.56 | 5.65e-8 | No |
| Streamed scalemax / INT8 | 607.57 | 6.88e-2 | No |

Peak deltas include two logical streams and the local collective emulator's
snapshot buffers. They are single-device diagnostics, **not per-T4 VRAM or
latency estimates**. Allocation order also affects smaller-shape peaks.
INT8 provides no material additional VRAM saving over tiled FP32 here. Its
accuracy cost is substantial; it is not a recommended quality preset.

Native learned projection outputs are byte-exact with streamed INT32 at
S257/512/8192. Random-input activation payload/scales were also byte-exact at
S33/257/512/513/2048/8192. Zero, cancellation and same-sign overflow-bound
collective tests pass. See `semantic_results/h3_optional_reduction.json` and
`semantic_results/streamed_swiglu.json`.

## Multi-LoRA implementation

Chain `MiniMaxH3TP2LoRAStack` nodes and connect the result to the loader's
optional `lora_stack`. Each entry supplies a local path and independent
strength. Negative and zero strengths are supported. Unknown tensor formats,
missing factors, incompatible dimensions, duplicate canonical targets and
invalid alpha are rejected.

The metadata reader preserves **per-module** alpha/rank. QKV and gate/up output
rows are partitioned component by component. Row projections read only the
owned A columns and replicate B; a single FP32 reduction of concatenated
rank activations precedes B, so replicated B is added once. Column projections
replicate A and read only owned B rows. Runtime factors are not merged into a
dense weight. B-projection temporaries are bounded to 64 MiB.

Adapter identities enter transformer/refiner cache keys without changing the
base-checkpoint identity used by existing offline shard manifests. Prepared
factor bytes and construction temporaries enter HostBudget admission. Refiner
factors enter auxiliary reservations. Changing a source file invalidates the
session; loader configuration signatures force service replacement on reload.

Native fused FC2 dispatch skips ordinary module-forward hooks. The refiner
therefore uses an explicit MLP wrapper that preserves fused base FC2 and adds
the SwiGLU LoRA bypass separately. Adapter buffers survive CPU/GPU shell copies.

### DMAD test identity and limitation

File: `ZhengmingYu/DMAD/minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors`.
Revision: `439136a1dd4f435b8137a084f73d558d2cecda2a`.
SHA256: `d3ee9f599ef1ac2d76850e90372ce9064fdc2c16e7f3de6a7ba9bc90cb91eb1b`.
208 adapted modules: four projections in each of 50 DiT and two refiner blocks.
QKV rank/alpha is 384; ordinary projection rank/alpha is 128. FC1 is already in
ComfyUI gate/value order. No TE LoRA is present.

The adapter metadata describes a separately trained critic backbone and T2VA
student. Matching tensor names against the public Ref2VA INT8 base establishes
loading mechanics, **not** that its intended four-step generation quality is
reproduced on this base.

### Online qualification: PARTIAL, strict full-block gate FAIL

Twenty actual projection tests across DiT blocks 0/25/49 and both refiner
blocks pass the existing numerical gate against upstream `LoRAAdapter.h`,
using two concatenated adapters with strengths +0.6 and −0.2. Relative L2 is
approximately 1.9e-7–6.1e-7. Packed row ordering and row-parallel B placement
are covered. Unadapted full blocks remain Native-byte-exact at S33/257/512.

However, concatenated online blocks have relative L2 2.28e-4–3.41e-4 and maximum
absolute error up to 8.96 against the independent sequential Native bypass.
They **fail** the existing 1e-5 / 0.005 gate. A single strength-1 adapter passes
S33/512 but fails S257 at 8.04e-5. The copied full refiner passes S37; S257 has
relative L2 8.29e-7 but maximum absolute error 0.00684 and also fails that gate.

Confirmed evidence: factors/projections are closely aligned; FC1 input to
quantization diverges and the resulting INT8 payload differs. The receipts
record changed payload counts and row scales. Candidate causes include
FP32 reduction order, concatenated GEMM geometry, and folding scale into B;
their individual contributions have not been isolated. No gate was relaxed.
Online concatenation is exposed as experimental, not accepted as Native
equivalent or promoted for physical performance benchmarking.

Receipts: `semantic_results/h3_lora.json`, `semantic_results/h3_lora_single.json`.

### Offline qualification: PASS within tested scope

`benchmarks/merge_h3_lora.py` creates a new checkpoint without changing or
uploading the original. It applies upstream Native FP32 sequential merges,
then Native deterministic **full-width** ConvRot requantization for INT8
weights. Dense refiner weights preserve their original BF16 storage dtype.
Quantizing row shards independently is prohibited: that would change scales
and invalidate the shared-scale INT32 reduction.

All four tested merged block-0 weight/scale pairs match an independent Native
merge/requantization byte-for-byte. A dense BF16 refiner out-projection also
matches independently merged bytes. Loading the merged block through Native
and TP gives byte-exact outputs at S33/257/512, for both scalemax variants.
This validates the **merged-model** reference, not equality to online bypass.
Full 208-target conversion and full-pipeline generation remain unverified.

Example full conversion (new output, sufficient disk required):

```powershell
python benchmarks/merge_h3_lora.py --base D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors --lora D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors 1 --output D:/minimaxH3/MiniMax_H3_DMAD_merged_int8.safetensors
```

Select the new file as `diffusion_model`, leaving `lora_stack` disconnected.
Old offline rank-local shards belong to the old source identity and must not
be reused for the merged checkpoint. Read the new checkpoint directly or
regenerate and independently validate its rank-local shards.

For Kaggle use, download and convert on Kaggle; no weights were uploaded from
this machine. Only block fixtures have been converted locally so far.

## Validation and remaining work

Reproduction scripts: `qualify_h3_reduction.py`, `qualify_streamed_swiglu.py`,
`qualify_h3_lora.py`, `qualify_offline_h3_lora.py` under `benchmarks/`.
The online qualifier intentionally exits nonzero while the strict gate fails.
Nine existing residency/identity/lifetime tests pass.

Physical dual-T4 NCCL support, accuracy at larger/full-model workloads,
latency, end-to-end memory and generation quality are pending. No new physical
job was submitted with the unqualified online implementation. Keep the existing
default for production; use streamed FP32 as an explicit local-tested candidate,
and treat INT8 communication and online concatenation as experimental.
