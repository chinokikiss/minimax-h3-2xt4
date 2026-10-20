# DiT temporary tensor lifetimes: one unique block sequence

Here T means an **allocation/release boundary inside one TP2 DiT block**.
The block sequence is described once, not repeated for 50 layers or each NFE.
This is a source-derived tensor-storage calculation, not NVML or a new physical
peak measurement. No inference implementation has been changed.

**Historical ledger:** the worked IDs below predate the FC2 lifetime fix.
The current calculator defaults to `--padding-lifetime released`; pass
`--padding-lifetime before` to reproduce this table. The fixed ledger is in
`semantic_results/fc2_quant_memory/ledger_lifetime_fixed.json` and peaks at
1345.107422 MiB of block-created payload at S8192/U3, including AdaLN. The
separate optimization report records its validation and owned-rotation study.
For the runtime's default FC2 path, use `--fc2-rotation owned_fp32` in the ledger tool. Its ledger
is `semantic_results/fc2_quant_memory/ledger_owned_fp32.json`; FC2's reduction
moves this configuration's block payload maximum to the O projection. See
`FC2_QUANTIZER_MEMORY.md` for physical per-scope results and UI activation.

The reference ledger follows the current checkout: FP32, TP2, Kitchen 0.2.37
dense INT8 attention, packed row-linear and fused SwiGLU padding on SM75, two
DMAD entries, optional Row Partial enabled, selected FC1 B enabled. The current
source now contains operator dispatch and selected B integration; older reports
describe earlier source and cannot supply its exact liveness table.

DMAD ranks were checked in the actual safetensors header: QKV **384**, FC1/O/FC2
**128**, no AdaLN adapter. Do not use rank128 for its QKV hidden buffer.

## Calculation at any T

Each allocation j has a unique storage, byte size b[j], allocation boundary
birth[j] and last-owner release boundary release[j]:

`temporary_bytes(T) = sum(b[j] for j if birth[j] <= T < release[j])`

An allocation boundary includes the new storage; a release boundary excludes
the released storage. Views/aliases add zero. During a GEMM, its input and new
output coexist until the recorded input release. Different adapter iterations
are consecutive, so their B outputs are not summed across the entire stack.

This computes explicit named tensor payload bytes. To estimate total device
memory, separately add outer-live inputs/weights, persistent library buffers,
operator-internal workspace, deferred stream retirement, allocator rounding and
inactive reservations. None of those is silently treated as zero GPU usage.

## Shapes and units

S is the complete packed sequence, including reference/text/audio/video tokens;
U is the number of unique timestep rows. Per rank:

| Storage class | Shape / dtype | Bytes | S8192, MiB |
|---|---|---:|---:|
| Residual x (external baseline) | S x 5376, FP32 | 21504S | 168 |
| Each normalized hidden | S x 5376, FP32 | 21504S | 168 |
| Packed local QKV | S x 10752, FP32 | 43008S | 336 |
| Each local Q/K/V or attention result | S x 3584, FP32 | 14336S | 112 |
| Packed FC1 gate/up | S x 14336, FP32 | 57344S | 448 |
| Owned SwiGLU activation | S x 7168, FP32 | 28672S | 224 |
| FC2 full padding or full rotation | S x 14336, FP32 | 57344S | 448 |
| FC2 owned INT8 payload | S x 7168, INT8 | 7168S | 56 |
| FC2 INT32 accumulator or FP32 output | S x 5376 | 21504S | 168 |
| Row maximum / row scale, each | S x 1, FP32 | 4S | 0.03125 |
| One rank128 LoRA hidden | S x 128, FP32 | 512S | 4 |
| One rank384 QKV LoRA hidden | S x 384, FP32 | 1536S | 12 |
| AdaLN shared backing | U x 96768, FP32 | 387072U | 1.107422 at U3 |

RoPE's shared per-forward table is outside block temporaries: S x 48 x 2 x 2
FP32 = 768S bytes (6 MiB at S8192). It survives all blocks of that forward.
Cached Hadamard256 is persistent 0.25 MiB, not reallocated per projection.

## Unique execution order and release rules

| Operation | Newly allocated storage | Still retained / actual release |
|---|---|---|
| AdaLN | U x 96768 | All six modulation chunks are views; backing survives to block return |
| RMSNorm1 + modulation | S x 5376 | `h` aliases `normed_x`; in-place modulation adds no second Sx5376 buffer |
| QKV base ConvRot | rotated Sx5376, INT8 Sx5376, row scales | Rotation dies after quantization; q/scales survive base GEMM |
| QKV base GEMM | two Sx10752 outputs on CK Turing route | Eager unused output and helper output coexist; unused output dies on base-call return |
| QKV LoRA, each adapter | Sx384 hidden + Sx21504 full delta | Hidden dies after B; delta dies after in-place scale/add; local QKV remains |
| RMS/RoPE | new Q and K, each Sx3584 | Original packed QKV remains via `qkv` and V's view; replacing Q/K does not release it |
| Kitchen attention | Q8/K8/V8, scales, anchors, BF16 output, FP32 output | Original FP32 Q/K/V remain live; BF16 and FP32 output coexist during conversion |
| Attention layout | contiguous token-major Sx3584 | Transpose is a view, reshape requires a copy; old head-major output dies after dispatcher return |
| O row-linear | full padding/rotation, max, local q/scales, INT32 partial, FP32 output | All block attention tensors remain alive throughout; details below |
| O LoRA | A hidden + Sx5376 delta; or gather/reorder input | Output and original attention input coexist until O projection returns |
| Attention residual | no new residual storage | In-place add; then explicit `del` releases norm1, packed QKV, new Q/K, attention views and O output |
| RMSNorm2 + modulation | Sx5376 | `h2` aliases `normed_x2`, held through FC1 base and all its adapters |
| FC1 base | rotation/q/scales and two Sx14336 outputs | Same Turing duplicate-output lifetime; one packed output survives |
| FC1 LoRA | Sx128 hidden + selected Sx14336 or full Sx28672 delta | Hidden and delta briefly overlap; delta released before next adapter; norm2 only released after FC1 returns |
| FC2 base | padding/rotation, max, q/scale, INT32 partial, FP32 output | Packed FC1 output remains live in `fc1_out`, `mlp_act` and caller argument |
| FC2 Row Partial LoRA | activated Sx7168, small staged hiddens, Sx5376 B deltas | Activation dies after bounded hidden staging; packed FC1 output does not die |
| MLP residual + return | no new residual storage | FC1/FC2 outputs and AdaLN backing survive residual update, then die on block frame return |

### Attention's explicit buffers

With B1/H28/D128 and no attention mask, let C=128 for S>1024, otherwise64,
and Sp=ceil(S/C)*C. Kitchen allocates:

- Q8 and K8: 3584S bytes each; V8: 3584Sp bytes.
- Q scales: 28*ceil(S/128)*32*4 bytes.
- K scales: 28*ceil(S/C)*4*4 bytes; V scales: 3584*4 bytes.
- Anchor indices: 28*4 bytes.
- BF16 output: 7168S bytes; returned FP32 output: 14336S bytes.

They coexist at FP32 conversion; the package's frame return drops quantization
buffers/BF16 output. No dense SxS attention matrix is allocated by this path.
RMSNorm/kernel-internal scratch is a separate workspace term, not listed as a
known persistent Python tensor.

### Packed FC2: the caller reference matters

In `tilelang_swiglu_pad.global_row_quantize`, `padded` is a local variable passed
to `_quantize_padded`. The helper's `del padded,flat` only drops its own aliases.
The caller still holds its `padded` until the helper and caller return. Therefore
FC1 packed output + padded + full rotation + q + maximum + scale coexist.
Deleting the helper alias does not free 448 MiB at that point. This is a source
lifetime finding, not a newly measured T4 peak or an optimization change.

The O path in `tilelang_row_quant.global_row_quantize` differs: it actually
releases its own `padded` after rotation. Its input view still refers to the
attention output owned by the block. In both paths, q/scales and INT32 partial
remain live while dequantized output is allocated, then disappear on return.

### Row-LoRA branch changes the live set

Row Partial with two rank128 FC2 adapters stages 8 MiB of hiddens at S8192
(within the 32 MiB cap), replacing a 224 MiB activation before B. B deltas are
sequential, each168 MiB. With default Native-aligned row LoRA instead, gather
and reorder briefly require two full Sx14336 FP32 inputs (448 MiB each), while
packed FC1 (448 MiB) and base FC2 output (168 MiB) remain live. That is1512 MiB
before AdaLN/outer state. The calculator exposes both choices.

## Worked T example: S8192, U3, two DMAD entries, Row Partial + selected B

IDs refer to the supplied `S8192_partial_selected.json` / `.csv`:

| T | Boundary | Live block-created tensor payload, MiB |
|---:|---|---:|
| 108 | Release norm2/h2 after FC1 | 449.107422 |
| 110 | Allocate FC2 padded activation | 897.107422 |
| 111 | Allocate full rotation; padded remains owned by caller | 1345.107422 |
| 113 | Allocate per-row maximum | 1345.138672 |
| 115 | Allocate owned INT8 payload | 1401.138672 |
| **116** | **Allocate row scale: named-storage peak** | **1401.169922** |
| 117 | Helper frame returns: full rotation released | 953.169922 |
| 118 | Helper's maximum released | 953.138672 |
| 119 | Caller frame returns: padded finally released | 505.138672 |
| 120 | Allocate INT32 GEMM partial | 673.138672 |
| 122 | Allocate FP32 dequantized result | 841.138672 |
| 125 | Base FC2 returned; partial/q/scale gone | 617.107422 |
| 145 | Block return: only external x/outer state remains | 0 |

At T116 explicitly:

`448 FC1 + 448 padded + 448 rotation + 56 INT8 + .03125 max + .03125 scale`
`+ 1.107421875 AdaLN = 1401.169921875 MiB per rank`.

Residual x adds168 MiB if included; shared RoPE adds6 MiB. Neither should be
counted again for every block. This is not the older physical1344 MiB quantizer
result: that used a different row path/lifetime and excluded whole-block state.

## Calculator and complete ledger

`benchmarks/dit_tensor_ledger.py` generates the complete allocation/alias/release
sequence, storage shapes/bytes/birth/release, live sets at each T, and source
SHA256 bindings. It imports no torch and executes no model. Example:

```powershell
python benchmarks/dit_tensor_ledger.py --rows 8192 --time-values 3 --lora-ranks 128 128 --qkv-lora-ranks 384 384 --row-lora partial --padding-lifetime before --step 116 --output semantic_results/dit_tensor_ledger/S8192_partial_selected.json
```

Use `--row-lora gather` for Native-aligned row LoRA, `--no-selective-b` for
full FC1 B, or omit rank lists for no LoRA. Non128 FC1 ranks automatically use
full B, matching the current selection gate. `--row-backend native` selects the
older scalemax tensor sequence. T IDs depend on the chosen branches; use the
generated CSV for that configuration. Formula and live sets, not a wall-clock
timestamp, define T.

The table assumes uniform scalar modulation rows, FP32 and inference-only
execution. Per-token AdaLN indexing, sparse/native attention, different dtype,
graphs, custom patches, exceptional kernels and opaque library workspaces need
their own branch; this calculator does not pretend to measure those buffers.
An inner `del` permits reuse only after **all** owners and required CUDA stream
dependencies permit it. Reserved allocator memory is not released tensor payload.
