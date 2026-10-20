# Attention compatibility on Turing / Tesla T4

Verified by installed-source inspection and primary upstream source on
2026-10-12. T4 is SM75. This check does not claim a physical T4 kernel
execution or a performance result, and does not change the running diagnostic.

| Entry | SM75 finding |
|---|---|
| ComfyUI `attention_sage` wrapper | Depends on the installed `sageattention`; exceptions and certain masked/low-precision-disabled calls use PyTorch attention. Wrapper success does not prove a Sage kernel ran. |
| Official SageAttention v2.2.0 / inspected main `sageattn` | Standard dispatcher has no SM75 branch and rejects it as unsupported. Official CUDA build targets begin at SM80. |
| Installed Windows SageAttention `2.2.0+cu130torch2.10.0andhigher.post5` | Explicit SM75 branch selects `sageattn_qk_int8_pv_fp16_triton`. This is the woct0rdho fork route, not an SM80 CUDA kernel. Actual execution still needs a working compatible Triton/compiler stack and input validation. Not tested on T4 here. |
| `comfy_kitchen/backends/eager/sol_attn.py` | PyTorch reference without an SM80 requirement; FP16 inputs are converted to FP32 for computation. Architecturally compatible with T4, subject to memory and input constraints. |
| Kitchen fused CUDA `sol_attn` / `sol_attn_chunked` | Requires SM80+; registry and direct-call checks reject SM75. ComfyUI sparse nodes test fused availability and retain dense attention when unavailable. |
| Kitchen dense `int8_attention` | Separate implementation; minimum capability is SM75 and the CUDA extension must load. A candidate for a later physical correctness/performance study, not evidence that fused Sol-Attn supports T4. |

The installed Kitchen package reports 0.2.33; the running Kaggle diagnostic
pins 0.2.37. The 0.2.37 tagged sources confirm the same Sol SM80 gate,
eager memory formula, and dense INT8 minimum capability.

The pinned ComfyUI wrapper also checks whether `attn_mask` is an explicit
parameter of `sageattn`. The installed Windows entry has `**kwargs` rather
than an explicit mask parameter, so masked calls fall back even though an
inner Triton function can accept a mask. This matters for masked TE calls.

Eager Sol forms the entire FP32 score matrix, costing `4 * B * H * S^2`
bytes before its other buffers. The source rejects a score tensor larger
than 4 GiB. The project's DiT block has 56 heads, hence 28 heads per TP2
rank: at B1/S8192, S16384 and S32768 the score tensor alone would cost
7, 28 and 112 GiB per rank. All three exceed that guard. Eager Sol is
therefore unsuitable for the recommended DiT workloads despite its
architecture compatibility.

Sol pools unrouted key/value blocks, and Sage/Kitchen INT8 attention
quantizes attention operands. Neither is automatically a Native bit-exact
replacement. A new backend must undergo its own accepted numerical/quality
validation; it must not be switched into the current causality comparison.

Primary sources:

- [Pinned ComfyUI attention wrapper](https://github.com/Comfy-Org/ComfyUI/blob/d49e888586dd8ae012c0667b33466b815fee07f7/comfy/ldm/modules/attention.py)
- [Official SageAttention v2.2.0 dispatcher](https://github.com/thu-ml/SageAttention/blob/v2.2.0/sageattention/core.py)
- [Official SageAttention build targets](https://github.com/thu-ml/SageAttention/blob/main/setup.py)
- [woct0rdho SM75 dispatcher](https://github.com/woct0rdho/SageAttention/blob/main/sageattention/core.py)
- [Kitchen 0.2.37 eager Sol reference](https://github.com/Comfy-Org/comfy-kitchen/blob/v0.2.37/comfy_kitchen/backends/eager/sol_attn.py)
- [Kitchen 0.2.37 CUDA Sol guard](https://github.com/Comfy-Org/comfy-kitchen/blob/v0.2.37/comfy_kitchen/backends/cuda/__init__.py)
- [Kitchen 0.2.37 dense INT8 attention](https://github.com/Comfy-Org/comfy-kitchen/blob/v0.2.37/comfy_kitchen/sage_attention.py)
