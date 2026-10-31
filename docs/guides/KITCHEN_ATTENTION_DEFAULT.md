# H3 Kitchen attention default

The existing `kitchen_causal` TE and `kitchen` DiT selections share
`patches.kitchen_attention.int8_attention`. Query/TP Video attention also uses
this adapter. Native H3 Video decoding enters a scoped adapter around the
unchanged legacy attention API. FP32 inference writes the final FP32
accumulators directly to FP32 output. FP16/BF16 input calls use the pinned
Kitchen implementation and retain their existing output dtype.

## Direct output epilogue

Kitchen 0.2.37 quantizes Q/K/V and probabilities, keeps its online softmax and
output accumulators in FP32, then normally rounds FP32-input output through a
BF16 shared-memory/global-memory store and converts it back to FP32. That store
choice also applies to non-causal attention.

The H3 kernel preserves the quantizer, operand/scaling layouts, GQA ownership,
score/probability arithmetic and mask handling. Its epilogue maps each MMA
accumulator pair to its owned output row/channel and writes a `float2` directly
to global output. Stores use the original channel count rather than the padded
kernel dimension; odd channel counts use scalar stores for alignment and the
last channel. This removes the BF16 staging/store, its global temporary and
the following dtype conversion without an extra padded FP32 output. The public
result remains contiguous `[B,H,Q,D]` FP32 with the original strides.

The launcher follows the pinned key/query tile selection for unmasked, dense
masked, broadcast-key and prepared-key-mask calls. It preserves prepared tile
descriptors and padding/bias values. It retains original head-dimension padding,
original-dimension softmax scaling, compact KV heads and unequal Q/K lengths;
it does not expand a causal triangle or repeat KV heads in the adapter.

This removes output-store rounding only. It does not restore information lost
to INT8 operand/probability quantization, provide exact FP32 attention math or
repair floating-input causal prefix invariance. The H3 FP16 tensor flow remains
FP16; choosing the corrected Kitchen default does not select a different tensor
flow or increase the precision of existing FP16 attention outputs.

## Runtime and ownership

The installed Kitchen package and frozen ComfyUI source remain unchanged. The
adapter keeps the Kitchen keyword API (`scale`, `attn_mask`) and inference
boundary. Calls with gradient-bearing inputs delegate to the original API.
The direct FP32 backend targets SM75; SM120 supports local validation. Other
architectures receive an explicit qualification error for this FP32 path.

The source manifest verifies every retained Kitchen vendor header before
derivation. A build receipt binds the library to its source hashes, target,
compiler and binary SHA256. SM75 requires CUDA12.8.93. Building uses existing
tools without installing packages or replacing runtime libraries. Separate
architecture/source cache directories and an exclusive build lock allow ranks
to share one completed build. Corrupt or mismatched libraries are rejected.

FP16/BF16 calls do not look up or build this library. Legacy FP32 TP workers
prepare it at startup; other FP32 callers prepare it on their first call. Warm
the adapter before graph capture. An explicitly supplied library can be selected
with `H3_KITCHEN_FP32_LIBRARY`; it must pass the same receipt validation.
`H3_KITCHEN_FP32_CACHE` chooses the local build-cache parent.

The Native Video integration temporarily proxies `comfy.quant_ops.ck` while an
H3 decode is active. A context variable selects the corrected attention only
inside that H3 scope; other methods and unrelated threads delegate to the
original module. Nested/concurrent H3 scopes share the proxy, and the last
scope restores the original object. `comfy_kitchen.int8_attention` itself is
never replaced. The legacy `pre_norm`, `residual`, and `residual_scale` attention
signature remains intact. Audio remains Native single-GPU.

Sources: [shared adapter](../../patches/kitchen_attention.py),
[CUDA launcher](../../patches/kitchen_attention_native/attention.cu),
[CPU contracts](../../benchmarks/test_kitchen_attention_default.py),
[physical operator matrix](../../benchmarks/kitchen_attention_default_matrix.py).
Measurements and qualification status belong in the separate
[validation report](../reports/KITCHEN_ATTENTION_DEFAULT_20261027.md).
