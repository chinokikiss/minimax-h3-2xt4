"""Sequence-sharded block boundaries with quantization-preserving TP inside.

This repaired schedule gathers native activations. It makes no claim of the
legacy INT8-AllGather payload saving. Global modulation and RoPE stay global.
"""
import torch.distributed as dist
from patches.tp_dit import TPDiTBlockINT8
from patches.parallel_int8 import gather, quantize, int_mm, dequantize

quantize_int8_activation_convrot = quantize

def w8a8_gemm(q_act, scale_act, weight, weight_scale, out_dtype=None):
    import torch
    return dequantize(int_mm(q_act, weight), scale_act, weight_scale, out_dtype or torch.float16)

class SPDiTBlockINT8(TPDiTBlockINT8):
    def forward(self, x_local, t_emb, mod_segments, rope_freqs=None, transformer_options=None):
        full, start, length = gather(x_local, dist, 0)
        out = super().forward(full, t_emb, mod_segments, rope_freqs, transformer_options or {})
        return out[start:start + length].contiguous()
