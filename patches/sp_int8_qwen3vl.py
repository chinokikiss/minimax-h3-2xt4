"""Sequence-sharded boundaries with native masks, global RoPE and repaired TP.

The revised schedule gathers activations, not legacy INT8 sequence payloads.
"""
import torch.distributed as dist
from patches.tp_qwen3vl import TPTransformerBlockINT8
from patches.parallel_int8 import gather
from patches.sp_int8_dit import quantize_int8_activation_convrot, w8a8_gemm

class SPTransformerBlockINT8(TPTransformerBlockINT8):
    def forward(self, x_local, attention_mask=None, freqs_cis=None):
        full, start, length = gather(x_local, dist, 1)
        out = super().forward(full, attention_mask, freqs_cis)
        return out[:, start:start + length].contiguous()
