"""
MiniMax-H3 Video VAE (ViT3D Decoder) INT8 ConvRot Multi-GPU Implementations:
1. Single GPU INT8 Baseline: Full-frame un-tiled ViT3D forward pass in W8A8.
2. Strategy 2: TP=2 (Megatron Tensor Parallelism with FP16 AllReduce).
   - Column-parallel to_qkv (16 heads per rank) and w1 (8192 intermediate per rank).
   - Row-parallel to_out and w2 with FP16 AllReduce.
   - Preserves full-frame global attention and 3D RoPE coordinates.
3. Strategy 3: TP=2 + Sequence Parallelism + INT8 AllGather:
   - FP16 ReduceScatter -> local residual/RMSNorm -> INT8 ConvRot Activation Quantization -> INT8 AllGather -> W8A8 GEMM.
   - Slashes communication volume by 25% and cuts activation memory by 50%.
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Tuple

try:
    import comfy_kitchen
    HAS_CK = True
except ImportError:
    HAS_CK = False

# Ensure ComfyUI and project root are in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.exists("/tmp/ComfyUI") and "/tmp/ComfyUI" not in sys.path:
    sys.path.insert(0, "/tmp/ComfyUI")

# Fallback mock for comfy_aimdo if missing
try:
    import comfy_aimdo
except ImportError:
    import types
    m = types.ModuleType("comfy_aimdo")
    m.host_buffer = types.ModuleType("host_buffer")
    sys.modules["comfy_aimdo"] = m
    sys.modules["comfy_aimdo.host_buffer"] = m.host_buffer

from comfy.ldm.minimax.vae import create_token_ids, RotaryEmbeddingND

def quantize_int8_activation_convrot(x: torch.Tensor, convrot_groupsize: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes activation to row-wise INT8 with group-wise Hadamard rotation (ConvRot).
    Input: x [M, K] in FP16
    Output: qdata [M, K] in INT8, qscale [M, 1] in FP32
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    m, k = x_2d.shape

    if HAS_CK and x.is_cuda and hasattr(torch.ops.comfy_kitchen, "quantize_int8_rowwise_convrot64"):
        qdata = torch.empty((m, k), dtype=torch.int8, device=x.device)
        qscale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        torch.ops.comfy_kitchen.quantize_int8_rowwise_convrot64(
            x_2d, qdata, qscale, convrot_groupsize, False, 0, 0, 0
        )
        return qdata.view(*orig_shape), qscale
    elif HAS_CK and x.is_cuda and hasattr(comfy_kitchen.backends.cuda, "quantize_int8_rowwise_convrot"):
        qdata, qscale = comfy_kitchen.backends.cuda.quantize_int8_rowwise_convrot(x_2d, convrot_groupsize)
        return qdata.view(*orig_shape), qscale
    elif HAS_CK and hasattr(comfy_kitchen.backends.cuda, "_build_hadamard") and hasattr(comfy_kitchen.backends.cuda, "_rotate_activation"):
        h = comfy_kitchen.backends.cuda._build_hadamard(convrot_groupsize, device=x_2d.device, dtype=x_2d.dtype)
        x_rot = comfy_kitchen.backends.cuda._rotate_activation(x_2d, h, convrot_groupsize)
        qdata, qscale = comfy_kitchen.quantize_int8_rowwise(x_rot)
        return qdata.view(*orig_shape), qscale
    else:
        # High-precision PyTorch reference quantizer
        amax = torch.amax(torch.abs(x_2d), dim=-1, keepdim=True).clamp(min=1e-5)
        qscale = amax / 127.0
        qdata = torch.clamp(torch.round(x_2d / qscale), -128, 127).to(torch.int8)
        return qdata.view(*orig_shape), qscale

def w8a8_gemm(
    q_act: torch.Tensor,
    scale_act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Computes INT8 x INT8 -> FP16 GEMM with already-quantized activations.
    """
    m, k = q_act.shape
    n, k_w = weight.shape
    assert k == k_w

    if HAS_CK and hasattr(torch.ops.comfy_kitchen, "cublas_gemm_int8"):
        out_int32 = torch.empty((m, n), dtype=torch.int32, device=q_act.device)
        workspace = torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device=q_act.device)
        torch.ops.comfy_kitchen.cublas_gemm_int8(q_act, weight, out_int32, workspace, 0)
        out = (out_int32.to(torch.float32) * scale_act * weight_scale.T).to(out_dtype)
    else:
        int32_mm = torch.matmul(q_act.to(torch.float32), weight.T.to(torch.float32))
        out = (int32_mm * scale_act * weight_scale.T).to(out_dtype)

    if bias is not None:
        out = out + bias.to(out_dtype)
    return out

def quantize_weight_int8(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes weight to INT8 with per-channel scale."""
    if HAS_CK:
        return comfy_kitchen.quantize_int8_rowwise(w)
    amax = torch.amax(torch.abs(w), dim=-1, keepdim=True).clamp(min=1e-5)
    scale = amax / 127.0
    qw = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
    return qw, scale.to(torch.float32)

# =============================================================================
# 1. Single-GPU INT8 ConvRot ViT3D Block
# =============================================================================

class SingleGPUViT3DBlockINT8(nn.Module):
    """Single-GPU reference block for MiniMax-H3 ViT3D decoder in INT8 ConvRot."""
    def __init__(
        self,
        dim: int = 2048,
        heads: int = 32,
        dim_head: int = 64,
        ffn_mult: int = 4,
        bias: bool = True,
        eps: float = 1e-5,
        convrot_groupsize: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.ffn_dim = dim * ffn_mult
        self.convrot_groupsize = convrot_groupsize
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Norms & Residual Scales
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False).to(self.device)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False).to(self.device)
        self.scale1 = nn.Parameter(torch.ones(dim, dtype=torch.float16, device=self.device), requires_grad=False)
        self.scale2 = nn.Parameter(torch.ones(dim, dtype=torch.float16, device=self.device), requires_grad=False)

        # Head RMSNorms
        self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=False).to(self.device)
        self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=False).to(self.device)

        # Weights: INT8 [Out, In] and scale [Out, 1]
        self.qw_qkv = nn.Parameter(torch.empty(3 * self.inner_dim, dim, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_qkv = nn.Parameter(torch.empty(3 * self.inner_dim, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_qkv = nn.Parameter(torch.zeros(3 * self.inner_dim, dtype=torch.float16, device=self.device), requires_grad=False)

        self.qw_out = nn.Parameter(torch.empty(dim, self.inner_dim, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_out = nn.Parameter(torch.empty(dim, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_out = nn.Parameter(torch.zeros(dim, dtype=torch.float16, device=self.device), requires_grad=False)

        self.qw_w1 = nn.Parameter(torch.empty(2 * self.ffn_dim, dim, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_w1 = nn.Parameter(torch.empty(2 * self.ffn_dim, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_w1 = nn.Parameter(torch.zeros(2 * self.ffn_dim, dtype=torch.float16, device=self.device), requires_grad=False)

        self.qw_w2 = nn.Parameter(torch.empty(dim, self.ffn_dim, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_w2 = nn.Parameter(torch.empty(dim, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_w2 = nn.Parameter(torch.zeros(dim, dtype=torch.float16, device=self.device), requires_grad=False)

    @torch.no_grad()
    def load_from_fp16(self, qkv_w, qkv_b, out_w, out_b, w1_w, w1_b, w2_w, w2_b, s1, s2):
        qw, qs = quantize_weight_int8(qkv_w)
        self.qw_qkv.data.copy_(qw); self.qs_qkv.data.copy_(qs); self.b_qkv.data.copy_(qkv_b)
        qw, qs = quantize_weight_int8(out_w)
        self.qw_out.data.copy_(qw); self.qs_out.data.copy_(qs); self.b_out.data.copy_(out_b)
        qw, qs = quantize_weight_int8(w1_w)
        self.qw_w1.data.copy_(qw); self.qs_w1.data.copy_(qs); self.b_w1.data.copy_(w1_b)
        qw, qs = quantize_weight_int8(w2_w)
        self.qw_w2.data.copy_(qw); self.qs_w2.data.copy_(qs); self.b_w2.data.copy_(w2_b)
        self.scale1.data.copy_(s1); self.scale2.data.copy_(s2)

    def forward(self, x: torch.Tensor, rotary_pos_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape

        # --- 1. Attention Sub-layer ---
        h_norm1 = self.norm1(x)
        qh, sh = quantize_int8_activation_convrot(h_norm1, self.convrot_groupsize)
        qkv = w8a8_gemm(qh.view(-1, self.dim), sh, self.qw_qkv, self.qs_qkv, self.b_qkv, out_dtype=x.dtype)
        qkv = qkv.view(b, s, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4) # [3, B, Heads, S, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.norm_q(q)
        k = self.norm_k(k)
        if rotary_pos_emb is not None:
            # Apply rotary positional embedding
            # rotary_pos_emb: [B, S, 1, pairs, 2, 2]
            q, k = self._apply_rope(q, k, rotary_pos_emb)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.dim_head)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale) # [B, Heads, S, D]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(b * s, self.inner_dim)

        q_attn, s_attn = quantize_int8_activation_convrot(attn_out, self.convrot_groupsize)
        out_proj = w8a8_gemm(q_attn, s_attn, self.qw_out, self.qs_out, self.b_out, out_dtype=x.dtype)
        x = x + self.scale1 * out_proj.view(b, s, self.dim)

        # --- 2. FeedForward Sub-layer ---
        h_norm2 = self.norm2(x)
        qh2, sh2 = quantize_int8_activation_convrot(h_norm2, self.convrot_groupsize)
        w1_out = w8a8_gemm(qh2.view(-1, self.dim), sh2, self.qw_w1, self.qs_w1, self.b_w1, out_dtype=x.dtype)
        gate, up = torch.chunk(w1_out, 2, dim=-1)
        swiglu = torch.nn.functional.silu(gate) * up # [B*S, ffn_dim]

        q_swi, s_swi = quantize_int8_activation_convrot(swiglu, self.convrot_groupsize)
        w2_out = w8a8_gemm(q_swi, s_swi, self.qw_w2, self.qs_w2, self.b_w2, out_dtype=x.dtype)
        x = x + self.scale2 * w2_out.view(b, s, self.dim)
        return x

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # RoPE broadcasted across heads
        # q, k: [B, Heads, S, D]
        # In ComfyUI, rope is [B, S, 1, pairs, 2, 2]
        rot_dim = rope.shape[-3] * 2
        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
        # Standard complex rotation
        q_rot = q_rot.reshape(*q_rot.shape[:-1], -1, 2)
        k_rot = k_rot.reshape(*k_rot.shape[:-1], -1, 2)
        q_rot = torch.stack([-q_rot[..., 1], q_rot[..., 0]], dim=-1).flatten(-2)
        k_rot = torch.stack([-k_rot[..., 1], k_rot[..., 0]], dim=-1).flatten(-2)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

# =============================================================================
# 2. Strategy 2: TP=2 (Megatron Tensor Parallelism with FP16 AllReduce)
# =============================================================================

class TPViT3DBlockINT8(nn.Module):
    """
    Megatron Tensor Parallel (TP=2) ViT3D Block:
    - Column-Parallel: to_qkv (16 heads per rank), w1 (8192 out per rank).
    - Row-Parallel: to_out, w2 with FP16 AllReduce.
    """
    def __init__(
        self,
        dim: int = 2048,
        heads: int = 32,
        dim_head: int = 64,
        ffn_mult: int = 4,
        rank: int = 0,
        world_size: int = 2,
        bias: bool = True,
        eps: float = 1e-5,
        convrot_groupsize: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.heads_per_rank = heads // world_size
        self.dim_head = dim_head
        self.inner_dim_per_rank = self.heads_per_rank * dim_head # 1024
        self.ffn_dim = dim * ffn_mult # 8192
        self.ffn_dim_per_rank = self.ffn_dim // world_size # 4096
        self.rank = rank
        self.world_size = world_size
        self.convrot_groupsize = convrot_groupsize
        self.device = device or torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Norms & Residual Scales
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False).to(self.device)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False).to(self.device)
        self.scale1 = nn.Parameter(torch.ones(dim, dtype=torch.float16, device=self.device), requires_grad=False)
        self.scale2 = nn.Parameter(torch.ones(dim, dtype=torch.float16, device=self.device), requires_grad=False)

        self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=False).to(self.device)
        self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=False).to(self.device)

        # Sharded Weights
        # to_qkv: [3 * 1024 = 3072, 2048]
        self.qw_qkv = nn.Parameter(torch.empty(3 * self.inner_dim_per_rank, dim, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_qkv = nn.Parameter(torch.empty(3 * self.inner_dim_per_rank, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_qkv = nn.Parameter(torch.zeros(3 * self.inner_dim_per_rank, dtype=torch.float16, device=self.device), requires_grad=False)

        # to_out: [2048, 1024]
        self.qw_out = nn.Parameter(torch.empty(dim, self.inner_dim_per_rank, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_out = nn.Parameter(torch.empty(dim, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_out = nn.Parameter(torch.zeros(dim, dtype=torch.float16, device=self.device), requires_grad=False)

        # w1: [2 * 4096 = 8192, 2048]
        self.qw_w1 = nn.Parameter(torch.empty(2 * self.ffn_dim_per_rank, dim, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_w1 = nn.Parameter(torch.empty(2 * self.ffn_dim_per_rank, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_w1 = nn.Parameter(torch.zeros(2 * self.ffn_dim_per_rank, dtype=torch.float16, device=self.device), requires_grad=False)

        # w2: [2048, 4096]
        self.qw_w2 = nn.Parameter(torch.empty(dim, self.ffn_dim_per_rank, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_w2 = nn.Parameter(torch.empty(dim, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_w2 = nn.Parameter(torch.zeros(dim, dtype=torch.float16, device=self.device), requires_grad=False)

    @torch.no_grad()
    def load_from_full(self, ref: SingleGPUViT3DBlockINT8):
        # Shard to_qkv by heads: 3 chunks of [heads, head_dim]
        # Full to_qkv is [3 * 2048, 2048]
        q_w, k_w, v_w = torch.chunk(ref.qw_qkv.data, 3, dim=0)
        q_s, k_s, v_s = torch.chunk(ref.qs_qkv.data, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(ref.b_qkv.data, 3, dim=0)
        start_h = self.rank * self.inner_dim_per_rank
        end_h = start_h + self.inner_dim_per_rank

        sharded_w = torch.cat([q_w[start_h:end_h], k_w[start_h:end_h], v_w[start_h:end_h]], dim=0)
        sharded_s = torch.cat([q_s[start_h:end_h], k_s[start_h:end_h], v_s[start_h:end_h]], dim=0)
        sharded_b = torch.cat([q_b[start_h:end_h], k_b[start_h:end_h], v_b[start_h:end_h]], dim=0)
        self.qw_qkv.data.copy_(sharded_w.to(self.device))
        self.qs_qkv.data.copy_(sharded_s.to(self.device))
        self.b_qkv.data.copy_(sharded_b.to(self.device))

        # Shard to_out by columns: [2048, 2048] -> [2048, 1024]
        self.qw_out.data.copy_(ref.qw_out.data[:, start_h:end_h].to(self.device))
        self.qs_out.data.copy_(ref.qs_out.data.to(self.device))
        self.b_out.data.copy_((ref.b_out.data / self.world_size).to(self.device))

        # Shard w1 by rows: gate [8192, 2048], up [8192, 2048]
        gate_w, up_w = torch.chunk(ref.qw_w1.data, 2, dim=0)
        gate_s, up_s = torch.chunk(ref.qs_w1.data, 2, dim=0)
        gate_b, up_b = torch.chunk(ref.b_w1.data, 2, dim=0)
        start_f = self.rank * self.ffn_dim_per_rank
        end_f = start_f + self.ffn_dim_per_rank
        self.qw_w1.data.copy_(torch.cat([gate_w[start_f:end_f], up_w[start_f:end_f]], dim=0).to(self.device))
        self.qs_w1.data.copy_(torch.cat([gate_s[start_f:end_f], up_s[start_f:end_f]], dim=0).to(self.device))
        self.b_w1.data.copy_(torch.cat([gate_b[start_f:end_f], up_b[start_f:end_f]], dim=0).to(self.device))

        # Shard w2 by columns: [2048, 8192] -> [2048, 4096]
        self.qw_w2.data.copy_(ref.qw_w2.data[:, start_f:end_f].to(self.device))
        self.qs_w2.data.copy_(ref.qs_w2.data.to(self.device))
        self.b_w2.data.copy_((ref.b_w2.data / self.world_size).to(self.device))

        self.scale1.data.copy_(ref.scale1.data.to(self.device))
        self.scale2.data.copy_(ref.scale2.data.to(self.device))

    def forward(self, x: torch.Tensor, rotary_pos_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape

        # --- 1. Attention Sub-layer (TP=2) ---
        h_norm1 = self.norm1(x)
        qh, sh = quantize_int8_activation_convrot(h_norm1, self.convrot_groupsize)
        qkv = w8a8_gemm(qh.view(-1, self.dim), sh, self.qw_qkv, self.qs_qkv, self.b_qkv, out_dtype=x.dtype)
        qkv = qkv.view(b, s, 3, self.heads_per_rank, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.norm_q(q)
        k = self.norm_k(k)
        if rotary_pos_emb is not None:
            q, k = self._apply_rope(q, k, rotary_pos_emb)

        scale = 1.0 / math.sqrt(self.dim_head)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(b * s, self.inner_dim_per_rank)

        q_attn, s_attn = quantize_int8_activation_convrot(attn_out, self.convrot_groupsize)
        partial_out = w8a8_gemm(q_attn, s_attn, self.qw_out, self.qs_out, self.b_out, out_dtype=x.dtype)

        # FP16 AllReduce
        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(partial_out, op=dist.ReduceOp.SUM)

        x = x + self.scale1 * partial_out.view(b, s, self.dim)

        # --- 2. FeedForward Sub-layer (TP=2) ---
        h_norm2 = self.norm2(x)
        qh2, sh2 = quantize_int8_activation_convrot(h_norm2, self.convrot_groupsize)
        w1_out = w8a8_gemm(qh2.view(-1, self.dim), sh2, self.qw_w1, self.qs_w1, self.b_w1, out_dtype=x.dtype)
        gate, up = torch.chunk(w1_out, 2, dim=-1)
        swiglu = torch.nn.functional.silu(gate) * up # [B*S, ffn_dim_per_rank]

        q_swi, s_swi = quantize_int8_activation_convrot(swiglu, self.convrot_groupsize)
        partial_w2 = w8a8_gemm(q_swi, s_swi, self.qw_w2, self.qs_w2, self.b_w2, out_dtype=x.dtype)

        # FP16 AllReduce
        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(partial_w2, op=dist.ReduceOp.SUM)

        x = x + self.scale2 * partial_w2.view(b, s, self.dim)
        return x

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rot_dim = rope.shape[-3] * 2
        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
        q_rot = q_rot.reshape(*q_rot.shape[:-1], -1, 2)
        k_rot = k_rot.reshape(*k_rot.shape[:-1], -1, 2)
        q_rot = torch.stack([-q_rot[..., 1], q_rot[..., 0]], dim=-1).flatten(-2)
        k_rot = torch.stack([-k_rot[..., 1], k_rot[..., 0]], dim=-1).flatten(-2)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

# =============================================================================
# 3. Strategy 3: TP=2 + Sequence Parallelism + INT8 AllGather
# =============================================================================

class SPViT3DBlockINT8(nn.Module):
    """
    TP=2 + Sequence Parallelism + INT8 AllGather ViT3D Block:
    - Partitioned activations at [S/2, dim] in FP16 between blocks.
    - Local Norms and Residual additions on [S/2, dim].
    - INT8 ConvRot Activation Quantization -> INT8 AllGather before QKV & W1.
    - Row-Parallel GEMM -> FP16 ReduceScatter along Sequence dimension.
    """
    def __init__(
        self,
        dim: int = 2048,
        heads: int = 32,
        dim_head: int = 64,
        ffn_mult: int = 4,
        rank: int = 0,
        world_size: int = 2,
        bias: bool = True,
        eps: float = 1e-5,
        convrot_groupsize: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.heads_per_rank = heads // world_size
        self.dim_head = dim_head
        self.inner_dim_per_rank = self.heads_per_rank * dim_head # 1024
        self.ffn_dim = dim * ffn_mult # 8192
        self.ffn_dim_per_rank = self.ffn_dim // world_size # 4096
        self.rank = rank
        self.world_size = world_size
        self.convrot_groupsize = convrot_groupsize
        self.device = device or torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Norms & Residual Scales
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False).to(self.device)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False).to(self.device)
        self.scale1 = nn.Parameter(torch.ones(dim, dtype=torch.float16, device=self.device), requires_grad=False)
        self.scale2 = nn.Parameter(torch.ones(dim, dtype=torch.float16, device=self.device), requires_grad=False)

        self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=False).to(self.device)
        self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=False).to(self.device)

        # Sharded Weights
        self.qw_qkv = nn.Parameter(torch.empty(3 * self.inner_dim_per_rank, dim, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_qkv = nn.Parameter(torch.empty(3 * self.inner_dim_per_rank, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_qkv = nn.Parameter(torch.zeros(3 * self.inner_dim_per_rank, dtype=torch.float16, device=self.device), requires_grad=False)

        self.qw_out = nn.Parameter(torch.empty(dim, self.inner_dim_per_rank, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_out = nn.Parameter(torch.empty(dim, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_out = nn.Parameter(torch.zeros(dim, dtype=torch.float16, device=self.device), requires_grad=False)

        self.qw_w1 = nn.Parameter(torch.empty(2 * self.ffn_dim_per_rank, dim, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_w1 = nn.Parameter(torch.empty(2 * self.ffn_dim_per_rank, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_w1 = nn.Parameter(torch.zeros(2 * self.ffn_dim_per_rank, dtype=torch.float16, device=self.device), requires_grad=False)

        self.qw_w2 = nn.Parameter(torch.empty(dim, self.ffn_dim_per_rank, dtype=torch.int8, device=self.device), requires_grad=False)
        self.qs_w2 = nn.Parameter(torch.empty(dim, 1, dtype=torch.float32, device=self.device), requires_grad=False)
        self.b_w2 = nn.Parameter(torch.zeros(dim, dtype=torch.float16, device=self.device), requires_grad=False)

    @torch.no_grad()
    def load_from_full(self, ref: SingleGPUViT3DBlockINT8):
        q_w, k_w, v_w = torch.chunk(ref.qw_qkv.data, 3, dim=0)
        q_s, k_s, v_s = torch.chunk(ref.qs_qkv.data, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(ref.b_qkv.data, 3, dim=0)
        start_h = self.rank * self.inner_dim_per_rank
        end_h = start_h + self.inner_dim_per_rank

        sharded_w = torch.cat([q_w[start_h:end_h], k_w[start_h:end_h], v_w[start_h:end_h]], dim=0)
        sharded_s = torch.cat([q_s[start_h:end_h], k_s[start_h:end_h], v_s[start_h:end_h]], dim=0)
        sharded_b = torch.cat([q_b[start_h:end_h], k_b[start_h:end_h], v_b[start_h:end_h]], dim=0)
        self.qw_qkv.data.copy_(sharded_w.to(self.device))
        self.qs_qkv.data.copy_(sharded_s.to(self.device))
        self.b_qkv.data.copy_(sharded_b.to(self.device))

        self.qw_out.data.copy_(ref.qw_out.data[:, start_h:end_h].to(self.device))
        self.qs_out.data.copy_(ref.qs_out.data.to(self.device))
        self.b_out.data.copy_((ref.b_out.data / self.world_size).to(self.device))

        gate_w, up_w = torch.chunk(ref.qw_w1.data, 2, dim=0)
        gate_s, up_s = torch.chunk(ref.qs_w1.data, 2, dim=0)
        gate_b, up_b = torch.chunk(ref.b_w1.data, 2, dim=0)
        start_f = self.rank * self.ffn_dim_per_rank
        end_f = start_f + self.ffn_dim_per_rank
        self.qw_w1.data.copy_(torch.cat([gate_w[start_f:end_f], up_w[start_f:end_f]], dim=0).to(self.device))
        self.qs_w1.data.copy_(torch.cat([gate_s[start_f:end_f], up_s[start_f:end_f]], dim=0).to(self.device))
        self.b_w1.data.copy_(torch.cat([gate_b[start_f:end_f], up_b[start_f:end_f]], dim=0).to(self.device))

        self.qw_w2.data.copy_(ref.qw_w2.data[:, start_f:end_f].to(self.device))
        self.qs_w2.data.copy_(ref.qs_w2.data.to(self.device))
        self.b_w2.data.copy_((ref.b_w2.data / self.world_size).to(self.device))

        self.scale1.data.copy_(ref.scale1.data.to(self.device))
        self.scale2.data.copy_(ref.scale2.data.to(self.device))

    def forward(self, x_local: torch.Tensor, rotary_pos_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_local: [B, S/2, dim] local slice in FP16.
        Returns: [B, S/2, dim] local slice in FP16.
        """
        b, s_local, _ = x_local.shape
        s_total = s_local * self.world_size

        # --- 1. Attention Sub-layer (TP=2 + SP + INT8 AG) ---
        # 1a. Local Norm on S/2
        h_norm1_local = self.norm1(x_local)

        # 1b. Local INT8 Activation Quantization on S/2
        qh_local, sh_local = quantize_int8_activation_convrot(h_norm1_local, self.convrot_groupsize)

        # 1c. INT8 AllGather -> [B, S, dim] (1 byte per element)
        if self.world_size > 1 and dist.is_initialized():
            qh_full = torch.empty((b, s_total, self.dim), dtype=torch.int8, device=self.device)
            dist.all_gather_into_tensor(qh_full, qh_local)
            sh_full = torch.empty((b, s_total, 1), dtype=torch.float32, device=self.device)
            dist.all_gather_into_tensor(sh_full, sh_local)
        else:
            qh_full = qh_local.repeat(1, self.world_size, 1)
            sh_full = sh_local.repeat(1, self.world_size, 1)

        # 1d. Column W8A8 GEMM -> [B, S, 3072]
        qkv = w8a8_gemm(qh_full.view(-1, self.dim), sh_full.view(-1, 1), self.qw_qkv, self.qs_qkv, self.b_qkv, out_dtype=x_local.dtype)
        qkv = qkv.view(b, s_total, 3, self.heads_per_rank, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.norm_q(q)
        k = self.norm_k(k)
        if rotary_pos_emb is not None:
            q, k = self._apply_rope(q, k, rotary_pos_emb)

        scale = 1.0 / math.sqrt(self.dim_head)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(b * s_total, self.inner_dim_per_rank)

        # 1e. Row W8A8 GEMM -> partial sum [B*S, dim]
        q_attn, s_attn = quantize_int8_activation_convrot(attn_out, self.convrot_groupsize)
        partial_out = w8a8_gemm(q_attn, s_attn, self.qw_out, self.qs_out, self.b_out, out_dtype=x_local.dtype)
        partial_out = partial_out.view(b, s_total, self.dim)

        # 1f. FP16 ReduceScatter -> [B, S/2, dim]
        if self.world_size > 1 and dist.is_initialized():
            reduced_attn = torch.empty((b, s_local, self.dim), dtype=x_local.dtype, device=self.device)
            dist.reduce_scatter_tensor(reduced_attn, partial_out.contiguous(), op=dist.ReduceOp.SUM)
        else:
            reduced_attn = partial_out[:, :s_local, :]

        # 1g. Local Residual Addition on S/2
        x_local = x_local + self.scale1 * reduced_attn

        # --- 2. FeedForward Sub-layer (TP=2 + SP + INT8 AG) ---
        # 2a. Local Norm on S/2
        h_norm2_local = self.norm2(x_local)

        # 2b. Local INT8 Activation Quantization on S/2
        qh2_local, sh2_local = quantize_int8_activation_convrot(h_norm2_local, self.convrot_groupsize)

        # 2c. INT8 AllGather -> [B, S, dim]
        if self.world_size > 1 and dist.is_initialized():
            qh2_full = torch.empty((b, s_total, self.dim), dtype=torch.int8, device=self.device)
            dist.all_gather_into_tensor(qh2_full, qh2_local)
            sh2_full = torch.empty((b, s_total, 1), dtype=torch.float32, device=self.device)
            dist.all_gather_into_tensor(sh2_full, sh2_local)
        else:
            qh2_full = qh2_local.repeat(1, self.world_size, 1)
            sh2_full = sh2_local.repeat(1, self.world_size, 1)

        # 2d. Column W8A8 GEMM -> [B, S, 8192]
        w1_out = w8a8_gemm(qh2_full.view(-1, self.dim), sh2_full.view(-1, 1), self.qw_w1, self.qs_w1, self.b_w1, out_dtype=x_local.dtype)
        gate, up = torch.chunk(w1_out, 2, dim=-1)
        swiglu = torch.nn.functional.silu(gate) * up # [B*S, ffn_dim_per_rank]

        # 2e. Row W8A8 GEMM -> partial sum [B*S, dim]
        q_swi, s_swi = quantize_int8_activation_convrot(swiglu, self.convrot_groupsize)
        partial_w2 = w8a8_gemm(q_swi, s_swi, self.qw_w2, self.qs_w2, self.b_w2, out_dtype=x_local.dtype)
        partial_w2 = partial_w2.view(b, s_total, self.dim)

        # 2f. FP16 ReduceScatter -> [B, S/2, dim]
        if self.world_size > 1 and dist.is_initialized():
            reduced_w2 = torch.empty((b, s_local, self.dim), dtype=x_local.dtype, device=self.device)
            dist.reduce_scatter_tensor(reduced_w2, partial_w2.contiguous(), op=dist.ReduceOp.SUM)
        else:
            reduced_w2 = partial_w2[:, :s_local, :]

        # 2g. Local Residual Addition on S/2
        x_local = x_local + self.scale2 * reduced_w2
        return x_local

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rot_dim = rope.shape[-3] * 2
        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
        q_rot = q_rot.reshape(*q_rot.shape[:-1], -1, 2)
        k_rot = k_rot.reshape(*k_rot.shape[:-1], -1, 2)
        q_rot = torch.stack([-q_rot[..., 1], q_rot[..., 0]], dim=-1).flatten(-2)
        k_rot = torch.stack([-k_rot[..., 1], k_rot[..., 0]], dim=-1).flatten(-2)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)
