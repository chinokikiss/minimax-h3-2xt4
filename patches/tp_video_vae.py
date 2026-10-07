"""
Production Multi-GPU Inference Implementation for MiniMax-H3 Video VAE INT8 ConvRot.
Supports:
  1. Single GPU INT8 ConvRot Baseline (Real Checkpoint, Native Path)
  2. TP=2 + FP16 AllReduce (Megatron-style Column/Row parallelism over NCCL)
  3. TP=2 + Sequence Parallelism + FP16 ReduceScatter + INT8 AllGather (Over NCCL)

Operates on the actual checkpoint:
  Comfy-Org/MiniMax-H3/vae/minimax_h3_video_vae_int8_convrot.safetensors
"""

import os
import sys
import math
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Ensure project and ComfyUI roots are in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.join(PROJECT_ROOT, "ComfyUI")
for p in [PROJECT_ROOT, COMFY_ROOT, "/tmp/minimax_repo", "/tmp/ComfyUI"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Try importing comfy and comfy_kitchen
try:
    import comfy_kitchen
    HAS_CK = True
except ImportError:
    HAS_CK = False

try:
    from comfy.ldm.minimax.vae import create_token_ids, RotaryEmbeddingND
except ImportError:
    def create_token_ids(patch_dims, device, dtype):
        coords_list = []
        for dim_size in patch_dims:
            coords = torch.arange(0.5, dim_size, dtype=dtype, device=device)
            coords = coords / dim_size
            coords = 2.0 * coords - 1.0
            coords_list.append(coords)
        coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
        return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0)

    class RotaryEmbeddingND(nn.Module):
        def __init__(self, dim, rotary_base=100.0, n_dim=3):
            super().__init__()
            self.n_dim = n_dim
            self.angle_scale = 2.0 * math.pi
            inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, img_ids):
            angles = (
                self.angle_scale
                * img_ids[:, :, :, None].float()
                * self.inv_freq.to(img_ids.device)[None, None, None, :]
            )
            angles = angles.flatten(2, 3)
            c, s = torch.cos(angles), torch.sin(angles)
            table = torch.stack([c, -s, s, c], dim=-1).reshape(*angles.shape[:2], 1, angles.shape[-1], 2, 2)
            return table.to(img_ids.dtype)


def quantize_int8_rowwise_convrot(
    x_2d: torch.Tensor,
    convrot_groupsize: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused online ConvRot Hadamard rotation + per-row INT8 quantization.
    Input: [M, K] in FP16 or BF16.
    Output: qdata [M, K] in INT8, qscale [M, 1] in FP32.
    """
    orig_shape = x_2d.shape
    x_flat = x_2d.reshape(-1, orig_shape[-1])
    m, k = x_flat.shape

    if HAS_CK and x_flat.is_cuda and hasattr(comfy_kitchen.backends.cuda, "quantize_int8_rowwise_convrot"):
        qdata, qscale = comfy_kitchen.backends.cuda.quantize_int8_rowwise_convrot(x_flat, convrot_groupsize)
        return qdata.view(*orig_shape), qscale
    elif HAS_CK and hasattr(comfy_kitchen.backends.cuda, "_build_hadamard") and hasattr(comfy_kitchen.backends.cuda, "_rotate_activation"):
        h = comfy_kitchen.backends.cuda._build_hadamard(convrot_groupsize, device=x_flat.device, dtype=x_flat.dtype)
        x_rot = comfy_kitchen.backends.cuda._rotate_activation(x_flat, h, convrot_groupsize)
        qdata, qscale = comfy_kitchen.quantize_int8_rowwise(x_rot)
        return qdata.view(*orig_shape), qscale
    else:
        # PyTorch reference implementation
        amax = torch.amax(torch.abs(x_flat), dim=-1, keepdim=True).clamp(min=1e-5)
        qscale = amax / 127.0
        qdata = torch.clamp(torch.round(x_flat / qscale), -128, 127).to(torch.int8)
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
    INT8 x INT8 -> FP16 GEMM with already-quantized activations and per-channel weights.
    q_act: [M, K] INT8
    scale_act: [M, 1] FP32
    weight: [N, K] INT8
    weight_scale: [N, 1] FP32
    """
    m, k = q_act.shape
    n, k_w = weight.shape
    assert k == k_w, f"K mismatch: act {k} vs weight {k_w}"

    if HAS_CK and q_act.is_cuda and hasattr(torch.ops.comfy_kitchen, "cublas_gemm_int8"):
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


def create_tp_block_from_sd(
    sd: Dict[str, torch.Tensor],
    block_idx: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Extracts and shards checkpoint weights for block block_idx for rank.
    """
    prefix = f"decoder.transformer_blocks.{block_idx}."

    # 1. to_qkv: [6144, 2048]
    w_qkv = sd[f"{prefix}attn.to_qkv.weight"]
    s_qkv = sd[f"{prefix}attn.to_qkv.weight_scale"]
    b_qkv = sd.get(f"{prefix}attn.to_qkv.bias", None)

    # to_qkv is ordered by heads: 32 heads, each head has [Q (64), K (64), V (64)] = 192 rows.
    # Total rows = 32 * 192 = 6144.
    # For TP=2: Rank 0 gets heads 0..15 (rows 0..3072), Rank 1 gets heads 16..31 (rows 3072..6144).
    qkv_rows_per_rank = (32 // world_size) * 192  # 3072
    qkv_start = rank * qkv_rows_per_rank
    qkv_end = qkv_start + qkv_rows_per_rank

    sharded_w_qkv = w_qkv[qkv_start:qkv_end]
    sharded_s_qkv = s_qkv[qkv_start:qkv_end]
    sharded_b_qkv = b_qkv[qkv_start:qkv_end] if b_qkv is not None else None

    # 2. to_out: [2048, 2048]
    w_out = sd[f"{prefix}attn.to_out.weight"]
    s_out = sd[f"{prefix}attn.to_out.weight_scale"]
    b_out = sd.get(f"{prefix}attn.to_out.bias", None)

    sharded_w_out = w_out[:, h_start:h_end]
    sharded_s_out = s_out
    sharded_b_out = (b_out / world_size) if b_out is not None else None

    # 3. w1: [16384, 2048]
    w_w1 = sd[f"{prefix}ff.w1.weight"]
    s_w1 = sd[f"{prefix}ff.w1.weight_scale"]
    b_w1 = sd.get(f"{prefix}ff.w1.bias", None)

    ffn_dim_per_rank = 4096
    f_start = rank * ffn_dim_per_rank
    f_end = f_start + ffn_dim_per_rank

    gate_w, up_w = w_w1[:8192], w_w1[8192:]
    gate_s, up_s = s_w1[:8192], s_w1[8192:]
    gate_b, up_b = (b_w1[:8192], b_w1[8192:]) if b_w1 is not None else (None, None)

    sharded_w_w1 = torch.cat([gate_w[f_start:f_end], up_w[f_start:f_end]], dim=0)
    sharded_s_w1 = torch.cat([gate_s[f_start:f_end], up_s[f_start:f_end]], dim=0)
    sharded_b_w1 = torch.cat([gate_b[f_start:f_end], up_b[f_start:f_end]], dim=0) if b_w1 is not None else None

    # 4. w2: [2048, 8192]
    w_w2 = sd[f"{prefix}ff.w2.weight"]
    s_w2 = sd[f"{prefix}ff.w2.weight_scale"]
    b_w2 = sd.get(f"{prefix}ff.w2.bias", None)

    sharded_w_w2 = w_w2[:, f_start:f_end]
    sharded_s_w2 = s_w2
    sharded_b_w2 = (b_w2 / world_size) if b_w2 is not None else None

    # Norms and scales
    norm1_w = sd[f"{prefix}norm1.weight"]
    norm2_w = sd[f"{prefix}norm2.weight"]
    scale1 = sd[f"{prefix}scale1"]
    scale2 = sd[f"{prefix}scale2"]

    return {
        "qw_qkv": nn.Parameter(sharded_w_qkv.to(device=device, dtype=torch.int8), requires_grad=False),
        "qs_qkv": nn.Parameter(sharded_s_qkv.to(device=device, dtype=torch.float32), requires_grad=False),
        "b_qkv": nn.Parameter(sharded_b_qkv.to(device=device, dtype=torch.float16), requires_grad=False) if sharded_b_qkv is not None else None,
        "qw_out": nn.Parameter(sharded_w_out.to(device=device, dtype=torch.int8), requires_grad=False),
        "qs_out": nn.Parameter(sharded_s_out.to(device=device, dtype=torch.float32), requires_grad=False),
        "b_out": nn.Parameter(sharded_b_out.to(device=device, dtype=torch.float16), requires_grad=False) if sharded_b_out is not None else None,
        "qw_w1": nn.Parameter(sharded_w_w1.to(device=device, dtype=torch.int8), requires_grad=False),
        "qs_w1": nn.Parameter(sharded_s_w1.to(device=device, dtype=torch.float32), requires_grad=False),
        "b_w1": nn.Parameter(sharded_b_w1.to(device=device, dtype=torch.float16), requires_grad=False) if sharded_b_w1 is not None else None,
        "qw_w2": nn.Parameter(sharded_w_w2.to(device=device, dtype=torch.int8), requires_grad=False),
        "qs_w2": nn.Parameter(sharded_s_w2.to(device=device, dtype=torch.float32), requires_grad=False),
        "b_w2": nn.Parameter(sharded_b_w2.to(device=device, dtype=torch.float16), requires_grad=False) if sharded_b_w2 is not None else None,
        "norm1_w": nn.Parameter(norm1_w.to(device=device, dtype=torch.float16), requires_grad=False),
        "norm2_w": nn.Parameter(norm2_w.to(device=device, dtype=torch.float16), requires_grad=False),
        "scale1": nn.Parameter(scale1.to(device=device, dtype=torch.float16), requires_grad=False),
        "scale2": nn.Parameter(scale2.to(device=device, dtype=torch.float16), requires_grad=False),
    }


class RealTPTransformerBlock(nn.Module):
    """
    Megatron Tensor Parallelism (TP=2) for MiniMax-H3 ViT3D Transformer Block.
    Uses FP16 AllReduce for row-parallel attention output and MLP down projection.
    """
    def __init__(self, block_dict: Dict[str, Any], eps: float = 1e-5, convrot_groupsize: int = 256, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = 2048
        self.heads_per_rank = 16
        self.dim_head = 64
        self.inner_dim_per_rank = 1024
        self.ffn_dim_per_rank = 4096
        self.convrot_groupsize = convrot_groupsize
        self.device = device
        self.eps = eps

        self.qw_qkv = block_dict["qw_qkv"]
        self.qs_qkv = block_dict["qs_qkv"]
        self.b_qkv = block_dict["b_qkv"]

        self.qw_out = block_dict["qw_out"]
        self.qs_out = block_dict["qs_out"]
        self.b_out = block_dict["b_out"]

        self.qw_w1 = block_dict["qw_w1"]
        self.qs_w1 = block_dict["qs_w1"]
        self.b_w1 = block_dict["b_w1"]

        self.qw_w2 = block_dict["qw_w2"]
        self.qs_w2 = block_dict["qs_w2"]
        self.b_w2 = block_dict["b_w2"]

        self.norm1_w = block_dict["norm1_w"]
        self.norm2_w = block_dict["norm2_w"]
        self.scale1 = block_dict["scale1"]
        self.scale2 = block_dict["scale2"]

        self.register_buffer("qk_norm_scale", torch.ones(self.dim_head, device=device, dtype=torch.float16), persistent=False)

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * weight

    def _apply_rope_eager(self, q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pairs = rope.shape[-3]
        rot_dim = pairs * 2
        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
        q_rot = q_rot.reshape(*q.shape[:-1], pairs, 1, 2)
        k_rot = k_rot.reshape(*k.shape[:-1], pairs, 1, 2)
        q_rot = torch.matmul(q_rot, rope).squeeze(-2).flatten(-2)
        k_rot = torch.matmul(k_rot, rope).squeeze(-2).flatten(-2)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    def forward(self, x: torch.Tensor, rotary_pos_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, dim = x.shape

        # 1. Attention Sublayer
        h1 = self._rms_norm(x, self.norm1_w)
        q_act, s_act = quantize_int8_rowwise_convrot(h1.view(-1, dim), self.convrot_groupsize)
        qkv = w8a8_gemm(q_act, s_act, self.qw_qkv, self.qs_qkv, self.b_qkv, out_dtype=x.dtype)
        qkv = qkv.view(b, s, self.heads_per_rank, 3 * self.dim_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        if rotary_pos_emb is not None and HAS_CK and hasattr(comfy_kitchen.backends.cuda, "rms_rope_split_half"):
            q, k = comfy_kitchen.backends.cuda.rms_rope_split_half(
                q, k, rotary_pos_emb, self.qk_norm_scale,
                epsilon=self.eps, rot_dim=rotary_pos_emb.shape[-3] * 2
            )
        else:
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + self.eps)
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + self.eps)
            if rotary_pos_emb is not None:
                q, k = self._apply_rope_eager(q, k, rotary_pos_emb)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v)) # [B, Heads, S, D]
        scale = 1.0 / math.sqrt(self.dim_head)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=scale)
        attn_out = torch.nan_to_num(attn_out).transpose(1, 2).reshape(b * s, self.inner_dim_per_rank)

        q_attn, s_attn = quantize_int8_rowwise_convrot(attn_out, self.convrot_groupsize)
        partial_out = w8a8_gemm(q_attn, s_attn, self.qw_out, self.qs_out, self.b_out, out_dtype=x.dtype)
        partial_out = partial_out.view(b, s, dim)

        # Real NCCL FP16 AllReduce
        if dist.is_initialized():
            dist.all_reduce(partial_out, op=dist.ReduceOp.SUM)

        x = x + self.scale1 * partial_out

        # 2. FeedForward Sublayer
        h2 = self._rms_norm(x, self.norm2_w)
        q_act2, s_act2 = quantize_int8_rowwise_convrot(h2.view(-1, dim), self.convrot_groupsize)
        w1_out = w8a8_gemm(q_act2, s_act2, self.qw_w1, self.qs_w1, self.b_w1, out_dtype=x.dtype)
        gate, up = torch.chunk(w1_out, 2, dim=-1)
        swiglu = F.silu(gate) * up # [B*S, 4096]

        q_swi, s_swi = quantize_int8_rowwise_convrot(swiglu, self.convrot_groupsize)
        partial_w2 = w8a8_gemm(q_swi, s_swi, self.qw_w2, self.qs_w2, self.b_w2, out_dtype=x.dtype)
        partial_w2 = partial_w2.view(b, s, dim)

        # Real NCCL FP16 AllReduce
        if dist.is_initialized():
            dist.all_reduce(partial_w2, op=dist.ReduceOp.SUM)

        x = x + self.scale2 * partial_w2
        return x


class RealSPTransformerBlock(nn.Module):
    """
    TP=2 + Sequence Parallelism + FP16 ReduceScatter + INT8 AllGather ViT3D Block.
    Activations are partitioned along S: [B, S/2, 2048] per rank.
    Quantization occurs post-reduction on local S/2 slices.
    """
    def __init__(self, block_dict: Dict[str, Any], world_size: int = 2, eps: float = 1e-5, convrot_groupsize: int = 256, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = 2048
        self.world_size = world_size
        self.heads_per_rank = 16
        self.dim_head = 64
        self.inner_dim_per_rank = 1024
        self.ffn_dim_per_rank = 4096
        self.convrot_groupsize = convrot_groupsize
        self.device = device
        self.eps = eps

        self.qw_qkv = block_dict["qw_qkv"]
        self.qs_qkv = block_dict["qs_qkv"]
        self.b_qkv = block_dict["b_qkv"]

        self.qw_out = block_dict["qw_out"]
        self.qs_out = block_dict["qs_out"]
        self.b_out = block_dict["b_out"]

        self.qw_w1 = block_dict["qw_w1"]
        self.qs_w1 = block_dict["qs_w1"]
        self.b_w1 = block_dict["b_w1"]

        self.qw_w2 = block_dict["qw_w2"]
        self.qs_w2 = block_dict["qs_w2"]
        self.b_w2 = block_dict["b_w2"]

        self.norm1_w = block_dict["norm1_w"]
        self.norm2_w = block_dict["norm2_w"]
        self.scale1 = block_dict["scale1"]
        self.scale2 = block_dict["scale2"]

        self.register_buffer("qk_norm_scale", torch.ones(self.dim_head, device=device, dtype=torch.float16), persistent=False)

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * weight

    def _apply_rope_eager(self, q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pairs = rope.shape[-3]
        rot_dim = pairs * 2
        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
        q_rot = q_rot.reshape(*q.shape[:-1], pairs, 1, 2)
        k_rot = k_rot.reshape(*k.shape[:-1], pairs, 1, 2)
        q_rot = torch.matmul(q_rot, rope).squeeze(-2).flatten(-2)
        k_rot = torch.matmul(k_rot, rope).squeeze(-2).flatten(-2)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    def forward(self, x_local: torch.Tensor, rotary_pos_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s_local, dim = x_local.shape
        s_full = s_local * self.world_size

        # --- 1. Attention Sublayer ---
        # 1a. Local Pre-Norm on S/2
        h1_local = self._rms_norm(x_local, self.norm1_w)

        # 1b. Local INT8 ConvRot Activation Quantization on S/2
        qh_local, sh_local = quantize_int8_rowwise_convrot(h1_local.view(-1, dim), self.convrot_groupsize)
        qh_local = qh_local.view(b, s_local, dim)
        sh_local = sh_local.view(b, s_local, 1)

        # 1c. INT8 AllGather over NCCL
        if dist.is_initialized():
            qh_full = torch.empty((b, s_full, dim), dtype=torch.int8, device=self.device)
            dist.all_gather_into_tensor(qh_full, qh_local)
            sh_full = torch.empty((b, s_full, 1), dtype=torch.float32, device=self.device)
            dist.all_gather_into_tensor(sh_full, sh_local)
        else:
            qh_full = qh_local.repeat(1, self.world_size, 1)
            sh_full = sh_local.repeat(1, self.world_size, 1)

        # 1d. Column QKV GEMM from gathered INT8
        qkv = w8a8_gemm(qh_full.view(-1, dim), sh_full.view(-1, 1), self.qw_qkv, self.qs_qkv, self.b_qkv, out_dtype=x_local.dtype)
        qkv = qkv.view(b, s_full, self.heads_per_rank, 3 * self.dim_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 1e. RoPE across full S tokens
        if rotary_pos_emb is not None and HAS_CK and hasattr(comfy_kitchen.backends.cuda, "rms_rope_split_half"):
            q, k = comfy_kitchen.backends.cuda.rms_rope_split_half(
                q, k, rotary_pos_emb, self.qk_norm_scale,
                epsilon=self.eps, rot_dim=rotary_pos_emb.shape[-3] * 2
            )
        else:
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + self.eps)
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + self.eps)
            if rotary_pos_emb is not None:
                q, k = self._apply_rope_eager(q, k, rotary_pos_emb)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scale = 1.0 / math.sqrt(self.dim_head)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=scale)
        attn_out = torch.nan_to_num(attn_out).transpose(1, 2).reshape(b * s_full, self.inner_dim_per_rank)

        # 1f. Row to_out GEMM
        q_attn, s_attn = quantize_int8_rowwise_convrot(attn_out, self.convrot_groupsize)
        partial_out = w8a8_gemm(q_attn, s_attn, self.qw_out, self.qs_out, self.b_out, out_dtype=x_local.dtype)
        partial_out = partial_out.view(b, s_full, dim)

        # 1g. FP16 ReduceScatter over NCCL
        if dist.is_initialized():
            reduced_attn = torch.empty((b, s_local, dim), dtype=x_local.dtype, device=self.device)
            dist.reduce_scatter_tensor(reduced_attn, partial_out.contiguous(), op=dist.ReduceOp.SUM)
        else:
            reduced_attn = partial_out[:, :s_local, :]

        # 1h. Local Residual Addition on S/2
        x_local = x_local + self.scale1 * reduced_attn

        # --- 2. FeedForward Sublayer ---
        # 2a. Local Pre-Norm on S/2
        h2_local = self._rms_norm(x_local, self.norm2_w)

        # 2b. Local INT8 ConvRot Activation Quantization on S/2
        qh2_local, sh2_local = quantize_int8_rowwise_convrot(h2_local.view(-1, dim), self.convrot_groupsize)
        qh2_local = qh2_local.view(b, s_local, dim)
        sh2_local = sh2_local.view(b, s_local, 1)

        # 2c. INT8 AllGather over NCCL
        if dist.is_initialized():
            qh2_full = torch.empty((b, s_full, dim), dtype=torch.int8, device=self.device)
            dist.all_gather_into_tensor(qh2_full, qh2_local)
            sh2_full = torch.empty((b, s_full, 1), dtype=torch.float32, device=self.device)
            dist.all_gather_into_tensor(sh2_full, sh2_local)
        else:
            qh2_full = qh2_local.repeat(1, self.world_size, 1)
            sh2_full = sh2_local.repeat(1, self.world_size, 1)

        # 2d. Column w1 GEMM
        w1_out = w8a8_gemm(qh2_full.view(-1, dim), sh2_full.view(-1, 1), self.qw_w1, self.qs_w1, self.b_w1, out_dtype=x_local.dtype)
        gate, up = torch.chunk(w1_out, 2, dim=-1)
        swiglu = F.silu(gate) * up # [B*S, 4096]

        # 2e. Row w2 GEMM
        q_swi, s_swi = quantize_int8_rowwise_convrot(swiglu, self.convrot_groupsize)
        partial_w2 = w8a8_gemm(q_swi, s_swi, self.qw_w2, self.qs_w2, self.b_w2, out_dtype=x_local.dtype)
        partial_w2 = partial_w2.view(b, s_full, dim)

        # 2f. FP16 ReduceScatter over NCCL
        if dist.is_initialized():
            reduced_w2 = torch.empty((b, s_local, dim), dtype=x_local.dtype, device=self.device)
            dist.reduce_scatter_tensor(reduced_w2, partial_w2.contiguous(), op=dist.ReduceOp.SUM)
        else:
            reduced_w2 = partial_w2[:, :s_local, :]

        # 2g. Local Residual Addition on S/2
        x_local = x_local + self.scale2 * reduced_w2
        return x_local


def patch_video_vae_tp(model: nn.Module, sd: Dict[str, torch.Tensor], rank: int, world_size: int, device: torch.device) -> nn.Module:
    """
    Patches real MiniMaxH3VideoVAE decoder transformer blocks with RealTPTransformerBlock.
    """
    for i in range(len(model.decoder.transformer_blocks)):
        block_dict = create_tp_block_from_sd(sd, i, rank, world_size, device)
        model.decoder.transformer_blocks[i] = RealTPTransformerBlock(
            block_dict, convrot_groupsize=256, device=device
        )
    return model


def patch_video_vae_sp(model: nn.Module, sd: Dict[str, torch.Tensor], rank: int, world_size: int, device: torch.device) -> nn.Module:
    """
    Patches real MiniMaxH3VideoVAE decoder transformer blocks with RealSPTransformerBlock,
    and wraps decoder.forward to partition activations across the sequence dimension.
    """
    for i in range(len(model.decoder.transformer_blocks)):
        block_dict = create_tp_block_from_sd(sd, i, rank, world_size, device)
        model.decoder.transformer_blocks[i] = RealSPTransformerBlock(
            block_dict, world_size=world_size, convrot_groupsize=256, device=device
        )

    orig_decoder_forward = model.decoder.forward

    def sp_decoder_forward(x: torch.Tensor) -> torch.Tensor:
        B, C, latent_T, latent_H, latent_W = x.shape
        h = model.decoder.x_embedder(x.flatten(2).transpose(1, 2))
        num_patches = h.shape[1]

        # Suffix tokens: ensure total S is divisible by world_size
        num_reg = model.decoder.num_register_tokens
        reg = model.decoder.register_tokens.expand(B, -1, -1)
        raw_s = num_patches + num_reg + 1
        num_pad = 1 if (raw_s % world_size == 0) else (1 + (world_size - (raw_s % world_size)))
        num_suffix = num_reg + num_pad

        h = torch.cat([h, reg, torch.zeros((B, num_pad, h.shape[-1]), device=h.device, dtype=h.dtype)], dim=1)

        img_ids = create_token_ids((latent_T, latent_H, latent_W), x.device, x.dtype).expand(B, -1, -1)
        suffix_ids = torch.zeros((B, num_suffix, 3), device=x.device, dtype=img_ids.dtype)
        img_ids = torch.cat([img_ids, suffix_ids], dim=1)
        rotary_pos_emb = model.decoder.pos_embed(img_ids)

        total_s = h.shape[1]
        s_local = total_s // world_size
        h_local = h[:, rank * s_local : (rank + 1) * s_local, :].contiguous()

        for block in model.decoder.transformer_blocks:
            h_local = block(h_local, rotary_pos_emb)

        # Single FP16 AllGather at the end
        if dist.is_initialized():
            h_full = torch.empty((B, total_s, model.decoder.x_embedder.out_features), dtype=h_local.dtype, device=device)
            dist.all_gather_into_tensor(h_full, h_local)
        else:
            h_full = h_local.repeat(1, world_size, 1)

        output = model.decoder.proj_out(model.decoder.norm_out(h_full))
        output = output[:, :num_patches, :]

        output = output.view(
            B, latent_T, latent_H, latent_W,
            model.decoder.out_channels, model.decoder.patch_size_t, model.decoder.patch_size, model.decoder.patch_size,
        )
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        output = output.reshape(
            B, model.decoder.out_channels,
            latent_T * model.decoder.patch_size_t,
            latent_H * model.decoder.patch_size,
            latent_W * model.decoder.patch_size,
        )
        return output

    model.decoder.forward = sp_decoder_forward
    return model


def verify_shard_quantization_invariance(dim: int = 2048, groupsize: int = 256, device: str = "cuda:0") -> Dict[str, Any]:
    """
    Experimental verification that quantizing local sequence shards [S/2, 2048]
    produces bit-exact identical INT8 activations and scales as quantizing the full sequence.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    s_full = 1024
    s_half = s_full // 2
    x = torch.randn(s_full, dim, dtype=torch.float16, device=dev)

    q_full, s_full_scales = quantize_int8_rowwise_convrot(x, groupsize)
    q_0, s_0 = quantize_int8_rowwise_convrot(x[:s_half], groupsize)
    q_1, s_1 = quantize_int8_rowwise_convrot(x[s_half:], groupsize)

    q_cat = torch.cat([q_0, q_1], dim=0)
    s_cat = torch.cat([s_0, s_1], dim=0)

    q_equal = torch.equal(q_full, q_cat)
    s_equal = torch.equal(s_full_scales, s_cat)
    q_max_diff = (q_full.float() - q_cat.float()).abs().max().item()
    s_max_diff = (s_full_scales - s_cat).abs().max().item()

    return {
        "q_equal": q_equal,
        "s_equal": s_equal,
        "q_max_diff": q_max_diff,
        "s_max_diff": s_max_diff,
        "is_bit_exact": q_equal and s_equal,
    }
