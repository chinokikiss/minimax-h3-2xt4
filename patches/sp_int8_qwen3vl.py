"""
Strategy 3: TP=2 + Sequence Parallelism (SP) + INT8 AllGather for MiniMax-H3 Qwen3-VL-32B Text Encoder.
Optimized for 2x NVIDIA Tesla T4 GPUs (16 GB VRAM each, PCIe Gen3, No P2P).

Architecture:
  - Hidden Size: H = 5120
  - Attention Heads: 64 Q heads (head_dim 128 -> 8192 total), 8 KV heads (head_dim 128 -> 1024 total, GQA)
  - MLP Intermediate: I = 25600 (SwiGLU gate_proj, up_proj, down_proj)
  - Layers: 50 layers

Per Layer Execution Pipeline:
  1. Input: Local sequence slice x_local [S/2, H] in FP16.
  2. Local input_layernorm (RMSNorm) computed on [S/2, H] (2x faster than full sequence).
  3. Local INT8 Activation Quantization (with ConvRot Hadamard rotation) on [S/2, H].
  4. INT8 AllGather #1: Gathers quantized tokens [S, H] across GPUs at 1 byte/token (half FP16 volume).
     *Shared across Q, K, V projections to eliminate redundant transfers.*
  5. ColumnParallel QKV GEMMs (W8A8):
     - Q: [S, 4096] (32 heads / rank)
     - K: [S, 512]  (4 heads / rank)
     - V: [S, 512]  (4 heads / rank)
  6. Local QK-Norm (RMSNorm on head_dim 128) + Local GQA Attention compute -> attn_out [S, 4096].
  7. RowParallel O projection GEMM -> partial sum [S, H].
  8. FP16 ReduceScatter #1: Reduces partial sums and scatters along sequence dimension -> [S/2, H].
  9. Local Residual addition: x_local = x_local + attn_out_local on [S/2, H].
  10. Local post_attention_layernorm (RMSNorm) computed on [S/2, H].
  11. Local INT8 Activation Quantization on [S/2, H].
  12. INT8 AllGather #2: Gathers quantized tokens [S, H] at 1 byte/token.
      *Shared across Gate and Up projections.*
  13. ColumnParallel Gate & Up GEMMs (W8A8) -> [S, 12800] each.
  14. Local SwiGLU: silu(gate) * up -> mlp_act [S, 12800].
  15. RowParallel Down projection GEMM -> partial sum [S, H].
  16. FP16 ReduceScatter #2: Reduces partial sums and scatters along sequence dimension -> [S/2, H].
  17. Local Residual addition: x_local = x_local + mlp_out_local on [S/2, H].

Communication per layer:
  - Standard TP=2: 2 AllReduces = 4 * S * H bytes.
  - Strategy 3 (SP + INT8 AllGather): 2 INT8 AllGathers (1 SH) + 2 FP16 ReduceScatters (2 SH) = 3 * S * H bytes.
  -> 25% Reduction in communication payload over PCIe Gen3!
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Tuple

try:
    import comfy_kitchen
    HAS_CK = True
except ImportError:
    HAS_CK = False

def quantize_int8_activation_convrot(x: torch.Tensor, convrot_groupsize: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes activation to row-wise INT8 with group-wise Hadamard rotation (ConvRot).
    Input: x [M, K] in FP16/BF16
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
        # Fallback preserving exact Hadamard ConvRot semantics
        h = comfy_kitchen.backends.cuda._build_hadamard(convrot_groupsize, device=x_2d.device, dtype=x_2d.dtype)
        x_rot = comfy_kitchen.backends.cuda._rotate_activation(x_2d, h, convrot_groupsize)
        qdata, qscale = comfy_kitchen.quantize_int8_rowwise(x_rot)
        return qdata.view(*orig_shape), qscale
    else:
        # High-precision reference quantizer
        amax = torch.amax(torch.abs(x_2d), dim=-1, keepdim=True).clamp(min=1e-5)
        qscale = amax / 127.0
        qdata = torch.clamp(torch.round(x_2d / qscale), -128, 127).to(torch.int8)
        return qdata.view(*orig_shape), qscale

def w8a8_gemm(
    q_act: torch.Tensor,
    scale_act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Computes INT8 x INT8 -> FP16 GEMM with already-quantized activations.
    q_act: [M, K] INT8
    scale_act: [M, 1] FP32
    weight: [N, K] INT8
    weight_scale: [N, 1] or scalar FP32
    """
    m, k = q_act.shape
    n, k_w = weight.shape
    assert k == k_w

    if HAS_CK and hasattr(torch.ops.comfy_kitchen, "cublas_gemm_int8"):
        out_int32 = torch.empty((m, n), dtype=torch.int32, device=q_act.device)
        workspace = torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device=q_act.device)
        torch.ops.comfy_kitchen.cublas_gemm_int8(q_act, weight, out_int32, workspace, 0)
        out = (out_int32.to(torch.float32) * scale_act * weight_scale.T).to(out_dtype)
        return out
    else:
        int32_mm = torch.matmul(q_act.to(torch.float32), weight.T.to(torch.float32))
        out = (int32_mm * scale_act * weight_scale.T).to(out_dtype)
        return out

class SPColumnParallelLinearINT8(nn.Module):
    """
    Sequence-Parallel Column Linear layer for Qwen3-VL:
    Slices output features across TP ranks.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 0,
        world_size: int = 2,
        convrot_groupsize: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert out_features % world_size == 0
        self.in_features = in_features
        self.out_features_per_rank = out_features // world_size
        self.rank = rank
        self.world_size = world_size
        self.convrot_groupsize = convrot_groupsize
        self.device = device or torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features, dtype=torch.int8, device=self.device),
            requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(self.out_features_per_rank, 1, dtype=torch.float32, device=self.device),
            requires_grad=False
        )

    def load_shard_from_full(self, full_weight: torch.Tensor, full_scale: torch.Tensor):
        start = self.rank * self.out_features_per_rank
        end = start + self.out_features_per_rank
        self.weight.data.copy_(full_weight[start:end, :].to(self.device))
        self.weight_scale.data.copy_(full_scale[start:end, :].to(self.device))

    def forward_with_gathered_act(self, q_full: torch.Tensor, s_full_scale: torch.Tensor, out_dtype=torch.float16) -> torch.Tensor:
        """
        Executes GEMM directly on already gathered INT8 activations.
        """
        return w8a8_gemm(q_full, s_full_scale, self.weight, self.weight_scale, out_dtype=out_dtype)

class SPRowParallelLinearINT8(nn.Module):
    """
    Sequence-Parallel Row Linear layer for Qwen3-VL:
    Slices input features across TP ranks.
    Produces local GEMM partial sum -> FP16 ReduceScatter along sequence dimension.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 0,
        world_size: int = 2,
        convrot_groupsize: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features_per_rank = in_features // world_size
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.convrot_groupsize = convrot_groupsize
        self.device = device or torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank, dtype=torch.int8, device=self.device),
            requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32, device=self.device),
            requires_grad=False
        )

    def load_shard_from_full(self, full_weight: torch.Tensor, full_scale: torch.Tensor):
        start = self.rank * self.in_features_per_rank
        end = start + self.in_features_per_rank
        self.weight.data.copy_(full_weight[:, start:end].to(self.device))
        self.weight_scale.data.copy_(full_scale.to(self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [S, in_features_per_rank] in FP16.
        Returns: [S/2, out_features] scattered local chunk in FP16.
        """
        s, in_dim = x.shape
        s_local = s // self.world_size

        # 1. Local Row GEMM -> partial sum [S, out_features]
        q_x, s_x = quantize_int8_activation_convrot(x, self.convrot_groupsize)
        partial_out = w8a8_gemm(q_x, s_x, self.weight, self.weight_scale, out_dtype=x.dtype)

        # 2. FP16 ReduceScatter along Sequence Dimension
        if self.world_size > 1 and dist.is_initialized():
            reduced_chunk = torch.empty((s_local, self.out_features), dtype=x.dtype, device=self.device)
            dist.reduce_scatter_tensor(reduced_chunk, partial_out.contiguous(), op=dist.ReduceOp.SUM)
            return reduced_chunk
        else:
            return partial_out[:s_local, :]

class SPTransformerBlockINT8(nn.Module):
    """
    Sequence Parallel Transformer Block for Qwen3-VL-32B (TP=2 + SP + INT8 AllGather):
    - Activations between blocks remain partitioned at [S/2, H] in FP16.
    - Local RMSNorm on [S/2, H].
    - Single INT8 AllGather per sub-layer (shared across QKV and Gate/Up).
    - Local Attention over full sequence S tokens for rank's head subset.
    - FP16 ReduceScatter after Out and Down projections.
    """
    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 25600,
        num_heads: int = 64,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        rank: int = 0,
        world_size: int = 2,
        convrot_groupsize: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rank = rank
        self.world_size = world_size
        self.convrot_groupsize = convrot_groupsize
        self.device = device or torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        self.num_heads_per_rank = num_heads // world_size       # 32 heads
        self.num_kv_heads_per_rank = num_kv_heads // world_size # 4 heads
        self.q_dim_per_rank = self.num_heads_per_rank * head_dim # 4096
        self.kv_dim_per_rank = self.num_kv_heads_per_rank * head_dim # 512

        # 1. Attention Norms & Linears
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6).to(device=self.device, dtype=torch.float16)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6).to(device=self.device, dtype=torch.float16)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6).to(device=self.device, dtype=torch.float16)

        self.q_proj = SPColumnParallelLinearINT8(hidden_size, num_heads * head_dim, rank, world_size, convrot_groupsize, self.device)
        self.k_proj = SPColumnParallelLinearINT8(hidden_size, num_kv_heads * head_dim, rank, world_size, convrot_groupsize, self.device)
        self.v_proj = SPColumnParallelLinearINT8(hidden_size, num_kv_heads * head_dim, rank, world_size, convrot_groupsize, self.device)
        self.o_proj = SPRowParallelLinearINT8(num_heads * head_dim, hidden_size, rank, world_size, convrot_groupsize, self.device)

        # 2. MLP Norm & Linears
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6).to(device=self.device, dtype=torch.float16)
        self.gate_proj = SPColumnParallelLinearINT8(hidden_size, intermediate_size, rank, world_size, convrot_groupsize, self.device)
        self.up_proj = SPColumnParallelLinearINT8(hidden_size, intermediate_size, rank, world_size, convrot_groupsize, self.device)
        self.down_proj = SPRowParallelLinearINT8(intermediate_size, hidden_size, rank, world_size, convrot_groupsize, self.device)

    def load_from_full_block(self, full_block):
        """Loads and shards weights from full reference block."""
        self.input_layernorm.load_state_dict(full_block.input_layernorm.state_dict())
        self.post_attention_layernorm.load_state_dict(full_block.post_attention_layernorm.state_dict())
        self.q_norm.load_state_dict(full_block.q_norm.state_dict())
        self.k_norm.load_state_dict(full_block.k_norm.state_dict())
        self.q_proj.load_shard_from_full(full_block.q_proj.weight, full_block.q_proj.weight_scale)
        self.k_proj.load_shard_from_full(full_block.k_proj.weight, full_block.k_proj.weight_scale)
        self.v_proj.load_shard_from_full(full_block.v_proj.weight, full_block.v_proj.weight_scale)
        self.o_proj.load_shard_from_full(full_block.o_proj.weight, full_block.o_proj.weight_scale)
        self.gate_proj.load_shard_from_full(full_block.gate_proj.weight, full_block.gate_proj.weight_scale)
        self.up_proj.load_shard_from_full(full_block.up_proj.weight, full_block.up_proj.weight_scale)
        self.down_proj.load_shard_from_full(full_block.down_proj.weight, full_block.down_proj.weight_scale)

    def forward(self, x_local: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_local: [S/2, H] in FP16.
        Returns: [S/2, H] in FP16.
        """
        s_local, H = x_local.shape
        total_s = s_local * self.world_size

        # -------------------------------------------------------------
        # 1. Attention Sub-Layer
        # -------------------------------------------------------------
        residual = x_local

        # 1a. Local RMSNorm on [S/2, H]
        normed_local = self.input_layernorm(x_local)

        # 1b. Local INT8 Activation Quantization on [S/2, H]
        q_local, s_local_scale = quantize_int8_activation_convrot(normed_local, self.convrot_groupsize)

        # 1c. Shared INT8 AllGather across GPUs (1 byte/elem vs 2 bytes FP16)
        if self.world_size > 1 and dist.is_initialized():
            q_full = torch.empty((total_s, H), dtype=torch.int8, device=self.device)
            dist.all_gather_into_tensor(q_full, q_local)
            s_full_scale = torch.empty((total_s, 1), dtype=torch.float32, device=self.device)
            dist.all_gather_into_tensor(s_full_scale, s_local_scale)
        else:
            q_full = q_local.repeat(self.world_size, 1)
            s_full_scale = s_local_scale.repeat(self.world_size, 1)

        # 1d. Shared INT8 activations fed to ColumnParallel Q, K, V
        q = self.q_proj.forward_with_gathered_act(q_full, s_full_scale, out_dtype=x_local.dtype) # [S, 4096]
        k = self.k_proj.forward_with_gathered_act(q_full, s_full_scale, out_dtype=x_local.dtype) # [S, 512]
        v = self.v_proj.forward_with_gathered_act(q_full, s_full_scale, out_dtype=x_local.dtype) # [S, 512]

        # 1e. Head reshaping & QK-Norm
        q = q.view(total_s, self.num_heads_per_rank, self.head_dim)
        k = k.view(total_s, self.num_kv_heads_per_rank, self.head_dim)
        v = v.view(total_s, self.num_kv_heads_per_rank, self.head_dim)

        q = self.q_norm(q).transpose(0, 1).unsqueeze(0) # [1, 32, S, 128]
        k = self.k_norm(k).transpose(0, 1).unsqueeze(0) # [1, 4, S, 128]
        v = v.transpose(0, 1).unsqueeze(0)              # [1, 4, S, 128]

        # 1f. GQA head repeating
        if self.num_heads_per_rank != self.num_kv_heads_per_rank:
            ratio = self.num_heads_per_rank // self.num_kv_heads_per_rank
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)

        # Local GQA Attention compute
        if HAS_CK and callable(getattr(comfy_kitchen, "flash_attention", None)):
            attn_out = comfy_kitchen.flash_attention(q, k, v)
        else:
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.squeeze(0).transpose(0, 1).contiguous().view(total_s, self.q_dim_per_rank) # [S, 4096]

        # 1g. RowParallel O projection + FP16 ReduceScatter -> scattered [S/2, H]
        attn_out_local = self.o_proj(attn_out)

        # 1h. Local Residual Addition on [S/2, H]
        x_local = residual + attn_out_local

        # -------------------------------------------------------------
        # 2. MLP Sub-Layer
        # -------------------------------------------------------------
        residual = x_local

        # 2a. Local RMSNorm on [S/2, H]
        normed2_local = self.post_attention_layernorm(x_local)

        # 2b. Local INT8 Activation Quantization on [S/2, H]
        q2_local, s2_local_scale = quantize_int8_activation_convrot(normed2_local, self.convrot_groupsize)

        # 2c. Shared INT8 AllGather across GPUs (1 byte/elem)
        if self.world_size > 1 and dist.is_initialized():
            q2_full = torch.empty((total_s, H), dtype=torch.int8, device=self.device)
            dist.all_gather_into_tensor(q2_full, q2_local)
            s2_full_scale = torch.empty((total_s, 1), dtype=torch.float32, device=self.device)
            dist.all_gather_into_tensor(s2_full_scale, s2_local_scale)
        else:
            q2_full = q2_local.repeat(self.world_size, 1)
            s2_full_scale = s2_local_scale.repeat(self.world_size, 1)

        # 2d. Shared INT8 activations fed to Gate & Up
        gate = self.gate_proj.forward_with_gathered_act(q2_full, s2_full_scale, out_dtype=x_local.dtype) # [S, 12800]
        up = self.up_proj.forward_with_gathered_act(q2_full, s2_full_scale, out_dtype=x_local.dtype)     # [S, 12800]

        # 2e. Local SwiGLU activation
        mlp_act = torch.nn.functional.silu(gate) * up # [S, 12800]

        # 2f. RowParallel Down projection + FP16 ReduceScatter -> scattered [S/2, H]
        mlp_out_local = self.down_proj(mlp_act)

        # 2g. Local Residual Addition on [S/2, H]
        x_local = residual + mlp_out_local

        return x_local
