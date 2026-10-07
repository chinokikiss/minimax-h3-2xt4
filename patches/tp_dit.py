"""
Megatron-style Tensor Parallelism (TP=2) for MiniMax-H3 Ref2VA Pruned INT8 ConvRot DiT.
Features:
  - ColumnParallelLinearINT8: qkv_proj (28 heads per rank), fc1 (14336 intermediate per rank)
  - RowParallelLinearINT8: out_proj (3584 in -> 5376 out), fc2 (7168 in -> 5376 out)
  - FP16 AllReduce after out_proj and fc2 (2 AllReduces per block, 100 AllReduces total for 50 blocks)
  - Replicated AdaLN projection and RMSNorms executed locally without communication
  - Preserves exact INT8 ConvRot Hadamard block-orthogonal quantization semantics
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, List

try:
    import comfy_kitchen
    HAS_CK = True
except ImportError:
    HAS_CK = False

def run_int8_linear_forward(x, weight, weight_scale, convrot=True, convrot_groupsize=256, out_dtype=torch.float16):
    """
    Executes INT8 ConvRot linear projection with comfy_kitchen or fallback.
    """
    if HAS_CK:
        return comfy_kitchen.int8_linear(
            x, weight, weight_scale, None, out_dtype,
            convrot=convrot, convrot_groupsize=convrot_groupsize
        )
    else:
        # Fallback: dequantize on the fly
        w_f16 = weight.to(out_dtype) * weight_scale.to(out_dtype)
        return torch.nn.functional.linear(x, w_f16)

class ColumnParallelLinearINT8(nn.Module):
    """
    Column Parallel INT8 Linear layer:
    Splits output dimension across TP ranks.
    Input: [S, in_features]
    Output: [S, out_features // world_size]
    Communication: None (independent local GEMM)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 0,
        world_size: int = 2,
        convrot: bool = True,
        convrot_groupsize: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert out_features % world_size == 0
        self.in_features = in_features
        self.out_features_per_rank = out_features // world_size
        self.rank = rank
        self.world_size = world_size
        self.convrot = convrot
        self.convrot_groupsize = convrot_groupsize
        self.device = device or torch.device(f"cuda:{rank}")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.view(-1, self.in_features)
        out = run_int8_linear_forward(
            x_2d, self.weight, self.weight_scale,
            convrot=self.convrot, convrot_groupsize=self.convrot_groupsize,
            out_dtype=x.dtype
        )
        return out.view(*orig_shape[:-1], self.out_features_per_rank)

class RowParallelLinearINT8(nn.Module):
    """
    Row Parallel INT8 Linear layer:
    Splits input dimension across TP ranks.
    Input: [S, in_features // world_size]
    Output: [S, out_features]
    Communication: AllReduce (SUM) across all TP ranks
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 0,
        world_size: int = 2,
        convrot: bool = True,
        convrot_groupsize: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features_per_rank = in_features // world_size
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.convrot = convrot
        self.convrot_groupsize = convrot_groupsize
        self.device = device or torch.device(f"cuda:{rank}")

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

    def forward(self, x: torch.Tensor, reduce_sum: bool = True) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.view(-1, self.in_features_per_rank)
        out = run_int8_linear_forward(
            x_2d, self.weight, self.weight_scale,
            convrot=self.convrot, convrot_groupsize=self.convrot_groupsize,
            out_dtype=x.dtype
        )
        out = out.view(*orig_shape[:-1], self.out_features)

        if reduce_sum and self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)

        return out

class TPDiTBlockINT8(nn.Module):
    """
    Tensor Parallel DiT Block for MiniMax-H3:
    Hidden: 5376, Heads: 56 (28/rank), HeadDim: 128, FFN: 14336 (7168/rank).
    """
    def __init__(
        self,
        hidden: int = 5376,
        heads: int = 56,
        head_dim: int = 128,
        ffn: int = 14336,
        t_dim: int = 8,
        rank: int = 0,
        world_size: int = 2,
        eps: float = 1e-6,
        qk_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.rank = rank
        self.world_size = world_size
        self.heads_per_rank = heads // world_size
        self.inner_per_rank = self.heads_per_rank * head_dim
        self.device = torch.device(f"cuda:{rank}")

        # Replicated modules (local compute, no communication)
        self.norm1 = nn.RMSNorm(hidden, eps=eps).to(device=self.device, dtype=torch.float16)
        self.norm2 = nn.RMSNorm(hidden, eps=eps).to(device=self.device, dtype=torch.float16)
        self.q_norm = nn.RMSNorm(head_dim, eps=qk_eps).to(device=self.device, dtype=torch.float16)
        self.k_norm = nn.RMSNorm(head_dim, eps=qk_eps).to(device=self.device, dtype=torch.float16)

        # Adaln projection: [t_dim -> 6 * 3 * hidden] = [8 -> 96768] (tiny, replicated)
        self.adaln_proj = nn.Linear(t_dim, 6 * 3 * hidden, bias=True).to(device=self.device, dtype=torch.float16)

        # Attention: ColumnParallel QKV + RowParallel Out
        self.qkv_proj = ColumnParallelLinearINT8(
            in_features=hidden,
            out_features=heads * head_dim * 3,
            rank=rank,
            world_size=world_size,
            device=self.device,
        )
        self.out_proj = RowParallelLinearINT8(
            in_features=heads * head_dim,
            out_features=hidden,
            rank=rank,
            world_size=world_size,
            device=self.device,
        )

        # MLP: ColumnParallel FC1 (SwiGLU gate+up) + RowParallel FC2
        self.fc1 = ColumnParallelLinearINT8(
            in_features=hidden,
            out_features=ffn * 2,
            rank=rank,
            world_size=world_size,
            device=self.device,
        )
        self.fc2 = RowParallelLinearINT8(
            in_features=ffn,
            out_features=hidden,
            rank=rank,
            world_size=world_size,
            device=self.device,
        )

    def load_from_full_block(self, full_block):
        """Loads and shards weights from full-sized reference block."""
        self.norm1.load_state_dict(full_block.norm1.state_dict())
        self.norm2.load_state_dict(full_block.norm2.state_dict())
        self.q_norm.load_state_dict(full_block.q_norm.state_dict())
        self.k_norm.load_state_dict(full_block.k_norm.state_dict())
        self.adaln_proj.load_state_dict(full_block.adaln_proj.state_dict())
        self.qkv_proj.load_shard_from_full(full_block.qkv_proj.weight, full_block.qkv_proj.weight_scale)
        self.out_proj.load_shard_from_full(full_block.out_proj.weight, full_block.out_proj.weight_scale)
        self.fc1.load_shard_from_full(full_block.fc1.weight, full_block.fc1.weight_scale)
        self.fc2.load_shard_from_full(full_block.fc2.weight, full_block.fc2.weight_scale)

    def _mod_row(self, vecs, row, dtype):
        return vecs[row].to(dtype)

    def _mod_scale_shift(self, h, shift, scale, segments):
        for a, b, row in segments:
            h[a:b].mul_(1.0 + self._mod_row(scale, row, h.dtype)).add_(self._mod_row(shift, row, h.dtype))
        return h

    def _mod_gate(self, x, gate, other, segments):
        for a, b, row in segments:
            x[a:b].addcmul_(other[a:b], self._mod_row(gate, row, x.dtype))
        return x

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        mod_segments: List[Any],
        rope_freqs: Optional[torch.Tensor] = None,
        transformer_options: Dict[str, Any] = {},
    ) -> torch.Tensor:
        # 1. Modulation parameters
        adaln_out = self.adaln_proj(nn.functional.silu(t_emb))
        adaln_out = adaln_out.view(-1, 6 * self.hidden)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_out.chunk(6, dim=-1)

        # 2. Attention Block
        normed_x = self.norm1(x)
        h = self._mod_scale_shift(normed_x, shift_msa, scale_msa, mod_segments)

        # ColumnParallel QKV
        qkv = self.qkv_proj(h) # [S, 3 * inner_per_rank]
        q, k, v = qkv.split(self.inner_per_rank, dim=-1)
        S = q.shape[0]
        q = q.view(S, self.heads_per_rank, self.head_dim)
        k = k.view(S, self.heads_per_rank, self.head_dim)
        v = v.view(S, self.heads_per_rank, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Local Attention
        q_heads = q.transpose(0, 1).unsqueeze(0) # [1, heads_per_rank, S, head_dim]
        k_heads = k.transpose(0, 1).unsqueeze(0)
        v_heads = v.transpose(0, 1).unsqueeze(0)

        attn_out = nn.functional.scaled_dot_product_attention(q_heads, k_heads, v_heads)
        attn_out = attn_out.squeeze(0).transpose(0, 1).contiguous().view(S, self.inner_per_rank)

        # RowParallel Out + AllReduce #1 (SUM, FP16)
        attn_out = self.out_proj(attn_out, reduce_sum=True)
        x = self._mod_gate(x, gate_msa, attn_out, mod_segments)

        # 3. MLP Block (SwiGLU)
        normed_x2 = self.norm2(x)
        h2 = self._mod_scale_shift(normed_x2, shift_mlp, scale_mlp, mod_segments)

        fc1_out = self.fc1(h2) # [S, 2 * ffn_per_rank]
        gate, up = fc1_out.chunk(2, dim=-1)
        mlp_act = nn.functional.silu(gate) * up

        # RowParallel FC2 + AllReduce #2 (SUM, FP16)
        mlp_out = self.fc2(mlp_act, reduce_sum=True)
        x = self._mod_gate(x, gate_mlp, mlp_out, mod_segments)

        return x
