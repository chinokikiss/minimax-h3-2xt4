"""
Tensor Parallelism (TP=2) for MiniMax-H3 Qwen3-VL-32B Text Encoder
Supports PyTorch Distributed (NCCL) with INT8 ConvRot execution.

Topology:
  - World Size: 2 (Rank 0 = GPU 0, Rank 1 = GPU 1)
  - Interconnect: PCIe Gen3 x16 (NO P2P -> NCCL_P2P_DISABLE=1)

Per Layer Operations:
  - ColumnParallelLinear: q_proj (32 heads), k_proj (4 heads), v_proj (4 heads)
  - Attention local compute
  - RowParallelLinear: o_proj + AllReduce(SUM)
  - ColumnParallelLinear: gate_proj (12800), up_proj (12800)
  - Local SwiGLU activation
  - RowParallelLinear: down_proj + AllReduce(SUM)
  - Total: 2 AllReduces per layer (100 AllReduces for 50 layers)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import comfy_kitchen

class ColumnParallelLinearINT8(nn.Module):
    """
    Column Parallel Linear layer:
    Splits output features across TP ranks.
    Input: [B, S, H]
    Output: [B, S, Out_features // TP_WORLD_SIZE]
    Communication: None (independent local GEMM)
    """
    def __init__(self, in_features, out_features, rank=0, world_size=2, convrot=True, convrot_groupsize=256):
        super().__init__()
        assert out_features % world_size == 0
        self.in_features = in_features
        self.out_features_per_rank = out_features // world_size
        self.rank = rank
        self.world_size = world_size
        self.convrot = convrot
        self.convrot_groupsize = convrot_groupsize

        # Sliced INT8 weights: [out_features_per_rank, in_features]
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features, dtype=torch.int8),
            requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(self.out_features_per_rank, 1, dtype=torch.float32),
            requires_grad=False
        )

    def load_shard_from_full(self, full_weight, full_scale):
        start = self.rank * self.out_features_per_rank
        end = start + self.out_features_per_rank
        self.weight.data.copy_(full_weight[start:end, :])
        self.weight_scale.data.copy_(full_scale[start:end, :])

    def forward(self, x):
        # x: [B, S, in_features]
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        out = comfy_kitchen.int8_linear(
            x_flat, self.weight, self.weight_scale, None, x.dtype,
            convrot=self.convrot, convrot_groupsize=self.convrot_groupsize
        )
        return out.view(*orig_shape[:-1], self.out_features_per_rank)


class RowParallelLinearINT8(nn.Module):
    """
    Row Parallel Linear layer:
    Splits input features across TP ranks.
    Input: [B, S, In_features // TP_WORLD_SIZE]
    Output: [B, S, out_features]
    Communication: AllReduce (SUM) across all TP ranks
    """
    def __init__(self, in_features, out_features, rank=0, world_size=2, convrot=True, convrot_groupsize=256):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features_per_rank = in_features // world_size
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.convrot = convrot
        self.convrot_groupsize = convrot_groupsize

        # Sliced INT8 weights: [out_features, in_features_per_rank]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank, dtype=torch.int8),
            requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32),
            requires_grad=False
        )

    def load_shard_from_full(self, full_weight, full_scale):
        start = self.rank * self.in_features_per_rank
        end = start + self.in_features_per_rank
        self.weight.data.copy_(full_weight[:, start:end])
        self.weight_scale.data.copy_(full_scale)

    def forward(self, x, reduce_sum=True):
        # x: [B, S, in_features_per_rank]
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features_per_rank)
        out = comfy_kitchen.int8_linear(
            x_flat, self.weight, self.weight_scale, None, x.dtype,
            convrot=self.convrot, convrot_groupsize=self.convrot_groupsize
        )
        out = out.view(*orig_shape[:-1], self.out_features)

        if reduce_sum and self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)

        return out


class TPTransformerBlockINT8(nn.Module):
    """
    Tensor Parallel Transformer block for Qwen3-VL-32B:
    Attention:
      - Q: ColumnParallel (32 heads / rank)
      - K: ColumnParallel (4 heads / rank)
      - V: ColumnParallel (4 heads / rank)
      - O: RowParallel + AllReduce
    MLP:
      - Gate: ColumnParallel (12800 / rank)
      - Up: ColumnParallel (12800 / rank)
      - Down: RowParallel + AllReduce
    """
    def __init__(self, hidden_size=5120, intermediate_size=25600, num_heads=64, num_kv_heads=8, head_dim=128, rank=0, world_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.world_size = world_size
        self.num_heads_per_rank = num_heads // world_size
        self.num_kv_heads_per_rank = num_kv_heads // world_size
        self.head_dim = head_dim

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.q_proj = ColumnParallelLinearINT8(hidden_size, num_heads * head_dim, rank, world_size)
        self.k_proj = ColumnParallelLinearINT8(hidden_size, num_kv_heads * head_dim, rank, world_size)
        self.v_proj = ColumnParallelLinearINT8(hidden_size, num_kv_heads * head_dim, rank, world_size)
        self.o_proj = RowParallelLinearINT8(num_heads * head_dim, hidden_size, rank, world_size)

        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.gate_proj = ColumnParallelLinearINT8(hidden_size, intermediate_size, rank, world_size)
        self.up_proj = ColumnParallelLinearINT8(hidden_size, intermediate_size, rank, world_size)
        self.down_proj = RowParallelLinearINT8(intermediate_size, hidden_size, rank, world_size)

    def forward(self, x, attention_mask=None, freqs_cis=None):
        # 1. Self Attention
        residual = x
        normed_x = self.input_layernorm(x)

        q = self.q_proj(normed_x) # [B, S, 32 * 128 = 4096]
        k = self.k_proj(normed_x) # [B, S, 4 * 128 = 512]
        v = self.v_proj(normed_x) # [B, S, 4 * 128 = 512]

        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads_per_rank, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads_per_rank, self.head_dim).transpose(1, 2)

        # GQA local attention
        attn_out = comfy_kitchen.flash_attention(q, k, v) if hasattr(comfy_kitchen, "flash_attention") else torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)

        # RowParallel O projection + AllReduce 1
        attn_out = self.o_proj(attn_out, reduce_sum=True)
        x = residual + attn_out

        # 2. MLP
        residual = x
        normed_x = self.post_attention_layernorm(x)

        gate = self.gate_proj(normed_x) # [B, S, 12800]
        up = self.up_proj(normed_x)     # [B, S, 12800]
        mlp_act = torch.nn.functional.silu(gate) * up

        # RowParallel Down projection + AllReduce 2
        mlp_out = self.down_proj(mlp_act, reduce_sum=True)
        x = residual + mlp_out

        return x
