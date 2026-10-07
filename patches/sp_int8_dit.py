"""
Strategy 3: TP=2 + Sequence Parallelism (SP) + INT8 AllGather for MiniMax-H3 DiT.
Pipeline:
  1. RowParallel projection produces partial sums.
  2. FP16 ReduceScatter -> reduces partial sums, scattering local [S/2, H] chunk to each GPU.
  3. Local Residual & RMSNorm/AdaLN computed entirely on local [S/2, H] slice.
  4. INT8 Activation Quantization (with ConvRot Hadamard rotation) applied locally on [S/2, H].
  5. INT8 AllGather -> gathers quantized INT8 tokens [S, H] (cutting communication volume by 50% vs FP16).
  6. W8A8 GEMM executed on gathered INT8 activations.

Reuses the existing comfy_kitchen / INT8 ConvRot activation quantizer to guarantee
zero additional quantization error beyond the baseline W8A8 format.
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
    x_2d = x.view(-1, orig_shape[-1])
    m, k = x_2d.shape

    if HAS_CK and hasattr(torch.ops.comfy_kitchen, "quantize_int8_rowwise_convrot64"):
        qdata = torch.empty((m, k), dtype=torch.int8, device=x.device)
        qscale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        # Call comfy_kitchen fused quantizer
        torch.ops.comfy_kitchen.quantize_int8_rowwise_convrot64(
            x_2d, qdata, qscale, convrot_groupsize, False, 0, 0, 0
        )
        return qdata.view(*orig_shape), qscale
    elif HAS_CK and hasattr(comfy_kitchen, "quantize_int8_rowwise"):
        # Builtin row-wise quantizer
        qdata, qscale = comfy_kitchen.quantize_int8_rowwise(x_2d)
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
        # Dequantize: int32 * scale_act * weight_scale -> out_dtype
        out = (out_int32.to(torch.float32) * scale_act * weight_scale.T).to(out_dtype)
        return out
    else:
        # Native PyTorch INT8 GEMM / fallback
        int32_mm = torch.matmul(q_act.to(torch.float32), weight.T.to(torch.float32))
        out = (int32_mm * scale_act * weight_scale.T).to(out_dtype)
        return out

class SPColumnParallelLinearINT8(nn.Module):
    """
    Sequence-Parallel Column Linear:
    Receives local sequence chunk [S/2, H], quantizes to INT8, performs INT8 AllGather -> [S, H],
    then computes column-parallel GEMM -> [S, Out // world_size].
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

    def forward(self, x_local: torch.Tensor) -> torch.Tensor:
        """
        x_local: [S/2, in_features] in FP16.
        Returns: [S, out_features_per_rank] in FP16.
        """
        s_local, h = x_local.shape
        total_s = s_local * self.world_size

        # 1. Local INT8 Activation Quantization on [S/2, H]
        q_local, s_local_scale = quantize_int8_activation_convrot(x_local, self.convrot_groupsize)

        # 2. INT8 AllGather over PCIe: 1 byte per element instead of 2!
        if self.world_size > 1 and dist.is_initialized():
            q_full = torch.empty((total_s, h), dtype=torch.int8, device=self.device)
            dist.all_gather_into_tensor(q_full, q_local)

            s_full_scale = torch.empty((total_s, 1), dtype=torch.float32, device=self.device)
            dist.all_gather_into_tensor(s_full_scale, s_local_scale)
        else:
            # Single device / simulation
            q_full = q_local.repeat(self.world_size, 1)
            s_full_scale = s_local_scale.repeat(self.world_size, 1)

        # 3. W8A8 GEMM on gathered INT8 activations
        out = w8a8_gemm(
            q_full, s_full_scale, self.weight, self.weight_scale, out_dtype=x_local.dtype
        )
        return out # [S, out_features_per_rank]

class SPRowParallelLinearINT8(nn.Module):
    """
    Sequence-Parallel Row Linear:
    Computes local GEMM [S, In // world_size] -> partial sum [S, Out],
    then performs FP16 ReduceScatter -> scattered local chunk [S/2, Out].
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [S, in_features_per_rank] (intermediate output from ColumnParallel/Attention/MLP)
        Returns: [S/2, out_features] scattered local chunk in FP16.
        """
        s, in_dim = x.shape
        s_local = s // self.world_size

        # 1. Local Row GEMM -> partial sum [S, out_features]
        # Quantize x [S, in_dim] locally
        q_x, s_x = quantize_int8_activation_convrot(x, self.convrot_groupsize)
        partial_out = w8a8_gemm(q_x, s_x, self.weight, self.weight_scale, out_dtype=x.dtype)

        # 2. FP16 ReduceScatter along Sequence Dimension
        if self.world_size > 1 and dist.is_initialized():
            reduced_chunk = torch.empty((s_local, self.out_features), dtype=x.dtype, device=self.device)
            dist.reduce_scatter_tensor(reduced_chunk, partial_out.contiguous(), op=dist.ReduceOp.SUM)
            return reduced_chunk
        else:
            # Single device / simulation
            return partial_out[:s_local, :]

class SPDiTBlockINT8(nn.Module):
    """
    DiT Block with Strategy 3 (TP=2 + Sequence Parallelism + INT8 AllGather):
    - Activations between blocks remain partitioned at [S/2, H] in FP16.
    - ReduceScatter after RowParallel projections.
    - Local Residual & AdaLN/RMSNorm on [S/2, H].
    - INT8 AllGather before Attention QKV and MLP FC1.
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

        # Local RMSNorms & AdaLN (all operating on [S/2, H])
        self.norm1 = nn.RMSNorm(hidden, eps=eps).to(device=self.device, dtype=torch.float16)
        self.norm2 = nn.RMSNorm(hidden, eps=eps).to(device=self.device, dtype=torch.float16)
        self.q_norm = nn.RMSNorm(head_dim, eps=qk_eps).to(device=self.device, dtype=torch.float16)
        self.k_norm = nn.RMSNorm(head_dim, eps=qk_eps).to(device=self.device, dtype=torch.float16)

        self.adaln_proj = nn.Linear(t_dim, 6 * 3 * hidden, bias=True).to(device=self.device, dtype=torch.float16)

        # Sequence-Parallel Attention
        self.qkv_proj = SPColumnParallelLinearINT8(
            in_features=hidden,
            out_features=heads * head_dim * 3,
            rank=rank,
            world_size=world_size,
            device=self.device,
        )
        self.out_proj = SPRowParallelLinearINT8(
            in_features=heads * head_dim,
            out_features=hidden,
            rank=rank,
            world_size=world_size,
            device=self.device,
        )

        # Sequence-Parallel MLP
        self.fc1 = SPColumnParallelLinearINT8(
            in_features=hidden,
            out_features=ffn * 2,
            rank=rank,
            world_size=world_size,
            device=self.device,
        )
        self.fc2 = SPRowParallelLinearINT8(
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
            if a < h.shape[0]:
                end = min(b, h.shape[0])
                h[a:end].mul_(1.0 + self._mod_row(scale, row, h.dtype)).add_(self._mod_row(shift, row, h.dtype))
        return h

    def _mod_gate(self, x, gate, other, segments):
        for a, b, row in segments:
            if a < x.shape[0]:
                end = min(b, x.shape[0])
                x[a:end].addcmul_(other[a:end], self._mod_row(gate, row, x.dtype))
        return x

    def forward(
        self,
        x_local: torch.Tensor,
        t_emb: torch.Tensor,
        mod_segments: List[Any],
        rope_freqs: Optional[torch.Tensor] = None,
        transformer_options: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """
        x_local: [S/2, H] in FP16.
        Returns: [S/2, H] in FP16.
        """
        # 1. Local Modulation Vectors
        adaln_out = self.adaln_proj(nn.functional.silu(t_emb)).view(-1, 6 * self.hidden)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_out.chunk(6, dim=-1)

        # 2. Local Norm & Modulation on [S/2, H]
        normed_x_local = self.norm1(x_local)
        h_local = self._mod_scale_shift(normed_x_local, shift_msa, scale_msa, mod_segments)

        # 3. INT8 AllGather + ColumnParallel QKV -> [S, 3 * inner_per_rank]
        qkv = self.qkv_proj(h_local)
        q, k, v = qkv.split(self.inner_per_rank, dim=-1)
        S = q.shape[0]
        q = q.view(S, self.heads_per_rank, self.head_dim)
        k = k.view(S, self.heads_per_rank, self.head_dim)
        v = v.view(S, self.heads_per_rank, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Local Attention Compute
        q_heads = q.transpose(0, 1).unsqueeze(0)
        k_heads = k.transpose(0, 1).unsqueeze(0)
        v_heads = v.transpose(0, 1).unsqueeze(0)

        attn_out = nn.functional.scaled_dot_product_attention(q_heads, k_heads, v_heads)
        attn_out = attn_out.squeeze(0).transpose(0, 1).contiguous().view(S, self.inner_per_rank)

        # 5. RowParallel Out + FP16 ReduceScatter -> [S/2, H]
        attn_out_local = self.out_proj(attn_out)

        # 6. Local Residual on [S/2, H]
        x_local = self._mod_gate(x_local, gate_msa, attn_out_local, mod_segments)

        # 7. Local Norm & Modulation on [S/2, H]
        normed_x2_local = self.norm2(x_local)
        h2_local = self._mod_scale_shift(normed_x2_local, shift_mlp, scale_mlp, mod_segments)

        # 8. INT8 AllGather + ColumnParallel FC1 -> [S, 2 * ffn_per_rank]
        fc1_out = self.fc1(h2_local)
        gate, up = fc1_out.chunk(2, dim=-1)
        mlp_act = nn.functional.silu(gate) * up

        # 9. RowParallel FC2 + FP16 ReduceScatter -> [S/2, H]
        mlp_out_local = self.fc2(mlp_act)

        # 10. Local Residual on [S/2, H]
        x_local = self._mod_gate(x_local, gate_mlp, mlp_out_local, mod_segments)

        return x_local
