"""
Megatron-style Tensor Parallelism (TP=2) for MiniMax-H3 Ref2VA Pruned INT8 ConvRot DiT.
Features:
  - ColumnParallelLinearINT8: qkv_proj (28 heads per rank), fc1 (14336 intermediate per rank)
  - RowParallelLinearINT8: out_proj (3584 in -> 5376 out), fc2 (7168 in -> 5376 out)
  - Gather complete activation rows, then INT32 AllReduce before dequantization
  - Replicated AdaLN projection and RMSNorms executed locally without communication
  - Preserves exact INT8 ConvRot Hadamard block-orthogonal quantization semantics
"""

from patches.parallel_int8 import row_linear, packed_rows
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, List

import comfy_kitchen

def run_int8_linear_forward(x, weight, weight_scale, convrot=True, convrot_groupsize=256, out_dtype=torch.float16,backend='preallocated'):
    from patches.dit_column_linear import column_linear
    return column_linear(x, weight, weight_scale, out_dtype=out_dtype,
        convrot=convrot, convrot_groupsize=convrot_groupsize,backend=backend)

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
        components: int = 1,
    ):
        super().__init__()
        assert out_features % world_size == 0
        self.components = components
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
        w, scale = packed_rows(full_weight, full_scale, self.rank, self.world_size, self.components)
        self.weight.data.copy_(w)
        self.weight_scale.data.copy_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.view(-1, self.in_features)
        policy=getattr(self,'operator_policy',None)
        options={} if policy is None or policy.column_linear_backend=='preallocated' else dict(backend=policy.column_linear_backend)
        out = run_int8_linear_forward(
            x_2d, self.weight, self.weight_scale,
            convrot=self.convrot, convrot_groupsize=self.convrot_groupsize,
            out_dtype=x.dtype,**options
        )
        out=out.view(*orig_shape[:-1], self.out_features_per_rank)
        adapter=getattr(self,'lora_adapter',None)
        return adapter.add_to(x,out) if adapter is not None else out

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

    def forward(self, x: torch.Tensor, reduce_sum: bool = True, input_act=None,
                *, residual=None,gate=None,segments=None) -> torch.Tensor:
        if not reduce_sum and self.world_size > 1:
            raise ValueError("Quantization-preserving projection requires reduction")
        if residual is not None:
            from patches.tilelang_dit_fusion import row_gated
            fused=row_gated(self,x,residual,gate,segments,dist,input_act)
            if fused is not None:return fused
        from patches.h3_operator_policy import row_linear as dispatch_row
        out=dispatch_row(self,x,row_linear,dist,input_act)
        adapter=getattr(self,'lora_adapter',None)
        if adapter is not None:
            # Fused preparation/transport returns FP32 until LoRA is added.
            # Preserve the old path's dtype/order when no FP32 island is used.
            source=x.float() if out.dtype==torch.float32 else x
            out=adapter.add_to(source,out,comm=dist if self.world_size>1 else None,input_act=input_act)
        out=out.to(x.dtype)
        if residual is not None:
            from patches.h3_tensor_flow import gated_add
            return gated_add(residual,gate,out,segments,
                fused=getattr(getattr(self,'operator_policy',None),'dit_fused_gate',False))
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
        eps: float = 1e-5,
        qk_eps: float = 1e-5,
        device=None, dtype=torch.float32, apply_silu=False,
    ):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.rank = rank
        self.world_size = world_size
        self.heads_per_rank = heads // world_size
        self.inner_per_rank = self.heads_per_rank * head_dim
        self.device = torch.device(device or f"cuda:{rank}")
        self.apply_silu = apply_silu

        # Replicated modules (local compute, no communication)
        self.norm1 = nn.RMSNorm(hidden, eps=eps).to(device=self.device, dtype=dtype)
        self.norm2 = nn.RMSNorm(hidden, eps=eps).to(device=self.device, dtype=dtype)
        self.q_norm = nn.RMSNorm(head_dim, eps=qk_eps).to(device=self.device, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=qk_eps).to(device=self.device, dtype=dtype)

        # Adaln projection: [t_dim -> 6 * 3 * hidden] = [8 -> 96768] (tiny, replicated)
        self.adaln_proj = nn.Linear(t_dim, 6 * 3 * hidden, bias=True).to(device=self.device, dtype=dtype)

        # Attention: ColumnParallel QKV + RowParallel Out
        self.qkv_proj = ColumnParallelLinearINT8(
            in_features=hidden,
            out_features=heads * head_dim * 3,
            components=3,
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
            components=2,
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
        sd = full_block.state_dict()
        for name in ['norm1', 'norm2', 'q_norm', 'k_norm']:
            key = name if name.startswith('norm') else 'attn.' + name
            getattr(self, name).load_state_dict({'weight': sd[key + '.weight']})
        self.adaln_proj.load_state_dict({'weight': sd['adaln_proj.linear.weight'], 'bias': sd['adaln_proj.linear.bias']})
        self.apply_silu = full_block.adaln_proj.apply_silu
        for name, key in {'qkv_proj': 'attn.qkv_proj', 'out_proj': 'attn.out_proj', 'fc1': 'mlp.fc1', 'fc2': 'mlp.fc2'}.items():
            getattr(self, name).load_shard_from_full(sd[key + '.weight'], sd[key + '.weight_scale'])

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
        flow=getattr(self,'tensor_flow',None)
        policy=getattr(self,'operator_policy',None)
        if flow is not None:
            if x.dtype!=flow.dtype:raise ValueError('DiT residual dtype does not match tensor flow')
            from patches.h3_tensor_flow import modulated_norm
        # 1. Modulation parameters
        adaln_out = self.adaln_proj(nn.functional.silu(t_emb) if self.apply_silu else t_emb)
        adaln_out = adaln_out.view(-1, 6 * self.hidden)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_out.chunk(6, dim=-1)

        # 2. Attention Block
        if flow is not None:
            normed_x = None
            h=modulated_norm(self.norm1,x,shift_msa,scale_msa,mod_segments,
                fused=getattr(policy,'dit_fused_norm',False))
        else:
            normed_x = self.norm1(x)
            h = self._mod_scale_shift(normed_x, shift_msa, scale_msa, mod_segments)

        # ColumnParallel QKV
        qkv = self.qkv_proj(h) # [S, 3 * inner_per_rank]
        del normed_x, h
        q, k, v = qkv.split(self.inner_per_rank, dim=-1)
        S = q.shape[0]
        q = q.view(S, self.heads_per_rank, self.head_dim)
        k = k.view(S, self.heads_per_rank, self.head_dim)
        v = v.view(S, self.heads_per_rank, self.head_dim)

        if rope_freqs is not None:
            from patches.tilelang_fp32_rms_rope import supported,execute
            if (flow is not None and getattr(policy,'dit_rms_rope','native')=='tilelang_fp32'
                    and supported(qkv,rope_freqs,self.heads_per_rank,self.head_dim)):
                q,k=execute(qkv,rope_freqs,self.q_norm.weight,self.k_norm.weight,
                    heads=self.heads_per_rank,dim=self.head_dim,epsilon=self.q_norm.eps,out_dtype=flow.dtype)
            else:
                if flow is not None:q,k=q.float(),k.float()
                q, k = comfy_kitchen.rms_rope_split_half(q[None], k[None], rope_freqs.float() if flow is not None else rope_freqs,
                    self.q_norm.weight, self.k_norm.weight, epsilon=self.q_norm.eps,
                    rot_dim=rope_freqs.shape[-3] * 2)
                q, k = q[0], k[0]
        else:
            q, k = self.q_norm(q), self.k_norm(k)
        if flow is not None:q,k=q.to(flow.dtype),k.to(flow.dtype)

        # Local Attention
        q_heads = q.transpose(0, 1).unsqueeze(0) # [1, heads_per_rank, S, head_dim]
        k_heads = k.transpose(0, 1).unsqueeze(0)
        v_heads = v.transpose(0, 1).unsqueeze(0)

        from patches.h3_operator_policy import dit_attention,prepare_dit_attention,execute_dit_attention
        prepared=prepare_dit_attention(self,q_heads,k_heads,v_heads,transformer_options) if flow is not None else None
        if prepared is not None:
            del qkv,q,k,v,q_heads,k_heads,v_heads
            attn_out=execute_dit_attention(prepared)
            del prepared
        else:
            attn_out=dit_attention(self,q_heads,k_heads,v_heads,transformer_options)
            del qkv,q,k,v,q_heads,k_heads,v_heads
        attn_out = attn_out.reshape(S,self.inner_per_rank)

        # RowParallel Out + AllReduce #1 (native INT32 SUM by default)
        if flow is not None:
            x=self.out_proj(attn_out,reduce_sum=True,residual=x,gate=gate_msa,segments=mod_segments)
        else:
            attn_out=self.out_proj(attn_out,reduce_sum=True)
            x=self._mod_gate(x,gate_msa,attn_out,mod_segments)
        del attn_out

        # 3. MLP Block (SwiGLU)
        if flow is not None:
            normed_x2 = None
            h2=modulated_norm(self.norm2,x,shift_mlp,scale_mlp,mod_segments,
                fused=getattr(policy,'dit_fused_norm',False))
        else:
            normed_x2 = self.norm2(x)
            h2 = self._mod_scale_shift(normed_x2, shift_mlp, scale_mlp, mod_segments)

        fc1_out = self.fc1(h2) # [S, 2 * ffn_per_rank]
        del normed_x2, h2
        # RowParallel FC2 + AllReduce #2 (native INT32 SUM by default)
        if flow is not None:
            x=self.fc2(fc1_out,input_act='swiglu',residual=x,gate=gate_mlp,segments=mod_segments)
            del fc1_out
        else:
            mlp_out=self.fc2(fc1_out,input_act='swiglu')
            del fc1_out
            x=self._mod_gate(x,gate_mlp,mlp_out,mod_segments)
            del mlp_out

        return x
