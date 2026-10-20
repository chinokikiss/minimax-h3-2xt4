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
  - Complete activation rows are gathered before ConvRot quantization; integer partial sums are reduced before casting
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import comfy_kitchen
from patches.parallel_int8 import row_linear

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
        if not reduce_sum and self.world_size > 1:
            raise ValueError("Quantization-preserving row projection requires reduction")
        from patches.h3_operator_policy import row_linear as dispatch_row
        return dispatch_row(self,x,row_linear,dist)

    def forward_parts(self,gate,up):
        from patches.h3_operator_policy import row_linear_parts
        return row_linear_parts(self,gate,up,row_linear,dist)


class TPTransformerBlockINT8(nn.Module):
    """
    Tensor Parallel Transformer block for Qwen3-VL-32B:
    Attention:
      - Q: ColumnParallel (32 heads / rank) + Q-Norm (RMSNorm)
      - K: ColumnParallel (4 heads / rank) + K-Norm (RMSNorm)
      - V: ColumnParallel (4 heads / rank)
      - O: RowParallel + AllReduce
    MLP:
      - Gate: ColumnParallel (12800 / rank)
      - Up: ColumnParallel (12800 / rank)
      - Down: RowParallel + AllReduce
    """
    def __init__(self, hidden_size=5120, intermediate_size=25600, num_heads=64, num_kv_heads=8, head_dim=128, rank=0, world_size=2, device=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.world_size = world_size
        self.num_heads_per_rank = num_heads // world_size
        self.num_kv_heads_per_rank = num_kv_heads // world_size
        self.head_dim = head_dim
        self.device = device or torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6).to(device=self.device, dtype=torch.float16)
        self.q_proj = ColumnParallelLinearINT8(hidden_size, num_heads * head_dim, rank, world_size)
        self.k_proj = ColumnParallelLinearINT8(hidden_size, num_kv_heads * head_dim, rank, world_size)
        self.v_proj = ColumnParallelLinearINT8(hidden_size, num_kv_heads * head_dim, rank, world_size)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6).to(device=self.device, dtype=torch.float16)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6).to(device=self.device, dtype=torch.float16)
        self.o_proj = RowParallelLinearINT8(num_heads * head_dim, hidden_size, rank, world_size)

        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6).to(device=self.device, dtype=torch.float16)
        self.gate_proj = ColumnParallelLinearINT8(hidden_size, intermediate_size, rank, world_size)
        self.up_proj = ColumnParallelLinearINT8(hidden_size, intermediate_size, rank, world_size)
        self.down_proj = RowParallelLinearINT8(intermediate_size, hidden_size, rank, world_size)

    def load_from_full_block(self, full_block):
        sd = full_block.state_dict()
        native = 'self_attn.q_proj.weight' in sd
        for name in ['input_layernorm', 'post_attention_layernorm', 'q_norm', 'k_norm']:
            key = ('self_attn.' if native and name in ['q_norm', 'k_norm'] else '') + name
            getattr(self, name).load_state_dict({'weight': sd[key + '.weight']})
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            key = (('self_attn.' if name in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] else 'mlp.') if native else '') + name
            getattr(self, name).load_shard_from_full(sd[key + '.weight'], sd[key + '.weight_scale'])

    def forward(self, x, attention_mask=None, freqs_cis=None):
        # 1. Self Attention
        residual = x
        # Native ComfyUI casts norm weights to the activation dtype. The
        # CLIP frontend carries FP32 activations even with FP16 TE weights.
        from comfy.ldm.common_dit import rms_norm
        normed_x = rms_norm(x, self.input_layernorm.weight, self.input_layernorm.eps)

        q = self.q_proj(normed_x) # [B, S, 32 * 128 = 4096]
        k = self.k_proj(normed_x) # [B, S, 4 * 128 = 512]
        v = self.v_proj(normed_x) # [B, S, 4 * 128 = 512]

        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads_per_rank, self.head_dim)
        k = k.view(B, S, self.num_kv_heads_per_rank, self.head_dim)
        v = v.view(B, S, self.num_kv_heads_per_rank, self.head_dim)

        q = rms_norm(q.transpose(1, 2), self.q_norm.weight, self.q_norm.eps)
        k = rms_norm(k.transpose(1, 2), self.k_norm.weight, self.k_norm.eps)
        v = v.transpose(1, 2)

        from comfy.text_encoders.llama import apply_rope
        from comfy.ldm.modules.attention import optimized_attention_for_device
        q, k = apply_rope(q, k, freqs_cis=freqs_cis)
        from patches.h3_operator_policy import te_attention
        attn_out=te_attention(self,q,k,v,attention_mask)
        if attn_out is None:
            attention = optimized_attention_for_device(x.device, mask=attention_mask is not None, small_input=True)
            def attend():
                return attention(q, k, v, self.num_heads_per_rank, mask=attention_mask,
                    skip_reshape=True, enable_gqa=self.num_heads_per_rank != self.num_kv_heads_per_rank)
            # The pinned native SDPA wrapper changes backend priority at this
            # global query size. A head shard must retain the full-model policy.
            import comfy.ops
            if (q.numel() * self.world_size >= 1024 * 128
                    and hasattr(comfy.ops, 'SDPA_BACKEND_PRIORITY')):
                from torch.nn.attention import sdpa_kernel
                with sdpa_kernel(comfy.ops.SDPA_BACKEND_PRIORITY, set_priority=True):
                    # Native's global-size branch also explicitly repeats KV
                    # when no selected backend supports masked GQA. Merely
                    # entering its backend context misses this for short prompts
                    # whose local head shard falls below the dispatch threshold.
                    if (q.numel() < 1024 * 128 and attention_mask is not None
                            and self.num_heads_per_rank != self.num_kv_heads_per_rank):
                        mask = attention_mask
                        if mask.ndim == 2:
                            mask = mask.unsqueeze(0)
                        if mask.ndim == 3:
                            mask = mask.unsqueeze(1)
                        params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0., False, True)
                        supports_gqa = (torch.backends.cuda.can_use_flash_attention(params)
                            or torch.backends.cuda.can_use_cudnn_attention(params)
                            or torch.backends.cuda.can_use_efficient_attention(params))
                        del params
                        if not supports_gqa:
                            k, v = comfy.ops.repeat_kv_for_gqa(k, v, self.num_heads_per_rank, -3)
                            attn_out = attention(q, k, v, self.num_heads_per_rank,
                                mask=attention_mask, skip_reshape=True, enable_gqa=False)
                        else:
                            attn_out = attend()
                    else:
                        attn_out = attend()
            else:
                attn_out = attend()

        # RowParallel O projection + AllReduce 1
        attn_out = self.o_proj(attn_out, reduce_sum=True)
        x = residual + attn_out
        del normed_x, q, k, v, attn_out

        # 2. MLP
        residual = x
        normed_x = rms_norm(x, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)

        gate = self.gate_proj(normed_x) # [B, S, 12800]
        up = self.up_proj(normed_x)     # [B, S, 12800]
        policy=getattr(self,'operator_policy',None)
        # Consume separate producer tensors before materializing SwiGLU.
        if policy is not None and policy.swiglu_padding:
            mlp_out=self.down_proj.forward_parts(gate,up)
        else:
            mlp_act = torch.nn.functional.silu(gate) * up
            mlp_out = self.down_proj(mlp_act, reduce_sum=True)
            del mlp_act
        del normed_x, gate, up
        x = residual + mlp_out
        del mlp_out

        return x
