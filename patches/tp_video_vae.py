"""Repaired two-rank INT8 video decoder.

Column projections and attention heads are sharded. Output projections gather
complete activations and use replicated native weights, preserving the native
fused quantization/bias/residual epilogue. SP shards sequence boundaries without
inserting extra attention tokens. This replaces the legacy payload claims.
"""
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.distributed as dist
import comfy_kitchen as ck
from comfy.ldm.minimax.vae import create_token_ids, RotaryEmbeddingND
from patches.parallel_int8 import gather, quantize, row_linear, int_mm, dequantize, packed_rows

quantize_int8_rowwise_convrot = quantize

def w8a8_gemm(q_act, scale_act, weight, weight_scale, bias=None, out_dtype=torch.float16):
    return dequantize(int_mm(q_act, weight), scale_act, weight_scale, out_dtype, bias)


def create_tp_block_from_sd(sd, block_idx, rank, world_size, device):
    prefix = f"decoder.transformer_blocks.{block_idx}."
    result = {"world_size": world_size, "dim_head": 64}
    for short, suffix, style in [('qkv', 'attn.to_qkv', 'heads'), ('out', 'attn.to_out', 'row'),
                                  ('w1', 'ff.w1', 'packed'), ('w2', 'ff.w2', 'row')]:
        w, scale = sd[prefix + suffix + '.weight'], sd[prefix + suffix + '.weight_scale']
        bias = sd.get(prefix + suffix + '.bias')
        if style == 'row':
            result['qw_' + short + '_full'] = nn.Parameter(w.to(device=device, dtype=torch.int8).contiguous(), requires_grad=False)
            result['qs_' + short + '_full'] = nn.Parameter(scale.to(device=device, dtype=torch.float32).contiguous(), requires_grad=False)
            w = w.chunk(world_size, 1)[rank]
            # Bias is applied once after the complete integer reduction.
        elif style == 'packed':
            w, scale = packed_rows(w, scale, rank, world_size, 2)
            if bias is not None:
                bias = torch.cat([b.chunk(world_size)[rank] for b in bias.chunk(2)])
        else:
            w, scale = w.chunk(world_size)[rank], scale.chunk(world_size)[rank]
            if bias is not None:
                bias = bias.chunk(world_size)[rank]
        for key, value in [('qw_' + short, w), ('qs_' + short, scale), ('b_' + short, bias)]:
            dtype = torch.int8 if key.startswith('qw') else torch.float32 if key.startswith('qs') else torch.float16
            result[key] = nn.Parameter(value.to(device=device, dtype=dtype).contiguous(), requires_grad=False) if value is not None else None
    for short, suffix in [('norm1_w', 'norm1.weight'), ('norm2_w', 'norm2.weight'), ('scale1', 'scale1'), ('scale2', 'scale2')]:
        result[short] = nn.Parameter(sd[prefix + suffix].to(device=device, dtype=torch.float16), requires_grad=False)
    return result


class RealTPTransformerBlock(nn.Module):
    def __init__(self, block_dict, eps=1e-5, convrot_groupsize=256, device=None):
        super().__init__()
        for key, value in block_dict.items():
            setattr(self, key, value)
        self.device, self.eps, self.convrot_groupsize = device, eps, convrot_groupsize
        self.dim = self.qw_qkv.shape[1]
        self.inner_dim_per_rank = self.qw_out.shape[1]
        self.heads_per_rank = self.inner_dim_per_rank // self.dim_head
        self.register_buffer('qk_norm_scale', torch.ones(self.dim_head, device=device), persistent=False)

    def _column(self, x, suffix, norm):
        return ck.int8_linear(x.reshape(-1, self.dim), getattr(self, 'qw_' + suffix),
            getattr(self, 'qs_' + suffix), getattr(self, 'b_' + suffix), x.dtype,
            convrot=True, convrot_groupsize=self.convrot_groupsize,
            input_act='rms_norm', input_act_weight=norm, input_act_eps=self.eps)

    def forward(self, x, rotary_pos_emb=None):
        b, s, _ = x.shape
        qkv = self._column(x, 'qkv', self.norm1_w).view(b, s, self.heads_per_rank, 3 * self.dim_head)
        q, k, v = qkv.chunk(3, -1)
        if rotary_pos_emb is not None:
            # Match native inference's in-place dispatch and QKV view layout.
            # The out-of-place result has a different layout/kernel path.
            rms_rope = ck.rms_rope_split_half if torch.is_grad_enabled() else ck.rms_rope_split_half_
            q, k = rms_rope(q, k, rotary_pos_emb, self.qk_norm_scale,
                epsilon=self.eps, rot_dim=rotary_pos_emb.shape[-3] * 2)
        else:
            q = torch.nn.functional.rms_norm(q, (self.dim_head,), eps=self.eps)
            k = torch.nn.functional.rms_norm(k, (self.dim_head,), eps=self.eps)
        q, k, v = (a.transpose(1, 2) for a in [q, k, v])
        from patches.kitchen_attention import int8_attention
        out = int8_attention(q, k, v).transpose(1, 2).reshape(b, s, self.inner_dim_per_rank)
        full, _, _ = gather(torch.nan_to_num(out), dist, -1)
        x = ck.int8_linear(full, self.qw_out_full, self.qs_out_full, self.b_out, x.dtype,
            convrot=True, convrot_groupsize=self.convrot_groupsize, residual=x, residual_scale=self.scale1)
        packed = self._column(x, 'w1', self.norm2_w).view(b, s, -1)
        full, _, _ = gather(packed, dist, -1)
        pieces = full.chunk(self.world_size, -1)
        full = torch.cat([p.chunk(2, -1)[0] for p in pieces] + [p.chunk(2, -1)[1] for p in pieces], -1)
        return ck.int8_linear(full, self.qw_w2_full, self.qs_w2_full, self.b_w2, x.dtype,
            convrot=True, convrot_groupsize=self.convrot_groupsize, input_act='swiglu',
            residual=x, residual_scale=self.scale2)


class RealSPTransformerBlock(RealTPTransformerBlock):
    def __init__(self, block_dict, world_size=2, **kwargs):
        super().__init__(block_dict, **kwargs)
        self.world_size = world_size

    def forward(self, x_local, rotary_pos_emb=None):
        full, start, length = gather(x_local, dist, 1)
        out = super().forward(full, rotary_pos_emb)
        return out[:, start:start + length].contiguous()


def patch_video_vae_tp(model, sd, rank, world_size, device):
    for i in range(len(model.decoder.transformer_blocks)):
        model.decoder.transformer_blocks[i] = RealTPTransformerBlock(
            create_tp_block_from_sd(sd, i, rank, world_size, device), device=device)
    return model


def patch_video_vae_sp(model, sd, rank, world_size, device):
    for i in range(len(model.decoder.transformer_blocks)):
        model.decoder.transformer_blocks[i] = RealSPTransformerBlock(
            create_tp_block_from_sd(sd, i, rank, world_size, device), world_size=world_size, device=device)

    def sp_decoder_forward(x):
        decoder = model.decoder
        B, C, T, H, W = x.shape
        h = decoder.x_embedder(x.flatten(2).transpose(1, 2))
        patches = h.shape[1]
        reg = decoder.register_tokens.expand(B, -1, -1)
        # Preserve precisely the native register tokens plus ONE zero token.
        h = torch.cat([h, reg, torch.zeros((B, 1, h.shape[-1]), device=h.device, dtype=h.dtype)], 1)
        ids = create_token_ids((T, H, W), x.device, x.dtype).expand(B, -1, -1)
        suffix = torch.zeros((B, decoder.num_register_tokens + 1, 3), device=x.device, dtype=x.dtype)
        rope = decoder.pos_embed(torch.cat([ids, suffix], 1))
        local = torch.tensor_split(h, world_size, dim=1)[rank].contiguous()
        for block in decoder.transformer_blocks:
            local = block(local, rope)
        full, _, _ = gather(local, dist, 1)
        out = decoder.proj_out(decoder.norm_out(full))[:, :patches]
        out = out.view(B, T, H, W, decoder.out_channels, decoder.patch_size_t, decoder.patch_size, decoder.patch_size)
        return out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().reshape(
            B, decoder.out_channels, T * decoder.patch_size_t, H * decoder.patch_size, W * decoder.patch_size)
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
