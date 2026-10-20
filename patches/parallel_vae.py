"""Native-aligned video queries and independent FP32 stereo audio channels.

Runs inside the existing two persistent ranks. Public VAE planning remains on
the parent: only raw video pixels / the audio model's decode are dispatched.
Audio channel splitting is intentionally not advertised as Native bit-exact.
"""
from types import MethodType

import torch
from torch import nn

from patches.windowed_tp import CheckpointReader


def query_partitioning_status(model):
    """Explain the qualified query contract without confusing ABI and build support."""
    import inspect
    import comfy.ops
    import comfy.ldm.minimax.vae as native
    import comfy.ldm.modules.attention as attention
    blocks=list(model.decoder.transformer_blocks)
    native_flag=getattr(native,'COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE',None)
    unquantized=[];signature_mismatch=[];missing_scale=[];signatures=set()
    for i,block in enumerate(blocks):
        if not isinstance(block.attn.to_qkv.weight,comfy.ops.QuantizedTensor):unquantized.append(i)
        try:
            signature=inspect.signature(block.attn.forward);parameters=signature.parameters
        except (TypeError,ValueError):signature='<uninspectable>';parameters={}
        signatures.add(str(signature))
        if not {'pre_norm','residual','residual_scale'}.issubset(parameters):signature_mismatch.append(i)
        if not hasattr(block.attn,'qk_norm_scale'):missing_scale.append(i)
    reasons=[]
    if not blocks:reasons.append('decoder has no transformer blocks')
    if native_flag is None:reasons.append('Native VAE no longer exports its legacy Kitchen dispatch flag')
    elif not native_flag:reasons.append('Native VAE Kitchen dispatch is disabled')
    if unquantized:reasons.append(f'QKV weights are not QuantizedTensor in {len(unquantized)} blocks')
    if signature_mismatch:reasons.append(f'attention call signature differs in {len(signature_mismatch)} blocks')
    if missing_scale:reasons.append(f'fused qk_norm_scale is absent in {len(missing_scale)} blocks')
    return dict(available=not reasons,reason='; '.join(reasons) if reasons else None,
        block_count=len(blocks),native_vae_kitchen_flag=native_flag,
        kitchen_build_available=bool(attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE),
        native_global_attention_backend=getattr(native.optimized_attention,'__name__',type(native.optimized_attention).__name__),
        attention_signatures=sorted(signatures),unquantized_blocks=unquantized,
        signature_mismatch_blocks=signature_mismatch,missing_qk_norm_scale_blocks=missing_scale)


def query_partitioning_available(model):
    return query_partitioning_status(model)['available']


class VideoDecodeWeights(nn.Module):
    def __init__(self):
        super().__init__()
        import comfy.ops
        from comfy.ldm.minimax.vae import ViT3DDecoder
        ops = comfy.ops.mixed_precision_ops({'mixed_ops': True}, torch.float16)
        self.post_quant_conv = comfy.ops.disable_weight_init.Conv3d(24, 24, 1)
        self.decoder = ViT3DDecoder(operations=ops, num_layers=36)

    def _decode_pixels(self, z):
        return self.decoder(self.post_quant_conv(z))


class AudioDecodeWeights(nn.Module):
    def __init__(self):
        super().__init__()
        import comfy.ops
        from comfy.ldm.minimax.audio_vae import BigVGAN
        self.dec_in_proj = comfy.ops.disable_weight_init.Conv1d(32, 2048, 1)
        self.decoder = BigVGAN(num_mels=2048, upsample_initial_channel=1024)
        self.register_buffer('latents_mean', torch.empty(32))
        self.register_buffer('latents_std', torch.empty(32))

    def decode(self, z):
        from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
        return MiniMaxH3AudioVAE.decode(self, z)


def checkpoint_plan(path, kind):
    reader = CheckpointReader(path)
    prefixes = ('decoder.', 'post_quant_conv.') if kind == 'video' else (
        'decoder.', 'dec_in_proj.', 'latents_mean', 'latents_std')
    names = [n for n in reader.header if n != '__metadata__' and n.startswith(prefixes)]
    if kind == 'video':
        depth = sum(n.startswith('decoder.transformer_blocks.') and n.endswith('.scale1') for n in names)
        if depth != 36:
            raise ValueError('Query partitioning requires the trained 36-layer video checkpoint')
    elif kind != 'audio':
        raise ValueError(kind)
    payload = sum(reader.header[n]['data_offsets'][1] - reader.header[n]['data_offsets'][0] for n in names)
    # CPU checkpoint + model backing + allocator/construction allowance.
    return reader, names, 2 * payload + 256 * 2**20


@torch.inference_mode(False)
@torch.no_grad()
def load_decode_weights(reader, names, kind, device):
    # Module.to inside inference_mode can turn a QuantizedTensor parameter
    # into a plain Parameter, changing Native attention/operator dispatch.
    # Construct/transfer like Native loading; inference_mode resumes for decode.
    model = VideoDecodeWeights() if kind == 'video' else AudioDecodeWeights()
    state = {n: reader.read_tensor(n) for n in names}
    model.load_state_dict(state, strict=True)
    del state
    dtype = torch.float16 if kind == 'video' else torch.float32
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if kind == 'video':
        status=query_partitioning_status(model)
        if not status['available']:
            raise ValueError('Query partitioning unavailable: '+status['reason'])
    return model


def aligned_ranges(length):
    """Keep the physical qualification's 128-query scale-group boundary."""
    cut = ((length + 255) // 256) * 128
    return ((0, cut), (cut, length)) if 0 < cut < length else None


class AlignedVideoDecode:
    def __init__(self, model, z, comm, rank):
        self.model, self.z, self.comm, self.rank = model, z, comm, rank
        self.buffers = {}
        self.patches = z.shape[2] * z.shape[3] * z.shape[4]
        self.ranges = aligned_ranges(self.patches + 5)

    def exchange(self, x, sizes, name, dim):
        y = x.movedim(dim, 0).contiguous()
        maximum = max(sizes)
        key = (name, tuple(y.shape[1:]), maximum, y.dtype)
        if key not in self.buffers:
            self.buffers[key] = (
                torch.empty((maximum, *y.shape[1:]), dtype=y.dtype, device=y.device),
                torch.empty((2 * maximum, *y.shape[1:]), dtype=y.dtype, device=y.device))
        padded, full = self.buffers[key]
        padded[:y.shape[0]].copy_(y)
        padded[y.shape[0]:].zero_()
        self.comm.all_gather_into_tensor(full, padded)
        return torch.cat((full[:sizes[0]], full[maximum:maximum + sizes[1]]), 0).movedim(0, dim).contiguous()

    def run(self):
        import comfy.ops
        from comfy.ldm.minimax.vae import create_token_ids
        if self.ranges is None:
            return self.model._decode_pixels(self.z) if self.rank == 0 else None
        decoder = self.model.decoder
        originals = [block.attn.forward for block in decoder.transformer_blocks]
        try:
            for block in decoder.transformer_blocks:
                block.attn._query_run = self
                block.attn.forward = MethodType(parallel_attention, block.attn)
            b, _, t, h, w = self.z.shape
            lo, hi = self.ranges[self.rank]
            otherlo, otherhi = self.ranges[1]
            remote = torch.empty((b, otherhi - otherlo, decoder.register_tokens.shape[-1]),
                                 dtype=self.z.dtype, device=self.z.device)
            if self.rank == 0:
                full = decoder.x_embedder(self.model.post_quant_conv(self.z).flatten(2).transpose(1, 2))
                full = torch.cat((full, comfy.ops.cast_to_input(decoder.register_tokens, full).expand(b, -1, -1),
                                  torch.zeros_like(full[:, :1])), 1)
                remote.copy_(full[:, otherlo:otherhi])
                local = full[:, lo:hi].contiguous()
                del full
            self.comm.broadcast(remote, src=0)
            if self.rank == 1:
                local = remote
            coords = create_token_ids((t, h, w), self.z.device, self.z.dtype).expand(b, -1, -1)
            coords = torch.cat((coords, torch.zeros((b, 5, 3), device=self.z.device, dtype=self.z.dtype)), 1)
            rope = decoder.pos_embed(coords[:, lo:hi].contiguous())
            for block in decoder.transformer_blocks:
                local = block(local, rope)
            local = decoder.proj_out(decoder.norm_out(local))
            projected = self.exchange(local, [end - start for start, end in self.ranges], 'projected', 1)
            if self.rank == 1:
                return None
            projected = projected[:, :self.patches].view(
                b, t, h, w, decoder.out_channels, decoder.patch_size_t, decoder.patch_size, decoder.patch_size)
            return projected.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().reshape(
                b, decoder.out_channels, t * decoder.patch_size_t, h * decoder.patch_size, w * decoder.patch_size)
        finally:
            for block, forward in zip(decoder.transformer_blocks, originals):
                block.attn.forward = forward
                if hasattr(block.attn, '_query_run'):
                    del block.attn._query_run
            self.buffers.clear()


def parallel_attention(self, x, rotary_pos_emb, pre_norm, residual, residual_scale):
    import comfy.ops
    import comfy.quant_ops
    run = self._query_run
    b, s, _ = x.shape
    with comfy.ops.CastBiasWeightContext(pre_norm, x, offloadable=True) as (weight, bias):
        qkv = comfy.ops.linear_input_act(self.to_qkv, x, 'rms_norm', weight, pre_norm.eps)
    q, k, v = qkv.view(b, s, -1, 3 * self.dim_head).chunk(3, -1)
    q, k = comfy.quant_ops.ck.rms_rope_split_half_(
        q, k, rotary_pos_emb, self.qk_norm_scale.to(q.device), epsilon=self.norm_q.eps,
        rot_dim=rotary_pos_emb.shape[-3] * 2)
    q, k, v = (a.transpose(1, 2) for a in (q, k, v))
    sizes = [end - start for start, end in run.ranges]
    full_k, full_v = run.exchange(torch.stack((k, v)), sizes, 'kv', 3).unbind(0)
    out = comfy.quant_ops.ck.int8_attention(q, full_k, full_v)
    out = out.transpose(1, 2).reshape(b, s, -1)
    return comfy.ops.linear_input_act(self.to_out, torch.nan_to_num(out), None,
                                     residual=residual, residual_scale=residual_scale)


def channel_decode(model, z, comm, rank):
    from patches.audio_channel_decode import channel_decode as decode
    return decode(model,z,comm,rank)
