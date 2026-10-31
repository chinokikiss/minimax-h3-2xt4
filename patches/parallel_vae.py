"""Native-aligned video query partitioning inside the persistent TP2 ranks.

Runs inside the existing two persistent ranks. Public VAE planning remains on
the parent: only raw video pixels are dispatched. Audio VAE decoding is Native
on a single GPU and never loads a decoder replica into these ranks.
"""
from types import MethodType

import torch
from torch import nn

from patches.windowed_tp import CheckpointReader


def query_partitioning_status(model, *, checkpoint_reader=None, attention_backend='kitchen'):
    """Explain the qualified query contract without confusing ABI and build support."""
    import inspect
    import comfy.ops
    import comfy.ldm.minimax.vae as native
    import comfy.ldm.modules.attention as attention
    blocks=list(model.decoder.transformer_blocks)
    native_flag=getattr(native,'COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE',None)
    unquantized=[];signature_mismatch=[];missing_scale=[];signatures=set()
    for i,block in enumerate(blocks):
        if checkpoint_reader is None:
            quantized=isinstance(block.attn.to_qkv.weight,comfy.ops.QuantizedTensor)
        else:
            # Parent metadata gate only. Workers still qualify the actual
            # constructed QuantizedTensor and API before collective decode.
            import json
            prefix=f'decoder.transformer_blocks.{i}.attn.to_qkv.'
            weight=checkpoint_reader.header.get(prefix+'weight',{})
            quantized=weight.get('dtype')=='I8' and prefix+'comfy_quant' in checkpoint_reader.header
            if quantized:
                raw=checkpoint_reader.read_tensor(prefix+'comfy_quant')
                quantized=json.loads(bytes(raw.tolist()))==dict(format='int8_tensorwise',convrot=True,convrot_groupsize=256)
        if not quantized:unquantized.append(i)
        try:
            signature=inspect.signature(block.attn.forward);parameters=signature.parameters
        except (TypeError,ValueError):signature='<uninspectable>';parameters={}
        signatures.add(str(signature))
        if not {'pre_norm','residual','residual_scale'}.issubset(parameters):signature_mismatch.append(i)
        if not hasattr(block.attn,'qk_norm_scale'):missing_scale.append(i)
    if attention_backend not in ('kitchen','native'):
        raise ValueError('Unknown Video VAE attention backend')
    reasons=[]
    if not blocks:reasons.append('decoder has no transformer blocks')
    if attention_backend=='kitchen':
        if native_flag is None:reasons.append('Native VAE no longer exports its legacy Kitchen dispatch flag')
        elif not native_flag:reasons.append('Native VAE Kitchen dispatch is disabled')
        if not bool(attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE):
            reasons.append('Kitchen attention build is unavailable')
    if unquantized:reasons.append(f'QKV weights are not QuantizedTensor in {len(unquantized)} blocks')
    if signature_mismatch:reasons.append(f'attention call signature differs in {len(signature_mismatch)} blocks')
    if missing_scale:reasons.append(f'fused qk_norm_scale is absent in {len(missing_scale)} blocks')
    return dict(available=not reasons,reason='; '.join(reasons) if reasons else None,
        block_count=len(blocks),native_vae_kitchen_flag=native_flag,
        kitchen_build_available=bool(attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE),
        attention_backend=attention_backend,
        native_global_attention_backend=getattr(native.optimized_attention,'__name__',type(native.optimized_attention).__name__),
        attention_signatures=sorted(signatures),unquantized_blocks=unquantized,
        validation='checkpoint_metadata' if checkpoint_reader is not None else 'constructed_weights',
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
        from patches.kitchen_attention import native_scope
        with native_scope():return self.decoder(self.post_quant_conv(z))


def checkpoint_plan(path, kind):
    if kind != 'video':
        raise ValueError('Parallel VAE decoding is video-only; use Native single-GPU Audio VAE decoding')
    reader = CheckpointReader(path)
    prefixes = ('decoder.', 'post_quant_conv.')
    names = [n for n in reader.header if n != '__metadata__' and n.startswith(prefixes)]
    depth = sum(n.startswith('decoder.transformer_blocks.') and n.endswith('.scale1') for n in names)
    if depth != 36:
        raise ValueError('Query partitioning requires the trained 36-layer video checkpoint')
    payload = sum(reader.header[n]['data_offsets'][1] - reader.header[n]['data_offsets'][0] for n in names)
    # CPU checkpoint + model backing + allocator/construction allowance.
    return reader, names, 2 * payload + 256 * 2**20


@torch.inference_mode(False)
@torch.no_grad()
def load_decode_weights(reader, names, kind, device, *, parallelism='query', attention_backend='kitchen'):
    if kind != 'video':
        raise ValueError('Parallel VAE decoding is video-only; use Native single-GPU Audio VAE decoding')
    if parallelism not in ('query', 'temporal'):
        raise ValueError(parallelism)
    if attention_backend not in ('kitchen','native'):
        raise ValueError('Unknown Video VAE attention backend')
    # Module.to inside inference_mode can turn a QuantizedTensor parameter
    # into a plain Parameter, changing Native attention/operator dispatch.
    # Construct/transfer like Native loading; inference_mode resumes for decode.
    model = VideoDecodeWeights()
    state = {n: reader.read_tensor(n) for n in names}
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
    from patches.h3_tensor_flow import prepare_video,policy_for
    prepare_video(model,policy_for(reader))
    for block in model.decoder.transformer_blocks:
        block.attn._h3_attention_backend=attention_backend
        if attention_backend=='native':
            block.attn.forward=MethodType(full_video_attention,block.attn)
    if parallelism == 'query':
        status=query_partitioning_status(model,attention_backend=attention_backend)
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


def _video_qkv(self, x, rotary_pos_emb, pre_norm):
    import comfy.ops
    import comfy.quant_ops
    b,s,_=x.shape
    with comfy.ops.CastBiasWeightContext(pre_norm,x,offloadable=True) as (weight,bias):
        qkv=comfy.ops.linear_input_act(self.to_qkv,x,'rms_norm',weight,pre_norm.eps)
    q,k,v=qkv.view(b,s,-1,3*self.dim_head).chunk(3,-1)
    q,k=comfy.quant_ops.ck.rms_rope_split_half_(
        q,k,rotary_pos_emb,self.qk_norm_scale.to(q.device),epsilon=self.norm_q.eps,
        rot_dim=rotary_pos_emb.shape[-3]*2)
    return tuple(a.transpose(1,2) for a in (q,k,v))


def _video_attention(self,q,k,v):
    backend=getattr(self,'_h3_attention_backend','kitchen')
    if backend=='kitchen':
        from patches.kitchen_attention import int8_attention
        return int8_attention(q,k,v)
    if backend=='native':
        return torch.nn.functional.scaled_dot_product_attention(q,k,v)
    raise ValueError('Unknown Video VAE attention backend')


def full_video_attention(self,x,rotary_pos_emb,pre_norm,residual,residual_scale):
    import comfy.ops
    q,k,v=_video_qkv(self,x,rotary_pos_emb,pre_norm)
    out=_video_attention(self,q,k,v).transpose(1,2).reshape(x.shape[0],x.shape[1],-1)
    return comfy.ops.linear_input_act(self.to_out,torch.nan_to_num(out),None,
        residual=residual,residual_scale=residual_scale)


def parallel_attention(self,x,rotary_pos_emb,pre_norm,residual,residual_scale):
    import comfy.ops
    run=self._query_run
    q,k,v=_video_qkv(self,x,rotary_pos_emb,pre_norm)
    sizes=[end-start for start,end in run.ranges]
    full_k,full_v=run.exchange(torch.stack((k,v)),sizes,'kv',3).unbind(0)
    backend=getattr(self,'_h3_attention_backend','kitchen')
    if backend=='kitchen':
        from patches.kitchen_attention import int8_attention
        out=int8_attention(q,full_k,full_v)
    elif backend=='native':
        out=torch.nn.functional.scaled_dot_product_attention(q,full_k,full_v)
    else:
        raise ValueError('Unknown Video VAE attention backend')
    out=out.transpose(1,2).reshape(x.shape[0],x.shape[1],-1)
    return comfy.ops.linear_input_act(self.to_out,torch.nan_to_num(out),None,
        residual=residual,residual_scale=residual_scale)
