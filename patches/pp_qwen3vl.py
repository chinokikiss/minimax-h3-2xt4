"""Two-stage placement preserving the pinned native Qwen forward contract.

The forward mathematics below are from ComfyUI d49e888 Llama2_.forward;
changes only place activations, masks, positions, caches and intermediates.
CUDA graph capture is disabled for this multi-device schedule.
"""
import types
import torch
import torch.nn as nn
import comfy.model_prefetch
from comfy.text_encoders.llama import FixedKV, FixedKVBias, rope_matrix
from comfy.ldm.modules.attention import optimized_attention_for_device


def _place(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_place(v, device) for v in value)
    if isinstance(value, list):
        return [_place(v, device) for v in value]
    if isinstance(value, FixedKV):
        for key, item in vars(value).items():
            if isinstance(item, torch.Tensor):
                setattr(value, key, item.to(device))
    return value


class PipelineParallelQwen3VL(nn.Module):
    def __init__(self, model, dev0='cuda:0', dev1='cuda:1', split_layer=25):
        super().__init__()
        self.model = model
        self.dev0, self.dev1 = torch.device(dev0), torch.device(dev1)
        self.split_layer = split_layer
        inner = getattr(model, 'model', model)
        if not 0 < split_layer < len(inner.layers):
            raise ValueError('Both pipeline stages need at least one layer')
        if getattr(model, 'visual', None) is not None:
            model.visual.to(self.dev0)
        inner.embed_tokens.to(self.dev0)
        inner._pp_devices = [self.dev0 if i < split_layer else self.dev1 for i in range(len(inner.layers))]
        for layer, device in zip(inner.layers, inner._pp_devices):
            layer.to(device)
        if inner.norm is not None:
            inner.norm.to(self.dev1)
        inner.graph_dynamic_vbar_blocks = False
        inner.forward = types.MethodType(_placed_forward, inner)

    def forward(self, input_ids=None, embeds=None, attention_mask=None, **kwargs):
        with torch.cuda.device(self.dev0):
            return self.model(_place(input_ids, self.dev0), embeds=_place(embeds, self.dev0),
                              attention_mask=_place(attention_mask, self.dev0), **kwargs)


def _placed_forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True,
            dtype=None, position_ids=None, embeds_info=[], past_key_values=None, input_ids=None,deepstack_embeds=None, visual_pos_masks=None, decode_buffers=None):
    x = _place(x, self._pp_devices[0])
    embeds = _place(embeds, self._pp_devices[0])
    attention_mask = _place(attention_mask, self._pp_devices[0])
    position_ids = _place(position_ids, self._pp_devices[0])
    if embeds is not None:
        x = embeds
    else:
        x = self.embed_tokens(x, out_dtype=dtype)

    seq_len = x.shape[1]
    past_len = 0
    if past_key_values is not None and len(past_key_values) > 0:
        past_len = self.get_past_len(past_key_values)
    fixed_kv = past_key_values is not None and len(past_key_values) > 0 and isinstance(past_key_values[0], FixedKV)
    fixed_kv_decode = fixed_kv and past_len > 0 and seq_len == 1
    if fixed_kv_decode:
        attention_mask = None

    if position_ids is None:
        position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)

    freqs_cis = self.compute_freqs_cis(position_ids, x.device)

    mask = None
    if attention_mask is not None:
        mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
        mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

    spec_decode = fixed_kv and any(isinstance(kv, FixedKVBias) for kv in past_key_values) and 2 <= seq_len <= 6 and past_len > 0 and attention_mask is None
    if seq_len > 1 and not spec_decode:  # spec verify: the staircase decode bias is causal
        causal_mask = torch.empty(past_len + seq_len, past_len + seq_len, dtype=x.dtype, device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
        if mask is not None:
            mask += causal_mask
        else:
            mask = causal_mask

    optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

    enable_graph = self.graph_dynamic_vbar_blocks and (fixed_kv_decode or spec_decode)
    if enable_graph:
        if decode_buffers is None:
            x = x.clone()
        else:
            hidden_buffer, rotary_buffer = decode_buffers
            hidden_buffer.copy_(x)
            x = hidden_buffer
            rotary_buffer.copy_(rope_matrix(freqs_cis))
            freqs_cis = rotary_buffer

    intermediate = None
    all_intermediate = None
    only_layers = None
    if intermediate_output is not None:
        if isinstance(intermediate_output, list):
            all_intermediate = []
            only_layers = set(intermediate_output)
        elif intermediate_output == "all":
            all_intermediate = []
            intermediate_output = None
        elif intermediate_output < 0:
            intermediate_output = len(self.layers) + intermediate_output

    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {"prefetch_dynamic_vbars": self.prefetch_dynamic_vbars and past_key_values is not None})
    next_key_values = list(past_key_values) if past_key_values is not None else []
    for i, layer in enumerate(self.layers):
        target = self._pp_devices[i]
        x, mask, freqs_cis = _place(x, target), _place(mask, target), _place(freqs_cis, target)
        if all_intermediate is not None:
            if only_layers is None or (i in only_layers):
                all_intermediate.append(x.to(self._pp_devices[-1]).unsqueeze(1).clone())

        past_kv = None
        if past_key_values is not None:
            past_kv = _place(past_key_values[i], target) if len(past_key_values) > 0 else []

        if fixed_kv:
            past_kv.prepare(seq_len)

        def core():
            nonlocal x
            with torch.cuda.device(target):
                output, current_kv = layer(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=optimized_attention,
                past_key_value=past_kv,
            )
            if enable_graph:
                x.copy_(output)
            else:
                x = output
            if next_key_values:
                next_key_values[i] = current_kv

        comfy.model_prefetch.prefetch_queue_pop(
            prefetch_queue, x.device, layer, x.dtype, core=core, enable_graph=enable_graph,
            malloc_scope="block"
        )
        if fixed_kv:
            next_key_values[i].advance(seq_len)

        # DeepStack: add per-layer visual features into the first len() decoder layers at image positions (Qwen3-VL)
        if deepstack_embeds is not None and i < len(deepstack_embeds):
            local_visual_mask = visual_pos_masks.to(x.device)
            x[local_visual_mask] = x[local_visual_mask] + deepstack_embeds[i].to(x)

        if i == intermediate_output:
            intermediate = x.clone()

    comfy.model_prefetch.prefetch_queue_pop(
        prefetch_queue, x.device, None,
        malloc_scope="block"
    )

    if self.norm is not None:
        with torch.cuda.device(self._pp_devices[-1]):
            x = self.norm(x.to(self._pp_devices[-1]))

    if all_intermediate is not None:
        if only_layers is None or ((i + 1) in only_layers):
            all_intermediate.append(x.to(self._pp_devices[-1]).unsqueeze(1).clone())

    if all_intermediate is not None:
        intermediate = torch.cat(all_intermediate, dim=1)

    if intermediate is not None and final_layer_norm_intermediate and self.norm is not None:
        with torch.cuda.device(self._pp_devices[-1]):
            intermediate = self.norm(intermediate.to(self._pp_devices[-1]))

    if next_key_values:
        return x, intermediate, next_key_values
    else:
        return x, intermediate
