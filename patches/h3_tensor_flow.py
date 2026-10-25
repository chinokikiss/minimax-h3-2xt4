"""Explicit activation dtype and residual gauges for the pinned H3 runtime.

Checkpoint bytes and Native source are immutable. Folds apply once to privately
prepared modules. For y=x/c, norm epsilon is eps/c**2 and branch outputs are
divided by c before their storage cast. Public conditioning is unscaled FP32;
sampler state, time curves, statistics and final DiT heads remain FP32 islands.
"""
from dataclasses import dataclass, replace
import inspect
import types
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TensorFlowPolicy:
    mode: str = 'legacy'
    residual_exponent: int = 6
    vision_residual_exponent: int = 0
    video_residual_exponent: int = 0

    def __post_init__(self):
        if self.mode not in ('legacy', 'fp16', 'bf16_reference'):
            raise ValueError('Unknown H3 tensor flow')
        for value in (self.residual_exponent,self.vision_residual_exponent,self.video_residual_exponent):
            if type(value) is not int or not 0 <= value <= 12:
                raise ValueError('Residual exponent must be an integer in [0,12]')

    @property
    def enabled(self):return self.mode != 'legacy'

    @property
    def dtype(self):return torch.bfloat16 if self.mode == 'bf16_reference' else torch.float16

    @property
    def divisor(self):return float(2**self.residual_exponent)


def policy_for(reader):return getattr(reader, 'tensor_flow', TensorFlowPolicy())


def activation_cast(value, dtype):
    """Storage boundary after an FP32 numerical island."""
    return value.to(dtype)


class StableNorm(nn.Module):
    """FP32 statistics/gains/epsilon, followed by activation storage rounding."""
    def __init__(self, norm, divisor=1.):
        super().__init__()
        self.weight = norm.weight
        self.bias = getattr(norm, 'bias', None)
        self.normalized_shape = tuple(norm.normalized_shape)
        self.eps = float(norm.eps)/(divisor*divisor)
        self.layer_norm = 'LayerNorm' in type(norm).__name__
        # The legacy fused VAE API passes norms to CastBiasWeightContext.
        self.weight_function = list(getattr(norm, 'weight_function', ()))
        self.bias_function = list(getattr(norm, 'bias_function', ()))
        self.comfy_cast_weights = True

    def forward(self, x):
        weight = self.weight.to(device=x.device,dtype=torch.float32) if self.weight is not None else None
        for function in self.weight_function:weight=function(weight)
        if self.layer_norm:
            bias = self.bias.to(device=x.device,dtype=torch.float32) if self.bias is not None else None
            for function in self.bias_function:bias=function(bias)
            out = F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps)
        else:
            out = F.rms_norm(x.float(), self.normalized_shape, weight, self.eps)
        return activation_cast(out,x.dtype)


def stable_norms(module):
    for name, child in list(module.named_children()):
        if isinstance(child, StableNorm):continue
        if hasattr(child, 'normalized_shape') and hasattr(child, 'eps'):
            setattr(module, name, StableNorm(child))
        else:stable_norms(child)


def fold_linear(module, divisor):
    """Fold a scalar into dense weights or FP32 INT8 scales, plus bias/LoRA.

    No nonuniform gain is commuted through ConvRot, and no INT8 requantization
    is performed. NativeLoraLinear and TP adapters retain post-B scale order.
    """
    base = getattr(module, 'base', module)
    weight = base.weight
    if hasattr(base, 'weight_scale'):
        base.weight_scale = nn.Parameter(base.weight_scale.detach().float()/divisor, requires_grad=False)
    elif hasattr(weight, '_qdata'):
        if weight._params.scale.dtype != torch.float32:
            raise ValueError('Residual folding requires FP32 INT8 weight scales')
        params = replace(weight._params, scale=weight._params.scale/divisor)
        folded = weight._copy_with(qdata=weight._qdata, params=params, clone_params=False)
        folded._is_param = True
        base.weight = folded
    else:
        base.weight = nn.Parameter(weight.detach()/divisor, requires_grad=False)
    bias = getattr(base, 'bias', None)
    if bias is not None:base.bias = nn.Parameter(bias.detach()/divisor, requires_grad=False)
    fold_adapter(getattr(module, 'lora_adapter', None), divisor)


def fold_adapter(adapter, divisor):
    for term in getattr(adapter, 'terms', ()):
        term.scale /= divisor
    # LoRA A/B products use FP32 before post-B scaling in the FP16 flow. A
    # small post-B coefficient cannot rescue an already overflowing FP16 B.
    if adapter is not None:adapter.fp32_compute = True


def prepare_tp(block, kind, policy):
    if not policy.enabled:return block
    if hasattr(block, 'tensor_flow'):raise ValueError('Tensor flow already prepared')
    block.tensor_flow = policy
    c = policy.divisor
    for name in (('o_proj', 'down_proj') if kind == 'te' else ('out_proj', 'fc2')):
        fold_linear(getattr(block, name), c)
    stable_norms(block)
    for name in (('input_layernorm', 'post_attention_layernorm') if kind == 'te' else ('norm1', 'norm2')):
        getattr(block, name).eps /= c*c
    for child in block.modules():
        adapter = getattr(child, 'lora_adapter', None)
        if adapter is not None:adapter.fp32_compute = True
    return block


def residual_add(x, branch):
    return activation_cast(x.float()+branch.float(),x.dtype)


def modulated_norm(norm, x, shift, scale, segments):
    h = F.rms_norm(x.float(), norm.normalized_shape, norm.weight.float(), norm.eps)
    for a,b,row in segments:
        h[a:b].mul_(1.+scale[row].float()).add_(shift[row].float())
    return activation_cast(h,x.dtype)


def gated_add(x, gate, branch, segments):
    for a,b,row in segments:
        x[a:b].copy_(activation_cast(torch.addcmul(x[a:b].float(), branch[a:b].float(), gate[row].float()),x.dtype))
    return x


def _cast_input(module, args):
    return (args[0].to(module.tensor_flow.dtype), *args[1:])


def _te_input(module, args, kwargs):
    bound = inspect.signature(type(module).forward).bind_partial(module, *args, **kwargs)
    values = dict(bound.arguments);values.pop('self')
    p = module.tensor_flow
    embeds = values.get('embeds')
    if embeds is None:embeds = module.embed_tokens(values['x'], out_dtype=torch.float32)
    values['embeds'] = activation_cast(embeds.float()/p.divisor,p.dtype)
    deep = values.get('deepstack_embeds')
    if deep is not None:values['deepstack_embeds'] = [activation_cast(d.float()/p.divisor,p.dtype) for d in deep]
    values['dtype'] = p.dtype
    return (), values


def _te_output(module, args, output):
    # H3 exports an unnormalized hidden state. Restore its physical magnitude
    # in FP32; casting this export to FP16 could reintroduce the overflow.
    return tuple(t.float()*module.tensor_flow.divisor if i<2 and t is not None else t
                 for i,t in enumerate(output))


def prepare_te(model, policy):
    if not policy.enabled:return
    text = model.model
    text.tensor_flow = policy
    text.register_forward_pre_hook(_te_input, with_kwargs=True)
    text.register_forward_hook(_te_output)
    vision = model.visual
    vision.tensor_flow = policy
    vision.register_forward_pre_hook(_cast_input)
    c = float(2**policy.vision_residual_exponent)
    if c==1:return
    stable_norms(vision)
    fold_linear(vision.patch_embed.proj, c)
    vision.pos_embed.weight = nn.Parameter(vision.pos_embed.weight.detach()/c, requires_grad=False)
    for block in vision.blocks:
        block.norm1.eps /= c*c;block.norm2.eps /= c*c
        fold_linear(block.attn.proj, c);fold_linear(block.mlp.linear_fc2, c)
    for merger in [vision.merger, *vision.deepstack_merger_list]:merger.norm.eps /= c*c


def _scaled_pack(self, video, audio, context, layout, payload, options):
    p = self.tensor_flow
    return self._h3_pack_function(self, video, audio, (context.float()/p.divisor).to(p.dtype), layout, payload, options)


def _refiner_input(module, args):
    return (args[0].to(module.tensor_flow.dtype), *args[1:])


def prepare_dit(model, policy, rank=None):
    if not policy.enabled:return
    model.tensor_flow = policy;model.dtype = policy.dtype
    c = policy.divisor
    model._h3_pack_function = type(model)._embed_and_pack
    model._embed_and_pack = types.MethodType(_scaled_pack, model)
    model.final_layer.norm = StableNorm(model.final_layer.norm, c)
    if rank == 1:return
    # Refiner has its own gauge, ending at its final normalization. Its Native
    # SDPA, QK normalization, projections and online adapters are retained.
    fold_linear(model.condition_proj, c)
    model.condition_proj.tensor_flow = policy
    # TE export is FP32; divide in the condition projection before storage cast.
    # Keep this one projection FP32, then cast its scaled output at refiner entry.
    model.token_refiner.tensor_flow = policy
    model.token_refiner.register_forward_pre_hook(_refiner_input)
    stable_norms(model.token_refiner)
    for block in model.token_refiner.blocks:
        block.norm1.eps /= c*c;block.norm2.eps /= c*c
        attn = getattr(block.attn, 'base', block.attn)
        fold_linear(attn.out_proj, c);fold_linear(block.mlp.fc2, c)
        fold_adapter(getattr(block.mlp, 'fc2_adapter', None), c)
    model.token_refiner.final_norm.eps /= c*c
    for name in ('video_patch_proj', 'audio_patch_proj'):fold_linear(getattr(model, name), c)
    for child in model.token_refiner.modules():
        for name in ('lora_adapter','fc1_adapter','fc2_adapter'):
            adapter=getattr(child,name,None)
            if adapter is not None:adapter.fp32_compute=True


def prepare_video(model, policy):
    """Encoder is already FP16; decoder residual fixes preserve both decode paths.

    Encoder GN -> SiLU -> Conv is not homogeneous: retain that Native FP16
    computation rather than incorrectly folding a scale through SiLU.
    """
    if not policy.enabled:return model
    if hasattr(model, 'tensor_flow'):raise ValueError('Video tensor flow already prepared')
    model.tensor_flow = policy
    if hasattr(model, 'decoder') and policy.video_residual_exponent:
        decoder = model.decoder;c = float(2**policy.video_residual_exponent)
        stable_norms(decoder)
        fold_linear(decoder.x_embedder, c)
        decoder.register_tokens = nn.Parameter(decoder.register_tokens.detach()/c, requires_grad=False)
        for block in decoder.transformer_blocks:
            block.norm1.eps /= c*c;block.norm2.eps /= c*c
            fold_linear(block.attn.to_out, c);fold_linear(block.ff.w2, c)
        decoder.norm_out.eps /= c*c
    return model
