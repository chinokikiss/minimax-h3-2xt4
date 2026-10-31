"""Per-session operator selection, carried through CPU/GPU residency copies."""
from dataclasses import dataclass
from functools import lru_cache
import logging
import torch
from torch import nn


@dataclass(frozen=True)
class H3OperatorPolicy:
    te_attention: str = 'kitchen_causal'
    dit_attention: str = 'kitchen'
    row_linear_backend: str = 'tilelang_scaled_fp16'
    fc2_quantizer: str = 'fused_stored'
    column_linear_backend: str = 'cuda_int8_tilelang'
    dit_rms_rope: str = 'tilelang_fp32'
    aot_precompile: bool = True
    dit_fused_norm: bool = True
    dit_fused_gate: bool = True
    dit_fused_row_gate: bool = True
    dit_prepared_attention: bool = True

    def __post_init__(self):
        for name in ('dit_fused_norm','dit_fused_gate','dit_fused_row_gate','dit_prepared_attention'):
            if type(getattr(self,name)) is not bool:raise ValueError(name+' must be boolean')
        if self.dit_rms_rope not in ('native','tilelang_fp32'):raise ValueError('Unknown DiT RMS-RoPE backend')
        if self.te_attention not in ('kitchen_causal','native'):raise ValueError('Unknown TE attention')
        if self.dit_attention not in ('kitchen','sol','native'):raise ValueError('Unknown DiT attention')
        if self.row_linear_backend not in ('tilelang_packed','tilelang_scaled_fp16','native'):raise ValueError('Unknown row-linear backend')
        if self.fc2_quantizer not in ('native','owned_fp32','fused_stored','fused_recompute'):
            raise ValueError('Unknown FC2 quantizer')
        if self.column_linear_backend not in ('preallocated','cuda_int8_tilelang'):
            raise ValueError('Unknown column linear backend')


@lru_cache(maxsize=32)
def _fallback(name,device,capability):
    logging.warning('H3 %s: TileLang SM75 path unavailable on %s (%s); using native operator',name,device,capability)


def sm75(x,operator):
    capability=torch.cuda.get_device_capability(x.device) if x.is_cuda else None
    if capability==(7,5):return True
    _fallback(operator,str(x.device),capability)
    return False


def configure_block(block,kind,policy):
    block.operator_policy=policy
    for name in (('o_proj','down_proj') if kind=='te' else ('out_proj','fc2')):
        getattr(block,name).operator_policy=policy
    if kind=='dit':
        for name in ('qkv_proj','fc1'):
            column=getattr(block,name,None)
            if column is not None:column.operator_policy=policy
    return block


def te_attention(block,q,k,v,mask):
    policy=getattr(block,'operator_policy',None)
    if policy is None or policy.te_attention=='native':return None
    if policy.te_attention=='kitchen_causal':
        if not q.is_cuda:return None
        from patches.kitchen_te_attention import attention
        return attention(q,k,v,mask=mask)
    raise ValueError('Unknown TE attention')


def protected_prefix(options,tokens):
    """Return the exact-protected conditioning prefix before generated targets."""
    layout=options.get('minimax_h3_layout')
    segments=getattr(layout,'segments',None)
    if not segments or getattr(layout,'seq_len',None)!=tokens:
        raise ValueError('Sol DiT attention requires the matching Native PackedLayout')
    if len(segments)<2:
        raise ValueError('Sol attention requires Native target audio/video segments')
    audio_a,audio_b,audio_kind=segments[-2]
    video_a,video_b,video_kind=segments[-1]
    if (audio_kind!='audio' or video_kind!='video' or audio_b!=video_a
            or video_b!=tokens or not 0<=audio_a<=audio_b<=video_b):
        raise ValueError('Sol attention requires final Native audio/video target segments')
    # Protect only conditioning/reference rows. Target audio and target video
    # remain generated tokens and use normal Sol routing. Native Q64/K64 routing
    # rounds the protected boundary outward to a complete block.
    return audio_a


def sol_v2_routing(options,tokens):
    """Return the experimental H3-aware Sol prefix and video token topology."""
    options=options or {}
    layout=options.get('minimax_h3_layout')
    segments=getattr(layout,'segments',None)
    if not segments or getattr(layout,'seq_len',None)!=tokens or len(segments)<2:
        raise ValueError('Sol v2 requires the matching Native PackedLayout')
    audio_a,audio_b,audio_kind=segments[-2]
    video_a,video_b,video_kind=segments[-1]
    if (audio_kind!='audio' or video_kind!='video' or audio_b!=video_a
            or video_b!=tokens or not 0<=audio_a<=audio_b<=video_b):
        raise ValueError('Sol v2 requires final Native audio/video target segments')
    signature=getattr(layout,'signature',None)
    if not signature or len(signature)<2:
        raise ValueError('Sol v2 requires PackedLayout latent geometry')
    latent_t=int(signature[1])
    video_tokens=video_b-video_a
    if latent_t<=0 or video_tokens<=0 or video_tokens%latent_t:
        raise ValueError('Sol v2 video segment is not frame-regular')
    frame_tokens=video_tokens//latent_t
    topology=dict(
        kind='dit',
        nfe=options.get('_h3_nfe'),
        start_token=video_a,
        end_token=video_b,
        frame_stride_tokens=frame_tokens,
        temporal_radius=int(options.get('_h3_sol_temporal_radius',1)),
        local_radius_tokens=int(options.get('_h3_sol_local_radius_tokens',64)),
        protect_frame_boundaries=bool(options.get('_h3_sol_protect_frame_boundaries',True)),
    )
    # Protect conditioning + generated audio. Outward Q64/K64 rounding also
    # absorbs the mixed audio/video boundary block.
    return video_a,topology


# Historical name retained for archived benchmark readers.
packed_prefix = protected_prefix


def _dit_mode(block,options):
    policy=getattr(block,'operator_policy',None)
    default=policy.dit_attention if policy is not None else 'native'
    mode=(options or {}).get('_h3_dit_attention',default)
    if mode not in ('kitchen','sol','native'):
        raise ValueError('Unknown DiT attention')
    return mode


def dit_attention(block,q,k,v,options):
    mode=_dit_mode(block,options)
    if mode=='native':
        out=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    elif mode=='kitchen':
        from patches.kitchen_attention import int8_attention
        out=int8_attention(q,k,v)
    else:
        from patches.sol_attention import attention
        options=options or {}
        if options.get('_h3_sol_topology_v2',True):
            prefix,topology=sol_v2_routing(options,q.shape[-2])
        else:
            prefix,topology=protected_prefix(options,q.shape[-2]),None
        tau=options.get('_h3_sol_tau',1.0)
        out=attention(
            q,k,v,protected_tokens=prefix,tau=tau,topology=topology,
            sync_free_lut=bool(options.get('_h3_sol_sync_free_lut',True)),
            collect_route_stats=bool(options.get('_h3_sol_collect_route_stats',False)),
        )
    return out.transpose(1,2).reshape(q.shape[0],q.shape[2],-1)


def prepare_dit_attention(block,q,k,v,options=None):
    """Prepare attention so tp_dit can release all floating QKV owners early."""
    policy=getattr(block,'operator_policy',None)
    mode=_dit_mode(block,options)
    if (policy is None or mode not in ('kitchen','sol')
            or not getattr(policy,'dit_prepared_attention',False)
            or torch.is_grad_enabled() or torch.compiler.is_compiling()
            or not q.is_cuda or q.dtype!=torch.float16 or q.shape!=k.shape or q.shape!=v.shape
            or q.ndim!=4 or q.shape[0]!=1 or q.shape[1]!=28 or q.shape[-1]!=128
            or k.dtype!=q.dtype or v.dtype!=q.dtype or k.device!=q.device or v.device!=q.device
            or torch.cuda.get_device_capability(q.device)!=(7,5)):return None
    if mode=='kitchen':
        import comfy_kitchen as ck
        if not callable(getattr(ck,'prequantize_int8_attention',None)) or not callable(getattr(ck,'int8_attention_from_prequantized',None)):
            return None
        return ck.prequantize_int8_attention(q,k,v)

    from patches.sol_attention import prepare_attention
    options=options or {}
    if options.get('_h3_sol_topology_v2',True):
        prefix,topology=sol_v2_routing(options,q.shape[-2])
    else:
        prefix,topology=protected_prefix(options,q.shape[-2]),None
    return prepare_attention(
        q,k,v,protected_tokens=prefix,tau=options.get('_h3_sol_tau',1.0),
        topology=topology,
        sync_free_lut=bool(options.get('_h3_sol_sync_free_lut',True)),
        collect_route_stats=bool(options.get('_h3_sol_collect_route_stats',False)),
    )


def execute_dit_attention(prepared):
    if isinstance(prepared,dict) and prepared.get('_h3_backend')=='sol':
        from patches.sol_attention import execute_prepared
        out=execute_prepared(prepared)
    else:
        import comfy_kitchen as ck
        out=ck.int8_attention_from_prequantized(prepared)
    return out.transpose(1,2).reshape(out.shape[0],out.shape[2],-1)


def row_linear(module,x,fallback,comm,input_act=None):
    policy=getattr(module,'operator_policy',None)
    fused=(policy is not None and policy.fc2_quantizer in ('fused_stored','fused_recompute')
        and input_act=='swiglu' and policy.row_linear_backend!='native')
    if fused:
        from patches import tilelang_swiglu_quant as quant
        if quant.supported(x,module.weight,module.weight_scale,module.world_size,module.convrot_groupsize):
            return quant.row_linear(x,module.weight,module.weight_scale,comm,mode=policy.fc2_quantizer,
                scaled_transport=policy.row_linear_backend=='tilelang_scaled_fp16')
        # Keep the branch-product FP32 island on fallback hardware/geometry.
        # This cast happens before SwiGLU, never after an overflowing product.
        sm75(x,'FP32 SwiGLU quantization')
        x=x.float()
    elif policy is not None and policy.row_linear_backend=='tilelang_scaled_fp16':
        # FP16 residual storage does not require FP16 quantization arithmetic.
        x=x.float()
    args=(x,module.weight,module.weight_scale,comm,module.world_size,module.convrot_groupsize,input_act)
    if policy is not None and policy.row_linear_backend=='tilelang_scaled_fp16' and sm75(x,'scaled FP16 row-linear'):
        from patches.tilelang_transport_p2 import row_linear as run
        return run(*args)
    if policy is not None and policy.row_linear_backend=='tilelang_packed' and sm75(x,'packed row-linear'):
        from patches.tilelang_swiglu_pad import row_linear as run
        rotation=policy.fc2_quantizer if policy.fc2_quantizer in ('native','owned_fp32') else 'owned_fp32'
        return run(*args,variant='packed',rotation_mode=rotation)
    return fallback(*args)


def row_linear_parts(module,gate,up,fallback,comm):
    policy=getattr(module,'operator_policy',None)
    if (policy is not None and policy.row_linear_backend!='native'
            and policy.fc2_quantizer in ('fused_stored','fused_recompute')):
        if gate.shape!=up.shape or gate.dtype!=up.dtype or gate.device!=up.device:
            raise ValueError('Matching gate/up required')
        return row_linear(module,torch.cat((gate,up),-1),fallback,comm,'swiglu')
    if (policy is not None and policy.row_linear_backend=='tilelang_scaled_fp16'
            and sm75(gate,'scaled FP16 SwiGLU padding')):
        from patches.tilelang_transport_p2 import row_linear_parts as run
        return run(gate,up,module.weight,module.weight_scale,comm,module.world_size,module.convrot_groupsize)
    if (policy is not None and policy.row_linear_backend=='tilelang_packed'
            and sm75(gate,'SwiGLU padding')):
        from patches.tilelang_swiglu_pad import row_linear_parts as run
        return run(gate,up,module.weight,module.weight_scale,comm,module.world_size,module.convrot_groupsize)
    return row_linear(module,torch.nn.functional.silu(gate)*up,fallback,comm)


class NativeRefinerAttention(nn.Module):
    """Keep Native refiner projections/norm/RoPE, selecting PyTorch SDPA.

    A module wrapper survives residency copies without bound methods referring
    to a CPU canonical module. Refiner tokens are never sparsified.
    """
    def __init__(self,base):super().__init__();self.base=base

    def forward(self,x,rope_freqs=None,transformer_options=None):
        import comfy_kitchen as ck
        import comfy.model_management
        base=self.base;s=x.shape[0];d=base.head_dim;h=base.heads
        q,k,v=base.qkv_proj(x).split(h*d,dim=-1)
        v=v.view(s,h,d)
        if rope_freqs is not None:
            q=q.view(1,s,h,d);k=k.view(1,s,h,d)
            qw=comfy.model_management.cast_to(base.q_norm.weight,device=x.device)
            kw=comfy.model_management.cast_to(base.k_norm.weight,device=x.device)
            if comfy.model_management.in_training:raise ValueError('TP2 refiner integration is inference-only')
            ck.rms_rope_split_half_(q,k,rope_freqs,qw,kw,epsilon=base.q_norm.eps,rot_dim=rope_freqs.shape[-3]*2)
            q,k=q[0],k[0]
        else:q,k=base.q_norm(q.view(s,h,d)),base.k_norm(k.view(s,h,d))
        q,k,v=(a.transpose(0,1).unsqueeze(0) for a in (q,k,v))
        out=torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(s,h*d)
        del q,k,v
        return base.out_proj(out)


# Retain the historical import name for existing callers and serialized modules.
KitchenRefinerAttention = NativeRefinerAttention
