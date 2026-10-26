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
    scalar_score_workspace_mib: int = 96
    fc2_quantizer: str = 'fused_stored'
    column_linear_backend: str = 'cuda_int8_tilelang'

    def __post_init__(self):
        if self.te_attention not in ('kitchen_causal','tilelang_group32','native'):raise ValueError('Unknown TE attention')
        if self.dit_attention not in ('kitchen','tilelang_dense','tilelang_drop','tilelang_scalar','native'):
            raise ValueError('Unknown DiT attention')
        if self.row_linear_backend not in ('tilelang_packed','tilelang_scaled_fp16','native'):raise ValueError('Unknown row-linear backend')
        if self.fc2_quantizer not in ('native','owned_fp32','fused_stored','fused_recompute'):
            raise ValueError('Unknown FC2 quantizer')
        if type(self.scalar_score_workspace_mib) is not int or self.scalar_score_workspace_mib<0:
            raise ValueError('Nonnegative scalar score workspace MiB required')
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
    if not sm75(q,'TE Group32 attention'):return None
    from patches.tilelang_group32_masked import attention
    return attention(q,k,v,mask=mask)


def packed_prefix(options,tokens):
    layout=options.get('minimax_h3_layout')
    segments=getattr(layout,'segments',None)
    if not segments or getattr(layout,'seq_len',None)!=tokens:
        raise ValueError('Sparse DiT attention requires the matching Native PackedLayout')
    video=[(a,b) for a,b,kind in segments if kind=='video']
    if len(video)!=1 or video[0][1]!=tokens or not 0<=video[0][0]<tokens:
        raise ValueError('Sparse attention requires one final generated-video span')
    # All text, references, conditioning and generated audio remain in the dense prefix.
    return video[0][0]


def dit_attention(block,q,k,v,options):
    policy=getattr(block,'operator_policy',None)
    mode=policy.dit_attention if policy is not None else 'native'
    if mode=='native':out=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    elif mode=='kitchen':
        import comfy_kitchen as ck
        out=ck.int8_attention(q,k,v)
    elif not sm75(q,'DiT '+mode):
        out=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    elif mode=='tilelang_dense':
        from patches import tilelang_grouped_tp_kitchen_v2 as dense
        p=dense.prepare(q,k,v,query_tile=64)
        return dense.execute(p,cache='q',layout='BND',output_dtype=q.dtype,threads=128,warp_policy='row')
    else:
        from patches import tilelang_grouped_tp_optimized as sparse
        prefix=packed_prefix(options,q.shape[-2])
        p=sparse.prepare(q.half(),k.half(),v.half(),fused=True,
            mode='drop' if mode=='tilelang_drop' else 'scalar',query_tile=64)
        routes=sparse.routes(p,prefix_tokens=prefix,fast_tau0=True)
        if mode=='tilelang_scalar':
            b,h,np,d=p.q.shape;nb=p.kc.shape[2];workspace=b*h*np*nb*4
            if workspace<=policy.scalar_score_workspace_mib*2**20:
                from patches import tilelang_attention_p2 as split
                scores=split.centroid_scores(b,h,np,nb,d,omitted=True)(p.q,p.kc,routes)
                return split.execute(p,routes,output_dtype=q.dtype,centroids=scores)
        return sparse.execute(p,routes,layout='BND',output_dtype=q.dtype)
    return out.transpose(1,2).reshape(q.shape[0],q.shape[2],-1)


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
