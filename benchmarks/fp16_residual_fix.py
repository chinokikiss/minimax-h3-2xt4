"""Experimental residual-stream gauge; never imported by the H3 runtime.

For y=x/c, RMSNorm uses eps/c**2 and every residual branch contributes b/c.
Only positive power-of-two c is accepted. FP32 norm/AdaLN/RoPE islands remain.
This is an approximate FP16 experiment, not a byte-exact runtime backend.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Gauge:
    exponent: int = 0
    storage_dtype: torch.dtype = torch.float16
    branch_dtype: torch.dtype = torch.float32
    fold_output: bool = False
    norm_gain_exponent: int = 0

    def __post_init__(self):
        if type(self.exponent) is not int or not 0 <= self.exponent <= 16:
            raise ValueError('Expected an integer power-of-two exponent in [0,16]')
        if self.storage_dtype not in (torch.float16, torch.float32):raise ValueError('Storage dtype')
        if self.branch_dtype not in (torch.float16, torch.float32):raise ValueError('Branch dtype')
        if type(self.norm_gain_exponent) is not int or not 0<=self.norm_gain_exponent<=8:
            raise ValueError('Norm gain exponent must be an integer in [0,8]')

    @property
    def divisor(self):return float(2**self.exponent)


def rms_from_scaled(y, weight, eps, divisor):
    # epsilon must remain FP32: eps/c**2 can underflow if cast to FP16.
    return F.rms_norm(y.float(), (y.shape[-1],), weight.float(), eps/(divisor*divisor))


def range_stats(x):
    x=x.detach().float();finite=torch.isfinite(x);absolute=x.abs()
    safe=absolute.masked_fill(~finite,0)
    return dict(elements=x.numel(),nonfinite=int((~finite).sum()),zeros=int((x==0).sum()),
                fp16_subnormal=int(((absolute>0)&(absolute<2**-14)&finite).sum()),
                above_fp16_max=int(((absolute>65504)&finite).sum()),maximum=float(safe.max()))


def storage_stats(exact, stored):
    result=range_stats(stored)
    result['cast_nonzero_to_zero']=int(((exact!=0)&(stored==0)&torch.isfinite(exact)).sum())
    result['cast_finite_to_nonfinite']=int((torch.isfinite(exact)&~torch.isfinite(stored)).sum())
    return result


@contextmanager
def folded_rows(block, divisor):
    """Fold b/c into FP32 dequant scales and post-B LoRA strengths.

    No INT8 weight bytes are edited; no LoRA strength is moved before B GEMM.
    Mutations apply to a private experimental block and are always restored.
    """
    if not math.isfinite(divisor) or divisor<=0:raise ValueError('Positive divisor required')
    records=[]
    try:
        for name in ('out_proj','fc2'):
            module=getattr(block,name);original=module.weight_scale
            if original.dtype!=torch.float32:raise ValueError('FP32 weight scales required')
            folded=original.detach()/divisor
            if not torch.isfinite(folded).all() or ((original!=0)&(folded==0)).any():
                raise ValueError('Folded weight scale loses finite/nonzero values')
            terms=[]
            records.append((module,original,terms))
            module.weight_scale=torch.nn.Parameter(folded,requires_grad=False)
            # H3 row projections have no bias. Reject new schemas explicitly.
            if getattr(module,'bias',None) is not None:raise ValueError('Bias folding requires its own qualified path')
            adapter=getattr(module,'lora_adapter',None)
            for term in getattr(adapter,'terms',()):
                terms.append((term,term.scale));term.scale=term.scale/divisor
        yield
    finally:
        for module,original,terms in reversed(records):
            module.weight_scale=original
            for term,scale in terms:term.scale=scale


@contextmanager
def boosted_norm_gain(block, exponent):
    """Scalar norm gain folding without requantizing ConvRot weights.

    Multiply norm gains AND AdaLN shifts by p, divide following QKV/FC1
    dequant scales and post-B adapter strength by p. Dynamic modulation scales
    and residual gates are unchanged. Nonlinearities see compensated outputs.
    """
    if type(exponent) is not int or not 0<=exponent<=8:raise ValueError('Norm gain exponent')
    if not isinstance(block.adaln_proj,torch.nn.Linear):
        raise ValueError('AdaLN adapters require additional row-wise compensation; not qualified')
    p=float(2**exponent);records=[];terms=[]
    def replace(module,name,value):
        original=getattr(module,name)
        if not torch.isfinite(value).all():raise ValueError('Nonfinite folded parameter')
        records.append((module,name,original))
        setattr(module,name,torch.nn.Parameter(value,requires_grad=False))
    try:
        for norm in (block.norm1,block.norm2):replace(norm,'weight',norm.weight.detach()*p)
        w=block.adaln_proj.weight.detach().clone();b=block.adaln_proj.bias.detach().clone()
        for index in (0,3):
            w.view(-1,6,block.hidden,w.shape[-1])[:,index].mul_(p)
            b.view(-1,6,block.hidden)[:,index].mul_(p)
        replace(block.adaln_proj,'weight',w);replace(block.adaln_proj,'bias',b)
        for projection in (block.qkv_proj,block.fc1):
            replace(projection,'weight_scale',projection.weight_scale.detach()/p)
            for term in getattr(getattr(projection,'lora_adapter',None),'terms',()):
                terms.append((term,term.scale));term.scale/=p
        yield
    finally:
        for module,name,original in reversed(records):setattr(module,name,original)
        for term,scale in terms:term.scale=scale


def dit_block(block, scaled, t_emb, segments, rope, gauge, observations=None):
    """One private TP2 DiT block; caller provides actual (or simulated) comm.

    FP16 branch mode keeps FP32 accumulation islands, but changes projection
    output/activation precision and possibly Native ConvRot dispatch.
    """
    import comfy_kitchen as ck
    from patches.h3_operator_policy import dit_attention
    c=gauge.divisor;y=scaled.clone()
    mod=block.adaln_proj(F.silu(t_emb) if block.apply_silu else t_emb).view(-1,6*block.hidden)
    shift_a,scale_a,gate_a,shift_m,scale_m,gate_m=mod.chunk(6,-1)
    def observe(name,x):
        if observations is not None:observations[name]=range_stats(x)
    def norm_mod(norm,shift,scale):
        h=rms_from_scaled(y,norm.weight,norm.eps,c)
        block._mod_scale_shift(h,shift,scale,segments)
        name='modulated_'+('attention' if norm is block.norm1 else 'mlp')
        observe(name,h);cast=h.to(gauge.branch_dtype)
        if observations is not None:observations[name+'_cast']=storage_stats(h,cast)
        return cast
    def update(branch,gate,label):
        observe(label+'_projection',branch)
        value=branch.float()
        if not gauge.fold_output:value=value/c
        lost=nonzero=underflow=overflow=0
        for a,b,row in segments:
            g=block._mod_row(gate,row,torch.float32)
            before=y[a:b].float();increment=value[a:b]*g
            exact=torch.addcmul(before,value[a:b],g)
            stored=exact.to(gauge.storage_dtype)
            if observations is not None:
                valid=torch.isfinite(before)&torch.isfinite(increment)&(increment!=0)
                lost+=int((valid&(stored.float()==before)).sum());nonzero+=int(valid.sum())
                underflow+=int(((exact!=0)&(stored==0)&torch.isfinite(exact)).sum())
                overflow+=int((torch.isfinite(exact)&~torch.isfinite(stored)).sum())
            y[a:b].copy_(stored)
        if observations is not None:
            observations[label+'_residual']=dict(**range_stats(y),nonzero_updates=nonzero,
                rounded_away_updates=lost,cast_nonzero_to_zero=underflow,cast_finite_to_nonfinite=overflow)
    h=norm_mod(block.norm1,shift_a,scale_a)
    qkv=block.qkv_proj(h);s=h.shape[0];del h
    q,k,v=[x.view(s,block.heads_per_rank,block.head_dim) for x in qkv.split(block.inner_per_rank,-1)]
    # Norm, learned gains and RoPE remain FP32 islands in both ablation arms.
    if rope is not None:
        q,k=ck.rms_rope_split_half(q.float()[None],k.float()[None],rope.float(),
            block.q_norm.weight.float(),block.k_norm.weight.float(),
            epsilon=block.q_norm.eps,rot_dim=rope.shape[-3]*2)
        q,k=q[0],k[0]
    else:
        q=F.rms_norm(q.float(),(block.head_dim,),block.q_norm.weight.float(),block.q_norm.eps)
        k=F.rms_norm(k.float(),(block.head_dim,),block.k_norm.weight.float(),block.k_norm.eps)
    q,k=q.to(gauge.branch_dtype),k.to(gauge.branch_dtype)
    observe('q',q);observe('k',k);observe('v',v)
    local=dit_attention(block,q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],{})
    out=block.out_proj(local.reshape(s,block.inner_per_rank));del local,q,k,v,qkv
    update(out,gate_a,'attention');del out
    packed=block.fc1(norm_mod(block.norm2,shift_m,scale_m));observe('fc1',packed)
    out=block.fc2(packed,input_act='swiglu');del packed
    update(out,gate_m,'mlp');del out
    return y


def fold_rms_gain_int8(weight, scale, gamma, *, modulated=False):
    """TE-style unmodulated input gain folding; introduces requantization.

    Dequantize through Native ConvRot, multiply input columns in the original
    basis, then Native requantize full width. Multiplying rotated INT8 columns
    directly by gamma is mathematically wrong in general.
    """
    if modulated:raise ValueError('Dynamic AdaLN shifts need compensation; static gain folding is not valid as-is')
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as Layout
    if gamma.shape!=(weight.shape[1],):raise ValueError('Input-channel gamma required')
    params=Layout.Params(scale=scale,orig_dtype=torch.float32,orig_shape=tuple(weight.shape),
                         is_weight=True,convrot=True,convrot_groupsize=256)
    original=Layout.dequantize(weight,params)
    folded=original*gamma.float()[None]
    q,params=Layout.quantize(folded,per_channel=True,convrot=True,convrot_groupsize=256,stochastic_rounding=0)
    return q,params.scale,original,folded
