"""Historical dense-FP32 experiment; excluded from Native FP32-flow evaluation.

This helper changes operators and is NOT the FP32 tensor-flow reference requested
by the user. Production/native-flow evaluation must not enter this context.

Uses Native inverse-ConvRot dequantization and dense FP32 linear/attention
computation. Query chunking retains every key and the exact mask coordinates.
It never modifies checkpoint bytes or production operator selection.
"""
from contextlib import ExitStack,contextmanager
from unittest.mock import patch
import torch
from torch.nn import functional as F


def query_chunked_sdpa(original,q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,
                       scale=None,enable_gqa=False,query_chunk=128):
    if any(t.dtype!=torch.float32 for t in (q,k,v)):
        raise ValueError('FP32 ground-truth attention received a low-precision tensor')
    if dropout_p:raise ValueError('Ground truth is deterministic inference only')
    length=q.shape[-2];parts=[]
    for start in range(0,length,query_chunk):
        end=min(start+query_chunk,length);mask=attn_mask
        if mask is not None and mask.ndim>=2 and mask.shape[-2]==length:
            mask=mask[...,start:end,:]
        if is_causal:
            allowed=torch.arange(k.shape[-2],device=q.device)[None,:]<=torch.arange(start,end,device=q.device)[:,None]
            if mask is None:mask=allowed
            elif mask.dtype==torch.bool:mask=mask&allowed
            else:mask=mask.masked_fill(~allowed,float('-inf'))
        parts.append(original(q[...,start:end,:],k,v,attn_mask=mask,dropout_p=0.,
            is_causal=False,scale=scale,enable_gqa=enable_gqa))
    return torch.cat(parts,dim=-2)


@contextmanager
def dense_fp32(model):
    """Force dense GEMMs and SDPA, and fail if INT8 linear or low dtype leaks in."""
    import comfy_kitchen as ck
    from comfy.ldm.modules.attention import attention_pytorch
    records=dict(dense_linear_calls=0,dense_conv_calls=0,sdpa_calls=0,
        kitchen_attention_redirects=0,int8_linear_calls=0,query_chunk=128,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        linear_input_dtypes=set(),linear_weight_dtypes=set(),attention_dtypes=set())
    if records['tf32_matmul'] or records['tf32_cudnn']:
        raise ValueError('Disable TF32 before FP32 ground-truth execution')
    linear=F.linear;conv=F.conv3d;sdpa=F.scaled_dot_product_attention
    def checked_linear(x,w,b=None):
        if x.dtype!=torch.float32 or w.dtype!=torch.float32 or (b is not None and b.dtype!=torch.float32):
            raise ValueError('Dense FP32 ground truth received a low-precision linear')
        records['dense_linear_calls']+=1
        records['linear_input_dtypes'].add(str(x.dtype));records['linear_weight_dtypes'].add(str(w.dtype))
        return linear(x,w,b)
    def checked_conv(x,w,b=None,*args,**kwargs):
        if x.dtype!=torch.float32 or w.dtype!=torch.float32 or (b is not None and b.dtype!=torch.float32):
            raise ValueError('Dense FP32 ground truth received a low-precision convolution')
        records['dense_conv_calls']+=1
        return conv(x,w,b,*args,**kwargs)
    def checked_sdpa(q,k,v,*args,**kwargs):
        records['sdpa_calls']+=1;records['attention_dtypes'].update(str(t.dtype) for t in (q,k,v))
        result=query_chunked_sdpa(sdpa,q,k,v,*args,**kwargs)
        if not torch.isfinite(result).all():raise ValueError('Nonfinite FP32 attention before Native sanitization')
        return result
    def redirect(q,k,v,*args,**kwargs):
        if args or kwargs:raise ValueError('Unexpected Kitchen attention arguments in reference')
        records['kitchen_attention_redirects']+=1
        return checked_sdpa(q,k,v)
    def reject_int8(*args,**kwargs):
        records['int8_linear_calls']+=1
        raise ValueError('INT8 activation GEMM leaked into FP32 ground truth')
    with ExitStack() as stack:
        for module in model.modules():
            if hasattr(module,'_full_precision_mm'):stack.enter_context(patch.object(module,'_full_precision_mm',True))
            preferred=getattr(module,'comfy_attention',None)
            if preferred is not None:stack.enter_context(patch.object(preferred,'function',attention_pytorch))
        stack.enter_context(patch.object(F,'linear',checked_linear))
        stack.enter_context(patch.object(F,'conv3d',checked_conv))
        stack.enter_context(patch.object(F,'scaled_dot_product_attention',checked_sdpa))
        stack.enter_context(patch.object(ck,'int8_attention',redirect))
        stack.enter_context(patch.object(ck,'int8_linear',reject_int8))
        try:yield records
        finally:
            for key in ('linear_input_dtypes','linear_weight_dtypes','attention_dtypes'):
                records[key]=sorted(records[key])
