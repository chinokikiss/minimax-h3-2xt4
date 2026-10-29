"""T4 DiT fusion with FP32 arithmetic islands and FP16 storage boundaries.

Factories are promoted unchanged from the physical 2026-10-29 operator run.
Unsupported layouts, legacy/BF16 flow and online LoRA retain existing operators.
"""
from functools import lru_cache
import threading
import torch

@lru_cache(maxsize=32)
def norm_kernel(rows, width, modulation_rows, epsilon, arch='sm_75'):
    import tilelang
    import tilelang.language as T
    padded = 1 << (width-1).bit_length()
    @T.prim_func
    def main(X:T.Tensor((rows,width),'float16'),W:T.Tensor((width,),'float32'),
             Shift:T.Tensor((modulation_rows,width),'float32'),
             Scale:T.Tensor((modulation_rows,width),'float32'),
             Labels:T.Tensor((rows,),'int32'),Y:T.Tensor((rows,width),'float16')):
        with T.Kernel(rows,threads=256) as row:
            squares=T.alloc_fragment((padded,),'float32')
            total=T.alloc_fragment((1,),'float32')
            for i in T.Parallel(padded):
                value=T.if_then_else(i<width,T.cast(X[row,i],'float32'),T.float32(0.))
                squares[i]=T.call_pure_extern('float32','__fmul_rn',value,value)
            T.reduce_sum(squares,total,dim=0)
            for i in T.Parallel(width):
                mean=T.call_pure_extern('float32','__fmul_rn',total[0],T.float32(1./width))
                inv=T.call_pure_extern('float32','rsqrtf',T.call_pure_extern('float32','__fadd_rn',mean,T.float32(epsilon)))
                value=T.call_pure_extern('float32','__fmul_rn',T.cast(X[row,i],'float32'),inv)
                gain=T.call_pure_extern('float32','__fmul_rn',value,W[i])
                factor=T.call_pure_extern('float32','__fadd_rn',T.float32(1.),Scale[Labels[row],i])
                scaled=T.call_pure_extern('float32','__fmul_rn',gain,factor)
                Y[row,i]=T.cast(T.call_pure_extern('float32','__fadd_rn',scaled,Shift[Labels[row],i]),'float16')
    return tilelang.compile(main,out_idx=[5],target={'kind':'cuda','arch':arch},
                            pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=32)
def gate_kernel(rows,width,modulation_rows,arch='sm_75'):
    import tilelang
    import tilelang.language as T
    @T.prim_func
    def main(X:T.Tensor((rows,width),'float16'),B:T.Tensor((rows,width),'float16'),
             Gate:T.Tensor((modulation_rows,width),'float32'),Labels:T.Tensor((rows,),'int32')):
        with T.Kernel(rows,T.ceildiv(width,1024),threads=128) as (r,bx):
            for i in T.Parallel(1024):
                c=bx*1024+i
                if c<width:
                    value=T.call_pure_extern('float32','__fmaf_rn',T.cast(B[r,c],'float32'),Gate[Labels[r],c],T.cast(X[r,c],'float32'))
                    X[r,c]=T.cast(value,'float16')
    return tilelang.compile(main,out_idx=[],target={'kind':'cuda','arch':arch},
                            pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=32)
def row_gate_kernel(rows,width,modulation_rows,shift,arch='sm_75'):
    import tilelang
    import tilelang.language as T
    @T.prim_func
    def main(P:T.Tensor((rows,width),'float16'),A:T.Tensor((rows,1),'float32'),
             W:T.Tensor((width,),'float32'),X:T.Tensor((rows,width),'float16'),
             Gate:T.Tensor((modulation_rows,width),'float32'),Labels:T.Tensor((rows,),'int32')):
        with T.Kernel(rows,T.ceildiv(width,1024),threads=128) as (r,bx):
            for i in T.Parallel(1024):
                c=bx*1024+i
                if c<width:
                    total=T.call_pure_extern('float32','__fmul_rn',T.cast(P[r,c],'float32'),T.float32(2.**shift))
                    restored=T.cast(total,'int32')
                    value=T.call_pure_extern('float32','__fmul_rn',T.cast(restored,'float32'),A[r,0])
                    dequant=T.call_pure_extern('float32','__fmul_rn',value,W[c])
                    # This rounding is part of the existing operator, even
                    # though the branch is never stored in a full FP16 array.
                    branch=T.cast(T.cast(dequant,'float16'),'float32')
                    X[r,c]=T.cast(T.call_pure_extern('float32','__fmaf_rn',branch,Gate[Labels[r],c],T.cast(X[r,c],'float32')),'float16')
    return tilelang.compile(main,out_idx=[],target={'kind':'cuda','arch':arch},
                            pass_configs={'tl.enable_async_copy':False})


def supported(x):
    return (not torch.is_grad_enabled() and not torch.compiler.is_compiling()
        and x.is_cuda and x.ndim==2 and x.shape[0]>0 and x.shape[1]==5376
        and x.dtype==torch.float16 and x.is_contiguous()
        and torch.cuda.get_device_capability(x.device)==(7,5))


def _table(value,x):
    return (value.ndim==2 and value.shape==(9,x.shape[1]) and value.dtype==torch.float32
            and value.device==x.device)


def segment_key(rows,segments):
    key=tuple(tuple(s) for s in segments)
    cursor=0
    for a,b,row in sorted(key):
        if (any(type(v) is not int for v in (a,b,row)) or a!=cursor
                or not a<b<=rows or not 0<=row<9):return None
        cursor=b
    return key if cursor==rows else None


_label_lock=threading.RLock()


@lru_cache(maxsize=4)
def _labels(rows,segments,device):
    labels=torch.empty(rows,dtype=torch.int32)
    for a,b,row in segments:labels[a:b]=row
    labels=labels.to(device)
    ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(device))
    return labels,ready


def labels_for(x,segments):
    key=segment_key(x.shape[0],segments)
    if key is None:return None
    # The immutable cache is bounded; record_stream protects an evicted tensor
    # when another request/stream is still consuming it asynchronously.
    with _label_lock:
        labels,ready=_labels(x.shape[0],key,str(x.device))
    stream=torch.cuda.current_stream(x.device)
    stream.wait_event(ready);labels.record_stream(stream)
    return labels


def modulated_norm(norm,x,shift,scale,segments):
    if (not supported(x) or not _table(shift,x) or not _table(scale,x)
            or getattr(norm,'layer_norm',False) or tuple(norm.normalized_shape)!=(5376,)
            or norm.weight.dtype!=torch.float32 or norm.weight.device!=x.device):return None
    labels=labels_for(x,segments)
    if labels is None:return None
    return norm_kernel(*x.shape,9,float(norm.eps))(
        x,norm.weight.contiguous(),shift.contiguous(),scale.contiguous(),labels)


def gated_add(x,gate,branch,segments):
    if (not supported(x) or not _table(gate,x) or branch.shape!=x.shape
            or branch.dtype!=x.dtype or branch.device!=x.device or not branch.is_contiguous()):return None
    labels=labels_for(x,segments)
    if labels is None:return None
    gate_kernel(*x.shape,9)(x,branch,gate.contiguous(),labels)
    return x


def finish_row(payload,scale,weights,residual,gate,labels,local_k):
    from patches.tilelang_transport_tradeoffs import shift_for
    row_gate_kernel(*residual.shape,9,shift_for(local_k))(
        payload,scale,weights.reshape(-1),residual,gate,labels)
    return residual


def row_gated(module,value,residual,gate,segments,comm,input_act=None):
    """Return None before any collective when fusion cannot preserve the route."""
    policy=getattr(module,'operator_policy',None)
    if (policy is None or not getattr(policy,'dit_fused_row_gate',False)
            or policy.row_linear_backend!='tilelang_scaled_fp16'
            or getattr(module,'lora_adapter',None) is not None or not supported(residual)
            or not _table(gate,residual) or value.ndim!=2 or value.shape[0]!=residual.shape[0]
            or value.dtype!=torch.float16 or value.device!=residual.device or not value.is_contiguous()
            or module.world_size!=2 or not module.convrot or module.convrot_groupsize!=256):return None
    width=3584 if input_act is None else 7168
    if (input_act not in (None,'swiglu') or value.shape[1]!=width*(2 if input_act else 1)
            or module.weight.shape!=(5376,width) or module.weight.dtype!=torch.int8
            or module.weight.device!=value.device or not module.weight.is_contiguous()
            or module.weight_scale.shape!=(5376,1) or module.weight_scale.dtype!=torch.float32
            or module.weight_scale.device!=value.device or not module.weight_scale.is_contiguous()):return None
    if input_act and policy.fc2_quantizer not in ('fused_stored','fused_recompute'):return None
    labels=labels_for(residual,segments)
    if labels is None:return None
    gates=gate.contiguous()
    def finish(payload,scale,weights,local_k):
        return finish_row(payload,scale,weights,residual,gates,labels,local_k)
    if input_act:
        from patches.tilelang_swiglu_quant import row_linear
        return row_linear(value,module.weight,module.weight_scale,comm,
            mode=policy.fc2_quantizer,scaled_transport=True,finish_fn=finish)
    from patches.tilelang_transport_p2 import row_linear
    return row_linear(value.float(),module.weight,module.weight_scale,comm,2,256,finish_fn=finish)
