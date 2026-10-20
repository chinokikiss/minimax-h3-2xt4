"""Opt-in SM75 scaled FP16 transport + native no-bias FP32 dequantization.

The immutable codecs specify INT32 reconstruction before two FP32 products.
Current TP2 O/FC2 sites have vector weight scales and no bias. INT32 SUM
remains the default; these entry points must be selected explicitly.
"""
from functools import lru_cache
import logging
import torch
from patches import tilelang_transport_tradeoffs as old


@lru_cache(maxsize=32)
def epilogue(rows,width,scheme,shift=0):
    import tilelang
    import tilelang.language as T
    if scheme not in ('fp16','int8_gather') or width%128:raise ValueError('Whole group128 outputs required')
    packed=scheme=='int8_gather';groups=width//128;stride=132
    shape=(2*rows,groups*stride) if packed else (rows,width)
    dtype='uint8' if packed else 'float16'
    @T.prim_func
    def main(P:T.Tensor(shape,dtype),A:T.Tensor((rows,1),'float32'),W:T.Tensor((width,),'float32'),Y:T.Tensor((rows,width),'float32')):
        with T.Kernel(T.ceildiv(rows*width,1024),threads=128) as bx:
            for i in T.Parallel(1024):
                index=bx*1024+i
                if index<rows*width:
                    rx=index//width;col=index%width
                    if packed:
                        gx=col//128;offset=col%128
                        a=T.alloc_local((1,),'uint32');b=T.alloc_local((1,),'uint32')
                        a[0]=T.uint32(0);b[0]=T.uint32(0)
                        for j in T.serial(4):
                            a[0]=a[0] | (T.cast(P[rx,gx*stride+128+j],'uint32') << (j*8))
                            b[0]=b[0] | (T.cast(P[rows+rx,gx*stride+128+j],'uint32') << (j*8))
                        q0=T.cast(T.cast(P[rx,gx*stride+offset],'int8'),'float32')
                        q1=T.cast(T.cast(P[rows+rx,gx*stride+offset],'int8'),'float32')
                        x0=T.call_pure_extern('float32','__fmul_rn',q0,T.reinterpret('float32',a[0]))
                        x1=T.call_pure_extern('float32','__fmul_rn',q1,T.reinterpret('float32',b[0]))
                        total=T.call_pure_extern('float32','__fadd_rn',x0,x1)
                        restored=T.call_pure_extern('int32','__float2int_rn',total)
                    else:
                        total=T.call_pure_extern('float32','__fmul_rn',T.cast(P[rx,col],'float32'),T.float32(2.**shift))
                        restored=T.cast(total,'int32')
                    scaled=T.call_pure_extern('float32','__fmul_rn',T.cast(restored,'float32'),A[rx,0])
                    Y[rx,col]=T.call_pure_extern('float32','__fmul_rn',scaled,W[col])
    return tilelang.compile(main,out_idx=[3],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


def encode(partial,scheme,local_k):
    rows,width=partial.shape
    if scheme=='fp16':return old.fp16_codec(rows,width,old.shift_for(local_k))(partial)
    if scheme=='int8_gather':return old.pack_int8(rows,width)(partial)
    raise ValueError(scheme)


def collect(payload,scheme,comm):
    if scheme=='fp16':comm.all_reduce(payload,op=comm.ReduceOp.SUM);return payload
    if scheme=='int8_gather':
        gathered=torch.empty((payload.shape[0]*2,payload.shape[1]),device=payload.device,dtype=torch.uint8)
        comm.all_gather_into_tensor(gathered,payload);return gathered
    raise ValueError(scheme)


def finish(payload,activation_scale,weight_scale,scheme,local_k):
    if torch.cuda.get_device_capability(payload.device)!=(7,5):raise ValueError('Physical SM75 required')
    return epilogue(activation_scale.shape[0],weight_scale.numel(),scheme,old.shift_for(local_k))(
        payload,activation_scale,weight_scale.reshape(-1))


@lru_cache(maxsize=16)
def _native_fallback(reason):
    logging.warning('H3 scaled FP16 transport unavailable (%s); using native INT32 SUM',reason)


def _supported(x,weight,weight_scale,world_size,groupsize,input_act):
    width=x.shape[-1]//(2 if input_act=='swiglu' else 1)
    # These guards apply before quantization or any collective, on both ranks.
    # The fused epilogue contract is deliberately limited to current TP2 sites.
    if (world_size!=2 or x.dtype!=torch.float32 or not x.is_cuda
            or weight.ndim!=2 or weight.dtype!=torch.int8 or weight.shape[1]!=width
            or weight.shape[0]%128 or weight_scale.numel()!=weight.shape[0]
            or weight_scale.dtype!=torch.float32 or weight.device!=x.device
            or weight_scale.device!=x.device or not weight_scale.is_contiguous()
            or groupsize<=0 or width%groupsize):
        _native_fallback('requires TP2 FP32, aligned local groups and vector FP32 weight scales')
        return False
    if torch.cuda.get_device_capability(x.device)!=(7,5):
        _native_fallback('physical SM75 required');return False
    return True


def _finish_row(q,scale,shape,weight,weight_scale,comm):
    from patches.parallel_int8 import int_mm
    from patches.tilelang_row_quant import label
    local_k=q.shape[-1]
    with label('int8_gemm'):partial=int_mm(q,weight)
    del q
    with label('scaled_fp16_encode'):payload=encode(partial,'fp16',local_k)
    del partial
    with label('scaled_fp16_sum'):payload=collect(payload,'fp16',comm)
    with label('scaled_fp16_fused_epilogue'):out=finish(payload,scale,weight_scale,'fp16',local_k)
    return out.reshape(*shape[:-1],weight.shape[0])


def row_linear(x,weight,weight_scale,comm,world_size,groupsize=256,input_act=None,*,swiglu_padding=True):
    """Keep packed quantization/GEMM, replacing only SUM transport and epilogue."""
    from patches import tilelang_row_quant as row,tilelang_swiglu_pad as swiglu,optimized_tp,parallel_int8 as original
    prepare=swiglu if swiglu_padding else row
    if not _supported(x,weight,weight_scale,world_size,groupsize,input_act):
        return prepare.row_linear(x,weight,weight_scale,comm,world_size,groupsize,input_act,variant='packed')
    width=x.shape[-1]//(2 if input_act=='swiglu' else 1)
    if optimized_tp.use_fallback_rotation(x,width*world_size,groupsize):
        q,scale=prepare.global_row_quantize(x,comm,world_size,groupsize,input_act,variant='packed')
    else:
        # Preserve the Native fused ConvRot quantizer, just as tilelang_packed
        # does on this route. Only the subsequent SUM/epilogue is replaced.
        full,start,length=original.gather(x,comm,-1)
        if input_act=='swiglu':
            pieces=full.chunk(world_size,-1)
            full=torch.cat([p.chunk(2,-1)[0] for p in pieces]+[p.chunk(2,-1)[1] for p in pieces],-1)
            start//=2;length//=2
        quantized,scale=original.quantize(full,groupsize,input_act=input_act)
        q=quantized[:,start:start+length].contiguous()
        del full,quantized
    return _finish_row(q,scale,x.shape,weight,weight_scale,comm)


def row_linear_parts(gate,up,weight,weight_scale,comm,world_size,groupsize=256):
    """TE separate gate/up path, retaining fused SwiGLU + owned padding."""
    from patches import tilelang_swiglu_pad as swiglu,optimized_tp
    if gate.shape!=up.shape or gate.dtype!=up.dtype or gate.device!=up.device:
        raise ValueError('Matching gate/up required')
    if not _supported(gate,weight,weight_scale,world_size,groupsize,None):
        return swiglu.row_linear_parts(gate,up,weight,weight_scale,comm,world_size,groupsize)
    if not optimized_tp.use_fallback_rotation(gate,gate.shape[-1]*world_size,groupsize):
        return row_linear(torch.cat((gate,up),-1),weight,weight_scale,comm,world_size,groupsize,'swiglu')
    padded=swiglu.swiglu_pad(gate,up,rank=comm.get_rank(),world_size=world_size)
    q,scale=swiglu._quantize_padded(padded,gate.shape[-1],comm,world_size,groupsize)
    del padded
    return _finish_row(q,scale,gate.shape,weight,weight_scale,comm)
