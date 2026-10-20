"""Opt-in approximate TP2 transport; preserves INT32 interface after reconstruction.

FP16 uses a fixed worst-case power-of-two scale, avoiding extra MAX collectives.
INT8 sends independently scaled 128-output groups in one packed all-gather.
Both ranks reconstruct rank0 + rank1 in the same order, including local quantization.
"""
from functools import lru_cache
from contextlib import contextmanager
import math
import torch


@contextmanager
def label(name):
    torch.cuda.nvtx.range_push('linear/'+name)
    try:yield
    finally:torch.cuda.nvtx.range_pop()


def shift_for(local_k):
    # Bound includes both rank contributions and headroom below FP16 max.
    return max(0,math.ceil(math.log2(2*local_k*127*127/60000.)))


@lru_cache(maxsize=32)
def fp16_codec(rows,width,shift,decode=False):
    import tilelang
    import tilelang.language as T
    src='float16' if decode else 'int32';dst='int32' if decode else 'float16'
    factor=2.**(shift if decode else -shift)
    @T.prim_func
    def main(X:T.Tensor((rows,width),src),Y:T.Tensor((rows,width),dst)):
        with T.Kernel(T.ceildiv(rows*width,1024),threads=128) as bx:
            for i in T.Parallel(1024):
                index=bx*1024+i
                if index<rows*width:
                    z=T.call_pure_extern('float32','__fmul_rn',T.cast(X[index//width,index%width],'float32'),T.float32(factor))
                    Y[index//width,index%width]=T.cast(z,dst)
    return tilelang.compile(main,out_idx=[1],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=32)
def pack_int8(rows,width,group=128):
    import tilelang
    import tilelang.language as T
    if width%group:raise ValueError('Whole output groups required')
    groups=width//group;stride=group+4
    @T.prim_func
    def main(X:T.Tensor((rows,width),'int32'),Packed:T.Tensor((rows,groups*stride),'uint8')):
        with T.Kernel(groups,rows,threads=128) as (gx,rx):
            value=T.alloc_fragment((1,group),'float32');absolute=T.alloc_fragment((1,group),'float32')
            maximum=T.alloc_fragment((1,),'float32');scale=T.alloc_local((1,),'float32')
            for i in T.Parallel(group):
                value[0,i]=T.cast(X[rx,gx*group+i],'float32');absolute[0,i]=T.abs(value[0,i])
            T.reduce_max(absolute,maximum,dim=1)
            scale[0]=T.max(maximum[0]*T.float32(1./127.),T.float32(1e-8))
            for i in T.Parallel(group):
                z=T.call_pure_extern('float32','__fdiv_rn',value[0,i],scale[0])
                q=T.call_pure_extern('int32','__float2int_rn',z)
                Packed[rx,gx*stride+i]=T.cast(T.max(-127,T.min(127,q)),'uint8')
            if T.get_thread_binding()==0:
                bits=T.reinterpret('uint32',scale[0])
                for j in T.serial(4):Packed[rx,gx*stride+group+j]=T.cast((bits >> (j*8)) & T.uint32(255),'uint8')
    return tilelang.compile(main,out_idx=[1],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


@lru_cache(maxsize=32)
def unpack_int8(rows,width,group=128):
    import tilelang
    import tilelang.language as T
    groups=width//group;stride=group+4
    @T.prim_func
    def main(P:T.Tensor((2*rows,groups*stride),'uint8'),Y:T.Tensor((rows,width),'int32')):
        with T.Kernel(groups,rows,threads=128) as (gx,rx):
            a=T.alloc_local((1,),'uint32');b=T.alloc_local((1,),'uint32')
            a[0]=T.uint32(0);b[0]=T.uint32(0)
            for j in T.serial(4):
                a[0]=a[0] | (T.cast(P[rx,gx*stride+group+j],'uint32') << (j*8))
                b[0]=b[0] | (T.cast(P[rows+rx,gx*stride+group+j],'uint32') << (j*8))
            s0=T.reinterpret('float32',a[0]);s1=T.reinterpret('float32',b[0])
            for i in T.Parallel(group):
                q0=T.cast(T.cast(P[rx,gx*stride+i],'int8'),'float32')
                q1=T.cast(T.cast(P[rows+rx,gx*stride+i],'int8'),'float32')
                x0=T.call_pure_extern('float32','__fmul_rn',q0,s0)
                x1=T.call_pure_extern('float32','__fmul_rn',q1,s1)
                z=T.call_pure_extern('float32','__fadd_rn',x0,x1)
                Y[rx,gx*group+i]=T.call_pure_extern('int32','__float2int_rn',z)
    return tilelang.compile(main,out_idx=[1],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


def reduce(partial,comm,scheme,local_k):
    if partial.dtype!=torch.int32 or not partial.is_contiguous() or comm.get_world_size()!=2:raise ValueError('Contiguous TP2 INT32 partial required')
    if torch.cuda.get_device_capability(partial.device)!=(7,5):raise ValueError('Physical SM75 required')
    rows,width=partial.shape
    if scheme=='int32':comm.all_reduce(partial,op=comm.ReduceOp.SUM);return partial
    if scheme=='fp16':
        shift=shift_for(local_k)
        with label('encode_fp16'):payload=fp16_codec(rows,width,shift)(partial)
        comm.all_reduce(payload,op=comm.ReduceOp.SUM)
        with label('decode_fp16'):return fp16_codec(rows,width,shift,True)(payload)
    if scheme=='int8_gather':
        with label('pack_INT8_with_scales'):payload=pack_int8(rows,width)(partial)
        gathered=torch.empty((rows*2,payload.shape[1]),device=partial.device,dtype=torch.uint8)
        comm.all_gather_into_tensor(gathered,payload)
        with label('reconstruct_INT8_rank_order'):return unpack_int8(rows,width)(gathered)
    raise ValueError('Unknown transport scheme')
