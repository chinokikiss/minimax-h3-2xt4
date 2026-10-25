"""FP32 SwiGLU / regular ConvRot256 preparation qualified on physical TP2 T4.

Kernel arithmetic is promoted unchanged from the 2026-10-22 experiment.
Stored fusion retains rotated FP32 values; recompute is an optional memory path.
Neither removes the shared row-MAX collective. No benchmark imports at runtime.
"""
from functools import lru_cache
import torch


@lru_cache(maxsize=64)
def fused_kernel(rows,width,dtype,arch,phase='stored'):
    import tilelang
    import tilelang.language as T
    if arch not in ('sm_75','sm_120') or dtype not in ('float16','float32'):raise ValueError('Explicit supported target/dtype required')
    if width%256 or phase not in ('stored','maximum','recompute'):raise ValueError('Whole regular ConvRot256 groups required')
    groups=width//256;stored=phase=='stored'
    @T.macro
    def transform(X,A,B,row,group):
        for i in T.Parallel(256):
            gate=T.cast(X[row,group*256+i],'float32');up=T.cast(X[row,width+group*256+i],'float32')
            denominator=T.call_pure_extern('float32','__fadd_rn',T.float32(1.),T.call_pure_extern('float32','expf',-gate))
            activated=T.call_pure_extern('float32','__fdiv_rn',gate,denominator)
            A[i]=T.call_pure_extern('float32','__fmul_rn',activated,up)
        T.sync_threads()
        for stage in T.unroll(4):
            for i in T.Parallel(256):
                stride=1<<(2*stage);digit=(i//stride)%4;base=i-digit*stride
                a=A[base];b=A[base+stride];c=A[base+2*stride];d=A[base+3*stride]
                p=T.if_then_else(digit==3,-a,a);q=T.if_then_else(digit==2,-b,b)
                r=T.if_then_else(digit==1,-c,c);s=T.if_then_else(digit==0,-d,d)
                B[i]=T.call_pure_extern('float32','__fadd_rn',T.call_pure_extern('float32','__fadd_rn',
                    T.call_pure_extern('float32','__fadd_rn',p,q),r),s)
            T.sync_threads()
            for i in T.Parallel(256):A[i]=B[i]
            T.sync_threads()
    if phase!='recompute':
        @T.prim_func
        def main(X:T.Tensor((rows,2*width),dtype),R:T.Tensor((rows,width) if stored else (1,1),'float32'),
                 M:T.Tensor((rows,groups),'float32')):
            with T.Kernel(rows,groups,threads=256) as (row,group):
                a=T.alloc_shared((256,),'float32');b=T.alloc_shared((256,),'float32')
                absolute=T.alloc_fragment((256,),'float32');maximum=T.alloc_fragment((1,),'float32')
                transform(X,a,b,row,group)
                for i in T.Parallel(256):
                    value=a[i]*T.float32(1./16.)
                    absolute[i]=T.abs(value)
                    if stored:R[row,group*256+i]=value
                T.reduce_max(absolute,maximum,dim=0)
                if T.get_thread_binding()==0:M[row,group]=maximum[0]
        return tilelang.compile(main,out_idx=[1,2],target={'kind':'cuda','arch':arch},pass_configs={'tl.enable_async_copy':False})
    @T.prim_func
    def quant(X:T.Tensor((rows,2*width),dtype),M:T.Tensor((rows,1),'float32'),
              Q:T.Tensor((rows,width),'int8'),S:T.Tensor((rows,1),'float32')):
        with T.Kernel(rows,groups,threads=256) as (row,group):
            a=T.alloc_shared((256,),'float32');b=T.alloc_shared((256,),'float32')
            transform(X,a,b,row,group)
            scale=T.max(M[row,0]*T.float32(1./127.),T.float32(1.e-30))
            for i in T.Parallel(256):
                scaled=T.call_pure_extern('float32','__fdividef',a[i]*T.float32(1./16.),scale)
                rounded=T.call_pure_extern('float32','nearbyintf',scaled)
                Q[row,group*256+i]=T.cast(T.min(T.float32(127.),T.max(T.float32(-128.),rounded)),'int8')
            if (group==0)&(T.get_thread_binding()==0):S[row,0]=scale
    return tilelang.compile(quant,out_idx=[2,3],target={'kind':'cuda','arch':arch},pass_configs={'tl.enable_async_copy':False})


def prepare(x,comm,mode='fused_stored',*,arch='sm_75'):
    width=x.shape[-1]//2;flat=x.reshape(-1,2*width).contiguous()
    if mode=='control':
        from patches import tilelang_swiglu_pad as previous
        return previous.global_row_quantize(flat.float(),comm,2,256,'swiglu','packed','owned_fp32')
    if mode not in ('fused_stored','fused_recompute'):raise ValueError(mode)
    dtype={torch.float16:'float16',torch.float32:'float32'}[flat.dtype]
    phase='stored' if mode=='fused_stored' else 'maximum'
    rotated,groups=fused_kernel(flat.shape[0],width,dtype,arch,phase)(flat)
    maximum=groups.amax(-1,keepdim=True);del groups
    comm.all_reduce(maximum,op=comm.ReduceOp.MAX)
    if mode=='fused_recompute':
        del rotated
        return fused_kernel(flat.shape[0],width,dtype,arch,'recompute')(flat,maximum)
    from patches.tilelang_row_quant import quantize
    return quantize(rotated,maximum)


def supported(x,weight,weight_scale,world_size,groupsize):
    """Check the complete contract before either rank enters a collective."""
    width=x.shape[-1]//2
    return (x.is_cuda and x.dtype in (torch.float16,torch.float32)
        and x.numel()>0 and x.shape[-1]==2*width and width%256==0
        and world_size==2 and groupsize==256 and weight.ndim==2
        and weight.dtype==torch.int8 and weight.shape[1]==width
        and weight.shape[0]%128==0 and weight.device==x.device
        and weight_scale.dtype==torch.float32 and weight_scale.is_contiguous()
        and weight_scale.numel()==weight.shape[0] and weight_scale.device==x.device
        and torch.cuda.get_device_capability(x.device)==(7,5))


def row_linear(x,weight,weight_scale,comm,*,mode,scaled_transport):
    """Return FP32 through the caller's LoRA addition, then store only once."""
    from patches import parallel_int8 as original,tilelang_transport_p2 as codec
    from patches.tilelang_row_quant import label
    with label('fp32_swiglu_convrot_quantize'):q,scale=prepare(x,comm,mode)
    shape=x.shape
    del x
    local_k=q.shape[-1]
    with label('int8_gemm'):partial=original.int_mm(q,weight)
    del q
    if scaled_transport:
        with label('scaled_fp16_encode'):payload=codec.encode(partial,'fp16',local_k)
        del partial
        with label('scaled_fp16_sum'):payload=codec.collect(payload,'fp16',comm)
        with label('scaled_fp16_fused_epilogue'):out=codec.finish(payload,scale,weight_scale,'fp16',local_k)
    else:
        with label('int32_sum'):comm.all_reduce(partial,op=comm.ReduceOp.SUM)
        with label('dequantize'):out=original.dequantize(partial,scale,weight_scale,torch.float32)
    return out.reshape(*shape[:-1],weight.shape[0])

