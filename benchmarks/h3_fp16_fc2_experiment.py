"""Experimental FP16 storage / FP32 preparation; never imported by production.

Regular H4 ConvRot is retained (not a Sylvester Hadamard). Butterfly reduction
order differs from Native SGEMM; fused modes are measured numerical candidates.
TP2's global maximum remains a separate collective before final INT8 rounding.
"""
from functools import lru_cache
import types
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


def prepare(x,comm,mode,*,arch='sm_75'):
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


def row_forward(module,x,comm,mode='control',transport='int32'):
    """INT8 GEMM unchanged; F32 dequant/LoRA, then one F16 output store."""
    from patches import parallel_int8 as original,tilelang_transport_p2 as codec
    q,scale=prepare(x,comm,mode)
    partial=original.int_mm(q,module.weight);del q
    if transport=='int32':
        comm.all_reduce(partial,op=comm.ReduceOp.SUM)
        out=original.dequantize(partial,scale,module.weight_scale,torch.float32)
    elif transport=='scaled_fp16':
        payload=codec.encode(partial,'fp16',module.weight.shape[1]);del partial
        payload=codec.collect(payload,'fp16',comm)
        out=codec.finish(payload,scale,module.weight_scale,'fp16',module.weight.shape[1])
    else:raise ValueError(transport)
    out=out.reshape(*x.shape[:-1],module.weight.shape[0])
    adapter=getattr(module,'lora_adapter',None)
    if adapter is not None:out=adapter.add_to(x.float(),out,comm=comm,input_act='swiglu')
    return out.to(x.dtype)


def install(module,comm,mode='control',transport='int32',capture=None):
    if module.world_size!=2 or module.convrot_groupsize!=256:raise ValueError('Current TP2 ConvRot256 only')
    def forward(self,x,*args,**kwargs):
        if kwargs.get('input_act')!='swiglu':raise ValueError('FC2 SwiGLU entry point required')
        if capture is not None:capture(x)
        return row_forward(self,x,comm,mode,transport)
    module.forward=types.MethodType(forward,module)
    if hasattr(module,'forward_parts'):
        def parts(self,gate,up):return forward(self,torch.cat((gate,up),-1),input_act='swiglu')
        module.forward_parts=types.MethodType(parts,module)
