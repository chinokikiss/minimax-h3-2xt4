"""Private prepare/consume experiment. No runtime defaults are modified."""
from contextlib import nullcontext
from dataclasses import dataclass
import torch


@dataclass
class Operand:
    codes: torch.Tensor | None
    scale: torch.Tensor | None
    shape: tuple

    def take(self):
        if self.codes is None:raise RuntimeError('Prepared operand already consumed')
        codes,scale=self.codes,self.scale
        self.codes=self.scale=None
        return codes,scale


def row_labels(rows,segments,device,modulation_rows=3):
    labels=torch.full((rows,),-1,dtype=torch.int32,device='cpu')
    for start,stop,row in segments:
        if not (0<=start<stop<=rows and type(row) is int and 0<=row<modulation_rows):
            raise ValueError('Valid bounded modulation rows required')
        if bool((labels[start:stop]!=-1).any()):raise ValueError('Overlapping modulation segments')
        labels[start:stop]=row
    if bool((labels<0).any()):raise ValueError('Modulation segments must cover the input')
    return labels.to(device)


def prepare_column(x):
    from patches.dit_column_linear import _prepare_activation
    codes,scale=_prepare_activation(x,True,256)
    return Operand(codes,scale,tuple(x.shape))


def consume_column(prepared,module,chunk_rows=0):
    from patches.optimized_tp import native_int_mm
    from patches.tilelang_column_epilogue import dequantize
    from benchmarks.dit_vram_kernels import column_store_kernel
    codes,scale=prepared.take()
    rows=codes.shape[0];width=module.weight.shape[0]
    weights=module.weight_scale.reshape(-1)
    if chunk_rows==0:
        accumulator=native_int_mm(codes,module.weight)
        del codes
        return dequantize(accumulator,scale,weights,torch.float16)
    if type(chunk_rows) is not int or chunk_rows<1:raise ValueError('Positive chunk_rows required')
    output=torch.empty((rows,width),device=codes.device,dtype=torch.float16)
    for start in range(0,rows,chunk_rows):
        stop=min(start+chunk_rows,rows)
        accumulator=native_int_mm(codes[start:stop],module.weight)
        column_store_kernel(stop-start,width,weights.numel()==1)(accumulator,scale[start:stop],weights,output[start:stop])
        del accumulator
    return output


def prepare_row(module,x,comm,input_act=None):
    if input_act=='swiglu':
        from patches.tilelang_swiglu_quant import prepare
        q,scale=prepare(x,comm,'fused_stored')
    else:
        from patches import optimized_tp,parallel_int8 as original
        from patches.tilelang_swiglu_pad import global_row_quantize
        flat=x.float()
        if optimized_tp.use_fallback_rotation(flat,x.shape[-1]*2,256):
            q,scale=global_row_quantize(flat,comm,2,256,None,variant='packed')
        else:
            full,start,length=original.gather(flat,comm,-1)
            quantized,scale=original.quantize(full,256)
            q=quantized[:,start:start+length].contiguous()
    return Operand(q,scale,tuple(x.shape))


def consume_row(prepared,module,comm,*,residual=None,gates=None,labels=None):
    from patches.parallel_int8 import int_mm
    from patches import tilelang_transport_p2 as codec
    q,scale=prepared.take();local_k=q.shape[-1]
    partial=int_mm(q,module.weight);del q
    payload=codec.encode(partial,'fp16',local_k);del partial
    payload=codec.collect(payload,'fp16',comm)
    if residual is not None:
        from benchmarks.dit_vram_kernels import row_gate
        return row_gate(payload,scale,module.weight_scale,residual,gates,labels,local_k)
    return codec.finish(payload,scale,module.weight_scale,'fp16',local_k).half()


VARIANTS={
    'baseline':{},
    'ownership':dict(ownership=True),
    'norm_gate':dict(fuse=True),
    'fc1_4k':dict(chunk=4096),
    'fc1_8k':dict(chunk=8192),
    'fc1_16k':dict(chunk=16384),
    'combined_8k':dict(ownership=True,fuse=True,chunk=8192,row_epilogue=True),
}


def forward(block,x,time_embedding,segments,freqs,labels,variant,*,observer=None):
    if variant=='baseline':
        with observer.phase('production_block') if observer else nullcontext():
            return block(x,time_embedding,segments,freqs,{})
    from patches.h3_tensor_flow import modulated_norm,gated_add
    from patches.tilelang_fp32_rms_rope import execute as rope
    from patches.h3_operator_policy import dit_attention
    from benchmarks import dit_vram_kernels as fused
    import comfy_kitchen as ck
    import torch.distributed as comm
    options=VARIANTS[variant];owns=options.get('ownership',False);fusion=options.get('fuse',False)
    chunk=options.get('chunk',0);epilogue=options.get('row_epilogue',False)
    stage=observer.phase if observer else lambda name:nullcontext()
    norm=(lambda n,x,s,c:fused.norm(n,x,s,c,labels)) if fusion else lambda n,x,s,c:modulated_norm(n,x,s,c,segments)
    add=(lambda x,g,b:fused.gate(x,g,b,labels)) if fusion else lambda x,g,b:gated_add(x,g,b,segments)
    modulation=block.adaln_proj(time_embedding).view(-1,6*block.hidden)
    sm,cm,gm,sf,cf,gf=modulation.chunk(6,-1)
    with stage('norm1'):h=norm(block.norm1,x,sm,cm)
    with stage('qkv'):
        if owns:
            p=prepare_column(h);del h
            qkv=consume_column(p,block.qkv_proj)
        else:qkv=block.qkv_proj(h);del h
    with stage('rms_rope'):
        q,k=rope(qkv,freqs,block.q_norm.weight,block.k_norm.weight,epsilon=block.q_norm.eps)
        v=qkv[:,2*block.inner_per_rank:].view(qkv.shape[0],28,128)
        qh,kh,vh=(z.transpose(0,1).unsqueeze(0) for z in (q,k,v))
    with stage('attention'):
        if owns and callable(getattr(ck,'prequantize_int8_attention',None)):
            p=ck.prequantize_int8_attention(qh,kh,vh)
            del qkv,q,k,v,qh,kh,vh
            result=ck.int8_attention_from_prequantized(p)
            del p
            attended=result.transpose(1,2).reshape(x.shape[0],3584)
            del result
        else:
            # Match the production block's [S, local_inner] row-linear input.
            # The policy returns [B,S,D]; retaining B broke the fused gate's
            # shape contract even with B=1 and otherwise valid FP16 tensors.
            attended=dit_attention(block,qh,kh,vh,{}).reshape(x.shape[0],block.inner_per_rank)
            del qkv,q,k,v,qh,kh,vh
    with stage('out_projection'):
        if owns:
            p=prepare_row(block.out_proj,attended,comm);del attended
            if epilogue:x=consume_row(p,block.out_proj,comm,residual=x,gates=gm,labels=labels)
            else:branch=consume_row(p,block.out_proj,comm)
        else:branch=block.out_proj(attended);del attended
    if not epilogue:
        with stage('gate1'):x=add(x,gm,branch);del branch
    with stage('norm2'):h2=norm(block.norm2,x,sf,cf)
    with stage('fc1'):
        if owns or chunk:
            p=prepare_column(h2)
            if owns:del h2
            packed=consume_column(p,block.fc1,chunk)
            if not owns:del h2
        else:packed=block.fc1(h2);del h2
    with stage('fc2'):
        if owns:
            p=prepare_row(block.fc2,packed,comm,'swiglu');del packed
            if epilogue:x=consume_row(p,block.fc2,comm,residual=x,gates=gf,labels=labels)
            else:branch=consume_row(p,block.fc2,comm)
        else:branch=block.fc2(packed,input_act='swiglu');del packed
    if not epilogue:
        with stage('gate2'):x=add(x,gf,branch);del branch
    return x
