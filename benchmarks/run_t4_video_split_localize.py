"""Trained 261-token physical query split: first differing stage and M-only replays."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
from run_h3_tensor_flow import compare


def difference(a,b):
    a=a.detach().cpu();b=b.detach().cpu()
    return dict(byte_exact=torch.equal(a,b),different=int((a!=b).sum()),elements=a.numel(),**compare(a,b))


def replay(block,full,rope,lo,hi):
    import comfy.ops,comfy_kitchen as ck
    records={};local=full[:,lo:hi].contiguous();count=hi-lo
    def norm_linear(module,norm,value):
        with comfy.ops.CastBiasWeightContext(norm,value,offloadable=True) as (weight,bias):
            return comfy.ops.linear_input_act(module,value,'rms_norm',weight,norm.eps)
    def pad(value):
        return torch.cat((value,torch.zeros((1,128-count,value.shape[-1]),device=value.device,dtype=value.dtype)),1) if count<128 else value
    qkv=norm_linear(block.attn.to_qkv,block.norm1,full)
    small=norm_linear(block.attn.to_qkv,block.norm1,local)
    records['qkv_norm_projection']=difference(small,qkv[:,lo:hi])
    if count<128:records['qkv_norm_projection_padded128']=difference(norm_linear(block.attn.to_qkv,block.norm1,pad(local))[:,:count],qkv[:,lo:hi])
    def qkv_rope(value,table):
        q,k,v=value.view(1,value.shape[1],-1,3*block.attn.dim_head).chunk(3,-1)
        ck.rms_rope_split_half_(q,k,table,block.attn.qk_norm_scale,epsilon=block.attn.norm_q.eps,rot_dim=table.shape[-3]*2)
        return tuple(v.transpose(1,2) for v in (q,k,v))
    q,k,v=qkv_rope(qkv,rope);qs,ks,vs=qkv_rope(small,rope[:,lo:hi].contiguous())
    for name,a,b in [('q_rope',qs,q[:,:,lo:hi]),('k_rope',ks,k[:,:,lo:hi]),('v_projection',vs,v[:,:,lo:hi])]:records[name]=difference(a,b)
    attention=ck.int8_attention(q,k,v)
    same_inputs=ck.int8_attention(q[:,:,lo:hi].contiguous(),k,v)
    records['attention_query_shape_identical_qkv']=difference(same_inputs,attention[:,:,lo:hi])
    attention=attention.transpose(1,2).reshape(1,full.shape[1],-1)
    out=lambda a,res:comfy.ops.linear_input_act(block.attn.to_out,a,None,residual=res,residual_scale=block._residual_scale(block.scale1,res))
    projected=out(attention,full)
    records['attention_out_projection_identical_input']=difference(out(attention[:,lo:hi].contiguous(),local),projected[:,lo:hi])
    if count<128:records['attention_out_projection_padded128']=difference(out(pad(attention[:,lo:hi].contiguous()),pad(local))[:,:count],projected[:,lo:hi])
    packed=norm_linear(block.ff.w1,block.norm2,full)
    records['ff_w1_norm_projection']=difference(norm_linear(block.ff.w1,block.norm2,local),packed[:,lo:hi])
    if count<128:records['ff_w1_norm_projection_padded128']=difference(norm_linear(block.ff.w1,block.norm2,pad(local))[:,:count],packed[:,lo:hi])
    ff=lambda a,res:comfy.ops.linear_input_act(block.ff.w2,a,'swiglu',residual=res,residual_scale=block._residual_scale(block.scale2,res))
    target=ff(packed,full)
    records['ff_w2_swiglu_identical_input']=difference(ff(packed[:,lo:hi].contiguous(),local),target[:,lo:hi])
    if count<128:records['ff_w2_swiglu_padded128']=difference(ff(pad(packed[:,lo:hi].contiguous()),pad(local))[:,:count],target[:,lo:hi])
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();sys.argv=sys.argv[:1]
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    assert torch.cuda.device_count()==2 and torch.cuda.get_device_capability()==(7,5)
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(ROOT/'ComfyUI')
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    from patches.parallel_vae import checkpoint_plan,load_decode_weights,AlignedVideoDecode,query_partitioning_status
    from comfy.ldm.minimax.vae import LATENTS_MEAN,LATENTS_STD,IMAGENET_MEAN,IMAGENET_STD
    reader,names,_=checkpoint_plan(args.assets/'video_vae.safetensors','video')
    model=load_decode_weights(reader,names,'video',torch.device('cuda',rank))
    z=torch.randn((1,24,1,16,16),generator=torch.Generator().manual_seed(732)).half().cuda()
    z=z*torch.tensor(LATENTS_STD,device='cuda',dtype=z.dtype).view(1,-1,1,1,1)+torch.tensor(LATENTS_MEAN,device='cuda',dtype=z.dtype).view(1,-1,1,1,1)
    runner=AlignedVideoDecode(model,z,dist,rank);lo,hi=runner.ranges[rank]
    report=dict(status='running',rank=rank,ranges=runner.ranges,compatibility=query_partitioning_status(model),
        scope='Trained 36-block Video decoder, normalized input seed732/261 tokens; raw decoder tensor comparisons, no full H3 or visual quality claim',stages=[])
    output=args.output.with_name(args.output.stem+f'_rank{rank}.json');output.parent.mkdir(parents=True,exist_ok=True)
    gold={};handles=[];phase='native';first=None
    def before(name):
        def hook(module,inp,kwargs):
            if phase=='native':
                gold[name+'.input']=inp[0].detach().cpu()
                if len(inp)>1 and isinstance(inp[1],torch.Tensor):gold[name+'.rope']=inp[1].detach().cpu()
        return hook
    def after(name):
        def hook(module,inp,out):
            nonlocal first
            if phase=='native':gold[name+'.output']=out.detach().cpu()
            else:
                row=dict(stage=name,**difference(out,gold[name+'.output'][:,lo:hi]))
                report['stages'].append(row)
                if not row['byte_exact'] and first is None:first=name
        return hook
    try:
        modules={}
        for i,block in enumerate(model.decoder.transformer_blocks):
            modules[f'block{i}.attn']=block.attn;modules[f'block{i}.ff']=block.ff
        modules['norm_out']=model.decoder.norm_out;modules['proj_out']=model.decoder.proj_out
        for name,module in modules.items():
            handles.extend((module.register_forward_pre_hook(before(name),with_kwargs=True),module.register_forward_hook(after(name))))
        with torch.inference_mode():expected=model._decode_pixels(z).cpu()
        phase='parallel'
        with torch.inference_mode():actual=runner.run()
        if rank==0:
            report['raw_decoder_output']=difference(actual,expected)
            def pixels(raw):
                value=raw[:,:,-1:]*torch.tensor(IMAGENET_STD,device=raw.device,dtype=torch.float32).view(1,3,1,1,1)
                return value.add_(torch.tensor(IMAGENET_MEAN,device=raw.device,dtype=torch.float32).view(1,3,1,1,1)).clamp_(0,1)
            report['finalized_last_frame']=difference(pixels(actual),pixels(expected.cuda()))
        for handle in handles:handle.remove()
        handles=[];report['first_differing_stage']=first
        with torch.inference_mode():
            if first and first.startswith('block'):
                index=int(first.split('.')[0][5:]);block=model.decoder.transformer_blocks[index]
                # Replays hold the full Native stage input fixed, separating
                # local M geometry from accumulated upstream divergence.
                full=gold[first+'.input'].cuda()
                rope=gold[f'block{index}.attn.rope'].cuda()
                report['controlled_replay']=replay(block,full,rope,lo,hi)
            normalized=model.decoder.norm_out(gold['norm_out.input'].cuda())
            localnorm=model.decoder.norm_out(gold['norm_out.input'][:,lo:hi].contiguous().cuda())
            report['norm_out_identical_input']=difference(localnorm,normalized[:,lo:hi])
            report['proj_out_identical_input']=difference(model.decoder.proj_out(normalized[:,lo:hi].contiguous()),model.decoder.proj_out(normalized)[:,lo:hi])
        report['status']='completed_localization_not_identity_acceptance'
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        for handle in handles:handle.remove()
        output.write_text(json.dumps(report,indent=2));dist.destroy_process_group()


if __name__=='__main__':main()
