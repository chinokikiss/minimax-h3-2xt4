"""Trained local slab ablations; no physical multi-GPU performance claim."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import MethodType

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'benchmarks',ROOT/'ComfyUI',ROOT/'.retest-deps'):
    sys.path.insert(0,str(path))
import torch
import comfy.ops
import comfy.quant_ops
from comfy.ldm.minimax.vae import create_token_ids
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from patches.parallel_int8 import gather
from patches.slab_video_vae import SlabPlan,stitch_slabs
from retest import load_video_native


def metrics(a,b):
    a=a.detach().float();b=b.detach().to(a.device).float()
    assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    d=a-b
    return dict(bit_exact=torch.equal(a,b),rel_l2=(d.norm()/b.norm().clamp_min(1e-20)).item(),
                mae=d.abs().mean().item(),max_abs=d.abs().max().item())


def shared_attention(self,x,rotary_pos_emb,pre_norm,residual,residual_scale):
    state=self._slab_diagnostic;batch,sequence,_=x.shape
    with comfy.ops.CastBiasWeightContext(pre_norm,x,offloadable=True) as (weight,bias):
        qkv=comfy.ops.linear_input_act(self.to_qkv,x,'rms_norm',weight,pre_norm.eps)
    q,k,v=qkv.view(batch,sequence,-1,3*self.dim_head).chunk(3,-1)
    q,k=comfy.quant_ops.ck.rms_rope_split_half_(q,k,rotary_pos_emb,
        self.qk_norm_scale.to(q.device),epsilon=self.norm_q.eps,
        rot_dim=rotary_pos_emb.shape[-3]*2)
    q,k,v=(a.transpose(1,2) for a in (q,k,v))
    owned=state['owned']
    full_k,_,_=gather(k.index_select(2,owned),state['comm'],2)
    full_v,_,_=gather(v.index_select(2,owned),state['comm'],2)
    full_k=full_k.index_select(2,state['order'])
    full_v=full_v.index_select(2,state['order'])
    if self is state['first_attention']:
        state['first_q']=metrics(q,state['reference_q'].index_select(2,state['ids'].cpu()))
        state['first_global_k']=metrics(full_k,state['reference_k'])
        state['first_global_v']=metrics(full_v,state['reference_v'])
    if state['full_query_control'] or state['native_query_block_control']:
        full_q,_,_=gather(q.index_select(2,owned),state['comm'],2)
        full_q=full_q.index_select(2,state['order'])
        if state['native_query_block_control']:
            packed=comfy.quant_ops.ck.prequantize_int8_attention(full_q,full_k,full_v)
            length=full_q.shape[2];cut=((length+255)//256)*128
            lo,hi=(0,cut) if state['comm'].get_rank()==0 else (cut,length)
            local=replace(packed,q=packed.q[:,:,lo:hi].contiguous(),
                q_scale=packed.q_scale[:,:,lo//4:((hi+127)//128)*32].contiguous())
            partial=comfy.quant_ops.ck.int8_attention_from_prequantized(local)
            full,_,_=gather(partial,state['comm'],2)
            out=full.index_select(2,state['ids'])
            if self is state['first_attention']:
                state['first_preserved_global_q_pack']=metrics(packed.q,state['reference_packed'].q)
                state['first_preserved_global_q_scales']=metrics(packed.q_scale,state['reference_packed'].q_scale)
                state['first_query_execution_range']=[lo,hi]
        else:
            out=comfy.quant_ops.ck.int8_attention(full_q,full_k,full_v).index_select(2,state['ids'])
    else:
        out=comfy.quant_ops.ck.int8_attention(q,full_k,full_v)
    if self is state['first_attention']:
        state['first_attention_output']=metrics(out,state['reference_attention'].index_select(2,state['ids'].cpu()))
        local_packed=comfy.quant_ops.ck.prequantize_int8_attention(q,full_k,full_v)
        native_packed=state['reference_packed']
        state['first_quantized_q']=metrics(local_packed.q,native_packed.q.index_select(2,state['ids']))
        state['first_quantized_k']=metrics(local_packed.k,native_packed.k)
        state['first_k_scale']=metrics(local_packed.k_scale,native_packed.k_scale)
        state['first_quantized_v']=metrics(local_packed.v,native_packed.v)
        state['first_v_scale']=metrics(local_packed.v_scale,native_packed.v_scale)
    out=out.transpose(1,2).reshape(batch,sequence,-1)
    return comfy.ops.linear_input_act(self.to_out,torch.nan_to_num(out),None,
                                     residual=residual,residual_scale=residual_scale)


def aligned_query_decode(models,z,comm,global_ids,references,capture):
    """Candidate: only K/V exchange per block; no full-Q/output block gathers."""
    decoder=models[0].decoder
    x=models[0].post_quant_conv(z)
    h=decoder.x_embedder(x.flatten(2).transpose(1,2))
    patches=h.shape[1];batch=h.shape[0]
    h=torch.cat([h,comfy.ops.cast_to_input(decoder.register_tokens,h).expand(batch,-1,-1),
                 torch.zeros_like(h[:,:1])],1)
    rope=decoder.pos_embed(global_ids);length=h.shape[1];cut=((length+255)//256)*128
    ranges=[(0,cut),(cut,length)];states=[];restores=[];hooks=[]
    for rank,model in enumerate(models):
        lo,hi=ranges[rank];ids=torch.arange(lo,hi,device=z.device)
        state=dict(comm=comm,owned=torch.arange(hi-lo,device=z.device),
            order=torch.arange(length,device=z.device),ids=ids,reference_q=capture['q'],
            reference_k=capture['k'],reference_v=capture['v'],reference_packed=capture['packed'],
            reference_attention=capture['attention'],full_query_control=False,native_query_block_control=False,
            first_attention=model.decoder.transformer_blocks[0].attn,layer_errors={})
        states.append(state)
        for layer,block in enumerate(model.decoder.transformer_blocks):
            restores.append((block.attn,block.attn.forward));block.attn._slab_diagnostic=state
            block.attn.forward=MethodType(shared_attention,block.attn)
            if layer in references:
                def compare(module,args,out,index=layer,s=state):
                    s['layer_errors'][str(index+1)]=metrics(out,references[index].index_select(1,s['ids'].cpu()))
                hooks.append(block.register_forward_hook(compare))
    def worker(rank):
        lo,hi=ranges[rank];local=h[:,lo:hi].contiguous();decoder=models[rank].decoder
        for block in decoder.transformer_blocks:local=block(local,rope[:,lo:hi].contiguous())
        return decoder.proj_out(decoder.norm_out(local))
    try:projected=comm.run([lambda r=r:worker(r) for r in range(2)])
    finally:
        for module,original in restores:
            module.forward=original;del module._slab_diagnostic
        for hook in hooks:hook.remove()
    out=torch.cat(projected,1)[:,:patches]
    b,_,t,height,width=z.shape
    out=out.view(b,t,height,width,decoder.out_channels,decoder.patch_size_t,decoder.patch_size,decoder.patch_size)
    out=out.permute(0,4,1,5,2,6,3,7).contiguous().reshape(
        b,decoder.out_channels,t*decoder.patch_size_t,height*decoder.patch_size,width*decoder.patch_size)
    evidence=[{k:v for k,v in s.items() if k=='layer_errors' or k.startswith('first_') and k!='first_attention'} for s in states]
    return out,evidence,ranges


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=Path('D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors'))
    parser.add_argument('--output',type=Path,default=ROOT/'slab_results/kv_diagnosis.json')
    options=parser.parse_args();start=time.monotonic()
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    isolate_cublas_workspaces()
    with options.checkpoint.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    assert sha=='52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e'
    models=[]
    for rank in range(2):
        model,sd=load_video_native(options.checkpoint,'cuda:0');del sd;models.append(model)
    comm=LogicalDualGPU(timeout=120)
    result=dict(scope='5060 Ti, two logical CUDA ranks/streams, trained 36 blocks; semantic ablations only',
        physical_gpu_count=1,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        comfy_kitchen=importlib.metadata.version('comfy-kitchen'),checkpoint_sha256=sha,
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__),ROOT/'ComfyUI/comfy/ldm/minimax/vae.py',ROOT/'patches/slab_video_vae.py')},
        cases=[])
    options.output.parent.mkdir(parents=True,exist_ok=True)
    def save():options.output.write_text(json.dumps(result,indent=2))
    saved_manifest=json.loads((ROOT/'slab_results/saved_references/manifest.json').read_text())
    selected_layers=(0,1,2,5,11,23,35)
    for workload in ('768x768_1F','768x1344_2F'):
        path=ROOT/'slab_results/saved_references'/('input_'+workload+'.pt')
        cpu=torch.load(path,weights_only=True)
        expected=next(s for s in saved_manifest if s['file']==path.name)
        assert hashlib.sha256(cpu.numpy().tobytes()).hexdigest()==expected['tensor_sha256']
        z=cpu.cuda();batch,_,frames,height,width=z.shape;patches=frames*height*width
        references={};handles=[]
        for layer in selected_layers:
            def record(module,args,output,index=layer):references[index]=output.detach().cpu()
            handles.append(models[0].decoder.transformer_blocks[layer].register_forward_hook(record))
        native_attention=comfy.quant_ops.ck.int8_attention;capture={}
        def record_attention(q,k,v,*args,**kwargs):
            first=not capture
            if first:
                capture.update(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu())
                capture['packed']=comfy.quant_ops.ck.prequantize_int8_attention(q,k,v)
            out=native_attention(q,k,v,*args,**kwargs)
            if first:
                capture['attention']=out.detach().cpu()
                capture['packed_control']=metrics(comfy.quant_ops.ck.int8_attention_from_prequantized(capture['packed']),out)
            return out
        comfy.quant_ops.ck.int8_attention=record_attention
        try:
            with torch.inference_mode():native=models[0]._decode_pixels(z)
        finally:
            comfy.quant_ops.ck.int8_attention=native_attention
            for handle in handles:handle.remove()
        grid=torch.arange(patches,device=z.device).view(frames,height,width)
        suffix=torch.arange(patches,patches+5,device=z.device)
        global_ids=create_token_ids((frames,height,width),z.device,z.dtype).expand(batch,-1,-1)
        global_ids=torch.cat([global_ids,torch.zeros((batch,5,3),device=z.device,dtype=z.dtype)],1)
        for overlap in (64,128):
            plan=SlabPlan.create(width,overlap)
            slabs=plan.slices(z)
            ids=[torch.cat([grid[...,:plan.left_end].flatten(),suffix]),
                 torch.cat([grid[...,plan.right_start:].flatten(),suffix])]
            # Ownership is disjoint even though both ranks query the overlap.
            owners=[]
            for rank,indices in enumerate(ids):
                patch=(indices<patches)
                choose=patch & ((indices%width<width//2) if rank==0 else (indices%width>=width//2))
                if rank==0:choose|=~patch
                owners.append(torch.where(choose)[0])
            owned_ids=torch.cat([ids[r].index_select(0,owners[r]) for r in range(2)])
            assert torch.equal(owned_ids.sort().values,torch.arange(patches+5,device=z.device))
            order=owned_ids.argsort()
            for variant,positions,sharing in [('independent',False,False),('global_positions',True,False),
                                               ('shared_kv_local_positions',False,True),('shared_kv_global_positions',True,True),
                                               ('full_query_control',True,True),('native_query_block_control',True,True)]:
                states=[];restores=[];hooks=[]
                for rank,model in enumerate(models):
                    state=dict(comm=comm,owned=owners[rank],order=order,ids=ids[rank],
                        reference_q=capture['q'],reference_k=capture['k'],reference_v=capture['v'],
                        reference_attention=capture['attention'],reference_packed=capture['packed'],
                        full_query_control=variant=='full_query_control',
                        native_query_block_control=variant=='native_query_block_control',
                        first_attention=model.decoder.transformer_blocks[0].attn,layer_errors={})
                    states.append(state)
                    if positions:
                        position=model.decoder.pos_embed;old=position.forward
                        def global_position(unused,original=old,indices=ids[rank]):
                            return original(global_ids.index_select(1,indices))
                        position.forward=global_position;restores.append((position,old))
                    for layer,block in enumerate(model.decoder.transformer_blocks):
                        if sharing:
                            old=block.attn.forward;block.attn._slab_diagnostic=state
                            block.attn.forward=MethodType(shared_attention,block.attn)
                            restores.append((block.attn,old))
                        if layer in selected_layers:
                            def compare(module,args,out,index=layer,s=state):
                                s['layer_errors'][str(index+1)]=metrics(out,references[index].index_select(1,s['ids'].cpu()))
                            hooks.append(block.register_forward_hook(compare))
                try:
                    outputs=comm.run([lambda r=r:models[r]._decode_pixels(slabs[r].contiguous()) for r in range(2)])
                    decoded=stitch_slabs(*outputs,plan)
                    quality=metrics(decoded,native)
                    row=dict(workload=workload,shape=list(z.shape),overlap_pixels=overlap,variant=variant,
                        global_coordinates=positions,shared_kv_every_layer=sharing,vs_native=quality,
                        full_decoder_1_percent_gate=quality['rel_l2']<=.01,
                        key_token_count=patches+5 if sharing else [len(i) for i in ids],
                        suffix_policy='four registers plus one zero; shared KV owns each once on rank zero',
                        native_split_quantize_control=capture['packed_control'],
                        control_duplicates_full_attention=variant=='full_query_control',
                        control_gathers_full_q_and_attention_output=variant=='native_query_block_control',
                        rank_evidence=[{k:v for k,v in s.items() if k=='layer_errors' or k.startswith('first_') and k!='first_attention'}
                                       for s in states],
                        ideal_fp16_kv_exchange_bytes_all_36_layers=36*2*(patches+5)*2048*2 if sharing else 0)
                    result['cases'].append(row);save()
                    print('RESULT',workload,overlap,variant,'rel_l2',round(quality['rel_l2'],8),flush=True)
                finally:
                    for module,original in reversed(restores):module.forward=original
                    for hook in hooks:hook.remove()
                    for model in models:
                        for block in model.decoder.transformer_blocks:
                            if hasattr(block.attn,'_slab_diagnostic'):del block.attn._slab_diagnostic
                del outputs,decoded,states
        with torch.inference_mode():candidate,rank_evidence,ranges=aligned_query_decode(models,z,comm,global_ids,references,capture)
        quality=metrics(candidate,native)
        row=dict(workload=workload,shape=list(z.shape),variant='aligned_query_candidate',vs_native=quality,
            full_decoder_1_percent_gate=quality['rel_l2']<=.01,global_coordinates=True,
            shared_kv_every_layer=True,query_ranges=ranges,overlap_pixels=0,
            key_token_count=patches+5,rank_evidence=rank_evidence,
            full_q_gathers_per_layer=0,full_attention_output_gathers_per_layer=0,
            prefix_scope='Native full input projection once, then disjoint owned hidden tokens',
            suffix_policy='Native four registers and one zero owned once by rank one',
            output_scope='Join local projected patch outputs once; no overlap fade',
            ideal_fp16_kv_exchange_bytes_all_36_layers=36*2*(patches+5)*2048*2)
        result['cases'].append(row);save()
        print('RESULT',workload,'aligned_query_candidate','rel_l2',quality['rel_l2'],'bit_exact',quality['bit_exact'],flush=True)
        del z,native,references,capture,candidate
    result.update(completed=True,wall_seconds=time.monotonic()-start,
                  limits=['Random saved benchmark latents do not establish visual acceptability.',
                          'Logical transport has synchronization and no physical PCIe/NCCL cost.',
                          'Shared floating KV is quantized in native global order; local query quantization can still change grouping.'])
    save();print('COMPLETE',options.output,flush=True)


if __name__=='__main__':main()
