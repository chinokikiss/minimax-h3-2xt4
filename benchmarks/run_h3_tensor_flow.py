"""Bounded trained-component BF16 comparisons. Never runs the H3 sampler."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import gc, hashlib, json, sys, threading
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
from torch import nn
from patches.h3_tensor_flow import TensorFlowPolicy
from patches.windowed_tp import CheckpointReader,prepare_tp_block


def stats(x):
    dtype=str(x.dtype);x=x.detach().float();finite=torch.isfinite(x);a=x.abs()
    return dict(dtype=dtype,elements=x.numel(),nonfinite=int((~finite).sum()),
        maximum=float(a.masked_fill(~finite,0).max()),above_fp16=int((finite&(a>65504)).sum()),
        subnormal_fp16=int((finite&(a>0)&(a<2**-14)).sum()),zeros=int((x==0).sum()))


def compare(x,y):
    a=x.float();b=y.float();finite=torch.isfinite(a)&torch.isfinite(b)
    if not bool(finite.all()):return dict(nonfinite=int((~finite).sum()),relative_l2=None,max_abs=None)
    delta=a.double()-b.double();den=b.double().norm().clamp_min(1e-300)
    return dict(nonfinite=0,relative_l2=float(delta.norm()/den),max_abs=float(delta.abs().max()))


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


@contextmanager
def observe(model):
    records={};handles=[]
    def hook(name):
        def record(module,args,out):
            if isinstance(out,torch.Tensor):records[name]=dict(storage_dtype=str(out.dtype),**stats(out))
        return record
    for name,module in model.named_modules():
        if name and not list(module.children()):handles.append(module.register_forward_hook(hook(name)))
    import comfy_kitchen as ck
    import patches.h3_tensor_flow as flow_module
    attention=ck.int8_attention;attention_index=0
    cast=flow_module.activation_cast;cast_index=0
    probe_lock=threading.Lock()
    def cast_probe(value,dtype):
        nonlocal cast_index
        with probe_lock:index=cast_index;cast_index+=1
        out=cast(value,dtype)
        finite=torch.isfinite(value)
        records[f'__activation_storage_cast_{index}']=dict(storage_dtype=str(out.dtype),
            **stats(out),cast_nonzero_to_zero=int((finite&(value!=0)&(out==0)).sum()),
            cast_finite_to_nonfinite=int((finite&~torch.isfinite(out)).sum()))
        return out
    def probe(*args,**kwargs):
        nonlocal attention_index
        with probe_lock:index=attention_index;attention_index+=1
        out=attention(*args,**kwargs)
        records[f'__int8_attention_before_sanitizer_{index}']=dict(storage_dtype=str(out.dtype),**stats(out))
        return out
    try:
        with patch.object(ck,'int8_attention',probe),patch.object(flow_module,'activation_cast',cast_probe):yield records
    finally:
        for handle in handles:handle.remove()


def native_block(kind, reader, index, dtype):
    import comfy.ops
    # The pinned mixed-precision Linear factory ignores its per-layer dtype
    # argument when loading dense weights. A BF16 factory would silently round
    # the explicitly FP32 AdaLN curve weights to BF16. Keep that state in FP32;
    # Native linears still compute QKV/MLP in their BF16 input/output dtype.
    ops=comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.float32 if kind=='dit' else dtype)
    if kind=='dit':
        from comfy.ldm.minimax.model import DiTBlock
        block=DiTBlock(5376,56,128,14336,8,1e-5,1e-5,apply_silu=False,
            adaln_dtype=torch.float32,device='cpu',dtype=dtype,operations=ops)
        prefix=f'blocks.{index}.'
    else:
        from comfy.text_encoders.llama import Qwen3VL_32BConfig,TransformerBlock
        block=TransformerBlock(Qwen3VL_32BConfig(),index=index,device='cpu',dtype=dtype,ops=ops)
        prefix=f'model.layers.{index}.'
    block.load_state_dict(reader.read(prefix),strict=True)
    store=getattr(reader,'lora_store',None)
    if store is not None:
        from patches.h3_lora import NativeLoraLinear,NativeLoraMLP
        for name in ('qkv_proj','out_proj'):
            adapter=store.adapter(f'blocks.{index}.attn.{name}')
            if adapter is not None:setattr(block.attn,name,NativeLoraLinear(getattr(block.attn,name),adapter))
        a=store.adapter(f'blocks.{index}.mlp.fc1');b=store.adapter(f'blocks.{index}.mlp.fc2')
        if a is not None or b is not None:block.mlp=NativeLoraMLP(block.mlp,a,b)
        adapter=store.adapter(f'blocks.{index}.adaln_proj.linear')
        if adapter is not None:block.adaln_proj.linear=NativeLoraLinear(block.adaln_proj.linear,adapter)
    return block.cuda().eval().requires_grad_(False)


def block_cases(args, report):
    from patches import tp_dit,tp_qwen3vl
    from patches.h3_operator_policy import H3OperatorPolicy
    from patches.optimized_tp import enable
    from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
    kind=args.mode;reader=CheckpointReader(args.checkpoint)
    if args.lora:
        if kind!='dit':raise ValueError('Only DiT LoRA is supported')
        from patches.h3_lora import H3LoRAStore
        reader.lora_store=H3LoRAStore(((str(args.lora),1.),),reader,selective_b=True)
    if kind=='dit':
        fixture=torch.load(args.fixture,map_location='cpu',weights_only=False)
        ids=torch.arange(fixture['input'].shape[0]) if args.tokens==0 else torch.linspace(0,fixture['input'].shape[0]-1,args.tokens).long().unique()
        x=fixture['input'][ids].cuda();t=fixture['t_emb'].cuda();rope=fixture['rope'][:,ids].cuda()
        segments=[]
        for a,b,row in fixture['segments']:
            selected=((ids>=a)&(ids<b)).nonzero().flatten()
            if selected.numel():segments.append((int(selected[0]),int(selected[-1])+1,row if isinstance(row,int) else row[ids[selected]-a].cuda()))
        report['distribution']=dict(source=str(args.fixture),sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            provenance=fixture['provenance'],shape=list(x.shape),layer=args.layer,
            caveat='Real first-block input reconstruction; other layer indices are stress tests, not their native trajectories.')
        def call(block,value):return block(value.clone(),t,segments,rope.to(value.dtype),{})
    else:
        from fixture_inputs import block_input
        from comfy.text_encoders.llama import precompute_freqs_cis,Qwen3VL_32BConfig
        from comfy.ldm.modules.attention import optimized_attention_for_device
        x=block_input('te',args.tokens or 257,20261021)['input'].float().cuda()
        config=Qwen3VL_32BConfig();positions=torch.arange(x.shape[1],device=x.device)[None]
        rope=precompute_freqs_cis(128,positions,config.rope_theta,rope_dims=config.rope_dims,device=x.device,interleaved_mrope=True)
        def call(block,value):
            mask=torch.full((value.shape[1],value.shape[1]),torch.finfo(value.dtype).min/4,device=value.device,dtype=value.dtype).triu_(1)
            if isinstance(block,tp_qwen3vl.TPTransformerBlockINT8):return block(value.clone(),mask,rope)
            return block(value.clone(),attention_mask=mask,freqs_cis=rope,
                optimized_attention=optimized_attention_for_device(value.device,mask=True,small_input=True))[0]
        report['distribution']=dict(source='Seeded synthetic TE inputs; no real TE input/trajectory claim',shape=list(x.shape),layer=args.layer)
    baseline=native_block(kind,reader,args.layer,torch.bfloat16)
    with observe(baseline) as obs:
        call(baseline,x.bfloat16())  # warm-up; compilation/loading are not timed
        reference=call(baseline,x.bfloat16()).float().cpu()
    report['bf16_reference']=dict(output=stats(reference),operators=obs)
    qkv_name='attn.qkv_proj' if kind=='dit' else 'self_attn.q_proj'
    proof=obs.get(qkv_name,obs.get(qkv_name+'.base'))
    if proof is None or proof['storage_dtype']!='torch.bfloat16':raise ValueError('Reference did not execute BF16 projections')
    if not torch.isfinite(reference).all():raise ValueError('BF16 baseline is nonfinite')
    if kind=='dit':
        report['bf16_reference']['adaln_parameter_dtype']=str(baseline.adaln_proj.linear.weight.dtype)
        from comfy.ldm.modules import attention as native_attention
        report['bf16_reference']['preferred_attention']=getattr(baseline.attn.comfy_attention.function,'__name__',None)
        report['bf16_reference']['global_attention']=native_attention.optimized_attention.__name__
        baseline.attn.comfy_attention.function=native_attention.attention_comfy_kitchen_int8
        with observe(baseline) as obs:
            kitchen_reference=call(baseline,x.bfloat16()).float().cpu()
        report['matched_kitchen_bf16_reference']=dict(output=stats(kitchen_reference),operators=obs,
            vs_native_bf16=compare(kitchen_reference,reference))
    del baseline;gc.collect();torch.cuda.empty_cache()
    reader.operator_policy=H3OperatorPolicy()
    enable('scalemax');isolate_cublas_workspaces()
    owner=tp_dit if kind=='dit' else tp_qwen3vl;comm=LogicalDualGPU(timeout=120)
    cases=[TensorFlowPolicy('bf16_reference',0),TensorFlowPolicy('fp16',0),TensorFlowPolicy('fp16',4),TensorFlowPolicy('fp16',6)]
    runtime_bf16=None
    for flow in cases:
        reader.tensor_flow=flow
        blocks=[prepare_tp_block(reader,kind,args.layer,r).cuda() for r in (0,1)]
        value=(x/flow.divisor).to(flow.dtype)
        with patch.object(owner,'dist',comm),observe(blocks[0]) as observations:
            comm.run([lambda r=r:call(blocks[r],value) for r in (0,1)])
            outputs=comm.run([lambda r=r:call(blocks[r],value) for r in (0,1)])
            actual=outputs[0].float()*flow.divisor
            repeated=comm.run([lambda r=r:call(blocks[r],value) for r in (0,1)])
        row=dict(flow=flow.mode,exponent=flow.residual_exponent,output=stats(actual),
            vs_native_bf16=compare(actual.cpu(),reference),rank_equal=torch.equal(outputs[0],outputs[1]),
            repeat_equal=torch.equal(outputs[0],repeated[0]),operators=observations)
        exact_input=x/flow.divisor
        row['input_cast']=dict(storage_dtype=str(value.dtype),cast_nonzero_to_zero=int(((exact_input!=0)&(value==0)).sum()),
            cast_finite_to_nonfinite=int((torch.isfinite(exact_input)&~torch.isfinite(value)).sum()),**stats(value))
        if kind=='dit':
            row['vs_native_kitchen_bf16']=compare(actual.cpu(),kitchen_reference)
            native_policy=replace(reader.operator_policy,dit_attention='native')
            with patch.object(owner,'dist',comm),patch.object(blocks[0],'operator_policy',native_policy),patch.object(blocks[1],'operator_policy',native_policy):
                dense=comm.run([lambda r=r:call(blocks[r],value) for r in (0,1)])
            row['sdpa_attention_ablation_vs_native_bf16']=compare(dense[0].float().cpu()*flow.divisor,reference)
            del dense
        if flow.mode=='bf16_reference':
            runtime_bf16=actual.cpu()
            if kind=='dit':
                def native_norm(norm,value,shift,scale,spans):
                    h=torch.nn.functional.rms_norm(value,norm.normalized_shape,norm.weight.to(value.dtype),norm.eps)
                    for a,b,idx in spans:h[a:b].mul_(1.+scale[idx].to(value.dtype)).add_(shift[idx].to(value.dtype))
                    return h
                def native_gate(value,gate,branch,spans):
                    for a,b,idx in spans:value[a:b].addcmul_(branch[a:b],gate[idx].to(value.dtype))
                    return value
                # Controlled BF16 rounding ablation. QK remains the runtime's
                # FP32 island; this changes only pre-norm/modulation/residual.
                with patch.object(owner,'dist',comm),patch('patches.h3_tensor_flow.modulated_norm',native_norm),patch('patches.h3_tensor_flow.gated_add',native_gate):
                    ablated=comm.run([lambda r=r:call(blocks[r],value) for r in (0,1)])
                row['native_bf16_norm_modulation_residual_rounding_ablation']=compare(ablated[0].float().cpu(),reference)
                row['native_bf16_rounding_ablation_vs_matched_kitchen']=compare(ablated[0].float().cpu(),kitchen_reference)
                del ablated
        else:row['vs_runtime_bf16_with_same_fp32_islands']=compare(actual.cpu(),runtime_bf16)
        report['cases'].append(row);save(args.output,report)
        print(flow,row['vs_native_bf16'],flush=True)
        del blocks,outputs,repeated,actual;gc.collect();torch.cuda.empty_cache()


def frontend(args,report):
    from patches.comfy_tp_models import native_dit
    source=Path('D:/attention_eval/quality_constrained_20261015/quality_challenge_v1/generation_screen/challenge_inputs_seed20261013/conditioning.pt')
    context=torch.load(source,map_location='cpu',weights_only=False)['positive'][0][0].cuda().float()
    reader=CheckpointReader(args.checkpoint)
    # Native BF16 execution: operations and residual activations use BF16,
    # condition projection input remains an explicit FP32 boundary.
    model=native_dit(reader)
    # Preserve actual checkpoint float values; Native linears follow input dtype.
    # The reference refiner receives BF16, not an FP32 result rounded afterward.
    model=model.cuda()
    with observe(model.token_refiner) as obs:
        refined=model.token_refiner(model.condition_proj(context[0]).bfloat16())[None]
        reference=refined.float().cpu()
    report['bf16_reference']=dict(output=stats(reference),operators=obs)
    packed_reference=capture_pack(model,refined,source.parent,torch.bfloat16)
    del model;gc.collect();torch.cuda.empty_cache()
    reader.tensor_flow=TensorFlowPolicy('fp16',args.exponent);model=native_dit(reader).cuda()
    with observe(model.token_refiner) as obs:
        actual=model.preprocess_text_embeds(context)
    report['cases'].append(dict(stage='condition_proj_and_native_sdpa_refiner',vs_native_bf16=compare(actual.cpu(),reference),
        output=stats(actual),storage_dtype=str(actual.dtype),operators=obs))
    packed=capture_pack(model,actual,source.parent,torch.float16)
    report['cases'].append(dict(stage='all_modality_packing',storage_dtype=str(packed.dtype),
        shape=list(packed.shape),output=stats(packed),vs_native_bf16=compare(packed.float()*reader.tensor_flow.divisor,packed_reference)))
    report['distribution']=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),shape=list(context.shape))
    del model


def runtime_frontend(args,report):
    """Exercise the actual rank-zero leases and half-precision broadcasts."""
    from patches.comfy_tp_models import native_dit
    from patches.windowed_tp import DiTPreprocessing
    from patches.weight_residency import copy_module
    from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
    source=Path('D:/attention_eval/quality_constrained_20261015/quality_challenge_v1/generation_screen/challenge_inputs_seed20261013')
    context=torch.load(source/'conditioning.pt',map_location='cpu',weights_only=False)['positive'][0][0].cuda().float()
    reader=CheckpointReader(args.checkpoint);reader.tensor_flow=TensorFlowPolicy('fp16',args.exponent)
    models=[native_dit(reader,r) for r in (0,1)]
    comm=LogicalDualGPU(timeout=120);isolate_cublas_workspaces()
    owners=[DiTPreprocessing(m,r,'cuda:0',comm,('flow_probe',reader.tensor_flow)) for r,m in enumerate(models)]
    class Leases:
        def __init__(self,owner):
            self.modules={s.key:s.prepare().cuda() for s in owner.specs()}
        @contextmanager
        def use(self,key):yield self.modules[key]
    for model,owner in zip(models,owners):
        owner.window=Leases(owner)
        # Move only auxiliary heads/buffers, mirroring the runtime ownership.
        for name,child in list(model.named_children()):
            if name not in owner.canonical:setattr(model,name,copy_module(child,lambda t:t.cuda()))
    refined=comm.run([lambda r=r:models[r].preprocess_text_embeds(context) for r in (0,1)])
    packed=comm.run([lambda r=r:capture_pack(models[r],refined[r],source,torch.float16) for r in (0,1)])
    report['cases'].append(dict(stage='actual_preprocessing_leases_and_broadcasts',
        refined_dtype=str(refined[0].dtype),packed_dtype=str(packed[0].dtype),
        refined_rank_exact=torch.equal(refined[0],refined[1]),packed_rank_exact=torch.equal(packed[0],packed[1]),
        output=stats(packed[0]),collective_steps_released=not comm.steps))
    for owner in owners:owner.close()


def capture_pack(model,context,source,dtype):
    from patches.comfy_tp_runtime import packet
    meta=torch.load(source/'conditioning.pt',map_location='cpu',weights_only=False)['positive'][0][1]
    noise=torch.load(source/'initial_noise.pt',map_location='cpu',weights_only=False)
    cfg=json.loads((source/'run_state.json').read_text(encoding='utf-8'))['config']
    payload=dict(seed=cfg['seed'],audio_scale=1.)
    if 'minimax_token_tags' in meta:payload['text_token_tags']=meta['minimax_token_tags']
    for key,dest in [('minimax_refs','refs'),('minimax_keyframes','keyframes')]:
        if key in meta:payload[dest]=meta[key]
    items=payload.get('refs',[])+payload.get('keyframes',[])
    payload['cond_video_latents']=[r['latent'] for r in items if 'latent' in r]
    payload['cond_audio_latents']=[r['audio_latent'] for r in items if r.get('audio_latent') is not None]
    for key in ('minimax_visual_cond_noise_aug','minimax_audio_cond_noise_aug'):
        if key in meta:payload[key.removeprefix('minimax_')]=meta[key]
    captured=[]
    class Stop(RuntimeError):pass
    class Capture(nn.Module):
        def forward(self,h,*args,**kwargs):
            captured.append(h.cpu());raise Stop('Stop before main DiT blocks')
    model.blocks=nn.ModuleList([Capture()])
    try:
        model([x.cuda().to(dtype) for x in noise.unbind()],torch.tensor([1000.],device='cuda'),context,
              minimax_payload=packet(payload,'cuda'),transformer_options={})
    except Stop:pass
    if len(captured)!=1:raise RuntimeError('Native packing boundary was not reached')
    return captured[0]


def vision(args,report):
    import comfy.ops,numpy as np
    from PIL import Image
    from comfy.text_encoders.qwen3vl import Qwen3VLVisionModel,QWEN3VL_VISION,QWEN3VL_VISION_COMMON
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images
    from patches.h3_tensor_flow import prepare_te
    source=ROOT/'bench_assets/tokyo_neon.jpg'
    image=torch.from_numpy(np.array(Image.open(source).convert('RGB'),copy=True)).float()[None]/255.
    pixels,grid=process_qwen2vl_images(image,max_pixels=65536,patch_size=16,image_mean=[.5]*3,image_std=[.5]*3)
    reader=CheckpointReader(args.checkpoint)
    config={**QWEN3VL_VISION_COMMON,**QWEN3VL_VISION['qwen3vl_32b'],'out_hidden_size':5120}
    def load(dtype,flow=None):
        model=Qwen3VLVisionModel(config,dtype=dtype,device='cpu',ops=comfy.ops.mixed_precision_ops({'mixed_ops':True},dtype))
        model.load_state_dict(reader.read('visual.'),strict=True)
        if flow is not None:
            shell=nn.Module();shell.model=nn.Identity();shell.visual=model;prepare_te(shell,flow)
        return model.cuda().eval().requires_grad_(False)
    baseline=load(torch.bfloat16)
    with observe(baseline) as obs:merged,deep=baseline(pixels.cuda().bfloat16(),grid.cuda())
    reference=[merged.float().cpu(),*[x.float().cpu() for x in deep]]
    report['bf16_reference']=dict(operators=obs,output=[stats(x) for x in reference])
    del baseline,merged,deep;gc.collect();torch.cuda.empty_cache()
    model=load(torch.float16,TensorFlowPolicy('fp16',4))
    with observe(model) as obs:merged,deep=model(pixels.cuda(),grid.cuda())
    original_half=[x.float().cpu() for x in [merged,*deep]]
    report['cases'].append(dict(stage='default_fp16_trained_27_block_vision_and_all_deepstack_mergers',
        vs_native_bf16=[compare(a.cpu(),b) for a,b in zip([merged,*deep],reference)],operators=obs))
    del model,merged,deep;gc.collect();torch.cuda.empty_cache()
    model=load(torch.float16,TensorFlowPolicy('fp16',4,vision_residual_exponent=4))
    with observe(model) as obs:merged,deep=model(pixels.cuda(),grid.cuda())
    report['cases'].append(dict(stage='optional_c16_vision_fold',
        vs_native_bf16=[compare(a.cpu(),b) for a,b in zip([merged,*deep],reference)],
        vs_unscaled_fp16=[compare(a.cpu(),b) for a,b in zip([merged,*deep],original_half)],operators=obs))
    report['distribution']=dict(image=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        max_pixels=65536,grid=grid.tolist(),caveat='Native Tokyo image preprocessing at bounded resolution; not the full reference workload.')


@torch.inference_mode(False)
@torch.no_grad()
def reference_video(reader,names,stage):
    """Load checkpoint -> BF16 directly, avoiding an FP16 intermediate."""
    import comfy.ops
    if stage=='decode':
        from patches.parallel_vae import VideoDecodeWeights
        model=VideoDecodeWeights()
        model.load_state_dict({n:reader.read_tensor(n) for n in names},strict=True)
    else:
        from patches.staged_vae import component_names
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE,LATENTS_MEAN,LATENTS_STD,IMAGENET_MEAN,IMAGENET_STD
        with torch.device('meta'):
            model=MiniMaxH3VideoVAE(operations=comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.bfloat16))
        del model.decoder,model.post_quant_conv
        model.to_empty(device='cpu')
        model.latents_mean=torch.tensor(LATENTS_MEAN);model.latents_std=torch.tensor(LATENTS_STD)
        model.pixel_mean=torch.tensor(IMAGENET_MEAN).view(1,3,1,1,1)
        model.pixel_std=torch.tensor(IMAGENET_STD).view(1,3,1,1,1)
        missing,extra=model.load_state_dict({n:reader.read_tensor(n) for n in component_names(reader,'video','encode')},strict=False,assign=True)
        if set(missing)-{'latents_mean','latents_std'} or extra:raise ValueError('Incomplete BF16 encoder reference')
    return model.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)


def video(args,report):
    from patches.parallel_vae import checkpoint_plan,load_decode_weights,AlignedVideoDecode
    from patches.staged_vae import load_component
    from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
    reader,names,_=checkpoint_plan(args.checkpoint,'video');isolate_cublas_workspaces()
    z=torch.randn((1,24,1,4,4),generator=torch.Generator().manual_seed(20261021)).cuda()
    reference_model=reference_video(reader,names,'decode')
    with observe(reference_model.decoder) as obs:reference=reference_model._decode_pixels(z.bfloat16()).float().cpu()
    report['bf16_reference']=dict(output=stats(reference),operators=obs)
    count=sum(k.startswith('__int8_attention_before_sanitizer_') for k in obs)
    if count!=36:raise ValueError('BF16 reference changed the legacy INT8 attention dispatch')
    report['bf16_reference']['checkpoint_to_bf16_direct']=True
    del reference_model;gc.collect();torch.cuda.empty_cache()
    unscaled=load_decode_weights(reader,names,'video','cuda')
    original_half=unscaled._decode_pixels(z.half()).float().cpu()
    from patches.h3_tensor_flow import prepare_video
    prepare_video(unscaled,TensorFlowPolicy('fp16',4))
    default_half=unscaled._decode_pixels(z.half()).float().cpu()
    report['cases'].append(dict(stage='default_fp16_decoder',native_fp16_exact=torch.equal(default_half,original_half),
        output=stats(default_half),vs_native_bf16=compare(default_half,reference)))
    del unscaled;gc.collect();torch.cuda.empty_cache()
    reader.tensor_flow=TensorFlowPolicy('fp16',4,video_residual_exponent=4)
    models=[load_decode_weights(reader,names,'video','cuda') for _ in range(2)]
    with observe(models[0].decoder) as obs:actual=models[0]._decode_pixels(z.half())
    comm=LogicalDualGPU(timeout=120)
    runners=[AlignedVideoDecode(m,z.half(),comm,r) for r,m in enumerate(models)]
    # 16 patches + 5 suffix tokens cannot be split at 128-query boundaries.
    # Use a second 261-token probe to actually exercise both query ranks.
    large=torch.randn((1,24,1,16,16),generator=torch.Generator().manual_seed(20261021)).cuda().half()
    expected=models[0]._decode_pixels(large)
    runners=[AlignedVideoDecode(m,large.clone(),comm,r) for r,m in enumerate(models)]
    results=comm.run([r.run for r in runners])
    report['cases'].append(dict(stage='optional_c16_trained_36_block_decoder',vs_native_bf16=compare(actual.cpu(),reference),output=stats(actual),
        vs_unscaled_native_fp16=compare(actual.cpu(),original_half),unscaled_fp16_vs_bf16=compare(original_half,reference),
        query_tokens=261,query_parallel_exact=torch.equal(expected,results[0]),query_rank1_none=results[1] is None,operators=obs))
    del models,runners,results,actual,expected;gc.collect();torch.cuda.empty_cache()
    # Encoder has no residual gauge fold across GN->SiLU. Confirm the existing
    # FP16 computation separately against BF16, using a tiny trained encode.
    reader.tensor_flow=TensorFlowPolicy()
    encoder=reference_video(reader,None,'encode')
    pixels=torch.randn((1,3,1,32,32),generator=torch.Generator().manual_seed(20261021)).cuda()
    ref=encoder._encode_moments(pixels.bfloat16()).float().cpu()
    del encoder;gc.collect();torch.cuda.empty_cache()
    reader.tensor_flow=TensorFlowPolicy('fp16',4)
    encoder=load_component(reader,'video','encode').cuda()
    actual=encoder._encode_moments(pixels.half())
    report['cases'].append(dict(stage='trained_video_encoder',vs_native_bf16=compare(actual.cpu(),ref),output=stats(actual),
        caveat='Existing FP16 encoder; synthetic pixels. Candidate freshly reloads original checkpoint after BF16 baseline.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['dit','te','frontend','runtime','video','vision'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tokens',type=int,default=513)
    parser.add_argument('--layer',type=int,default=0);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--exponent',type=int,default=6)
    parser.add_argument('--fixture',type=Path,default=Path('D:/attention_eval/fp16_residual_fix_20261020/first_block.pt'))
    parser.add_argument('--lora',type=Path)
    args=parser.parse_args();sys.argv=sys.argv[:1]
    from patches.legacy_runtime import validate_source,validate_packages
    validate_source(ROOT/'ComfyUI');validate_packages()
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Actual BF16-capable GPU required')
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if args.checkpoint is None:
        args.checkpoint=Path('D:/minimaxH3')/dict(dit='MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
            te='qwen3vl_32b_minimax_h3_int8_convrot.safetensors',frontend='MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
            video='minimax_h3_video_vae_int8_convrot.safetensors',vision='qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
            runtime='MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')[args.mode]
    report=dict(status='running',scope='Bounded trained components on one BF16-capable local GPU. Logical TP2 transport; no NCCL/T4 timing or full generation qualification.',
        hardware=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),torch=torch.__version__,checkpoint=str(args.checkpoint),
        configuration=dict(mode=args.mode,tokens=args.tokens,layer=args.layer,exponent=args.exponent,lora=str(args.lora) if args.lora else None),
        runtime_lock_sha256=hashlib.sha256((ROOT/'runtime/local_conda.json').read_bytes()).hexdigest(),cases=[])
    try:
        with torch.inference_mode():
            if args.mode in ('dit','te'):block_cases(args,report)
            elif args.mode=='frontend':frontend(args,report)
            elif args.mode=='vision':vision(args,report)
            elif args.mode=='runtime':runtime_frontend(args,report)
            else:video(args,report)
        report['status']='complete'
    except Exception as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:
        sources=['patches/h3_tensor_flow.py','patches/tp_dit.py','patches/tp_qwen3vl.py','patches/h3_lora.py',
            'patches/comfy_tp_models.py','patches/comfy_tp_runtime.py','patches/comfy_sampler.py','patches/windowed_tp.py',
            'patches/staged_vae.py','patches/parallel_vae.py','benchmarks/run_h3_tensor_flow.py']
        report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
        save(args.output,report)


if __name__=='__main__':main()
