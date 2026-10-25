"""OUT OF SCOPE: historical forced-dense-FP32 experiment, not the user's reference.

The requested reference is Native FP32 tensor flow with the current INT8
operators retained. Use run_h3_native_fp32_flow.py for that comparison.

No full H3, dependency change or Kaggle submission. Historical BF16 pairwise
records remain untouched. Fixed INT8 checkpoint weights define this study's
weight values; their original pre-quantization error cannot be recovered.
"""
import argparse,gc,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
from torch import nn
from fp32_ground_truth import dense_fp32
from run_h3_tensor_flow import stats,compare,save,observe,native_block,capture_pack
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.h3_tensor_flow import TensorFlowPolicy


def cleanup():gc.collect();torch.cuda.empty_cache()


def golden_record(output,contract):
    values=output if isinstance(output,list) else [output]
    if any(not torch.isfinite(x).all() for x in values):raise ValueError('Nonfinite FP32 ground truth')
    return dict(role='ground_truth',activation_dtype='torch.float32',
        linear='dense FP32 on Native inverse-ConvRot dequantized checkpoint weights',
        attention='dense FP32 SDPA; all keys; query chunk 128; no routing or INT8 attention',
        output=[stats(x) for x in values],contract=contract,
        output_sha256=[hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest() for x in values])


def row(name,value,gold,**extra):
    return dict(name=name,role='candidate',output=stats(value),vs_fp32_ground_truth=compare(value.cpu(),gold),**extra)


def block_inputs(args):
    if args.mode=='dit':
        f=torch.load(args.fixture,map_location='cpu',weights_only=False)
        ids=torch.arange(f['input'].shape[0]) if args.tokens==0 else torch.linspace(0,f['input'].shape[0]-1,args.tokens).long().unique()
        x=f['input'][ids].cuda().float();t=f['t_emb'].cuda().float();rope=f['rope'][:,ids].cuda().float();segments=[]
        for a,b,idx in f['segments']:
            selected=((ids>=a)&(ids<b)).nonzero().flatten()
            if selected.numel():segments.append((int(selected[0]),int(selected[-1])+1,idx if isinstance(idx,int) else idx[ids[selected]-a].cuda()))
        def call(model,value):return model(value.clone(),t,segments,rope.to(value.dtype),{})
        info=dict(source=str(args.fixture),sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),provenance=f['provenance'],
            shape=list(x.shape),layer=args.layer,caveat='Only layer 0 uses its reconstructed native input; late layers reuse it as a stress probe.')
    else:
        from fixture_inputs import block_input
        from comfy.text_encoders.llama import precompute_freqs_cis,Qwen3VL_32BConfig
        from comfy.ldm.modules.attention import optimized_attention_for_device
        from patches.tp_qwen3vl import TPTransformerBlockINT8
        x=block_input('te',args.tokens or 257,20261021)['input'].cuda().float()
        config=Qwen3VL_32BConfig()
        rope=precompute_freqs_cis(128,torch.arange(x.shape[1],device=x.device)[None],config.rope_theta,
            rope_dims=config.rope_dims,device=x.device,interleaved_mrope=True)
        def call(model,value):
            mask=torch.full((value.shape[1],value.shape[1]),torch.finfo(value.dtype).min/4,device=value.device,dtype=value.dtype).triu_(1)
            if isinstance(model,TPTransformerBlockINT8):return model(value.clone(),mask,rope)
            return model(value.clone(),attention_mask=mask,freqs_cis=rope,
                optimized_attention=optimized_attention_for_device(value.device,mask=True,small_input=True))[0]
        info=dict(source='Identical seeded synthetic TE inputs, half-rounded before this study',shape=list(x.shape),layer=args.layer)
    return x,call,info


def blocks(args,report):
    from patches import tp_dit,tp_qwen3vl
    from patches.optimized_tp import enable
    from patches.h3_operator_policy import H3OperatorPolicy
    from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
    reader=CheckpointReader(args.checkpoint)
    if args.lora:
        from patches.h3_lora import H3LoRAStore
        reader.lora_store=H3LoRAStore(((str(args.lora),1.),),reader,selective_b=True)
    x,call,report['distribution']=block_inputs(args)
    model=native_block(args.mode,reader,args.layer,torch.float32)
    with dense_fp32(model) as contract:
        call(model,x)
        gold=call(model,x).cpu()
        repeat=call(model,x).cpu()
    if not torch.equal(gold,repeat):raise ValueError('FP32 ground truth is not repeatable')
    report['fp32_ground_truth']=golden_record(gold,contract)
    report['fp32_ground_truth']['repeat_equal']=True
    actual=call(model,x)
    report['cases'].append(row('native_fp32_w8a8_sdpa_control',actual,gold))
    del model,actual,repeat;cleanup()
    model=native_block(args.mode,reader,args.layer,torch.bfloat16)
    with observe(model) as proof:actual=call(model,x.bfloat16())
    if actual.dtype!=torch.bfloat16:raise ValueError('BF16 candidate did not execute in BF16')
    report['cases'].append(row('native_bf16_w8a8_sdpa',actual,gold,operators=proof))
    if args.mode=='dit':
        from comfy.ldm.modules.attention import attention_comfy_kitchen_int8
        model.attn.comfy_attention.function=attention_comfy_kitchen_int8
        actual=call(model,x.bfloat16())
        report['cases'].append(row('native_bf16_w8a8_kitchen',actual,gold))
    del model,actual;cleanup()
    reader.operator_policy=H3OperatorPolicy()
    owner=tp_dit if args.mode=='dit' else tp_qwen3vl
    enable('scalemax');isolate_cublas_workspaces();comm=LogicalDualGPU(timeout=180)
    configs=[('tp_fp32_w8a8_control',TensorFlowPolicy()),
             ('tp_bf16_w8a8',TensorFlowPolicy('bf16_reference',0)),
             ('tp_fp16_unscaled',TensorFlowPolicy('fp16',0)),
             ('tp_fp16_fold_c16',TensorFlowPolicy('fp16',4)),
             ('tp_fp16_fold_c64',TensorFlowPolicy('fp16',6))]
    for name,policy in configs:
        reader.tensor_flow=policy
        models=[prepare_tp_block(reader,args.mode,args.layer,r).cuda() for r in (0,1)]
        divisor=policy.divisor if policy.enabled else 1.
        dtype=policy.dtype if policy.enabled else torch.float32
        value=(x/divisor).to(dtype)
        with patch.object(owner,'dist',comm),observe(models[0]) as proof:
            comm.run([lambda r=r:call(models[r],value) for r in (0,1)])
            actual=comm.run([lambda r=r:call(models[r],value) for r in (0,1)])
            repeated=comm.run([lambda r=r:call(models[r],value) for r in (0,1)])
        result=row(name,actual[0].float()*divisor,gold,operators=proof,divisor=divisor,
            stored_dtype=str(dtype),rank_equal=torch.equal(actual[0],actual[1]),repeat_equal=torch.equal(actual[0],repeated[0]))
        report['cases'].append(result);save(args.output,report)
        print(name,result['vs_fp32_ground_truth'],flush=True)
        del models,actual,repeated;cleanup()


def frontend(args,report):
    from patches.comfy_tp_models import native_dit
    source=Path('D:/attention_eval/quality_constrained_20261015/quality_challenge_v1/generation_screen/challenge_inputs_seed20261013')
    context=torch.load(source/'conditioning.pt',map_location='cpu',weights_only=False)['positive'][0][0].cuda().float()
    reader=CheckpointReader(args.checkpoint);model=native_dit(reader).cuda()
    with dense_fp32(model) as contract:
        refined=model.preprocess_text_embeds(context)
        gold_ref=refined.cpu();gold_pack=capture_pack(model,refined,source,torch.float32)
    report['fp32_ground_truth']=golden_record([gold_ref,gold_pack],contract)
    del model,refined;cleanup()
    model=native_dit(reader).cuda()
    refined=model.token_refiner(model.condition_proj(context[0]).bfloat16())[None]
    report['cases'].append(row('bf16_native_refiner',refined,gold_ref))
    packed=capture_pack(model,refined,source,torch.bfloat16)
    report['cases'].append(row('bf16_native_packing',packed,gold_pack))
    del model,refined,packed;cleanup()
    reader.tensor_flow=TensorFlowPolicy('fp16',6);model=native_dit(reader).cuda()
    refined=model.preprocess_text_embeds(context)
    report['cases'].append(row('fp16_refiner_c64',refined,gold_ref))
    packed=capture_pack(model,refined,source,torch.float16)
    report['cases'].append(row('fp16_packing_c64',packed.float()*64,gold_pack,stored_dtype=str(packed.dtype)))
    report['distribution']=dict(source=str(source/'conditioning.pt'),sha256=hashlib.sha256((source/'conditioning.pt').read_bytes()).hexdigest(),
        caveat='Common retained conditioning/reference/noise. This does not recapture a whole FP32 TE trajectory.')


def vision(args,report):
    import comfy.ops,numpy as np
    from PIL import Image
    from comfy.text_encoders.qwen3vl import Qwen3VLVisionModel,QWEN3VL_VISION,QWEN3VL_VISION_COMMON
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images
    from patches.h3_tensor_flow import prepare_te
    source=ROOT/'bench_assets/tokyo_neon.jpg'
    image=torch.from_numpy(np.array(Image.open(source).convert('RGB'),copy=True)).float()[None]/255.
    pixels,grid=process_qwen2vl_images(image,max_pixels=65536,patch_size=16,image_mean=[.5]*3,image_std=[.5]*3)
    pixels,grid=pixels.cuda(),grid.cuda();reader=CheckpointReader(args.checkpoint)
    config={**QWEN3VL_VISION_COMMON,**QWEN3VL_VISION['qwen3vl_32b'],'out_hidden_size':5120}
    def load(dtype,policy=None):
        model=Qwen3VLVisionModel(config,dtype=dtype,device='cpu',ops=comfy.ops.mixed_precision_ops({'mixed_ops':True},dtype))
        model.load_state_dict(reader.read('visual.'),strict=True)
        if policy is not None:
            shell=nn.Module();shell.model=nn.Identity();shell.visual=model;prepare_te(shell,policy)
        return model.cuda().eval().requires_grad_(False)
    model=load(torch.float32)
    with dense_fp32(model) as contract:merged,deep=model(pixels,grid)
    gold=[a.cpu() for a in [merged,*deep]]
    report['fp32_ground_truth']=golden_record(gold,contract)
    del model,merged,deep;cleanup()
    for name,dtype,policy in [('bf16_native',torch.bfloat16,None),('fp16_native',torch.float16,TensorFlowPolicy('fp16',6)),
                              ('fp16_optional_vision_c16',torch.float16,TensorFlowPolicy('fp16',6,vision_residual_exponent=4))]:
        model=load(dtype,policy);merged,deep=model(pixels.to(dtype),grid)
        report['cases'].append(dict(name=name,role='candidate',vs_fp32_ground_truth=[compare(a.cpu(),b) for a,b in zip([merged,*deep],gold)],
            output=[stats(a) for a in [merged,*deep]]))
        del model,merged,deep;cleanup()
    report['distribution']=dict(image=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),grid=grid.tolist(),max_pixels=65536)


@torch.inference_mode(False)
@torch.no_grad()
def load_video_dtype(reader,names,stage,dtype=torch.float32):
    import comfy.ops
    if stage=='decode':
        from patches.parallel_vae import VideoDecodeWeights
        from comfy.ldm.minimax.vae import ViT3DDecoder
        model=VideoDecodeWeights()
        # MixedPrecision Linear ignores its per-module constructor dtype.
        # Replace the unloaded shell before loading any payload so dense F32
        # checkpoint weights never pass through an FP16 factory in the gold.
        model.decoder=ViT3DDecoder(operations=comfy.ops.mixed_precision_ops({'mixed_ops':True},dtype),num_layers=36)
        model.load_state_dict({n:reader.read_tensor(n) for n in names},strict=True)
    else:
        from patches.staged_vae import component_names
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE,LATENTS_MEAN,LATENTS_STD,IMAGENET_MEAN,IMAGENET_STD
        with torch.device('meta'):
            model=MiniMaxH3VideoVAE(operations=comfy.ops.mixed_precision_ops({'mixed_ops':True},dtype))
        del model.decoder,model.post_quant_conv;model.to_empty(device='cpu')
        model.latents_mean=torch.tensor(LATENTS_MEAN);model.latents_std=torch.tensor(LATENTS_STD)
        model.pixel_mean=torch.tensor(IMAGENET_MEAN).view(1,3,1,1,1);model.pixel_std=torch.tensor(IMAGENET_STD).view(1,3,1,1,1)
        missing,extra=model.load_state_dict({n:reader.read_tensor(n) for n in component_names(reader,'video','encode')},strict=False,assign=True)
        if set(missing)-{'latents_mean','latents_std'} or extra:raise ValueError('Incomplete FP32 encoder')
    model=model.to(device='cuda',dtype=dtype).eval().requires_grad_(False)
    if stage=='decode' and dtype==torch.float32:
        for name in ('decoder.x_embedder.weight','decoder.proj_out.weight'):
            module_name,parameter=name.rsplit('.',1)
            actual=getattr(model.get_submodule(module_name),parameter)
            if not torch.equal(actual,reader.read_tensor(name).to(device='cuda',dtype=torch.float32)):
                raise ValueError('FP32 ground truth rounded original checkpoint parameter: '+name)
    return model


def video(args,report):
    from patches.parallel_vae import checkpoint_plan,load_decode_weights
    from patches.staged_vae import load_component
    from patches.h3_tensor_flow import prepare_video
    reader,names,_=checkpoint_plan(args.checkpoint,'video')
    z=torch.randn((1,24,1,4,4),generator=torch.Generator().manual_seed(20261021)).cuda().float()
    model=load_video_dtype(reader,names,'decode')
    with dense_fp32(model) as contract:gold=model._decode_pixels(z).cpu()
    contract['original_fp32_dense_weights_exact']=True
    report['fp32_ground_truth_decoder']=golden_record(gold,contract)
    del model;cleanup()
    model=load_video_dtype(reader,names,'decode',torch.bfloat16);actual=model._decode_pixels(z.bfloat16())
    report['cases'].append(row('bf16_native_decoder',actual,gold));del model,actual;cleanup()
    model=load_decode_weights(reader,names,'video','cuda');actual=model._decode_pixels(z.half())
    report['cases'].append(row('fp16_native_decoder',actual,gold))
    prepare_video(model,TensorFlowPolicy('fp16',6,video_residual_exponent=4));folded=model._decode_pixels(z.half())
    report['cases'].append(row('fp16_optional_decoder_c16',folded,gold,vs_unfolded_fp16=compare(folded,actual)))
    del model,actual,folded;cleanup()
    pixels=torch.randn((1,3,1,32,32),generator=torch.Generator().manual_seed(20261021)).cuda().float()
    model=load_video_dtype(reader,None,'encode')
    with dense_fp32(model) as contract:gold=model._encode_moments(pixels).cpu()
    report['fp32_ground_truth_encoder']=golden_record(gold,contract)
    del model;cleanup()
    model=load_video_dtype(reader,None,'encode',torch.bfloat16);actual=model._encode_moments(pixels.bfloat16())
    report['cases'].append(row('bf16_native_encoder',actual,gold));del model,actual;cleanup()
    model=load_component(reader,'video','encode').cuda();actual=model._encode_moments(pixels.half())
    report['cases'].append(row('fp16_native_encoder',actual,gold))
    report['distribution']=dict(decoder_latent_shape=list(z.shape),encoder_pixel_shape=list(pixels.shape),seed=20261021,kind='synthetic tiny component probes')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['dit','te','frontend','vision','video'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tokens',type=int,default=513)
    parser.add_argument('--layer',type=int,default=0);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--lora',type=Path)
    parser.add_argument('--fixture',type=Path,default=Path('D:/attention_eval/fp16_residual_fix_20261020/first_block.pt'))
    args=parser.parse_args();sys.argv=sys.argv[:1]
    if args.output.exists():raise FileExistsError('Preserve existing results; choose a new output path')
    from patches.legacy_runtime import validate_source,validate_packages
    validate_source(ROOT/'ComfyUI');validate_packages()
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16-capable local GPU required for both candidates')
    torch.set_grad_enabled(False);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if args.checkpoint is None:
        args.checkpoint=Path('D:/minimaxH3')/dict(dit='MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',te='qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
            frontend='MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',vision='qwen3vl_32b_minimax_h3_int8_convrot.safetensors',video='minimax_h3_video_vae_int8_convrot.safetensors')[args.mode]
    reader=CheckpointReader(args.checkpoint)
    report=dict(schema='fp32-ground-truth-v1',status='running',reference_role='FP32 only; BF16 and FP16 are both candidates',
        scope='Local numerical components on SM120; no full H3, T4/NCCL, performance or generation qualification.',
        hardware=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),torch=torch.__version__,
        configuration=dict(mode=args.mode,tokens=args.tokens,layer=args.layer,lora=str(args.lora) if args.lora else None),
        checkpoint_identity=reader.identity,checkpoint_header_sha256=hashlib.sha256(json.dumps(reader.header,sort_keys=True).encode()).hexdigest(),
        runtime_lock_sha256=hashlib.sha256((ROOT/'runtime/local_conda.json').read_bytes()).hexdigest(),cases=[])
    try:
        with torch.inference_mode():
            if args.mode in ('dit','te'):blocks(args,report)
            else:{'frontend':frontend,'vision':vision,'video':video}[args.mode](args,report)
        report['status']='complete'
    except Exception as exc:report['status']='failed';report['error']=repr(exc);raise
    finally:
        sources=['benchmarks/run_h3_fp32_ground_truth.py','benchmarks/fp32_ground_truth.py','benchmarks/run_h3_tensor_flow.py',
            'patches/h3_tensor_flow.py','patches/tp_dit.py','patches/tp_qwen3vl.py','patches/h3_lora.py','patches/comfy_tp_models.py',
            'patches/parallel_int8.py','patches/optimized_tp.py','patches/parallel_vae.py','patches/staged_vae.py']
        report['source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources}
        save(args.output,report)


if __name__=='__main__':main()
