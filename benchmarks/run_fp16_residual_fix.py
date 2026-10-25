"""Bounded local FP16 gauge/weight-fold experiments; no full H3 or Kaggle run."""
import argparse
from contextlib import ExitStack
import gc, hashlib, json, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
from torch import nn
from fp16_residual_fix import Gauge,dit_block,folded_rows,boosted_norm_gain,range_stats,storage_stats,fold_rms_gain_int8
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.legacy_runtime import validate_source,validate_packages


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def metrics(actual,reference):
    a=actual.detach().float();b=reference.detach().float();finite=torch.isfinite(a)&torch.isfinite(b)
    if not bool(finite.all()):return dict(nonfinite=int((~finite).sum()),relative_l2=None,maximum_error=None)
    # FP32 global norms/dots over millions of elements can even report cosine>1.
    af=a.flatten().double();bf=b.flatten().double();delta=af-bf
    return dict(nonfinite=0,relative_l2=float(delta.norm()/bf.norm().clamp_min(1e-300)),
        maximum_error=float(delta.abs().max()),cosine=float((torch.dot(af,bf)/(af.norm()*bf.norm()).clamp_min(1e-300)).clamp(-1.,1.)),
        exact_elements=int((a==b).sum()),elements=a.numel())


def prepare(args):
    """Run only Native front-end operators; intercept before the first DiT block."""
    from patches.comfy_tp_models import native_dit
    from patches.comfy_tp_runtime import packet
    source=args.inputs
    cached=torch.load(source/'conditioning.pt',map_location='cpu',weights_only=False)
    noise=torch.load(source/'initial_noise.pt',map_location='cpu',weights_only=False)
    sigmas=torch.load(source/'sigmas.pt',map_location='cpu',weights_only=True)
    if float(sigmas[0])!=1.:raise ValueError('This initial-noise-only fixture requires sigma0=1')
    state=json.loads((source/'run_state.json').read_text(encoding='utf-8'))
    cfg=state['config'];context,meta=cached['positive'][0]
    payload=dict(seed=cfg['seed'],audio_scale=1.)
    if 'minimax_token_tags' in meta:payload['text_token_tags']=meta['minimax_token_tags']
    for key,dest in [('minimax_refs','refs'),('minimax_keyframes','keyframes')]:
        if key in meta:payload[dest]=meta[key]
    payload['cond_video_latents']=[r['latent'] for r in payload.get('refs',[])+payload.get('keyframes',[]) if 'latent' in r]
    payload['cond_audio_latents']=[r['audio_latent'] for r in payload.get('refs',[])+payload.get('keyframes',[]) if r.get('audio_latent') is not None]
    for key in ('minimax_visual_cond_noise_aug','minimax_audio_cond_noise_aug'):
        if key in meta:payload[key.removeprefix('minimax_')]=meta[key]
    reader=CheckpointReader(args.dit)
    model=native_dit(reader).cuda()
    captured={}
    class Stop(RuntimeError):pass
    class Capture(nn.Module):
        def forward(self,h,t_emb,segments,rope,transformer_options=None,**kw):
            captured.update(input=h.cpu(),t_emb=t_emb.cpu(),segments=packet(segments,'cpu'),rope=rope.cpu())
            captured['layout_segments']=transformer_options['minimax_h3_layout'].segments
            raise Stop('Stopped before the first main DiT block')
    model.blocks=nn.ModuleList([Capture()])
    streams=[x.cuda().float() for x in noise.unbind()]
    try:
        model(streams,torch.tensor([1000.],device='cuda'),context.cuda(),
              minimax_payload=packet(payload,'cuda'),transformer_options={})
    except Stop:pass
    if 'input' not in captured:raise RuntimeError('Native block-input intercept did not run')
    captured['provenance']=dict(kind='current pinned Native first-block reconstruction from retained real conditioning/reference/noise',
        seed=cfg['seed'],width=cfg['width'],height=cfg['height'],frames=cfg['frames'],sigma=1.,
        source_files={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('conditioning.pt','initial_noise.pt','sigmas.pt')},
        scope='Native front-end operators and text refiner only; zero main DiT blocks, no TE/VAE/generation')
    torch.save(captured,args.output/'first_block.pt')
    x=captured['input'].cuda()
    info=dict(provenance=captured['provenance'],shape=list(x.shape),range=range_stats(x),
        segments=[dict(start=a,end=b,kind=kind,range=range_stats(x[a:b])) for a,b,kind in captured['layout_segments']])
    save(args.output/'fixture.json',info);print(json.dumps(info,indent=2),flush=True)


def screen(args):
    import patches.tp_dit as tp
    from patches.optimized_tp import enable
    from patches.h3_operator_policy import H3OperatorPolicy,configure_block
    from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
    fixture=torch.load(args.output/'first_block.pt',map_location='cpu',weights_only=False)
    selected=torch.arange(fixture['input'].shape[0]) if args.tokens==0 else torch.linspace(0,fixture['input'].shape[0]-1,args.tokens).long().unique()
    x=fixture['input'][selected].cuda();rope=fixture['rope'][:,selected].cuda();t=fixture['t_emb'].cuda()
    # Project original modality rows onto the selected, order-preserving token subset.
    segments=[]
    for a,b,row in fixture['segments']:
        ids=((selected>=a)&(selected<b)).nonzero().flatten()
        if ids.numel():segments.append((int(ids[0]),int(ids[-1])+1,row if isinstance(row,int) else row[selected[ids]-a].cuda()))
    reader=CheckpointReader(args.dit)
    if args.lora:
        from patches.h3_lora import H3LoRAStore
        reader.lora_store=H3LoRAStore(((args.lora,1.),),reader,selective_b=True)
    policy=H3OperatorPolicy();reader.operator_policy=policy
    enable('scalemax');isolate_cublas_workspaces()
    blocks=[prepare_tp_block(reader,'dit',0,r).cuda() for r in (0,1)]
    comm=LogicalDualGPU(timeout=90);original=tp.dist;tp.dist=comm
    state=dict(status='running',scope='Local SM120 logical TP2, real trained block0 and reconstructed real sigma=1 input; no T4 performance or generation-quality claim',
        hardware=torch.cuda.get_device_name(),torch=torch.__version__,comfy_revision='d49e888586dd8ae012c0667b33466b815fee07f7',
        fixture_shape=list(fixture['input'].shape),tested_shape=list(x.shape),lora=bool(args.lora),cases=[],
        retained_islands=['FP32 RMSNorm accumulation/gain','FP32 AdaLN','FP32 QK norm/RoPE','INT32 GEMM SUM','Native final output heads outside this screen'])
    suffix='_lora' if args.lora else ''
    path=args.output/f'screen_s{x.shape[0]}{suffix}.json'
    def run(g,observed=True):
        observations=[{},{}] if observed else [None,None]
        with ExitStack() as stack:
            if g.norm_gain_exponent:
                for b in blocks:stack.enter_context(boosted_norm_gain(b,g.norm_gain_exponent))
            if g.fold_output:
                for b in blocks:stack.enter_context(folded_rows(b,g.divisor))
            values=comm.run([lambda r=r:dit_block(blocks[r],(x/g.divisor).to(g.storage_dtype),t,segments,rope,g,observations[r]) for r in (0,1)])
        return [v.float()*g.divisor for v in values],observations
    try:
        reference=comm.run([lambda r=r:blocks[r](x.clone(),t,segments,rope,{}) for r in (0,1)])
        ref=[v.cpu() for v in reference];del reference
        state['reference_sha256']=[hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for v in ref]
        state['fixture_sha256']=hashlib.sha256((args.output/'first_block.pt').read_bytes()).hexdigest()
        configs=[('fp32_control',Gauge(storage_dtype=torch.float32)),
                 ('fp32_gauge_c16',Gauge(4,torch.float32,torch.float32)),
                 ('fp32_fold_c16',Gauge(4,torch.float32,torch.float32,True)),
                 ('fp32_normboost64_control',Gauge(0,torch.float32,torch.float32,False,6)),
                 ('fp32_resid_fp16_branch',Gauge(0,torch.float32,torch.float16)),
                 ('fp16_flow_fold_c16_normboost64',Gauge(4,torch.float16,torch.float16,True,6))]
        for k in (0,2,4,6,8,12):configs.append((f'fp16_store_c{2**k}',Gauge(k)))
        for k in (0,4,8):
            configs.extend([(f'fp16_flow_c{2**k}',Gauge(k,branch_dtype=torch.float16)),
                            (f'fp16_flow_fold_c{2**k}',Gauge(k,branch_dtype=torch.float16,fold_output=True))])
        for name,g in configs:
            if g.norm_gain_exponent and any(not isinstance(b.adaln_proj,nn.Linear) for b in blocks):
                state['cases'].append(dict(name=name,status='contract_rejected',reason='AdaLN adapter row compensation not qualified'))
                save(path,state);continue
            values,obs=run(g);scores=[metrics(v.cpu(),r) for v,r in zip(values,ref)]
            row=dict(name=name,gauge=dict(exponent=g.exponent,storage=str(g.storage_dtype),branch=str(g.branch_dtype),fold=g.fold_output,norm_gain_exponent=g.norm_gain_exponent),
                     input_cast=storage_stats(x/g.divisor,(x/g.divisor).to(g.storage_dtype)),ranks=scores,observations=obs,
                     output_sha256=[hashlib.sha256(v.cpu().contiguous().numpy().tobytes()).hexdigest() for v in values])
            if name=='fp32_control' and any(s['exact_elements']!=s['elements'] for s in scores):
                row['failure']='Experimental FP32 control differs from current TP2 block';state['cases'].append(row);save(path,state);raise ValueError(row['failure'])
            state['cases'].append(row);save(path,state)
            print(name,json.dumps(scores),flush=True);del values
        state['status']='complete';state['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
            'benchmarks/fp16_residual_fix.py','benchmarks/run_fp16_residual_fix.py','patches/tp_dit.py','patches/h3_operator_policy.py','patches/optimized_tp.py']}
        save(path,state)
    finally:tp.dist=original


def norm_fold(args):
    import comfy_kitchen as ck
    reader=CheckpointReader(args.te);prefix='model.layers.0.'
    # Full K, selected output rows: an operator probe, not a complete TE block.
    w=reader.read_tensor(prefix+'self_attn.q_proj.weight',rows=[(0,512)]).cuda()
    scale=reader.read_tensor(prefix+'self_attn.q_proj.weight_scale',rows=[(0,512)]).cuda()
    gamma=reader.read_tensor(prefix+'input_layernorm.weight').cuda().float()
    q,qs,original,folded=fold_rms_gain_int8(w,scale,gamma)
    gen=torch.Generator(device='cuda').manual_seed(20261020)
    z=torch.randn(257,w.shape[1],generator=gen,device='cuda')
    z=z*torch.rsqrt((z*z).mean(-1,keepdim=True)+1e-6)
    before=ck.int8_linear(z*gamma,w,scale,out_dtype=torch.float32,convrot=True,convrot_groupsize=256)
    after=ck.int8_linear(z,q,qs,out_dtype=torch.float32,convrot=True,convrot_groupsize=256)
    dense_before=torch.nn.functional.linear(z*gamma,original)
    dense_after=torch.nn.functional.linear(z,folded)
    naive=ck.int8_linear(z,w,(scale),out_dtype=torch.float32,convrot=True,convrot_groupsize=256)
    boosted=ck.int8_linear((z*(gamma*64)).half(),w,scale/64,out_dtype=torch.float32,convrot=True,convrot_groupsize=256)
    direct_half=ck.int8_linear((z*gamma).half(),w,scale,out_dtype=torch.float32,convrot=True,convrot_groupsize=256)
    result=dict(scope='Trained TE layer0 Q projection, 512 output rows/full K5120, seeded synthetic normalized inputs; not real TE distribution or generation acceptance',
        dense_fold=metrics(dense_after,dense_before),int8_fold=metrics(after,before),removing_gain_without_folding=metrics(naive,before),
        fp16_input=metrics(direct_half,before),fp16_input_power2_gain_fold=metrics(boosted,before),
        original_input_cast=storage_stats(z*gamma,(z*gamma).half()),boosted_input_cast=storage_stats(z*(gamma*64),(z*(gamma*64)).half()),
        gamma=range_stats(gamma),requantized_weight_changed=int((q!=w).sum()),
        note='Native inverse ConvRot -> input-channel gamma -> Native full-width requantization. Dynamic AdaLN folding rejected.')
    save(args.output/'norm_fold.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','screen','norm-fold'])
    parser.add_argument('--output',type=Path,default=Path('D:/attention_eval/fp16_residual_fix_20261020'))
    parser.add_argument('--inputs',type=Path,default=Path('D:/attention_eval/quality_constrained_20261015/quality_challenge_v1/generation_screen/challenge_inputs_seed20261013'))
    parser.add_argument('--dit',default='D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')
    parser.add_argument('--te',default='D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors')
    parser.add_argument('--tokens',type=int,default=513);parser.add_argument('--lora',default='')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    validate_source(ROOT/'ComfyUI');validate_packages(preview=True)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    with torch.inference_mode():{'prepare':prepare,'screen':screen,'norm-fold':norm_fold}[args.mode](args)


if __name__=='__main__':main()
