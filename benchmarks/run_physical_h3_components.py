"""Bounded refiner and actual dual-T4 VAE/idle service regressions; no sampler."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import weakref

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI')]


def refiner(assets,report):
    import torch,comfy.ops,comfy_kitchen as ck
    from comfy.ldm.minimax.model import Attention
    from patches.h3_operator_policy import NativeRefinerAttention
    from patches.windowed_tp import CheckpointReader
    reader=CheckpointReader(assets/'refiner_attention.safetensors')
    # This is the Native FP32 frontend factory, retaining checkpoint BF16 values.
    ops=comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.float32)
    base=Attention(5376,56,128,1e-5,device='cpu',dtype=torch.float32,operations=ops)
    base.load_state_dict(reader.read('token_refiner.blocks.0.attn.'),strict=True)
    base=base.cuda().eval().requires_grad_(False);wrapped=NativeRefinerAttention(base)
    x=torch.randn(353,5376,device='cuda',generator=torch.Generator(device='cuda').manual_seed(829))
    for use_rope in (False,True):
        rope=None
        if use_rope:
            angle=torch.arange(353,device='cuda').float()[:,None]*torch.linspace(.001,1,48,device='cuda')[None]
            c,s=angle.cos(),angle.sin();rope=torch.stack((c,-s,s,c),-1).reshape(1,353,1,48,2,2)
        def reference():
            q,k,v=base.qkv_proj(x).split(base.heads*base.head_dim,-1)
            q,k,v=(a.reshape(353,56,128) for a in (q,k,v))
            if rope is None:q,k=base.q_norm(q),base.k_norm(k)
            else:
                q=q[None];k=k[None]
                ck.rms_rope_split_half_(q,k,rope,base.q_norm.weight,base.k_norm.weight,epsilon=base.q_norm.eps,rot_dim=96)
                q,k=q[0],k[0]
            q,k,v=(a.transpose(0,1)[None] for a in (q,k,v))
            y=torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(353,56*128)
            return base.out_proj(y)
        expected=reference();actual=wrapped(x,rope)
        assert torch.equal(actual,expected) and torch.isfinite(actual).all()
        report['records'].append(dict(component='native_refiner_sdpa',trained_weights=True,
            rope=use_rope,input_shape=list(x.shape),dtype=str(x.dtype),byte_exact=True,
            scope='Native projection/norm/RoPE + independent SDPA expression; no INT8 attention substitution'))
    del base,wrapped,x;gc.collect();torch.cuda.empty_cache()


def vae(assets,report,save):
    import torch,comfy.sd,comfy.utils,comfy.model_management as management
    from patches.h3_operator_policy import H3OperatorPolicy
    from patches.comfy_tp_runtime import PhysicalTPService,RuntimeConfig
    from patches.staged_vae import StagedVAE
    from patches.comfy_sampler import ParallelDecodeHandoff
    from patches.parallel_vae import query_partitioning_status
    @torch.inference_mode(False)
    @torch.no_grad()
    def native_vae(path,dtype):
        return comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(path)),device=torch.device('cuda:0'),dtype=dtype)
    failures=[]
    GiB=2**30
    config=RuntimeConfig(str(assets/'te_block0.safetensors'),str(assets/'dit_blocks.safetensors'),layers=1,
        weight_bytes=GiB,pinned_bytes=2*GiB,cache_bytes=GiB,host_bytes=20*GiB,reserve_bytes=GiB,
        prefetch_lifecycle=True,idle_residency=True,idle_weight_bytes=GiB,idle_reserve_bytes=2*GiB,
        idle_delay_seconds=.05,cpu_lookahead=1,decode_te_blocks=1,adaptive_lookahead=True,
        operators=H3OperatorPolicy(te_attention='native',dit_attention='kitchen',row_linear_backend='native'))
    # No TL arithmetic runs in this ownership test. The separate operator gate
    # qualifies the pinned TileLang version; its historical production gate stays intact.
    service=PhysicalTPService(config);commands=[];call=service.call
    def logged(command,*args,**kwargs):commands.append(command);return call(command,*args,**kwargs)
    service.call=logged
    report['ownership_operator_policy']=vars(config.operators)
    try:
        video=StagedVAE(assets/'video_vae.safetensors',service)
        assert video.stage is None and video.reader.logical_read_bytes==0
        service.configure_idle_vaes({
            'v':dict(class_type='MiniMaxH3ParallelVAELoader',inputs={'vae_name':'video_vae.safetensors'}),
            'ref':dict(class_type='MiniMaxH3ReferenceToVideo',inputs={'vae':['v',0],'ref_image_1':['pixels',0]})})
        assert service.request_idle()
        thread=service.idle_vae_thread
        if thread is not None:thread.join(120)
        if thread is not None and thread.is_alive():raise TimeoutError('Encoder preload did not finish')
        # Internal service RPCs observe its real worker event loop, not Kaggle.
        deadline=time.monotonic()+60
        while True:
            idle=service.call('idle_stats')
            if all(r['loads']>=1 for r in idle):break
            if time.monotonic()>=deadline:raise TimeoutError('Physical TE idle prefix did not load')
            time.sleep(.2)
        assert service.idle_vae_stats['loads']>=1
        assert video.stage=='encode' and not hasattr(video.first_stage_model,'decoder')
        report['records'].append(dict(component='idle_encoder_and_te_coexist',status='passed',
            worker_idle=idle,parent_encoder=service.idle_vae_stats));save()
        # Hold completed idle weights while exercising Native parent encoding.
        baseline=native_vae(assets/'video_vae.safetensors',torch.float16)
        report['baseline_query_contract']=query_partitioning_status(baseline.first_stage_model);save()
        pixels=torch.rand(1,32,32,3,generator=torch.Generator().manual_seed(419))
        expected=baseline.encode(pixels.clone());actual=video.encode(pixels.clone())
        assert torch.equal(expected,actual)
        report['records'].append(dict(component='staged_video_encode',byte_exact=True,shape=list(actual.shape)))
        encoder=weakref.ref(video.first_stage_model)
        wrapper=ParallelDecodeHandoff(video,service,assets/'video_vae.safetensors','auto')
        for name,shape in [('short_query',(1,24,1,4,4)),('query_split',(1,24,1,16,16)),
                           ('temporal_query',(1,24,3,16,16))]:
            z=torch.randn(shape,generator=torch.Generator().manual_seed(732)).half()
            expected=baseline.decode(z).cpu();actual=wrapper.decode(z).cpu()
            byte_exact=torch.equal(actual,expected);finite=bool(torch.isfinite(actual).all())
            assert wrapper.last_strategy=='native_aligned_query'
            row=dict(component=name,status='passed' if byte_exact and finite else 'failed',byte_exact=byte_exact,
                finite=finite,output_shape=list(actual.shape),strategy=wrapper.last_strategy,fallback=wrapper.last_fallback)
            if not byte_exact:
                from run_h3_tensor_flow import compare
                row['vs_native']=compare(actual,expected)
                row['differing_elements']=int((actual!=expected).sum())
                row['native_repeat_exact']=torch.equal(expected,baseline.decode(z).cpu())
                row['parallel_repeat_exact']=torch.equal(actual,wrapper.decode(z).cpu())
                native_wrapper=ParallelDecodeHandoff(video,service,assets/'video_vae.safetensors','native')
                native_actual=native_wrapper.decode(z).cpu()
                row['staged_native_vs_native']=compare(native_actual,expected)
                row['staged_native_byte_exact']=torch.equal(native_actual,expected)
                failures.append(name)
            report['records'].append(row);save()
        gc.collect();assert encoder() is None and video.stage=='decode'
        assert not hasattr(video.first_stage_model,'encoder')
        # Explicitly exercise declined-worker admission with actual Native T4
        # decoding. Eligibility is controlled; model arithmetic is not mocked.
        def declined(command,*args,**kwargs):
            if command=='vae_prepare':return dict(accepted=False,reason='Controlled replica admission decline')
            return logged(command,*args,**kwargs)
        service.call=declined
        z=torch.randn((1,24,1,4,4),generator=torch.Generator().manual_seed(733)).half()
        expected=baseline.decode(z).cpu();actual=wrapper.decode(z).cpu()
        assert torch.equal(actual,expected) and wrapper.last_strategy=='native'
        report['records'].append(dict(component='video_native_admission_fallback',byte_exact=True,
            strategy=wrapper.last_strategy,reason=wrapper.last_fallback,controlled_eligibility=True));save()
        service.call=logged
        video.release();assert service.request_idle()
        service.idle_vae_cancel.set()
        if service.idle_vae_thread is not None:service.idle_vae_thread.join(120)
        service.pause_idle(keep=False,pressure=True)
        assert video.stage is None
        pressure=service.call('idle_stats');assert all(r['live_bytes']==0 for r in pressure)
        report['records'].append(dict(component='idle_cancel_and_pressure_release',status='passed',worker_idle=pressure));save()
        management.unload_model_and_clones(baseline.patcher)
        del baseline,video,wrapper,expected,actual,z;gc.collect();management.soft_empty_cache()
        audio=StagedVAE(assets/'audio_vae.safetensors',service)
        native=native_vae(assets/'audio_vae.safetensors',torch.float32)
        waveform=torch.rand(1,1600,2,generator=torch.Generator().manual_seed(419))*2-1
        expected=native.encode(waveform.clone());actual=audio.encode(waveform.clone())
        assert torch.equal(actual,expected)
        report['records'].append(dict(component='staged_audio_encode',byte_exact=True,
            dtype=str(actual.dtype),latent_shape=list(actual.shape)));save()
        samples=torch.randn((1,32,2,16),generator=torch.Generator().manual_seed(20261021))
        expected=native.decode(samples).cpu()
        before=len(commands)
        for mode in ('auto','native'):
            wrapper=ParallelDecodeHandoff(audio,service,assets/'audio_vae.safetensors',mode)
            actual=wrapper.decode(samples_in=samples).cpu()
            assert torch.equal(actual,expected) and torch.isfinite(actual).all()
            assert wrapper.last_strategy=='native'
            report['records'].append(dict(component='native_single_gpu_audio',mode=mode,byte_exact=True,
                dtype=str(actual.dtype),output_shape=list(actual.shape)));save()
        assert not any(c in ('vae_prepare','audio_channels','video_pixels') for c in commands[before:])
        audio.release();management.unload_model_and_clones(native.patcher)
        report['worker_stats']=service.call('stats')
        assert all(r['vae_phase'] is None for r in report['worker_stats'])
        report['commands']=commands
        if failures:raise AssertionError('Video output identity failed: '+', '.join(failures))
    finally:
        for staged in tuple(service.staged_vaes):staged.release()
        service.close();report['budget_after_close']=service.budget.snapshot()
        assert report['budget_after_close']['reserved_bytes']==0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=['refiner','vae'])
    parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=sys.argv[:1]
    import torch
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in (0,1)):
        raise RuntimeError('Two actual T4s required')
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(ROOT/'ComfyUI')
    torch.cuda.set_device(0);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    report=dict(status='running',scope='Bounded trained components and physical idle/decoder ownership; no TE tokenizer, sampler or full H3 generation',records=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2))
    try:
        with torch.inference_mode():
            if args.part=='refiner':refiner(args.assets,report)
            else:vae(args.assets,report,save)
        report['status']='passed'
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
