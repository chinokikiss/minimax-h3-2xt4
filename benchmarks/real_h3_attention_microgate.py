"""Capture one real H3 post-RoPE DiT attention and replay Sol variants in-place."""
from __future__ import annotations
import argparse,hashlib,json,math,os,statistics,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import torch.nn.functional as F
import ref2va_e2e_run as observation
import run_h3_l1_turbo_e2e as current
from patches import comfy_tp_runtime as rt

class CaptureStop(Exception):pass
class ParentCaptureComplete(Exception):pass

def tensor_stats(x):
    y=x.float()
    return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),
        finite=bool(torch.isfinite(y).all()),mean=float(y.mean()),
        rms=float(y.square().mean().sqrt()),minimum=float(y.min()),maximum=float(y.max()))

def error_metrics(actual,reference):
    a=actual.float();r=reference.float();d=a-r
    denom=torch.linalg.vector_norm(r).clamp_min(1e-12)
    return dict(rel_l2=float(torch.linalg.vector_norm(d)/denom),
        mae=float(d.abs().mean()),max_abs=float(d.abs().max()),
        rms=float(d.square().mean().sqrt()),
        cosine=float(F.cosine_similarity(a.flatten(),r.flatten(),dim=0)),
        finite=bool(torch.isfinite(a).all()))

def subset_metrics(actual,reference,segments):
    out={'all':error_metrics(actual,reference)}
    for name,start,end in segments:
        if end>start:out[name]=error_metrics(actual[:,:,start:end],reference[:,:,start:end])
    return out

def frame_metrics(actual,reference,video_start,video_end,frame_tokens):
    rows=[]
    for start in range(video_start,video_end,frame_tokens):
        end=min(start+frame_tokens,video_end)
        rows.append(error_metrics(actual[:,:,start:end],reference[:,:,start:end])['rel_l2'])
    return dict(count=len(rows),mean=float(statistics.mean(rows)),maximum=float(max(rows)),
        minimum=float(min(rows)),per_frame_rel_l2=rows)

def measure(fn,warmup=2,repeats=4):
    for _ in range(warmup):fn()
    torch.cuda.synchronize()
    gpu=[];wall=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        begin=time.perf_counter();a.record();fn();b.record();b.synchronize()
        wall.append((time.perf_counter()-begin)*1000);gpu.append(a.elapsed_time(b))
    return dict(repeats=repeats,gpu_median_ms=float(statistics.median(gpu)),
        wall_median_ms=float(statistics.median(wall)),gpu_ms=gpu,wall_ms=wall)

def peak_temp(fn):
    torch.cuda.synchronize();base=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats();fn();torch.cuda.synchronize()
    return int(max(0,torch.cuda.max_memory_allocated()-base))


def profile_sol_stages(q,k,v,*,tau,prefix,topology,expected):
    """GPU-stage timings for the unchanged Sol v2 output contract.

    This is diagnostic only: route/quantize/execute are independently replayed
    on the identical captured QKV.  Warmup and allocation state make stage
    medians non-additive; only the complete-call timing is a throughput claim.
    """
    from patches import sol_sm75_backend as native
    from patches import sol_attention as sol
    options=dict(protected_tokens=prefix,protected_start=0,tau=tau,
        allow_d64=q.shape[-1]==64,topology=topology,
        sync_free_lut=True,collect_route_stats=False)
    prepare=lambda:sol.prepare_attention(q,k,v,**options)
    route=lambda:native.prepare(q,k,v,tau=tau,sink_tokens=prefix,
        sink_start=0 if prefix else None,quantize_qk=True,topology=topology,
        sync_free_lut=True,collect_route_stats=False)
    # Release each profiling owner's buffers before retaining the next set.
    route_ms=measure(route,warmup=1,repeats=3)
    routing=route()
    # Frozen pre-math-fix Sol uses K-centroid quantization grouped by SOL_BLOCK (64).
    key_quant=lambda:native.quantize(routing['k_centroid'],64)
    value_quant=lambda:native.prepare_all_int8_values(
        v,routing['v_centroid'],int(routing['centroid_padded']))
    key_ms=measure(key_quant,warmup=1,repeats=3)
    value_ms=measure(value_quant,warmup=1,repeats=3)
    del routing,key_quant,value_quant
    prepare_ms=measure(prepare,warmup=1,repeats=3)
    prepared=prepare()
    prepared_out=sol.execute_prepared(prepared)
    fidelity=error_metrics(prepared_out,expected)
    if fidelity['rel_l2']>1e-6:
        raise RuntimeError('Prepared Sol v2 changed the monolithic output')
    execute=lambda:sol.execute_prepared(prepared)
    execute_ms=measure(execute,warmup=2,repeats=5)
    timings=dict(complete_prepare=prepare_ms,route_and_qk_quant=route_ms,
        k_centroid_quant=key_ms,v_and_centroid_quant=value_ms,
        prepared_execute=execute_ms)
    return dict(timings=timings,prepared_vs_complete=fidelity,
        timing_note='Independent CUDA event medians are not additive')

def routing_stats(q,k,v,*,tau,prefix,topology,sync_free):
    from patches import sol_sm75_backend as native
    routing=native.prepare(q,k,v,tau=float(tau),sink_tokens=int(prefix),
        sink_start=0,quantize_qk=True,topology=topology,
        sync_free_lut=bool(sync_free),collect_route_stats=not sync_free)
    blocks=routing['exact_mask'].shape[-1]
    total=int(routing['row_count'].sum().item())
    rows=routing['row_count'].numel()
    return routing,dict(density=total/(rows*blocks),minimum=int(routing['row_count'].min().item()),
        maximum=int(routing['row_count'].max().item()),blocks=blocks)

def centroid_norm_retention(k,video_start,video_end):
    b,h,n,d=k.shape;block=64;nb=(n+block-1)//block
    pad=nb*block-n
    x=F.pad(k.float(),(0,0,0,pad)).reshape(b,h,nb,block,d)
    counts=(n-torch.arange(nb,device=k.device)*block).clamp(0,block).float()
    cent=x.sum(-2)/counts.view(1,1,nb,1)
    mean_norm=x.norm(dim=-1).sum(-1)/counts.view(1,1,nb)
    ratio=cent.norm(dim=-1)/mean_norm.clamp_min(1e-12)
    starts=torch.arange(nb,device=k.device)*block
    video=(starts<video_end)&(starts+block>video_start)
    boundary=[]
    frame_tokens=(video_end-video_start)//72 if (video_end-video_start)%72==0 else None
    if frame_tokens:
        for bi in range(nb):
            s=bi*block;e=min((bi+1)*block,n)
            if s<video_end and e>video_start:
                a=max(s,video_start)-video_start
                z=min(e,video_end)-1-video_start
                if a//frame_tokens!=z//frame_tokens:boundary.append(bi)
    def summarize(values):
        values=values.flatten()
        return dict(mean=float(values.mean()),median=float(values.median()),
            p10=float(torch.quantile(values,.1)),p90=float(torch.quantile(values,.9)),
            minimum=float(values.min()),maximum=float(values.max()),count=values.numel())
    result=dict(all=summarize(ratio),video=summarize(ratio[:,:,video]))
    if boundary:result['frame_boundary_blocks']=summarize(ratio[:,:,torch.tensor(boundary,device=k.device)])
    return result

def temporal_recall(mask,video_start,video_end,frame_tokens,radius=1):
    # Exact-mask recall for every K64 block intersecting the true token-space
    # interval of q shifted by +/- frame_tokens.
    m=mask.detach().cpu();heads=m.shape[1];blocks=m.shape[-1]
    required=[]
    for qb in range(blocks):
        q0=max(qb*64,video_start);q1=min((qb+1)*64,video_end)
        if q0>=q1:continue
        for dt in range(1,radius+1):
            for sign in (-1,1):
                t0=q0+sign*dt*frame_tokens;t1=q1+sign*dt*frame_tokens
                if t1<=video_start or t0>=video_end:continue
                t0=max(t0,video_start);t1=min(t1,video_end)
                kb0=max(0,t0//64);kb1=min(blocks-1,(t1-1)//64)
                for kb in range(kb0,kb1+1):required.append((qb,kb))
    required=sorted(set(required))
    if not required:return dict(pairs=0,recall=None)
    hits=0
    for qb,kb in required:hits+=int(m[0,:,qb,kb].sum())
    total=len(required)*heads
    return dict(pairs=len(required),head_pairs=total,hits=hits,recall=hits/total)

def boundary_key_coverage(mask,video_start,video_end,frame_tokens):
    m=mask.detach().cpu();blocks=m.shape[-1];keys=[]
    for bi in range(blocks):
        s=bi*64;e=min((bi+1)*64,video_end)
        if s<video_end and e>video_start:
            a=max(s,video_start)-video_start;z=min(e,video_end)-1-video_start
            if a//frame_tokens!=z//frame_tokens:keys.append(bi)
    if not keys:return dict(blocks=0,coverage=None)
    selected=m[0,:,:,keys]
    return dict(blocks=len(keys),coverage=float(selected.float().mean()))

def microgate(owner,q,k,v,options):
    from patches import h3_operator_policy as policy
    from patches.kitchen_attention import int8_attention as kitchen
    from patches.sol_attention import attention as sol
    rank=owner.rank;outdir=Path(os.environ['H3_REAL_ATTN_CAPTURE_DIR']);outdir.mkdir(parents=True,exist_ok=True)
    layout=options.get('minimax_h3_layout')
    if layout is None or layout.seq_len!=q.shape[-2]:raise ValueError('Real-H3 microgate requires matching PackedLayout')
    segments=[tuple(x) for x in layout.segments]
    old_prefix=policy.protected_prefix(options,q.shape[-2])
    v2_prefix,topology=policy.sol_v2_routing(options,q.shape[-2])
    video_start,video_end,video_kind=segments[-1]
    if video_kind!='video':raise ValueError(segments)
    video_tokens=video_end-video_start
    latent_t=int(layout.signature[1])
    if video_tokens%latent_t:raise ValueError('non-regular video token packing')
    frame_tokens=video_tokens//latent_t
    # Four contiguous local heads preserve production BHLD transpose-view strides.
    hs=slice(0,min(4,q.shape[1]));qs,ks,vs=q[:,hs],k[:,hs],v[:,hs]
    state=dict(status='running',rank=rank,nfe=owner.context.get('nfe'),
        source='real H3 post-RMS/post-RoPE QKV intercepted at prepare_dit_attention',
        q=tensor_stats(qs),k=tensor_stats(ks),v=tensor_stats(vs),
        full_q_shape=list(q.shape),full_q_stride=list(q.stride()),
        layout=dict(sequence_tokens=layout.seq_len,segments=segments,signature=list(layout.signature),
            old_protected_prefix=old_prefix,v2_protected_prefix=v2_prefix,
            video_tokens=video_tokens,latent_t=latent_t,frame_tokens=frame_tokens),
        centroid_norm_retention=centroid_norm_retention(ks,video_start,video_end),candidates={})
    path=outdir/f'rank{rank}.json'
    def save():path.write_text(json.dumps(state,indent=2))
    save()
    # Reference outputs. Kitchen is the trajectory baseline; SDPA is a secondary dense reference.
    kitchen_out=kitchen(qs,ks,vs)
    sdpa_out=F.scaled_dot_product_attention(qs,ks,vs)
    query_segments=[('prefix',0,v2_prefix),('video',video_start,video_end)]
    state['kitchen_vs_sdpa']=subset_metrics(kitchen_out,sdpa_out,query_segments)
    state['kitchen_timing']=measure(lambda:kitchen(qs,ks,vs),warmup=2,repeats=3)
    state['kitchen_peak_temporary']=peak_temp(lambda:kitchen(qs,ks,vs));save()

    candidates=[
        ('old_tau1_compact',1.0,old_prefix,None,False),
        ('old_tau1_sync_free',1.0,old_prefix,None,True),
        ('v2_tau1',1.0,v2_prefix,topology,True),
        ('v2_tau0p7',0.7,v2_prefix,topology,True),
        ('v2_tau0p5',0.5,v2_prefix,topology,True),
    ]
    bitmap_ab=os.environ.get('H3_REAL_SOL_BITMAP_AB','0')=='1'
    if bitmap_ab:
        candidates.append(('v2_bitmap_tau1',1.0,v2_prefix,topology,True))
    baseline_v2_cpu=None
    previous_bitmap_flag=os.environ.get('H3_SOL_LUT_BITMAP')
    for name,tau,prefix,topo,sync_free in candidates:
        # Baseline and bitmap modes share the same captured Q/K/V and compiled
        # shared library. Switch only the opt-in routing representation.
        os.environ['H3_SOL_LUT_BITMAP']='1' if name=='v2_bitmap_tau1' else '0'
        routing,rstats=routing_stats(qs,ks,vs,tau=tau,prefix=prefix,topology=topo,sync_free=sync_free)
        fn=lambda tau=tau,prefix=prefix,topo=topo,sync_free=sync_free:sol(
            qs,ks,vs,protected_tokens=prefix,protected_start=0,tau=tau,
            topology=topo,sync_free_lut=sync_free,collect_route_stats=not sync_free)
        actual=fn()
        # Explicit output-norm retention reveals numerator/denominator scale bias.
        norm_ratios={}
        for label,start,end in [('all',0,qs.shape[-2]),*query_segments]:
            if end>start:
                a=actual[:,:,start:end].float()
                r=kitchen_out[:,:,start:end].float()
                norm_ratios[label]=float(torch.linalg.vector_norm(a)/torch.linalg.vector_norm(r).clamp_min(1e-12))
        row=dict(tau=tau,prefix=prefix,topology=topo,route=rstats,
            output_norm_ratio_vs_kitchen=norm_ratios,
            vs_kitchen=subset_metrics(actual,kitchen_out,query_segments),
            vs_sdpa=subset_metrics(actual,sdpa_out,query_segments),
            video_frame_error=frame_metrics(actual,kitchen_out,video_start,video_end,frame_tokens),
            temporal_exact_recall=temporal_recall(routing['exact_mask'],video_start,video_end,frame_tokens),
            frame_boundary_key_coverage=boundary_key_coverage(routing['exact_mask'],video_start,video_end,frame_tokens),
            timing=measure(fn,warmup=2,repeats=3),peak_temporary=peak_temp(fn))
        # The routing statistics were materialized above; do not hold an
        # additional Q/K/V quantization workspace during stage profiling.
        del routing
        # Keep the performance microscope opt-in to the production Sol v2
        # policy so old-tau and alternate-tau replay contracts stay unchanged.
        if name=='v2_tau1':
            row['pipeline_stages']=profile_sol_stages(
                qs,ks,vs,tau=tau,prefix=prefix,topology=topo,expected=actual)
        if name=='v2_tau1' and bitmap_ab:
            baseline_v2_cpu=actual.detach().cpu().clone()
        if name=='v2_bitmap_tau1':
            if baseline_v2_cpu is None:
                raise RuntimeError('Bitmap A/B baseline output is missing')
            row['vs_v2_tau1']=error_metrics(actual.detach().cpu(),baseline_v2_cpu)
            row['semantics_gate']=bool(
                row['vs_v2_tau1']['finite'] and
                row['vs_v2_tau1']['rel_l2'] <= 1.0e-6 and
                row['vs_v2_tau1']['cosine'] >= 0.9999)
            if not row['semantics_gate']:
                state['candidates'][name]=row;save()
                raise AssertionError('Real-H3 bitmap LUT changed Sol v2 output')
        state['candidates'][name]=row;save()
        del actual
    if previous_bitmap_flag is None:
        os.environ.pop('H3_SOL_LUT_BITMAP',None)
    else:
        os.environ['H3_SOL_LUT_BITMAP']=previous_bitmap_flag
    # Compact summary for qualification decisions.
    state['summary']=dict(
        old_rel_l2=state['candidates']['old_tau1_sync_free']['vs_kitchen']['all']['rel_l2'],
        v2_tau1_rel_l2=state['candidates']['v2_tau1']['vs_kitchen']['all']['rel_l2'],
        v2_tau0p7_rel_l2=state['candidates']['v2_tau0p7']['vs_kitchen']['all']['rel_l2'],
        v2_tau0p5_rel_l2=state['candidates']['v2_tau0p5']['vs_kitchen']['all']['rel_l2'],
        old_density=state['candidates']['old_tau1_sync_free']['route']['density'],
        v2_tau1_density=state['candidates']['v2_tau1']['route']['density'],
        old_temporal_recall=state['candidates']['old_tau1_sync_free']['temporal_exact_recall']['recall'],
        v2_temporal_recall=state['candidates']['v2_tau1']['temporal_exact_recall']['recall'])
    state['status']='complete';save()

class CaptureRank(current.CurrentRank):
    def _install_observers(self):
        super()._install_observers()
        import patches.h3_operator_policy as policy
        original=policy.prepare_dit_attention
        owner=self
        def capture(block,q,k,v,options=None):
            if owner.context.get('command')=='denoise' and not getattr(owner,'_real_h3_captured',False):
                owner._real_h3_captured=True
                owner.record_layout(q,options or {})
                microgate(owner,q,k,v,options or {})
                raise CaptureStop('real H3 attention captured')
            return original(block,q,k,v,options)
        policy.prepare_dit_attention=capture
    def dispatch(self,command,args=(),kwargs=None):
        if command=='denoise':
            try:return super().dispatch(command,args,kwargs)
            except CaptureStop:
                return {'_real_h3_capture_complete':True,'rank':self.rank}
        return super().dispatch(command,args,kwargs)

def worker(*args):
    rt.RankRuntime=CaptureRank
    rt.ResidencyWindow=observation.ObservedWindow
    observation.BASE_WORKER(*args)

class CaptureService(observation.Service):
    def call(self,command,*args,**kwargs):
        result=super().call(command,*args,**kwargs)
        if command=='denoise' and isinstance(result,dict) and result.get('_real_h3_capture_complete'):
            raise ParentCaptureComplete()
        return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    args=ap.parse_args();config=json.loads(args.config.read_text());sys.argv=sys.argv[:1]
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',scope='real-H3 first-attention capture/replay only; no denoise trajectory or media generation',
        config=config,stages={})
    def save():(output/'result.json').write_text(json.dumps(state,indent=2,default=str))
    save();os.environ['H3_REAL_ATTN_CAPTURE_DIR']=str(output/'rank_results')
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in (0,1))
    torch.set_grad_enabled(False);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(ROOT/'ComfyUI')
    from patches.comfy_sampler import create_clip,create_model
    from patches.staged_vae import StagedVAE
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo,MiniMaxH3SigmaShift
    from comfy.nested_tensor import NestedTensor
    from PIL import Image
    import comfy.samplers
    import comfy.model_management as management
    import numpy as np
    runtime=current.runtime_config(config);state['runtime']=current.asdict(runtime)
    rt._worker=worker;service=None;video_vae=None
    def stage(name,fn):
        begin=time.monotonic();state['status']=name;save()
        try:
            value=fn();state['stages'][name]=dict(seconds=time.monotonic()-begin,status='complete');return value
        except ParentCaptureComplete:
            state['stages'][name]=dict(seconds=time.monotonic()-begin,status='capture_complete');raise
        except BaseException:
            state['stages'][name]=dict(seconds=time.monotonic()-begin,status='failed');raise
        finally:save()
    try:
        service=CaptureService(runtime);clip=create_clip(service);model=create_model(service)
        model=MiniMaxH3SigmaShift.execute(model,config['video_shift'],config['audio_shift'])[0]
        images={f'ref_image_{i+1}':torch.from_numpy(np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255)[None]
            for i,path in enumerate(config['images'])}
        video_vae=StagedVAE(config['paths']['video'],service)
        positive,latent=stage('reference_and_TE',lambda:tuple(MiniMaxH3ReferenceToVideo.execute(
            clip,config['prompt'],config['width'],config['height'],config['frames'],ref_image_size='match',
            vae=video_vae,ref_images=images))[:2])
        latent=latent['samples'];gen=torch.Generator().manual_seed(config['seed'])
        noise=NestedTensor([torch.randn(t.shape,generator=gen) for t in latent.unbind()])
        state['inputs']={k:observation.digest(t) for k,t in [('conditioning',positive[0][0])]+
            [(f'noise_{i}',t) for i,t in enumerate(noise.unbind())]}
        guider=comfy.samplers.CFGGuider(model);guider.set_conds(positive,positive);guider.set_cfg(1.)
        sigmas=comfy.samplers.calculate_sigmas(model.model.model_sampling,'simple',config['steps'])
        state['sigmas']=sigmas.cpu().tolist();save()
        try:
            stage('first_denoise_capture',lambda:guider.sample(noise,latent,
                comfy.samplers.sampler_object(config['sampler']),sigmas,
                disable_pbar=True,seed=config['seed']))
            raise AssertionError('Sampler returned without capture sentinel')
        except ParentCaptureComplete:
            pass
        ranks=[]
        for rank in (0,1):
            path=output/'rank_results'/f'rank{rank}.json'
            if not path.is_file():raise FileNotFoundError(path)
            row=json.loads(path.read_text())
            if row.get('status')!='complete':raise RuntimeError(f'rank{rank} microgate incomplete')
            ranks.append(row)
        state['ranks']=ranks
        state['summary']=dict(
            old_rel_l2=[r['summary']['old_rel_l2'] for r in ranks],
            v2_tau1_rel_l2=[r['summary']['v2_tau1_rel_l2'] for r in ranks],
            v2_tau0p7_rel_l2=[r['summary']['v2_tau0p7_rel_l2'] for r in ranks],
            v2_tau0p5_rel_l2=[r['summary']['v2_tau0p5_rel_l2'] for r in ranks],
            old_density=[r['summary']['old_density'] for r in ranks],
            v2_tau1_density=[r['summary']['v2_tau1_density'] for r in ranks],
            old_temporal_recall=[r['summary']['old_temporal_recall'] for r in ranks],
            v2_temporal_recall=[r['summary']['v2_temporal_recall'] for r in ranks])
        state['status']='complete';save()
    except BaseException as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc());save();raise
    finally:
        if video_vae is not None:
            try:video_vae.release()
            except Exception:pass
        if service is not None:
            try:service.close()
            except Exception:pass
        try:management.unload_all_models()
        except Exception:pass
        save()

if __name__=='__main__':main()
