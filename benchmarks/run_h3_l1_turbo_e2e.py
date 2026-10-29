"""One explicit Turbo REF2VA request with current defaults and disk-baked TP2 weights."""
import argparse
from dataclasses import asdict
import gc
import gzip
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import ref2va_e2e_run as observation
from patches import comfy_tp_runtime as rt
from patches.h3_operator_policy import H3OperatorPolicy
from patches.h3_tensor_flow import TensorFlowPolicy
from patches.host_weight_cache import GiB
from benchmarks.e2e_attention_validation import kitchen_denoiser_calls


class CurrentRank(observation.ObservedRank):
    def _install_observers(self):
        super()._install_observers()
        from patches import tilelang_swiglu_quant as fused,tilelang_transport_p2 as codec,tilelang_dit_fusion as dit_fusion
        for module,name,label in ((fused,'prepare','current/fused_swiglu_quant'),
                (codec,'finish','current/scaled_fp16_epilogue'),
                (dit_fusion,'finish_row','current/scaled_fp16_epilogue')):
            original=getattr(module,name)
            def observe(*args,_fn=original,_label=label,**kwargs):
                self.backends[_label]+=1
                return _fn(*args,**kwargs)
            setattr(module,name,observe)
        from patches import dit_column_linear as column
        original=column._column_accumulate
        def column_accumulate(codes,weight):
            if column._column_view_eligible(codes,weight):self.backends['TE/column_view']+=1
            return original(codes,weight)
        column._column_accumulate=column_accumulate

    def __init__(self,*args):
        super().__init__(*args)
        assert self.config.bundle_offline_lora or not self.config.lora_stack,'Offline checkpoint must not attach online LoRA'
        assert not any(getattr(r,'lora_store',None) for r in self.readers.values())

    def dispatch(self,command,args=(),kwargs=None):
        result=super().dispatch(command,args,kwargs)
        if command=='stats':
            result['benchmark']['tensor_flow']=asdict(self.config.tensor_flow)
            result['benchmark']['operators']=asdict(self.config.operators)
            result['benchmark']['online_lora_attached']=False
        return result


def worker(*args):
    rt.RankRuntime=CurrentRank
    rt.ResidencyWindow=observation.ObservedWindow
    observation.BASE_WORKER(*args)


def summaries(value):
    """Compact CPU summaries of already delivered sampler tensors, no payload retention."""
    if isinstance(value,torch.Tensor):
        x=value.detach().float().cpu()
        return dict(shape=list(value.shape),dtype=str(value.dtype),finite=bool(torch.isfinite(x).all()),
            minimum=float(x.min()),maximum=float(x.max()),mean=float(x.mean()),rms=float(x.square().mean().sqrt()))
    if hasattr(value,'unbind') and not isinstance(value,(list,tuple)):
        value=value.unbind()
    if isinstance(value,(list,tuple)):return [summaries(x) for x in value]
    if isinstance(value,dict):return {k:summaries(v) for k,v in value.items()}
    raise TypeError(type(value).__name__)


def all_finite(value):
    if isinstance(value,list):return all(all_finite(v) for v in value)
    return value['finite']


class CurrentService(observation.Service):
    def call(self,command,*args,**kwargs):
        result=super().call(command,*args,**kwargs)
        if command=='denoise' and hasattr(self,'on_denoise'):
            self.on_denoise(summaries(result))
        return result


def runtime_config(config):
    if not config.get('offline_lora') or not config.get('disk_baked_lora'):
        raise ValueError('This long E2E requires persisted offline-baked TP2 weights')
    if config['width']%32 or config['height']%32:
        raise ValueError('Use 32-pixel-aligned dimensions')
    return rt.RuntimeConfig(te_path=config['paths']['te'],dit_path=config['paths']['dit'],
        lora_stack=(),bundle_offline_lora=False,
        operators=H3OperatorPolicy(),tensor_flow=TensorFlowPolicy('fp16'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());sys.argv=sys.argv[:1]
    if config.get('workload_contract')=='anime_ref2va_8step':
        from benchmarks.anime_e2e_contract import validate
        validate(config,config['reference_images'])
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',config=config,stages={},denoiser_outputs=[],sampler_steps=[])
    def save():(output/'result.json').write_text(json.dumps(state,indent=2,default=str))
    save()
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in (0,1))
    torch.set_grad_enabled(False);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(ROOT/'ComfyUI')
    from patches.comfy_sampler import create_clip,create_model,ParallelDecodeHandoff
    from patches.staged_vae import StagedVAE
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo,MiniMaxH3SigmaShift
    from comfy.nested_tensor import NestedTensor
    from PIL import Image
    import comfy.samplers
    import comfy.model_management as management
    import numpy as np
    runtime=runtime_config(config);state['runtime']=asdict(runtime)
    rt._worker=worker
    service=None;vaes={};monitor=observation.Monitor()
    def stage(name,fn):
        state['status']=name;save();start=time.monotonic_ns()
        print('L1_STAGE_START',name,flush=True)
        try:
            result=fn();torch.cuda.synchronize(0)
            state['stages'][name]=dict(start_ns=start,end_ns=time.monotonic_ns(),status='complete')
            return result
        except BaseException:
            state['stages'][name]=dict(start_ns=start,end_ns=time.monotonic_ns(),status='failed');raise
        finally:save();print('L1_STAGE_END',name,state['stages'][name],flush=True)
    def denoised(record):
        state['denoiser_outputs'].append(record);save()
        print('L1_DENOISER_COMPLETE',len(state['denoiser_outputs']),record,flush=True)
        if not all_finite(record):raise FloatingPointError('Nonfinite denoiser output')
    def callback(step,x0,x,total):
        record=dict(step=step,total=total,denoised=summaries(x0),latent=summaries(x))
        state['sampler_steps'].append(record);save();print('L1_SAMPLER_STEP',step+1,total,flush=True)
        if not all_finite(record['denoised']) or not all_finite(record['latent']):raise FloatingPointError('Nonfinite sampler state')
    try:
        setup=time.monotonic();service=CurrentService(runtime);service.on_denoise=denoised
        clip=create_clip(service);model=create_model(service)
        model=MiniMaxH3SigmaShift.execute(model,config['video_shift'],config['audio_shift'])[0]
        images={f'ref_image_{i+1}':torch.from_numpy(np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255)[None]
            for i,path in enumerate(config['images'])}
        state['setup_seconds']=time.monotonic()-setup;state['trigger_ns']=time.monotonic_ns()
        vaes=stage('vae_metadata',lambda:{k:StagedVAE(config['paths'][k],service) for k in ('video','audio')})
        assert all(v.stage is None and v.reader.logical_read_bytes==0 for v in vaes.values())
        positive,latent=stage('reference_and_TE',lambda:tuple(MiniMaxH3ReferenceToVideo.execute(
            clip,config['prompt'],config['width'],config['height'],config['frames'],ref_image_size='match',
            vae=vaes['video'],ref_images=images))[:2])
        assert vaes['audio'].stage is None,'Image-only references must not load the Audio encoder'
        state['vae_before_sampling']={k:dict(stage=v.stage,read_bytes=v.reader.logical_read_bytes) for k,v in vaes.items()}
        latent=latent['samples'];generator=torch.Generator().manual_seed(config['seed'])
        noise=NestedTensor([torch.randn(t.shape,generator=generator) for t in latent.unbind()])
        state['inputs']={k:observation.digest(t) for k,t in [('conditioning',positive[0][0])]+[(f'noise_{i}',t) for i,t in enumerate(noise.unbind())]}
        guider=comfy.samplers.CFGGuider(model);guider.set_conds(positive,positive);guider.set_cfg(1.)
        sigmas=comfy.samplers.calculate_sigmas(model.model.model_sampling,'simple',config['steps'])
        state['sigmas']=sigmas.cpu().tolist();save()
        sampled=stage('sampler',lambda:guider.sample(noise,latent,comfy.samplers.sampler_object(config['sampler']),
            sigmas,callback=callback,disable_pbar=False,seed=config['seed']).cpu())
        assert all(v.stage is None for v in vaes.values()),'Encoders must be released before denoising'
        video,audio=sampled.unbind()
        # Save compact latents before either decoder, including successful runs,
        # to recover a decode/export failure without repeating the long sampler.
        torch.save(dict(video=video.cpu(),audio=audio.cpu()),output/'decode_inputs.pt')
        decoders={k:ParallelDecodeHandoff(v,service,config['paths'][k],'auto') for k,v in vaes.items()}
        pixels=stage('video_decode',lambda:decoders['video'].decode(video).cpu())
        waveform=stage('audio_decode',lambda:decoders['audio'].decode(audio).cpu())
        assert decoders['audio'].last_strategy=='native'
        if config.get('workload_contract')=='anime_ref2va_8step':
            shape=list(pixels.shape[-4:])
            assert shape[:3]==[config['frames'],config['height'],config['width']], 'Requested video shape changed'
        state['decoded_shapes']=dict(pixels=list(pixels.shape),waveform=list(waveform.shape))
        state['media']=stage('file_export',lambda:observation.export_media(pixels,waveform,output))
        state['foreground_seconds']=(time.monotonic_ns()-state['trigger_ns'])/1e9
        state['outputs']={k:observation.digest(t) for k,t in dict(video=video,audio=audio,pixels=pixels,waveform=waveform).items()}
        if not all(v['finite'] for v in state['outputs'].values()):raise FloatingPointError('Nonfinite final output')
        state['decode']={k:dict(strategy=d.last_strategy,fallback=d.last_fallback) for k,d in decoders.items()}
        state['vae_events']={k:v.events for k,v in vaes.items()}
        state['ranks']=service.call('stats');state['service_calls']=service.spans
        assert len(state['sampler_steps'])==config['steps'] and len(state['denoiser_outputs'])==config['steps']
        for rank in state['ranks']:
            bench=rank['benchmark'];calls=bench['executed_backends']
            assert bench['layouts'] and calls.get('TE/kitchen_causal')
            assert kitchen_denoiser_calls(bench) == 50*config['steps']
            assert rank['bundle_preparation']['dit']['disk_baked']
            assert rank['bundle_preparation']['dit']['merged_matrices']==0
            assert calls.get('current/fused_swiglu_quant') and calls.get('current/scaled_fp16_epilogue')
            assert not any(v for k,v in calls.items() if k.startswith('LoRA/'))
            assert max(c['peak_allocated'] for c in bench['commands'] if c['command']=='denoise')<14.5*GiB
            if config.get('workload_contract')=='anime_ref2va_8step':
                assert calls.get('TE/column_view')==250, 'Expected five optimized columns in each of 50 TE layers'
        state.update(status='complete',quality_acceptance='Pending visual/audio review; no dense Native comparison or generation-equivalence claim')
    except BaseException as error:
        state.update(status='failed',error=repr(error));(output/'exception.txt').write_text(traceback.format_exc())
        if service and not service.closed:
            try:state['ranks']=service.call('stats')
            except Exception as capture_error:state['failure_metrics_error']=repr(capture_error)
        raise
    finally:
        for vae in vaes.values():vae.release()
        if service:service.close()
        vaes.clear();gc.collect();management.unload_all_models();torch.cuda.empty_cache();monitor.close()
        with gzip.open(output/'memory_samples.json.gz','wt') as stream:json.dump(monitor.rows,stream)
        save()


if __name__=='__main__':main()
