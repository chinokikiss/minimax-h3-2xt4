"""One full trained eight-hop PTD request, using existing Native H3 adapters."""
import argparse
from dataclasses import asdict
import gc
import gzip
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'ComfyUI'), str(ROOT/'benchmarks')]
import torch
import ref2va_e2e_run as observation
from run_h3_l1_turbo_e2e import CurrentService, worker, summaries, all_finite
from benchmarks.e2e_attention_validation import kitchen_denoiser_calls
from benchmarks.ptd_e2e_contract import validate
from patches import comfy_tp_runtime as rt
from patches.host_weight_cache import GiB


class PTDService(CurrentService):
    def __init__(self, config):
        self.hop_states = []
        super().__init__(config)

    def call(self, command, *args, **kwargs):
        result = super().call(command, *args, **kwargs)
        if command == 'denoise':
            self.hop_states.append(super().call('ptd_state'))
        return result


def runtime_config(config):
    validate(config)
    # Preserve the proven long-sequence GPU reserve. CPU cache must fit the
    # ~3 GiB refiner group plus a block; the normal 3 GiB cache cannot do both.
    return rt.RuntimeConfig(config['paths']['te'], config['paths']['dit'],
        lora_stack=(), bundle_offline_lora=False, cache_bytes=4*GiB,
        host_bytes=20*GiB, reserve_bytes=9*GiB, timeout_seconds=1800)


def validate_bank(bank):
    from patches.ptd.checkpoint import RELEASE
    if (bank.layers != 50 or len(bank.hops) != 8
            or bank.recipe['checkpoint_sha256'] != RELEASE['sha256']
            or (bank.recipe['video_shift'], bank.recipe['audio_shift']) != (12.,3.)):
        raise ValueError('Full PTD bank and trained clock required')
    counts = [len(receipt['native_verified_targets']) for _, receipt in bank.hops]
    if counts != [208]*8:
        raise ValueError('Every full expert needs 208 independent Native merge proofs')
    return dict(bank_sha256=bank.digest, recipe_sha256=bank.recipe_sha,
        verified_targets_by_hop=counts, layers=50, hops=8, auxiliary_layout='monolithic',online_lora=False)


def ptd_options(config):
    from patches.ptd.runtime import PTDOptions
    validate(config)
    return PTDOptions(enabled=True,allow_pruned_adaln=True,hop_handoff='queue',
        future_cache=config['future_cache'],prefetch_refiner=config.get('prefetch_refiner',True))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(); config=json.loads(args.config.read_text()); sys.argv=sys.argv[:1]
    validate(config)
    output=Path(config['output']); output.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',config=config,stages={},denoiser_outputs=[],sampler_steps=[])
    def save(): (output/'result.json').write_text(json.dumps(state,indent=2,default=str))
    save()
    if (torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5)
            or 'T4' not in torch.cuda.get_device_name(i) for i in (0,1))):
        raise RuntimeError('Two physical T4 devices required')
    torch.set_grad_enabled(False); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages(); validate_source(ROOT/'ComfyUI')
    from patches.comfy_sampler import create_clip,create_model,ParallelDecodeHandoff
    from patches.ptd.bank import PTDBank
    from patches.ptd.runtime import adapt_model
    from patches.ptd.sampler import make_sampler
    from patches.staged_vae import StagedVAE
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo,MiniMaxH3SigmaShift
    from comfy.nested_tensor import NestedTensor
    from PIL import Image
    import comfy.samplers
    import comfy.model_management as management
    import numpy as np
    bank=PTDBank(config['paths']['bank'],expected_sha256=config['bank_sha256'])
    state['bank']=validate_bank(bank)
    runtime=runtime_config(config); state['runtime']=asdict(runtime)
    rt._worker=worker
    service=None; vaes={}; monitor=observation.Monitor()
    def stage(name,fn):
        state['status']=name; save(); start=time.monotonic_ns()
        print('PTD_E2E_STAGE_START',name,flush=True)
        try:
            result=fn(); torch.cuda.synchronize(0)
            state['stages'][name]=dict(start_ns=start,end_ns=time.monotonic_ns(),status='complete')
            return result
        except BaseException:
            state['stages'][name]=dict(start_ns=start,end_ns=time.monotonic_ns(),status='failed'); raise
        finally:
            save(); print('PTD_E2E_STAGE_END',name,state['stages'][name],flush=True)
    def denoised(record):
        state['denoiser_outputs'].append(record); save()
        print('PTD_E2E_DENOISER_COMPLETE',len(state['denoiser_outputs']),flush=True)
        if not all_finite(record): raise FloatingPointError('Nonfinite denoiser output')
    def callback(step,x0,x,total):
        record=dict(step=step,total=total,denoised=summaries(x0),latent=summaries(x))
        state['sampler_steps'].append(record)
        state['ptd_hops']=service.hop_states; save()
        print('PTD_E2E_SAMPLER_STEP',step+1,total,flush=True)
        if not all_finite(record['denoised']) or not all_finite(record['latent']):
            raise FloatingPointError('Nonfinite PTD sampler state')
    try:
        service=PTDService(runtime); service.on_denoise=denoised
        clip=create_clip(service); model=create_model(service)
        model=MiniMaxH3SigmaShift.execute(model,12.,3.)[0]
        options=ptd_options(config)
        model,controller=adapt_model(model,service,bank.path,options,expected_bank_sha256=bank.digest)
        state['ptd_options']=asdict(options); state['trigger_ns']=time.monotonic_ns()
        images={f'ref_image_{i+1}':torch.from_numpy(np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255)[None]
                for i,path in enumerate(config['images'])}
        vaes=stage('vae_metadata',lambda:{k:StagedVAE(config['paths'][k],service) for k in ('video','audio')})
        assert all(v.stage is None and v.reader.logical_read_bytes==0 for v in vaes.values())
        positive,latent=stage('reference_and_TE',lambda:tuple(MiniMaxH3ReferenceToVideo.execute(
            clip,config['prompt'],config['width'],config['height'],config['frames'],
            ref_image_size='match',vae=vaes['video'],ref_images=images))[:2])
        assert vaes['audio'].stage is None,'Image references must not load Audio encoder'
        latent=latent['samples']; generator=torch.Generator().manual_seed(config['seed'])
        noise=NestedTensor([torch.randn(t.shape,generator=generator) for t in latent.unbind()])
        state['inputs']={k:observation.digest(t) for k,t in [('conditioning',positive[0][0])]+
                        [(f'noise_{i}',t) for i,t in enumerate(noise.unbind())]}
        guider=comfy.samplers.CFGGuider(model); guider.set_conds(positive,positive); guider.set_cfg(1.)
        state['sigmas']=controller.sigmas.tolist(); save()
        sampled=stage('sampler',lambda:guider.sample(noise,latent,make_sampler(controller),
            controller.sigmas.float(),callback=callback,disable_pbar=False,seed=config['seed']).cpu())
        assert all(v.stage is None for v in vaes.values()),'Release encoders before denoising'
        assert [h['calls'] for h in service.hop_states]==[list(range(i+1)) for i in range(8)]
        for key in ('window_id','shell_id'):
            assert len({h['queue'][key] for h in service.hop_states})==1,'Queue was rebuilt between hops'
        video,audio=sampled.unbind()
        torch.save(dict(video=video.cpu(),audio=audio.cpu()),output/'decode_inputs.pt')
        from patches.tp2_bundle import sha
        state['latent_checkpoint_sha256']=sha(output/'decode_inputs.pt'); save()
        decoders={k:ParallelDecodeHandoff(v,service,config['paths'][k],'temporal' if k=='video' else 'native')
                  for k,v in vaes.items()}
        pixels=stage('video_decode',lambda:decoders['video'].decode(video).cpu())
        waveform=stage('audio_decode',lambda:decoders['audio'].decode(audio).cpu())
        assert decoders['video'].last_strategy=='temporal' and not decoders['video'].last_fallback
        assert decoders['audio'].last_strategy=='native'
        assert list(pixels.shape[-4:-1])==[config['frames'],config['height'],config['width']]
        state['decoded_shapes']=dict(pixels=list(pixels.shape),waveform=list(waveform.shape))
        state['media']=stage('file_export',lambda:observation.export_media(pixels,waveform,output))
        state['foreground_seconds']=(time.monotonic_ns()-state['trigger_ns'])/1e9
        state['outputs']={k:observation.digest(t) for k,t in dict(video=video,audio=audio,pixels=pixels,waveform=waveform).items()}
        if not all(v['finite'] for v in state['outputs'].values()): raise FloatingPointError('Nonfinite final output')
        state['decode']={k:dict(strategy=d.last_strategy,fallback=d.last_fallback) for k,d in decoders.items()}
        state['vae_events']={k:v.events for k,v in vaes.items()}
        state['ranks']=service.call('stats'); state['service_calls']=service.spans
        assert len(state['sampler_steps'])==len(state['denoiser_outputs'])==8
        for rank in state['ranks']:
            bench=rank['benchmark']; calls=bench['executed_backends']
            assert kitchen_denoiser_calls(bench)==400
            assert calls.get('TE/kitchen_causal') and calls.get('TE/column_view')==250
            assert calls.get('current/fused_swiglu_quant') and calls.get('current/scaled_fp16_epilogue')
            assert not any(v for k,v in calls.items() if k.startswith('LoRA/'))
            assert any(row['sequence_tokens']==config['expected_packed_tokens'] for row in bench['layouts'])
            assert max(c['peak_allocated'] for c in bench['commands'] if c['command']=='denoise')<14.5*GiB
        state.update(status='complete',quality_acceptance='Pending visual/audio review; pruned-base approximation, no quality-equivalence claim')
    except BaseException as error:
        state.update(status='failed',error=repr(error)); (output/'exception.txt').write_text(traceback.format_exc())
        if service and not service.closed:
            try: state['ranks']=service.call('stats')
            except Exception as capture_error: state['failure_metrics_error']=repr(capture_error)
        raise
    finally:
        if service: state['ptd_hops']=service.hop_states
        for vae in vaes.values(): vae.release()
        if service: service.close()
        vaes.clear(); gc.collect(); management.unload_all_models(); torch.cuda.empty_cache(); monitor.close()
        with gzip.open(output/'memory_samples.json.gz','wt') as stream: json.dump(monitor.rows,stream)
        save()


if __name__=='__main__': main()
