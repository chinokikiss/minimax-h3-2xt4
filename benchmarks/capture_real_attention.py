"""Native REF2VA trajectory capture; no attention approximation in baseline."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


class AttentionCapture:
    def __init__(self,config,output,mode='native',policy=None):
        self.config=config;self.output=output;self.component=None;self.layer=-1
        self.call=-1;self.timestep=None;self.events=[];self.records=[]
        self.model_calls=[];self.patch=[];self.seen=set()
        self.mode=mode;self.policy=policy or {};self.routes_profile=[]
        self.video_call=-1
        (output/'qkv').mkdir(parents=True,exist_ok=True)

    @contextmanager
    def scope(self,component,layer=-1):
        old=self.component,self.layer;self.component,self.layer=component,layer
        try:yield
        finally:self.component,self.layer=old

    def wrap(self,original):
        import torch
        def attention(q,k,v,heads,*args,**kwargs):
            tensors=[x.peek() if hasattr(x,'peek') else x for x in (q,k,v)]
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            if self.mode!='native' and self.component in ('te','dit','video') and self.layer>=0:
                from patches.tilelang_sparse_attention import prepare,make_routes,execute
                if not kwargs.get('skip_reshape'):raise ValueError('BHND attention required')
                p=prepare(*[t.half() for t in tensors],enable_gqa=kwargs.get('enable_gqa',False))
                routes=make_routes(p,'dense')
                causal=self.component=='te';pooled=False
                if self.mode in ('adaptive','sparse') and self.component=='dit':
                    if self.mode=='sparse':routes=make_routes(p,'sol',tau=0.)
                    else:
                        head_policy=self.policy.get(str(self.layer),{})
                        candidates={}
                        for head,selected_mode in head_policy.items():
                            if selected_mode=='dense_int8_fp16_pv':continue
                            if selected_mode not in candidates:
                                tau={'sol_tau0':0.,'sol_tau0p5':.5,'sol_tau1':1.}[selected_mode]
                                candidates[selected_mode]=make_routes(p,'sol',tau=tau)
                            routes[:,int(head)]=candidates[selected_mode][:,int(head)]
                    # Full text/reference/audio prefix: preserve conditioning queries and keys.
                    prefix=math.ceil((tensors[0].shape[2]-27*11*20)/64)
                    routes[...,:prefix]=1;routes[...,:prefix,:]=1;pooled=True
                out,_=execute(p,routes,pooled=pooled,causal=causal)
                out=out.to(tensors[0].dtype)
                native=out if kwargs.get('skip_output_reshape') else out.transpose(1,2).reshape(out.shape[0],out.shape[2],-1)
                self.routes_profile.append({'component':self.component,'layer':self.layer,'call':self.call,
                                            'selected_density':routes.float().mean().item()})
                for value in (q,k,v):
                    if hasattr(value,'peek'):value.take()
            else:native=original(q,k,v,heads,*args,**kwargs)
            end.record()
            row={'component':self.component,'layer':self.layer,'denoise_call':self.call,
                 'timestep':self.timestep,'q_shape':list(tensors[0].shape),'k_shape':list(tensors[1].shape),
                 'video_decode_call':self.video_call if self.component=='video' else None,
                 'dtype':str(tensors[0].dtype),
                 'q_stride':list(tensors[0].stride()),'k_stride':list(tensors[1].stride()),'v_stride':list(tensors[2].stride()),
                 'backend':f'tilelang_{self.mode}' if self.mode!='native' and self.component in ('te','dit','video') and self.layer>=0 else original.__name__,
                 'device':str(tensors[0].device)}
            self.events.append((row,start,end))
            key=(self.component,self.layer,self.call if self.component=='dit' else (self.video_call if self.component=='video' else -1))
            selected=self.mode=='native' and self.layer in self.config['capture_layers'].get(self.component,[])
            if self.component=='dit':selected &= self.call in self.config['capture_denoise_calls']
            if self.component=='video':selected &= self.video_call in self.config.get('capture_video_calls',[0])
            if selected and key not in self.seen:
                self.seen.add(key)
                if not kwargs.get('skip_reshape') or tensors[0].ndim!=4:
                    raise ValueError('Expected native BHND attention boundary')
                mask=kwargs.get('mask',args[0] if args else None)
                # Native output BND -> BHND; preserve every query/key/head.
                b,h,n,d=tensors[0].shape
                native_bhnd=native if kwargs.get('skip_output_reshape',False) else native.reshape(b,n,h,d).transpose(1,2)
                data={name:t.detach().cpu().contiguous() for name,t in zip(['q','k','v'],tensors)}
                data.update(native_output=native_bhnd.detach().cpu().contiguous(),
                            mask=mask.detach().cpu() if mask is not None else None,
                            enable_gqa=kwargs.get('enable_gqa',False),scale=kwargs.get('scale'),metadata=row)
                path=self.output/'qkv'/f'{self.component}_layer{self.layer:02d}_call{key[2]:02d}.pt'
                torch.save(data,path)
                record={**row,'file':str(path.relative_to(self.output)),'bytes':path.stat().st_size,
                        'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
                self.records.append(record);self.save()
                print('CAPTURE',path.name,record['bytes'],flush=True)
            return native
        return attention

    def install(self):
        import comfy.ldm.minimax.model as dit
        import comfy.ldm.minimax.vae as video
        for module in (dit,video):
            original=module.optimized_attention
            module.optimized_attention=self.wrap(original)
            self.patch.append((module,original))

    def save(self):
        path=self.output/'capture_manifest.json'
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps({'records':self.records},indent=2));temp.replace(path)

    def finish(self):
        import torch
        torch.cuda.synchronize()
        for row,start,end in self.events:row['attention_ms']=start.elapsed_time(end)
        (self.output/'attention_profile.json').write_text(json.dumps([x[0] for x in self.events],indent=2))
        (self.output/'route_profile.json').write_text(json.dumps(self.routes_profile,indent=2))
        for module,original in self.patch:module.optimized_attention=original
        self.save()


def main(config_path,mode='native'):
    config=json.loads(config_path.read_text())
    baseline=Path(config['output_directory'])
    output=baseline if mode=='native' else baseline/'runs'/mode
    output.mkdir(parents=True,exist_ok=True)
    if mode!='native' and not (baseline/'native_pixels.pt').exists():raise FileNotFoundError('Complete native baseline required')
    os.environ['COMFY_ROOT']=config['comfy_root']
    sys.path[:0]=[config['comfy_root'],str(ROOT),str(ROOT/'benchmarks')]
    import torch
    from torch import nn
    import comfy_kitchen as ck
    kitchen_native_linear=ck.int8_linear
    import retest
    import comfy.ops
    if not hasattr(comfy.ops,'_eager_input_act'):
        # This ComfyUI-master version uses Kitchen's existing input_act API.
        # The older benchmark helper targets a different fused-residual API.
        ck.int8_linear=kitchen_native_linear
        retest.KITCHEN_EAGER_COMPAT=False
    from full_models import StreamedBlock
    from patches.windowed_tp import CheckpointReader
    from patches.comfy_tp_runtime import native_te,native_dit,packet
    from patches.comfy_sampler import create_model
    import comfy.model_management
    import comfy.samplers
    import comfy.sd
    from comfy.nested_tensor import NestedTensor
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    from PIL import Image
    import numpy as np
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if config.get('seed_global_rng',False):
        torch.manual_seed(config['seed']);np.random.seed(config['seed']%(2**32))
    assets=ROOT/config['reference_image']
    if not assets.exists():raise FileNotFoundError('Exact reference image is required')
    paths={kind:str(Path(config['model_directory'])/name) for kind,name in {
        'te':'qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'dit':'MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
        'video':'minimax_h3_video_vae_int8_convrot.safetensors',
        'audio':'minimax_h3_audio_vae_fp32_clean.safetensors'}.items()}
    state={'status':'initializing','mode':mode,'config':config,'gpu':torch.cuda.get_device_name(),
           'torch':torch.__version__,'kitchen_eager_compat':retest.KITCHEN_EAGER_COMPAT,
           'asset_sha256':hashlib.sha256(assets.read_bytes()).hexdigest(),
           'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [Path(__file__),Path(config['comfy_root'])/'comfy/ldm/minimax/model.py',
                Path(config['comfy_root'])/'comfy/ldm/minimax/vae.py',
                Path(config['comfy_root'])/'comfy/text_encoders/llama.py']},'stages':{}}
    def save(): (output/'run_state.json').write_text(json.dumps(state,indent=2))
    policy=json.loads((baseline/'sparse_policy.json').read_text())['layers'] if mode=='adaptive' else None
    save();probe=AttentionCapture(config,output,mode,policy);probe.install()
    (output/'executed_capture_source.py').write_bytes(Path(__file__).read_bytes())
    def timed(label,fn):
        state['status']=label;save();torch.cuda.synchronize();start=time.perf_counter()
        result=fn();torch.cuda.synchronize()
        state['stages'][label]={'wall_seconds':time.perf_counter()-start};save()
        print('STAGE',label,state['stages'][label],flush=True);return result

    class CapturingBlock(StreamedBlock):
        def forward(self,*args,**kwargs):
            with probe.scope(self.kind,self.index):
                if self.kind=='te':kwargs['optimized_attention']=probe.wrap(kwargs['optimized_attention'])
                return super().forward(*args,**kwargs)

    class NativeClip:
        def __init__(self):self.tokenizer=MiniMaxH3Tokenizer()
        def tokenize(self,*args,**kwargs):return self.tokenizer.tokenize_with_weights(*args,**kwargs)
        def encode_from_tokens_scheduled(self,tokens):
            model=native_te(CheckpointReader(paths['te']),50).cuda()
            model.qwen3vl_32b.transformer.model.layers=nn.ModuleList([
                CapturingBlock('te',paths['te'],i,'cuda:0') for i in range(50)])
            model.set_clip_options({'execution_device':torch.device('cuda:0')})
            try:
                with torch.inference_mode(),probe.scope('te'):
                    out=packet(model.encode_token_weights(tokens),'cpu')
            finally:del model;retest.cleanup()
            return [[out[0],out[2]]]

    class LazyNative(nn.Module):
        dtype=torch.float32;patch_size=(1,2,2);hidden_size=5376
        def __init__(self):super().__init__();self.inner=None;self.resident=None
        def load(self):
            if self.inner is None:
                if config.get('dit_loading')=='resident_two_device':
                    from native_resident_reference import ResidentNativeDiT
                    from full_models import PlacedBlock
                    class ScopedPlaced(PlacedBlock):
                        def forward(self,*a,**kw):
                            with probe.scope(self.kind,self.index):return super().forward(*a,**kw)
                    self.resident=ResidentNativeDiT(CheckpointReader(paths['dit']),50,placed_factory=ScopedPlaced,
                        profile_loading=config.get('profile_attention',False))
                    self.inner=self.resident
                else:
                    self.inner=native_dit(CheckpointReader(paths['dit'])).cuda()
                    self.inner.blocks=nn.ModuleList([
                        CapturingBlock('dit',paths['dit'],i,'cuda:0') for i in range(50)])
            return self.inner
        def preprocess_text_embeds(self,*args,**kwargs):return self.load().preprocess_text_embeds(*args,**kwargs)
        def release(self):
            if self.resident is not None:self.resident.release();self.resident=None
            self.inner=None
        def forward(self,x,timestep,*args,**kwargs):
            probe.call+=1;probe.timestep=float(timestep.flatten()[0])
            print('DENOISE',probe.call,probe.timestep,flush=True)
            began=time.perf_counter();cpu_profile=None
            if config.get('cpu_profile_step')==probe.call:
                import cProfile
                cpu_profile=cProfile.Profile();cpu_profile.enable()
            if probe.call==0 and config.get('qualify_resident_loading',False):
                # Same real first denoiser input, original Native operators.
                # Qualification runs outside sampling timing claims; no dump.
                streamed=native_dit(CheckpointReader(paths['dit'])).cuda()
                streamed.blocks=nn.ModuleList([CapturingBlock('dit',paths['dit'],i,'cuda:0') for i in range(50)])
                expected=packet(streamed(x,timestep,*args,**kwargs),'cpu')
                del streamed;retest.cleanup()
                result=self.load()(x,timestep,*args,**kwargs)
                actual=packet(result,'cpu')
                items=lambda value:list(value.unbind()) if hasattr(value,'unbind') else list(value)
                checks=[]
                for a,e in zip(items(actual),items(expected)):
                    checks.append({'relative_l2':float((a.float()-e.float()).norm()/e.float().norm().clamp_min(1e-30)),
                                   'max_absolute':float((a.float()-e.float()).abs().max())})
                if len(checks)!=2 or any(not math.isfinite(c['relative_l2']) or c['relative_l2']>1e-5 for c in checks):raise ValueError('Resident loading changed Native denoiser implementation: '+str(checks))
                (output/'loading_qualification.json').write_text(json.dumps({'checks':checks,'scope':'exact same real first denoiser input, full50layers; not generation quality gate'},indent=2))
                print('LOADING_QUALIFICATION',checks,flush=True)
            else:result=self.load()(x,timestep,*args,**kwargs)
            if config.get('profile_attention',False):
                for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
                probe.model_calls.append({'call':probe.call,'timestep':probe.timestep,'wall_seconds':time.perf_counter()-began,
                    'scope':'instrumented denoiser; call0 includes loading/compilation, later calls are not independent requests'})
                if self.resident is not None:
                    state['stages']['dit_header_loading']={'wall_seconds':self.resident.header_load_seconds}
                    state['stages']['dit_block_loading']={'wall_seconds':self.resident.block_load_seconds}
                save()
            if cpu_profile is not None:
                cpu_profile.disable();cpu_profile.dump_stats(str(output/f'denoiser_step{probe.call:02d}_cpu.prof'))
            # Keep denoiser outputs, not huge per-layer states, at every call.
            if config.get('save_denoiser_tensors',True):
                torch.save({'timestep':probe.timestep,'output':packet(result,'cpu')},
                           output/f'denoiser_call{probe.call:02d}.pt')
            return result

    try:
        video=timed('video_vae_load',lambda:comfy.sd.VAE(sd=CheckpointReader(paths['video']).read(''),
                                   device=torch.device('cuda:0'),dtype=torch.float16))
        for i,block in enumerate(video.first_stage_model.decoder.transformer_blocks):
            original=block.forward
            def scoped(*args,_original=original,_i=i,**kwargs):
                if _i==0:probe.video_call+=1
                with probe.scope('video',_i):return _original(*args,**kwargs)
            block.forward=scoped
        image=torch.from_numpy(np.asarray(Image.open(assets).convert('RGB')).copy()).float()[None]/255
        def condition():
            with torch.inference_mode():
                return MiniMaxH3ReferenceToVideo.execute(NativeClip(),config['prompt'],config['width'],config['height'],config['frames'],
                    ref_image_size=config['ref_image_size'],vae=video,ref_images={'ref_image_1':image})
        reused=config.get('reuse_native_inputs')
        if reused:
            reused=Path(reused)
            cached=torch.load(reused/'conditioning.pt',map_location='cpu',weights_only=False)
            data=[cached['positive'],{'samples':cached['latent']}]
            state['stages']['reference_and_TE']={'reused_inputs':str(reused),'scope':'not measured; cached conditioning'};save()
        else:data=timed('reference_and_TE',condition)
        positive,latent=data[0],data[1]['samples']
        if not reused:torch.save({'positive':packet(positive,'cpu'),'latent':packet(latent,'cpu')},output/'conditioning.pt')
        comfy.model_management.unload_all_models();retest.cleanup()
        class Placeholder:closed=False;lock=threading.RLock()
        model=create_model(Placeholder())
        model.remove_wrappers_with_key('outer_sample','minimax_tp2_session')
        model.model.diffusion_model=LazyNative()
        gen=torch.Generator().manual_seed(config['seed'])
        if reused:noise=torch.load(reused/'initial_noise.pt',map_location='cpu',weights_only=False)
        else:noise=NestedTensor([torch.randn(t.shape,generator=gen) for t in latent.unbind()]) if mode=='native' else torch.load(baseline/'initial_noise.pt',map_location='cpu',weights_only=False)
        if not reused:torch.save(packet(noise,'cpu'),output/'initial_noise.pt')
        guider=comfy.samplers.CFGGuider(model);guider.set_conds(positive,positive);guider.set_cfg(config['cfg'])
        sigmas=comfy.samplers.calculate_sigmas(model.model.model_sampling,config['scheduler'],config['steps'])
        if reused:
            original_sigmas=torch.load(reused/'sigmas.pt',map_location='cpu',weights_only=True)
            if not torch.equal(sigmas.cpu(),original_sigmas):raise ValueError('Reused native sigma schedule differs')
            sigmas=original_sigmas.to(sigmas.device)
            (output/'reused_inputs.json').write_text(json.dumps({name:hashlib.file_digest((reused/name).open('rb'),'sha256').hexdigest()
                for name in ['conditioning.pt','initial_noise.pt','sigmas.pt']},indent=2))
        else:torch.save(sigmas.cpu(),output/'sigmas.pt')
        if config.get('prepare_inputs_only',False):
            probe.finish();state['status']='inputs_prepared';save();return
        def sample():
            with torch.inference_mode():
                return guider.sample(noise,latent,comfy.samplers.sampler_object(config['sampler']),sigmas,
                                     disable_pbar=True,seed=config['seed']).cpu()
        sampled=timed('native_sampling',sample)
        video_latent,audio_latent=sampled.unbind()
        torch.save(video_latent,output/'native_video_latent.pt');torch.save(audio_latent,output/'native_audio_latent.pt')
        model.model.diffusion_model.release();del model,guider
        comfy.model_management.unload_all_models();retest.cleanup()
        with torch.inference_mode():pixels=timed('native_video_decode',lambda:video.decode(video_latent).cpu())
        if config.get('save_pixel_tensor',True):torch.save(pixels,output/'native_pixels.pt')
        state['video_shape']=list(pixels.shape)
        frames=pixels[0] if pixels.ndim==5 and pixels.shape[0]==1 else pixels
        # Export an inspectable baseline clip. No interpolation or frame changes.
        import av
        with av.open(str(output/'native_video.mp4'),'w') as container:
            stream=container.add_stream('libx264',rate=config['fps']);stream.width=frames.shape[-2];stream.height=frames.shape[-3];stream.pix_fmt='yuv420p'
            for frame in frames:
                array=frame.clamp(0,1).mul(255).round().byte().numpy()
                for pkt in stream.encode(av.VideoFrame.from_ndarray(array,format='rgb24')):container.mux(pkt)
            for pkt in stream.encode():container.mux(pkt)
        if mode!='native':
            expected=torch.load(baseline/'native_pixels.pt',map_location='cpu',weights_only=True)
            expected=expected[0] if expected.ndim==5 and expected.shape[0]==1 else expected
            per_frame=[]
            for i in range(frames.shape[0]):
                diff=frames[i].float()-expected[i].float()
                mse=diff.square().mean().item()
                entry={'frame':i,'seconds':i/config['fps'],'rmse':math.sqrt(mse),
                       'relative_l2':(diff.norm()/expected[i].float().norm().clamp_min(1e-12)).item(),
                       'psnr_db':-10*math.log10(max(mse,1e-30))}
                if i:
                    delta=(frames[i]-frames[i-1])-(expected[i]-expected[i-1])
                    entry['temporal_difference_rmse']=delta.float().square().mean().sqrt().item()
                per_frame.append(entry)
            (output/'frame_errors.json').write_text(json.dumps(per_frame,indent=2))
        probe.finish();state['status']='complete';state['captures']=len(probe.records);save()
        print('COMPLETE',len(probe.records),flush=True)
    except Exception:
        state['status']='failed';state['traceback']=traceback.format_exc();save();probe.finish();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'bench_assets/tokyo_ref2va.json')
    parser.add_argument('--mode',choices=['native','dense','sparse','adaptive'],default='native')
    args=parser.parse_args();main(args.config,args.mode)
