"""H3 component-only checkpoint loading behind Native public VAE methods."""
from concurrent.futures import CancelledError
import logging
import threading
import torch
from patches.windowed_tp import CheckpointReader


def checkpoint_kind(reader):
    names=reader.header
    if 'decoder.transformer_blocks.0.scale1' in names and 'encoder.down.5.block.0.conv1.weight' in names:return 'video'
    if 'pre_block.attn.zero_k_bias' in names:return 'audio'
    return None


def component_names(reader,kind,stage):
    if stage not in ('encode','decode'):raise ValueError(stage)
    if kind=='video':prefixes=('encoder.','quant_conv.') if stage=='encode' else ('decoder.','post_quant_conv.')
    elif kind=='audio':prefixes=('encoder.','pre_block.','mean_proj.','logs_proj.') if stage=='encode' else ('decoder.','dec_in_proj.')
    else:raise ValueError(kind)
    return [n for n in reader.header if n!='__metadata__' and (n.startswith(prefixes) or n in ('latents_mean','latents_std'))]


def component_payload(reader,names):
    return sum(reader.header[n]['data_offsets'][1]-reader.header[n]['data_offsets'][0] for n in names)


def native_shell(reader,kind):
    """Use Native VAE metadata/planning; all discarded model weights are meta.

    Only architecture sentinels are supplied, never checkpoint payload. The
    shell's model is immediately discarded and replaced at stage admission.
    """
    import comfy.sd
    sentinels=(('decoder.transformer_blocks.0.scale1','encoder.down.5.block.0.conv1.weight')
               if kind=='video' else ('pre_block.attn.zero_k_bias',))
    sd={n:torch.empty(reader.header[n]['shape'],dtype=reader.DTYPES[reader.header[n]['dtype']],device='meta') for n in sentinels}
    class QuietMissing(logging.Filter):
        owner=threading.get_ident()
        def filter(self,record):return not (record.thread==self.owner and record.getMessage().startswith('Missing VAE keys'))
    quiet=QuietMissing();logger=logging.getLogger();logger.addFilter(quiet)
    try:
        with torch.device('meta'):
            vae=comfy.sd.VAE(sd=sd,dtype=torch.float16 if kind=='video' else torch.float32)
    finally:logger.removeFilter(quiet)
    vae.first_stage_model=None;vae.patcher=None;vae.size=None
    return vae


@torch.inference_mode(False)
@torch.no_grad()
def load_component(reader,kind,stage,cancel=None):
    import comfy.ops
    if cancel is not None and cancel.is_set():raise CancelledError('VAE preload cancelled')
    # Construct Native shape/configuration on meta, delete the inactive half
    # before allocating any parameter storage. No global class monkeypatches.
    with torch.device('meta'):
        if kind=='video':
            from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
            ops=comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.float16)
            model=MiniMaxH3VideoVAE(operations=ops)
        else:
            from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
            model=MiniMaxH3AudioVAE()
    remove=(('decoder','post_quant_conv') if stage=='encode' else ('encoder','quant_conv')) if kind=='video' else (
        ('decoder','dec_in_proj') if stage=='encode' else ('encoder','pre_block','mean_proj','logs_proj'))
    for name in remove:delattr(model,name)
    model.to(dtype=torch.float16 if kind=='video' else torch.float32)
    model.to_empty(device='cpu')
    if kind=='video':
        from comfy.ldm.minimax.vae import LATENTS_MEAN,LATENTS_STD,IMAGENET_MEAN,IMAGENET_STD
        model.latents_mean=torch.tensor(LATENTS_MEAN);model.latents_std=torch.tensor(LATENTS_STD)
        model.pixel_mean=torch.tensor(IMAGENET_MEAN).view(1,3,1,1,1)
        model.pixel_std=torch.tensor(IMAGENET_STD).view(1,3,1,1,1)
        if stage=='decode':
            # This deterministic RoPE buffer is nonpersistent and therefore
            # absent from safetensors. to_empty must not leave it uninitialized.
            from comfy.ldm.minimax.vae import RotaryEmbeddingND
            rotary=model.decoder.pos_embed
            model.decoder.pos_embed=RotaryEmbeddingND(2*rotary.n_dim*rotary.inv_freq.numel(),n_dim=rotary.n_dim)
            for block in model.decoder.transformer_blocks:
                if hasattr(block.attn,'qk_norm_scale'):
                    # The legacy fused attention API also keeps a unit-norm
                    # buffer outside state_dict. Native fallback needs it too.
                    block.attn.qk_norm_scale=torch.ones(block.attn.dim_head)
    state={}
    for name in component_names(reader,kind,stage):
        if cancel is not None and cancel.is_set():raise CancelledError('VAE preload cancelled')
        state[name]=reader.read_tensor(name)
    missing,unexpected=model.load_state_dict(state,strict=False,assign=True)
    allowed={'latents_mean','latents_std'} if kind=='video' else set()
    if set(missing)-allowed or unexpected:raise ValueError(f'Incomplete H3 {kind} {stage}: {missing}, {unexpected}')
    del state
    model=model.to(dtype=torch.float16 if kind=='video' else torch.float32).eval().requires_grad_(False)
    if kind=='video':
        from patches.h3_tensor_flow import prepare_video,policy_for
        prepare_video(model,policy_for(reader))
    return model


def video_planning_template(reader):
    """Architecture and tiny normalization buffers; never learned payloads."""
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE,LATENTS_MEAN,LATENTS_STD,IMAGENET_MEAN,IMAGENET_STD
    import comfy.ops
    with torch.device('meta'):
        ops=comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.float16)
        model=MiniMaxH3VideoVAE(operations=ops)
    for name,values in (('latents_mean',LATENTS_MEAN),('latents_std',LATENTS_STD),
                        ('pixel_mean',IMAGENET_MEAN),('pixel_std',IMAGENET_STD)):
        value=reader.read_tensor(name) if name in reader.header else torch.tensor(values)
        if name.startswith('pixel_'):value=value.view(1,3,1,1,1)
        setattr(model,name,value.to(device='cpu',dtype=torch.float16))
    return model


class StagedVAE:
    """Demand-load one H3 component; Native still owns encode/decode planning."""
    def __init__(self,path,service):
        self.reader=CheckpointReader(path);self.kind=checkpoint_kind(self.reader)
        from patches.h3_tensor_flow import TensorFlowPolicy
        self.reader.tensor_flow=getattr(service.config,'tensor_flow',TensorFlowPolicy())
        if self.kind is None:raise ValueError('Staged loader requires a Native H3 VAE checkpoint')
        self.service=service;self.lock=threading.RLock();self.stage=None;self.reservation=0
        self.vae=native_shell(self.reader,self.kind)
        self.events=[];self.encoder_used=False
        service.register_staged_vae(self)

    def __getattr__(self,name):return getattr(self.vae,name)

    def __del__(self):
        # ComfyUI cache eviction can discard a VAE node without closing its
        # still-live TP service. Return the parent lease in that case too.
        if self.__dict__.get('stage') is not None:
            try:self.release()
            except Exception:logging.debug('H3 staged VAE final release failed',exc_info=True)

    def check_identity(self):
        stat=self.reader.path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=self.reader.signature:raise RuntimeError('VAE checkpoint changed; reload its loader node')

    def model_size(self):
        return 0 if self.stage is None else self.vae.model_size()

    def throw_exception_if_invalid(self):
        # A recognized, unchanged lazy checkpoint is valid without weight storage.
        # Full component/state validation still runs at the first actual use.
        self.check_identity()

    def _release(self):
        if self.stage is None:return
        import comfy.model_management as management
        if self.vae.patcher is not None:
            # Unload only this patcher, preserving independently managed models.
            from patches.comfy_idle_lifecycle import preserve_te
            with preserve_te(self.service):management.unload_model_and_clones(self.vae.patcher)
        self.vae.first_stage_model=None;self.vae.patcher=None;self.vae.size=None
        self.stage=None
        if self.reservation:
            self.service.budget.release(self.reservation);self.reservation=0

    def release(self):
        with self.service.lock,self.lock:self._release()

    def _ensure(self,stage,cancel=None):
        self.check_identity()
        if self.stage==stage:return
        self._release()
        names=component_names(self.reader,self.kind,stage)
        # Cover model storage + checkpoint staging. The conservative lease stays
        # live until release, even after Native transfers weights to the GPU.
        reservation=2*component_payload(self.reader,names)+32*2**20
        if not self.service.budget.acquire(reservation,speculative=cancel is not None):
            raise MemoryError('Host budget cannot admit H3 VAE '+stage)
        self.reservation=reservation
        try:
            model=load_component(self.reader,self.kind,stage,cancel)
            if cancel is not None and cancel.is_set():raise CancelledError('VAE preload cancelled')
            import comfy.model_patcher
            import comfy.model_management as management
            self.vae.first_stage_model=model
            self.vae.patcher=comfy.model_patcher.CoreModelPatcher(model,load_device=self.vae.device,
                offload_device=management.vae_offload_device())
            management.archive_model_dtypes(model)
            self.vae.size=None;self.stage=stage
            self.events.append(dict(stage=stage,read_bytes=component_payload(self.reader,names)))
            self.events=self.events[-8:]
        except BaseException:
            self.vae.first_stage_model=None;self.vae.patcher=None
            self.service.budget.release(self.reservation);self.reservation=0
            raise

    def prepare_decode(self):
        if hasattr(self.service,'idle_vae_cancel'):self.service.idle_vae_cancel.set()
        with self.service.lock,self.lock:self._ensure('decode')

    def prepare_remote_decode(self):
        if self.kind!='video':raise ValueError('Remote decode is video-only')
        if hasattr(self.service,'idle_vae_cancel'):self.service.idle_vae_cancel.set()
        with self.service.lock,self.lock:
            self.check_identity()
            self._release()
            return video_planning_template(self.reader)

    def _run(self,stage,method,*args,**kwargs):
        if hasattr(self.service,'idle_vae_cancel'):self.service.idle_vae_cancel.set()
        with self.service.lock,self.lock:
            if not (stage=='decode' and self.__dict__.get('remote_decode_active',False)):
                self._ensure(stage)
            from patches.comfy_idle_lifecycle import preserve_te
            with preserve_te(self.service):output=getattr(self.vae,method)(*args,**kwargs)
            if stage=='encode':self.encoder_used=True
            return output

    def encode(self,*args,**kwargs):return self._run('encode','encode',*args,**kwargs)
    def encode_tiled(self,*args,**kwargs):return self._run('encode','encode_tiled',*args,**kwargs)
    def decode(self,*args,**kwargs):return self._run('decode','decode',*args,**kwargs)
    def decode_tiled(self,*args,**kwargs):return self._run('decode','decode_tiled',*args,**kwargs)

    def preload_idle(self,cancel):
        with self.service.lock,self.lock:
            self._ensure('encode',cancel)
            if cancel.is_set():raise CancelledError('VAE preload cancelled')
            import comfy.model_management as management
            device=torch.device(self.vae.device)
            if device.type!='cuda':return False
            free=management.get_free_memory(device)
            needed=self.vae.model_size()+self.service.config.idle_reserve_bytes
            if free<needed:return False
            # Honor worker TE ownership; don't evict another request's models.
            from patches.comfy_idle_lifecycle import preserve_te
            with preserve_te(self.service):
                management.load_models_gpu([self.vae.patcher],force_full_load=True)
            return True
