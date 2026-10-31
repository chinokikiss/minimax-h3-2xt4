"""ComfyUI MODEL/CLIP/VAE adapters using Native guider/sampler interfaces."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from patches.comfy_tp_runtime import packet


def reclaim_parent_models(device,service=None):
    """Make ComfyUI's real models relinquish VRAM before proxy execution.

    Proxy parameter counts cannot describe the worker's checkpoint window.
    Native model management still owns the VAE's transfer and CPU backing.
    """
    import comfy.model_management
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        keep=service.idle_vae_patchers() if service is not None and hasattr(service,'idle_vae_patchers') else []
        comfy.model_management.free_memory(
            torch.cuda.get_device_properties(device).total_memory, device,keep_loaded=keep)


class RemoteConditioning(nn.Module):
    def __init__(self, service):
        super().__init__()
        self.service = service
        self.dtypes = {torch.float16}
        self.options = {}
        self.register_buffer('device_anchor', torch.zeros(1), persistent=False)

    def reset_clip_options(self):
        self.options = {}

    def set_clip_options(self, options):
        self.options.update(options)

    def encode_token_weights(self, tokens):
        options = dict(self.options)
        device = options.pop('execution_device', self.device_anchor.device)
        if self.service.config.idle_residency:
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('TP2 idle residency does not support CUDA Graph capture')
            from patches.comfy_idle_lifecycle import preserve_te
            with preserve_te(self.service):reclaim_parent_models(device,self.service)
        else:reclaim_parent_models(device)
        return packet(self.service.call('te', tokens, **options), device)

    def get_dynamic_vram__units(self):
        return [], []


class RemoteDenoiser(nn.Module):
    dtype = torch.float32
    patch_size = (1, 2, 2)
    hidden_size = 5376

    def __init__(self, service):
        super().__init__()
        self.service = service
        self._aot_shapes=set()
        flow=getattr(service.config,'tensor_flow',None)
        if flow is not None and flow.enabled:self.dtype=flow.dtype
        self.register_buffer('device_anchor', torch.zeros(1), persistent=False)

    def preprocess_text_embeds(self, context):
        flow=getattr(self.service.config,'tensor_flow',None)
        if flow is not None and flow.enabled and context.shape[-1]==self.hidden_size:
            return context.to(flow.dtype)
        return packet(self.service.call('refine', context), context.device)

    def prepare_payload(self,payload):
        if not getattr(self.service,'operator_aot',False):return
        layout=(payload or {}).get('layout')
        if layout is None or not getattr(getattr(self.service.config,'operators',None),'aot_precompile',False):return
        rows=layout.seq_len
        if rows not in self._aot_shapes:
            self.service.call('prepare_dit',rows)
            self._aot_shapes.add(rows)

    def forward(self, x, timestep, context=None, control=None, transformer_options=None, **kwargs):
        if control is not None:
            raise ValueError('ControlNet has not been qualified for this TP2 adapter')
        options = dict(transformer_options or {})
        # Our lifecycle wrapper executes in the main Native sampler. Never
        # serialize its bound service or silently drop another extension.
        wrappers = {k:dict(v) for k,v in options.get('wrappers', {}).items()}
        outer = wrappers.get('outer_sample', {})
        outer.pop('minimax_tp2_session', None)
        if not outer:
            wrappers.pop('outer_sample', None)
        if wrappers:
            raise ValueError('Additional model wrappers require TP qualification')
        options.pop('wrappers', None)
        self.prepare_payload(kwargs.get('minimax_payload'))
        result = self.service.call('denoise', x, timestep, context=context,
            control=None, transformer_options=options, **kwargs)
        device = x[0].device if isinstance(x, list) else x.device
        return packet(result, device)


def create_model(service, device='cuda:0', *, adapter=None):
    import comfy.model_base
    import comfy.model_patcher
    import comfy.patcher_extension
    import comfy.supported_models
    config = comfy.supported_models.MiniMaxH3({'image_model':'minimax_h3',
        'disable_unet_model_creation':True})
    flow=getattr(service.config,'tensor_flow',None)
    config.set_inference_dtype(flow.dtype if flow is not None and flow.enabled else torch.float32, None)
    base_class=comfy.model_base.MiniMaxH3
    if (flow is not None and flow.enabled) or adapter is not None:
        class ScaledConditioningH3(base_class):
            def extra_conds(self, **kwargs):
                # Native casts unnormalized TE states to inference dtype before
                # condition_proj. Preserve this boundary in FP32 until the
                # projection's folded output scale has made FP16 storage safe.
                cross=kwargs.get('cross_attn')
                if cross is not None:
                    kwargs['cross_attn']=self.diffusion_model.preprocess_text_embeds(
                        cross.to(device=kwargs['device'],dtype=torch.float32))
                # Retain Native layout construction, references, masks and tags.
                # The proxy recognizes the already-refined hidden width.
                out=super().extra_conds(**kwargs)
                if adapter is not None:out=adapter.conditions(out,kwargs)
                self.diffusion_model.prepare_payload(out['minimax_payload'].cond)
                return out
        base_class=ScaledConditioningH3
    native = base_class(config, device=torch.device(device))
    native.diffusion_model = RemoteDenoiser(service) if adapter is None else adapter.denoiser(service)
    patcher = comfy.model_patcher.ModelPatcher(native, torch.device(device), torch.device('cpu'))

    def sample_session(executor, *args, **kwargs):
        active_patcher = executor.class_obj.model_patcher
        if active_patcher.patches or active_patcher.forced_hooks:
            raise ValueError('LoRA/weight hooks require separate TP checkpoint qualification')
        # Lock spans the whole sample, preventing another workflow from
        # changing worker state between two denoiser calls.
        with service.lock:
            if hasattr(service,'release_staged_encoders'):service.release_staged_encoders()
            reclaim_parent_models(active_patcher.load_device)
            if adapter is not None:adapter.begin()
            try:
                service.call('begin_sample')
                try:
                    return executor(*args, **kwargs)
                finally:
                    if not service.closed:service.call('end_sample')
            finally:
                if adapter is not None and not service.closed:adapter.end()
    patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        'minimax_tp2_session', sample_session)
    return patcher


def create_clip(service, device='cuda:0'):
    import comfy.sd
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    # Native CLIP scheduling/tokenization; Native SDClipModel conditioning is
    # executed on the workers, including vision, DeepStack and token tags.
    target = SimpleNamespace(params={}, tokenizer=MiniMaxH3Tokenizer,
        clip=lambda **kwargs: RemoteConditioning(service))
    return comfy.sd.CLIP(target=target, model_options={'load_device':torch.device(device),
        'offload_device':torch.device('cpu'), 'initial_device':torch.device('cpu'),
        'dtype':torch.float16}, disable_dynamic=True)


class DecodeHandoff:
    """Delegate Native VAE; prepare next TE CPU weights during decode."""
    def __init__(self, vae, service):
        self.vae, self.service = vae, service

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def _decode(self, method, *args, **kwargs):
        with self.service.lock:
            self.service.call('decode_start')
            from patches.kitchen_attention import native_scope
            with native_scope():output = method(*args, **kwargs)
            device = torch.device(self.vae.device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            # Result delivery does not wait for the CPU prefetch future.
            self.service.call('decode_complete')
            return output

    def decode(self, *args, **kwargs):
        return self._decode(self.vae.decode, *args, **kwargs)

    def decode_tiled(self, *args, **kwargs):
        return self._decode(self.vae.decode_tiled, *args, **kwargs)


class ParallelDecodeHandoff(DecodeHandoff):
    """Lease video-only parallel decoding; audio always uses Native single-GPU.

    Checkpoint path comes from the loader that constructed this Native VAE.
    Single-device / unsupported dtype / memory admission uses Native. Once a
    collective decode starts, execution errors propagate; no partial fallback.
    """
    def __init__(self, vae, service, checkpoint, mode='temporal'):
        super().__init__(vae, service)
        if mode not in ('auto', 'native', 'query', 'temporal'):
            raise ValueError(mode)
        # Saved workflows using auto follow the promoted parallel strategy.
        self.checkpoint, self.mode = str(checkpoint), 'temporal' if mode == 'auto' else mode
        self.last_strategy = 'native'
        self.last_fallback = None
        self.last_query_partitioning = None
        self.last_admission = None

    def _decode(self, method, *args, **kwargs):
        with self.service.lock:
            from patches.staged_vae import StagedVAE
            template=None
            if (isinstance(self.vae,StagedVAE) and getattr(self.vae,'kind',None)=='video'
                    and self.mode in ('auto', 'query', 'temporal') and self.vae.vae_dtype==torch.float16
                    and not getattr(self.vae.patcher,'patches',None)
                    and not getattr(self.vae.patcher,'forced_hooks',None)
                    and torch.device(self.vae.device).type=='cuda'):
                template=self.vae.prepare_remote_decode()
            return self._decode_prepared(method,*args,_metadata_model=template,**kwargs)

    def _native_decode(self,method,*args,**kwargs):
        from patches.staged_vae import StagedVAE
        if isinstance(self.vae,StagedVAE):self.vae.prepare_decode()
        return super()._decode(method,*args,**kwargs)

    def _decode_prepared(self, method, *args, _metadata_model=None, **kwargs):
        import logging
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
        model = _metadata_model if _metadata_model is not None else self.vae.first_stage_model
        kind = 'video' if isinstance(model, MiniMaxH3VideoVAE) else None
        dtype = torch.float16
        self.last_strategy, self.last_fallback = 'native', None
        self.last_query_partitioning = None
        self.last_admission = None
        if (self.mode == 'native' or kind is None or self.vae.vae_dtype != dtype or
                getattr(self.vae.patcher, 'patches', None) or
                getattr(self.vae.patcher, 'forced_hooks', None) or
                torch.device(self.vae.device).type != 'cuda'):
            return self._native_decode(method, *args, **kwargs)
        if self.mode != 'temporal':
            from patches.parallel_vae import query_partitioning_status
            backend=getattr(getattr(self.service,'config',None),'video_attention','kitchen')
            self.last_query_partitioning=(query_partitioning_status(model,checkpoint_reader=self.vae.reader,attention_backend=backend)
                if _metadata_model is not None else query_partitioning_status(model,attention_backend=backend))
            if not self.last_query_partitioning['available']:
                self.last_fallback='Video VAE query-parallel unavailable: '+self.last_query_partitioning['reason']
                logging.info('MiniMax video VAE using Native: %s',self.last_fallback)
                return self._native_decode(method, *args, **kwargs)
        if type(model) is not MiniMaxH3VideoVAE:
            self.last_fallback = 'Remote planner requires the frozen Native Video VAE class'
            return self._native_decode(method, *args, **kwargs)
        samples = args[0] if args else kwargs.get('samples_in', kwargs.get('samples'))
        if samples is None:
            raise ValueError('VAE decode requires samples_in')
        with self.service.lock:
            memory = self.vae.memory_used_decode(samples.shape, dtype)
            numeric = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic)
            def remote(z):
                output = self.service.call('video_pixels', z)
                return output.to(device=z.device)
            def remote_pair(clips):
                return self.service.call('video_temporal_pair', clips)

            from patches.video_decode_planner import remote_video_planner
            temporal = self.mode == 'temporal'
            with remote_video_planner(self.vae, self.service, remote,template=_metadata_model,
                    decode_pair=remote_pair if temporal else None) as planner:
                prepare_args = (self.checkpoint, kind, int(memory), numeric) + (('temporal',) if temporal else ())
                prepared = (self.service.call('vae_prepare', *prepare_args)
                    if planner is not None else
                    {'accepted':False, 'reason':'Parent Video VAE could not offload to CPU'})
                self.last_admission = prepared.get('admission')
                if prepared['accepted']:
                    self.last_strategy = 'native_temporal_pairs' if temporal else 'native_aligned_query'
                    try:
                        self.service.call('decode_start')
                        output = method(*args, **kwargs)
                        torch.cuda.synchronize(self.vae.device)
                    finally:
                        if not self.service.closed:
                            self.service.call('vae_release')
                    self.service.call('decode_complete')
                    return output
            # Restore the trained parent and release the planner before fallback.
            self.last_fallback = prepared['reason']
            logging.info('MiniMax %s VAE using Native: %s', kind, self.last_fallback)
            return self._native_decode(method, *args, **kwargs)
