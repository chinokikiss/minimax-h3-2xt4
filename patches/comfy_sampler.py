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
        self.register_buffer('device_anchor', torch.zeros(1), persistent=False)

    def preprocess_text_embeds(self, context):
        return packet(self.service.call('refine', context), context.device)

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
        result = self.service.call('denoise', x, timestep, context=context,
            control=None, transformer_options=options, **kwargs)
        device = x[0].device if isinstance(x, list) else x.device
        return packet(result, device)


def create_model(service, device='cuda:0'):
    import comfy.model_base
    import comfy.model_patcher
    import comfy.patcher_extension
    import comfy.supported_models
    config = comfy.supported_models.MiniMaxH3({'image_model':'minimax_h3',
        'disable_unet_model_creation':True})
    config.set_inference_dtype(torch.float32, None)
    native = comfy.model_base.MiniMaxH3(config, device=torch.device(device))
    native.diffusion_model = RemoteDenoiser(service)
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
            service.call('begin_sample')
            try:
                return executor(*args, **kwargs)
            finally:
                if not service.closed:
                    service.call('end_sample')
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
            output = method(*args, **kwargs)
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
    """Lease parallel raw decoding inside the unchanged Native public VAE.

    Checkpoint path comes from the loader that constructed this Native VAE.
    Single-device / unsupported dtype / memory admission uses Native. Once a
    collective decode starts, execution errors propagate; no partial fallback.
    """
    def __init__(self, vae, service, checkpoint, mode='auto'):
        super().__init__(vae, service)
        if mode not in ('auto', 'native'):
            raise ValueError(mode)
        self.checkpoint, self.mode = str(checkpoint), mode
        self.last_strategy = 'native'
        self.last_fallback = None
        self.last_query_partitioning = None

    def _decode(self, method, *args, **kwargs):
        with self.service.lock:
            if hasattr(self.vae,'prepare_decode'):self.vae.prepare_decode()
            return self._decode_prepared(method,*args,**kwargs)

    def _decode_prepared(self, method, *args, **kwargs):
        import logging
        import comfy.model_management
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
        from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
        model = self.vae.first_stage_model
        kind = ('video' if isinstance(model, MiniMaxH3VideoVAE) else
                'audio' if isinstance(model, MiniMaxH3AudioVAE) else None)
        dtype = torch.float16 if kind == 'video' else torch.float32
        self.last_strategy, self.last_fallback = 'native', None
        self.last_query_partitioning = None
        if (self.mode == 'native' or kind is None or self.vae.vae_dtype != dtype or
                getattr(self.vae.patcher, 'patches', None) or
                getattr(self.vae.patcher, 'forced_hooks', None) or
                torch.device(self.vae.device).type != 'cuda'):
            return super()._decode(method, *args, **kwargs)
        if kind == 'video':
            from patches.parallel_vae import query_partitioning_status
            self.last_query_partitioning=query_partitioning_status(model)
            if not self.last_query_partitioning['available']:
                self.last_fallback='Video VAE query-parallel unavailable: '+self.last_query_partitioning['reason']
                logging.info('MiniMax video VAE using Native: %s',self.last_fallback)
                return super()._decode(method, *args, **kwargs)
        samples = args[0] if args else kwargs.get('samples_in', kwargs.get('samples'))
        if samples is None:
            raise ValueError('VAE decode requires samples_in')
        if kind == 'audio' and (samples.ndim != 4 or samples.shape[2] != 2):
            return super()._decode(method, *args, **kwargs)
        with self.service.lock:
            memory = self.vae.memory_used_decode(samples.shape, dtype)
            # Account for the parent model before admitting rank replicas.
            # The subsequent Native public call owns batch/tile choices.
            try:
                with comfy.model_management.cuda_device_context(self.vae.device):
                    comfy.model_management.load_models_gpu([self.vae.patcher],
                        memory_required=memory, force_full_load=self.vae.disable_offload)
            except comfy.model_management.OOM_EXCEPTION:
                self.last_fallback = 'Parent VAE memory admission'
                return super()._decode(method, *args, **kwargs)
            numeric = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic)
            prepared = self.service.call('vae_prepare', self.checkpoint, kind, int(memory), numeric)
            if not prepared['accepted']:
                self.last_fallback = prepared['reason']
                logging.info('MiniMax %s VAE using Native: %s', kind, self.last_fallback)
                return super()._decode(method, *args, **kwargs)
            target = '_decode_pixels' if kind == 'video' else 'decode'
            had_override = target in model.__dict__
            override = model.__dict__.get(target)

            def remote(z):
                command = 'video_pixels' if kind == 'video' else 'audio_channels'
                output = self.service.call(command, z)
                return output.to(device=z.device)

            self.last_strategy = 'native_aligned_query' if kind == 'video' else 'channel_parallel'
            try:
                setattr(model, target, remote)
                self.service.call('decode_start')
                output = method(*args, **kwargs)
                torch.cuda.synchronize(self.vae.device)
            finally:
                # Restore the exact instance state, including after cancellation.
                if had_override:
                    setattr(model, target, override)
                else:
                    delattr(model, target)
                if not self.service.closed:
                    self.service.call('vae_release')
            self.service.call('decode_complete')
            return output
