"""Opt-in adapters around the existing TP2 service; no Native forward replacement."""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from patches.comfy_sampler import RemoteDenoiser, create_model
from patches.windowed_tp import tp_specs
from patches.tp2_bundle import BundleReader
from .bank import PTDBank
from .schedule import T_MAX


@dataclass(frozen=True)
class PTDOptions:
    enabled: bool = False
    allow_pruned_adaln: bool = False
    use_hop_refiner: bool = True
    use_pdd_heads: bool = True
    audio_chord_correction: bool = True
    stream_weights: bool = True
    allow_fixture: bool = False
    hop_handoff: str = 'queue'
    future_cache: bool = False
    prefetch_refiner: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for key,value in vars(self).items() if key != 'hop_handoff'):
            raise ValueError('PTD options require explicit booleans')
        if self.hop_handoff not in ('queue','rebuild'):
            raise ValueError('Unknown PTD hop handoff')
        if self.enabled and not self.stream_weights and self.hop_handoff == 'queue':
            raise ValueError('Resident-expert ablation requires hop_handoff=rebuild')
        if (self.future_cache or self.prefetch_refiner) and self.hop_handoff != 'queue':
            raise ValueError('CPU scheduling experiments require the queue handoff')
        if self.prefetch_refiner and (not self.future_cache or not self.use_hop_refiner):
            raise ValueError('Refiner prefetch requires a future cache and per-hop refiners')


def audio_sigma(time, ratio=.25):
    return ratio * time * T_MAX / (T_MAX + (ratio - 1.) * time)


def native_audio_shift(time, ratio=.25):
    if time == T_MAX:
        return 1.
    a = audio_sigma(time, ratio)
    return a * (1. - time) / (time * (1. - a))


class WorkerAdapter:
    def __init__(self, runtime, path, options, expected_bank_sha256=None):
        if isinstance(options,dict):options=PTDOptions(**options)
        if not isinstance(options, PTDOptions) or not options.enabled:
            raise ValueError('Explicit enabled PTD options required')
        self.runtime, self.options = runtime, options
        self.bank = PTDBank(path, allow_fixture=options.allow_fixture, expected_sha256=expected_bank_sha256)
        self.original = runtime.readers['dit']
        if (type(self.original) is not BundleReader or self.original.path.resolve() != self.bank.base
                or runtime.config.lora_stack):
            raise ValueError('PTD requires its original TP2 base, without a second Turbo/online LoRA')
        if self.bank.layers != runtime.config.layers:
            raise ValueError('Runtime layer count differs from the baked PTD bank')
        if 'adaln_t_table' in self.original.header and not options.allow_pruned_adaln:
            raise ValueError('Current pruned AdaLN base needs explicit allow_pruned_adaln approximation consent')
        if not options.stream_weights:
            from patches.rank_local_int8 import rank_block_bytes
            needed = sum(rank_block_bytes(self.original, 'dit', i, runtime.rank) for i in range(self.bank.layers))
            free = torch.cuda.mem_get_info(runtime.device)[0]
            if needed + runtime.config.reserve_bytes > free:
                raise MemoryError('Resident PTD expert does not fit beside the activation reserve; enable streaming')
            # The existing window can hold the entire active expert. Other
            # seven experts remain on disk; no eight-model GPU allocation.
            self.original_config = runtime.config
            runtime.config = __import__('dataclasses').replace(runtime.config, weight_bytes=needed + 2**20)
        self.hop = None
        self.calls = []
        self.head_hook = None
        self.queue = None
        self.select(0, float(self.bank.sigmas[0]), preparing=True)

    def select(self, hop, sigma, *, preparing=False):
        if type(hop) is not int or not 0 <= hop < 8 or abs(sigma - float(self.bank.sigmas[hop])) > 2e-6:
            raise ValueError('PTD expert and trained sigma must match')
        if not preparing and (not self.runtime.sample_active or hop != len(self.calls)):
            raise ValueError('PTD requires ordered, exactly-once hop evaluation inside the sampling session')
        self.bank.check()
        if self.hop != hop:
            if self.options.hop_handoff == 'queue':
                if self.queue is not None:
                    self.queue.select(hop)
                elif preparing and hop == 0:
                    reader = self.bank.reader(0, refiner=self.options.use_hop_refiner,
                        heads=self.options.use_pdd_heads,base_reader=self.original)
                    reader.operator_policy = self.runtime.config.operators
                    reader.tensor_flow = self.runtime.config.tensor_flow
                    self.runtime.readers['dit'] = reader
                else:
                    raise RuntimeError('The initial expert must execute before selecting the next one')
            else:
                if self.head_hook is not None:
                    self.head_hook.remove(); self.head_hook = None
                self.runtime._drop_model()
                reader = self.bank.reader(hop, refiner=self.options.use_hop_refiner, heads=self.options.use_pdd_heads)
                reader.operator_policy = self.runtime.config.operators
                reader.tensor_flow = self.runtime.config.tensor_flow
                self.runtime.readers['dit'] = reader
                specs = tp_specs(reader, 'dit', self.runtime.rank, self.runtime.config.layers)
                self.runtime.cache.begin_request(specs, discard_unwanted=True)
            self.hop = hop
        return dict(hop=hop, sigma=sigma, identity=self.runtime.readers['dit'].identity)

    def prepare(self, args, kwargs):
        if not self.runtime.sample_active:
            raise ValueError('PTD denoising requires the sampling session')
        if self.hop != len(self.calls):
            raise ValueError('Select the next PTD hop before calling the denoiser; CFG must be 1')
        self.runtime._phase('dit')
        if self.options.hop_handoff == 'queue' and self.queue is None:
            from .queue import ExpertQueue
            self.queue = ExpertQueue(self)
        sigma = float(args[1].flatten()[0]) / 1000.
        expected = float(self.bank.sigmas[self.hop])
        if abs(sigma - expected) > 2e-6:
            raise ValueError('Denoiser timestep differs from active PTD expert')
        payload = dict(kwargs.get('minimax_payload') or {})
        context = payload.pop('ptd_text_states', None)
        if context is None or context.shape[-1] != 5120:
            raise ValueError('PTD needs raw TE conditioning; per-hop refiner cannot reuse a previous hop')
        from patches.comfy_tp_runtime import packet
        context = packet(context, self.runtime.device)
        self.runtime.window.extend_uses([self.runtime.preprocessing.refine_key] if self.runtime.rank == 0 else [])
        kwargs = dict(kwargs, context=self.runtime.model.preprocess_text_embeds(context))
        options = dict(kwargs.get('transformer_options') or {})
        ratio = self.bank.recipe['audio_shift'] / self.bank.recipe['video_shift']
        options.update(minimax_h3_sigma_shift_video=1., minimax_h3_sigma_shift_audio=native_audio_shift(expected, ratio))
        kwargs['transformer_options'] = options
        kwargs['minimax_payload'] = dict(payload, audio_scale=1.)
        if self.head_hook is None:
            next_sigma = float(self.bank.sigmas[self.hop + 1])
            chord = (audio_sigma(expected, ratio) - audio_sigma(next_sigma, ratio)) / (expected - next_sigma)
            factor = chord if self.options.audio_chord_correction else 1.
            # Match the reference's head-output multiplication BEFORE the
            # Native latent-storage cast, not a folded weight multiplication.
            if self.queue is not None:
                self.queue.final.audio_factor = factor
            else:
                self.head_hook = self.runtime.model.final_layer.audio_out.register_forward_hook(
                    lambda module, inputs, output: output * factor)
        self.calls.append(self.hop)
        self.runtime.events.append(dict(phase='ptd_hop', hop=self.hop, sigma=expected,
            refiner_each_hop=True, online_lora=False,hop_handoff=self.options.hop_handoff,
            pruned_adaln_approximation=self.options.allow_pruned_adaln))
        return args, kwargs

    def close(self):
        if self.head_hook is not None:
            self.head_hook.remove(); self.head_hook = None
        if self.queue is not None:self.queue.finish()
        if (self.options.hop_handoff == 'rebuild' or any(getattr(self.runtime,name,None) is not None
                for name in ('model','window','preprocessing'))):
            self.runtime._drop_model()
        self.queue = None
        self.runtime.readers['dit'] = self.original
        if hasattr(self, 'original_config'):
            self.runtime.config = self.original_config
        self.runtime.cache.begin_request((), discard_unwanted=True)


def control(runtime, command, args):
    if command == 'ptd_configure':
        if runtime.sample_active or getattr(runtime, '_ptd_adapter', None) is not None:
            raise ValueError('PTD configuration requires an idle, unpatched session')
        runtime.stop_idle(keep=False)
        runtime._ptd_adapter = WorkerAdapter(runtime, *args)
        return dict(enabled=True, hop=0)
    adapter = getattr(runtime, '_ptd_adapter', None)
    if adapter is None:
        raise ValueError('PTD is not configured')
    if command == 'ptd_hop':
        return adapter.select(*args)
    if command == 'ptd_state':
        return dict(hop=adapter.hop, calls=adapter.calls,
                    queue=adapter.queue.state() if adapter.queue is not None else None)
    if command == 'ptd_release':
        if runtime.sample_active:
            raise ValueError('End the sample before releasing its PTD adapter')
        adapter.close(); runtime._ptd_adapter = None
        return dict(released=True)
    raise ValueError('Unknown PTD service command')


class PTDRemoteDenoiser(RemoteDenoiser):
    def __init__(self, service, controller):
        super().__init__(service)
        self.controller = controller

    def preprocess_text_embeds(self, context):
        if context.shape[-1] != 5120:
            raise ValueError('PTD requires raw TE embeddings')
        return context  # Refine once per active expert inside the rank service.

    def forward(self, *args, **kwargs):
        if self.controller.selection.get() is None:
            raise ValueError('Use the PTD sampler with this PTD model')
        return super().forward(*args, **kwargs)


class Controller:
    def __init__(self, service, path, options, *, expected_bank_sha256=None):
        self.service, self.path, self.options = service, str(Path(path).resolve()), options
        self.bank = PTDBank(path, allow_fixture=options.allow_fixture, expected_sha256=expected_bank_sha256)
        self.sigmas = self.bank.sigmas
        self.hops = 8
        self.selection = ContextVar(f'ptd_tp2_{id(self)}', default=None)

    def denoiser(self, service):
        return PTDRemoteDenoiser(service, self)

    def begin(self):
        self.service.call('ptd_configure', self.path, asdict(self.options), self.bank.digest)

    def conditions(self, conditions, inputs):
        import comfy.conds
        raw = inputs.get('cross_attn')
        if raw is None or raw.shape[-1] != 5120:
            raise ValueError('PTD requires raw Qwen3-VL conditioning')
        result = dict(conditions)
        # Native casts c_crossattn to the residual dtype before dispatch.
        # Carry raw TE values in its existing FP32-safe payload instead; the
        # worker refines them with each hop's baked refiner before any cast.
        payload = dict(result['minimax_payload'].cond, ptd_text_states=raw.float())
        result['minimax_payload'] = comfy.conds.CONDConstant(payload)
        result['c_crossattn'] = comfy.conds.CONDRegular(torch.zeros_like(raw, dtype=torch.float16))
        return result

    def end(self):
        self.service.call('ptd_release')

    @contextmanager
    def activate(self, hop, sigma):
        self.service.call('ptd_hop', hop, sigma)
        token = self.selection.set((hop, sigma))
        try:
            yield self
        finally:
            self.selection.reset(token)


def adapt_model(model, service, path, options, *, expected_bank_sha256=None):
    if not options.enabled:
        return model, None
    if (getattr(model.model.diffusion_model,'service',None) is not service
            or isinstance(model.model.diffusion_model, PTDRemoteDenoiser) or model.patches or model.forced_hooks):
        raise ValueError('Use an unpatched MODEL from this TP2 session, without another LoRA')
    if set(model.object_patches) - {'model_sampling'}:
        raise ValueError('PTD does not silently discard additional MODEL object patches')
    # A fresh proxy owns its own session wrapper. Retain callers' supported
    # transformer settings; additional wrappers still meet the existing gate.
    if set(model.model_options) - {'transformer_options'}:
        raise ValueError('Additional guider/CFG model options require PTD qualification')
    if any(kind != 'outer_sample' or set(entries) != {'minimax_tp2_session'}
           for kind, entries in model.wrappers.items()):
        raise ValueError('Additional sampler wrappers require PTD qualification')
    controller = Controller(service, path, options, expected_bank_sha256=expected_bank_sha256)
    # Create a private existing Native proxy shell; never mutate the MODEL
    # supplied by the caller or globally replace a Native implementation.
    clone = create_model(service, device=model.load_device, adapter=controller)
    clone.model_options['transformer_options'] = copy.deepcopy(model.model_options['transformer_options'])
    sampling = copy.deepcopy(model.get_model_object('model_sampling'))
    sampling.audio_shift = sampling.shift  # physical audio carry and VAE export
    clone.add_object_patch('model_sampling', sampling)
    return clone, controller
