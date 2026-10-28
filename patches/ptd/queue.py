"""One bounded residency queue across PTD experts; Native math stays intact."""
from dataclasses import replace
import math

from torch import nn

from patches.weight_residency import WeightSpec, copy_module
from patches.windowed_tp import tp_specs


def _projection(refiner, name):
    parent, _, attribute = name.rpartition('.')
    owner = refiner.get_submodule(parent)
    return getattr(getattr(owner, 'base', owner), attribute)


def refiner_group(canonical, reader, flow):
    """Same load-to-FP32 and output-weight folding as Native construction.

    Immutable norms/condition projection share CPU backing. Only the selected
    hop's eight refiner projection matrices are replaced. No online LoRA,
    quantization or repeated normalization/epsilon folding is performed.
    """
    group = copy_module(nn.ModuleDict({name:canonical[name]
        for name in ('condition_proj', 'token_refiner')}), lambda tensor:tensor)
    targets = reader.bank.hops[reader.hop][1]['auxiliary']['verified_targets']
    for target in targets:
        relative = target.removeprefix('token_refiner.')
        module = _projection(group['token_refiner'], relative)
        if getattr(module, 'bias', None) is not None:
            raise ValueError('Queued refiner requires the frozen bias-free projections')
        value = reader.read_tensor(target+'.weight').to(dtype=module.weight.dtype)
        if flow.enabled and relative.endswith(('attn.out_proj', 'mlp.fc2')):
            value = value / flow.divisor
        module.weight = nn.Parameter(value.contiguous(), requires_grad=False)
    return group


def head_group(templates, reader):
    result = nn.ModuleDict()
    for name, template in templates.items():
        head = copy_module(template, lambda tensor:tensor)
        for attribute in ('weight', 'bias'):
            value = reader.read_tensor(f'final_layer.{name}.{attribute}')
            setattr(head, attribute, nn.Parameter(value.to(dtype=getattr(template,attribute).dtype),
                                                 requires_grad=False))
        result[name] = head
    return result


class PlannedWindow:
    """Adapt Native's per-call append notifications to an already queued plan."""
    def __init__(self, window): self.base = window

    def __getattr__(self, name): return getattr(self.base, name)

    def extend_uses(self, keys):
        keys = tuple(keys)
        if self.base.closed or self.base.uses[self.base.cursor:self.base.cursor+len(keys)] != keys:
            raise RuntimeError('Native call does not match the selected PTD queue prefix')


class QueuedFinalLayer(nn.Module):
    def __init__(self, native, window):
        super().__init__()
        self.native, self.window = native, window
        self.head_key, self.audio_factor = None, 1.
        # Norm/AdaLN stay in the existing shell. Only the hop's complete heads
        # are leased here, after the final DiT block, including both biases.
        native.video_out = nn.Identity(); native.audio_out = nn.Identity()

    def forward(self, *args, **kwargs):
        with self.window.use(self.head_key) as heads:
            old_video, old_audio = self.native.video_out, self.native.audio_out
            self.native.video_out, self.native.audio_out = heads['video_out'], heads['audio_out']
            try:
                video, audio = self.native(*args, **kwargs)
                return video, audio * self.audio_factor
            finally:
                self.native.video_out, self.native.audio_out = old_video, old_audio


class ExpertQueue:
    def __init__(self, adapter):
        self.adapter, self.runtime = adapter, adapter.runtime
        self.finished = False
        runtime = self.runtime
        self.window = runtime.window
        if self.window.uses or self.window.cursor:
            raise RuntimeError('Install the PTD queue before its first refinement')
        self.view = PlannedWindow(self.window)
        self.preprocessing = runtime.preprocessing
        self.canonical = self.preprocessing.canonical
        self.refine_spec = self.window.specs.get(self.preprocessing.refine_key)
        self.project_spec = self.window.specs.get(self.preprocessing.project_key)
        self.initial_blocks = tuple(self.window.specs[key] for key in runtime.keys)
        self.readers = {0:runtime.readers['dit']}
        self.plans, self.starts, self.block_keys, self.refine_keys, self.head_keys = {}, {}, {}, {}, {}
        native_final = runtime.model.final_layer
        self.head_templates = {name:copy_module(getattr(native_final,name),lambda tensor:tensor.to('meta'))
                               for name in ('video_out','audio_out')}
        self.final = QueuedFinalLayer(native_final, self.view)
        runtime.model.final_layer = self.final
        runtime.window = self.view
        self.preprocessing.window = self.view
        # Every expert's unique groups retire at their last recorded use.
        # Shared Native input projection stays reusable under the same guards.
        keep = {self.project_spec.key} if self.project_spec is not None else set()
        if self.refine_spec is not None and not adapter.options.use_hop_refiner:
            keep.add(self.refine_spec.key)
        self.window.retain_keys = frozenset(keep)
        self._append(0)
        self._append(1)
        self.select(0)

    def reader(self, hop):
        if hop not in self.readers:
            reader = self.adapter.bank.reader(hop, refiner=self.adapter.options.use_hop_refiner,
                heads=self.adapter.options.use_pdd_heads, base_reader=self.adapter.original, lazy_overlay=True)
            reader.operator_policy = self.runtime.config.operators
            reader.tensor_flow = self.runtime.config.tensor_flow
            self.readers[hop] = reader
        return self.readers[hop]

    def _append(self, hop):
        if hop >= 8 or hop in self.plans: return
        reader, runtime = self.reader(hop), self.runtime
        blocks = self.initial_blocks if hop == 0 else tuple(tp_specs(reader, 'dit', runtime.rank, runtime.config.layers))
        specs, uses = list(blocks), []
        if runtime.rank == 0:
            refiner = self.refine_spec
            if hop and self.adapter.options.use_hop_refiner:
                # Native auxiliary construction is not repeated. This one
                # pageable group transfers its existing cache reservation to
                # the GPU window until its H2D ready event completes.
                largest = max((math.prod(info['shape']) * 4
                    for name,info in reader.header.items() if name.startswith('token_refiner.blocks.')
                    and name.endswith('.weight')), default=0)
                refiner = replace(refiner, key=((reader.identity,runtime.config.tensor_flow),'ptd_refiner',runtime.rank),
                    prepare=lambda reader=reader:refiner_group(self.canonical, reader, runtime.config.tensor_flow),
                    prepare_peak_bytes=refiner.max_bytes+largest+16*2**20,
                    resident_backing=False, pin_backing=False)
            self.refine_keys[hop] = refiner.key
            specs += [refiner, self.project_spec]
            uses += [refiner.key] + [self.project_spec.key]*self.preprocessing.project_uses
        uses += [spec.key for spec in blocks]
        head_key = (reader.identity,'ptd_heads',runtime.rank)
        head_bytes = sum(math.prod(reader.header[f'final_layer.{name}.{attribute}']['shape'])*4
                         for name in ('video_out','audio_out') for attribute in ('weight','bias'))
        heads = WeightSpec(head_key, lambda reader=reader:head_group(self.head_templates,reader),
            head_bytes+2**20, prepare_peak_bytes=2*head_bytes+2**20)
        specs.append(heads); uses.append(head_key)
        self.starts[hop] = len(self.window.uses)
        self.window.append_plan(tuple({spec.key:spec for spec in specs}.values()),uses)
        self.plans[hop], self.block_keys[hop], self.head_keys[hop] = tuple(uses), [spec.key for spec in blocks], head_key

    def select(self, hop):
        self._append(hop)
        if self.window.cursor != self.starts[hop]:
            raise RuntimeError('Previous expert did not finish its queued weight uses')
        self._append(hop+1)
        runtime = self.runtime
        runtime.readers['dit'] = self.reader(hop)
        runtime.keys = self.block_keys[hop]
        for block,key in zip(runtime.model.blocks,runtime.keys): block.key = key
        if runtime.rank == 0: self.preprocessing.refine_key = self.refine_keys[hop]
        self.final.head_key = self.head_keys[hop]
        # No trim, close, GC, global CUDA sync or allocator flush at a hop.
        runtime.cache.begin_request([self.window.specs[key] for key in
            dict.fromkeys(self.window.uses[self.window.cursor:])], discard_unwanted=False)

    def state(self):
        return dict(window_id=id(self.window),shell_id=id(self.runtime.model),
            queued_hops=sorted(self.plans),cursor=self.window.cursor,planned_uses=len(self.window.uses),
            peak_live_weight_bytes=self.window.stats['peak_live_weight_bytes'],
            host_wait_ms=self.window.stats['host_wait_ms'],h2d_bytes=self.window.stats['h2d_bytes'])

    def finish(self):
        """Drain once at sample end, before releasing shared CPU auxiliary backing."""
        if self.finished: return
        # Release completed copy-source leases before waiting for CPU work
        # which may itself need that host-budget headroom.
        self.window.copy_stream.synchronize()
        self.window._reap()
        self.window.close()
        self.runtime.cache.begin_request((), discard_unwanted=True)
        self.runtime.cache.wait_idle(timeout=self.runtime.config.timeout_seconds)
        # Break preparation closures and the final-layer/window owner cycle
        # before Native stage teardown releases its auxiliary reservation.
        self.window.specs.clear()
        self.head_templates.clear()
        self.canonical = self.preprocessing = self.final = None
        self.readers.clear()
        self.finished = True
