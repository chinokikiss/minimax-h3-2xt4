"""One bounded residency queue across PTD experts; Native math stays intact."""
from contextlib import contextmanager
import math

import torch
from torch import nn

from patches.weight_residency import WeightSpec, copy_module, module_bytes
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


def _refiner_targets(reader):
    targets = tuple(sorted(reader.bank.hops[reader.hop][1]['auxiliary']['verified_targets']))
    if len(targets) != 8 or any(not name.startswith('token_refiner.blocks.') for name in targets):
        raise ValueError('PTD queue requires exactly eight baked refiner projections per hop')
    return targets


def refiner_static_group(canonical, targets):
    """One GPU shell for immutable refiner tensors, excluding all hop projections."""
    group = copy_module(nn.ModuleDict({name: canonical[name]
        for name in ('condition_proj', 'token_refiner')}), lambda tensor: tensor)
    for target in targets:
        weight = _projection(group['token_refiner'], target.removeprefix('token_refiner.'))
        if getattr(weight, 'bias', None) is not None:
            raise ValueError('Queued refiner requires frozen bias-free projections')
        weight.weight = nn.Parameter(torch.empty(0, dtype=weight.weight.dtype), requires_grad=False)
    return group


def refiner_projection_group(canonical, reader, flow, targets):
    """Only the eight changing FP32 projection weights; no duplicated static CPU backing."""
    weights = []
    batch = (reader.read_tensors(tuple(target+'.weight' for target in targets))
             if reader.hop > 0 and reader.bank.http_range2_enabled else {})
    for target in targets:
        projection = _projection(canonical['token_refiner'], target.removeprefix('token_refiner.'))
        if reader.hop == 0:
            # Hop 0 was already decoded, folded and retained by Native _phase.
            value = projection.weight.detach()
        else:
            value = (batch[target+'.weight'] if batch else reader.read_tensor(target+'.weight')).to(dtype=projection.weight.dtype)
            if flow.enabled and target.endswith(('attn.out_proj', 'mlp.fc2')):
                value = value / flow.divisor
        if value.shape != projection.weight.shape:
            raise ValueError('PTD projection does not match canonical refiner shape')
        weights.append(nn.Parameter(value.contiguous(), requires_grad=False))
    return nn.ParameterList(weights)


def head_group(templates, reader):
    result = nn.ModuleDict()
    batch = (reader.read_tensors(tuple(f'final_layer.{name}.{attribute}'
             for name in templates for attribute in ('weight', 'bias')))
             if reader.bank.http_range2_enabled else {})
    for name, template in templates.items():
        head = copy_module(template, lambda tensor:tensor)
        for attribute in ('weight', 'bias'):
            key = f'final_layer.{name}.{attribute}'
            value = batch[key] if batch else reader.read_tensor(key)
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
        self.split_refiner = runtime.rank == 0 and adapter.options.use_hop_refiner
        self.static_refine_spec = None
        if self.split_refiner:
            self.refiner_targets = _refiner_targets(runtime.readers['dit'])
            static_bound = module_bytes(refiner_static_group(self.preprocessing.canonical,
                self.refiner_targets)) + 2**20
            self.static_refine_spec = WeightSpec((self.refine_spec.key, 'ptd_static'),
                lambda: refiner_static_group(self.canonical, self.refiner_targets),
                static_bound, prepare_peak_bytes=static_bound, resident_backing=True)
        self.initial_blocks = tuple(self.window.specs[key] for key in runtime.keys)
        self.readers = {0:runtime.readers['dit']}
        self.plans, self.starts, self.block_keys, self.refine_keys, self.head_keys = {}, {}, {}, {}, {}
        native_final = runtime.model.final_layer
        self.initial_heads = getattr(runtime, '_ptd_initial_heads', None)
        self.head_templates = {name:copy_module(getattr(native_final,name),lambda tensor:tensor.to('meta'))
                               for name in ('video_out','audio_out')}
        self.final = QueuedFinalLayer(native_final, self.view)
        if self.split_refiner:
            self.preprocessing.ptd_refiner_context = self._refine_context
        runtime.model.final_layer = self.final
        runtime.window = self.view
        self.preprocessing.window = self.view
        # Every expert's unique groups retire at their last recorded use.
        # Shared Native input projection stays reusable under the same guards.
        keep = {self.project_spec.key} if self.project_spec is not None else set()
        if self.split_refiner:
            keep.add(self.static_refine_spec.key)
        elif self.refine_spec is not None and not adapter.options.use_hop_refiner:
            keep.add(self.refine_spec.key)
        self.window.retain_keys = frozenset(keep)
        self._append(0)
        if not adapter.options.use_pdd_heads:
            self.window.retain_keys |= {self.head_keys[0]}
        self._append(1)
        if adapter.options.future_cache:
            for hop in range(2, 8): self._append(hop)
            priorities = ({self.starts[hop]+int(self.split_refiner):self.refine_keys[hop] for hop in range(1,8)}
                if runtime.rank == 0 and adapter.options.prefetch_refiner else {})
            self.window.configure_cpu_schedule(refiner_by_use=priorities)
        self.select(0)

    @contextmanager
    def _refine_context(self):
        """Lease retained static refiner, then bind the selected GPU projections."""
        # WeightWindow advances its ordered cursor when a lease exits, not
        # when it enters. The static group remains resident via retain_keys.
        with self.view.use(self.static_refine_spec.key) as shared:
            pass
        with self.view.use(self.preprocessing.refine_key) as weights:
            if len(weights) != len(self.refiner_targets):
                raise RuntimeError('Queued PTD projection count changed')
            previous, model = [], self.runtime.model
            original = {name:getattr(model,name) for name in shared}
            try:
                for target, weight in zip(self.refiner_targets, weights):
                    projection = _projection(shared['token_refiner'],
                        target.removeprefix('token_refiner.'))
                    previous.append((projection, projection.weight))
                    projection.weight = weight
                for name,module in shared.items():setattr(model,name,module)
                yield
            finally:
                for projection,weight in reversed(previous):projection.weight = weight
                for name,module in original.items():setattr(model,name,module)

    def reader(self, hop):
        if hop not in self.readers:
            reader = self.adapter.bank.reader(hop, refiner=self.adapter.options.use_hop_refiner,
                heads=self.adapter.options.use_pdd_heads, base_reader=self.adapter.original)
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
            if self.split_refiner:
                targets = _refiner_targets(reader)
                if targets != self.refiner_targets:
                    raise ValueError('PTD refiner projection layout differs between hops')
                dynamic_bytes = sum(math.prod(_projection(self.canonical['token_refiner'],
                    target.removeprefix('token_refiner.')).weight.shape) * 4 for target in targets)
                largest = max((math.prod(_projection(self.canonical['token_refiner'],
                    target.removeprefix('token_refiner.')).weight.shape) * 4 for target in targets))
                refiner = WeightSpec(((reader.identity,runtime.config.tensor_flow),'ptd_refiner',runtime.rank),
                    lambda reader=reader: refiner_projection_group(self.canonical, reader,
                        runtime.config.tensor_flow, self.refiner_targets),
                    dynamic_bytes + 2**20, prepare_peak_bytes=dynamic_bytes + largest + 16*2**20,
                    resident_backing=hop == 0, pin_backing=False)
                specs += [self.static_refine_spec, refiner, self.project_spec]
                uses += [self.static_refine_spec.key,refiner.key]
            else:
                refiner = self.refine_spec
                specs += [refiner, self.project_spec]
                uses += [refiner.key]
            self.refine_keys[hop] = refiner.key
            uses += [self.project_spec.key]*self.preprocessing.project_uses
        uses += [spec.key for spec in blocks]
        if hop and not self.adapter.options.use_pdd_heads:
            head_key = self.head_keys[0]
            heads = self.window.specs[head_key]
        else:
            head_key = (reader.identity,'ptd_heads',runtime.rank)
            head_bytes = sum(math.prod(reader.header[f'final_layer.{name}.{attribute}']['shape'])*4
                             for name in ('video_out','audio_out') for attribute in ('weight','bias'))
            if hop == 0 and self.initial_heads is not None:
                heads = WeightSpec(head_key, lambda: self.initial_heads,
                    head_bytes+2**20, prepare_peak_bytes=head_bytes+2**20, resident_backing=True)
            else:
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
        if runtime.rank == 0:
            self.preprocessing.refine_key = self.refine_keys[hop]
            self.preprocessing.refine_uses = ((self.static_refine_spec.key,self.refine_keys[hop])
                if self.split_refiner else (self.refine_keys[hop],))
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
        if self.adapter.options.future_cache:
            self.runtime.cache.set_access_plan(self.window.uses, len(self.window.uses))
        self.runtime.cache.begin_request((), discard_unwanted=True)
        self.runtime.cache.wait_idle(timeout=self.runtime.config.timeout_seconds)
        if self.adapter.options.future_cache:self.runtime.cache.reset_access_plan()
        # Break preparation closures and the final-layer/window owner cycle
        # before Native stage teardown releases its auxiliary reservation.
        self.window.specs.clear()
        self.head_templates.clear()
        if hasattr(self.preprocessing, 'ptd_refiner_context'):
            self.preprocessing.ptd_refiner_context = None
            self.preprocessing.refine_uses = ()
        self.initial_heads = None
        self.canonical = self.preprocessing = self.final = None
        self.readers.clear()
        self.finished = True
