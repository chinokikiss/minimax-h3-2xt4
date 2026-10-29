"""Persistent TP2 ranks behind Native ComfyUI conditioning and denoising.

The first transport uses bounded synchronous CPU messages. This preserves
semantics and gives cancellation/error boundaries; it is not an overlap claim.
Worker weights and pageable caches persist independently of ComfyUI's proxies.
"""
from __future__ import annotations

import atexit
import copy
import gc
import math
from dataclasses import dataclass,field
from pathlib import Path
from datetime import timedelta
import threading
import time
import traceback
import uuid
import weakref

import torch
from torch import nn

from patches.host_weight_cache import HostBudget, HostWeightCache, GiB
from patches.diagnostic_metrics import RecentEvents
from patches.runtime_diagnostics import rank_snapshot, idle_snapshot
# Re-export factory names for existing standalone qualification callers.
from patches.comfy_tp_models import native_te, native_dit, StagedVision, CPUStartupState
from patches.weight_residency import ResidencyWindow, WeightSpec, copy_module, module_bytes
from patches.windowed_tp import CheckpointReader, tp_specs, attach_window, DiTPreprocessing
from patches.h3_operator_policy import H3OperatorPolicy
from patches.h3_tensor_flow import TensorFlowPolicy


@dataclass(frozen=True)
class RuntimeConfig:
    te_path: str
    dit_path: str
    layers: int = 50
    weight_bytes: int = 6 * GiB
    pinned_bytes: int = 6 * GiB
    cache_bytes: int = 3 * GiB
    host_bytes: int = 16 * GiB
    reserve_bytes: int = 9 * GiB
    timeout_seconds: int = 1500
    te_weight_bytes: int | None = None
    cpu_lookahead: int = 16
    gpu_lookahead: int = 2
    cross_request_te: bool = True
    rank_checkpoint_dir: str | None = None
    cross_model_cpu: bool = True
    prefetch_lifecycle: bool = True
    memory_pressure_aware: bool = True
    decode_te_blocks: int = 2
    idle_residency: bool = False
    idle_weight_bytes: int = 4*GiB
    idle_reserve_bytes: int = 4*GiB
    idle_delay_seconds: float = .25
    lora_stack: tuple = ()
    lora_signature: tuple = field(init=False,default=())
    lora_row_partial_reduce: bool = True
    operators: H3OperatorPolicy = field(default_factory=H3OperatorPolicy)
    adaptive_lookahead: bool = True
    tensor_flow: TensorFlowPolicy = field(default_factory=lambda: TensorFlowPolicy('fp16'))
    # Decode has no DiT activation window; do not inherit its sampling reserve.
    video_decode_reserve_bytes: int = 2 * GiB
    bundle_offline_lora: bool = False
    lora_selective_qkv_b: bool = True

    def __post_init__(self):
        if not isinstance(self.tensor_flow,TensorFlowPolicy):raise ValueError('TensorFlowPolicy required')
        if self.tensor_flow.mode=='bf16_reference':raise ValueError('BF16 reference is a local component test mode, not a T4 deployment mode')
        if not isinstance(self.operators,H3OperatorPolicy):raise ValueError('H3OperatorPolicy required')
        object.__setattr__(self,'lora_signature',tuple((str(Path(p).resolve()),Path(p).stat().st_size,
            Path(p).stat().st_mtime_ns) for p,_ in self.lora_stack))
        if not 1 <= self.layers <= 50:
            raise ValueError('Expected 1–50 trained layers')
        if min(self.weight_bytes, self.pinned_bytes, self.cache_bytes, self.host_bytes) <= 0:
            raise ValueError('Positive memory budgets are required')
        if self.te_weight_bytes is not None and self.te_weight_bytes <= 0:
            raise ValueError('Positive TE memory budget is required')
        if not 1 <= self.cpu_lookahead <= 32 or not 1 <= self.gpu_lookahead <= 2:
            raise ValueError('CPU horizon must be 1–32 and JIT GPU horizon 1–2')
        if not 1 <= self.decode_te_blocks <= 4:
            raise ValueError('Decode TE prefix must be 1–4 blocks')
        if self.idle_weight_bytes<=0 or self.idle_reserve_bytes<0 or self.idle_delay_seconds<0:
            raise ValueError('Invalid idle residency bounds')
        if self.idle_residency and not self.prefetch_lifecycle:
            raise ValueError('Idle residency requires P0 lifecycle')
        if self.video_decode_reserve_bytes < 0:
            raise ValueError('Video decode reserve must be nonnegative')
        if self.timeout_seconds <= 0 or self.reserve_bytes < 0:
            raise ValueError('Positive command timeout and nonnegative DiT reserve required')


def packet(value, device):
    """Copy Native tensor containers without changing dtypes or layout metadata."""
    from comfy.ldm.minimax.model import PackedLayout
    from comfy.nested_tensor import NestedTensor
    if isinstance(value, PackedLayout):
        result = copy.copy(value)
        result.__dict__ = packet(value.__dict__, device)
        return result
    if isinstance(value, NestedTensor):
        return NestedTensor(packet(value.unbind(), device))
    if isinstance(value, dict):
        return {k: packet(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        result = [packet(v, device) for v in value]
        return tuple(result) if isinstance(value, tuple) else result
    if isinstance(value, torch.Tensor):
        # CUDA IPC is deliberately avoided; rank ownership is explicit.
        return value.detach().to(device, copy=True)
    if value is None or isinstance(value, (str, int, float, bool, torch.dtype, torch.device, uuid.UUID)):
        return value
    raise TypeError(f'Unsupported worker payload: {type(value).__name__}; custom patches require qualification')


def worker_reply(result, rank, command):
    """Rank 0, diagnostics and independent temporal owners return CPU payloads."""
    if rank != 0 and command not in ('stats', 'idle_stats', 'video_temporal_pair'):
        return None
    return packet(result, 'cpu')


class RankRuntime:
    def __init__(self, config, rank, device, comm, budget):
        self.config, self.rank, self.device, self.comm = config, rank, torch.device(device), comm
        from patches.tp2_bundle import open_checkpoint,BundleReader
        self.readers = {'te': open_checkpoint(config.te_path), 'dit': open_checkpoint(config.dit_path)}
        for reader in self.readers.values():
            reader.operator_policy=config.operators
            reader.tensor_flow=config.tensor_flow
        if config.bundle_offline_lora:
            if not isinstance(self.readers['dit'],BundleReader):raise ValueError('Bundle offline LoRA requires a TP2 descriptor')
            self.readers['dit'].set_offline_lora(config.lora_stack)
        elif config.lora_stack:
            from patches.h3_lora import H3LoRAStore
            self.readers['dit'].lora_store=H3LoRAStore(config.lora_stack,self.readers['dit'],
                row_partial=config.lora_row_partial_reduce,selective_b=True,
                selective_qkv_b=config.lora_selective_qkv_b)
        if config.rank_checkpoint_dir:
            for reader in self.readers.values():
                reader.rank_checkpoint_dir = config.rank_checkpoint_dir
        self.cache = HostWeightCache(budget, config.cache_bytes,reuse_aware=config.prefetch_lifecycle)
        # Separate bounded startup cache: a large front end must not consume
        # the transformer's entire preparation queue capacity.
        startup_bytes = sum(v['data_offsets'][1]-v['data_offsets'][0]
            for k,v in self.readers['te'].header.items()
            if k != '__metadata__' and not k.startswith('model.layers.')) + 32*2**20
        reader = self.readers['te']
        names = [n for n in reader.header if n != '__metadata__' and not n.startswith('model.layers.')]
        self.startup_specs = []
        # Vision first, then embeddings/metadata. Each independently admitted
        # group can make progress when the complete shell would not fit.
        for label,selected in [('vision',[n for n in names if n.startswith('visual.')]),
                               ('embedding',[n for n in names if not n.startswith('visual.')])]:
            bound = sum(reader.header[n]['data_offsets'][1]-reader.header[n]['data_offsets'][0]
                        for n in selected) + 2**20
            self.startup_specs.append(WeightSpec((reader.identity,'te_startup',label,rank),
                lambda selected=tuple(selected):CPUStartupState(reader,selected),bound,
                prepare_peak_bytes=bound+2**20))
        self.startup_cache = HostWeightCache(budget,startup_bytes)
        self.model = self.window = self.preprocessing = None
        self.kind = None
        self.aux_reservation = 0
        self.sample_active = False
        self.events = RecentEvents(16)
        self.calls = 0
        self.completed_decode = False
        self.decode_lookahead_started = False
        self.vae_model = None
        self.vae_kind = None
        self.vae_numeric_flags = None
        self.idle_window=None
        self.idle_after=None
        self.idle_stats=dict(starts=0,cancellations=0,ticks=0,loads=0,errors=0,
                             started_ns=None,last_load_ns=None)
        self.idle_identity_checked=0.

    def _check_idle_identity(self):
        reader=self.readers['te']
        if hasattr(reader,'check'):reader.check()
        stat=reader.path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=reader.signature:
            raise RuntimeError('TE checkpoint changed; reload the TP2 session before reuse')
        from pathlib import Path
        for name,signature in getattr(reader,'rank_offline_verified',{}).items():
            path=Path(name);stat=path.stat();manifest=path.with_suffix('.json').stat()
            if (stat.st_size,stat.st_mtime_ns,manifest.st_mtime_ns,manifest.st_size)!=signature:
                raise RuntimeError('Rank-local checkpoint changed; reload the TP2 session before reuse')

    def stop_idle(self,keep=False):
        self.idle_after=None
        if self.idle_window is not None:
            self.idle_window.cancel_idle(keep=keep)
            self.idle_stats['cancellations']+=1

    def start_idle(self):
        if not self.config.idle_residency or self.sample_active or self.vae_model is not None:
            return False
        if self.kind not in (None,'te'):return False
        self._check_idle_identity()
        if self.idle_window is None:
            if self.kind=='te':self.idle_window=self.window
            else:
                specs=tp_specs(self.readers['te'],'te',self.rank,self.config.layers)
                self.idle_window=ResidencyWindow(specs,(),self.device,
                    max_weight_bytes=self.config.te_weight_bytes or self.config.weight_bytes,
                    pinned_bytes=self.config.pinned_bytes,reserve_bytes=self.config.reserve_bytes,
                    host_cache=self.cache,adaptive=self.config.adaptive_lookahead,
                    lookahead=self.config.gpu_lookahead,cpu_lookahead=self.config.cpu_lookahead)
        window=self.idle_window
        if window.idle_mode:return True
        # Normal request order protects early weights; no predicted prompt or
        # conditioning tensors are allocated. A known queued switch skips this.
        keys=tuple(window.specs)
        self.cache.begin_request(window.specs.values())
        window.begin_idle(keys,max_bytes=self.config.idle_weight_bytes,
            reserve_bytes=self.config.idle_reserve_bytes)
        self.idle_after=time.monotonic()+self.config.idle_delay_seconds
        self.idle_stats.update(started_ns=time.monotonic_ns())
        self.idle_stats['starts']+=1
        return True

    def idle_tick(self):
        if self.idle_after is None or time.monotonic()<self.idle_after:return
        if self.sample_active or self.vae_model is not None:return
        window=self.idle_window
        if window is None or not window.idle_mode:return
        self.idle_stats['ticks']+=1
        try:
            if time.monotonic()-self.idle_identity_checked>=1:
                self._check_idle_identity();self.idle_identity_checked=time.monotonic()
            if window.idle_tick():
                self.idle_stats['loads']+=1;self.idle_stats['last_load_ns']=time.monotonic_ns()
        except Exception:
            self.idle_stats['errors']+=1;self.stop_idle(keep=False)

    def _drop_vae(self):
        self.vae_temporal_planner = None
        if self.vae_model is None and self.vae_kind is None and not getattr(self, 'vae_numeric_flags', None):
            return
        self.vae_model = None
        self.vae_kind = None
        gc.collect()
        torch.cuda.synchronize(self.device)
        if self.idle_window is not None:self.idle_window._reap()
        torch.cuda.empty_cache()
        flags = getattr(self, 'vae_numeric_flags', None)
        if flags:
            for owner, name, value in flags:
                setattr(owner, name, value)
        self.vae_numeric_flags = None

    def _prepare_vae(self, path, kind, activation_bytes, numeric_options=None, parallelism='query'):
        if parallelism not in ('query', 'temporal'):
            raise ValueError(parallelism)
        if kind != 'video':
            return {'accepted':False, 'reason':'Parallel VAE decoding is video-only; Audio VAE uses Native single-GPU decoding'}
        from patches.parallel_vae import checkpoint_plan, load_decode_weights
        if self.sample_active:
            raise RuntimeError('VAE preparation arrived during sampling')
        self._drop_vae()
        reader, names, reservation = checkpoint_plan(path, kind)
        reader.tensor_flow=self.config.tensor_flow
        payload = (reservation - 256 * 2**20) // 2
        # Parent owns only Native planning/normalization. The activation allowance
        # still covers global KV, transport scratch and public output allocations.
        free, total = torch.cuda.mem_get_info(self.device)
        required = payload + 2 * int(activation_bytes) + self.config.video_decode_reserve_bytes
        gpu_ready = free >= required
        ready = gpu_ready
        admitted = False
        if ready:
            admitted = self.cache.reserve_staging(reservation)
            if not admitted:
                self.startup_cache.relieve_pressure(reservation) if self.config.prefetch_lifecycle else self.startup_cache.trim()
                admitted = self.cache.reserve_staging(reservation)
            ready = admitted
        admission = dict(rank=self.rank, free_bytes=free, total_bytes=total,
            weight_bytes=payload, activation_estimate_bytes=int(activation_bytes),
            activation_allowance_bytes=2*int(activation_bytes),
            reserve_bytes=self.config.video_decode_reserve_bytes, required_free_bytes=required,
            staging_bytes=reservation, gpu_accepted=gpu_ready,
            host_accepted=admitted if gpu_ready else None)
        if gpu_ready and not admitted:
            admission['host_budget'] = self.cache.budget.snapshot()
        status = torch.tensor(int(ready), device=self.device, dtype=torch.int32)
        self.comm.all_reduce(status, op=self.comm.ReduceOp.MIN)
        if not status.item():
            if admitted:
                self.cache.budget.finish_staging(reservation)
                self.cache.budget.release(reservation)
            reason = ('VRAM admission' if not gpu_ready else
                      'Host RAM admission' if not admitted else 'Admission declined on another rank')
            self.events.append(dict(phase='vae_prepare', kind=kind, accepted=False,
                reason=reason, admission=admission, monotonic=time.monotonic()))
            return {'accepted':False, 'reason':f'{reason} (rank {self.rank})', 'admission':admission}
        error = None
        try:
            options = numeric_options or {}
            self.vae_numeric_flags = []
            for owner, name, key in (
                (torch.backends.cuda.matmul, 'allow_tf32', 'matmul_tf32'),
                (torch.backends.cudnn, 'allow_tf32', 'cudnn_tf32'),
                (torch.backends.cudnn, 'benchmark', 'cudnn_benchmark'),
                (torch.backends.cudnn, 'deterministic', 'cudnn_deterministic')):
                if key in options:
                    self.vae_numeric_flags.append((owner, name, getattr(owner, name)))
                    setattr(owner, name, options[key])
            if parallelism == 'query':
                self.vae_model = load_decode_weights(reader, names, kind, self.device)
            else:
                self.vae_model = load_decode_weights(reader, names, kind, self.device, parallelism='temporal')
                from patches.staged_vae import video_planning_template
                from patches.video_decode_planner import make_planner
                self.vae_temporal_planner = make_planner(video_planning_template(reader), self.vae_model._decode_pixels)
        except (torch.cuda.OutOfMemoryError, ValueError) as exc:
            error = str(exc)
        finally:
            self.cache.budget.finish_staging(reservation)
            self.cache.budget.release(reservation)
        status.fill_(int(error is None))
        self.comm.all_reduce(status, op=self.comm.ReduceOp.MIN)
        if not status.item():
            self._drop_vae()
            reason=error or 'VAE unavailable on the other rank'
            self.events.append(dict(phase='vae_prepare',kind=kind,accepted=False,
                reason=reason,admission=admission,monotonic=time.monotonic()))
            return {'accepted':False, 'reason':reason, 'admission':admission}
        self.vae_kind = kind
        self.vae_parallelism = parallelism
        self.events.append(dict(phase='vae_prepare',kind=kind,accepted=True,
                                checkpoint=reader.identity,admission=admission,monotonic=time.monotonic()))
        return {'accepted':True, 'kind':kind, 'admission':admission}

    def _drop_model(self):
        if self.window is not None:
            if self.window is self.idle_window:
                self.stop_idle(keep=False);self.idle_window=None
            stats = self.window.close()
            self.events.append({'phase':self.kind, 'calls':self.calls, 'window':stats})
        if self.preprocessing is not None:
            self.preprocessing.close()
        self.window = self.model = self.preprocessing = None
        gc.collect()  # Native bound-method/module cycles, only at stage boundaries
        if self.aux_reservation:
            self.cache.budget.release(self.aux_reservation)
            self.aux_reservation = 0
        self.kind = None
        torch.cuda.synchronize(self.device)
        if self.idle_window is not None:self.idle_window._reap()
        torch.cuda.empty_cache()
        # Retired pin buffers remain in PyTorch's host cache unless purged.
        # Reclaim inactive blocks at stage boundaries, after copy/use fences;
        # checked-out buffers belonging to another rank are left intact.
        clear_host = getattr(torch._C, '_host_emptyCache', None)
        if clear_host is not None:
            clear_host()

    def _phase(self, kind):
        if kind == self.kind:
            return
        self._drop_model()
        reader = self.readers[kind]
        specs = tp_specs(reader, kind, self.rank, self.config.layers)
        self.cache.begin_request(specs)
        prefix = 'model.layers.' if kind == 'te' else 'blocks.'
        exclude = (prefix,)
        if kind == 'dit' and self.rank == 1:
            exclude += ('condition_proj.','token_refiner.','video_patch_proj.','audio_patch_proj.')
        aux_bytes = sum(v['data_offsets'][1]-v['data_offsets'][0] for k,v in reader.header.items()
            if k != '__metadata__' and not k.startswith(exclude)) + 32*2**20
        store=getattr(reader,'lora_store',None) or getattr(reader,'offline_store',None)
        if kind=='dit' and self.rank==0 and store is not None:
            aux_bytes+=store.prefix_bytes('token_refiner.')
        # Native CPU construction + checkpoint state + copy can coexist. The
        # same shared host budget covers these otherwise unmanaged shells.
        # TE MixedPrecision Linear creates no initial dense matrix, and
        # FP16 checkpoint tensors become final CPU backing without a dtype
        # copy. Initial embedding + full checkpoint fit below 2x aux. DiT
        # expands FP16 refiner weights into FP32, requiring the 3x bound.
        reservation = 0 if kind == 'te' else 3*aux_bytes
        if kind=='dit' and hasattr(reader,'workspace_bytes'):
            # Refiner LoRA is merged before construction in the same CPU path.
            active=getattr(reader,'offline_targets',())
            global_work=max((math.prod(reader.header[t+'.weight']['shape'])*32
                for t in active if t.startswith('token_refiner.')),default=0)
            reservation+=global_work
        deadline = time.monotonic() + self.config.timeout_seconds
        while reservation and not self.cache.budget.acquire(reservation,staging=True):
            self.cache.relieve_pressure(reservation) if self.config.prefetch_lifecycle else self.cache.trim()
            if time.monotonic() > deadline:
                raise MemoryError('Host budget cannot admit Native auxiliary construction')
            time.sleep(.02)
        self.aux_reservation = reservation
        auxiliary_pending = reservation
        try:
            if kind == 'te':
                self.startup_cache.begin_request(self.startup_specs)
                # Reclaim transformer's speculative pageable tail before the
                # required embedding shell needs its independent reservation.
                if not self.config.prefetch_lifecycle:
                    self.cache.trim()
                groups = []; startup_state = {}; reservation = 0
                for spec in self.startup_specs:
                    group,size = self.startup_cache.take(spec)
                    groups.append(group); startup_state.update(group.tensors)
                    reservation += size
                    self.aux_reservation = reservation
                # Native's initial embedding coexists with the prefetched raw
                # checkpoint; mixed-precision linears do not allocate dense
                # matrices before loading. Keep casting/dispatch Native.
                extra = reader.header['model.embed_tokens.weight']['data_offsets']
                extra = extra[1]-extra[0]+32*2**20
                deadline = time.monotonic()+self.config.timeout_seconds
                while not self.cache.budget.acquire(extra,staging=True):
                    self.cache.relieve_pressure(extra) if self.config.prefetch_lifecycle else self.cache.trim()
                    if time.monotonic()>deadline:
                        raise MemoryError('Host budget cannot admit TE embedding construction')
                    time.sleep(.02)
                reservation += extra
                auxiliary_pending += extra
                self.aux_reservation = reservation
                self.model = native_te(reader,self.config.layers,startup_state=startup_state)
                vision = self.model.qwen3vl_32b.transformer.visual
                self.model.qwen3vl_32b.transformer.visual = nn.Identity()
                self.model = copy_module(self.model, lambda t: t.to(self.device))
                self.model.qwen3vl_32b.transformer.visual = StagedVision(vision, self.device)
                owner = self.model.qwen3vl_32b.transformer.model
                retained = module_bytes(vision)
                del vision
                del group,groups,startup_state
                gc.collect()  # release the consumed CPU embedding shell now
            else:
                self.model = native_dit(reader, self.rank)
                self.preprocessing = DiTPreprocessing(self.model, self.rank, self.device, self.comm,
                    (reader.identity,store.identity if store is not None else None,'native-int8-scalemax',self.config.operators,self.config.tensor_flow))
                specs += self.preprocessing.specs()
                # Keep canonical preprocessing on CPU; moving the outer shell
                # wholesale would resurrect duplicate refiner/projection weights.
                names = set(self.preprocessing.canonical) | {'condition_proj','token_refiner',
                    'video_patch_proj','audio_patch_proj'}
                for name, child in list(self.model.named_children()):
                    if name not in names:
                        setattr(self.model, name, copy_module(child, lambda t: t.to(self.device)))
                for name, value in list(self.model._parameters.items()):
                    if value is not None:
                        self.model._parameters[name] = nn.Parameter(value.to(self.device), requires_grad=False)
                for name, value in list(self.model._buffers.items()):
                    if value is not None:
                        self.model._buffers[name] = value.to(self.device)
                retained = sum(module_bytes(m) for m in self.preprocessing.canonical.values())
                owner = self.model
            self.cache.budget.finish_staging(auxiliary_pending)
            auxiliary_pending = 0
            self.cache.budget.release(reservation-retained)
            self.aux_reservation = retained
            self.keys = [s.key for s in specs if len(s.key) == 4 and s.key[1] == kind]
            reusable=self.idle_window if kind=='te' and self.config.idle_residency else None
            if reusable is not None:
                if set(reusable.specs)!={s.key for s in specs}:raise RuntimeError('Idle TE identity mismatch')
                reusable.resume();self.window=reusable;self.idle_window=None
            else:self.window = ResidencyWindow(specs, (), self.device,
                max_weight_bytes=(self.config.te_weight_bytes or self.config.weight_bytes)
                    if kind == 'te' else self.config.weight_bytes,
                pinned_bytes=self.config.pinned_bytes,
                reserve_bytes=self.config.reserve_bytes, adaptive=self.config.adaptive_lookahead, host_cache=self.cache,
                lookahead=self.config.gpu_lookahead,cpu_lookahead=self.config.cpu_lookahead,
                retain_keys=[s.key for s in specs] if kind == 'dit' else ())
            attach_window(owner, kind, self.window, self.keys)
            if self.preprocessing:
                self.preprocessing.window = self.window
            self.kind = kind
        except BaseException:
            self.cache.budget.finish_staging(auxiliary_pending)
            if kind == 'te':
                # Raw groups transfer their reservation to this runtime. Drop
                # transient local ownership before teardown releases it.
                if 'groups' in locals():
                    groups.clear()
                if 'startup_state' in locals():
                    startup_state.clear()
                group = None
            self._drop_model()
            raise

    def dispatch(self, command, args=(), kwargs=None):
        if command == 'audio_channels':
            raise RuntimeError('Audio VAE parallel decoding has been removed; use Native single-GPU decoding')
        store=getattr(self.readers['dit'],'lora_store',None) or getattr(self.readers['dit'],'offline_store',None)
        if store is not None and command not in ('close','stats','idle_stop','idle_release'):store.check()
        kwargs = kwargs or {}
        with torch.cuda.device(self.device), torch.inference_mode():
            if command.startswith('ptd_'):
                from patches.ptd.runtime import control
                return control(self,command,args)
            ptd_adapter=getattr(self,'_ptd_adapter',None)
            if command=='denoise' and ptd_adapter is not None:
                args,kwargs=ptd_adapter.prepare(args,kwargs)
            if self.config.idle_residency:
                if command=='idle_begin':return {'accepted':self.start_idle()}
                if command=='idle_stop':
                    self.stop_idle(keep=bool(kwargs.get('keep',False)));return None
                if command=='idle_release':
                    self.stop_idle(keep=False)
                    if self.idle_window is not None:
                        self.idle_window.copy_stream.synchronize();self.idle_window._reap()
                    return None
                if command=='idle_stats':
                    return idle_snapshot(self)
                if command=='te':
                    self.stop_idle(keep=True);self._check_idle_identity()
                    if self.kind=='te' and self.window is self.idle_window:
                        self.window.resume();self.idle_window=None
                elif command in ('begin_sample','vae_prepare','close'):
                    self.stop_idle(keep=False)
                    if command=='vae_prepare' and self.kind=='te':self._drop_model()
                if self.idle_window is not None:self.idle_window._reap()
            if command == 'begin_sample':
                if self.sample_active:
                    raise RuntimeError('Concurrent samples are unsupported')
                self.sample_active, self.completed_decode = True, False
                self.decode_lookahead_started = False
                self._drop_vae()
                self._drop_model()  # release TE CUDA weights before Native sampling
                # Unknown-next-request speculation cannot displace sampling.
                self.startup_cache.begin_request((),discard_unwanted=True)
                # Both ranks relinquish irrelevant TE CPU tails before rank
                # zero needs the multi-GB Native refiner staging group.
                self.cache.begin_request(tp_specs(self.readers['dit'], 'dit', self.rank,
                    self.config.layers), discard_unwanted=True)
            elif command == 'te':
                if self.sample_active:
                    raise RuntimeError('TE cannot run inside a denoising invocation')
                if self.vae_model is not None:
                    self._drop_vae()
                self._phase('te')
                if self.config.idle_residency:self.window.extend_uses(self.keys)
                # A new conditioning call can contain several vision blocks.
                # Drop the previous call's decoder cache before staging vision
                # rather than allowing both ranks to retain competing windows.
                if any(isinstance(t[0],dict) and t[0].get('type') == 'image'
                       for batch in args[0]['qwen3vl_32b'] for t in batch):
                    if self.config.idle_residency:
                        vision=self.model.qwen3vl_32b.transformer.visual.canonical
                        self.window.room_for_external(module_bytes(vision)+2*GiB+self.config.reserve_bytes)
                    else:
                        for key in list(self.window.entries):self.window._evict(key)
                    for entry in self.window.retired:
                        (entry.last_use or entry.ready).synchronize()
                    self.window._reap()
                if not self.config.idle_residency:self.window.extend_uses(self.keys)
                self.model.reset_clip_options()
                self.model.set_clip_options({**kwargs, 'execution_device':self.device})
                output = self.model.encode_token_weights(packet(args[0], self.device))
                if self.config.cross_model_cpu:
                    self.cache.prefetch(tp_specs(self.readers['dit'],'dit',self.rank,
                        self.config.layers)[:self.config.cpu_lookahead])
                return output
            elif command == 'prepare_dit':
                from patches.h3_operator_aot import prepare
                result=prepare(int(args[0]),self.rank,self.config.operators,
                    norm_epsilon=1e-5/self.config.tensor_flow.divisor**2)
                self.events.append(dict(phase='dit_aot',**result))
                return result
            elif command in ('refine', 'denoise'):
                if not self.sample_active:
                    raise RuntimeError('Denoising requires a Native sampler session')
                self._phase('dit')
                args, kwargs = packet(args, self.device), packet(kwargs, self.device)
                if command == 'refine':
                    keys = ([self.preprocessing.refine_key] if self.rank == 0 and args[0].shape[-1] !=5376 else [])
                    self.window.extend_uses(keys)
                    return self.model.preprocess_text_embeds(*args)
                context=kwargs.get('context',args[2] if len(args)>2 else None)
                if context is not None and context.shape[-1]!=self.model.hidden_size:
                    raise ValueError('Refine Native text conditioning once before denoising')
                keys = ([self.preprocessing.project_key]*self.preprocessing.project_uses if self.rank == 0 else []) + self.keys
                self.window.extend_uses(keys)
                self.calls += 1
                return self.model(*args, **kwargs)
            elif command == 'end_sample':
                if ptd_adapter is not None and ptd_adapter.queue is not None:
                    ptd_adapter.queue.finish()
                self._drop_model()
                self.sample_active = False
            elif command == 'vae_prepare':
                return self._prepare_vae(*args, **kwargs)
            elif command == 'video_pixels':
                from patches.parallel_vae import AlignedVideoDecode
                if (self.sample_active or self.vae_model is None or self.vae_kind != 'video'
                        or getattr(self, 'vae_parallelism', 'query') != 'query'):
                    raise RuntimeError('Parallel VAE decode requires its prepared stage')
                z = packet(args[0], self.device)
                if z.ndim != 5 or z.shape[1] != 24 or z.dtype != torch.float16:
                    raise ValueError('Aligned video requires FP16 [B,24,T,H,W] raw latents')
                return AlignedVideoDecode(self.vae_model, z, self.comm, self.rank).run()
            elif command == 'video_temporal_pair':
                if (self.sample_active or self.vae_model is None or self.vae_kind != 'video'
                        or getattr(self, 'vae_parallelism', None) != 'temporal'):
                    raise RuntimeError('Temporal VAE decode requires its prepared stage')
                if len(args[0]) != 2:
                    raise ValueError('Temporal dispatch requires exactly two clip slots')
                clip = args[0][self.rank]
                if clip is None:
                    return None
                clip = packet(clip, self.device)
                if clip.ndim != 5 or clip.shape[1] != 24 or clip.dtype != torch.float16:
                    raise ValueError('Temporal video requires FP16 [B,24,T,H,W] raw latents')
                # Full Native spatial tiling on each independent temporal clip.
                # No attention patching or per-layer NCCL communication.
                return self.vae_temporal_planner._adaptive_decode(clip)
            elif command == 'vae_release':
                self._drop_vae()
            elif command == 'decode_start':
                if self.sample_active:
                    raise RuntimeError('Decode lookahead arrived during sampling')
                if self.config.cross_request_te and not self.decode_lookahead_started:
                    self.decode_lookahead_started = True
                    # Model-invariant startup/vision only: no guess about the
                    # next prompt, reference, shape or conditioning output.
                    self.cache.begin_request((),discard_unwanted=True)
                    self.startup_cache.prefetch(self.startup_specs)
                    if self.config.prefetch_lifecycle:
                        self.cache.prefetch(tp_specs(self.readers['te'],'te',self.rank,
                            self.config.layers)[:self.config.decode_te_blocks])
                    self.events.append(dict(phase='decode_start',cpu_only=True,
                        checkpoint=str(self.readers['te'].identity),unknown_next_request=True,
                        monotonic=time.monotonic()))
            elif command == 'decode_complete':
                if self.sample_active:
                    raise RuntimeError('Decode handoff arrived during sampling')
                # Caller observes decode completion before sending this command;
                # workers have already fenced and dropped GPU weights at end_sample.
                self.completed_decode = True
                self.events.append(dict(phase='decode_complete',monotonic=time.monotonic()))
                if (self.config.cross_request_te and not self.config.prefetch_lifecycle and
                    not self.startup_cache.snapshot()['pending']):
                    self.cache.prefetch(tp_specs(self.readers['te'], 'te', self.rank, self.config.layers))
            elif command == 'stats':
                return rank_snapshot(self)
            elif command == 'close':
                self._drop_model()
                self._drop_vae()
                if self.idle_window is not None:
                    self.idle_window.close();self.idle_window=None
                self.cache.close()
                self.startup_cache.close()
            else:
                raise ValueError(command)


def _worker(rank, config, budget, port, connection):
    import os
    import torch.distributed as dist
    from patches.optimized_tp import enable
    runtime = None
    try:
        # Keep the measured no-P2P TP2 topology local to these workers.
        os.environ.update(NCCL_P2P_DISABLE='1',NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4')
        os.environ.pop('NCCL_PROTO',None)
        torch.cuda.set_device(rank)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        if torch.cuda.get_device_capability(rank)==(7,5):
            from patches.h3_operator_dependencies import configure_compiler
            configure_compiler(config.operators)
        enable()
        store = dist.TCPStore('127.0.0.1', port, 3, False,
            timedelta(seconds=config.timeout_seconds))
        dist.init_process_group('nccl', store=store, rank=rank,
            world_size=2, timeout=timedelta(seconds=config.timeout_seconds))
        runtime = RankRuntime(config, rank, f'cuda:{rank}', dist, budget)
        connection.send(('ready', None))
        while True:
            if config.idle_residency:
                while not connection.poll(.05):
                    with torch.cuda.device(rank),torch.inference_mode():runtime.idle_tick()
            sequence, command, args, kwargs = connection.recv()
            result = runtime.dispatch(command, args, kwargs)
            result = worker_reply(result, rank, command)
            ptd = getattr(runtime, '_ptd_adapter', None)
            queued_reply = (ptd is not None and ptd.options.hop_handoff == 'queue'
                and command in ('ptd_configure','ptd_hop','ptd_state','denoise'))
            if not command.startswith('idle_') and not queued_reply:torch.cuda.synchronize(rank)
            connection.send((sequence, result))
            if command == 'close':
                break
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if runtime is not None:
            runtime.cache.close()
            runtime.startup_cache.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


class PhysicalTPService:
    """Two persistent NCCL processes. One in-flight command, bounded deadline."""
    operator_aot = True
    def __init__(self, config):
        from patches.legacy_runtime import validate_imported_source
        validate_imported_source()
        import torch.multiprocessing as mp
        import torch.distributed as dist
        if torch.cuda.device_count() < 2:
            raise RuntimeError('Physical TP2 requires two CUDA GPUs; use the local qualification harness on one GPU')
        if not dist.is_nccl_available():raise RuntimeError('Physical TP2 requires a PyTorch build with NCCL')
        if torch.cuda.get_device_capability(0)!=torch.cuda.get_device_capability(1):
            raise RuntimeError('Use matching GPU architectures so both ranks select the same collective paths')
        from patches.h3_operator_dependencies import validate_dependencies
        validate_dependencies(config.operators,t4=any(torch.cuda.get_device_capability(i)==(7,5) for i in range(2)))
        self.config, self.lock, self.sequence = config, threading.RLock(), 0
        self.idle_held=False
        self.staged_vaes=weakref.WeakSet()
        self.idle_vae_names=None
        self.idle_vae_cancel=threading.Event()
        self.idle_vae_thread=None
        self.idle_vae_stats=dict(loads=0,errors=0,last_error=None)
        self.closed = False
        context = mp.get_context('spawn')
        self.budget = HostBudget(config.host_bytes, context=context,
            pressure_aware=config.memory_pressure_aware)
        self.store = dist.TCPStore('127.0.0.1', 0, 3, True,
            timedelta(seconds=config.timeout_seconds), wait_for_workers=False)
        self.connections, self.processes = [], []
        try:
            for rank in range(2):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(rank,config,self.budget,self.store.port,child), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                message = self._receive(connection)
                if message[0] != 'ready':
                    raise RuntimeError(str(message))
        except BaseException:
            self.abort()
            raise
        atexit.register(self.close)
        if config.idle_residency:
            from patches.comfy_idle_lifecycle import register
            register(self)

    def request_idle(self):
        if not self.config.idle_residency or self.closed:return False
        if not self.lock.acquire(blocking=False):return False
        try:
            vaes=self._idle_vaes()
            if vaes:
                if self.idle_vae_thread is not None and self.idle_vae_thread.is_alive():return True
                self.idle_vae_cancel=threading.Event()
                self.idle_held=True
                self.idle_vae_thread=threading.Thread(target=self._preload_idle_vaes,
                    args=(vaes,self.idle_vae_cancel),name='h3-idle-vae',daemon=True)
                self.idle_vae_thread.start()
                return True
            result=self.call('idle_begin');self.idle_held=bool(result['accepted'])
            return self.idle_held
        finally:self.lock.release()

    def register_staged_vae(self,vae):
        with self.lock:self.staged_vaes.add(vae)

    def _idle_vaes(self):
        names=getattr(self,'idle_vae_names',None)
        return [v for v in getattr(self,'staged_vaes',())
                if (getattr(v,'encoder_used',False) if names is None else v.reader.path.name in names)]

    def configure_idle_vaes(self,prompt):
        from patches.h3_reference_plan import encoder_names
        self.idle_vae_names=encoder_names(prompt)
        for vae in tuple(getattr(self,'staged_vaes',())):
            if vae.reader.path.name not in self.idle_vae_names:vae.release()

    def release_staged_encoders(self):
        for vae in tuple(getattr(self,'staged_vaes',())):
            if vae.stage=='encode':vae.release()

    def idle_vae_patchers(self):
        return [v.vae.patcher for v in self._idle_vaes() if v.stage=='encode' and v.vae.patcher is not None]

    def _preload_idle_vaes(self,vaes,cancel):
        from concurrent.futures import CancelledError
        with self.lock:
            try:
                for vae in vaes:
                    if cancel.is_set() or self.closed:return
                    if vae.preload_idle(cancel):self.idle_vae_stats['loads']+=1
                if not cancel.is_set() and not self.closed:
                    result=self.call('idle_begin');self.idle_held=bool(result['accepted'])
            except CancelledError:pass
            except Exception as exc:
                self.idle_vae_stats.update(errors=self.idle_vae_stats['errors']+1,
                    last_error=(type(exc).__name__+': '+str(exc))[:256])
                # Encoder speculation must not disable the existing TE prefetch.
                if not cancel.is_set() and not self.closed:
                    result=self.call('idle_begin');self.idle_held=bool(result['accepted'])
            finally:
                if cancel.is_set() or self.closed:
                    for vae in vaes:vae.release()
                    if not self.closed:self.call('idle_stop',keep=False)
                    self.idle_held=False

    def pause_idle(self,*,keep=True,pressure=False):
        if hasattr(self,'idle_vae_cancel'):self.idle_vae_cancel.set()
        if not self.config.idle_residency or self.closed:return False
        vae_held=any(v.stage is not None for v in getattr(self,'staged_vaes',()))
        if not self.idle_held and not ((pressure or not keep) and vae_held):return False
        if not self.lock.acquire(blocking=False):return False
        try:
            self.call('idle_release' if pressure else 'idle_stop',keep=keep)
            if pressure or not keep:
                for vae in tuple(getattr(self,'staged_vaes',())):vae.release()
            self.idle_held=keep and not pressure
            return True
        finally:self.lock.release()

    def _receive(self, connection):
        import comfy.model_management
        deadline = time.monotonic()+self.config.timeout_seconds
        while not connection.poll(.1):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if time.monotonic() > deadline or any(not p.is_alive() for p in self.processes):
                raise TimeoutError('TP worker exited or command exceeded its deadline')
        result = connection.recv()
        if result[0] == 'error':
            raise RuntimeError(result[1])
        return result

    def call(self, command, *args, **kwargs):
        if command in ('te','begin_sample','vae_prepare','close') and hasattr(self,'idle_vae_cancel'):
            self.idle_vae_cancel.set()
        with self.lock:
            if self.closed:
                raise RuntimeError('TP service is closed; reload its ComfyUI loader node')
            if command=='begin_sample':self.release_staged_encoders()
            if command=='close':
                for vae in tuple(getattr(self,'staged_vaes',())):vae.release()
            self.sequence += 1
            message = (self.sequence, command, packet(args,'cpu'), packet(kwargs,'cpu'))
            try:
                for connection in self.connections:
                    connection.send(message)
                results = [self._receive(c) for c in self.connections]
                if any(r[0] != self.sequence for r in results):
                    raise RuntimeError('Worker acknowledgement order mismatch')
                if command in ('te','begin_sample','vae_prepare','close'):self.idle_held=False
                if command in ('stats','idle_stats'):
                    records=[r[1] for r in results]
                    for record in records:record['parent_vae_idle']=dict(self.idle_vae_stats)
                    return records
                if command == 'video_temporal_pair':
                    return tuple(r[1] for r in results)
                return results[0][1]
            except BaseException:
                # A rank may be inside NCCL. Terminating both prevents a stale
                # collective from consuming the next request after cancellation.
                self.abort()
                raise

    def abort(self):
        # atexit otherwise retains every replaced session through its bound
        # close method, even after both workers and their leases are gone.
        atexit.unregister(self.close)
        self.closed = True
        if hasattr(self,'idle_vae_cancel'):self.idle_vae_cancel.set()
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()
        self.store = None
        if all(not process.is_alive() for process in self.processes):
            import torch.multiprocessing as mp
            # This budget belongs only to this service's rank workers. Forced
            # exit frees their physical memory but can leave reservations and
            # even a process-shared lock whose owner died. No owner remains:
            # replace that lock and reconcile the ledger before diagnostics.
            self.budget.lock = mp.get_context('spawn').RLock()
            # Parent VAE component leases survive worker termination until
            # their models are released; do not erase those live reservations.
            self.budget.used.value = sum(v.reservation for v in getattr(self,'staged_vaes',()))
            self.budget.preparing.value = 0
            self.budget.pending_bytes.value = 0
            with self.lock:
                for vae in tuple(getattr(self,'staged_vaes',())):vae.release()

    def close(self):
        if not self.closed:
            try:
                self.call('close')
            finally:
                self.abort()
