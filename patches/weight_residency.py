"""Event-owned weight streaming across one request's ordered module uses.

No attention or quantization math is changed. Each GPU/rank owns its window;
the caller supplies the exact order, including repeated denoiser invocations.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, CancelledError
from patches.diagnostic_metrics import detail_events
from contextlib import contextmanager
import copy
from dataclasses import dataclass, fields, is_dataclass, replace
import math
import time
import threading
from typing import Callable, Hashable

import torch
from torch import nn


def _is_quantized(value):
    return isinstance(value, torch.Tensor) and hasattr(value, '_qdata') and hasattr(value, '_params')


def tensor_leaves(value):
    """Include physical INT8 storage and every tensor-valued layout field."""
    if _is_quantized(value):
        yield from tensor_leaves(value._qdata)
        yield from tensor_leaves(value._params)
    elif isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from tensor_leaves(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)


def module_tensors(module):
    seen = set()
    for child in module.modules():
        for value in child.__dict__.values():
            # Child modules are visited above; hooks and callables are opaque.
            for tensor in tensor_leaves(value):
                if id(tensor) not in seen:
                    seen.add(id(tensor))
                    yield tensor


def module_bytes(module):
    return sum(t.numel() * t.element_size() for t in module_tensors(module))


def map_tensors(value, transform, memo=None):
    """Copy physical payloads without dequantizing or changing layout metadata.

    kitchen 0.2.37 QuantizedTensor.to ignores non_blocking; transferring the
    payload and layout fields explicitly also avoids metadata clone kernels.
    """
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        if id(value) in memo:
            return memo[id(value)]
        if _is_quantized(value):
            result = value._copy_with(
                qdata=map_tensors(value._qdata, transform, memo),
                params=map_tensors(value._params, transform, memo), clone_params=False)
            if isinstance(value, nn.Parameter):
                result._is_param = True
        else:
            result = transform(value.detach())
            if isinstance(value, nn.Parameter):
                result = nn.Parameter(result, requires_grad=False)
        memo[id(value)] = result
        return result
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{f.name: map_tensors(getattr(value, f.name), transform, memo)
                                 for f in fields(value)})
    if isinstance(value, dict):
        # Module hook registries are weak-referenceable OrderedDict instances.
        # Preserve their type (and defaultdict factories), not just contents.
        result = copy.copy(value)
        result.clear()
        result.update((k, map_tensors(v, transform, memo)) for k, v in value.items())
        return result
    if isinstance(value, list):
        return [map_tensors(v, transform, memo) for v in value]
    if isinstance(value, tuple):
        items = [map_tensors(v, transform, memo) for v in value]
        return type(value)(*items) if hasattr(value, '_fields') else tuple(items)
    return value


def copy_module(module, transform):
    """Clone module shells, preserving shared modules/tensors, then move tensors.

    Use on checkpoint-loaded inference modules before their first forward.
    Mutable forward caches on an already-used module are outside this API.
    """
    tensors, modules = {}, {}
    def visit(source):
        if id(source) in modules:
            return modules[id(source)]
        target = copy.copy(source)
        modules[id(source)] = target
        for name, value in source.__dict__.items():
            if name == '_modules':
                target._modules = {k: visit(v) if v is not None else None for k, v in value.items()}
            else:
                target.__dict__[name] = map_tensors(value, transform, tensors)
        return target
    try:
        return visit(module)
    finally:
        # The recursive closure refers to itself. Its memo dictionaries would
        # otherwise own all copied tensors until cyclic GC, including weights
        # already evicted from a bounded GPU window and retired pin buffers.
        tensors.clear()
        modules.clear()


def pinned_module(module):
    def pin(tensor):
        if tensor.device.type != 'cpu':
            raise ValueError('Weight preparation must return a CPU module')
        return tensor.contiguous().pin_memory()
    return copy_module(module, pin)


@dataclass(frozen=True)
class WeightSpec:
    key: Hashable
    prepare: Callable[[], nn.Module]
    # Upper bound used to admit preparation without exceeding pinned budget.
    max_bytes: int
    # Total CPU preparation peak, including final pageable weights. None uses
    # the conservative generic 4x bound; specialized loaders can supply one.
    prepare_peak_bytes: int | None = None
    # Immutable CPU backing already owned and budgeted by the caller. Lease
    # it directly: a second multi-GB pinned copy defeats host headroom. CUDA
    # may stage pageable transfers synchronously; ready still fences use.
    resident_backing: bool = False
    # Transfer a budgeted pageable cache entry directly, instead of making
    # another large pinned copy. Ownership lasts through the copy-ready event.
    pin_backing: bool = True


@dataclass
class _Entry:
    module: nn.Module
    host: nn.Module | None
    size: int
    ready: torch.cuda.Event
    copy_start: torch.cuda.Event
    last_use: torch.cuda.Event | None = None
    active: bool = False
    copy_reported: bool = False
    host_reservation: int = 0
    pinned: bool = True


class ResidencyWindow:
    """Fixed bounded lookahead with reuse-aware eviction and event lifetimes.

    max_weight_bytes bounds live/pending physical weight tensors. reserve_bytes
    is an additional free-device-memory guard for activations/other consumers.
    pinned_bytes bounds admitted pinned preparation, not total process RAM.
    Preparation uses one CPU worker; native construction has transient RAM.
    """
    def __init__(self, specs, uses, device, *, max_weight_bytes, pinned_bytes,
                 lookahead=2, reserve_bytes=512 * 2**20, copy_stream=None,
                 adaptive=False, max_lookahead=2, host_cache=None, retain_keys=(),
                 cpu_lookahead=16, max_cpu_lookahead=32):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or self.device.index is None:
            raise ValueError('An explicit CUDA device is required')
        specs = tuple(specs)
        self.specs = {s.key: s for s in specs}
        if len(self.specs) != len(specs):
            raise ValueError('Duplicate weight specification keys')
        self.host_cache = host_cache
        self.host_reservations = {}
        self.uses = tuple(uses)
        if any(k not in self.specs for k in self.uses):
            raise ValueError('Every ordered use must have a weight specification')
        self.retain_keys = frozenset(retain_keys)
        if not self.retain_keys.issubset(self.specs):
            raise ValueError('Retained weights must have specifications')
        if lookahead < 1 or max_weight_bytes <= 0 or pinned_bytes <= 0 or reserve_bytes < 0:
            raise ValueError('Invalid residency budget or lookahead')
        self.max_weight_bytes = int(max_weight_bytes)
        self.pinned_bytes = int(pinned_bytes)
        if any(s.max_bytes <= 0 or (not s.resident_backing and s.pin_backing
                                   and s.max_bytes > pinned_bytes) for s in specs):
            raise ValueError('Every preparation must fit the pinned budget')
        self.reserve_bytes = int(reserve_bytes)
        self.lookahead = int(lookahead)
        self.adaptive = bool(adaptive)
        self.max_lookahead = max(int(max_lookahead), self.lookahead)
        if cpu_lookahead < 1 or max_cpu_lookahead < cpu_lookahead:
            raise ValueError('Invalid CPU preparation horizon')
        self.cpu_lookahead = int(cpu_lookahead)
        self.max_cpu_lookahead = int(max_cpu_lookahead)
        self.observations = []
        self.compute_ms = self.copy_ms = self.prepare_ms = None
        self.samples = 0
        self.copy_stream = copy_stream or torch.cuda.Stream(device=self.device)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='weight-prepare')
        self.pending: dict[Hashable, Future] = {}
        self.unmaterialized_staging = {}
        self.entries: dict[Hashable, _Entry] = {}
        self.retired: list[_Entry] = []
        self.cursor = 0
        self.remaining = Counter(self.uses)
        self.closed = False
        self.idle_mode=False
        self.idle_cancel=None
        self.idle_pending=set()
        self.idle_plan=()
        self.active_budgets=None
        self.stats = {'loads': {}, 'hits': 0, 'h2d_bytes': 0, 'evictions': 0,
                      'peak_live_weight_bytes': 0, 'peak_pinned_admitted_bytes': 0,
                      'host_wait_ms': 0., 'copies': detail_events(), 'transitions': detail_events(),
                      'lookahead_changes': detail_events(), 'cpu_lookahead_changes': detail_events(),
                      'cpu_plans': detail_events(), 'compute': detail_events()}
        self.stats.update(idle_loads=0,idle_h2d_bytes=0,idle_cancellations=0,
                          idle_errors=0,idle_last_error=None,idle_gpu_hits=0,idle_avoided_h2d_bytes=0)
        self.idle_loaded=set()
        self.future_cpu = False
        self.cpu_fences = []
        self.cpu_priority_by_use = {}

    def configure_cpu_schedule(self, *, refiner_by_use=None):
        """PTD-only complete sequence; no CUDA allocations or math changes."""
        if self.cursor or self.host_cache is None:
            raise RuntimeError('Configure the CPU schedule before consumption')
        self.future_cpu = True
        self.cpu_priority_by_use = dict(refiner_by_use or {})
        self.host_cache.set_access_plan(self.uses, self.cursor)
        if self.cpu_priority_by_use:
            self.host_cache.enable_priority_lane(set(self.cpu_priority_by_use.values()))

    @contextmanager
    def _cpu_guard(self, key):
        if not getattr(self, 'future_cpu', False):
            yield
            return
        self.host_cache.hold(key)
        try:
            yield
        finally:
            # Protect pageable ownership conservatively through GPU consumption,
            # even when DMA already uses an independently owned pinned copy.
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.cpu_fences.append((key, event))

    def _drop_pending(self,key):
        self.pending.pop(key,None)
        self.idle_pending.discard(key)
        staging=self.unmaterialized_staging.pop(key,0)
        if staging:self.host_cache.budget.finish_staging(staging)
        reservation=self.host_reservations.pop(key,0)
        if reservation:self.host_cache.budget.release(reservation)

    def begin_idle(self,keys,*,max_bytes,reserve_bytes):
        """No CUDA allocation here; the owner calls idle_tick between RPCs."""
        # An idle plan is prospective, not an unfinished model invocation.
        # Cancellation leaves it unconsumed; reset it before re-admission.
        if self.active_budgets is not None:
            self.resume()
        if self.closed or self.cursor!=len(self.uses) or any(e.active for e in self.entries.values()):
            raise RuntimeError('Idle admission requires a completed invocation')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Idle residency cannot mutate CUDA Graph pointers')
        if self.idle_mode:self.cancel_idle(keep=True)
        if self.active_budgets is None:
            self.active_budgets=(self.max_weight_bytes,self.reserve_bytes,self.retain_keys)
        self.idle_cancel=threading.Event();self.idle_mode=True
        self.idle_plan=tuple(dict.fromkeys(keys))
        self.uses=self.idle_plan;self.cursor=0;self.remaining=Counter(self.uses)
        self.max_weight_bytes=min(int(max_bytes),self.active_budgets[0])
        self.reserve_bytes=int(reserve_bytes)
        self.retain_keys=frozenset(self.idle_plan)
        while self._live_bytes()>self.max_weight_bytes:
            candidates=[k for k,e in self.entries.items() if not e.active]
            if not candidates:break
            self._evict(max(candidates,key=self._next_use));self._reap()

    def idle_tick(self):
        """Admit at most one copy, one CPU staging task; never await a Future."""
        if not self.idle_mode or self.closed:return False
        self._reap()
        if self.idle_cancel.is_set():return False
        for key in self.idle_plan:
            if key in self.entries:continue
            spec=self.specs[key]
            if key not in self.pending:
                if self.pending:return False
                if self._live_bytes()+spec.max_bytes>self.max_weight_bytes:return False
                if not self._room(spec.max_bytes,self._next_use(key),False):return False
                staging=self._staging_bytes(spec)
                if self._host_bytes()+staging>self.pinned_bytes:return False
                self.host_cache.prefetch([spec])
                if staging and not self.host_cache.reserve_staging(staging,foreground=False):return False
                self.host_reservations[key]=staging
                if staging:self.unmaterialized_staging[key]=staging
                self.idle_pending.add(key)
                self.pending[key]=self.pool.submit(ResidencyWindow._prepare,self,spec,
                    cancel_event=self.idle_cancel,speculative=True)
            future=self.pending[key]
            if not future.done():return False
            try:host,size=future.result()
            except Exception as error:
                self.stats['idle_errors']+=1;self.stats['idle_last_error']=(type(error).__name__+': '+str(error))[:256]
                self._drop_pending(key);self.cancel_idle(keep=False);return False
            if host is None:
                self._drop_pending(key);return False
            if not self._room(size,self._next_use(key),False):return False
            try:self._stage(key,host,size)
            except torch.cuda.OutOfMemoryError:
                self.stats['idle_errors']+=1;self._drop_pending(key);self.cancel_idle(keep=False);return False
            self.idle_pending.discard(key);self.idle_loaded.add(key)
            self.stats['idle_loads']+=1;self.stats['idle_h2d_bytes']+=size
            return True
        return False

    def cancel_idle(self,*,keep):
        if self.idle_cancel is not None:self.idle_cancel.set()
        self.idle_mode=False
        for key in tuple(self.idle_pending):
            future=self.pending.get(key)
            if future is not None and future.cancel():
                self.stats['idle_cancellations']+=1;self._drop_pending(key)
            elif future is not None and future.done() and not keep:
                del future
                self._drop_pending(key)
        if not keep:
            for key in list(self.entries):
                if not self.entries[key].active:self._evict(key)
            self.idle_loaded.clear()
        self._reap()

    def resume(self):
        self.cancel_idle(keep=True)
        if self.active_budgets is not None:
            self.max_weight_bytes,self.reserve_bytes,self.retain_keys=self.active_budgets
            self.active_budgets=None
        self.uses=();self.cursor=0;self.remaining.clear();self.idle_plan=()

    def room_for_external(self,required_bytes):
        """Yield cached weights for the Native vision working set, farthest first."""
        self._reap()
        while True:
            free,_=torch.cuda.mem_get_info(self.device)
            allocator=torch.cuda.memory_stats(self.device)
            free+=max(0,allocator.get('reserved_bytes.all.current',0)-allocator.get('active_bytes.all.current',0))
            if free>=required_bytes:return True
            candidates=[k for k,e in self.entries.items() if not e.active]
            if not candidates:return False
            self._evict(max(candidates,key=self._next_use));self._reap()

    def _host_bytes(self):
        # PyTorch's pinned allocator rounds each tensor to a power of two.
        # Twice payload is a conservative allocation bound, not just qdata.
        return (sum(self._staging_bytes(self.specs[k]) for k in self.pending)
                + sum(e.host_reservation for e in self.entries.values() if e.host is not None and e.pinned)
                + sum(e.host_reservation for e in self.retired if e.host is not None and e.pinned)
                if self.host_cache is not None else
                sum(self._staging_bytes(self.specs[k]) for k in self.pending)
                + sum(2*e.size for e in self.entries.values() if e.host is not None and e.pinned)
                + sum(2*e.size for e in self.retired if e.host is not None and e.pinned))

    @staticmethod
    def _staging_bytes(spec):
        return 0 if spec.resident_backing or not spec.pin_backing else 2*spec.max_bytes

    def append_plan(self, specs, keys):
        """Append immutable future groups without draining the existing queue.

        Owner-thread only. Active/retired entries retain their CUDA fences;
        admission and the existing CPU/GPU lookahead still limit materialization.
        This registers metadata, never an entire expert's resident allocation.
        """
        specs, keys = tuple(specs), tuple(keys)
        if self.closed or self.idle_mode:
            raise RuntimeError('Append requires an open foreground window')
        additions = {s.key:s for s in specs}
        if len(additions) != len(specs):
            raise ValueError('Duplicate appended specification keys')
        for key, spec in additions.items():
            if key in self.specs and self.specs[key] is not spec:
                raise ValueError('An admitted weight identity cannot be rebound')
            if spec.max_bytes <= 0 or (not spec.resident_backing and spec.pin_backing
                                      and spec.max_bytes > self.pinned_bytes):
                raise ValueError('Appended preparation exceeds its budget')
        if any(key not in self.specs and key not in additions for key in keys):
            raise ValueError('Unknown appended weight use')
        self.specs.update(additions)
        self.uses += keys
        self.remaining.update(keys)

    def extend_uses(self, keys):
        """Append one actual invocation, without predicting the sampler's NFE.

        Called only by the owning rank between forwards. Retained entries may
        survive an unknown next call, but remain subject to byte/free guards.
        """
        keys = tuple(keys)
        if self.closed or self.cursor != len(self.uses):
            raise RuntimeError('Append only after the preceding invocation completes')
        if any(k not in self.specs for k in keys):
            raise ValueError('Unknown appended weight use')
        self.uses += keys
        self.remaining.update(keys)

    def _live_bytes(self):
        return sum(e.size for e in self.entries.values()) + sum(e.size for e in self.retired)

    def _prepare(self, spec, *, cancel_event=None, speculative=False):
        begin = time.perf_counter()
        try:
            if not spec.resident_backing and not spec.pin_backing and self.host_cache is not None:
                module, lease = self.host_cache.take(spec)
                self.host_reservations[spec.key] = lease
            else:
                module = (spec.prepare() if spec.resident_backing or not spec.pin_backing else
                      self.host_cache.pin(spec,cancel_event=cancel_event,speculative=speculative)
                          if self.host_cache is not None and cancel_event is not None else
                      self.host_cache.pin(spec) if self.host_cache is not None else
                      pinned_module(spec.prepare().eval().requires_grad_(False)))
        except CancelledError:
            return None,0
        finally:
            staging = self.unmaterialized_staging.pop(spec.key,0)
            if staging:
                self.host_cache.budget.finish_staging(staging)
        module._residency_prepare_ms = (time.perf_counter() - begin) * 1000
        size = module_bytes(module)
        if size > spec.max_bytes:
            raise ValueError(f'Weight bound underestimated for {spec.key}: {size} > {spec.max_bytes}')
        return module, size

    def _reap(self):
        fences = []
        for key, event in getattr(self, 'cpu_fences', ()):
            if event.query(): self.host_cache.release_hold(key)
            else: fences.append((key, event))
        self.cpu_fences = fences
        for key in tuple(self.idle_pending):
            future=self.pending.get(key)
            if future is None:self.idle_pending.discard(key)
            elif future.done():
                failed=future.cancelled() or future.exception() is not None
                if failed:
                    self.stats['idle_errors']+=int(not future.cancelled())
                    if not future.cancelled():
                        error=future.exception()
                        self.stats['idle_last_error']=(type(error).__name__+': '+str(error))[:256]
                    if self.idle_cancel is not None:self.idle_cancel.set()
                if failed or future.result()[0] is None:self._drop_pending(key)
        for entry in list(self.entries.values()) + self.retired:
            if entry.host is not None and entry.ready.query():
                entry.host = None
                if entry.host_reservation:
                    self.host_cache.budget.release(entry.host_reservation)
                    entry.host_reservation = 0
            if not entry.copy_reported and entry.ready.query():
                ms = entry.copy_start.elapsed_time(entry.ready)
                self.stats['copies'].append({'bytes': entry.size,
                    'cuda_ms': ms})
                self.copy_ms = ms if self.copy_ms is None else .8*self.copy_ms + .2*ms
                entry.copy_reported = True
        pending = []
        for key, start, end in self.observations:
            if not end.query():
                pending.append((key, start, end))
                continue
            ms = start.elapsed_time(end)
            self.stats['compute'].append({'key': str(key), 'cuda_ms': ms})
            self.compute_ms = ms if self.compute_ms is None else .8*self.compute_ms + .2*ms
            self.samples += 1
            if self.adaptive and self.samples % 3 == 0 and self.compute_ms > 0:
                cpu_prepare = (self.host_cache.stats['prepare_ema_ms']
                               if self.host_cache is not None else self.prepare_ms)
                cpu_target = max(2, min(self.max_cpu_lookahead,
                    1 + math.ceil(((cpu_prepare or 0)+(self.copy_ms or 0))/self.compute_ms)))
                next_cpu = self.cpu_lookahead + (cpu_target > self.cpu_lookahead) - (cpu_target < self.cpu_lookahead)
                if next_cpu != self.cpu_lookahead:
                    self.stats['cpu_lookahead_changes'].append(dict(position=self.cursor,
                        before=self.cpu_lookahead, after=next_cpu))
                    self.cpu_lookahead = next_cpu
        self.observations = pending
        self.retired = [e for e in self.retired
                        if not (e.last_use or e.ready).query()]

    def _next_use(self, key):
        return next((i for i in range(self.cursor, len(self.uses)) if self.uses[i] == key), float('inf'))

    def _evict(self, key):
        entry = self.entries[key]
        if entry.active:
            raise RuntimeError('Cannot evict active weights')
        self.retired.append(self.entries.pop(key))
        self.idle_loaded.discard(key)
        self.stats['evictions'] += 1

    def _room(self, size, deadline, blocking):
        self._reap()
        reclaimed = False
        while True:
            driver_free, _ = torch.cuda.mem_get_info(self.device)
            # The process can reuse its own unused caching-allocator storage.
            # Driver-only free memory falsely rejects a request after a large
            # native attention pass has populated the allocator. Allocated is
            # a subset of reserved; neither is counted twice here.
            allocator = torch.cuda.memory_stats(self.device)
            # active_bytes also includes blocks awaiting side-stream completion.
            cached = max(0, allocator.get('reserved_bytes.all.current', 0)
                         - allocator.get('active_bytes.all.current', 0))
            free = driver_free + cached
            if self._live_bytes() + size <= self.max_weight_bytes and free >= size + self.reserve_bytes:
                return True
            candidates = [(self._next_use(k), k) for k, e in self.entries.items() if not e.active]
            if candidates:
                next_use, key = max(candidates, key=lambda item: item[0])
                # Speculation must not displace a weight needed sooner.
                if next_use > deadline:
                    self._evict(key)
                    self._reap()
                    continue
            if blocking and self.retired:
                entry = self.retired[0]
                (entry.last_use or entry.ready).synchronize()
                self._reap()
                continue
            if blocking and not reclaimed:
                # record_stream can leave logically freed weights in the CUDA
                # allocator's active pending-free list even after our last-use
                # events completed. memory_stats alone does not reap that list.
                # Reclaim only on foreground admission pressure, preserving
                # every consumer event and rechecking the real memory guard.
                begin = time.perf_counter()
                torch.cuda.empty_cache()
                self.stats['foreground_reclaims'] = self.stats.get('foreground_reclaims', 0) + 1
                self.stats['reclaim_ms'] = self.stats.get('reclaim_ms', 0.) + (time.perf_counter()-begin)*1000
                reclaimed = True
                self._reap()
                continue
            return False

    def _adapt_gpu_lookahead(self):
        """Choose one or two upcoming groups without allocating or evicting.

        Count reusable allocator memory just as foreground admission does.
        Admission still rechecks free memory immediately before each H2D copy.
        """
        if not self.adaptive:
            return
        keys = []
        for key in self.uses[self.cursor:]:
            if key not in keys:
                keys.append(key)
            if len(keys) == min(2, self.max_lookahead):
                break
        missing = sum(self.specs[k].max_bytes for k in keys if k not in self.entries)
        driver_free, _ = torch.cuda.mem_get_info(self.device)
        allocator = torch.cuda.memory_stats(self.device)
        free = driver_free + max(0, allocator.get('reserved_bytes.all.current', 0)
                                - allocator.get('active_bytes.all.current', 0))
        depth = (2 if len(keys)==2 and missing+self._live_bytes()<=self.max_weight_bytes
                 and free>=missing+self.reserve_bytes else 1)
        if depth != self.lookahead:
            self.stats['lookahead_changes'].append(dict(position=self.cursor,
                before=self.lookahead,after=depth,reason='vram_headroom',
                free_bytes=free,missing_weight_bytes=missing))
            self.lookahead = depth

    def prefetch(self):
        if self.closed:
            raise RuntimeError('Window is closed')
        with torch.cuda.device(self.device):
            self._reap()
            self._adapt_gpu_lookahead()
            # Pageable CPU preparation has its own farther horizon. It does
            # not reserve pin buffers or allocate CUDA tensors. Deadlines are
            # predicted next-use times, not a fixed-depth H2D permission.
            if self.host_cache is not None:
                if self.future_cpu:
                    self.host_cache.set_access_plan(self.uses, self.cursor)
                cpu_keys = []
                now = time.monotonic()
                for position in range(self.cursor, len(self.uses)):
                    key = self.uses[position]
                    if (not self.specs[key].resident_backing and key not in self.entries
                            and key not in [k for _,k in cpu_keys]):
                        cpu_keys.append((position,key))
                    limit = min(self.cpu_lookahead, 2) if self.cpu_priority_by_use else self.cpu_lookahead
                    if len(cpu_keys) >= limit:
                        break
                deadlines = {k: now+(p-self.cursor)*(self.compute_ms or 100.)/1000
                             for p,k in cpu_keys}
                if self.host_cache.budget.pressure_aware and cpu_keys:
                    unit = max(self.specs[k].prepare_peak_bytes or 4*self.specs[k].max_bytes for _,k in cpu_keys)
                    depth = self.host_cache.budget.speculation_depth(len(cpu_keys),unit)
                    cpu_keys = cpu_keys[:depth]
                if self.cpu_priority_by_use:
                    # The next refiner gets a separate CPU lane while ordinary
                    # preparation keeps a small imminent prefix. Both lanes use
                    # the same capacity, process-shared bytes and slot guards.
                    next_refiner = next(((p,k) for p,k in sorted(self.cpu_priority_by_use.items())
                        if p > self.cursor and k not in self.entries and k not in [x for _,x in cpu_keys]), None)
                    if next_refiner is not None:
                        p,k = next_refiner
                        deadlines[k] = now+(p-self.cursor)*(self.compute_ms or 100.)/1000
                        cpu_keys.append((p,k))
                self.host_cache.prefetch([self.specs[k] for _,k in cpu_keys],deadlines=deadlines)
                self.stats['cpu_plans'].append(dict(position=self.cursor,
                    cpu_depth=len(cpu_keys),gpu_depth=self.lookahead))
            upcoming = []
            for position in range(self.cursor, len(self.uses)):
                key = self.uses[position]
                if key not in [k for _, k in upcoming]:
                    upcoming.append((position, key))
                if len(upcoming) == self.lookahead:
                    break
            for position, key in upcoming:
                if key in self.entries:
                    continue
                if key not in self.pending:
                    spec = self.specs[key]
                    staging = self._staging_bytes(spec)
                    if self._host_bytes() + staging > self.pinned_bytes:
                        break
                    if self.host_cache is not None:
                        if staging and not self.host_cache.reserve_staging(staging,
                                foreground=position==self.cursor):
                            break
                        self.host_reservations[key] = staging
                        if staging:
                            self.unmaterialized_staging[key] = staging
                    self.pending[key] = self.pool.submit(self._prepare, spec)
                    self.stats['peak_pinned_admitted_bytes'] = max(
                        self.stats['peak_pinned_admitted_bytes'], self._host_bytes())
                future = self.pending[key]
                if future.done():
                    host, size = future.result()
                    if host is None:
                        self._drop_pending(key);continue
                    if self._room(size, position, False):
                        self._stage(key, host, size)
            return upcoming

    def _stage(self, key, host, size):
        ms = host._residency_prepare_ms
        self.prepare_ms = ms if self.prepare_ms is None else .8*self.prepare_ms + .2*ms
        with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
            start = torch.cuda.Event(enable_timing=True)
            ready = torch.cuda.Event(enable_timing=True)
            start.record()
            module = copy_module(host, lambda t: t.to(self.device, non_blocking=True))
            for child in module.modules():
                if hasattr(child, 'device') and isinstance(child.device, (str, torch.device)):
                    child.device = self.device
            ready.record()
        self.entries[key] = _Entry(module, host, size, ready, start,
                                  host_reservation=self.host_reservations.pop(key, 0),
                                  pinned=not self.specs[key].resident_backing and self.specs[key].pin_backing)
        del self.pending[key]
        self.idle_pending.discard(key)
        self.stats['loads'][str(key)] = self.stats['loads'].get(str(key), 0) + 1
        self.stats['h2d_bytes'] += size
        self.stats['peak_live_weight_bytes'] = max(self.stats['peak_live_weight_bytes'], self._live_bytes())
        self.stats['transitions'].append({'position': self.cursor, 'loaded': str(key),
                                         'bytes': size, 'deadline': self._next_use(key)})

    @contextmanager
    def use(self, key):
        if self.closed or self.cursor >= len(self.uses) or self.uses[self.cursor] != key:
            raise RuntimeError(f'Unexpected weight use at {self.cursor}: {key}')
        with torch.cuda.device(self.device), self._cpu_guard(key):
            self.prefetch()
            if key not in self.entries:
                if key not in self.pending:
                    # Earlier speculative copy sources may still own pinned space.
                    for entry in list(self.entries.values()) + self.retired:
                        if entry.host is not None:
                            entry.ready.synchronize()
                    self.prefetch()
                if key not in self.pending:
                    raise MemoryError('Pinned staging budget cannot admit the next required weights')
                begin = time.perf_counter()
                host, size = self.pending[key].result()
                if host is None:
                    self._drop_pending(key);self.prefetch()
                    if key not in self.pending:raise MemoryError('Cancelled idle staging cannot be readmitted')
                    host,size=self.pending[key].result()
                self.stats['host_wait_ms'] += (time.perf_counter() - begin) * 1000
                if not self._room(size, self.cursor, True):
                    free, total = torch.cuda.mem_get_info(self.device)
                    raise MemoryError(f'Required weights cannot fit: position={self.cursor}, '
                        f'required={size}, live={self._live_bytes()}, cap={self.max_weight_bytes}, '
                        f'driver_free={free}, total={total}, reserve={self.reserve_bytes}')
                self._stage(key, host, size)
            else:
                self.stats['hits'] += 1
                if key in self.idle_loaded:
                    self.stats['idle_gpu_hits']+=1
                    self.stats['idle_avoided_h2d_bytes']+=self.entries[key].size
                    self.idle_loaded.discard(key)
            entry = self.entries[key]
            if entry.active:
                raise RuntimeError('A weight lease is already active')
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(entry.ready)
            started = torch.cuda.Event(enable_timing=True)
            started.record(stream)
            entry.active = True
            # Allocation occurred on the copy stream; every physical payload
            # also belongs to this consumer until queued native kernels finish.
            for tensor in module_tensors(entry.module):
                tensor.record_stream(stream)
            self.prefetch()
            try:
                yield entry.module
            finally:
                entry.last_use = torch.cuda.Event(enable_timing=True)
                entry.last_use.record(stream)
                self.observations.append((key, started, entry.last_use))
                entry.active = False
                self.remaining[key] -= 1
                self.cursor += 1
                if not self.remaining[key] and key not in self.retain_keys:
                    self._evict(key)
                self.prefetch()

    def finish(self):
        if self.cursor != len(self.uses):
            raise RuntimeError(f'Request ended with {len(self.uses)-self.cursor} unconsumed weight uses')
        return self.close()

    def close(self):
        if self.closed:
            return self.stats
        if self.idle_cancel is not None:self.cancel_idle(keep=True)
        if self.future_cpu:
            # CPU preparation can be waiting for cache space held by a finished
            # GPU consumer. Release those fences BEFORE joining the CPU pool.
            with torch.cuda.device(self.device):
                for _, event in self.cpu_fences: event.synchronize()
                self._reap()
        self.pool.shutdown(wait=True, cancel_futures=True)
        with torch.cuda.device(self.device):
            self.copy_stream.synchronize()
            for entry in list(self.entries.values()) + self.retired:
                if entry.last_use is not None:
                    entry.last_use.synchronize()
            for _, event in self.cpu_fences: event.synchronize()
            self._reap()
        self.entries.clear()
        self.retired.clear()
        self.pending.clear()
        for size in self.unmaterialized_staging.values():
            self.host_cache.budget.finish_staging(size)
        self.unmaterialized_staging.clear()
        for size in self.host_reservations.values():
            self.host_cache.budget.release(size)
        self.host_reservations.clear()
        self.observations.clear()
        self.closed = True
        return self.stats


class ActivationHandoff:
    """A real copy with explicit producer/copy/consumer dependencies.

    Same-device simulations force separate storage. Source tensors remain
    owned until copy completion; destination payloads are recorded on consumer.
    """
    def __init__(self, device, copy_stream=None, max_pending=2):
        self.device = torch.device(device)
        self.stream = copy_stream or torch.cuda.Stream(device=self.device)
        self.pending = []
        self.sources = {}
        self.max_pending = max(1, int(max_pending))

    def copy(self, value, producer_stream, consumer_stream):
        self.pending = [(owner, event) for owner, event in self.pending if not event.query()]
        if len(self.pending) >= self.max_pending:
            owner, event = self.pending.pop(0)
            event.synchronize()
            del owner
        produced = torch.cuda.Event()
        produced.record(producer_stream)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.stream.wait_event(produced)
            used_sources = set()
            def transfer(tensor):
                if tensor.is_cuda and tensor.device != producer_stream.device:
                    raise ValueError('All CUDA handoff payloads must belong to the producer stream device')
                if tensor.is_cuda and tensor.device != self.device:
                    if tensor.device not in self.sources:
                        self.sources[tensor.device] = torch.cuda.Stream(device=tensor.device)
                    source = self.sources[tensor.device]
                    source.wait_event(produced)
                    used_sources.add(tensor.device)
                    # Cross-GPU PyTorch copies can enqueue on the source's
                    # current stream; control that stream explicitly too.
                    with torch.cuda.stream(source):
                        return tensor.to(self.device, non_blocking=True, copy=True)
                return tensor.to(self.device, non_blocking=True, copy=True)
            result = map_tensors(value, transfer)
            for device in used_sources:
                source_done = torch.cuda.Event()
                source_done.record(self.sources[device])
                self.stream.wait_event(source_done)
            copied = torch.cuda.Event()
            copied.record()
        self.pending.append((value, copied))
        consumer_stream.wait_event(copied)
        for tensor in tensor_leaves(result):
            tensor.record_stream(consumer_stream)
        return result

    def close(self):
        self.stream.synchronize()
        for stream in self.sources.values():
            stream.synchronize()
        self.pending.clear()
