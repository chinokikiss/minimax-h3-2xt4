"""Persistent pageable weight cache, with a host budget shared by GPU workers.

All admission is byte bounded. Speculation runs on one CPU worker per cache;
foreground demand takes priority at module boundaries. No CUDA work runs here.
"""
from collections import OrderedDict
from concurrent.futures import Future, CancelledError
from dataclasses import dataclass
import multiprocessing as mp
import math
import threading
import time

import psutil
import torch

from patches.weight_residency import module_bytes, module_tensors, pinned_module
from patches.diagnostic_metrics import detail_events
from patches.host_memory_pressure import CgroupMemory


GiB = 2**30


class HostBudget:
    """Pass the same instance to spawned rank workers; never multiply by ranks.

    The default 16 GiB managed allocation limit leaves headroom on a 32 GB
    host. The available-RAM guard also accounts for unmanaged allocations.
    Reservations cover pageable weights, construction and pinned staging.
    """
    def __init__(self, limit_bytes=16*GiB, reserve_bytes=3*GiB, context=None,
                 pressure_aware=False, file_cache_bytes=6*GiB, preparation_slots=2):
        if limit_bytes <= 0 or reserve_bytes < 0:
            raise ValueError('Invalid host memory budget')
        context = context or mp.get_context('spawn')
        self.limit_bytes, self.reserve_bytes = int(limit_bytes), int(reserve_bytes)
        self.lock = context.RLock()
        self.used = context.Value('q', 0, lock=False)
        self.peak = context.Value('q', 0, lock=False)
        self.denials = context.Value('q', 0, lock=False)
        self.pressure_aware = pressure_aware
        self.file_cache_bytes = int(file_cache_bytes)
        self.preparation_slots = int(preparation_slots)
        self.preparing = context.Value('q', 0, lock=False)
        self.pending_bytes = context.Value('q', 0, lock=False)
        self.memory = CgroupMemory()

    def pressure(self):
        return self.memory.snapshot() if self.pressure_aware else {'available':False}

    def speculation_depth(self, requested, unit_bytes):
        state = self.pressure()
        if not state['available']:
            return requested
        # Protect existing useful file cache as well as anonymous/kernel charges.
        room = state['limit']-state['nonfile_bytes']-min(state['file_bytes'],self.file_cache_bytes)-self.reserve_bytes
        if state['psi'].get('full_avg10',0)>2 or state['current']>state['limit']-GiB:
            requested = min(requested,2)
        return max(1,min(requested,int(max(0,room)//max(1,unit_bytes))))

    def acquire(self, size, speculative=False, preparation=False, staging=False):
        size = int(size)
        if size < 0:
            raise ValueError('Negative reservation')
        with self.lock:
            state = self.pressure()
            floor = self.file_cache_bytes if speculative else min(self.file_cache_bytes,2*GiB)
            projected = (state.get('nonfile_bytes',0)+self.pending_bytes.value+size+
                         min(state.get('file_bytes',0),floor)+self.reserve_bytes)
            if (self.used.value + size > self.limit_bytes or
                    psutil.virtual_memory().available < size + self.reserve_bytes or
                    (state['available'] and projected>state['limit']) or
                    (self.pressure_aware and preparation and self.preparing.value>=self.preparation_slots)):
                self.denials.value += 1
                return False
            self.used.value += size
            self.peak.value = max(self.peak.value, self.used.value)
            if self.pressure_aware and preparation:
                self.preparing.value += 1
            if self.pressure_aware and (preparation or staging):
                self.pending_bytes.value += size
            return True

    def finish_preparation(self, reservation):
        if self.pressure_aware:
            with self.lock:
                if self.preparing.value<=0 or reservation>self.pending_bytes.value:
                    raise RuntimeError('Unbalanced preparation lease')
                self.preparing.value-=1; self.pending_bytes.value-=reservation

    def finish_staging(self, reservation):
        if self.pressure_aware:
            with self.lock:
                if reservation>self.pending_bytes.value:
                    raise RuntimeError('Unbalanced staging lease')
                self.pending_bytes.value-=reservation

    def release(self, size):
        with self.lock:
            if size < 0 or size > self.used.value:
                raise RuntimeError('Unbalanced host reservation')
            self.used.value -= size

    def snapshot(self):
        with self.lock:
            return dict(limit_bytes=self.limit_bytes, reserved_bytes=self.used.value,
                        peak_reserved_bytes=self.peak.value, admission_denials=self.denials.value,
                        available_bytes=psutil.virtual_memory().available,
                        physical_bytes=psutil.virtual_memory().total,
                        pressure_aware=self.pressure_aware,cgroup=self.pressure(),
                        pending_preparation_bytes=self.pending_bytes.value,
                        preparing=self.preparing.value,file_cache_reserve_bytes=self.file_cache_bytes)


@dataclass
class _Cached:
    module: object
    size: int
    readers: int = 0
    uses: int = 0
    reload_ms: float = 0.


class HostWeightCache:
    """One persistent cache per rank, using one process-shared HostBudget.

    Cache keys must include the checkpoint and effective model configuration.
    A 4 GiB per-rank cache gives at most 8 GiB retained weights on two ranks.
    Pinning is charged separately by ResidencyWindow before it submits work.
    """
    def __init__(self, budget, capacity_bytes=4*GiB, background_headroom_bytes=2*GiB,
                 reuse_aware=False):
        if not 0 < capacity_bytes <= budget.limit_bytes:
            raise ValueError('Cache capacity must fit the shared host budget')
        self.budget, self.capacity_bytes = budget, int(capacity_bytes)
        self.background_headroom_bytes = int(background_headroom_bytes)
        self.condition = threading.Condition(threading.RLock())
        self.entries = OrderedDict()
        self.tasks = OrderedDict()
        self.wanted = set()
        self.reuse_aware = reuse_aware
        self.next_use = {}
        self.closed = False
        self.stats = dict(preparations=0, hits=0, misses=0, evictions=0,
                          discarded_preparations=0, cancelled_prefetches=0,
                          prepare_ms=0., wait_ms=0., peak_cache_bytes=0,
                          prepare_ema_ms=None,
                          preparation_counts={}, events=detail_events())
        self.stats.update(evictions_before_use=0,pressure_yields=0)
        self.stats.update(future_wait_ms=0.,future_waits=0)
        self.worker = threading.Thread(target=self._run, name='disk-weight-cache', daemon=True)
        self.worker.start()

    def _bytes(self):
        return sum(e.size for e in self.entries.values())

    def _evict_one(self, incoming=None, foreground=True):
        # Expired/model-tail entries go first, rather than evicting the next
        # use just to admit a farther speculative preparation.
        keys = sorted(self.entries,key=lambda k: k in self.wanted)
        if self.reuse_aware:
            keys = sorted(self.entries,key=lambda k:(k not in self.wanted,
                self.next_use.get(k,float('inf')),-self.entries[k].reload_ms),reverse=True)
        for key in keys:
            entry = self.entries[key]
            if (self.reuse_aware and not foreground and key in self.wanted and
                self.next_use.get(key,float('inf'))<=self.next_use.get(incoming,float('inf'))):
                continue
            if not entry.readers:
                size = entry.size
                self.stats['evictions_before_use'] += int(entry.uses==0)
                del self.entries[key]
                # Drop physical ownership before releasing the reservation.
                del entry
                self.budget.release(size)
                self.stats['evictions'] += 1
                return True
        return False

    def reserve_staging(self, size, foreground=True):
        with self.condition:
            while not self.budget.acquire(size, speculative=not foreground,staging=True):
                if not foreground or not self._evict_one():
                    return False
            return True

    def prefetch(self, specs, deadlines=None):
        """Replace idle disk lookahead; returns immediately, with no GPU load."""
        selected, planned = [], 0
        deadlines = deadlines or {}
        for spec in sorted(specs,key=lambda s: deadlines.get(s.key,float('inf'))):
            if planned + spec.max_bytes > self.capacity_bytes:
                break
            selected.append(spec)
            planned += spec.max_bytes
        specs = tuple(selected)
        with self.condition:
            if self.closed:
                raise RuntimeError('Host cache is closed')
            self.wanted = {s.key for s in specs}
            self.next_use = {s.key:deadlines.get(s.key,i) for i,s in enumerate(specs)}
            for key, task in list(self.tasks.items()):
                if not task['foreground'] and key not in self.wanted and not task['running']:
                    task['future'].cancel()
                    del self.tasks[key]
                    self.stats['cancelled_prefetches'] += 1
            for spec in specs:
                if spec.max_bytes > self.capacity_bytes:
                    break
                self._queue(spec, foreground=False, deadline=deadlines.get(spec.key,float('inf')))
            self.stats['events'].append(dict(event='idle_prefetch', keys=[str(s.key) for s in specs]))
            self.condition.notify_all()

    def begin_request(self, specs, discard_unwanted=False):
        """Reprioritize for the actual request, cancelling unrelated idle work."""
        specs = tuple(specs)
        # Retain matching ready entries; no eager loading of the entire graph.
        with self.condition:
            self.wanted = {s.key for s in specs}
            self.next_use = {s.key:i for i,s in enumerate(specs)}
            if discard_unwanted:
                # A completed model's tail must not starve another rank's
                # required startup allocation. Borrowed entries keep ownership.
                for key in list(self.entries):
                    entry = self.entries[key]
                    if key not in self.wanted and not entry.readers:
                        size = entry.size
                        self.stats['evictions_before_use'] += int(entry.uses==0)
                        del self.entries[key]
                        del entry
                        self.budget.release(size)
                        self.stats['evictions'] += 1
            for key, task in list(self.tasks.items()):
                if not task['foreground'] and key not in self.wanted and not task['running']:
                    task['future'].cancel()
                    del self.tasks[key]
                    self.stats['cancelled_prefetches'] += 1
            self.stats['events'].append(dict(event='begin_request'))
            self.condition.notify_all()

    def _queue(self, spec, foreground, deadline=float('inf')):
        if spec.key in self.entries:
            return None
        if spec.key not in self.tasks:
            self.tasks[spec.key] = dict(spec=spec, future=Future(), foreground=foreground,
                                       running=False, deadline=deadline)
        task = self.tasks[spec.key]
        task['foreground'] |= foreground
        task['deadline'] = min(task['deadline'],deadline) if self.reuse_aware else deadline
        return task['future']

    def pin(self, spec, *, cancel_event=None, speculative=False):
        """Borrow an immutable CPU entry only through the actual pin copy."""
        begin = time.perf_counter()
        counted_miss = False
        while True:
            with self.condition:
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError('Idle preparation cancelled')
                if self.closed:
                    raise RuntimeError('Host cache is closed')
                if spec.max_bytes > self.capacity_bytes:
                    raise MemoryError('Required weight group exceeds host cache capacity')
                entry = self.entries.get(spec.key)
                if entry is not None:
                    entry.readers += 1
                    entry.uses += 1
                    self.entries.move_to_end(spec.key)
                    if not counted_miss:
                        self.stats['hits'] += 1
                    break
                if not counted_miss:
                    self.stats['misses'] += 1
                    counted_miss = True
                future = self._queue(spec, foreground=not speculative)
                self.condition.notify_all()
            waiting = time.perf_counter()
            if cancel_event is not None:
                deadline=time.monotonic()+180
                while not future.done():
                    if cancel_event.wait(.02):raise CancelledError('Idle preparation cancelled')
                    if time.monotonic()>deadline:raise TimeoutError('Idle host preparation timed out')
            future.result(timeout=180)
            with self.condition:
                self.stats['future_wait_ms'] += (time.perf_counter()-waiting)*1000
                self.stats['future_waits'] += 1
        try:
            with torch.inference_mode(False), torch.no_grad():
                result = pinned_module(entry.module)
        finally:
            with self.condition:
                entry.readers -= 1
                self.stats['wait_ms'] += (time.perf_counter()-begin)*1000
                self.condition.notify_all()
        return result

    def take(self, spec):
        """Transfer CPU entry ownership AND its reservation to the caller.

        Used for startup shells: consume once without copying the embedding
        working set into another pageable cache. Caller must release bytes
        only after dropping physical tensor ownership.
        """
        begin = time.perf_counter()
        counted_miss = False
        while True:
            with self.condition:
                if self.closed:
                    raise RuntimeError('Host cache is closed')
                entry = self.entries.get(spec.key)
                if entry is not None and not entry.readers:
                    entry.uses += 1
                    del self.entries[spec.key]
                    if not counted_miss:
                        self.stats['hits'] += 1
                    self.stats['wait_ms'] += (time.perf_counter()-begin)*1000
                    return entry.module, entry.size
                if not counted_miss:
                    self.stats['misses'] += 1
                    counted_miss = True
                future = self._queue(spec,foreground=True)
                self.condition.notify_all()
            if future is not None:
                waiting = time.perf_counter()
                future.result(timeout=180)
                with self.condition:
                    self.stats['future_wait_ms'] += (time.perf_counter()-waiting)*1000
                    self.stats['future_waits'] += 1
            else:
                with self.condition:
                    self.condition.wait(timeout=.01)


    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.tasks)
                if self.closed:
                    return
                key = min(self.tasks,key=lambda k: (not self.tasks[k]['foreground'],self.tasks[k]['deadline']))
                task = self.tasks[key]
                spec = task['spec']
                # Reserve the final entry plus transient checkpoint/module
                # construction. Actual final storage replaces the upper bound.
                reservation = spec.prepare_peak_bytes or 4*spec.max_bytes
                if reservation < spec.max_bytes:
                    task['future'].set_exception(ValueError('Preparation peak is below final weight bound'))
                    del self.tasks[key]
                    continue
                if self.budget.pressure_aware and self.budget.preparing.value>=self.budget.preparation_slots:
                    self.stats['pressure_yields'] += 1
                    self.condition.wait(timeout=.1)
                    continue
                if (not task['foreground'] and psutil.virtual_memory().available <
                    reservation+self.budget.reserve_bytes+self.background_headroom_bytes):
                    self.condition.wait(timeout=.1)
                    continue
                while self._bytes() + spec.max_bytes > self.capacity_bytes:
                    if not self._evict_one(key,task['foreground']):
                        break
                fits = self._bytes() + spec.max_bytes <= self.capacity_bytes
                admitted = fits and self.budget.acquire(reservation,
                    speculative=not task['foreground'],preparation=True)
                while not admitted and self._evict_one(key,task['foreground']):
                    admitted = (self._bytes() + spec.max_bytes <= self.capacity_bytes
                                and self.budget.acquire(reservation,
                                    speculative=not task['foreground'],preparation=True))
                if not admitted:
                    # Idle preparation yields to memory pressure. Foreground
                    # fails explicitly rather than silently exceeding RAM.
                    if task['foreground'] and not self.budget.pressure_aware:
                        task['future'].set_exception(MemoryError('Shared host budget cannot admit weight preparation'))
                        del self.tasks[key]
                    else:
                        self.stats['pressure_yields'] += 1
                        self.condition.wait(timeout=.1)
                    continue
                task['running'] = True
                self.stats['events'].append(dict(event='prepare_started',key=str(key),
                    monotonic=time.monotonic(),
                    deadline=task['deadline'] if math.isfinite(task['deadline']) else None,
                    foreground=task['foreground']))
            begin = time.perf_counter()
            module = None
            preparation_lease = reservation
            try:
                with torch.inference_mode(False), torch.no_grad():
                    module = spec.prepare().eval().requires_grad_(False)
                size = module_bytes(module)
                if size > spec.max_bytes or any(t.device.type != 'cpu' for t in module_tensors(module)):
                    raise ValueError('Prepared weights exceed their bound or are not CPU resident')
                with self.condition:
                    ms = (time.perf_counter()-begin)*1000
                    self.stats['preparations'] += 1
                    self.stats['events'].append(dict(event='prepare_finished',key=str(key),
                        monotonic=time.monotonic(),prepare_ms=ms))
                    self.stats['prepare_ms'] += ms
                    ema = self.stats['prepare_ema_ms']
                    self.stats['prepare_ema_ms'] = ms if ema is None else .8*ema+.2*ms
                    name = str(key)
                    self.stats['preparation_counts'][name] = self.stats['preparation_counts'].get(name, 0)+1
                    if self.closed or (not task['foreground'] and key not in self.wanted):
                        del module
                        module = None
                        self.budget.release(reservation)
                        reservation = 0
                        self.stats['discarded_preparations'] += 1
                    else:
                        self.entries[key] = _Cached(module, size, reload_ms=ms)
                        module = None
                        self.budget.release(reservation-size)
                        reservation = 0
                        self.stats['peak_cache_bytes'] = max(self.stats['peak_cache_bytes'], self._bytes())
                    task['future'].set_result(True)
            except Exception as error:
                del module
                module = None
                if reservation:
                    self.budget.release(reservation)
                task['future'].set_exception(error)
            finally:
                self.budget.finish_preparation(preparation_lease)
                with self.condition:
                    self.tasks.pop(key, None)
                    self.condition.notify_all()

    def wait_idle(self, timeout=180):
        """Diagnostic/drain operation; result delivery must not call this."""
        with self.condition:
            if not self.condition.wait_for(lambda: not self.tasks, timeout):
                raise TimeoutError('Host prefetch did not drain within its budget')

    def trim(self):
        """Release unborrowed pageable entries under foreground memory pressure."""
        with self.condition:
            while self._evict_one():
                pass

    def relieve_pressure(self, required_bytes):
        """Free only enough local ownership for a foreground allocation.

        Does not reserve bytes: the caller must retry shared admission because
        another rank can acquire the released headroom first.
        """
        with self.condition:
            released = 0
            while released < required_bytes:
                before = self._bytes()
                if not self._evict_one():
                    break
                released += before-self._bytes()
            return released

    def snapshot(self):
        with self.condition:
            return dict(self.stats, cache_bytes=self._bytes(), capacity_bytes=self.capacity_bytes,
                        pending=len(self.tasks), shared_budget=self.budget.snapshot())

    def close(self):
        with self.condition:
            self.closed = True
            for key, task in list(self.tasks.items()):
                if not task['running']:
                    task['future'].cancel()
                    del self.tasks[key]
            self.condition.notify_all()
        self.worker.join(timeout=180)
        if self.worker.is_alive():
            raise TimeoutError('Host preparation did not stop')
        with self.condition:
            if any(e.readers for e in self.entries.values()):
                raise RuntimeError('Close GPU windows before their host cache')
            while self._evict_one():
                pass
