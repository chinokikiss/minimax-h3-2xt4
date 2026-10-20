"""Diagnostic observers only. No scheduling decisions or inference math live here.

CPU spans use the common host monotonic clock. CUDA intervals use an event
anchor fenced BEFORE the measured batch, with its host alignment uncertainty
reported. Kineto remains the authority for kernel/memcpy/API correlations.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from concurrent.futures import Future, ThreadPoolExecutor
import functools
import json
import math
import os
from pathlib import Path
import threading
import time
import torch

ACTIVE = None
ORIGINALS = []


def safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    return str(value)


class Recorder:
    def __init__(self, output, rank, device):
        self.output, self.rank, self.device = Path(output), rank, device
        self.output.mkdir(parents=True, exist_ok=True)
        self.events, self.gpu, self.loads, self.futures = [], [], {}, {}
        self.context = dict(request='setup', stage='setup', command=0, nfe=0)
        self.local = threading.local()
        self.enabled = False
        self.serial = self.collective = 0
        self.anchor = None
        self.profiler = None

    def fields(self):
        return {**self.context, **getattr(self.local, 'fields', {})}

    @contextmanager
    def scope(self, **fields):
        before = getattr(self.local, 'fields', {})
        self.local.fields = {**before, **fields}
        try:
            yield
        finally:
            self.local.fields = before

    def emit(self, name, start=None, end=None, **fields):
        if not self.enabled:
            return
        now = time.monotonic_ns()
        self.events.append(dict({"name":name, "start_ns":start or now,
            "end_ns":end or now, "pid":os.getpid(), "tid":threading.get_native_id(),
            "rank":self.rank, **self.fields(), **safe(fields)}))

    @contextmanager
    def span(self, name, **fields):
        if not self.enabled:
            yield
            return
        start = time.monotonic_ns()
        snapshot = {**self.fields(), **fields}
        label = ('causal/r%s/req=%s/stage=%s/cmd=%s/nfe=%s/%s' %
            (self.rank, snapshot['request'], snapshot['stage'],
             snapshot['command'], snapshot['nfe'], name))
        if snapshot.get('block') is not None:
            label += '/block=' + str(snapshot['block'])
        torch.cuda.nvtx.range_push(label)
        try:
            with torch.profiler.record_function(label):
                yield
        except BaseException as error:
            fields['error'] = type(error).__name__
            raise
        finally:
            torch.cuda.nvtx.range_pop()
            self.emit(name, start, time.monotonic_ns(), **{**snapshot, **fields})

    def event(self, stream=None):
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream or torch.cuda.current_stream(self.device))
        return event

    def interval(self, name, begin, end, **fields):
        self.gpu.append((name, begin, end, {**self.fields(), **safe(fields)}))

    def start(self, tag, kineto=True):
        self.events.clear(); self.gpu.clear(); self.loads.clear(); self.futures.clear()
        self.tag = tag
        # Calibration and profiler startup are outside request timing.
        self.anchor = self.event()
        self.anchor.synchronize()
        before = time.monotonic_ns()
        alignment = self.event()
        alignment.synchronize()
        after = time.monotonic_ns()
        self.anchor_ns = (before + after)//2 - int(self.anchor.elapsed_time(alignment)*1e6)
        self.uncertainty_ns = (after-before)//2
        if kineto:
            activities=getattr(self,'kineto_activities',None) or [
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            self.profiler = torch.profiler.profile(activities=activities,
                record_shapes=False, profile_memory=False, with_stack=False)
            self.profiler.__enter__()
        self.enabled = True
        self.emit('capture_start', clock='host_monotonic_ns',
                  alignment_uncertainty_ns=self.uncertainty_ns)

    def stop(self):
        # All fences, event resolution and trace export occur AFTER the batch.
        torch.cuda.synchronize(self.device)
        self.emit('capture_end')
        self.enabled = False
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            self.profiler.export_chrome_trace(str(self.output/f'{self.tag}_r{self.rank}_kineto.json'))
            self.profiler = None
        intervals = []
        for name, begin, end, fields in self.gpu:
            intervals.append(dict(name=name, rank=self.rank,
                start_ns=self.anchor_ns + int(self.anchor.elapsed_time(begin)*1e6),
                end_ns=self.anchor_ns + int(self.anchor.elapsed_time(end)*1e6),
                **fields))
        path = self.output/f'{self.tag}_r{self.rank}_causal.json'
        path.write_text(json.dumps(dict(schema=1, rank=self.rank, pid=os.getpid(),
            clock='host_monotonic_ns', alignment_uncertainty_ns=self.uncertainty_ns,
            cuda_interval_semantics='stream elapsed intervals; not kernel busy time',
            events=self.events, gpu_intervals=intervals), indent=2), encoding='utf-8')
        self.gpu.clear(); self.loads.clear(); self.futures.clear()
        return str(path)


def patch(owner, name, factory):
    original = getattr(owner, name)
    ORIGINALS.append((owner, name, original))
    setattr(owner, name, factory(original))


def restore():
    global ACTIVE
    for owner, name, original in reversed(ORIGINALS):
        setattr(owner, name, original)
    ORIGINALS.clear()
    ACTIVE = None


def install(recorder):
    """Install process-local observers; originals run exactly once, unchanged."""
    global ACTIVE
    ACTIVE = recorder
    from patches import weight_residency as residency
    from patches.host_weight_cache import HostWeightCache, HostBudget
    from patches.windowed_tp import CheckpointReader
    import torch.distributed as dist

    def span_method(name, extra=None):
        def factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                fields = extra(args, kwargs) if extra else {}
                with recorder.span(name, **fields):
                    return original(*args, **kwargs)
            return wrapped
        return factory

    # Future waits are measured on the caller thread, distinct from pin copies.
    def future_factory(original):
        @functools.wraps(original)
        def wrapped(future, *args, **kwargs):
            fields = recorder.futures.get(id(future))
            if fields is None or not recorder.enabled:
                return original(future, *args, **kwargs)
            with recorder.span('cpu_future_wait', **fields, ready_on_entry=future.done()):
                return original(future, *args, **kwargs)
        return wrapped
    patch(Future, 'result', future_factory)

    def submit_factory(original):
        def wrapped(pool, fn, *args, **kwargs):
            if not recorder.enabled:
                return original(pool, fn, *args, **kwargs)
            origin = recorder.fields()
            if args and hasattr(args[0], 'key'):
                origin['block'] = str(args[0].key)
            def task():
                with recorder.scope(**origin):
                    return fn(*args, **kwargs)
            future = original(pool, task)
            recorder.futures[id(future)] = dict(block=origin.get('block'),
                dependency='CPU staging task', requested_by=origin['request'])
            recorder.emit('cpu_staging_queue', **origin)
            return future
        return wrapped
    patch(ThreadPoolExecutor, 'submit', submit_factory)

    def queue_factory(original):
        def wrapped(cache, spec, foreground, deadline=float('inf')):
            existing = cache.tasks.get(spec.key)
            if existing is None and spec.key not in cache.entries:
                origin = recorder.fields()
                prepare = spec.prepare
                @functools.wraps(prepare)
                def traced_prepare():
                    with recorder.scope(**{**origin, 'block':str(spec.key)}):
                        with recorder.span('cpu_weight_prepare', source=str(spec.key[0]),
                            destination='pageable_RAM', requested_by=origin['request'],
                            executing_during=recorder.context['request']):
                            return prepare()
                spec = replace(spec, prepare=traced_prepare)
            result = original(cache, spec, foreground, deadline)
            fields = dict(block=str(spec.key), cache=id(cache), foreground=foreground,
                          predicted_use_ns=int(deadline*1e9) if deadline != float('inf') else None)
            if result is not None:
                recorder.futures[id(result)] = fields
            recorder.emit('cpu_queue', **fields, cached=result is None,
                          already_queued=existing is not None)
            return result
        return wrapped
    patch(HostWeightCache, '_queue', queue_factory)

    for name in ('pin', 'take'):
        def cache_factory(original):
            def wrapped(cache, spec, *args, **kwargs):
                before = {k:cache.stats[k] for k in ('hits','misses')}
                with recorder.scope(block=str(spec.key)):
                    with recorder.span('host_cache_'+original.__name__, cache=id(cache)):
                        result = original(cache, spec, *args, **kwargs)
                    recorder.emit('host_cache_outcome', cache=id(cache),
                        hit=cache.stats['hits']>before['hits'],
                        miss=cache.stats['misses']>before['misses'])
                return result
            return wrapped
        patch(HostWeightCache, name, cache_factory)
    patch(residency, 'pinned_module', span_method('cpu_pin', lambda a,k:dict(destination='pinned_RAM')))
    # host_weight_cache imports pinned_module by value.
    import patches.host_weight_cache as host_module
    patch(host_module, 'pinned_module', span_method('cpu_pin', lambda a,k:dict(destination='pinned_RAM')))

    def evict_factory(original):
        def wrapped(cache, *args, **kwargs):
            before = {str(k):e.size for k,e in cache.entries.items()}
            out = original(cache, *args, **kwargs)
            after = {str(k) for k in cache.entries}
            for key in before.keys()-after:
                recorder.emit('host_evict', block=key, bytes=before[key], cache=id(cache))
            return out
        return wrapped
    for name in ('_evict_one', 'begin_request', 'trim'):
        patch(HostWeightCache, name, evict_factory)

    def budget_factory(original):
        def wrapped(budget, size, *args, **kwargs):
            result = original(budget, size, *args, **kwargs)
            recorder.emit('host_budget_'+original.__name__, bytes=size,
                admitted=result if original.__name__=='acquire' else None,
                used=budget.used.value, limit=budget.limit_bytes)
            return result
        return wrapped
    for name in ('acquire', 'release'):
        patch(HostBudget, name, budget_factory)

    def prepare_factory(original):
        def wrapped(window, spec, *args, **kwargs):
            with recorder.scope(block=str(spec.key)):
                with recorder.span('cpu_staging', window=id(window)):
                    return original(window, spec, *args, **kwargs)
        return wrapped
    patch(residency.ResidencyWindow, '_prepare', prepare_factory)

    def stage_factory(original):
        def wrapped(window, key, host, size):
            recorder.serial += 1
            load = f'r{recorder.rank}-load-{recorder.serial}'
            with recorder.scope(block=str(key), load=load):
                with recorder.span('h2d_enqueue', bytes=size, source='host_RAM',
                    destination=str(window.device), stream=int(window.copy_stream.cuda_stream)):
                    result = original(window,key,host,size)
                entry = window.entries[key]
                recorder.loads[id(entry.ready)] = dict(load=load, block=str(key))
                if recorder.enabled:
                    recorder.interval('h2d',entry.copy_start,entry.ready,load=load,
                        bytes=size, stream=int(window.copy_stream.cuda_stream))
            return result
        return wrapped
    patch(residency.ResidencyWindow, '_stage', stage_factory)

    def use_factory(original):
        @contextmanager
        def wrapped(window, key):
            with recorder.scope(block=str(key), window=id(window)):
                recorder.emit('gpu_cache_lookup', hit=key in window.entries, cursor=window.cursor)
                manager = original(window,key)
                with recorder.span('block_acquire'):
                    module = manager.__enter__()
                if recorder.enabled:
                    recorder.emit('weight_compute_ready',allocated=torch.cuda.memory_allocated(window.device),
                        reserved=torch.cuda.memory_reserved(window.device),
                        driver_free=torch.cuda.mem_get_info(window.device)[0],
                        live_weights=window._live_bytes(),gpu_depth=window.lookahead,
                        cpu_depth=window.cpu_lookahead)
                begin = recorder.event() if recorder.enabled else None
                try:
                    with recorder.span('block_compute_enqueue'):
                        yield module
                finally:
                    if recorder.enabled:
                        end = recorder.event()
                        recorder.interval('block_compute_stream',begin,end)
                    with recorder.span('block_release'):
                        manager.__exit__(*__import__('sys').exc_info())
        return wrapped
    patch(residency.ResidencyWindow, 'use', use_factory)

    def gpu_evict_factory(original):
        def wrapped(window,key):
            if not recorder.enabled:
                return original(window,key)
            entry=window.entries[key]
            recorder.emit('gpu_evict',block=str(key),bytes=entry.size,
                next_use=window._next_use(key),last_use_complete=(entry.last_use or entry.ready).query())
            return original(window,key)
        return wrapped
    patch(residency.ResidencyWindow,'_evict',gpu_evict_factory)
    patch(residency.ResidencyWindow,'_room',span_method('gpu_memory_admission',
        lambda a,k:dict(bytes=a[1],deadline=a[2],blocking=a[3] if len(a)>3 else k['blocking'])))

    def wait_factory(original):
        def wrapped(stream,event):
            fields=recorder.loads.get(id(event),dict(dependency='unmapped_event'))
            if not recorder.enabled:
                return original(stream,event)
            begin=recorder.event(stream)
            with recorder.span('stream_wait_event',**fields,
                               stream=int(stream.cuda_stream),ready_on_enqueue=event.query()):
                result=original(stream,event)
            end=recorder.event(stream)
            recorder.interval('gpu_dependency_wait_upper_bound',begin,end,**fields,
                              stream=int(stream.cuda_stream))
            return result
        return wrapped
    patch(torch.cuda.Stream,'wait_event',wait_factory)
    for owner,name in ((torch.cuda,'synchronize'),(torch.cuda,'empty_cache'),
                       (torch.cuda.Stream,'synchronize'),(torch.cuda.Event,'synchronize')):
        patch(owner,name,span_method('sync/'+owner.__name__+'.'+name))

    def read_factory(original):
        def wrapped(reader,*args,**kwargs):
            if not recorder.enabled:
                return original(reader,*args,**kwargs)
            before=reader.logical_read_bytes
            with recorder.span('checkpoint_read/'+original.__name__,source=str(reader.path),
                selection=safe(args),slice=safe(kwargs)):
                result=original(reader,*args,**kwargs)
            recorder.emit('checkpoint_read_bytes',bytes=reader.logical_read_bytes-before,
                          source=str(reader.path),logical_not_physical_disk_traffic=True,
                          nested_read=False)
            return result
        return wrapped
    for name in ('read','read_tensor'):
        patch(CheckpointReader,name,read_factory)

    for name in ('all_reduce','all_gather','all_gather_into_tensor','reduce_scatter_tensor',
                 'broadcast','send','recv','isend','irecv','barrier'):
        def collective_factory(original):
            def wrapped(*args,**kwargs):
                recorder.collective+=1
                sequence=recorder.collective
                from patches.weight_residency import tensor_leaves
                tensors=list(tensor_leaves((args,kwargs)))
                bytes_=sum(t.numel()*t.element_size() for t in tensors)
                with recorder.span('nccl/'+original.__name__,collective=sequence,
                    argument_bytes=bytes_,traffic_semantics='tensor arguments; not wire bytes'):
                    return original(*args,**kwargs)
            return wrapped
        patch(dist,name,collective_factory)

    from patches import comfy_tp_runtime as runtime
    patch(runtime,'packet',span_method('activation_packet',
        lambda a,k:dict(destination=str(a[1]))))
