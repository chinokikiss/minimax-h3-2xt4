"""Whole-block OS snapshots; no tensor inspection or inference policy changes."""
from collections import Counter
import functools
import threading
import time
import cpu_causal_trace as cpu
import te_io_telemetry as telemetry


def install(recorder):
    cpu.install(recorder)
    from patches import windowed_tp
    observer=telemetry.ReadObserver(period=.1)
    recorder.block_observer=observer
    from concurrent.futures import ThreadPoolExecutor
    from patches.host_weight_cache import HostWeightCache
    seen=set()
    def queue(original):
        def wrapped(cache,spec,foreground,deadline=float('inf')):
            if not recorder.enabled:return original(cache,spec,foreground,deadline)
            before=time.monotonic_ns();future=original(cache,spec,foreground,deadline);after=time.monotonic_ns()
            key=(id(cache),spec.key,id(future))
            if future is not None and key not in seen:
                seen.add(key)
                recorder.emit('cache_queue_bounds',before,after,block=str(spec.key),
                    future=recorder.identity(future,'future')['id'],foreground=foreground,
                    lower_ns=before,upper_ns=after)
            return future
        return wrapped
    cpu.base.patch(HostWeightCache,'_queue',queue)
    def submit(original):
        def wrapped(pool,fn,*args,**kwargs):
            if not recorder.enabled:return original(pool,fn,*args,**kwargs)
            before=time.monotonic_ns();future=original(pool,fn,*args,**kwargs);after=time.monotonic_ns()
            if args and hasattr(args[0],'key'):
                recorder.emit('executor_submit_bounds',before,after,block=str(args[0].key),
                    future=recorder.identity(future,'future')['id'],lower_ns=before,upper_ns=after)
            return future
        return wrapped
    cpu.base.patch(ThreadPoolExecutor,'submit',submit)
    def prepare(original):
        @functools.wraps(original)
        def wrapped(reader,kind,index,rank):
            if not recorder.enabled or kind!='te':return original(reader,kind,index,rank)
            tid=threading.get_native_id();observed=time.monotonic_ns()
            before=telemetry.snapshot()
            samples=dict(states=Counter(),wchans=Counter(),stacks=[],stack_unavailable_samples=0)
            with observer.lock:observer.active[tid]=samples
            start=time.monotonic_ns();error=None
            try:return original(reader,kind,index,rank)
            except BaseException as failure:error=type(failure).__name__;raise
            finally:
                end=time.monotonic_ns()
                with observer.lock:observer.active.pop(tid,None)
                after=telemetry.snapshot()
                recorder.emit('te_block_os_counters',start,end,te_index=index,
                    producer_tid=tid,error=error,counters=telemetry.delta(before,after),
                    memory_current_before=before['memory_current'],memory_current_after=after['memory_current'],
                    thread_cpu_ns=after['thread_cpu_ns']-before['thread_cpu_ns'],
                    sampled_waits={k:dict(v) if isinstance(v,Counter) else v for k,v in samples.items()},
                    observer_outside_prepare_ns=time.monotonic_ns()-observed-(end-start),
                    semantics='Whole-block producer counters; shared cgroup/device deltas are overlapping observations and must not be summed.')
        return wrapped
    cpu.base.patch(windowed_tp,'prepare_tp_block',prepare)


def close(recorder):
    observer=getattr(recorder,'block_observer',None)
    if observer is not None:observer.close();recorder.block_observer=None
