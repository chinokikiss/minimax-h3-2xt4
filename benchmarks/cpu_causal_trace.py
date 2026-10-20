"""Narrow CPU observers. Every original operation executes once, unchanged."""
from contextlib import contextmanager
from dataclasses import replace
import functools
import json
import os
from pathlib import Path
import threading
import time
import weakref

import torch
import causal_trace as base

LEGACY_INSTALL=base.install


class CPURecorder(base.Recorder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.objects={};self.object_serial=0

    def identity(self,obj,kind,fresh=False):
        key=(kind,id(obj));row=self.objects.get(key)
        native=kind in ('tensor','cuda_event')
        signature=int(obj._cdata) if kind=='tensor' else None
        if not fresh and row and ((native and row.get('native_signature')==signature) or
                                  (not native and row['ref']() is obj)):return row
        self.object_serial+=1
        row=dict(id=f'{self.rank}/{kind}/{self.object_serial}',kind=kind,generation=0)
        if native:row['native_signature']=signature
        else:row['ref']=weakref.ref(obj)
        self.objects[key]=row
        return row

    @contextmanager
    def span(self,name,**fields):
        stack=getattr(self.local,'cpu_stack',[])
        before=time.thread_time_ns();stack.append((name,before));self.local.cpu_stack=stack
        detail=name in ('cpu_weight_prepare','cpu_staging','cpu_pin','cpu_native_te_construct','cpu_native_dit_construct')
        try:
            with self.scope(cpu_detail=detail or self.fields().get('cpu_detail',False)):
                with super().span(name,**fields):yield
        finally:stack.pop()

    def emit(self,name,*args,**fields):
        stack=getattr(self.local,'cpu_stack',[])
        if stack and stack[-1][0]==name:
            before=stack[-1][1];after=time.thread_time_ns()
            fields.update(thread_cpu_start_ns=before,thread_cpu_end_ns=after,
                          thread_cpu_running_ns=after-before,
                          cpu_semantics='inclusive thread CPU counter; wall minus CPU is nonrunning, not necessarily blocked')
        super().emit(name,*args,**fields)

    def start(self,*args,**kwargs):
        super().start(*args,**kwargs)
        self.objects.clear();self.object_serial=0
        capability=dict(pid=os.getpid(),rank=self.rank,capture_start_ns=time.monotonic_ns(),
            torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            python_gil_enabled=getattr(__import__('sys'),'_is_gil_enabled',lambda:True)(),
            env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
        if os.name=='posix' and os.environ.get('CPU_OBSERVER_PARENT'):
            import ctypes
            libc=ctypes.CDLL(None,use_errno=True)
            capability['observer_ptrace_permission']=libc.prctl(0x59616d61,int(os.environ['CPU_OBSERVER_PARENT']),0,0,0)
            capability['observer_ptrace_errno']=ctypes.get_errno()
        (self.output/f'cpu_process_r{self.rank}.json').write_text(json.dumps(capability))

    def stop(self):
        result=super().stop()
        (self.output/f'cpu_done_r{self.rank}.json').write_text(json.dumps(dict(pid=os.getpid(),end_ns=time.monotonic_ns())))
        self.objects.clear()
        return result


def install(recorder):
    LEGACY_INSTALL(recorder)
    patch=base.patch

    def wrap_span(name,extra=None,scoped=False):
        def factory(original):
            @functools.wraps(original)
            def wrapped(*a,**k):
                if not recorder.enabled or (scoped and not recorder.fields().get('cpu_detail')):
                    return original(*a,**k)
                with recorder.span(name,**(extra(a,k) if extra else {})):return original(*a,**k)
            return wrapped
        return factory

    from concurrent.futures import Future
    from patches.host_weight_cache import HostWeightCache
    from patches import comfy_tp_runtime as runtime
    from patches import rank_local_int8 as loader

    def init_future(original):
        def wrapped(future,*a,**k):
            result=original(future,*a,**k)
            if recorder.enabled:
                row=recorder.identity(future,'future');row['condition']=recorder.identity(future._condition,'condition')['id']
                recorder.emit('future_created',future=row['id'],condition=row['condition'])
            return result
        return wrapped
    patch(Future,'__init__',init_future)

    def queue(original):
        def wrapped(cache,spec,foreground,deadline=float('inf')):
            if not recorder.enabled:return original(cache,spec,foreground,deadline)
            result=original(cache,spec,foreground,deadline)
            if result is not None:
                row=recorder.identity(result,'future')
                row.update(block=str(spec.key),cache=id(cache),condition=recorder.identity(result._condition,'condition')['id'],
                           foreground=foreground,requested_by=recorder.fields()['request'])
                recorder.emit('future_cache_link',future=row['id'],block=row['block'],cache=row['cache'],condition=row['condition'],
                    foreground=foreground,predicted_use_ns=int(deadline*1e9) if deadline!=float('inf') else None)
            return result
        return wrapped
    patch(HostWeightCache,'_queue',queue)

    def future_wait(original):
        def wrapped(future,*a,**k):
            if not recorder.enabled:return original(future,*a,**k)
            row=recorder.identity(future,'future')
            legacy=recorder.futures.get(id(future),{})
            with recorder.span('future_consumer_wait',future=row['id'],block=row.get('block'),
                               cache=row.get('cache'),legacy_block=legacy.get('block'),ready_on_entry=future.done()):
                return original(future,*a,**k)
        return wrapped
    patch(Future,'result',future_wait)
    for method in ('set_result','set_exception','cancel'):
        def publish(original):
            def wrapped(future,*a,**k):
                if not recorder.enabled:return original(future,*a,**k)
                row=recorder.identity(future,'future')
                begin=time.monotonic_ns()
                with recorder.span('future_producer_publish',future=row['id'],block=row.get('block'),
                    operation=original.__name__,readiness_semantics='transition is bounded by publish entry/exit; not an exact nanosecond'):
                    result=original(future,*a,**k)
                if future.done():recorder.emit('future_ready_bounds',future=row['id'],lower_ns=begin,
                    upper_ns=time.monotonic_ns(),block=row.get('block'),operation=original.__name__)
                return result
            return wrapped
        patch(Future,method,publish)

    def condition_wait(original):
        def wrapped(condition,*a,**k):
            if not recorder.enabled:return original(condition,*a,**k)
            row=recorder.identity(condition,'condition')
            with recorder.span('condition_wait',condition=row['id']):return original(condition,*a,**k)
        return wrapped
    patch(threading.Condition,'wait',condition_wait)
    for method in ('notify','notify_all'):
        def notify(original):
            def wrapped(condition,*a,**k):
                if recorder.enabled:
                    row=recorder.identity(condition,'condition')
                    recorder.emit('condition_notify',condition=row['id'],operation=original.__name__)
                return original(condition,*a,**k)
            return wrapped
        patch(threading.Condition,method,notify)

    for method in ('wait','set','clear'):
        def python_event(original):
            def wrapped(event,*a,**k):
                if not recorder.enabled:return original(event,*a,**k)
                row=recorder.identity(event,'python_event')
                with recorder.span('python_event/'+original.__name__,event=row['id'],
                                   ready_on_entry=event.is_set()):return original(event,*a,**k)
            return wrapped
        patch(threading.Event,method,python_event)

    for method in ('record','wait','synchronize','query'):
        def cuda_event(original):
            def wrapped(event,*a,**k):
                if not recorder.enabled:return original(event,*a,**k)
                row=recorder.identity(event,'cuda_event')
                if original.__name__=='record':row['generation']+=1
                fields=dict(event=row['id'],generation=row['generation'])
                if original.__name__=='query':
                    start=time.monotonic_ns();ready=original(event,*a,**k);end=time.monotonic_ns()
                    if row.get('last_query')!=ready:
                        recorder.emit('cuda_event_query',start,end,**fields,ready=ready,
                            readiness_semantics='false query bounds readiness below by query entry; true query bounds above by return')
                        row['last_query']=ready
                    return ready
                if original.__name__=='record':
                    row.pop('last_query',None)
                    stream=a[0] if a and a[0] is not None else k.get('stream')
                    fields['stream']=int((stream or torch.cuda.current_stream(recorder.device)).cuda_stream)
                with recorder.span('cuda_event/'+original.__name__,**fields):return original(event,*a,**k)
            return wrapped
        patch(torch.cuda.Event,method,cuda_event)

    def stream_wait(original):
        def wrapped(stream,event,*a,**k):
            if not recorder.enabled:return original(stream,event,*a,**k)
            row=recorder.identity(event,'cuda_event')
            with recorder.span('cuda_event_consumer_wait',event=row['id'],generation=row['generation'],
                               stream=int(stream.cuda_stream)):
                return original(stream,event,*a,**k)
        return wrapped
    patch(torch.cuda.Stream,'wait_event',stream_wait)
    def event_new(original):
        def wrapped(cls,*a,**k):
            event=original(cls,*a,**k)
            if recorder.enabled:recorder.identity(event,'cuda_event',fresh=True)
            return event
        return wrapped
    patch(torch.cuda.Event,'__new__',event_new)

    # These are existing public helpers, not reimplementations of preparation.
    for owner,name,label in ((runtime,'native_te','cpu_native_te_construct'),
                              (runtime,'native_dit','cpu_native_dit_construct'),
                              (loader,'shell','cpu_module_shell'),(loader,'plan','cpu_shard_plan'),
                              (loader,'state_hashes','cpu_tensor_hashes'),
                              (loader,'load_rank_block','cpu_rank_deserialize'),
                              (loader,'load_offline_block','cpu_offline_deserialize')):
        patch(owner,name,wrap_span(label))
    patch(torch.nn.Module,'load_state_dict',wrap_span('cpu_state_assignment',scoped=True))
    patch(HostWeightCache,'_evict_one',wrap_span('cpu_cache_eviction'))
    for name in ('empty','empty_like','empty_strided','frombuffer'):
        patch(torch,name,wrap_span('cpu_allocation/'+name,scoped=True))

    # Aggregate syscall spans per opened checkpoint, not one record per row.
    class ObservedFile:
        def __init__(self,file,path):
            self.file=file;self.path=path;self.stats=dict(read_calls=0,read_bytes=0,read_wall_ns=0,
                read_thread_cpu_ns=0,seek_calls=0,seek_wall_ns=0);self.start=time.monotonic_ns()
        def __getattr__(self,name):return getattr(self.file,name)
        def __enter__(self):self.file.__enter__();return self
        def __exit__(self,*a):
            try:return self.file.__exit__(*a)
            finally:recorder.emit('checkpoint_file_io',self.start,time.monotonic_ns(),source=str(self.path),
                **self.stats,semantics='aggregate calls; read wall minus calling-thread CPU is nonrunning, not proven disk wait')
        def read(self,*a,**k):return self._read('read',*a,**k)
        def readinto(self,*a,**k):return self._read('readinto',*a,**k)
        def _read(self,name,*a,**k):
            start=time.monotonic_ns();cpu=time.thread_time_ns()
            try:
                result=getattr(self.file,name)(*a,**k)
                self.stats['read_calls']+=1;self.stats['read_bytes']+=len(result) if name=='read' else (result or 0)
                return result
            finally:
                self.stats['read_wall_ns']+=time.monotonic_ns()-start
                self.stats['read_thread_cpu_ns']+=time.thread_time_ns()-cpu
        def seek(self,*a,**k):
            start=time.monotonic_ns()
            try:return self.file.seek(*a,**k)
            finally:self.stats['seek_calls']+=1;self.stats['seek_wall_ns']+=time.monotonic_ns()-start
    def open_file(original):
        def wrapped(path,*a,**k):
            file=original(path,*a,**k)
            mode=a[0] if a else k.get('mode','r')
            if recorder.enabled and recorder.fields().get('cpu_detail') and mode=='rb':
                return ObservedFile(file,path)
            return file
        return wrapped
    patch(Path,'open',open_file)

    def tensor_to(original):
        def wrapped(tensor,*a,**k):
            if not recorder.enabled:return original(tensor,*a,**k)
            destination=k.get('device',a[0] if a else None)
            if isinstance(destination,torch.Tensor):destination=destination.device
            if isinstance(destination,torch.dtype) and recorder.fields().get('cpu_detail'):
                target=tensor.device
            elif isinstance(destination,(str,torch.device,int)):
                target=torch.device('cuda',destination) if isinstance(destination,int) else torch.device(destination)
            else:return original(tensor,*a,**k)
            if tensor.device.type==target.type and not recorder.fields().get('cpu_detail'):
                return original(tensor,*a,**k)
            before=recorder.identity(tensor,'tensor')['id']
            with recorder.span('tensor_transfer_submit',tensor=before,source=str(tensor.device),
                destination=str(target),bytes=tensor.numel()*tensor.element_size(),shape=list(tensor.shape),
                non_blocking=k.get('non_blocking',a[2] if len(a)>2 else False)):
                result=original(tensor,*a,**k)
            recorder.emit('tensor_transfer_result',source_tensor=before,
                          result_tensor=recorder.identity(result,'tensor')['id'],destination=str(result.device))
            return result
        return wrapped
    patch(torch.Tensor,'to',tensor_to)
