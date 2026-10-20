"""Opt-in TE loading candidates. No attention, quantization or GPU math changes.

These policies are experimental until a physical, Native-checked ablation
selects them. Original loader validation and checksum checks stay intact.
"""
import threading
import time
import math

import torch

from patches import host_weight_cache as host
from patches import windowed_tp as windowed
from patches.weight_residency import module_bytes, module_tensors


class CoalescedTEReader(windowed.CheckpointReader):
    def read_tensor(self, name, *, rows=None, columns=None):
        if not (self.path.name.startswith('te.block') and
                self.path.suffix=='.safetensors' and rows is None and columns is None):
            return super().read_tensor(name, rows=rows, columns=columns)
        stat=self.path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=self.signature:
            raise RuntimeError('Checkpoint changed after the request was planned')
        if not hasattr(self,'_te_payload'):
            # One allocation and sequential read for a rank-local block;
            # disjoint tensor views retain this backing until preparation ends.
            self._te_payload=bytearray(stat.st_size-self.offset)
            destination=memoryview(self._te_payload)
            with self.path.open('rb',buffering=0) as stream:
                stream.seek(self.offset)
                while destination:
                    count=stream.readinto(destination)
                    if not count:raise IOError('Offline TE payload is incomplete')
                    destination=destination[count:]
                    self.range_read_calls+=1
            self.logical_read_bytes+=len(self._te_payload)
        info=self.header[name];start,end=info['data_offsets']
        dtype=self.DTYPES[info['dtype']]
        expected=math.prod(info['shape'])*torch.empty((),dtype=dtype).element_size()
        if not 0<=start<=end<=len(self._te_payload) or end-start!=expected:
            raise ValueError('Offline TE tensor extent is invalid')
        if expected==0:return torch.empty(info['shape'],dtype=dtype)
        return torch.frombuffer(self._te_payload,dtype=dtype,count=math.prod(info['shape']),
                                offset=start).reshape(info['shape'])


class BoundedTECache(host.HostWeightCache):
    """Only TE transformer tasks may use a second pageable preparation worker.

    All in-flight final payloads count against the per-rank capacity, and
    construction reservations use the original process-shared HostBudget.
    Pinning/H2D remain owned by the original bounded residency window.
    """
    te_workers=2

    def __init__(self,*args,**kwargs):
        self._slot_local=threading.local()
        super().__init__(*args,**kwargs)
        self._extra_workers=[]
        for slot in range(1,self.te_workers):
            worker=threading.Thread(target=self._run_slot,args=(slot,),
                name=f'te-pageable-{slot}',daemon=True)
            self._extra_workers.append(worker);worker.start()

    def _run_slot(self,slot):
        self._slot_local.slot=slot
        self._run()

    def _eligible(self):
        slot=getattr(self._slot_local,'slot',0)
        return [k for k,t in self.tasks.items() if not t['running'] and
                (slot==0 or (t['spec'].key[1]=='te' and k in self.wanted))]

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.closed or self._eligible())
                if self.closed:return
                key=min(self._eligible(),key=lambda k:(not self.tasks[k]['foreground'],self.tasks[k]['deadline']))
                task=self.tasks[key];spec=task['spec']
                reservation=spec.prepare_peak_bytes or 4*spec.max_bytes
                if reservation<spec.max_bytes:
                    task['future'].set_exception(ValueError('Preparation peak is below final weight bound'))
                    del self.tasks[key];continue
                inflight=sum(t['spec'].max_bytes for t in self.tasks.values() if t['running'])
                while self._bytes()+inflight+spec.max_bytes>self.capacity_bytes:
                    if not self._evict_one():break
                if self._bytes()+inflight+spec.max_bytes>self.capacity_bytes:
                    self.condition.wait(timeout=.01);continue
                # Retain the original speculation headroom policy.
                if (not task['foreground'] and
                    host.psutil.virtual_memory().available<reservation+self.budget.reserve_bytes+self.background_headroom_bytes):
                    self.condition.wait(timeout=.1);continue
                admitted=self.budget.acquire(reservation)
                while not admitted and self._evict_one():admitted=self.budget.acquire(reservation)
                if not admitted:
                    if task['foreground'] and not inflight:
                        task['future'].set_exception(MemoryError('Shared host budget cannot admit weight preparation'))
                        del self.tasks[key]
                    else:self.condition.wait(timeout=.1)
                    continue
                task['running']=True
                self.stats['events'].append(dict(event='prepare_started',key=str(key),monotonic=time.monotonic(),
                    deadline=task['deadline'] if math.isfinite(task['deadline']) else None,foreground=task['foreground']))
            begin=time.perf_counter();module=None
            try:
                with torch.inference_mode(False),torch.no_grad():
                    module=spec.prepare().eval().requires_grad_(False)
                size=module_bytes(module)
                if size>spec.max_bytes or any(t.device.type!='cpu' for t in module_tensors(module)):
                    raise ValueError('Prepared weights exceed their bound or are not CPU resident')
                with self.condition:
                    ms=(time.perf_counter()-begin)*1000
                    self.stats['preparations']+=1
                    self.stats['prepare_ms']+=ms
                    ema=self.stats['prepare_ema_ms'];self.stats['prepare_ema_ms']=ms if ema is None else .8*ema+.2*ms
                    name=str(key);self.stats['preparation_counts'][name]=self.stats['preparation_counts'].get(name,0)+1
                    self.stats['events'].append(dict(event='prepare_finished',key=name,monotonic=time.monotonic(),prepare_ms=ms))
                    if self.closed or (not task['foreground'] and key not in self.wanted):
                        module=None;self.budget.release(reservation);reservation=0
                        self.stats['discarded_preparations']+=1
                    else:
                        self.entries[key]=host._Cached(module,size);module=None
                        self.budget.release(reservation-size);reservation=0
                        self.stats['peak_cache_bytes']=max(self.stats['peak_cache_bytes'],self._bytes())
                    task['future'].set_result(True)
            except Exception as error:
                module=None
                if reservation:self.budget.release(reservation)
                task['future'].set_exception(error)
            finally:
                with self.condition:
                    self.tasks.pop(key,None);self.condition.notify_all()

    def close(self):
        super().close()
        for worker in self._extra_workers:
            worker.join(timeout=180)
            if worker.is_alive():raise TimeoutError('TE preparation did not stop')


def install(mode):
    """Call before constructing runtime readers/caches in each spawned rank."""
    if mode not in ('baseline','coalesced','coalesced_parallel','coalesced_parallel_priority'):
        raise ValueError('Unknown TE loading ablation')
    if mode=='baseline':return
    from patches import comfy_tp_runtime as runtime
    windowed.CheckpointReader=CoalescedTEReader
    # Rank-local loader imports CheckpointReader dynamically; runtime's original
    # model readers keep their original class and never use coalesced payloads.
    if mode in ('coalesced_parallel','coalesced_parallel_priority'):
        runtime.HostWeightCache=BoundedTECache
