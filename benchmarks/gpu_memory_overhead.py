"""Opt-in allocator lifetime and residency audit, with no tensor ownership changes.

Allocator free_requested/free_completed are authoritative. Shape observations
do not retain tensors or infer CUDA completion from Python object lifetimes.
"""
from collections import Counter
from contextlib import contextmanager,nullcontext
import gzip
import inspect
import json
import os
from pathlib import Path
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from patches.weight_residency import module_tensors


def storage(t):
    if not isinstance(t,torch.Tensor) or not t.is_cuda:return None
    if hasattr(t,'_qdata'):t=t._qdata
    try:
        s=t.untyped_storage()
        return dict(addr=s.data_ptr(),bytes=s.nbytes(),shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype))
    except (RuntimeError,NotImplementedError):return None


def tensors(value):
    if isinstance(value,torch.Tensor):
        if hasattr(value,'_qdata'):
            yield value._qdata
        else:yield value
    elif isinstance(value,dict):
        for v in value.values():yield from tensors(v)
    elif isinstance(value,(tuple,list)):
        for v in value:yield from tensors(v)


class Shapes(TorchDispatchMode):
    def __init__(self,audit):super().__init__();self.audit=audit
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        a=self.audit
        if len(a.shapes)<12000:
            seen=set()
            for t in tensors(result):
                item=storage(t)
                if item is not None and item['addr'] not in seen:
                    seen.add(item['addr'])
                    a.shapes.append(dict(time_ns=time.time_ns(),operator=str(func),scope=list(a.scopes),
                        block=a.block,**item))
        else:a.shape_dropped+=1
        return result


class MemoryAudit:
    def __init__(self,runtime,role=None,install_wrappers=True):
        self.runtime=runtime;self.rank=runtime.rank;self.device=runtime.device
        self.role=role or f'rank{self.rank}'
        self.directory=Path(os.environ['MINIMAX_H3_MEMORY_DIR'])/self.role
        self.directory.mkdir(parents=True,exist_ok=True)
        self.command='startup';self.nfe=0;self.block=None;self.scopes=[]
        self.markers=[];self.shapes=[];self.shape_dropped=0;self.static_tables=[]
        self.chunks=[];self.index=0;self.active=False;self.begin_segments=[]
        self.handles=[]
        if install_wrappers:self.install()

    def _segments(self,snapshot,intern=None):
        rows=[]
        for segment in snapshot.get('segments',[]):
            if segment.get('device',self.rank)!=self.rank:continue
            s={k:v for k,v in segment.items() if k not in ('blocks','frames')}
            s['blocks']=[]
            for block in segment['blocks']:
                b={k:v for k,v in block.items() if k!='frames'}
                if intern is not None:b['stack']=intern(block.get('frames',[]))
                s['blocks'].append(b)
            rows.append(s)
        return rows

    def begin(self,command):
        self.command=command
        self.block=None;self.scopes=[]
        self.nfe=self.runtime.context.get('nfe',0)
        self.markers=[];self.shapes=[];self.shape_dropped=0;self.static_tables=[]
        self.begin_segments=self._segments(torch.cuda.memory._snapshot())
        kwargs=dict(enabled='all',context='all',stacks='python',max_entries=50000,
                    device=self.device,clear_history=True,global_record_annotations=True)
        signature=inspect.signature(torch.cuda.memory._record_memory_history).parameters
        torch.cuda.memory._record_memory_history(**{k:v for k,v in kwargs.items() if k in signature})
        self.active=True
        self.mark('command_begin',static=True)

    def static_inventory(self):
        result={}
        def add(t,category,key):
            item=storage(t)
            if item is None:return
            if item['addr'] not in result:result[item['addr']]=dict(category=category,key=key,**item)
        window=self.runtime.window
        if window is not None:
            for key,entry in list(window.entries.items()):
                for t in module_tensors(entry.module):add(t,'resident_weights',str(key[1:]))
            for entry in list(window.retired):
                for t in module_tensors(entry.module):add(t,'retired_weights','pending CUDA lifetime')
        for name,model in [('model_auxiliary',self.runtime.model),('vae_weights',self.runtime.vae_model),
                           *getattr(self.runtime,'extra_models',[])]:
            if model is not None:
                for t in module_tensors(model):add(t,name,name)
        return list(result.values())

    def mark(self,label,static=False):
        if not self.active:return
        stats=torch.cuda.memory_stats(self.device)
        row=dict(time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),label=label,
            command=self.command,nfe=self.runtime.context.get('nfe',0),block=self.block,
            scope=list(self.scopes),allocated=torch.cuda.memory_allocated(self.device),
            reserved=torch.cuda.memory_reserved(self.device),
            active=stats.get('active_bytes.all.current'),
            inactive_split=stats.get('inactive_split_bytes.all.current'),
            allocation_retries=stats.get('num_alloc_retries'),ooms=stats.get('num_ooms'))
        if static:
            table=self.static_inventory();row['static_table']=len(self.static_tables)
            self.static_tables.append(table)
            sums=Counter()
            for item in table:sums[item['category']]+=item['bytes']
            row['known_static_storage_bytes']=dict(sums)
            row['other_allocated_bytes']=row['allocated']-sum(sums.values())
            free,total=torch.cuda.mem_get_info(self.device)
            row.update(driver_free=free,driver_total=total)
        if len(self.markers)<12000:self.markers.append(row)

    @contextmanager
    def scope(self,label):
        if not self.active:
            yield;return
        self.scopes.append(label);self.mark(label+'/begin')
        torch.cuda.nvtx.range_push('memory/'+label)
        try:
            with torch.profiler.record_function('memory/'+label):yield
        finally:
            self.mark(label+'/end');torch.cuda.nvtx.range_pop();self.scopes.pop()

    def install(self):
        import patches.tp_dit as dit
        import patches.h3_operator_policy as policy
        import patches.tilelang_swiglu_pad as swiglu
        import patches.tilelang_row_quant as quant
        import comfy_kitchen as ck
        audit=self
        original=dit.TPDiTBlockINT8.forward
        def block(module,*args,**kw):
            if audit.active:
                audit.mark('block_begin',static=True)
                audit.markers[-1]['inputs']=[i for t in tensors((args,kw)) if (i:=storage(t)) is not None]
            # Representative first/middle/last blocks get tensor shape/operator
            # observations. Allocator alloc/free history covers ALL blocks.
            selected=audit.active and audit.block in (0,24,49)
            try:
                with Shapes(audit) if selected else nullcontext():
                    return original(module,*args,**kw)
            finally:audit.mark('block_end',static=True)
        dit.TPDiTBlockINT8.forward=block
        for cls in (dit.ColumnParallelLinearINT8,dit.RowParallelLinearINT8):
            fn=cls.forward
            def projection(module,*args,_fn=fn,**kw):
                if hasattr(module,'components'):label='QKV' if module.components==3 else 'FC1'
                else:label='FC2' if module.in_features_per_rank==7168 else 'AttentionO'
                with audit.scope(label):return _fn(module,*args,**kw)
            cls.forward=projection
        for module,name,label in ((policy,'dit_attention','Attention'),(ck,'rms_rope_split_half','RoPE'),
                (swiglu,'global_row_quantize','FC2_global_quantizer'),
                (swiglu,'_finish_row','FC2_GEMM_INT32_SUM_dequantize'),
                (quant,'global_row_quantize','row_global_quantizer')):
            fn=getattr(module,name)
            def wrap(*args,_fn=fn,_label=label,**kw):
                with audit.scope(_label):return _fn(*args,**kw)
            setattr(module,name,wrap)

    def flush(self):
        if not self.active:return
        self.mark('command_acknowledgement',static=True)
        snapshot=torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        self.active=False
        stacks=[];stack_ids={}
        def intern(frames):
            # Python source frames retain the actual caller/operator. Avoid C++
            # stacks and duplicate path strings in every allocator event.
            key=tuple((f.get('filename'),f.get('line'),f.get('name')) for f in frames)
            if key not in stack_ids:stack_ids[key]=len(stacks);stacks.append(key)
            return stack_ids[key]
        traces=snapshot.get('device_traces',[])
        events=traces[self.rank] if self.rank<len(traces) else []
        compact=[]
        for event in events:
            record={k:v for k,v in event.items() if k not in ('frames','compile_context')}
            record['stack']=intern(event.get('frames',[]));compact.append(record)
        segments=self._segments(snapshot,intern)
        actions=Counter(e['action'] for e in compact)
        payload=dict(rank=self.rank,role=self.role,pid=os.getpid(),command=self.command,nfe=self.runtime.context.get('nfe',0),
            markers=self.markers,static_tables=self.static_tables,shape_observations=self.shapes,
            shape_observations_dropped=self.shape_dropped,initial_segments=self.begin_segments,
            final_segments=segments,events=compact,stacks=stacks,
            allocator_settings=snapshot.get('allocator_settings'),
            external_annotations=snapshot.get('external_annotations',[]),
            history_limit=50000,history_limit_reached=len(events)>=50000,
            interpretation='free_requested is host release; free_completed is allocator-observed CUDA reuse eligibility. '
                'Reserved inactive storage remains cached. Known static categories are storage-deduplicated; other allocated includes inputs, activations and library scratch.')
        filename=f'{self.index:03d}_{self.command}_nfe{payload["nfe"]}.json.gz'
        self.index+=1
        with gzip.open(self.directory/filename,'wt',compresslevel=3) as stream:json.dump(payload,stream)
        self.chunks.append(dict(file=filename,command=self.command,nfe=payload['nfe'],actions=dict(actions),
            trace_events=len(events),history_limit_reached=payload['history_limit_reached']))
        (self.directory/'index.json').write_text(json.dumps(self.chunks,indent=2))
        self.markers=[];self.static_tables=[];self.shapes=[];self.begin_segments=[]
