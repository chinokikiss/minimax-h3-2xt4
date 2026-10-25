"""Read-only line markers: never inspect or retain a frame's tensor locals."""
import inspect
import linecache
import sys
import gzip
import json
import os


class LineMarkers:
    def __init__(self, audit, functions, checkpoint=None, representative_only=False):
        self.audit = audit
        self.codes = {fn.__code__: name for name, fn in functions.items()}
        self.previous = None
        self.checkpoint = checkpoint
        self.representative_only = representative_only

    def trace(self, frame, event, arg):
        name = self.codes.get(frame.f_code)
        if name is None:
            return None
        if event == 'line' and self.audit.active:
            if self.representative_only and name != 'DiT' and self.audit.block not in (0,24,49):
                return self.trace
            # f_locals would risk extending lifetimes through a retained locals
            # dictionary. Source-line identity is sufficient for alloc stacks.
            text = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
            self.audit.mark(name + ':' + str(frame.f_lineno) + ':' + text[:160],
                            static=name == 'DiT' and 'fc1_out = self.fc1' in text)
            if (self.checkpoint is not None and name == 'DiT' and self.audit.block in (0,24,49)
                    and ('fc1_out = self.fc1' in text or 'del normed_x2, h2' in text)):
                self.checkpoint('b'+str(self.audit.block)+'_fc1_'+('before' if 'fc1_out = self.fc1' in text else 'after'))
        return self.trace

    def __enter__(self):
        self.previous = sys.gettrace()
        if self.previous is not None:
            raise RuntimeError('Do not replace another active Python tracer')
        sys.settrace(self.trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous)


def source_bindings(functions):
    return {name: dict(file=inspect.getsourcefile(fn), first_line=fn.__code__.co_firstlineno)
            for name, fn in functions.items()}


def checkpoint_history(audit, label):
    """Persist an OOM-safe snapshot without stopping or clearing the history."""
    import torch
    audit.mark('checkpoint/'+label,static=True)
    snapshot=torch.cuda.memory._snapshot()
    stacks=[];ids={}
    def intern(frames):
        key=tuple((f.get('filename'),f.get('line'),f.get('name')) for f in frames)
        if key not in ids:ids[key]=len(stacks);stacks.append(key)
        return ids[key]
    traces=snapshot.get('device_traces',[])
    raw=traces[audit.rank] if audit.rank<len(traces) else []
    events=[]
    for event in raw:
        item={k:v for k,v in event.items() if k not in ('frames','compile_context')}
        item['stack']=intern(event.get('frames',[]));events.append(item)
    payload=dict(rank=audit.rank,role=audit.role,pid=os.getpid(),command=audit.command,nfe=audit.nfe,
        checkpoint=True,checkpoint_label=label,markers=audit.markers,static_tables=audit.static_tables,
        shape_observations=audit.shapes,shape_observations_dropped=audit.shape_dropped,
        initial_segments=audit.begin_segments,final_segments=audit._segments(snapshot,intern),
        events=events,stacks=stacks,history_limit=50000,history_limit_reached=len(raw)>=50000,
        interpretation='Crash-safe prefix snapshot; recording continues without a history gap. Not a completed command.')
    target=audit.directory/f'checkpoint_{audit.command}_nfe{audit.nfe}_{label}.json.gz'
    temporary=target.with_suffix(target.suffix+'.partial')
    with gzip.open(temporary,'wt',compresslevel=3) as stream:json.dump(payload,stream)
    temporary.replace(target)
