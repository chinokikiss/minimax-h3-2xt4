"""CPU-only hash attribution inside benchmark workers, never runtime imports."""
import threading
import time
from pathlib import Path


class HashObserver:
    def __init__(self):
        from patches import tp2_bundle,tp2_disk_bake
        from patches.ptd import bank
        self.rows=[];self.originals=[]
        specs=[(tp2_bundle,'sha'),(tp2_disk_bake,'sha'),(bank,'sha'),(tp2_bundle,'state_hashes')]
        originals=[(module,name,getattr(module,name)) for module,name in specs]
        for module,name,original in originals:
            self.originals.append((module,name,original))
            def observed(*args,_fn=original,_name=name,**kwargs):
                start=time.perf_counter();cpu=time.thread_time()
                if _name=='sha':
                    path=Path(args[0]);size=path.stat().st_size;kind='payload' if path.suffix=='.safetensors' else 'metadata'
                    details=dict(path=str(path),bytes=size,kind=kind)
                else:
                    details=dict(bytes=sum(t.numel()*t.element_size() for t in args[0].state_dict().values()),kind='tensor_state')
                try:return _fn(*args,**kwargs)
                finally:self.rows.append(dict(function=_name,**details,start_monotonic=start,
                    wall_seconds=time.perf_counter()-start,thread_cpu_seconds=time.thread_time()-cpu,
                    tid=threading.get_native_id()))
            setattr(module,name,observed)

    def report(self):
        return dict(calls=list(self.rows),
            semantics='CPU hash spans can overlap prefetch/other threads; summed time is not critical-path savings')

    def close(self):
        for module,name,original in reversed(self.originals):setattr(module,name,original)
