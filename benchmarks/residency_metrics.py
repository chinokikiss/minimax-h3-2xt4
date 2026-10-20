"""Bounded, shared instrumentation for all residency timing policies."""
import time
import te_io_telemetry as io


def compact():
    state=io.snapshot()
    return {k:state[k] for k in ('memory_current','memory_stat','memory_events','psi_memory','psi_io','diskstats')}


def difference(a,b):
    result={}
    for key in ('memory_events','psi_memory','psi_io'):
        result[key]={k:b[key][k]-v for k,v in a[key].items() if k in b[key]}
    result['memory_stat']={k:b['memory_stat'][k]-v for k,v in a['memory_stat'].items()
        if k in b['memory_stat'] and k.startswith(('workingset_','pgscan_','pgsteal_'))}
    result['diskstats']={device:{k:b['diskstats'][device][k]-v for k,v in values.items() if k!='inflight'}
        for device,values in a['diskstats'].items() if device in b['diskstats']}
    result['memory_current_start']=a['memory_current'];result['memory_current_end']=b['memory_current']
    result['scope']='Shared cgroup/device counters; layered devices must not be summed. Boundary observation outside recorded request latency contributes to measured handoff gap.'
    return result


def install(pipeline):
    from patches.host_memory_pressure import CgroupMemory
    memory=CgroupMemory()
    native_request=pipeline.request
    def request(*args,**kwargs):
        a=compact();record,outputs=native_request(*args,**kwargs);b=compact()
        record['host_io']=difference(a,b)
        return record,outputs
    pipeline.request=request
    # Extend the existing sampler without introducing another telemetry thread.
    cls=pipeline.MemoryMonitor
    native_run=cls.run
    class Rows(list):
        def append(self,row):
            # NVML data remains from the same instant; retain one in five samples.
            now=time.monotonic()
            if now-getattr(self,'last',0)<.25:return
            self.last=now;row['cgroup']=memory.snapshot()
            if len(self)>=8192:del self[0]
            super().append(row)
    def run(self):
        self.rows=Rows();native_run(self)
    cls.run=run
