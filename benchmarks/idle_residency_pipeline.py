"""Full physical requests with idle work explicitly outside output-ready latency."""
import os
import time
from dataclasses import dataclass
import residency_pipeline_ablation as ablation
import residency_metrics as metrics

pipeline=ablation.pipeline
ENABLED=os.environ.get('MINIMAX_H3_IDLE_RESIDENCY','0')=='1'
GAP=float(os.environ.get('MINIMAX_H3_IDLE_GAP','0'))
if GAP not in (0.,5.,15.,60.):raise ValueError('Use the established idle-gap matrix')
if ENABLED and ablation.POLICY=='baseline':raise ValueError('Idle requires P0 lifecycle')


@dataclass(frozen=True)
class IdleConfig(pipeline.DiagnosticConfig):
    idle_residency: bool=ENABLED


pipeline.DiagnosticConfig=IdleConfig
# The real worker polls its actual pipe between bounded idle ticks.
pipeline.ObservedConnection.poll=lambda self,timeout=0:self.connection.poll(timeout)
native_request=pipeline.request
previous=None


def request(service,clip,model,vaes,args,recorder,index):
    global previous
    if previous is not None and GAP:
        before=metrics.compact();start=time.monotonic_ns()
        accepted=service.request_idle() if ENABLED else False
        deadline=previous['end_ns']/1e9+GAP
        remaining=deadline-time.monotonic()
        if remaining>0:time.sleep(remaining)
        # This compact control call is outside request latency and is reported
        # in the actual handoff gap; it does not synchronize the copy stream.
        idle=service.call('idle_stats') if ENABLED else None
        end=time.monotonic_ns();after=metrics.compact()
        previous['idle_after']=dict(configured_seconds=GAP,start_ns=start,end_ns=end,
            accepted=accepted,ranks=idle,host_io=metrics.difference(before,after),
            semantics='Background work and I/O moved into user think time, not eliminated; actual gap also includes boundary instrumentation')
    record,outputs=native_request(service,clip,model,vaes,args,recorder,index)
    record.update(idle_enabled=ENABLED,configured_idle_gap_seconds=GAP)
    previous=record
    return record,outputs


pipeline.request=request
if __name__=='__main__':pipeline.main()
