"""Independent P0/P1 policies; inherited real requests and Native validation."""
import os
import sys
from dataclasses import dataclass
import causal_pipeline_audit as pipeline

POLICY=os.environ.get('MINIMAX_H3_RESIDENCY_POLICY','baseline')
if POLICY not in ('baseline','p0','p0_p1'):
    raise ValueError('Unknown residency policy')
@dataclass(frozen=True)
class ResidencyConfig(pipeline.DiagnosticConfig):
    prefetch_lifecycle: bool = POLICY!='baseline'
    memory_pressure_aware: bool = POLICY=='p0_p1'


pipeline.DiagnosticConfig=ResidencyConfig
import residency_metrics
residency_metrics.install(pipeline)
if '--capture' in sys.argv and sys.argv[sys.argv.index('--capture')+1]=='kineto':
    from te_pipeline_ablation import TERecorder
    from contextlib import contextmanager
    import cpu_causal_trace as cpu
    class WarmTERecorder(TERecorder):
        def start(self,tag,kineto=True):
            self.warm_kineto=kineto
            super().start(tag,False)

        @contextmanager
        def span(self,name,**fields):
            if self.enabled and name in ('rank_dispatch/te','stage/reference_and_TE'):
                self.profile_requested=self.warm_kineto and self.fields().get('request')==1
            with super().span(name,**fields):
                yield
    pipeline.tracing.Recorder=WarmTERecorder
    pipeline.tracing.install=cpu.install

if __name__=='__main__':pipeline.main()
