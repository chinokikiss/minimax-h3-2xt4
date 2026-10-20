"""Actual full-request TE loading ablations; detailed capture is opt-in."""
from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import sys
import torch

# Same policy is imported in every spawned worker before runtime construction.
mode=os.environ.get('MINIMAX_H3_TE_ABLATION','baseline')
import causal_pipeline_audit as pipeline
import cpu_causal_trace as cpu
from patches.te_loading import install
install(mode)


class TERecorder(cpu.CPURecorder):
    """Capture Kineto only for the real TE/reference stage of one request.

    Captures are separate diagnostic passes. Profiler start/stop perturbation
    stays visible and is never used as unprofiled performance evidence.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.te_profiler=None;self.profile_requested=False;self.in_te=False

    def start(self,tag,kineto=True):
        self.profile_requested=kineto
        super().start(tag,False)

    @contextmanager
    def span(self,name,**fields):
        if not self.enabled:
            yield;return
        outer=name in ('rank_dispatch/te','stage/reference_and_TE')
        block=str(fields.get('block',self.fields().get('block','')))
        relevant=(outer or self.context.get('stage') in ('TE','reference_and_TE') or "'te'" in block or "'te_startup'" in block)
        if not relevant:
            yield;return
        started_here=outer and self.profile_requested and self.te_profiler is None
        if started_here:
            self.te_profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)
            self.te_profiler.__enter__();self.in_te=True
        try:
            with super().span(name,**fields):yield
        finally:
            # A logical-rank qualification nests worker scopes inside the
            # parent's TE stage. Only the creating scope/thread may close its
            # profiler session; nested dispatch scopes never take ownership.
            if started_here and self.in_te:
                self.te_profiler.__exit__(None,None,None);self.in_te=False

    def emit(self,name,*args,**fields):
        block=str(fields.get('block',self.fields().get('block','')))
        if (self.context.get('stage') in ('TE','reference_and_TE') or
            "'te'" in block or "'te_startup'" in block or name in ('capture_start','capture_end')):
            super().emit(name,*args,**fields)

    def interval(self,name,begin,end,**fields):
        block=str(fields.get('block',self.fields().get('block','')))
        if "'te'" in block or "'te_startup'" in block:super().interval(name,begin,end,**fields)

    def stop(self):
        path=super().stop()
        if self.te_profiler:
            self.te_profiler.export_chrome_trace(str(self.output/f'{self.tag}_r{self.rank}_kineto.json'))
            self.te_profiler=None
        return path


config_class=pipeline.DiagnosticConfig
if mode=='coalesced_parallel_priority':
    # Controlled priority ablation: delay speculative DiT loading until the
    # real begin_sample handoff. Native request/collective math is unchanged.
    def make_config(*args,**kwargs):
        kwargs['cross_model_cpu']=False
        return config_class(*args,**kwargs)
    pipeline.DiagnosticConfig=make_config

if '--capture' in sys.argv and sys.argv[sys.argv.index('--capture')+1]=='kineto':
    pipeline.tracing.Recorder=TERecorder
    pipeline.tracing.install=cpu.install


if __name__=='__main__':pipeline.main()
