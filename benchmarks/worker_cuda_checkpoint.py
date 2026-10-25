"""Benchmark-only explicit CUDA trace checkpoints on the owning worker thread."""
import json
import os
from pathlib import Path
import threading
import time


PROTOCOL='explicit_checkpoint_restart_v1'


class CUDAFlushCheckpoint:
    def __init__(self,rank,output,*,device=None,backend=None):
        if backend is None:
            import torch
            backend=torch.cuda
        self.backend=backend;self.rank=rank;self.device=rank if device is None else device
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.owner=threading.get_ident();self.records=[]

    def report(self):
        return dict(protocol=PROTOCOL,rank=self.rank,pid=os.getpid(),records=list(self.records),
            semantics='Explicit synchronize/stop/start instrumentation; wall duration includes waiting for preceding GPU work and is not subtractable as exclusive profiler overhead.')

    def flush(self,label,*,resume=True):
        if threading.get_ident()!=self.owner:raise RuntimeError('CUDA trace checkpoints require the owning worker thread')
        api=self.backend;start=time.monotonic_ns();cpu=time.thread_time_ns()
        row=dict(label=label,resume_requested=resume,stopped=False,restarted=False,status='started',start_ns=start)
        api.nvtx.range_push(f'H3_FLUSH/rank={self.rank}/{label}')
        try:
            api.synchronize(self.device)
            api.profiler.stop();row['stopped']=True
            if resume:
                api.profiler.start();row['restarted']=True
            row['status']='complete'
        except BaseException as error:
            row.update(status='failed',error=type(error).__name__);raise
        finally:
            api.nvtx.range_pop()
            row.update(end_ns=time.monotonic_ns(),wall_seconds=(time.monotonic_ns()-start)/1e9,
                thread_cpu_seconds=(time.thread_time_ns()-cpu)/1e9)
            self.records.append(row)
            path=self.output/f'rank{self.rank}_cuda_checkpoints.json';temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.report(),indent=2),encoding='utf-8');temporary.replace(path)
        return row
