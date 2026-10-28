"""Benchmark-only worker metrics; no profiling code in production operators."""
import json
import os
from pathlib import Path
import time


def worker(rank, config, budget, port, connection):
    import psutil
    import torch
    import patches.comfy_tp_runtime as runtime
    dispatch, snapshot = runtime.RankRuntime.dispatch, runtime.rank_snapshot
    spans = []

    def report():
        rows=[]
        for row,start,stop in spans:
            rows.append(dict(row,default_stream_interval_ms=start.elapsed_time(stop)
                             if stop.query() else None))
        return dict(rank=rank,spans=rows,peak_allocated=torch.cuda.max_memory_allocated(rank),
            peak_reserved=torch.cuda.max_memory_reserved(rank),rss_bytes=psutil.Process().memory_info().rss,
            measurement='RPC dispatch wall/thread CPU; CUDA default-stream intervals include idle/waits, not a kernel-time sum',
            observer='Two CUDA events per denoise/hop; identical for rebuild and queue')

    def measured(self, command, args=(), kwargs=None):
        if command not in ('denoise','ptd_hop'): return dispatch(self,command,args,kwargs)
        start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(rank));wall=time.perf_counter();cpu=time.thread_time()
        try: return dispatch(self,command,args,kwargs)
        finally:
            stop.record(torch.cuda.current_stream(rank))
            spans.append((dict(command=command,wall_seconds=time.perf_counter()-wall,
                               thread_cpu_seconds=time.thread_time()-cpu),start,stop))

    def persist(value):
        output=Path(os.environ['H3_PTD_PROBE_DIR']);output.mkdir(parents=True,exist_ok=True)
        path=output/f'worker_rank{rank}.json';partial=path.with_suffix('.partial')
        with partial.open('w',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        partial.replace(path)

    def measured_snapshot(owner):
        value=report();persist(value)  # Durable before the worker's stats ACK.
        return dict(snapshot(owner),ptd_probe=value)

    runtime.RankRuntime.dispatch=measured
    runtime.rank_snapshot=measured_snapshot
    try: runtime._worker(rank,config,budget,port,connection)
    finally:
        persist(report())
