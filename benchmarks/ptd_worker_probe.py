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
    preparations = []
    # CPU-only stage attribution lives in this benchmark process, not in the
    # production loader. Nested spans overlap and must never be added blindly.
    import patches.ptd.queue as queue
    import patches.ptd.bank as bank
    import patches.ptd.refiner_layout as layout
    import patches.ptd.auxiliary as auxiliary
    auxiliary_access = []

    def timed(name, function, select=None, reader_index=None):
        def call(*args, **kwargs):
            if select is not None and not select(*args, **kwargs): return function(*args, **kwargs)
            start, cpu = time.perf_counter(), time.thread_time()
            try: return function(*args, **kwargs)
            finally:
                reader = args[reader_index] if reader_index is not None else None
                preparations.append(dict(stage=name,hop=getattr(reader,'hop',None),
                    wall_seconds=time.perf_counter()-start,thread_cpu_seconds=time.thread_time()-cpu,
                    start_monotonic=start))
        return call

    queue.refiner_group = timed('refiner_total',queue.refiner_group,reader_index=1)
    bank.HopReader._verify_overlay = timed('raw_overlay_verify',bank.HopReader._verify_overlay,
        select=lambda reader:not reader.overlay_verified,reader_index=0)
    bank.HopReader.read_tensor = timed('raw_aux_read',bank.HopReader.read_tensor,
        select=lambda reader,name,**kw:name.startswith(('token_refiner.','final_layer.')),reader_index=0)
    layout.read_layout_tensor = timed('layout_read_and_verify',layout.read_layout_tensor,reader_index=0)
    layout.tensor_hash = timed('layout_tensor_checksum',layout.tensor_hash)
    auxiliary.tensor_hash = timed('split_tensor_checksum',auxiliary.tensor_hash)
    split_read = auxiliary.SplitAuxiliary.read_tensor
    def consumed(reader, name):
        unit = reader.by_name[name]
        start, cpu = time.perf_counter(), time.thread_time()
        value = split_read(reader, name)
        auxiliary_access.append(dict(hop=reader.hop,role=unit['role'],file=unit['file'],tensor=name,
            payload_bytes=value.numel()*value.element_size(),header_bytes=unit['header_bytes'],
            payload_verified=True,wall_seconds=time.perf_counter()-start,
            thread_cpu_seconds=time.thread_time()-cpu))
        return value
    auxiliary.SplitAuxiliary.read_tensor = consumed

    def report():
        rows=[]
        for row,start,stop in spans:
            rows.append(dict(row,default_stream_interval_ms=start.elapsed_time(stop)
                             if stop.query() else None))
        return dict(rank=rank,spans=rows,peak_allocated=torch.cuda.max_memory_allocated(rank),
            peak_reserved=torch.cuda.max_memory_reserved(rank),rss_bytes=psutil.Process().memory_info().rss,
            cpu_preparation_spans=list(preparations),
            auxiliary_consumption=list(auxiliary_access),
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
