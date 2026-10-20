"""Fixed-count warm-up for operations containing rank-ordered collectives."""
import statistics
import time
import torch


def measure(fn,warmup=10,repeats=10):
    # Call counts must agree across ranks. A wall-clock stopping condition in
    # each process can advance one rank into a different collective.
    # JIT between candidates can return T4 to idle clocks. Prewarm locally,
    # outside the measured operation, then align the two ranks. The following
    # operator warm-up still has an identical fixed call count on both ranks.
    import torch.distributed as dist
    if dist.is_initialized():dist.barrier()
    warm_device()
    if dist.is_initialized():dist.barrier()
    begin=time.perf_counter()
    for _ in range(warmup):fn()
    torch.cuda.synchronize();warm_seconds=time.perf_counter()-begin
    gpu=[];wall=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();a.record();fn();b.record();b.synchronize()
        gpu.append(a.elapsed_time(b));wall.append((time.perf_counter()-start)*1000)
    return {'warmup_actual_calls':warmup,'warmup_seconds':warm_seconds,'warmup_policy':'fixed identical count on both ranks; device clocks prewarmed separately',
            'repeats':repeats,'gpu_ms':gpu,'wall_ms':wall,'gpu_median_ms':statistics.median(gpu),'wall_median_ms':statistics.median(wall)}


def warm_device():
    a=torch.randn(1024,1024,device='cuda',dtype=torch.float16)
    until=time.perf_counter()+1.
    while time.perf_counter()<until:
        for _ in range(10):torch.mm(a,a)
        torch.cuda.synchronize()
