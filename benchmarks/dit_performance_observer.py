"""Benchmark-only NVTX/CPU/CUDA intervals; original operators run once unchanged."""
from contextlib import contextmanager
import functools
import json
import threading
import time

import torch


class Phases:
    def __init__(self, rank):
        self.rank, self.enabled, self.block = rank, False, None
        self.cpu, self.gpu, self.originals = [], [], []
        self.cpu_probe=None
        self.owner_thread=threading.get_ident()

    @contextmanager
    def span(self, name, gpu=True):
        if not self.enabled or threading.get_ident()!=self.owner_thread:
            yield
            return
        label=f'H3_DIT/rank={self.rank}/block={self.block}/{name}'
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        begin=torch.cuda.Event(enable_timing=True) if gpu else None
        if begin:begin.record()
        torch.cuda.nvtx.range_push(label)
        try:yield
        finally:
            torch.cuda.nvtx.range_pop()
            row=dict(name=name,rank=self.rank,block=self.block,pid=__import__('os').getpid(),
                tid=threading.get_native_id(),start_ns=start,end_ns=time.monotonic_ns(),
                thread_cpu_ns=time.thread_time_ns()-cpu)
            self.cpu.append(row)
            if begin:
                end=torch.cuda.Event(enable_timing=True);end.record()
                self.gpu.append((row.copy(),begin,end))

    def wrap(self, module, name, label, gpu=True):
        original=getattr(module,name)
        @functools.wraps(original)
        def observed(*args,**kwargs):
            name=label(args,kwargs) if callable(label) else label
            def body(*a,**kw):
                with self.span(name,gpu):return original(*a,**kw)
            if self.cpu_probe and threading.get_ident()==self.owner_thread and name in ('norm/modulate','residual/gated_add'):
                # Include Event.record/NVTX/context-manager overhead in the
                # worker CPU profile, as well as the operator itself.
                return self.cpu_probe.invoke(name,body,args,kwargs)
            return body(*args,**kwargs)
        self.originals.append((module,name,original));setattr(module,name,observed)

    def install(self):
        from patches import tp_dit,parallel_int8,tilelang_swiglu_quant,tilelang_transport_p2,h3_tensor_flow
        from comfy_kitchen.backends.eager import quantization as eager
        import comfy_kitchen as ck
        import tilelang
        from patches import tilelang_grouped_tp_optimized as sparse,tilelang_attention_p2 as split
        self.wrap(tp_dit.ColumnParallelLinearINT8,'forward',
            lambda a,k:'QKV/column' if a[0].components==3 else 'FC1/column')
        self.wrap(tp_dit.RowParallelLinearINT8,'forward',
            lambda a,k:'O/row' if a[0].in_features_per_rank==3584 else 'FC2/row')
        for module,name,label in (
            (h3_tensor_flow,'modulated_norm','norm/modulate'),
            (h3_tensor_flow,'gated_add','residual/gated_add'),
            (ck,'rms_rope_split_half','attention/QK_norm_RoPE'),
            (ck,'int8_attention','attention/Kitchen_fused_QK_softmax_PV'),
            (sparse,'prepare','attention/sparse_quantize_and_centroids'),
            (sparse,'routes','attention/sparse_routing'),
            (sparse,'execute','attention/sparse_execute'),
            (split,'execute','attention/scalar_split_execute'),
            (eager,'_rotate_activation','column/ConvRot'),
            (eager,'quantize_int8_rowwise','column/quantize'),
            (eager,'_int8_matmul_accumulate','column/INT8_GEMM'),
            (parallel_int8,'int_mm','row/INT8_GEMM'),
            (tilelang_swiglu_quant,'prepare','FC2/SwiGLU_ConvRot_quantize'),
            (tilelang_transport_p2,'encode','transport/encode'),
            (tilelang_transport_p2,'collect','transport/SUM'),
            (tilelang_transport_p2,'finish','transport/fused_epilogue')):
            self.wrap(module,name,label)
        self.wrap(tilelang,'compile','compile/TileLang',gpu=False)

    def report(self):
        torch.cuda.synchronize()
        return dict(cpu=self.cpu,gpu=[row|dict(stream_elapsed_ms=a.elapsed_time(b)) for row,a,b in self.gpu],
            semantics='Nested stream elapsed intervals include dependency/host launch gaps; not exclusive kernel busy time. Use Nsight CUDA correlations for busy time.')
