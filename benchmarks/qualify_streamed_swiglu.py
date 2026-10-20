"""Native INT8 payload equivalence and local CUDA memory, not T4 timings."""
import hashlib
import json
from pathlib import Path
import torch
import simulate_weight_residency as setup
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from patches.parallel_int8 import quantize
from patches.optimized_tp import streamed_row_quantize,global_row_quantize


def main():
    import comfy_kitchen as ck
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda _:49152
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;isolate_cublas_workspaces()
    rows=[];root=Path(__file__).resolve().parents[1]
    for count in (33,257,512,513,2048,8192):
        torch.manual_seed(1729);x=torch.randn((count,28672),device='cuda:0',dtype=torch.float32)
        gate,up=x.chunk(2,-1)
        inputs=[torch.cat((gate.chunk(2,-1)[r],up.chunk(2,-1)[r]),-1).contiguous() for r in (0,1)]
        qref,sref=quantize(x,input_act='swiglu');torch.cuda.synchronize()
        for mode in ('original','streamed'):
            comm=LogicalDualGPU(timeout=180);torch.cuda.reset_peak_memory_stats()
            base=torch.cuda.memory_allocated()
            fn=global_row_quantize if mode=='original' else streamed_row_quantize
            outputs=comm.run([lambda r=r:fn(inputs[r],comm,2,256,'swiglu') for r in (0,1)])
            peak=torch.cuda.max_memory_allocated()-base
            exact=all(torch.equal(q,qref.chunk(2,-1)[r]) and torch.equal(s,sref) for r,(q,s) in enumerate(outputs))
            rows.append(dict(rows=count,mode=mode,native_payload_bit_exact=exact,
                mismatched_int8=[int((q!=qref.chunk(2,-1)[r]).sum()) for r,(q,s) in enumerate(outputs)],
                temporary_peak_bytes=peak))
            print('SWIGLU',count,mode,'exact',exact,'temporary_MiB',round(peak/2**20,2),flush=True)
            del outputs
        del x,gate,up,inputs,qref,sref;torch.cuda.empty_cache()
    result=dict(passed=all(r['native_payload_bit_exact'] for r in rows),physical_t4=False,
        scope='Actual CUDA FP32 SwiGLU→ConvRot→global maximum→INT8, H3 FFN14336, logical ranks; no T4 latency claim',cases=rows,
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ('patches/optimized_tp.py','patches/parallel_int8.py','benchmarks/qualify_streamed_swiglu.py')})
    (root/'semantic_results/streamed_swiglu.json').write_text(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Streaming changes Native quantization; do not promote')


if __name__=='__main__':main()
