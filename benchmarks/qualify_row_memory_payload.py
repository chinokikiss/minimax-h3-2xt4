"""Local INT8 payload/scale gate for rotation buffer lifetime changes."""
import json,hashlib
from pathlib import Path
import simulate_weight_residency
import torch
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from patches.parallel_int8 import quantize
from patches.optimized_tp import global_row_quantize
from row_partial_memory_baseline import global_row_quantize as old_quantize

def main():
    import comfy_kitchen as ck
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda _:49152
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;isolate_cublas_workspaces()
    rows=[]
    for count in (33,257,513,8192):
        torch.manual_seed(1729);x=torch.randn(count,28672,device='cuda');gate,up=x.chunk(2,-1)
        parts=[torch.cat((gate.chunk(2,-1)[r],up.chunk(2,-1)[r]),-1).contiguous() for r in (0,1)]
        qref,sref=quantize(x,input_act='swiglu')
        for name,fn in (('before',old_quantize),('optimized',global_row_quantize)):
            comm=LogicalDualGPU(180)
            outputs=comm.run([lambda r=r:fn(parts[r],comm,2,256,'swiglu') for r in (0,1)])
            exact=all(torch.equal(q,qref.chunk(2,-1)[r]) and torch.equal(s.contiguous().view(torch.uint8),sref.contiguous().view(torch.uint8)) for r,(q,s) in enumerate(outputs))
            rows.append(dict(rows=count,variant=name,native_payload_and_scale_bytes_exact=exact))
            if not exact:raise RuntimeError('Quantized payload or scale differs')
            del outputs
        del x,gate,up,parts,qref,sref;torch.cuda.empty_cache()
    root=Path(__file__).resolve().parents[1]
    result=dict(passed=True,physical_t4=False,cases=rows,source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ('patches/optimized_tp.py','benchmarks/row_partial_memory_baseline.py','benchmarks/qualify_row_memory_payload.py')})
    (root/'semantic_results/h3_row_memory_payload.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('PAYLOAD_GATE_PASS',len(rows),'Native INT8 payload and scale byte checks',flush=True)

if __name__=='__main__':main()
