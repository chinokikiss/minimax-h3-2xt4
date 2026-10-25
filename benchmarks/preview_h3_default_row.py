"""Small local integration preview; SM120 logical TP2 is not T4 evidence."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
from patches import h3_operator_policy as policy,optimized_tp,parallel_int8 as pi
from patches import tilelang_swiglu_quant as fused,tilelang_transport_p2 as transport
from patches import tilelang_transport_tradeoffs as codec,tilelang_row_quant as row
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from tp2_tradeoffs import transport_reference


def rank_cases(comm,rank):
    records=[]
    for name,k,n in (('te_down',12800,5120),('dit_fc2',7168,5376),('te_o',4096,5120),('dit_o',3584,5376)):
        generator=torch.Generator(device='cuda').manual_seed(951+rank)
        activated=name in ('te_down','dit_fc2')
        x=torch.randn((1,3,k*(2 if activated else 1)),generator=generator,device='cuda',dtype=torch.float16)
        if activated:x[0,0,0]=1000.;x[0,0,k]=1000.
        weight=torch.randint(-127,128,(n,k),generator=generator,device='cuda',dtype=torch.int8)
        ws=torch.full((n,1),.0003,device='cuda');act='swiglu' if activated else None
        module=SimpleNamespace(weight=weight,weight_scale=ws,world_size=2,convrot_groupsize=256,operator_policy=policy.H3OperatorPolicy())
        if activated:q,scale=fused.prepare(x,comm,'fused_stored',arch='sm_120')
        else:
            full,start,length=pi.gather(x.float(),comm,-1)
            codes,scale=pi.quantize(full,256);q=codes[:,start:start+length].contiguous()
        partial=pi.int_mm(q,weight);partials=[torch.empty_like(partial) for _ in range(2)]
        comm.all_gather(partials,partial)
        expected=pi.dequantize(transport_reference(partials,'fp16',k),scale,ws,torch.float32).reshape(1,3,n)
        actual=(policy.row_linear_parts(module,*x.chunk(2,-1),optimized_tp.row_linear,comm) if name=='te_down'
            else policy.row_linear(module,x,optimized_tp.row_linear,comm,act))
        assert actual.dtype==torch.float32 and torch.equal(actual,expected),(name,rank)
        assert torch.isfinite(actual).all()
        records.append(dict(name=name,rank=rank,codec_reference_exact=True,finite=True))
    return records


def main():
    import tilelang
    assert torch.cuda.get_device_capability()==(12,0)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    optimized_tp.enable();isolate_cublas_workspaces();compile_real=tilelang.compile
    def compile_preview(fn,**kwargs):
        kwargs['target']={'kind':'cuda','arch':'sm_120'};return compile_real(fn,**kwargs)
    def finish(p,a,w,scheme,k):return transport.epilogue(a.shape[0],w.numel(),scheme,codec.shift_for(k))(p,a,w.reshape(-1))
    with ExitStack() as stack:
        stack.enter_context(patch.object(tilelang,'compile',new=compile_preview))
        stack.enter_context(patch.object(policy,'sm75',return_value=True))
        stack.enter_context(patch.object(fused,'supported',return_value=True))
        stack.enter_context(patch.object(transport,'_supported',return_value=True))
        stack.enter_context(patch.object(transport,'finish',new=finish))
        stack.enter_context(patch.object(row,'_device',side_effect=lambda x:None))
        comm=LogicalDualGPU(timeout=120)
        records=[r for group in comm.run([lambda i=i:rank_cases(comm,i) for i in (0,1)]) for r in group]
    output=Path(sys.argv[1]);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='passed',scope='Local SM120 integration preview, logical TP2, no T4/NCCL/timing qualification',records=records),indent=2))
    print('DEFAULT_ROW_PREVIEW_PASS',len(records))


if __name__=='__main__':main()
