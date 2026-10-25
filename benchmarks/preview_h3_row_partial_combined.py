"""Production FP16 row + fused stored + scaled codec + row-A SUM, local only."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import torch
from torch.nn import functional as F
import preview_h3_default_row as preview
from patches import tp_dit as dit
from patches.h3_lora import RowPartialLoRA


def rank_cases(comm,rank):
    dit.dist=comm;records=[]
    for name,k,n in (('dit_fc2',7168,5376),('dit_o',3584,5376)):
        activated=name=='dit_fc2';m=3;act='swiglu' if activated else None
        generator=torch.Generator(device='cuda').manual_seed(1151+rank)
        x=torch.randn((m,k*(2 if activated else 1)),generator=generator,device='cuda',dtype=torch.float16)
        if activated:x[0,0]=1000.;x[0,k]=1000.
        module=dit.RowParallelLinearINT8(2*k,n,rank=rank,device='cuda')
        module.weight.data.copy_(torch.randint(-127,128,(n,k),generator=generator,device='cuda',dtype=torch.int8))
        module.weight_scale.data.fill_(.0003/64.)
        module.operator_policy=preview.policy.H3OperatorPolicy()
        # Identical full factors across both owners, with distinct ordered terms.
        generator=torch.Generator(device='cuda').manual_seed(2234)
        terms=[(torch.randn(128,2*k,generator=generator,device='cuda')*.001,
                torch.randn(n,128,generator=generator,device='cuda')*.01,s/64.) for s in (.7,-.25)]
        if activated:q,scale=preview.fused.prepare(x,comm,'fused_stored',arch='sm_120')
        else:
            full,start,length=preview.pi.gather(x.float(),comm,-1)
            codes,scale=preview.pi.quantize(full,256);q=codes[:,start:start+length].contiguous()
        partial=preview.pi.int_mm(q,module.weight);partials=[torch.empty_like(partial) for _ in range(2)]
        comm.all_gather(partials,partial)
        expected=preview.pi.dequantize(preview.transport_reference(partials,'fp16',k),scale,module.weight_scale,torch.float32)
        own=(F.silu(x[:,:k].float())*x[:,k:].float()) if activated else x.float()
        for a,b,s in terms:
            hidden=F.linear(own,a[:,rank*k:(rank+1)*k]);parts=[torch.empty_like(hidden) for _ in range(2)]
            comm.all_gather(parts,hidden);expected.add_(F.linear(parts[0]+parts[1],b).mul_(s))
        events=[]
        class HiddenComm:
            ReduceOp=comm.ReduceOp
            def get_world_size(self):return 2
            def all_reduce(self,value,op):
                events.append(dict(dtype=str(value.dtype),shape=list(value.shape)))
                assert value.dtype==torch.float32 and tuple(value.shape)==(m,128)
                comm.all_reduce(value,op=op)
        class ObservedPartial(RowPartialLoRA):
            def add_to(self,x,out,comm=None,input_act=None):
                return super().add_to(x,out,HiddenComm(),input_act)
        adapter=ObservedPartial([(a[:,rank*k:(rank+1)*k],b,s) for a,b,s in terms],(rank*k,(rank+1)*k))
        adapter.fp32_compute=True;module.lora_adapter=adapter
        actual=module(x,input_act=act)
        assert actual.dtype==torch.float16 and torch.equal(actual,expected.half()),(name,rank)
        assert torch.isfinite(actual).all() and len(events)==2
        records.append(dict(name=name,rank=rank,partial_codec_reference_exact=True,finite=True,
            lora_collectives=events,lora_full_activation_gather=False,storage_dtype=str(actual.dtype)))
    return records


if __name__=='__main__':
    preview.rank_cases=rank_cases
    preview.main()
