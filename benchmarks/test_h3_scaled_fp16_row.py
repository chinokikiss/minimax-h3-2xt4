"""End-to-end row operator arithmetic, on physical TP2 or explicit local preview.

Small M with current TP2 K/N. No full H3 pipeline or performance claims.
Run torchrun --nproc_per_node=2 ... --distributed --output ... on SM75;
--local-preview uses one SM120 and two logical owners, never NCCL evidence.
"""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import torch
from patches import h3_operator_policy as policy, optimized_tp as native, parallel_int8 as pi
from patches import tilelang_transport_p2 as transport, tilelang_transport_tradeoffs as codec
from patches import tilelang_row_quant as row
from tp2_tradeoffs import transport_reference


def rank_cases(comm,rank,device):
    records=[];torch.manual_seed(438+rank)
    for kind,width,output in (('dit_o',3584,5376),('dit_fc2',7168,5376),
                              ('te_o',4096,5120),('te_down',12800,5120)):
        gate=torch.randn(1,17,width,device=device);up=torch.randn_like(gate)
        weight=torch.randint(-127,128,(output,width),device=device,dtype=torch.int8)
        ws=torch.full((output,1),.0003,device=device)
        parts=kind=='te_down';activated=kind in ('dit_fc2','te_down')
        x=torch.cat((gate,up),-1) if activated else gate
        act='swiglu' if activated else None
        module=SimpleNamespace(weight=weight,weight_scale=ws,world_size=2,convrot_groupsize=256,
            operator_policy=policy.H3OperatorPolicy())
        # Full gather and Native global-row quantization: independent of TileLang
        # preparation. Keep the pre-SUM partials for the lossy codec reference.
        full,start,length=pi.gather(x,comm,-1)
        if activated:
            pieces=full.chunk(2,-1)
            full=torch.cat([p.chunk(2,-1)[0] for p in pieces]+[p.chunk(2,-1)[1] for p in pieces],-1)
            start//=2;length//=2
        q,scale=pi.quantize(full,256,input_act=act)
        partial=pi.int_mm(q[:,start:start+length].contiguous(),weight)
        gathered=[torch.empty_like(partial) for _ in range(2)];comm.all_gather(gathered,partial)
        exact=pi.dequantize(gathered[0]+gathered[1],scale,ws,torch.float32).reshape(1,17,output)
        expected=pi.dequantize(transport_reference(gathered,'fp16',width),scale,ws,torch.float32).reshape(1,17,output)
        baseline=(policy.row_linear_parts(module,gate,up,native.row_linear,comm) if parts else
                  policy.row_linear(module,x,native.row_linear,comm,act))
        assert torch.equal(baseline,exact),(kind,'default INT32 SUM changed')
        module.operator_policy=replace(module.operator_policy,row_linear_backend='tilelang_scaled_fp16')
        actual=(policy.row_linear_parts(module,gate,up,native.row_linear,comm) if parts else
                policy.row_linear(module,x,native.row_linear,comm,act))
        assert torch.equal(actual,expected),(kind,'FP16 codec + fused epilogue mismatch',float((actual-expected).abs().max()))
        assert torch.isfinite(actual).all()
        records.append(dict(name=kind,rank=rank,M=17,K=width,N=output,
            default_native_int32='bitwise_pass',fp16_independent_reference='bitwise_pass'))
        if activated:
            module.operator_policy=replace(module.operator_policy,swiglu_padding=False)
            unfused=policy.row_linear(module,x,native.row_linear,comm,act)
            assert torch.equal(unfused,expected),(kind,'unfused activation path')
            records[-1]['unfused_activation']='bitwise_pass'
    return records


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--local-preview',action='store_true');mode.add_argument('--distributed',action='store_true')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();sys.argv=sys.argv[:1]
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    native.enable()
    with ExitStack() as stack:
        if args.local_preview:
            assert torch.cuda.get_device_capability()==(12,0)
            import tilelang
            from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
            compile_real=tilelang.compile
            def preview(fn,**kw):kw['target']={'kind':'cuda','arch':'sm_120'};return compile_real(fn,**kw)
            def device_check(x):assert x.is_cuda and x.dtype==torch.float32
            def finish(p,a,w,scheme,k):return transport.epilogue(a.shape[0],w.numel(),scheme,codec.shift_for(k))(p,a,w.reshape(-1))
            # Explicit test-only guard bypasses for factories retargeted to SM120.
            # Production hardware queries/SM75 guards remain untouched.
            stack.enter_context(patch.object(tilelang,'compile',new=preview))
            stack.enter_context(patch.object(row,'_device',new=device_check))
            stack.enter_context(patch.object(transport,'_supported',return_value=True))
            stack.enter_context(patch.object(transport,'finish',new=finish))
            stack.enter_context(patch.object(policy,'sm75',return_value=True))
            comm=LogicalDualGPU(timeout=120);isolate_cublas_workspaces()
            results=comm.run([lambda r=r:rank_cases(comm,r,'cuda:0') for r in (0,1)])
            records=[record for group in results for record in group];output=args.output
        else:
            import torch.distributed as comm
            rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
            assert torch.cuda.get_device_capability()==(7,5)
            comm.init_process_group('nccl')
            try:records=rank_cases(comm,rank,torch.device('cuda',rank));comm.barrier()
            finally:comm.destroy_process_group()
            output=args.output.with_name(args.output.stem+f'_rank{rank}.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(hardware=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
        scope='SM120 arithmetic preview with logical transport' if args.local_preview else 'Physical SM75 NCCL arithmetic gates',records=records),indent=2))
    print('SCALED_FP16_ROW_PASS',len(records),flush=True)


if __name__=='__main__':main()
