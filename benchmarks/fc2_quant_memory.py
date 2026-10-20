"""FC2 quantizer memory A/B: frozen before, lifetime fix, owned-rotation study."""
import argparse
from datetime import timedelta
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
if '--physical' not in sys.argv:import simulate_weight_residency
import torch
import torch.distributed as dist
import comfy_kitchen as ck
from patches import tilelang_swiglu_pad as fixed, optimized_tp as old, parallel_int8 as pi
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.h3_lora import H3LoRAStore
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from fc2_quant_candidates import owned_rotation


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf8');temp.replace(path)


def metrics(a,b):
    exact=torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if exact:return dict(bytes_exact=True,finite=finite,max_abs=0.,rel_l2=0.,cosine=1.)
    maximum=error=aa=bb=ab=0.
    for start in range(0,a.numel(),2**20):
        x=a.flatten()[start:start+2**20].double();y=b.flatten()[start:start+2**20].double();d=x-y
        maximum=max(maximum,float(d.abs().max()));error+=float(d.square().sum())
        aa+=float(x.square().sum());bb+=float(y.square().sum());ab+=float((x*y).sum())
    return dict(bytes_exact=False,finite=finite,max_abs=maximum,rel_l2=(error/max(bb,1e-300))**.5,
                cosine=ab/max((aa*bb)**.5,1e-300))


class LogicalComm(LogicalDualGPU):
    def __init__(self):super().__init__(180);self.maxima={}
    def all_reduce(self,x,op=None):
        super().all_reduce(x,op)
        if op==self.ReduceOp.MAX:self.maxima[self.get_rank()]=x.clone()


class FrozenMaximum:
    ReduceOp=dist.ReduceOp
    def __init__(self,rank,maximum):self.rank,self.maximum=rank,maximum
    def get_rank(self):return self.rank
    def all_reduce(self,x,op):
        if op!=self.ReduceOp.MAX:raise ValueError('Local diagnostic covers quantization only')
        x.copy_(self.maximum)


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true')
    p.add_argument('--output',required=True,type=Path);p.add_argument('--lengths',nargs='+',type=int,default=[513,8192,16384])
    p.add_argument('--checkpoint',default='D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')
    p.add_argument('--lora',default='D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['LOCAL_RANK']) if args.physical else 0
    torch.cuda.set_device(rank);torch.set_grad_enabled(False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.physical:
        assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in (0,1))
        dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=timedelta(seconds=180))
        adaptation=None
    else:
        from swiglu_padding import dependencies
        _,adaptation=dependencies(args.output);isolate_cublas_workspaces()
    pi.int_mm=old.native_int_mm
    path=ROOT/'semantic_results/fc2_quant_memory/baseline/tilelang_swiglu_pad.py'
    spec=importlib.util.spec_from_file_location('frozen_fc2_before',path);before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
    variants=dict(before=before.global_row_quantize,lifetime=fixed.global_row_quantize,owned=owned_rotation)
    sources=['patches/tilelang_swiglu_pad.py','patches/tilelang_row_quant.py','patches/optimized_tp.py','patches/h3_lora.py',
             'benchmarks/fc2_quant_memory.py','benchmarks/fc2_quant_candidates.py','patches/h3_operator_policy.py',str(path.relative_to(ROOT)).replace('\\','/')]
    state=dict(status='running',physical_dual_t4=args.physical,torch=torch.__version__,cuda=torch.version.cuda,
               hardware=torch.cuda.get_device_name(rank),adaptation=adaptation,cases=[],timings=[],
               scope='Real block0 FC1/FC2 projections with seeded inputs; no complete generation claim',
               owned_scope='Explicit opt-in owned_fp32 production quantizer; reduced FP32 GEMM M can change rounding; no universal byte guarantee',
               source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    def persist():
        if rank==0:save(args.output/'results.json',state)
    def gather(value):
        if not args.physical:return value
        out=[None,None];dist.all_gather_object(out,value);return out
    def execute(fn):
        if args.physical:return [fn(0,dist)],None
        comm=LogicalComm();return comm.run([lambda r=r:fn(r,comm) for r in (0,1)]),comm
    def full(fn,block,packed,comm):
        if args.physical and fn is owned_rotation:
            from patches.h3_operator_policy import H3OperatorPolicy,row_linear
            block.fc2.operator_policy=H3OperatorPolicy(fc2_quantizer='owned_fp32')
            out=row_linear(block.fc2,packed,old.row_linear,comm,'swiglu')
        else:
            q,s=fn(packed,comm,2)
            out=fixed._finish_row(q,s,packed.shape,packed.dtype,block.fc2.weight,block.fc2.weight_scale,comm,None,None,None)
            del q,s
        adapter=getattr(block.fc2,'lora_adapter',None)
        return adapter.add_to(packed,out,comm=comm,input_act='swiglu') if adapter is not None else out
    def release():gc.collect();torch.cuda.empty_cache()
    persist()
    try:
        base=CheckpointReader(args.checkpoint)
        for stack,strengths in (('none',()),('multi',(.6,-.2))):
            if strengths:
                base.lora_store=H3LoRAStore(tuple((args.lora,s) for s in strengths),base,row_partial=True,selective_b=True)
            elif hasattr(base,'lora_store'):del base.lora_store
            blocks=[prepare_tp_block(base,'dit',0,r).cuda() for r in ((rank,) if args.physical else (0,1))]
            for rows in args.lengths:
                torch.manual_seed(561);x=torch.randn(rows,5376,device='cuda')*.25
                packed=[block.fc1(x) for block in blocks]
                reference,baseline_comm=execute(lambda r,c:variants['before'](packed[r],c,2))
                expected,_=execute(lambda r,c:full(variants['before'],blocks[r],packed[r],c))
                for name,fn in variants.items():
                    actual,_=execute(lambda r,c:fn(packed[r],c,2))
                    output,_=execute(lambda r,c:full(fn,blocks[r],packed[r],c))
                    scores=[]
                    for r in range(len(blocks)):
                        q,s=actual[r];qr,sr=reference[r]
                        scores.append(dict(q=metrics(q,qr),scale=metrics(s,sr),output=metrics(output[r],expected[r]),
                                           changed_payload=int((q!=qr).sum()),payload_elements=q.numel(),
                                           changed_scale_rows=int((s.view(torch.uint8).reshape(rows,-1)!=sr.view(torch.uint8).reshape(rows,-1)).any(-1).sum())))
                    per_rank=gather(scores[0]) if args.physical else scores
                    state['cases'].append(dict(stack=stack,rows=rows,variant=name,per_rank=per_rank));persist()
                    assert all(m['output']['finite'] for m in per_rank),'Nonfinite output'
                    if name!='owned':assert all(m[k]['bytes_exact'] for m in per_rank for k in ('q','scale','output')),'Lifetime fix changed numerical output'
                    print('FC2_QUANT_GATE',stack,rows,name,'exact',all(m['output']['bytes_exact'] for m in per_rank),flush=True)
                    del actual,output,q,s,qr,sr
                del reference,expected;release()
                if args.physical:
                    from tp2_timing import measure
                    for turn in (0,1):
                        order=list(variants) if turn==0 else list(reversed(variants))
                        for name in order:
                            fn=variants[name]
                            scopes=dict(quant=lambda:fn(packed[0],dist,2),fc2=lambda:full(fn,blocks[0],packed[0],dist),
                                        ffn=lambda:full(fn,blocks[0],blocks[0].fc1(x),dist))
                            for scope,call in scopes.items():
                                release();torch.cuda.synchronize();dist.barrier()
                                # Warm/latency and peak-only passes are separate; no correctness tensors retained.
                                timing=measure(call,warmup=3,repeats=7)
                                torch.cuda.synchronize();start=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                                value=call();torch.cuda.synchronize()
                                memory=dict(entry=start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                                            temporary=torch.cuda.max_memory_allocated()-start);del value
                                state['timings'].append(dict(stack=stack,rows=rows,variant=name,round=turn,scope=scope,
                                                              per_rank=gather(dict(timing=timing,memory=memory))));persist()
                else:
                    for name,fn in variants.items():
                        memories=[]
                        for r in (0,1):
                            comm=FrozenMaximum(r,baseline_comm.maxima[r]);fn(packed[r],comm,2);torch.cuda.synchronize();release()
                            entry=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                            value=fn(packed[r],comm,2);torch.cuda.synchronize()
                            memories.append(dict(entry=entry,temporary=torch.cuda.max_memory_allocated()-entry));del value
                        state['timings'].append(dict(stack=stack,rows=rows,variant=name,scope='local_quant_memory_only',per_rank=memories));persist()
                del x,packed,baseline_comm;release()
            del blocks;release()
        state.update(status='complete',passed=True,
                     lifetime_exact=all(m[k]['bytes_exact'] for c in state['cases'] if c['variant']=='lifetime' for m in c['per_rank'] for k in ('q','scale','output')),
                     owned_exact=all(m[k]['bytes_exact'] for c in state['cases'] if c['variant']=='owned' for m in c['per_rank'] for k in ('q','scale','output')))
        persist()
    except BaseException as error:state.update(status='failed',passed=False,error_type=type(error).__name__);persist();raise
    finally:
        if args.physical:dist.destroy_process_group()


if __name__=='__main__':main()
