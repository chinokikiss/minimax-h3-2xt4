"""Physical current-production rows/blocks: gather vs FP32 low-rank SUM."""
import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
from patches.h3_lora import H3LoRAStore,RowPartialLoRA
from patches.h3_operator_policy import H3OperatorPolicy,row_linear
from patches.h3_tensor_flow import TensorFlowPolicy
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from run_h3_native_fp32_flow import inputs
from run_h3_tensor_flow import compare,stats


def paired(functions):
    result={name:dict(samples_ms=[],peak_increment_bytes=[],resident_before_bytes=[]) for name in functions}
    for name,fn in functions.items():
        dist.barrier();torch.cuda.synchronize();started=time.perf_counter();value=fn();torch.cuda.synchronize();del value
        result[name]['first_call_seconds']=time.perf_counter()-started
        for _ in range(2):value=fn();del value
    for round_index in range(2):
        order=list(functions) if round_index==0 else list(reversed(functions))
        for _ in range(5):
            for name in order:
                dist.barrier();torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter();value=functions[name]();torch.cuda.synchronize()
                result[name]['samples_ms'].append(1000*(time.perf_counter()-started));del value
                result[name]['resident_before_bytes'].append(before)
                result[name]['peak_increment_bytes'].append(torch.cuda.max_memory_allocated()-before)
    return result


def independent_lora(module,packed,activation,mode):
    """Hold the validated base FP32 output fixed; independent A/SUM/B reference."""
    from patches import optimized_tp
    base=row_linear(module,packed,optimized_tp.row_linear,dist,activation)
    source=packed.float().reshape(-1,packed.shape[-1])
    if activation=='swiglu':g,u=source.chunk(2,-1);source=torch.nn.functional.silu(g)*u
    adapter=module.lora_adapter
    if mode=='gather':
        parts=[torch.empty_like(source) for _ in range(2)];dist.all_gather(parts,source)
        source=torch.cat(parts,-1)
    for term in adapter.terms:
        hidden=torch.nn.functional.linear(source,term.down)
        if mode=='partial':
            parts=[torch.empty_like(hidden) for _ in range(2)];dist.all_gather(parts,hidden)
            hidden=parts[0]+parts[1]
        delta=torch.nn.functional.linear(hidden,term.up).mul_(term.scale)
        base.reshape(-1,base.shape[-1]).add_(delta)
    return base.to(packed.dtype).cpu()


def profile_row(module,packed,activation,mode,path):
    adapter=module.lora_adapter;original=adapter.add_to;events=[]
    class Comm:
        ReduceOp=dist.ReduceOp
        def get_world_size(self):return 2
        def all_reduce(self,value,op):
            events.append(dict(op='all_reduce',dtype=str(value.dtype),shape=list(value.shape),bytes=value.numel()*value.element_size()))
            return dist.all_reduce(value,op=op)
        def all_gather_into_tensor(self,out,value):
            events.append(dict(op='all_gather_into_tensor',dtype=str(value.dtype),shape=list(value.shape),bytes=value.numel()*value.element_size()))
            return dist.all_gather_into_tensor(out,value)
    def observe(x,out,comm=None,input_act=None):
        with torch.profiler.record_function('LoRA/'+mode):return original(x,out,Comm(),input_act)
    adapter.add_to=observe
    try:
        dist.barrier()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as profiler:
            value=module(packed,input_act=activation);torch.cuda.synchronize();del value
        profiler.export_chrome_trace(str(path))
    finally:adapter.add_to=original
    if mode=='partial':
        assert len(events)==len(adapter.terms) and all(e['op']=='all_reduce' and e['dtype']=='torch.float32' for e in events)
    else:assert len(events)==1 and events[0]['op']=='all_gather_into_tensor'
    return events


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--trace-only',action='store_true')
    args=parser.parse_args();sys.argv=sys.argv[:1]
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    assert torch.cuda.device_count()==2 and torch.cuda.get_device_capability()==(7,5) and 'T4' in torch.cuda.get_device_name()
    from patches.legacy_runtime import validate_packages,validate_source
    from patches.optimized_tp import enable
    validate_packages();validate_source(ROOT/'ComfyUI');enable('scalemax')
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=150))
    output=args.output.with_name(args.output.stem+f'_rank{rank}.json');output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',rank=rank,records=[],scope='Current production FP16-fix + fused_stored + scaled FP16 base SUM, FP32 LoRA SUM; no sampler',
        hardware=dict(name=torch.cuda.get_device_name(),uuid=str(torch.cuda.get_device_properties(rank).uuid),capability=[7,5]))
    def save():output.write_text(json.dumps(report,indent=2))
    try:
        configurations=[(0,6814,(.7,-.25))] if args.trace_only else [(layer,tokens,stack) for layer,tokens in ((0,6814),(49,513)) for stack in ((1.,),(.7,-.25))]
        for layer,tokens,strengths in configurations:
            x,call,provenance=inputs(SimpleNamespace(mode='dit',fixture=args.assets/'first_block.pt',tokens=tokens,layer=layer))
            reader=CheckpointReader(args.assets/'dit_blocks.safetensors')
            reader.operator_policy=H3OperatorPolicy(row_linear_backend='tilelang_packed',fc2_quantizer='owned_fp32')
            reader.tensor_flow=TensorFlowPolicy()
            reader.lora_store=H3LoRAStore(tuple((str(args.assets/'lora_blocks.safetensors'),s) for s in strengths),reader,selective_b=True,row_partial=False)
            gold=None
            if not args.trace_only:
                native=prepare_tp_block(reader,'dit',layer,rank).cuda().eval().requires_grad_(False)
                gold=call(native,x).float().cpu();del native;gc.collect();torch.cuda.empty_cache()
            reader.operator_policy=H3OperatorPolicy();reader.tensor_flow=TensorFlowPolicy('fp16',6)
            model=prepare_tp_block(reader,'dit',layer,rank).cuda().eval().requires_grad_(False)
            modules={n:getattr(model,n) for n in ('out_proj','fc2')};adapters={};captured={};handles=[]
            for name,module in modules.items():
                original=module.lora_adapter;assert original is not None and original.columns is not None
                lo,hi=original.columns
                partial=RowPartialLoRA([(t.down[:,lo:hi],t.up,t.scale) for t in original.terms],original.columns).cuda()
                partial.fp32_compute=True;adapters[name]=dict(gather=original,partial=partial)
                def capture(module,inp,name=name):captured[name]=inp[0].detach().clone()
                handles.append(module.register_forward_pre_hook(capture))
            value=(x/64.).half();control_block=call(model,value).float().cpu()*64.
            for handle in handles:handle.remove()
            def select(mode):
                for name,module in modules.items():module.lora_adapter=adapters[name][mode]
            metadata=dict(layer=layer,tokens=tokens,strengths=strengths,provenance=provenance,
                preparation='fused_stored',base_transport='scaled_fp16',lora_dtype='float32',residual_dtype='float16')
            for name,module in modules.items():
                packed=captured[name];act='swiglu' if name=='fc2' else None;controls={}
                for mode in ('gather','partial'):
                    select(mode)
                    if not args.trace_only:
                        expected=independent_lora(module,packed,act,mode)
                        actual=module(packed,input_act=act).cpu()
                        assert torch.equal(expected,actual) and torch.isfinite(actual).all(),(name,mode,'independent LoRA reference')
                        controls[mode]=actual;del expected
                if args.trace_only:
                    for mode in ('gather','partial'):
                        select(mode)
                        for _ in range(2):warm=module(packed,input_act=act);del warm
                        for _ in range(3):
                            with torch.cuda.nvtx.range('ProductionRow/'+name+'/'+mode):
                                warm=module(packed,input_act=act);torch.cuda.synchronize();del warm
                    continue
                def run(mode):
                    select(mode);return module(packed,input_act=act)
                timing=paired({m:lambda m=m:run(m) for m in ('gather','partial')})
                for mode in ('gather','partial'):
                    select(mode);trace=output.with_name(output.stem+f'_L{layer}_{len(strengths)}stack_{name}_{mode}_trace.json')
                    events=profile_row(module,packed,act,mode,trace) if tokens==6814 else None
                    report['records'].append(dict(**metadata,kind='row',projection=name,mode=mode,timing=timing[mode],
                        independent_lora_exact=True,finite=True,vs_gather=compare(controls[mode],controls['gather']),
                        lora_collectives=events,trace=trace.name if events is not None else None))
                    save()
                del controls
            if not args.trace_only:
                def block(mode):select(mode);return call(model,value)
                timing=paired({m:lambda m=m:block(m) for m in ('gather','partial')})
                for mode in ('gather','partial'):
                    actual=block(mode).float().cpu()*64.
                    report['records'].append(dict(**metadata,kind='block',projection='whole_block',mode=mode,timing=timing[mode],
                        output=stats(actual),vs_gather=compare(actual,control_block),vs_native_fp32=compare(actual,gold)))
                    assert torch.isfinite(actual).all();save();del actual
            del model,modules,adapters,captured,value,x,gold,control_block,reader,module,original,partial,packed
            gc.collect();torch.cuda.empty_cache()
        report['status']='completed_not_generation_acceptance'
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:save();dist.destroy_process_group()


if __name__=='__main__':main()
