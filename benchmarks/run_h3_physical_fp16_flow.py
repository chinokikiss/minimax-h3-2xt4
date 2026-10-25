"""Trained TE/DiT component gate on physical TP2, without a sampler.

The reference is the existing FP32 activation flow with its normal INT8
operators. BF16 is deliberately not executed on SM75. Inputs reused at another
layer remain stress inputs, not captures of that layer's native trajectory.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
from run_h3_native_fp32_flow import inputs,int8_calls,int8_identity
from run_h3_tensor_flow import observe,stats,compare,save
from patches.h3_tensor_flow import TensorFlowPolicy
from patches.h3_operator_policy import H3OperatorPolicy
from patches.windowed_tp import CheckpointReader,prepare_tp_block


def output_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def replicated(value):
    digest=torch.tensor(list(bytes.fromhex(output_hash(value))),device=value.device,dtype=torch.uint8)
    others=[torch.empty_like(digest) for _ in range(2)]
    dist.all_gather(others,digest)
    return torch.equal(others[0],others[1])


@contextmanager
def dispatch_counts():
    """Forward unchanged calls and expose optimized preparation vs fallback."""
    from contextlib import ExitStack
    from patches import tilelang_row_quant as row,optimized_tp as native
    from patches import tilelang_swiglu_pad as swiglu,tilelang_transport_p2 as transport
    from patches import tilelang_group32_masked as masked
    counts={}
    def wrap(name,function):
        def forward(*args,**kwargs):
            counts[name]=counts.get(name,0)+1
            return function(*args,**kwargs)
        return forward
    with ExitStack() as stack:
        for module,name in ((row,'global_row_quantize'),(native,'global_row_quantize'),
                            (swiglu,'swiglu_pad'),(swiglu,'_quantize_owned'),
                            (transport,'encode'),(transport,'finish'),(masked,'attention')):
            original=getattr(module,name)
            label=module.__name__+'.'+name
            stack.enter_context(patch.object(module,name,wrap(label,original)))
        yield counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['te','dit'])
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--layer',type=int,default=0)
    parser.add_argument('--tokens',type=int,default=513)
    parser.add_argument('--lora',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--flows',nargs='+',choices=['native_fp32_flow','fp16_unscaled','fp16_fold_c16','fp16_fold_c64'])
    parser.add_argument('--lifetime-ablation',action='store_true')
    args=parser.parse_args();sys.argv=sys.argv[:1]
    if args.mode=='dit' and args.fixture is None:parser.error('DiT requires the retained trained input fixture')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability()!=(7,5) or torch.cuda.device_count()!=2:
        raise RuntimeError('Two physical SM75 GPUs are required')
    if 'T4' not in torch.cuda.get_device_name():raise RuntimeError('Actual T4 hardware required')
    if args.repeats<3:raise ValueError('At least three steady-state repetitions required')
    if args.flows and args.flows[0]!='native_fp32_flow':raise ValueError('Native FP32 flow must be the first reference')
    from patches.legacy_runtime import validate_source,validate_packages
    validate_source(ROOT/'ComfyUI');validate_packages()
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from patches.optimized_tp import enable
    enable('scalemax')
    dist.init_process_group('nccl')
    output=args.output.with_name(args.output.stem+f'_rank{rank}.json')
    if output.exists():raise FileExistsError('Never overwrite retained measurements: '+str(output))
    report=dict(schema='physical-native-fp32-flow-v1',status='running',rank=rank,
        scope='Bounded trained component only; physical NCCL, no H3 sampler or full trajectory',
        reference='Existing Native FP32 activation flow, retaining INT8 weights/operators and normal mixed dtype islands',
        hardware=dict(name=torch.cuda.get_device_name(),capability=[7,5],uuid=str(torch.cuda.get_device_properties(rank).uuid)),
        operator_policy=asdict(H3OperatorPolicy(row_linear_backend='tilelang_packed',fc2_quantizer='owned_fp32')),cases=[])
    try:
        x,call,report['distribution']=inputs(args)
        reader=CheckpointReader(args.checkpoint);reader.operator_policy=H3OperatorPolicy(row_linear_backend='tilelang_packed',fc2_quantizer='owned_fp32')
        if args.lora:
            from patches.h3_lora import H3LoRAStore
            reader.lora_store=H3LoRAStore(((str(args.lora),1.),),reader,selective_b=True)
        gold=None;identities=None
        configurations=[('native_fp32_flow',TensorFlowPolicy()),
                          ('fp16_unscaled',TensorFlowPolicy('fp16',0)),
                          ('fp16_fold_c16',TensorFlowPolicy('fp16',4)),
                          ('fp16_fold_c64',TensorFlowPolicy('fp16',6))]
        if args.flows:configurations=[entry for entry in configurations if entry[0] in args.flows]
        for name,flow in configurations:
            reader.tensor_flow=flow
            started=time.perf_counter();model=prepare_tp_block(reader,args.mode,args.layer,rank).cuda()
            torch.cuda.synchronize();loading=time.perf_counter()-started
            current=int8_identity([model])
            if identities is None:identities=current
            if current!=identities:raise AssertionError('INT8 weight bytes changed across flows')
            if any(getattr(m,'_full_precision_mm',False) for m in model.modules()):
                raise AssertionError('Forced dense FP32 is not the reference')
            divisor=flow.divisor if flow.enabled else 1.
            dtype=flow.dtype if flow.enabled else torch.float32
            value=(x/divisor).to(dtype)
            # First call includes lazy compilation; exclude it from timing.
            dist.barrier();started=time.perf_counter()
            with torch.inference_mode():first=call(model,value)
            torch.cuda.synchronize();first_call=time.perf_counter()-started
            del first
            with torch.inference_mode(),observe(model) as ranges,int8_calls() as int8,dispatch_counts() as dispatch:
                actual=call(model,value)
                repeat=call(model,value)
            physical=actual.float()*divisor
            if gold is None:
                if not torch.isfinite(physical).all():raise AssertionError('Native FP32 flow is nonfinite')
                gold=physical.detach().cpu()
            row=dict(name=name,dtype=str(dtype),divisor=divisor,
                loading_seconds=loading,first_call_including_compile_seconds=first_call,
                output=stats(physical),vs_native_fp32_flow=compare(physical.cpu(),gold),
                repeat_equal=torch.equal(actual,repeat),rank_equal=replicated(actual),
                int8_weights_unchanged=True,int8_calls=int8,dispatch=dispatch,ranges=ranges)
            if name=='native_fp32_flow' and (not row['repeat_equal'] or not row['rank_equal']):
                raise AssertionError('Reference is not reproducible across ranks/repetitions')
            if name=='fp16_fold_c64' and row['output']['nonfinite']:
                raise AssertionError('Default FP16-fix overflow on this component input')
            del actual,repeat,physical
            torch.cuda.reset_peak_memory_stats()
            timings=[]
            with torch.inference_mode():
                for _ in range(args.repeats):
                    dist.barrier();torch.cuda.synchronize();started=time.perf_counter()
                    result=call(model,value)
                    torch.cuda.synchronize();timings.append(1000*(time.perf_counter()-started));del result
            row.update(steady_state_ms=dict(samples=timings,median=statistics.median(timings),minimum=min(timings)),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if args.lifetime_ablation and args.mode=='te' and name=='fp16_fold_c64':
                import ast,inspect,textwrap,types
                from patches import tp_qwen3vl as owner
                tree=ast.parse(textwrap.dedent(inspect.getsource(owner.TPTransformerBlockINT8.forward)))
                class WithoutDeletes(ast.NodeTransformer):
                    def visit_Delete(self,node):return None
                namespace=dict(owner.__dict__)
                exec(compile(ast.fix_missing_locations(WithoutDeletes().visit(tree)),'<TE-without-explicit-del>','exec'),namespace)
                with torch.inference_mode():expected=call(model,value).cpu()
                original=model.forward
                try:
                    model.forward=types.MethodType(namespace['forward'],model)
                    torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode():without_del=call(model,value)
                    torch.cuda.synchronize()
                    row['lifetime_ablation']=dict(byte_exact=torch.equal(expected,without_del.cpu()),
                        without_del_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        current_peak_allocated_bytes=row['peak_allocated_bytes'],
                        scope='Same trained block/input/operators; benchmark-only removal of del statements')
                    if not row['lifetime_ablation']['byte_exact']:raise AssertionError('Lifetime deletion changed arithmetic')
                    del without_del
                finally:model.forward=original
                del expected
            if args.profile and name in ('native_fp32_flow','fp16_fold_c64'):
                trace=output.with_name(output.stem+'_'+name+'_trace.json')
                with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                         torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as profiler:
                    result=call(model,value);torch.cuda.synchronize();del result
                profiler.export_chrome_trace(str(trace));row['trace']=trace.name
            report['cases'].append(row);save(output,report)
            del model,value;gc.collect();torch.cuda.empty_cache()
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list((ROOT/'patches').glob('*.py'))+[Path(__file__)]}
        save(output,report)
        dist.destroy_process_group()


if __name__=='__main__':main()
