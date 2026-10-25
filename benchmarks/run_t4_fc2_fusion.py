"""TP2 FP16 residual control, fused FP32 regular-ConvRot quantization, SUM codecs."""
import argparse
from dataclasses import asdict
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
from h3_fp16_fc2_experiment import install,prepare,row_forward
from run_h3_native_fp32_flow import inputs
from run_h3_tensor_flow import compare,stats
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.h3_tensor_flow import TensorFlowPolicy
from patches.h3_operator_policy import H3OperatorPolicy


def measure(function,repeats=7):
    dist.barrier();torch.cuda.synchronize();start=time.perf_counter();value=function();torch.cuda.synchronize()
    first=1000*(time.perf_counter()-start);del value
    for _ in range(2):value=function();del value
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();samples=[]
    for _ in range(repeats):
        dist.barrier();torch.cuda.synchronize();start=time.perf_counter();value=function();torch.cuda.synchronize()
        samples.append(1000*(time.perf_counter()-start));del value
    return dict(first_timed_call_ms=first,samples_ms=samples,median_ms=statistics.median(samples),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args();sys.argv=sys.argv[:1]
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    assert torch.cuda.device_count()==2 and torch.cuda.get_device_capability()==(7,5)
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(ROOT/'ComfyUI')
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from patches.optimized_tp import enable
    enable('scalemax');dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    output=args.output.with_name(args.output.stem+f'_rank{rank}.json');output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',rank=rank,scope='Trained DiT components only, no sampler; current TP2 geometry; sparse quality unchanged',
        hardware=dict(name=torch.cuda.get_device_name(),capability=[7,5],uuid=str(torch.cuda.get_device_properties(rank).uuid)),
        cases=[])
    def save():output.write_text(json.dumps(result,indent=2))
    try:
        for kind,layer,tokens,lora in (('te',0,353,False),('dit',0,6814,False),('dit',49,513,True)):
            x,call,provenance=inputs(SimpleNamespace(mode=kind,fixture=args.assets/'first_block.pt',tokens=tokens,layer=layer))
            reader=CheckpointReader(args.assets/('te_block0.safetensors' if kind=='te' else 'dit_blocks.safetensors'));reader.operator_policy=H3OperatorPolicy(row_linear_backend='tilelang_packed',fc2_quantizer='owned_fp32')
            if lora:
                from patches.h3_lora import H3LoRAStore
                reader.lora_store=H3LoRAStore(((str(args.assets/'lora_blocks.safetensors'),1.),),reader,selective_b=True)
            reader.tensor_flow=TensorFlowPolicy()
            native=prepare_tp_block(reader,kind,layer,rank).cuda().eval().requires_grad_(False)
            with torch.inference_mode():gold=call(native,x).float().cpu()
            del native;gc.collect();torch.cuda.empty_cache()
            reader.tensor_flow=TensorFlowPolicy('fp16',6)
            model=prepare_tp_block(reader,kind,layer,rank).cuda().eval().requires_grad_(False)
            module=model.down_proj if kind=='te' else model.fc2
            value=(x/64.).half();captured=[]
            install(module,dist,'control','int32',capture=lambda value:captured.append(value.detach().clone()))
            with torch.inference_mode():control_block=call(model,value).float().cpu()*64.
            packed=captured.pop();del captured
            install(module,dist,'control','int32')
            with torch.inference_mode():control_row=row_forward(module,packed,dist).float().cpu()
            base_q,base_scale=prepare(packed,dist,'control')
            for mode in ('control','fused_stored','fused_recompute'):
                for transport in ('int32','scaled_fp16'):
                    entry=dict(component=kind,layer=layer,tokens=tokens,lora=lora,preparation=mode,transport=transport,provenance=provenance,
                        residual_dtype='torch.float16',prep_math='FP32',gemm='original INT8 -> INT32',
                        sum_dtype='INT32' if transport=='int32' else 'power-of-two scaled FP16',epilogue_and_lora='FP32 before FP16 store')
                    try:
                        started=time.perf_counter();q,scale=prepare(packed,dist,mode);torch.cuda.synchronize()
                        entry['preparation_before_timing_seconds_including_first_compile']=time.perf_counter()-started
                        entry['quantization']=dict(code_disagreements=int((q!=base_q).sum()),elements=q.numel(),
                            scale_vs_control=compare(scale,base_scale),all_finite=bool(torch.isfinite(scale).all()))
                        from patches import parallel_int8 as pi
                        partial=pi.int_mm(q,module.weight)
                        gathered=[torch.empty_like(partial) for _ in range(2)];dist.all_gather(gathered,partial)
                        if transport=='int32':acc=gathered[0]+gathered[1]
                        else:
                            from tp2_tradeoffs import transport_reference
                            acc=transport_reference(gathered,'fp16',q.shape[1])
                        expected=pi.dequantize(acc,scale,module.weight_scale,torch.float32).reshape(*packed.shape[:-1],module.weight.shape[0])
                        adapter=getattr(module,'lora_adapter',None)
                        if adapter is not None:expected=adapter.add_to(packed.float(),expected,comm=dist,input_act='swiglu')
                        expected=expected.half().cpu();del gathered,partial,acc,q,scale
                        with torch.inference_mode():actual=row_forward(module,packed,dist,mode,transport).cpu()
                        entry.update(independent_codec_epilogue_exact=torch.equal(actual,expected),
                            row_vs_control=compare(actual,control_row),finite=bool(torch.isfinite(actual).all()))
                        if not entry['independent_codec_epilogue_exact'] or not entry['finite']:raise AssertionError('Independent row arithmetic mismatch/nonfinite')
                        del actual,expected
                        entry['row_complete_api']=measure(lambda:row_forward(module,packed,dist,mode,transport),args.repeats)
                        entry['preparation_complete_api']=measure(lambda:prepare(packed,dist,mode),args.repeats)
                        install(module,dist,mode,transport)
                        with torch.inference_mode():actual=call(model,value).float().cpu()*64.
                        entry.update(block_vs_native_fp32=compare(actual,gold),block_vs_fp16_control=compare(actual,control_block),
                            block_output=stats(actual));del actual
                        entry['block_complete_api']=measure(lambda:call(model,value),args.repeats)
                        if layer==49:
                            trace=output.with_name(output.stem+'_'+mode+'_'+transport+'_trace.json')
                            with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as profiler:
                                actual=row_forward(module,packed,dist,mode,transport);torch.cuda.synchronize();del actual
                            profiler.export_chrome_trace(str(trace));entry['trace']=trace.name
                        entry['status']='completed_operator_diagnostic'
                    except Exception as error:
                        entry.update(status='failed',error=repr(error));result['cases'].append(entry);save()
                        # Collective-order recovery after a failed candidate is
                        # unsafe. The next independent process handles Video.
                        raise
                    result['cases'].append(entry);save()
            del model,module,packed,value,x,base_q,base_scale,gold,control_block,control_row;gc.collect();torch.cuda.empty_cache()
        result['status']='completed_not_generation_acceptance'
    except BaseException as error:result.update(status='failed',error=repr(error));raise
    finally:save();dist.destroy_process_group()


if __name__=='__main__':main()
