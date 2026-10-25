"""Physical T4 controlled FP16/LoRA stage probes; no production arithmetic edits."""
import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import sys
import types
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
from patches.h3_tensor_flow import TensorFlowPolicy
from patches.h3_operator_policy import H3OperatorPolicy
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from run_h3_native_fp32_flow import inputs
from run_h3_tensor_flow import compare,stats


def physical_captures(captured,divisor):
    # Hooks run under inference_mode; never mutate their tensors after leaving it.
    return {name:tensor*divisor if name in ('out_proj.output','fc2.output') else tensor
            for name,tensor in captured.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();sys.argv=sys.argv[:1]
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    assert torch.cuda.device_count()==2 and torch.cuda.get_device_capability()==(7,5)
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(ROOT/'ComfyUI')
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from patches.optimized_tp import enable
    enable('scalemax');dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    from types import SimpleNamespace
    x,call,provenance=inputs(SimpleNamespace(mode='dit',fixture=args.assets/'first_block.pt',tokens=513,layer=49))
    result=dict(status='running',rank=rank,provenance=provenance,
        scope='Reused first-block input at layer 49; stress probe only. FP32 row islands are benchmark ablations, not a deployed fix.',cases=[])
    output=args.output.with_name(args.output.stem+f'_rank{rank}.json');gold={}
    try:
        for lora,mode in [(False,'native'),(False,'fp16'),(True,'native'),(True,'fp16'),
                          (True,'fp16_wide_fc2'),(True,'fp16_wide_rows')]:
            reader=CheckpointReader(args.assets/'dit_blocks.safetensors');reader.operator_policy=H3OperatorPolicy(row_linear_backend='tilelang_packed',fc2_quantizer='owned_fp32')
            if lora:
                from patches.h3_lora import H3LoRAStore
                reader.lora_store=H3LoRAStore(((str(args.assets/'lora_blocks.safetensors'),1.),),reader,selective_b=True)
            flow=TensorFlowPolicy() if mode=='native' else TensorFlowPolicy('fp16',6)
            reader.tensor_flow=flow;model=prepare_tp_block(reader,'dit',49,rank).cuda().eval().requires_grad_(False)
            divisor=1. if mode=='native' else 64.;dtype=torch.float32 if mode=='native' else torch.float16
            value=(x/divisor).to(dtype)
            for name in (('fc2',) if mode=='fp16_wide_fc2' else ('out_proj','fc2') if mode=='fp16_wide_rows' else ()):
                module=getattr(model,name);original=module.forward
                def wide(self,inp,*a,_original=original,**kw):return _original(inp.float(),*a,**kw).to(inp.dtype)
                module.forward=types.MethodType(wide,module)
            captured={};handles=[];activation={}
            def before(name):
                def hook(module,inp):
                    captured[name+'.input']=inp[0].detach().float().cpu()
                    if name=='fc2':
                        gate,up=inp[0].float().chunk(2,-1)
                        physical=torch.nn.functional.silu(gate)*up
                        activation.update(fp32_product=stats(physical),fp16_product=stats(physical.half()))
                return hook
            def after(name):
                def hook(module,inp,out):captured[name+'.output']=out.detach().float().cpu()
                return hook
            with torch.inference_mode():call(model,value);torch.cuda.synchronize()
            for name in ('adaln_proj','qkv_proj','out_proj','fc1','fc2'):
                module=getattr(model,name)
                handles.extend((module.register_forward_pre_hook(before(name)),module.register_forward_hook(after(name))))
            try:
                with torch.inference_mode():actual=call(model,value).float().cpu()*divisor
            finally:
                for handle in handles:handle.remove()
            captured=physical_captures(captured,divisor)
            if mode=='native':gold[lora]=(actual.clone(),captured)
            reference,stages=gold[lora]
            result['cases'].append(dict(lora=lora,mode=mode,output=stats(actual),vs_native_fp32=compare(actual,reference),
                stages={name:dict(range=stats(tensor),vs_native_fp32=compare(tensor,stages[name])) for name,tensor in captured.items()},
                swiglu_probe=activation))
            output.write_text(json.dumps(result,indent=2));del model,value,actual,captured
            gc.collect();torch.cuda.empty_cache()
        result['status']='completed_diagnostic_not_quality_acceptance'
    except BaseException as error:result.update(status='failed',error=repr(error));raise
    finally:output.write_text(json.dumps(result,indent=2));dist.destroy_process_group()


if __name__=='__main__':main()
