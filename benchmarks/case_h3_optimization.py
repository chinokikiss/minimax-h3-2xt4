"""Isolated RMS-RoPE or all-selected-operator AOT rehearsal on physical SM75."""
import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import time
import traceback


def rope(case,out,state):
    import torch
    from patches.tilelang_fp32_rms_rope import execute
    from comfy_kitchen.backends.eager.rope import rms_rope_split_half
    from case_cu128_dispatch import timed,error,profile
    s=case['rows'];dtype=getattr(torch,case.get('dtype','float16'));rot=case.get('rot',128)
    output_dtype=dtype;torch.manual_seed(20261028)
    packed=torch.randn(s,10752,device='cuda',dtype=dtype)*.2
    if case.get('stress'):
        packed[0].zero_();packed[1].fill_(1e-7);packed[2,::128]=65504.
    angle=torch.randn(1,s,1,rot//2,device='cuda')*.3;c=angle.cos();v=angle.sin()
    freqs=torch.stack((c,-v,v,c),-1).reshape(1,s,1,rot//2,2,2)
    qg=torch.rand(128,device='cuda')+.5;kg=torch.rand(128,device='cuda')+.5
    def reference():
        q,k,_=packed.split(3584,-1)
        q,k=rms_rope_split_half(q.view(1,s,28,128).float(),k.view(1,s,28,128).float(),
            freqs,qg,kg,epsilon=1e-6,rot_dim=rot)
        return q[0].to(output_dtype),k[0].to(output_dtype)
    fn=lambda:execute(packed,freqs,qg,kg,out_dtype=output_dtype)
    begin=time.monotonic();value=fn();torch.cuda.synchronize();state['cold_candidate_seconds']=time.monotonic()-begin
    ref=reference();state['error']=error(value,ref);del value,ref
    state['baseline']=timed(reference);state['candidate']=timed(fn)
    state['trace']=profile(fn,out/'candidate_trace.json')
    state['speedup']=state['baseline']['median_ms']/state['candidate']['median_ms']
    budget=5e-6 if dtype==torch.float32 else 1e-3
    assert all(e['finite'] and e['relative_rmse']<=budget for e in state['error']),state['error']
    assert state['trace']['kernel_count']==1
    state.update(status='passed_operator_budget',precision='FP32 RMS/gains/rotation; final cast matches reference storage')


def execute_plan(calls):
    import torch
    for module,name,args in calls:
        fn=getattr(import_module('patches.'+module),name)(*args);rows=args[0]
        if module=='tilelang_column_epilogue':
            width=args[1];inputs=(torch.zeros(rows,width,device='cuda',dtype=torch.int32),torch.ones(rows,1,device='cuda'),torch.ones(width,device='cuda'))
        elif module=='tilelang_row_quant':
            if name=='packing_kernel':inputs=(torch.zeros(rows,args[1],device='cuda'),)
            elif name=='maximum_kernel':inputs=(torch.zeros(rows,args[2],device='cuda'),)
            else:inputs=(torch.zeros(rows,args[2],device='cuda'),torch.ones(rows,1,device='cuda'))
        elif module=='tilelang_swiglu_quant':inputs=(torch.zeros(rows,2*args[1],device='cuda',dtype=torch.float16),)
        elif module=='tilelang_transport_tradeoffs':inputs=(torch.zeros(rows,args[1],device='cuda',dtype=torch.int32),)
        elif module=='tilelang_transport_p2':
            inputs=(torch.zeros(rows,args[1],device='cuda',dtype=torch.float16),torch.ones(rows,1,device='cuda'),torch.ones(args[1],device='cuda'))
        elif module=='tilelang_fp32_rms_rope':
            inputs=(torch.zeros(rows,10752,device='cuda',dtype=torch.float16),torch.zeros(rows,64,2,2,device='cuda'),torch.ones(128,device='cuda'),torch.ones(128,device='cuda'))
        else:raise ValueError(module)
        output=fn(*inputs);torch.cuda.synchronize();del output,inputs


def aot(case,out,state):
    import torch,tilelang
    from patches.h3_operator_aot import prepare,plan
    from patches.h3_operator_policy import H3OperatorPolicy
    from case_cu128_dispatch import profile
    policy=H3OperatorPolicy(dit_rms_rope='tilelang_fp32');stage=case['stage'];rows=case['rows'];rank=case['rank']
    counts=[0];original=tilelang.compile
    def compile(*a,**kw):counts[0]+=1;return original(*a,**kw)
    tilelang.compile=compile
    try:
        start=time.monotonic();state['preparation']=prepare(rows,rank,policy,stage)
        state['preparation_seconds']=time.monotonic()-start;state['preparation_compile_calls']=counts[0]
        calls=plan(rows,rank,policy,stage);counts[0]=0
        start=time.monotonic();execute_plan(calls);state['first_execution_seconds']=time.monotonic()-start
        state['first_execution_compile_calls']=counts[0]
        start=time.monotonic();execute_plan(calls);state['second_execution_seconds']=time.monotonic()-start
        state['second_execution_compile_calls']=counts[0]
        state['trace']=profile(lambda:execute_plan(calls),out/'prepared_trace.json')
        assert counts[0]==0 and state['preparation']['complete_default_custom_coverage']
        state.update(status='prepared_execution_has_zero_compiles',
            scope='All custom operators selected by the default stage; shape rehearsal, not denoiser latency',
            full_pipeline=False)
    finally:tilelang.compile=original


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);case=json.loads(args.case.read_text());state=dict(case=case,status='starting')
    sys.path[:0]=[str(args.source),str(args.source/'ComfyUI'),str(args.source/'benchmarks')];sys.argv=['h3-operator-case']
    cache=Path('/kaggle/temp')/('h3_aot_cache_'+case['label'])
    if case['kind']=='aot':os.environ['TILELANG_CACHE_DIR']=str(cache)
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.cli_args
        import torch
        torch.cuda.set_device(args.device);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
        assert torch.cuda.get_device_capability()==(7,5) and torch.__version__=='2.11.0+cu128'
        from patches.legacy_runtime import validate_packages,validate_source
        validate_packages();validate_source(args.source/'ComfyUI')
        (aot if case['kind']=='aot' else rope)(case,args.output,state)
    except BaseException as exc:
        state.update(status='failed',error_message=repr(exc));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:
        if case['kind']=='aot' and cache.is_dir():
            import shutil
            shutil.make_archive(str(args.output/'compiled_cache'),'gztar',root_dir=cache)
            state['compiled_cache_archive']='compiled_cache.tar.gz'
        (args.output/'result.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
