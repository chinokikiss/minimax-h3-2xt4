"""One isolated physical-T4 operator probe. Never reuse a failed CUDA context."""
import argparse
from contextlib import contextmanager
import importlib.metadata as md
import inspect
import json
from pathlib import Path
import statistics
import sys
import time
import traceback


def cases():
    rows=[]
    for batch in (1,4):
        for label,n,k,act in (('qkv',6144,2048,'rms_norm'),('out',2048,2048,None),
                             ('fc1',16384,2048,'rms_norm'),('fc2',2048,8192,'swiglu')):
            rows.append(dict(kind='linear',label=f'vae_b{batch}_{label}',m=batch*1797,n=n,k=k,act=act))
    # Include the known sensitive Turing row-count band as isolated probes.
    for m in (1,2,96,97,127,128,129,257):
        rows.append(dict(kind='linear',label=f'te_m{m}',m=m,n=512,k=5120,act=None))
    for m in (97,127):
        for n in (4096,12800):
            rows.append(dict(kind='linear',label=f'te_m{m}_n{n}',m=m,n=n,k=5120,act=None))
    rows.extend((dict(kind='rope',label='te_rope_gqa',layout='te',length=2392),
                 dict(kind='rope',label='vae_rope_partial',layout='vae',length=1797),
                 dict(kind='rope',label='dit_rope_fp32',layout='dit',length=59134)))
    for width in (512,4096,12800):
        rows.append(dict(kind='dequant',label=f'te_dequant_{width}',m=2392,n=width))
    for channels in (128,1024):
        rows.append(dict(kind='normpad',label=f'encoder_normpad_c{channels}',channels=channels))
    rows.append(dict(kind='conv',label='encoder_conv_optin',channels=128))
    return rows


@contextmanager
def extension_calls(backend,rows):
    original=backend._C
    class Proxy:
        def __getattr__(self,name):
            value=getattr(original,name)
            if not callable(value) or isinstance(value,type):return value
            def call(*args,**kwargs):
                rows[name]=rows.get(name,0)+1
                result=value(*args,**kwargs)
                if isinstance(result,bool):rows[name+'/returned_'+str(result)]=rows.get(name+'/returned_'+str(result),0)+1
                return result
            return call
    backend._C=Proxy()
    try:yield
    finally:backend._C=original


def timed(function):
    import torch
    for _ in range(2):value=function();del value
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();value=function();b.record();b.synchronize();samples.append(a.elapsed_time(b));del value
    return dict(gpu_ms=samples,median_ms=statistics.median(samples),peak_allocated=torch.cuda.max_memory_allocated())


def error(output,reference):
    import torch
    if isinstance(output,(tuple,list)):
        return [error(a,b) for a,b in zip(output,reference)]
    delta=output.float()-reference.float()
    return dict(finite=bool(torch.isfinite(output).all()),byte_exact=bool(torch.equal(output,reference)),
        max_abs=float(delta.abs().max()),relative_rmse=float(delta.square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-12)))


def profile(function,path):
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as capture:
        with torch.profiler.record_function('CU128_EXPLICIT_CANDIDATE'):value=function()
        torch.cuda.synchronize()
    capture.export_chrome_trace(str(path));del value
    events=json.loads(path.read_text())['traceEvents']
    kernels=[e for e in events if e.get('cat')=='kernel']
    if not kernels:raise RuntimeError('No actual GPU kernel events')
    return dict(kernel_count=len(kernels),kernel_work_ms=sum(e['dur'] for e in kernels)/1000,
                names=sorted({e['name'] for e in kernels}))


def execute(case,out):
    import torch,comfy_kitchen as ck
    from patches.kitchen_cu128_candidate import Candidate
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261028)
    before_cuda=ck.registry.list_backends().get('cuda')
    state=dict(case=case,extension_calls={},cuda_registry_before=before_cuda)
    backend=ck.backends.cuda
    def rand(*shape):return torch.randn(*shape,device='cuda',dtype=torch.float16)*.2
    def rotations(*shape):
        angle=torch.randn(shape,device='cuda',dtype=torch.float32)*.3
        c,s=angle.cos(),angle.sin()
        return torch.stack((c,-s,s,c),-1).reshape(*shape,2,2)
    kind=case['kind']
    if kind=='linear':
        m,n,k=case['m'],case['n'],case['k'];act=case['act']
        x=rand(m,k*(2 if act=='swiglu' else 1));w=torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8)
        scale=torch.full((n,),.001,device='cuda',dtype=torch.float32)
        kwargs=dict(out_dtype=torch.float16,convrot=True,convrot_groupsize=256,input_act=act,
                    input_act_weight=rand(k).add_(1) if act=='rms_norm' else None,input_act_eps=1e-5)
        candidate=Candidate(('int8_linear',));fn=lambda:candidate.call('int8_linear',x,w,scale,**kwargs)
        baseline=ck.registry.get_implementation('int8_linear',backend='eager')
        ref=lambda:baseline(x,w,scale,**kwargs)
    elif kind=='rope':
        length=case['length'];layout=case['layout']
        if layout=='te':
            q=rand(1,length,32,128).transpose(1,2);k=rand(1,length,4,128).transpose(1,2)
            matrix=rotations(1,1,length,64)
            name='apply_rope_split_half';args=(q,k,matrix);kwargs={}
        elif layout=='vae':
            packed=rand(1,length,32,192);q,k,_=packed.chunk(3,-1)
            matrix=rotations(1,length,1,24)
            name='rms_rope_split_half_';args=(q,k,matrix,torch.ones(64,device='cuda'));kwargs=dict(epsilon=1e-5,rot_dim=48)
        else:
            q=rand(1,length,28,128).float();k=rand(1,length,28,128).float()
            matrix=rotations(1,length,1,64)
            name='rms_rope_split_half';args=(q,k,matrix,torch.ones(128,device='cuda'));kwargs=dict(epsilon=1e-6,rot_dim=128)
        candidate=Candidate((name,))
        signature=inspect.signature(getattr(ck,name));bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        values=dict(bound.arguments)
        # Diagnose the user's proposed global enable only inside this isolated
        # process. Restore it before the explicit-alias call and all timings.
        disabled=ck.registry.list_backends()['cuda']['disabled']
        try:
            ck.registry.enable('cuda')
            selected=ck.registry.get_implementation(name,kwargs=values)
            state['after_global_enable_implementation']=selected.__module__+'.'+selected.__name__
        finally:
            if disabled:ck.registry.disable('cuda')
        if layout=='dit':
            try:ck.registry.get_implementation(name,backend='h3_cu128_candidate_cuda',kwargs=values)
            except Exception as exc:
                state.update(status='unsupported_dtype_confirmed',reason=str(exc),global_cuda_unchanged=candidate.unchanged_global_cuda())
                return state
            raise AssertionError('Unexpected FP32 CUDA RMS-RoPE support; audit before changing tensor flow')
        # In-place input must be reset outside timed intervals. Use functional
        # RMS-RoPE for latency/error; separately verify the in-place API once.
        if layout=='vae':
            name='rms_rope_split_half';candidate=Candidate((name,'rms_rope_split_half_'))
        fn=lambda:candidate.call(name,*args,**kwargs)
        ref_impl=ck.registry.get_implementation(name,backend='eager');ref=lambda:ref_impl(*args,**kwargs)
    elif kind=='dequant':
        from patches.dit_column_linear import _dequantize_eager
        from patches.tilelang_column_epilogue import dequantize
        m,n=case['m'],case['n'];acc=torch.randint(-2**28,2**28,(m,n),device='cuda',dtype=torch.int32)
        rows=torch.rand((m,1),device='cuda')*.0001;weights=torch.rand((n,),device='cuda')*.0001
        ref=lambda:_dequantize_eager(acc,rows,weights,torch.float16)
        fn=lambda:dequantize(acc,rows,weights,torch.float16);candidate=None
    elif kind=='normpad':
        c=case['channels'];x=rand(1,c,17,32,32);weight=rand(c).add_(1);bias=rand(c)
        args=(x,weight,bias,32,1e-6,[1,1,1,1,2],True)
        candidate=Candidate(('group_norm_silu_pad3d',));fn=lambda:candidate.call('group_norm_silu_pad3d',*args)
        eager=ck.registry.get_implementation('group_norm_silu_pad3d',backend='eager');ref=lambda:eager(*args)
    else:
        c=case['channels'];x=rand(1,c,7,16,16).contiguous(memory_format=torch.channels_last_3d)
        weight=rand(c,c,3,3,3);args=(x,weight,None,None,[1,1,1])
        candidate=Candidate(('fp16_conv3d',));fn=lambda:candidate.call('fp16_conv3d',*args)
        ref=lambda:torch.nn.functional.conv3d(x,weight,stride=1)
        state['native_fp16_accumulation_optin']=bool(getattr(torch.backends.cuda.matmul,'allow_fp16_accumulation',False))
    reference=ref()
    with extension_calls(backend,state['extension_calls']):
        if kind=='rope' and case['layout']=='vae':
            inplace=packed.clone();iq,ik,_=inplace.chunk(3,-1)
            value=candidate.call('rms_rope_split_half_',iq,ik,matrix,args[3],**kwargs)
            state['inplace_error_vs_functional_eager']=error(value,reference)
            del value,inplace,iq,ik
        value=fn();torch.cuda.synchronize();state['error_vs_eager']=error(value,reference);del value
        state['candidate']=timed(fn);state['trace']=profile(fn,out/'candidate_trace.json')
    state['baseline']=timed(ref)
    state['speedup']=state['baseline']['median_ms']/state['candidate']['median_ms']
    if kind=='dequant':assert state['error_vs_eager']['byte_exact'],'Eager multiplication boundaries changed'
    def finite(row):return all(finite(r) for r in row) if isinstance(row,list) else row['finite']
    assert finite(state['error_vs_eager']),'Nonfinite output'
    state['global_cuda_unchanged']=candidate.unchanged_global_cuda() if candidate else True
    state['cuda_registry_after']=ck.registry.list_backends().get('cuda')
    state['global_cuda_unchanged'] &= state['cuda_registry_after']==before_cuda
    assert state['global_cuda_unchanged']
    state['status']='executed_not_quality_qualified'
    # Only library paths, not the full process environment or activation values.
    state['loaded_cuda_libraries']=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if any(n in line for n in ('libcublas','libcudart','libcuda.so','backends/cuda/_C'))})
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--device',type=int,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);sys.path[:0]=[str(args.source),str(args.source/'ComfyUI'),str(args.source/'benchmarks')]
    sys.argv=['cu128-case']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import torch
    torch.cuda.set_device(args.device)
    import comfy.quant_ops
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(args.source/'ComfyUI')
    state={}
    try:state=execute(json.loads(args.case.read_text()),args.output)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:(args.output/'result.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
