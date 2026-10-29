"""One isolated two-rank operator/block section; never an H3 sampler."""
import argparse
from contextlib import contextmanager,ExitStack
from datetime import timedelta
import gc
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import statistics
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import torch

COMPILE_REQUESTS=[]


def compare(actual,reference):
    actual=actual.detach().cpu();reference=reference.detach().cpu()
    mismatched=0;delta2=ref2=0.;maximum=0.;finite=True
    for a,b in zip(actual.split(512),reference.split(512)):
        finite=finite and bool(torch.isfinite(a).all())
        mismatched+=int((a!=b).sum())
        difference=a.float()-b.float()
        maximum=max(maximum,float(difference.abs().max()))
        delta2+=float(difference.double().square().sum());ref2+=float(b.double().square().sum())
    return dict(finite=finite,byte_exact=mismatched==0,max_abs=maximum,
                mismatch_fraction=mismatched/actual.numel(),relative_rmse=(delta2/max(ref2,1e-30))**.5)


class Memory:
    def __init__(self):self.rows=[]
    @contextmanager
    def phase(self,name):
        initial=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        with torch.profiler.record_function(name):yield
        self.rows.append(dict(phase=name,before=initial,after=torch.cuda.memory_allocated(),
            peak_allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.memory_reserved(),
            device_free=torch.cuda.mem_get_info()[0]))


def balance(functions,barrier,*,warmups=2,repeats=5):
    """No observers, compilation, loading or error reductions in timed runs."""
    keys=list(functions);result={k:dict(samples=[]) for k in keys}
    for name in keys:
        start=time.monotonic();value=functions[name]();torch.cuda.synchronize();del value
        result[name]['first_call_after_validation_seconds']=time.monotonic()-start
    for _ in range(warmups):
        for name in keys:
            barrier();value=functions[name]();del value
    torch.cuda.synchronize()
    for repeat in range(repeats):
        order=keys[repeat%len(keys):]+keys[:repeat%len(keys)]
        if repeat%2:order=list(reversed(order))
        for name in order:
            barrier();torch.cuda.synchronize()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            compiles_before=len(COMPILE_REQUESTS)
            begin=time.perf_counter();start.record();value=functions[name]();end.record();end.synchronize()
            result[name]['samples'].append(dict(cuda_ms=start.elapsed_time(end),wall_ms=1000*(time.perf_counter()-begin),
                                               compile_requests=len(COMPILE_REQUESTS)-compiles_before))
            del value
    for name in keys:
        barrier();torch.cuda.synchronize();torch.cuda.empty_cache()
        initial=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        value=functions[name]();torch.cuda.synchronize()
        result[name].update(initial_allocated=initial,peak_allocated=torch.cuda.max_memory_allocated(),
            incremental_peak=torch.cuda.max_memory_allocated()-initial,peak_reserved=torch.cuda.max_memory_reserved(),
            device_used_after=torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0],
            median_cuda_ms=statistics.median(s['cuda_ms'] for s in result[name]['samples']),
            median_wall_ms=statistics.median(s['wall_ms'] for s in result[name]['samples']))
        del value
        if any(sample['compile_requests'] for sample in result[name]['samples']):
            raise RuntimeError('Unexpected compilation inside steady-state timing: '+name)
    return result


def make_inputs(rows,device):
    # Deployment geometry and Native modality ordering, with an explicitly
    # generated residual fixture rather than a private activation capture.
    # Native tag order is video=0, text=1, audio=2. Use three
    # timestep classes (video=.5, audio=.75, pinned text/references=1),
    # so the modulation table has nine rows rather than one shared clock.
    segments=[(a,min(b,rows),row) for a,b,row in
              ((0,2392,7),(2392,3892,6),(3892,4702,5),(4702,59134,0)) if a<rows]
    from benchmarks.dit_vram_candidates import row_labels
    labels=row_labels(rows,segments,device,modulation_rows=9)
    x=torch.empty(rows,5376,device=device,dtype=torch.float16).normal_(0.,1./64.)
    angle=torch.arange(64,device=device,dtype=torch.float32)[None,:]*.01+torch.arange(rows,device=device)[:,None]*.0001
    c,s=angle.cos(),angle.sin()
    freqs=torch.stack((c,-s,s,c),-1).reshape(1,rows,1,64,2,2).half()
    return x,segments,labels,freqs


@contextmanager
def count_collectives(dist,record):
    with ExitStack() as stack:
        for name in ('all_reduce','all_gather','all_gather_into_tensor'):
            fn=getattr(dist,name)
            def counted(*args,_fn=fn,_name=name,**kwargs):
                tensors=[a for a in args if isinstance(a,torch.Tensor)]
                record.append(dict(name=_name,op=str(kwargs.get('op')),shapes=[list(a.shape) for a in tensors],dtypes=[str(a.dtype) for a in tensors]))
                return _fn(*args,**kwargs)
            stack.enter_context(patch.object(dist,name,counted))
        yield


def profile(function,path):
    from torch.profiler import profile as capture,ProfilerActivity,record_function
    with capture(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as probe:
        value=torch.ones(32,device='cuda');value.add_(1);torch.cuda.synchronize()
    tiny=path.with_name(path.stem+'_probe.json');probe.export_chrome_trace(str(tiny));del value
    events=json.loads(tiny.read_text())['traceEvents']
    if not any(e.get('cat')=='kernel' for e in events):
        return dict(status='profiler_blocked',reason='Minimal kernel CUDA capture is empty; no attribution claimed')
    with capture(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as captured:
        with record_function('DIT_VRAM_OPERATOR'):value=function()
        torch.cuda.synchronize();del value
    captured.export_chrome_trace(str(path))
    events=json.loads(path.read_text())['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    result=dict(status='complete' if kernels else 'profiler_blocked',kernel_count=len(kernels),
                summed_kernel_ms=sum(e.get('dur',0) for e in kernels)/1000,
                memcpy_count=sum(e.get('cat')=='gpu_memcpy' for e in events),
                launch_api_count=sum('LaunchKernel' in e.get('name','') for e in events),
                limitation='Summed kernel work may overlap; not the critical path')
    for target in (tiny,path):
        target.with_suffix('.json.gz').write_bytes(gzip.compress(target.read_bytes(),mtime=0));target.unlink()
    return result


@contextmanager
def baseline_phases(block,observer):
    import patches.h3_tensor_flow as flow
    import patches.h3_operator_policy as policy
    import patches.tilelang_fp32_rms_rope as rope
    with ExitStack() as stack:
        for target,name,label in [(block.qkv_proj,'forward','qkv'),(block.out_proj,'forward','out_projection'),
                (block.fc1,'forward','fc1'),(block.fc2,'forward','fc2'),
                (flow,'modulated_norm','norm'),(flow,'gated_add','gate'),
                (policy,'dit_attention','attention'),(rope,'execute','rms_rope')]:
            fn=getattr(target,name)
            def wrapped(*args,_fn=fn,_label=label,**kwargs):
                with observer.phase(_label):return _fn(*args,**kwargs)
            stack.enter_context(patch.object(target,name,wrapped))
        yield


def run(section,block,device,barrier,dist,out,state):
    from benchmarks import dit_vram_candidates as candidate,dit_vram_kernels as kernels
    from patches.h3_tensor_flow import modulated_norm,gated_add
    from patches import tilelang_transport_p2 as codec
    rows=59134;x,segments,labels,freqs=make_inputs(rows,device)
    state.update(input_distribution='Seeded FP16 residual fixture through trained base block0, not captured real H3 activations',
                 tokens=rows,hidden=5376,local_heads=28,head_dim=128,local_ffn=7168,
                 segments=segments,full_h3_pipeline=False,warmups=2,repeats=5)
    if section=='norm_gate':
        shift=torch.randn(9,5376,device=device)*.1;scale=torch.randn_like(shift)*.1;gate=torch.randn_like(shift)*.1
        branch=torch.empty_like(x).normal_(0.,.02)
        # Also cover zero/tiny/max-FP16 rows and folded epsilon independently.
        tiny=x[:33].clone();tiny[0].zero_();tiny[1].fill_(1e-7);tiny[2,::128]=65504.
        small_segments=[(0,33,1)];small_labels=candidate.row_labels(33,small_segments,device)
        ref=modulated_norm(block.norm1,tiny,shift,scale,small_segments)
        state['stress_norm_error']=compare(kernels.norm(block.norm1,tiny,shift,scale,small_labels),ref)
        del tiny,ref
        functions={
            'norm_baseline':lambda:modulated_norm(block.norm1,x,shift,scale,segments),
            'norm_fused':lambda:kernels.norm(block.norm1,x,shift,scale,labels),
            'gate_baseline':lambda:gated_add(x.clone(),gate,branch,segments),
            'gate_fused':lambda:kernels.gate(x.clone(),gate,branch,labels)}
        ref=functions['norm_baseline']().cpu();value=functions['norm_fused']()
        state['norm_error']=compare(value,ref);del value,ref
        ref=functions['gate_baseline']().cpu();value=functions['gate_fused']()
        state['gate_error']=compare(value,ref);del value,ref
        assert state['gate_error']['byte_exact'] and state['norm_error']['finite']
        assert state['norm_error']['relative_rmse']<=1e-3 and state['stress_norm_error']['relative_rmse']<=1e-3
        state['measurements']=balance(functions,barrier)
        state['profiles']={name:profile(fn,out/(name+'.json')) for name,fn in functions.items()}
    elif section=='column':
        def call(name):
            source=x.clone()
            if name=='production':return block.fc1(source)
            p=candidate.prepare_column(source);del source
            return candidate.consume_column(p,block.fc1,dict(prepared=0,chunk_4k=4096,chunk_8k=8192,chunk_16k=16384)[name])
        functions={name:(lambda n=name:call(n)) for name in ('production','prepared','chunk_4k','chunk_8k','chunk_16k')}
        reference=functions['production']().cpu();state['errors']={}
        for name,fn in list(functions.items())[1:]:
            value=fn();state['errors'][name]=compare(value,reference);del value
        del reference
        # Timings remain informative even if this device selects a different
        # producer route. Such a case is explicitly unqualified for promotion.
        state['exact_qualified']=all(row['byte_exact'] for row in state['errors'].values())
        state['measurements']=balance(functions,barrier)
        state['profiles']={name:profile(functions[name],out/(name+'.json')) for name in ('production','chunk_8k')}
        q,s=candidate.prepare_column(x).take()
        signed=torch.full((32,5376),-128,device=device,dtype=torch.int8)
        from patches.optimized_tp import native_int_mm
        # Sampled signed-extreme GEMM control; integer products fit INT32.
        w=torch.full((128,5376),-128,device=device,dtype=torch.int8)
        state['signed_extreme_int32_exact']=bool((native_int_mm(signed,w)==5376*128*128).all())
        del q,s,signed,w
        # Explicit copy-stream handoff. Hold pinned source through copy-ready;
        # record_stream protects the GPU allocation after its Python owner dies.
        host=x[:513].cpu().pin_memory();stream=torch.cuda.Stream()
        ready=torch.cuda.Event()
        with torch.cuda.stream(stream):
            transferred=host.to(device,non_blocking=True);ready.record()
        torch.cuda.current_stream().wait_event(ready);transferred.record_stream(torch.cuda.current_stream())
        prepared=candidate.prepare_column(transferred);del transferred
        pressure=torch.empty((513,5376),device=device,dtype=torch.float16);pressure.fill_(7.)
        value=candidate.consume_column(prepared,block.fc1,256)
        expected=candidate.consume_column(candidate.prepare_column(host.to(device)),block.fc1,256)
        state['copy_stream_lifetime']=compare(value,expected)
        assert state['copy_stream_lifetime']['byte_exact']
        torch.cuda.synchronize();del host,pressure,value,expected
    elif section=='row_epilogue':
        from patches.tilelang_transport_tradeoffs import shift_for
        local_k=7168
        payload=torch.empty(rows,5376,device=device,dtype=torch.float16).uniform_(-64,64)
        scale=torch.full((rows,1),.001,device=device)
        weights=block.fc2.weight_scale
        gate=torch.randn(9,5376,device=device)*.1
        def baseline():
            branch=codec.finish(payload,scale,weights,'fp16',local_k).half()
            return gated_add(x.clone(),gate,branch,segments)
        functions=dict(baseline=baseline,fused=lambda:kernels.row_gate(payload,scale,weights,x.clone(),gate,labels,local_k))
        reference=baseline().cpu();value=functions['fused']();state['error']=compare(value,reference);del value,reference
        assert state['error']['byte_exact'],state['error']
        state['measurements']=balance(functions,barrier)
        state['profiles']={name:profile(fn,out/(name+'.json')) for name,fn in functions.items()}
        state['restoration_shift']=shift_for(local_k)
    elif section=='attention':
        import comfy_kitchen as ck
        if not callable(getattr(ck,'prequantize_int8_attention',None)):
            state.update(status='blocked_api',reason='Pinned 0.2.37 wheel lacks split prequantization API; not upgrading');return
        from patches.tilelang_fp32_rms_rope import execute
        from patches.kitchen_attention import int8_attention
        # Weights are real; this large input is generated through their QKV
        # projection, without rerunning the complete model or sampler.
        packed=block.qkv_proj(modulated_norm(block.norm1,x,torch.zeros(9,5376,device=device),torch.zeros(9,5376,device=device),segments))
        template=packed.cpu();del packed
        def call(split):
            qkv=template.to(device);q,k=execute(qkv,freqs,block.q_norm.weight,block.k_norm.weight,epsilon=block.q_norm.eps)
            v=qkv[:,7168:].reshape(rows,28,128)
            qh,kh,vh=(z.transpose(0,1).unsqueeze(0) for z in (q,k,v))
            if split:
                p=ck.prequantize_int8_attention(qh,kh,vh);del qkv,q,k,v,qh,kh,vh
                return ck.int8_attention_from_prequantized(p)
            return int8_attention(qh,kh,vh)
        functions=dict(baseline=lambda:call(False),split=lambda:call(True))
        reference=functions['baseline']().cpu();value=functions['split']();state['error']=compare(value,reference);del value,reference
        assert state['error']['byte_exact'],state['error']
        state['measurements']=balance(functions,barrier,warmups=1,repeats=3)
        state['profiles']={name:profile(fn,out/(name+'.json')) for name,fn in functions.items()}
        state['timing_caveat']='Includes identical CPU template H2D staging in both cases; block timing is GPU-resident'
    elif section=='block':
        times=torch.tensor([.5,.75,1.],device=device)
        time_embedding=times[:,None].pow(torch.arange(8,device=device)[None,:])
        excluded=state.get('excluded_variants',{})
        functions={name:(lambda n=name:candidate.forward(block,x.clone(),time_embedding,segments,freqs,labels,n))
                   for name in candidate.VARIANTS if name not in excluded}
        reference=functions['baseline']().cpu();state['errors']={}
        for name,fn in list(functions.items())[1:]:
            value=fn();state['errors'][name]=compare(value,reference);del value
        del reference
        state['exact_qualified']={name:row['byte_exact'] for name,row in state['errors'].items()}
        assert all(row['finite'] for row in state['errors'].values()),state['errors']
        state['measurements']=balance(functions,barrier,warmups=1,repeats=3)
        state['phase_memory']={};state['collectives']={};state['profiles']={}
        for name in ('baseline','ownership','combined_8k'):
            if name not in functions:continue
            barrier();observer=Memory();collectives=[]
            history=callable(getattr(torch.cuda.memory,'_record_memory_history',None))
            try:
                if history:torch.cuda.memory._record_memory_history(enabled='all',context='all',stacks='python',max_entries=20000)
                with count_collectives(dist,collectives), baseline_phases(block,observer) if name=='baseline' else ExitStack():
                    value=candidate.forward(block,x.clone(),time_embedding,segments,freqs,labels,name,observer=observer)
                    torch.cuda.synchronize();del value
                if history:
                    snapshot=pickle.dumps(torch.cuda.memory._snapshot(),protocol=5)
                    (out/(name+'_allocator_snapshot.pickle.gz')).write_bytes(gzip.compress(snapshot,mtime=0))
                    del snapshot
            finally:
                if history:torch.cuda.memory._record_memory_history(enabled=None)
            state['phase_memory'][name]=observer.rows;state['collectives'][name]=collectives
            barrier();state['profiles'][name]=profile(functions[name],out/(name+'.json'))
    else:raise ValueError(section)
    state['status']='complete'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--section',choices=('norm_gate','column','row_epilogue','attention','block'),required=True)
    parser.add_argument('--eligibility',type=Path)
    args=parser.parse_args();sys.argv=sys.argv[:1]
    sys.path[:0]=[str(args.source),str(args.source/'ComfyUI'),str(args.source/'benchmarks')]
    import comfy.cli_args
    import torch.distributed as dist
    from safetensors.torch import load_file
    from patches.rank_local_int8 import shell,state_hashes
    from patches.h3_tensor_flow import prepare_tp,TensorFlowPolicy
    from patches.h3_operator_policy import H3OperatorPolicy,configure_block
    from patches import optimized_tp
    rank=int(__import__('os').environ['LOCAL_RANK']);device=torch.device('cuda',rank)
    torch.cuda.set_device(device);torch.set_num_threads(2);torch.set_grad_enabled(False)
    torch.manual_seed(20261029)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True);out=args.output/f'rank{rank}';out.mkdir(exist_ok=True)
    state=dict(status='starting',section=args.section,rank=rank,hardware=dict(torch=str(torch.__version__),
        cuda=torch.version.cuda,name=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device))))
    if args.eligibility:state['excluded_variants']=json.loads(args.eligibility.read_text())
    import tilelang
    original_compile=tilelang.compile
    def tracked_compile(*a,**kw):
        caller=sys._getframe(1).f_globals.get('__name__');start=time.monotonic()
        try:return original_compile(*a,**kw)
        finally:COMPILE_REQUESTS.append(dict(module=caller,seconds=time.monotonic()-start,
                                            scope='Compile request or persistent cache load, not necessarily a cold compiler invocation'))
    tilelang.compile=tracked_compile
    try:
        from patches.legacy_runtime import validate_packages,validate_source
        validate_packages();validate_source(args.source/'ComfyUI')
        if torch.cuda.get_device_capability(device)!=(7,5) or 'T4' not in state['hardware']['name']:
            raise RuntimeError('Actual T4 SM75 required')
        dist.init_process_group('nccl',timeout=timedelta(seconds=240))
        cpu_group=dist.new_group(backend='gloo',timeout=timedelta(seconds=240))
        barrier=lambda:dist.barrier(group=cpu_group)
        weights=load_file(str(args.weights/f'dit.block00.rank{rank}.safetensors'))
        block=shell('dit',rank);block.load_state_dict(weights,assign=True)
        proof=json.loads((args.source/'bundle_provenance'/f'dit.block00.rank{rank}.json').read_text())
        assert state_hashes(block)==proof['state_sha256'],'Native trained shard tensor proof mismatch'
        prepare_tp(block,'dit',TensorFlowPolicy(mode='fp16'));configure_block(block,'dit',H3OperatorPolicy())
        block.to(device);block.eval();del weights
        optimized_tp.enable()
        state.update(resident_weight_bytes=sum(t.numel()*t.element_size() for t in block.state_dict().values()),
            trained_base='cnmds/minimax-h3-tp2-int8-561a6211@d3e6a4b131056d2c626a01acf6ba48123a885783',
            offline_lora=False,ptd_bake=False,weight_residency='One fixed resident TP2 block; no prefetch/window extrapolation')
        run(args.section,block,device,barrier,dist,out,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error));(out/'exception.txt').write_text(traceback.format_exc());raise
    finally:
        tilelang.compile=original_compile
        state['compile_requests']=COMPILE_REQUESTS
        (out/'result.json').write_text(json.dumps(state,indent=2));print('DIT_VRAM_RESULT',rank,args.section,state['status'],flush=True)
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
