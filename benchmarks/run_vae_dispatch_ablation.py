"""Real-latent VAE block replay: strict linear x attention factorial on T4.

No TE, DiT, audio or complete video generation. The two historical Native
tile batch sizes are replayed; private activation values never enter receipts.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata as md
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

CASES=(('eager','sdpa'),('cuda','sdpa'),('eager','kitchen'),('cuda','kitchen'))


@contextmanager
def block_scopes(block):
    """Name both FP16 and INT8 Native call sites in the CUDA profiler."""
    import comfy.ops as ops
    import comfy.ldm.minimax.vae as native
    import torch
    original=ops.linear_input_act
    original_attention=native.optimized_attention
    labels={id(block.attn.to_qkv):'qkv',id(block.attn.to_out):'attention_out',
            id(block.ff.w1):'fc1',id(block.ff.w2):'fc2'}
    def linear(module,*args,**kwargs):
        with torch.profiler.record_function('VAE_LINEAR_SITE/'+labels.get(id(module),'other')):
            return original(module,*args,**kwargs)
    def attention(*args,**kwargs):
        with torch.profiler.record_function('VAE_ATTENTION/native_sdpa'):
            return original_attention(*args,**kwargs)
    ops.linear_input_act=linear;native.optimized_attention=attention
    try:yield
    finally:ops.linear_input_act=original;native.optimized_attention=original_attention


@contextmanager
def dispatch(linear, attention, receipt):
    """Strict explicit linear implementation; no use_backend silent fallback."""
    import comfy_kitchen as ck
    import comfy.quant_ops as quant
    import comfy.ldm.minimax.vae as native
    import torch
    original=quant.ck
    registry=ck.registry
    signature=inspect.signature(ck.int8_linear)
    class Proxy:
        def __getattr__(self,name):return getattr(original,name)
        def int8_linear(self,*args,**kwargs):
            bound=signature.bind(*args,**kwargs);bound.apply_defaults();values=dict(bound.arguments)
            if values['out_dtype'] is None:values['out_dtype']=torch.bfloat16
            implementation=registry.get_implementation('int8_linear',backend=linear,kwargs=values)
            # Explicit backend lookup in the pinned API validates constraints
            # and raises rather than moving on to another backend.
            receipt.setdefault('linear_implementations',{})[implementation.__module__+'.'+implementation.__name__]=linear
            receipt['linear_calls']=receipt.get('linear_calls',0)+1
            with torch.profiler.record_function('VAE_LINEAR/'+linear+'/'+str(values.get('input_act'))):
                return implementation(**values)
        def int8_attention(self,q,k,v,*args,**kwargs):
            receipt['attention_calls']=receipt.get('attention_calls',0)+1
            with torch.profiler.record_function('VAE_ATTENTION/'+attention):
                if attention=='sdpa':
                    return torch.nn.functional.scaled_dot_product_attention(q,k,v)
                return ck.int8_attention(q,k,v,*args,**kwargs)
    quant.ck=Proxy()
    native_flag=native.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE
    # Only the INT8 checkpoint takes this branch. FP16 retains Native SDPA.
    native.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE=True
    try:yield
    finally:
        quant.ck=original;native.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE=native_flag


def measure(call, count=7):
    import torch
    wall=[];gpu=[]
    for _ in range(count):
        torch.cuda.synchronize();begin=time.perf_counter()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();out=call();b.record();b.synchronize()
        wall.append((time.perf_counter()-begin)*1000);gpu.append(a.elapsed_time(b));del out
    return dict(gpu_ms=gpu,wall_ms=wall,gpu_median_ms=statistics.median(gpu),wall_median_ms=statistics.median(wall))


def profile(call, path):
    import torch
    from torch.profiler import profile as capture,ProfilerActivity
    with capture(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as trace:
        with torch.profiler.record_function('VAE_MEASURED_BLOCK'):
            result=call()
        torch.cuda.synchronize()
    trace.export_chrome_trace(str(path))
    events=trace.events()
    kernels=[e for e in events if e.device_type==torch.autograd.DeviceType.CUDA]
    if not kernels:raise RuntimeError('CUPTI trace has no GPU events; no attribution accepted')
    kernel_totals={}
    for event in kernels:
        row=kernel_totals.setdefault(event.name,dict(calls=0,gpu_work_us=0.0))
        row['calls']+=1;row['gpu_work_us']+=event.device_time_total
    scopes=[dict(name=e.key,cpu_us=e.cpu_time_total,gpu_us=e.device_time_total,
                 self_cpu_us=e.self_cpu_time_total,self_gpu_us=e.self_device_time_total)
            for e in trace.key_averages() if e.key.startswith(('VAE_','aten::'))]
    del result
    return dict(cuda_events=len(kernels),scopes=scopes,kernels=kernel_totals,
                note='Nested GPU totals overlap; do not sum inclusive block and child scopes')


def summarize(cases):
    """Counterfactual deltas at identical inputs/shapes within CUDA13."""
    result=[]
    for batch,layer in sorted({(r['batch'],r['layer']) for r in cases}):
        group={(r['linear'],r['attention']):r for r in cases if r['batch']==batch and r['layer']==layer}
        latency=lambda linear,attention:group[linear,attention]['gpu_median_ms']
        eager_sdpa=latency('eager','sdpa');cuda_sdpa=latency('cuda','sdpa')
        eager_kitchen=latency('eager','kitchen');cuda_kitchen=latency('cuda','kitchen')
        result.append(dict(batch=batch,layer=layer,
            linear_cuda_saving_with_sdpa_ms=eager_sdpa-cuda_sdpa,
            linear_cuda_saving_with_kitchen_ms=eager_kitchen-cuda_kitchen,
            attention_kitchen_saving_with_eager_ms=eager_sdpa-eager_kitchen,
            attention_kitchen_saving_with_cuda_ms=cuda_sdpa-cuda_kitchen,
            interaction_ms=(eager_sdpa-cuda_sdpa)-(eager_kitchen-cuda_kitchen),
            cuda_int8_over_fp16_ratio=cuda_kitchen/latency('fp16_native','sdpa'),
            scope='Positive saving favors the changed backend; all other CUDA13 registry dispatch held constant'))
    return result


def device_gate():
    import torch,comfy_kitchen as ck
    devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
                  uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
    assert len(devices)==2 and len({d['uuid'] for d in devices})==2
    assert all(d['capability']==[7,5] and 'T4' in d['name'] for d in devices)
    driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip().splitlines()
    assert driver and all(int(d.split('.')[0])>=580 for d in driver),'CUDA13 driver unsupported'
    assert torch.__version__=='2.11.0+cu130' and torch.version.cuda=='13.0'
    assert 'sm_75' in torch.cuda.get_arch_list()
    from patches.legacy_runtime import validate_packages
    validate_packages()
    assert ck.registry.is_available('cuda'),'Kitchen CUDA backend unavailable; do not bypass frozen guard'
    return dict(devices=devices,driver=driver,torch=torch.__version__,cuda=torch.version.cuda,
                archs=torch.cuda.get_arch_list(),kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'),
                registry=ck.registry.list_backends())


def real_tiles(reader, latents):
    """Use Native normalization, chunk/span and spatial indices, without pixels."""
    import torch
    from patches.staged_vae import video_planning_template
    planner=video_planning_template(reader)
    z=latents.to(dtype=torch.float16)
    z=z*planner.latents_std.view(1,-1,1,1,1).to(z)+planner.latents_mean.view(1,-1,1,1,1).to(z)
    padding,count=planner._decode_temporal_chunks(z.shape[2])
    if padding:z=torch.cat((z,z[:,:,-1:].repeat(1,1,padding,1,1)),2)
    clip=z[:,:,:planner.tokens_chunk_size+planner.token_overlap]
    yi,yl,_=planner.split_tiles(clip.shape[-2]*planner.vae_ratio)
    xi,xl,_=planner.split_tiles(clip.shape[-1]*planner.vae_ratio)
    row=clip[...,yi[0]//planner.vae_ratio:(yi[0]+yl[0])//planner.vae_ratio,:]
    tiles=[row[...,x//planner.vae_ratio:(x+w)//planner.vae_ratio] for x,w in zip(xi,xl)]
    assert len(tiles)==5 and all(list(t.shape)==[1,24,7,16,16] for t in tiles)
    return {4:torch.cat(tiles[:4]),1:tiles[4].clone()}


def run_blocks(args, save, state):
    import torch,comfy_kitchen as ck
    from patches.windowed_tp import CheckpointReader
    from patches.parallel_vae import checkpoint_plan,load_decode_weights
    z=torch.load(args.latents,map_location='cpu',weights_only=True)['video']
    assert hashlib.sha256(z.contiguous().view(torch.uint8).numpy()).hexdigest()=='8fb3296f9dfc00641698e1a1aa26acb7f0fa98675b89b597dc95f10e64a9fa64'
    manifest=json.loads(args.models.read_text());state['cases']=[]
    # Each GPU independently replays one actual Native batch size. This is
    # block attribution, not a two-GPU full decoder throughput measurement.
    captured={}
    for device,batch in ((0,4),(1,1)):
        torch.cuda.set_device(device)
        reader,names,_=checkpoint_plan(manifest['int8_convrot']['local_path'],'video')
        tiles=real_tiles(reader,z)
        model=load_decode_weights(reader,names,'video',torch.device('cuda',device),parallelism='temporal')
        selected=(0,17,35);handles=[];inputs={}
        def hook(index):
            def capture(module,values):
                inputs[index]=tuple(v.detach().clone() if isinstance(v,torch.Tensor) else v for v in values)
            return capture
        for i in selected:handles.append(model.decoder.transformer_blocks[i].register_forward_pre_hook(hook(i)))
        with dispatch('eager','sdpa',{}):
            # Invoke the unchanged Native decoder directly: the production
            # FP32 attention adapter's scope would override this SDPA ablation.
            pixels=model.decoder(model.post_quant_conv(tiles[batch].to(device=device,dtype=torch.float16)))
        assert torch.isfinite(pixels).all();del pixels
        for h in handles:h.remove()
        # Reuse identical captured inputs and unchanged real INT8 weights for
        # every factorial case; capture cost is excluded from warm timings.
        for i in selected:
            block=model.decoder.transformer_blocks[i];values=inputs[i]
            original_hash=hashlib.sha256(values[0].cpu().contiguous().view(torch.uint8).numpy()).hexdigest()
            shift=(i+batch)%len(CASES)
            order=CASES[shift:]+CASES[:shift]
            for linear,attention in order:
                row=dict(device=device,batch=batch,layer=i,linear=linear,attention=attention,
                         input_sha256=original_hash,shape=list(values[0].shape),warmup=3,repetitions=7)
                with block_scopes(block),dispatch(linear,attention,row):
                    call=lambda:block(*values)
                    begin=time.perf_counter();out=call();torch.cuda.synchronize()
                    row['cold_or_cache_transition_ms']=(time.perf_counter()-begin)*1000
                    assert torch.isfinite(out).all();del out
                    for _ in range(2):out=call();del out
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    row.update(measure(call));row['peak_allocated']=torch.cuda.max_memory_allocated()
                    row['trace']=profile(call,args.output/f'gpu{device}_b{batch}_layer{i}_{linear}_{attention}.json')
                if hashlib.sha256(values[0].cpu().contiguous().view(torch.uint8).numpy()).hexdigest()!=original_hash:
                    raise RuntimeError('Block mutated the shared replay input')
                assert row['linear_calls']==44 and row['attention_calls']==11
                state['cases'].append(row);save();print('VAE_ABLATION_CASE',device,batch,i,linear,attention,row['gpu_median_ms'],flush=True)
        captured[device]={i:tuple(v.cpu() if isinstance(v,torch.Tensor) else v for v in values) for i,values in inputs.items()}
        del model,inputs,reader,tiles,values,block,call;gc.collect();torch.cuda.empty_cache()
        # FP16 checkpoint anchor uses the same captured pre-block activations.
        reader,names,_=checkpoint_plan(manifest['fp16']['local_path'],'video')
        model=load_decode_weights(reader,names,'video',torch.device('cuda',device),parallelism='temporal')
        for i in selected:
            block=model.decoder.transformer_blocks[i]
            values=tuple(v.to(device=device) if isinstance(v,torch.Tensor) else v for v in captured[device][i])
            row=dict(device=device,batch=batch,layer=i,linear='fp16_native',attention='sdpa',warmup=3,repetitions=7,
                     shape=list(values[0].shape),anchor='Same captured input; checkpoint differs, not a quality ground truth')
            call=lambda:block(*values)
            with block_scopes(block):
                for _ in range(3):out=call();del out
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                row.update(measure(call));row['peak_allocated']=torch.cuda.max_memory_allocated()
                row['trace']=profile(call,args.output/f'gpu{device}_b{batch}_layer{i}_fp16_sdpa.json')
            state['cases'].append(row);save()
        del model,reader,values,block,call;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-preflight',action='store_true');p.add_argument('--gate-only',action='store_true')
    p.add_argument('--latents',type=Path);p.add_argument('--models',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(args.source),str(args.source/'ComfyUI'),str(args.source/'benchmarks')]
    sys.argv=['vae-ablation','--cpu'] if args.cpu_preflight else ['vae-ablation']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    state=dict(status='starting',full_pipeline=False)
    def save():(args.output/'result.json').write_text(json.dumps(state,indent=2))
    try:
        import torch,comfy_kitchen as ck,comfy.quant_ops,tilelang,torchaudio,torchvision
        from patches.legacy_runtime import validate_packages,validate_source
        validate_packages();validate_source(args.source/'ComfyUI')
        assert torch.__version__=='2.11.0+cu130' and torch.version.cuda=='13.0'
        state['versions']={n:md.version(n) for n in ('torch','torchvision','torchaudio','numpy','comfy-kitchen','tilelang')}
        lock=json.loads((args.source/'vae_cuda13_lock.json').read_text())
        for name,row in lock['packages'].items():
            assert md.version(name)==row['version'],(name,md.version(name),row['version'])
        for name,version in lock['retained_dependencies'].items():
            assert md.version(name)==version,(name,md.version(name),version)
        if args.cpu_preflight:
            import unittest
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('benchmarks.test_vae_cuda13'))
            assert result.wasSuccessful()
            flags=torch._C._cuda_getArchFlags();assert 'sm_75' in flags
            from run_vae_bake_batch import qualify,sha
            plan,latent=qualify(args.source,Path('/kaggle/input'))
            state.update(status='cuda13_api_ready',tests=result.testsRun,compiled_arch_flags=flags,
                         input_sha256=sha(latent),
                         int8_linear_signature=str(inspect.signature(ck.int8_linear)),
                         int8_attention_signature=str(inspect.signature(ck.int8_attention)))
            return
        torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
        state['hardware']=device_gate();save()
        # Verify both devices, strict CUDA INT8 dispatch, attention and CUPTI
        # before any large weight download.
        for device in (0,1):
            torch.cuda.set_device(device)
            x=torch.ones((32,256),device=device,dtype=torch.float16)
            w=torch.ones((256,256),device=device,dtype=torch.int8)
            scale=torch.ones((),device=device,dtype=torch.float32)
            q=torch.randn((1,4,256,64),device=device,dtype=torch.float16)
            row={}
            with dispatch('cuda','kitchen',row):
                import comfy.quant_ops as quant
                out=quant.ck.int8_linear(x,w,scale,out_dtype=torch.float16)
                assert torch.isfinite(out).all() and torch.allclose(out,torch.full_like(out,256),atol=2,rtol=.02)
                value=quant.ck.int8_attention(q,q,q);assert torch.isfinite(value).all()
                state.setdefault('gate_traces',[]).append(profile(lambda:quant.ck.int8_linear(x,w,scale,out_dtype=torch.float16),args.output/f'gate_gpu{device}.json'))
            del x,w,scale,q,out,value
        if args.gate_only:state['status']='physical_cuda13_gate_passed';return
        run_blocks(args,save,state)
        assert len(state['cases'])==30
        state.update(status='complete',counterfactuals=summarize(state['cases']),
            other_ops='Unchanged automatic registry dispatch across all CUDA13 factorial cases',
            limitation='Block replay isolates linear/attention effects; no full-video quality or total decode speedup claim')
    except BaseException as error:
        state.update(status='failed',error=repr(error));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:save()


if __name__=='__main__':main()
