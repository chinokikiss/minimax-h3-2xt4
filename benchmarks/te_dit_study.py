"""Measured TE/DiT study. No SP, no latency extrapolation from block measurements."""
import argparse
from contextlib import contextmanager
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import socket
import statistics
import time

import retest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fixture_inputs import block_input

GATE = {'rel_l2_max': 1e-5, 'cosine_min': .999999, 'max_abs_max': .005}
CASES = {'te': [(37,1),(512,1),(512,2),(2048,1)],
         'dit': [(37,1),(1024,1),(1024,2),(4096,1)]}


def inputs(kind, path, length, batch, device):
    curve = retest.state(path, 'adaln_t_table')[''] if kind == 'dit' else None
    # Each DiT request is independent, never joined into one attention sequence.
    data = [block_input(kind, length, 900+i, curve) for i in range(batch)]
    if kind == 'te':
        x = torch.cat([d['input'] for d in data], 0).to(device)
        from comfy.text_encoders.llama import Qwen3VL_32BConfig, precompute_freqs_cis
        from comfy.ldm.modules.attention import optimized_attention_for_device
        c = Qwen3VL_32BConfig()
        rope = precompute_freqs_cis(128, torch.arange(length,device=device)[None], c.rope_theta,
            rope_dims=c.rope_dims, device=device, interleaved_mrope=True)
        mask = torch.full((length,length),torch.finfo(x.dtype).min/4,dtype=x.dtype,device=device).triu(1)
        attention = optimized_attention_for_device(device,mask=True,small_input=True)
        return x, (mask,rope,attention)
    return [d['input'].to(device) for d in data], [(d['t_emb'].to(device),d['segments'],d['rope'].to(device)) for d in data]


def invoke(kind, model, x, args, native):
    if kind == 'te':
        mask,rope,attention = args
        if native:
            return model(x.clone(), attention_mask=mask, freqs_cis=rope, optimized_attention=attention)[0]
        return model(x.clone(),mask,rope)
    return torch.stack([model(v.clone(),*a) for v,a in zip(x,args)])


def check(output, reference):
    score = retest.metrics(output.cpu(), reference)
    ok = retest.passes(score,GATE)
    if dist.is_initialized():
        flag = torch.tensor(int(ok),device=output.device)
        dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        ok = bool(flag.item())
    if not ok:
        raise RuntimeError('Native correctness gate failed: '+json.dumps(score))
    return score


def timing(call, reference, batch, warmups=3, repeats=9):
    pre = check(call(),reference)
    for _ in range(warmups): call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    wall,events=[],[]
    for _ in range(repeats):
        if dist.is_initialized(): dist.barrier()
        torch.cuda.synchronize()
        a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin=time.perf_counter(); a.record(); out=call(); b.record(); torch.cuda.synchronize()
        elapsed=(time.perf_counter()-begin)*1000
        values=torch.tensor([elapsed,a.elapsed_time(b)],device=out.device)
        if dist.is_initialized(): dist.all_reduce(values,op=dist.ReduceOp.MAX)
        wall.append(values[0].item());events.append(values[1].item())
    post=check(out,reference)
    memory=torch.tensor([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved(),resident],device=out.device,dtype=torch.float64)
    peaks=[memory]
    if dist.is_initialized():
        peaks=[torch.empty_like(memory) for _ in range(2)];dist.all_gather(peaks,memory)
    return {'wall_samples_ms':wall,'cuda_event_samples_ms':events,'median_ms':statistics.median(wall),
        'requests_per_second':batch*1000/statistics.median(wall),
        'peak_allocated_bytes':[int(p[0].item()) for p in peaks],
        'peak_reserved_bytes':[int(p[1].item()) for p in peaks],
        'resident_bytes':[int(p[2].item()) for p in peaks],
        'temporary_peak_increment_bytes':[int((p[0]-p[2]).item()) for p in peaks],
        'pre_accuracy':pre,'post_accuracy':post,'correctness_passed':True,
        'timing_boundary':'GPU-resident inputs; input clone included; masks/RoPE prepared outside timer; rank barrier outside timer; max rank synchronized wall',
        'warmups':warmups,'iterations':repeats}


@contextmanager
def instrumentation(model):
    """Instrument only the profiled pass; ordinary latency has no hooks."""
    import patches.parallel_int8 as math
    originals={};handles=[]
    import patches.optimized_tp as optimized
    saved_global=optimized.global_row_quantize
    def traced_global(*args,**kwargs):
        with torch.profiler.record_function('TP::global_row_quantize'):return saved_global(*args,**kwargs)
    optimized.global_row_quantize=traced_global
    for name in ['gather','quantize','int_mm','dequantize']:
        original=getattr(math,name);originals[name]=original
        def wrapped(*args,_f=original,_name=name,**kwargs):
            with torch.profiler.record_function('TP::'+_name):return _f(*args,**kwargs)
        setattr(math,name,wrapped)
    for name,module in model.named_modules():
        if not name:continue
        if any(v in name for v in ['proj','fc1','fc2','norm','mlp','attn']):
            def pre(m,a,_name=name):
                m._study_scope=torch.profiler.record_function('module::'+_name);m._study_scope.__enter__()
            def post(m,a,o):m._study_scope.__exit__(None,None,None)
            handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
    try:yield
    finally:
        for handle in handles:handle.remove()
        for name,value in originals.items():setattr(math,name,value)
        optimized.global_row_quantize=saved_global


def profile(call,model,output,label):
    if dist.is_initialized():dist.barrier()
    torch.cuda.synchronize()
    with instrumentation(model),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=True) as prof:
        with torch.profiler.record_function('study::forward'):call()
        torch.cuda.synchronize()
    trace=output/(label+'_trace.json')
    prof.export_chrome_trace(str(trace))
    raw=json.loads(trace.read_text())
    intervals=[];kernels=defaultdict(lambda:{'count':0,'duration_us':0.0});cpu=defaultdict(lambda:{'count':0,'self_us':0.0})
    for event in prof.events():
        if event.device_type==torch.autograd.DeviceType.CPU and event.name.startswith(('TP::','module::','c10d::','aten::item','cudaStreamSynchronize','cudaDeviceSynchronize')):
            cpu[event.name]['count']+=1;cpu[event.name]['self_us']+=event.self_cpu_time_total
    for event in raw['traceEvents']:
        if event.get('cat')=='kernel' and event.get('ph')=='X':
            intervals.append((event['ts'],event['ts']+event['dur']))
            kernels[event['name']]['count']+=1;kernels[event['name']]['duration_us']+=event['dur']
    union=[]
    for a,b in sorted(intervals):
        if union and a<=union[-1][1]:union[-1][1]=max(b,union[-1][1])
        else:union.append([a,b])
    span=union[-1][1]-union[0][0] if union else 0
    busy=sum(b-a for a,b in union)
    retest.write_json(output/(label+'_profile.json'),{'kernels':dict(kernels),'cpu_scopes':dict(cpu),
        'kernel_count':sum(v['count'] for v in kernels.values()),'kernel_span_ms':span/1000,
        'kernel_busy_union_ms':busy/1000,'between_kernel_idle_ms':(span-busy)/1000,
        'warning':'Profiled pass perturbs execution. Durations may overlap; idle excludes memcpy engines, and NCCL kernel duration is not wire latency.'})
    with gzip.open(str(trace)+'.gz','wb') as stream:stream.write(trace.read_bytes())
    trace.unlink()


def worker(rank,kind,path,output,port,mode):
    os.environ['MASTER_ADDR']='127.0.0.1';os.environ['MASTER_PORT']=str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',rank=rank,world_size=2,device_id=torch.device('cuda',rank))
    try:run(kind,path,Path(output),mode,rank)
    finally:dist.destroy_process_group()


def run(kind,path,output,mode,rank=0):
    output.mkdir(parents=True,exist_ok=True)
    native=mode=='native'
    if mode in ('tp_fixed','tp_scalemax','tp_p2p'):
        from patches.optimized_tp import enable
        enable('fixed' if mode=='tp_fixed' else 'scalemax')
    model=retest.load_native_block(kind,path,device=f'cuda:{rank}') if native else retest.load_proposal(kind,'tp',path,rank)
    result_path=output/(mode+'_results.json')
    rows=json.loads(result_path.read_text()) if os.environ.get('REF2VA_ISOLATED_CASE')=='1' and result_path.exists() else []
    with torch.inference_mode():
        for length,batch in CASES[kind]:
            name=f's{length}_b{batch}';x,args=inputs(kind,path,length,batch,torch.device('cuda',rank))
            call=lambda:invoke(kind,model,x,args,native)
            refpath=output/f'native_{name}.pt'
            if native:torch.save(call().cpu(),refpath)
            ref=torch.load(refpath,map_location='cpu',weights_only=True)
            row=timing(call,ref,batch);row.update(kind=kind,mode=mode,length=length,batch=batch,scope='trained block 0',batch_execution='native TE batch' if kind=='te' else 'independent sequential DiT requests')
            rows.append(row)
            if rank==0:
                retest.write_json(output/(mode+'_results.json'),rows)
                print('MEASURED',kind,mode,name,round(row['median_ms'],3),'exact',row['post_accuracy']['is_bit_exact'],flush=True)
            if batch==1 and length!=37:profile(call,model,output,f'{mode}_{name}_rank{rank}')
            del x,args,ref


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['te','dit'],required=True);parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--modes',nargs='+',default=['native','tp_legacy']);a=parser.parse_args()
    retest.validate_runtime()
    a.suite=a.kind+'_study';a.round=0;a.warmups=3;a.iterations=9;a.allow_non_t4=False
    retest.environment(a.output,a,a.checkpoint)
    for mode in a.modes:
        if mode=='native':run(a.kind,a.checkpoint,a.output,mode);retest.cleanup()
        else:
            os.environ['NCCL_P2P_DISABLE']='0' if mode=='tp_p2p' else '1'
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            mp.spawn(worker,args=(a.kind,a.checkpoint,str(a.output),port,mode),nprocs=2,join=True)


if __name__=='__main__':main()
