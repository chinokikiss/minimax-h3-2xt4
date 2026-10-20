"""Actual full-model latency and independent-request pipeline throughput."""
import argparse
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
from full_models import load_full,request,forward,PipelineRunner
from te_dit_study import GATE,check

CASES={'te':[(37,1),(512,1),(512,2),(1024,1)],'dit':[(512,1),(512,2),(1024,1)]}


def requests(kind,length,count,device,factory=request):return [factory(kind,length,1700+i,device) for i in range(count)]


def call_requests(kind,model,data,te_batch=False):
    if kind=='te' and te_batch:
        ids=torch.cat([d[0][0] for d in data],0)
        return forward(kind,model,((ids,),{'dtype':torch.float16}))
    return torch.stack([forward(kind,model,d) for d in data])


def sync(devices):
    for d in devices:torch.cuda.synchronize(d)


def measured(call,ref,devices,count,repeats=5,runner=None):
    pre=check(call(),ref)
    call();sync(devices)
    resident=[torch.cuda.memory_allocated(d) for d in devices]
    for d in devices:torch.cuda.reset_peak_memory_stats(d)
    times=[];events=[];stages=[]
    for _ in range(repeats):
        if dist.is_initialized():dist.barrier()
        sync(devices)
        event_a=torch.cuda.Event(enable_timing=True);event_b=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();event_a.record();out=call();event_b.record();sync(devices)
        elapsed=(time.perf_counter()-start)*1000
        value=torch.tensor([elapsed,event_a.elapsed_time(event_b)],device=devices[0],dtype=torch.float64)
        if dist.is_initialized():dist.all_reduce(value,op=dist.ReduceOp.MAX)
        times.append(value[0].item());events.append(value[1].item())
        if runner is not None:stages.append(runner.breakdown())
    post=check(out,ref)
    memory=[[torch.cuda.max_memory_allocated(d),torch.cuda.max_memory_reserved(d),resident[i]] for i,d in enumerate(devices)]
    if dist.is_initialized():
        local=torch.tensor(memory[0],device=devices[0],dtype=torch.float64)
        ranks=[torch.empty_like(local) for _ in range(2)];dist.all_gather(ranks,local);memory=[v.tolist() for v in ranks]
    record={'wall_samples_ms':times,'cuda_event_samples_ms':events,**retest.summary(times),
        'requests_per_second':count*1000/statistics.median(times),'requests':count,
        'peak_allocated_bytes':[int(v[0]) for v in memory],'peak_reserved_bytes':[int(v[1]) for v in memory],
        'resident_bytes':[int(v[2]) for v in memory],'temporary_peak_increment_bytes':[int(v[0]-v[2]) for v in memory],
        'pre_accuracy':pre,'post_accuracy':post,'correctness_passed':True,
        'boundary':'GPU resident input to complete TE hidden states / both native DiT latent velocities. Embedding/refiner/packing/AdaLN/heads included; output CPU copy excluded; fill/drain included.',
        'warmups':1,'pipeline_events':stages,
        'cuda_event_note':'Current rank/device event interval; synchronized wall is authoritative for total multi-device latency.'}
    return record


def trace(call,output,label):
    if dist.is_initialized():dist.barrier()
    for d in range(torch.cuda.device_count()):torch.cuda.synchronize(d)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False) as prof:
        with torch.profiler.record_function('full::'+label):call()
        for d in range(torch.cuda.device_count()):torch.cuda.synchronize(d)
    path=output/(label+'_trace.json');prof.export_chrome_trace(str(path))
    kernels={}
    for event in json.loads(path.read_text())['traceEvents']:
        if event.get('cat')=='kernel' and event.get('ph')=='X':
            v=kernels.setdefault(event['name'],{'count':0,'duration_us':0});v['count']+=1;v['duration_us']+=event['dur']
    retest.write_json(output/(label+'_profile.json'),{'kernels':kernels,'note':'Profiled pass only; CPU worker-thread scopes may be missing from Kineto. Per-stage CUDA events and all CUDA kernels remain separate evidence.'})
    with gzip.open(str(path)+'.gz','wb') as out:out.write(path.read_bytes())
    path.unlink()


def balance(kind,model,data,output):
    blocks=model.layers if kind=='te' else model.blocks
    events=[];handles=[]
    for index,block in enumerate(blocks):
        def pre(m,args,_index=index):
            with torch.cuda.device(m.device):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();m._balance_events=(a,b);events.append((_index,a,b))
        def post(m,args,out):
            with torch.cuda.device(m.device):m._balance_events[1].record()
        handles.extend([block.register_forward_pre_hook(pre),block.register_forward_hook(post)])
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(0):begin.record()
    forward(kind,model,data)
    with torch.cuda.device(1):end.record()
    sync([0,1])
    for handle in handles:handle.remove()
    values=[a.elapsed_time(b) for _,a,b in events]
    prefix=begin.elapsed_time(events[0][1]);suffix=events[-1][2].elapsed_time(end)
    choices=[]
    # TE's embedding table and DiT's FP32 refiners live on stage 0. Keep
    # activation headroom on 16 GB T4s when considering a new split.
    for split in range(22,26) if kind=='te' else range(22,27):
        s0=prefix+sum(values[:split]);s1=suffix+sum(values[split:])
        choices.append({'split':split,'stage0_compute_ms':s0,'stage1_compute_ms':s1,'max_stage_ms':max(s0,s1)})
    best=min(choices,key=lambda c:c['max_stage_ms'])
    retest.write_json(output/'balance.json',{'per_layer_cuda_ms':values,'prefix_cuda_ms':prefix,'suffix_cuda_ms':suffix,
        'candidates':choices,'selected':best,'note':'CUDA-event diagnostic pass; selects balanced compute including preprocessing/heads. Boundary transfer and bubbles remain in actual PP timing.'})
    return best['split']


def run(kind,path,output,mode,rank=0,variant='tp_scalemax',cases=None,request_factory=request,cohort_length=512,cohort_counts=(4,8),offload_dit_aux=False,latency_repeats=5,validate_placement=True):
    if mode=='tp_optimized':
        from patches.optimized_tp import enable
        enable('fixed' if variant=='tp_fixed' else 'scalemax')
    devices=[torch.device('cuda',rank)] if mode.startswith('tp') else [torch.device('cuda',i) for i in range(2)]
    split=json.loads((output/'balance.json').read_text())['selected']['split'] if mode in ('pp_overlap','pp_serial_balanced') else 25
    runner=PipelineRunner(kind,split=split) if mode=='pp_overlap' else None
    model=load_full(kind,path,'tp' if mode.startswith('tp') else 'placed',rank=rank,runner=runner,split=split,offload_dit_aux=offload_dit_aux)
    if runner:runner.model=model
    result_path=output/(mode+'_results.json')
    rows=json.loads(result_path.read_text()) if cases is not None and result_path.exists() else []
    with torch.inference_mode():
        # A full single-GPU streamed original forward validates layer placement.
        # Disk/load/offload time is never used as a resident performance baseline.
        if mode=='native' and validate_placement:
            small=37 if kind=='te' else 68
            data=requests(kind,small,1,devices[0],request_factory);placed=call_requests(kind,model,data).cpu()
            # Free placed weights so the reference fits on a single T4.
            del model;retest.cleanup()
            streamed=load_full(kind,path,'streamed')
            reference=call_requests(kind,streamed,data).cpu();check(placed,reference)
            retest.write_json(output/'placement_correctness.json',retest.metrics(placed,reference))
            del streamed,placed,reference,data;retest.cleanup()
            model=load_full(kind,path,'placed',offload_dit_aux=offload_dit_aux)
        for length,count in cases if cases is not None else CASES[kind]:
            name=f's{length}_b{count}';data=requests(kind,length,count,devices[0],request_factory)
            te_batch=kind=='te'
            if runner and count>1:
                # Compare independent microrequests against the same independent
                # native execution; this is separate from native TE batch 2.
                name+='_micro1';te_batch=False
            call=(lambda:runner(data)[0] if kind=='te' and count==1 else runner(data)) if runner else (lambda:call_requests(kind,model,data,te_batch))
            refpath=output/('reference_'+name+'.pt')
            if mode=='native':torch.save(call().cpu(),refpath)
            if runner and count>1:
                ref=torch.stack([torch.load(output/f'request_s{length}_{i}.pt',weights_only=True) for i in range(count)])
            else:ref=torch.load(refpath,map_location='cpu',weights_only=True)
            row=measured(call,ref,devices,count,repeats=latency_repeats,runner=runner);row.update(kind=kind,mode=mode,length=length,batch=count,scope='full text-only TE / complete audiovisual DiT denoiser',purpose='latency',batch_execution='TE native batch' if te_batch else 'independent requests',auxiliary_weight_staging=offload_dit_aux)
            rows.append(row)
            if mode=='native' and length==cohort_length:
                if count==1:
                    saved=torch.load(refpath,map_location='cpu',weights_only=True)
                    torch.save(saved if kind=='te' else saved[0],output/f'request_s{length}_0.pt')
                    del saved
                else:
                    for i,d in enumerate(data):torch.save(forward(kind,model,d).cpu(),output/f'request_s{length}_{i}.pt')
            if rank==0:
                retest.write_json(output/(mode+'_results.json'),rows)
                print('FULL',kind,mode,name,round(row['median_ms'],3),row['post_accuracy']['is_bit_exact'],flush=True)
            if length==cohort_length and count==1:
                trace(call,output,f'{mode}_{name}_rank{rank}')
                if mode=='native':balance(kind,model,data[0],output)
            del data,ref
        # Fair throughput cohort: every strategy executes the same M batch-1
        # independent requests, with all fill/drain and scheduling cost timed.
        for count in cohort_counts:
            length=cohort_length;name=f's{length}_m{count}_micro1';data=requests(kind,length,count,devices[0],request_factory)
            call=(lambda:runner(data)) if runner else (lambda:call_requests(kind,model,data))
            refpath=output/('reference_'+name+'.pt')
            if mode=='native':torch.save(call().cpu(),refpath)
            ref=torch.load(refpath,map_location='cpu',weights_only=True)
            row=measured(call,ref,devices,count,repeats=3,runner=runner)
            row.update(kind=kind,mode=mode,length=length,batch=count,scope='full model independent request cohort',purpose='throughput',batch_execution='M independent batch-1 requests, microbatch size 1')
            rows.append(row)
            if rank==0:
                retest.write_json(output/(mode+'_results.json'),rows)
                print('THROUGHPUT',kind,mode,count,round(row['median_ms'],3),row['post_accuracy']['is_bit_exact'],flush=True)
            if count==cohort_counts[0]:trace(call,output,f'{mode}_{name}_rank{rank}')
            del data,ref
    del model,runner;retest.cleanup()


def worker(rank,kind,path,output,port,mode,variant):
    os.environ['MASTER_ADDR']='127.0.0.1';os.environ['MASTER_PORT']=str(port);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',rank=rank,world_size=2,device_id=torch.device('cuda',rank))
    try:run(kind,path,Path(output),mode,rank,variant)
    finally:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['te','dit'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--mode',choices=['native','tp_legacy','tp_optimized','pp_overlap','pp_serial_balanced'],required=True);p.add_argument('--tp-variant',choices=['tp_fixed','tp_scalemax','tp_p2p'],default='tp_scalemax');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);retest.validate_runtime()
    a.suite=a.kind+'_full';a.round=0;a.warmups=1;a.iterations=5;a.allow_non_t4=False
    retest.environment(a.output,a,a.checkpoint)
    if a.mode.startswith('tp'):
        os.environ['NCCL_P2P_DISABLE']='0' if a.tp_variant=='tp_p2p' and a.mode=='tp_optimized' else '1'
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        mp.spawn(worker,args=(a.kind,a.checkpoint,str(a.output),port,a.mode,a.tp_variant),nprocs=2,join=True)
    else:run(a.kind,a.checkpoint,a.output,a.mode)


if __name__=='__main__':main()
