"""Trace actual VAE collectives and GEMM kernels; no microbenchmark extrapolation."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from retest import checkpoint, cleanup, digest_tensor, environment, load_video_native, metrics, write_json, validate_runtime

def trace(model, latent, output, label, rank=None):
    tokens=[]
    handle=model.decoder.transformer_blocks[0].register_forward_pre_hook(lambda module,args: tokens.append(list(args[0].shape)))
    with torch.inference_mode():
        model.decode(latent)
        torch.cuda.synchronize(latent.device)
        tokens.clear()
        if rank is not None:
            dist.barrier()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            start=time.perf_counter()
            out=model.decode(latent)
            torch.cuda.synchronize(latent.device)
            wall_ms=(time.perf_counter()-start)*1000
    handle.remove()
    suffix=f'{label}_rank{rank}' if rank is not None else label
    prof.export_chrome_trace(str(output/f'{suffix}_trace.json'))
    kernels=defaultdict(lambda:{'count':0,'duration_us':0.0})
    collective_calls=defaultdict(int)
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CPU and event.name.startswith('c10d::'):
            collective_calls[event.name]+=1
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name]['count']+=1
            kernels[event.name]['duration_us']+=event.device_time_total
    nccl={name:v for name,v in kernels.items() if 'nccl' in name.lower()}
    record={'profiled_wall_ms':wall_ms,'warning':'profiling perturbs timing; kernel duration sums may overlap and are not wire latency',
            'first_block_input_shapes':tokens,'input_sha256':digest_tensor(latent),
            'kernels':dict(kernels),'nccl_kernels':nccl,'collective_cpu_calls':dict(collective_calls),
            'nccl_kernel_count':sum(v['count'] for v in nccl.values()),
            'nccl_kernel_duration_sum_ms':sum(v['duration_us'] for v in nccl.values())/1000}
    write_json(output/f'{suffix}_profile.json',record)
    if rank in (None,0):
        torch.save(out.cpu(),output/f'{label}_output.pt')
        print('PROFILE',suffix,json.dumps({k:record[k] for k in ['profiled_wall_ms','first_block_input_shapes','nccl_kernel_count','nccl_kernel_duration_sum_ms']}),flush=True)

def worker(rank,mode,path,output,port):
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']=str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',rank=rank,world_size=2,device_id=torch.device(f'cuda:{rank}'))
    from patches.tp_video_vae import patch_video_vae_tp,patch_video_vae_sp
    try:
        model,sd=load_video_native(path,f'cuda:{rank}')
        model=(patch_video_vae_tp if mode=='tp' else patch_video_vae_sp)(model,sd,rank,2,torch.device(f'cuda:{rank}'))
        del sd
        model.requires_grad_(False)
        latent=torch.load(Path(output)/'input.pt',weights_only=True).to(f'cuda:{rank}')
        trace(model,latent,Path(output),mode,rank)
    finally:
        dist.destroy_process_group()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    args.suite='video_profile';args.round=0;args.warmups=1;args.iterations=1;args.allow_non_t4=False
    validate_runtime()
    path=checkpoint('video')
    environment(args.output,args,path)
    latent=torch.randn((1,24,2,48,84),generator=torch.Generator().manual_seed(46)).half()
    torch.save(latent,args.output/'input.pt')
    model,sd=load_video_native(path,'cuda:0')
    del sd
    trace(model,latent.cuda(),args.output,'native')
    del model
    cleanup()
    for mode in ['tp','sp']:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        mp.spawn(worker,args=(mode,path,str(args.output),port),nprocs=2,join=True)
    ref=torch.load(args.output/'native_output.pt',weights_only=True)
    values={mode:metrics(torch.load(args.output/f'{mode}_output.pt',weights_only=True),ref) for mode in ['tp','sp']}
    write_json(args.output/'accuracy.json',values)
    print('PROFILE_ACCURACY',json.dumps(values),flush=True)

if __name__=='__main__':
    main()
