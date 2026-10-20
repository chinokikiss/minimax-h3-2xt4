"""Physical T4 entry point for the prescribed TE/DiT token matrix."""
import argparse
import json
import os
from pathlib import Path
import socket
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import retest
import te_dit_study as blocks
import full_study as full
import ref2va_workloads as workloads

BASE_INPUTS=blocks.inputs
BASE_TIMING=blocks.timing


def inputs(kind,path,length,batch,device):
    if kind=='te':return BASE_INPUTS(kind,path,length,batch,device)
    rows=[torch.load(Path(os.environ['REF2VA_INPUTS'])/f's{length}_{i}.pt',map_location='cpu',weights_only=True) for i in range(batch)]
    return [r['input'].to(device) for r in rows],[(r['args'][0].to(device),r['args'][1],r['args'][2].to(device)) for r in rows]


def timing(call,reference,batch,warmups=3,repeats=9):
    # Correctness comes first. Keep short tests warm long enough for clocks to
    # settle, without adding dozens of warmups to expensive long sequences.
    blocks.check(call(),reference)
    if dist.is_initialized():dist.barrier()
    begin=time.monotonic();n=0
    while True:
        call();torch.cuda.synchronize();n+=1
        elapsed=torch.tensor(time.monotonic()-begin,device='cuda')
        if dist.is_initialized():dist.all_reduce(elapsed,op=dist.ReduceOp.MIN)
        if n>=3 and elapsed.item()>=.5:break
    row=BASE_TIMING(call,reference,batch,warmups=0,repeats=repeats)
    row.update(warmups=n,minimum_warmup_seconds=.5)
    return row


def configure():
    blocks.CASES={'te':[(s,1) for s in workloads.TE_LENGTHS]+[(512,2)],
                  'dit':[(s,1) for s in workloads.DIT_LENGTHS]+[(8192,2)]}
    blocks.inputs=inputs;blocks.timing=timing


def worker(rank,options,port):
    os.environ['MASTER_ADDR']='127.0.0.1';os.environ['MASTER_PORT']=str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',rank=rank,world_size=2,device_id=torch.device('cuda',rank))
    try:execute(options,rank)
    finally:dist.destroy_process_group()


def execute(options,rank=0):
    configure()
    if options.scope=='block':
        if options.length:
            if options.length not in (workloads.TE_LENGTHS if options.kind=='te' else workloads.DIT_LENGTHS):raise ValueError('Outside prescribed matrix')
            blocks.CASES[options.kind]=[(options.length,options.batch)]
            os.environ['REF2VA_ISOLATED_CASE']='1'
        return blocks.run(options.kind,options.checkpoint,options.output,options.mode,rank)
    cases=[] if options.throughput_only else ([(options.length,1)] if options.length else [(s,1) for s in workloads.TE_LENGTHS])
    cohort_length=8192 if options.kind=='dit' else 512
    cohort_counts=(4,) if options.kind=='dit' else (4,8)
    if options.length and options.length!=cohort_length:cohort_counts=()
    if options.latency_only:cohort_counts=()
    full.run(options.kind,options.checkpoint,options.output,options.mode,rank,
             cases=cases,request_factory=workloads.request,cohort_length=cohort_length,
             cohort_counts=cohort_counts,offload_dit_aux=options.kind=='dit',latency_repeats=3,
             validate_placement=not options.reuse_placement)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scope',choices=['prepare','block','full'],required=True)
    p.add_argument('--kind',choices=['te','dit'],required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--inputs',type=Path)
    p.add_argument('--mode',default='native',choices=['native','tp_legacy','tp_scalemax','tp_optimized','pp_overlap','pp_serial_balanced'])
    p.add_argument('--length',type=int)
    p.add_argument('--batch',type=int,choices=[1,2],default=1)
    phase=p.add_mutually_exclusive_group()
    phase.add_argument('--latency-only',action='store_true')
    phase.add_argument('--throughput-only',action='store_true')
    p.add_argument('--reuse-placement',action='store_true')
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    retest.validate_runtime()
    a.suite=a.kind+'_ref2va';a.round=0;a.warmups=3;a.iterations=9 if a.scope=='block' else 3;a.allow_non_t4=False
    retest.environment(a.output,a,a.checkpoint)
    if a.inputs:os.environ['REF2VA_INPUTS']=str(a.inputs)
    if a.scope=='prepare':
        a.inputs.mkdir(exist_ok=True,parents=True)
        for length in workloads.DIT_LENGTHS:
            for i in range(2 if length==8192 else 1):
                torch.save(workloads.capture_inputs(a.checkpoint,length,900+i),a.inputs/f's{length}_{i}.pt')
                print('CAPTURED native Ref2VA',length,i,flush=True)
        retest.write_json(a.output/'workloads.json',{str(s):workloads.description(s) for s in workloads.DIT_LENGTHS})
        return
    if a.reuse_placement:
        proof=json.loads((a.output/'placement_correctness.json').read_text())
        if not proof.get('is_bit_exact'):raise RuntimeError('Prior placement validation did not pass')
    if a.mode.startswith('tp'):
        os.environ['NCCL_P2P_DISABLE']='1'
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        mp.spawn(worker,args=(a,port),nprocs=2,join=True)
    else:execute(a)


if __name__=='__main__':main()
