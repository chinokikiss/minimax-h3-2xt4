"""Tiny spawned-worker CUDA workload for flush/exit capture qualification."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time


def worker(rank,connection,flush,dwell=0.,checkpoint_output=None,exit_mode='graceful'):
    import torch
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    x=torch.full((256,256),float(rank+1),device=f'cuda:{rank}')
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f'H3_CAPTURE/rank={rank}')
    if checkpoint_output is not None:
        from worker_cuda_checkpoint import CUDAFlushCheckpoint
        checkpoints=CUDAFlushCheckpoint(rank,checkpoint_output)
        for chunk in (0,1):
            torch.cuda.nvtx.range_push(f'H3_CAPTURE_CHUNK/rank={rank}/chunk={chunk}')
            for _ in range(4):x.add_(.125)
            torch.cuda.nvtx.range_pop()
            checkpoints.flush(f'chunk_{chunk}',resume=chunk==0 or exit_mode=='terminate')
        expected=rank+2.
    else:
        for _ in range(4):x.add_(.125)
        expected=rank+1.5
    torch.cuda.nvtx.range_pop();torch.cuda.synchronize()
    value=float(x[0,0].item());assert value==expected
    if flush:torch.cuda.profiler.stop()
    if dwell:time.sleep(dwell)
    connection.send(dict(rank=rank,pid=os.getpid(),value=value,flushed=flush or checkpoint_output is not None,
        checkpoint_restart=checkpoint_output is not None))
    if not connection.recv():raise RuntimeError('Parent did not acknowledge worker')
    connection.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--flush',choices=('yes','no'),required=True);p.add_argument('--exit',choices=('graceful','terminate'),required=True)
    p.add_argument('--dwell-seconds',type=float,default=0.)
    p.add_argument('--checkpoint-restart',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    context=mp.get_context('spawn');processes=[];pipes=[];rows=[]
    try:
        for rank in (0,1):
            parent,child=context.Pipe();process=context.Process(target=worker,args=(rank,child,args.flush=='yes',args.dwell_seconds,
                str(args.output) if args.checkpoint_restart else None,args.exit))
            process.start();child.close();processes.append(process);pipes.append(parent)
        for parent in pipes:
            if not parent.poll(75):raise TimeoutError('Minimal CUDA worker did not acknowledge')
            rows.append(parent.recv())
        if args.exit=='graceful':
            for parent in pipes:parent.send(True)
        else:
            for process in processes:process.terminate()
        for process in processes:process.join(15)
        if any(process.is_alive() for process in processes):raise TimeoutError('Worker exit did not finish')
        if args.exit=='graceful' and any(process.exitcode!=0 for process in processes):raise RuntimeError('Graceful worker failed')
        (args.output/'workers.json').write_text(json.dumps(dict(workers=rows,flush=args.flush,exit=args.exit,
            exit_codes=[x.exitcode for x in processes]),indent=2))
    finally:
        for process in processes:
            if process.is_alive():process.terminate();process.join(5)
        for parent in pipes:parent.close()


if __name__=='__main__':main()
