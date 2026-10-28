"""Minimal per-worker CUPTI capture/flush/exit gate; no model allocation."""
import argparse
import gzip
import json
from pathlib import Path
import torch
import torch.multiprocessing as mp


def worker(rank, output):
    torch.cuda.set_device(rank)
    a=torch.ones((128,128),device='cuda')
    for _ in range(2):a=a@a
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(3):a=torch.ones_like(a)@torch.ones_like(a)
        torch.cuda.synchronize()
    path=Path(output)/f'rank{rank}.json'
    profile.export_chrome_trace(str(path))
    path.with_suffix('.json.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    path.unlink()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    ctx=mp.get_context('spawn')
    workers=[ctx.Process(target=worker,args=(i,str(args.output))) for i in (0,1)]
    for process in workers:process.start()
    rows=[]
    for rank,process in enumerate(workers):
        process.join(90)
        if process.is_alive():process.terminate();process.join(10)
        path=args.output/f'rank{rank}.json.gz'
        events=json.loads(gzip.decompress(path.read_bytes()))['traceEvents'] if path.is_file() else []
        kernels=sum(e.get('cat')=='kernel' for e in events)
        rows.append(dict(rank=rank,exit_code=process.exitcode,kernel_events=kernels,
                         passed=process.exitcode==0 and kernels>=3))
    passed=all(r['passed'] for r in rows)
    (args.output/'gate.json').write_text(json.dumps(dict(passed=passed,workers=rows,checked_after_process_exit=True),indent=2))
    if not passed:raise SystemExit('CUPTI capture/flush/exit gate failed; kernel attribution blocked')


if __name__=='__main__':main()
