"""Deferred TP2 communication probe. No models, Kitchen, TileLang or uploads.

Run in fresh two-T4 processes, once with --protocol default and once Simple.
This file is prepared offline; writing it does not launch a Kaggle session.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import torch.distributed as dist
from tp2_timing import measure


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',choices=['default','Simple','LL'],default='default')
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--channels',type=int);ap.add_argument('--profile',action='store_true');args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text())
    rows=[r for r in manifest['GEMMs'] if r['component']=='dit' and r['partition']=='row']
    assert len(rows)==2 and len({(r['M'],r['N']) for r in rows})==1
    m,n=rows[0]['M'],rows[0]['N']
    video=sorted((r for r in manifest['attention'] if r['component']=='video'),key=lambda r:r['rank'])
    assert len(video)==2 and video[0]['k']==video[1]['k']
    vb,vh,vn,vd=video[0]['k'];maximum=max(r['q'][2] for r in video)
    assert sum(r['q'][2] for r in video)==vn
    if args.protocol!='default':os.environ['NCCL_PROTO']=args.protocol
    elif 'NCCL_PROTO' in os.environ:raise RuntimeError('Default requires a clean NCCL_PROTO environment')
    os.environ['NCCL_P2P_DISABLE']='1'
    if args.channels:
        if args.channels not in (2,4,8):raise ValueError('Bounded channel shortlist only')
        os.environ['NCCL_MIN_NCHANNELS']=os.environ['NCCL_MAX_NCHANNELS']=str(args.channels)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False)
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=60))
    assert dist.get_world_size()==2
    payloads=[('dit_INT32_SUM',torch.zeros((m,n),device=rank,dtype=torch.int32),None),
              ('row_FP32_MAX',torch.zeros((m,1),device=rank),None),
              ('video_padded_KV_all_gather',torch.full((maximum,2,vb,vh,vd),rank+1,device=rank,dtype=torch.float16),
               torch.empty((2*maximum,2,vb,vh,vd),device=rank,dtype=torch.float16))]
    result={'status':'running','rank':rank,'hardware':torch.cuda.get_device_name(rank),'torch':torch.__version__,
            'nccl_version':torch.cuda.nccl.version(),
            'settings':{k:v for k,v in os.environ.items() if k.startswith('NCCL_')},'cases':[],
            'TP_manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            'scope':'Actual TP2 message shapes only. Raw collectives exclude projection, packing, compute overlap and whole pipeline.'}
    args.output.mkdir(parents=True,exist_ok=True)
    for name,x,out in payloads:
        def operation():
            torch.cuda.nvtx.range_push(name)
            try:
                if out is not None:dist.all_gather_into_tensor(out,x)
                else:dist.all_reduce(x,op=dist.ReduceOp.MAX if name.endswith('MAX') else dist.ReduceOp.SUM)
            finally:torch.cuda.nvtx.range_pop()
        if out is None:
            x.fill_(rank+1);operation();assert torch.all(x==(2 if name.endswith('MAX') else 3)).item()
            # Stable zero input avoids accumulation overflow without charging
            # an unrelated device copy to each measured all-reduce.
            x.zero_()
        else:
            operation();assert torch.all(out[:maximum]==1).item() and torch.all(out[maximum:]==2).item()
        dist.barrier()
        row={'name':name,'shape':list(x.shape),'dtype':str(x.dtype),'input_bytes':x.numel()*x.element_size()}
        if args.profile:
            for _ in range(10):operation()
            torch.cuda.synchronize();dist.barrier();torch.cuda.profiler.start()
            for _ in range(3):operation()
            torch.cuda.synchronize();torch.cuda.profiler.stop()
        else:row['timings']=measure(operation,warmup=10,repeats=15)
        result['cases'].append(row)
    result['status']='complete';(args.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
