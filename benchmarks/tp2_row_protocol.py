"""Current row-linear API under one NCCL setting; no models or quality study."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
from tp2_timing import measure
from tp2_gemm_operators import instrument
from patches import optimized_tp as opt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--protocol',choices=['default','Simple','LL'],default='default');ap.add_argument('--channels',type=int);ap.add_argument('--profile',action='store_true');args=ap.parse_args()
    if args.protocol!='default':os.environ['NCCL_PROTO']=args.protocol
    elif 'NCCL_PROTO' in os.environ:raise RuntimeError('Default requires clean NCCL_PROTO')
    os.environ['NCCL_P2P_DISABLE']='1'
    if args.channels:os.environ['NCCL_MIN_NCHANNELS']=os.environ['NCCL_MAX_NCHANNELS']=str(args.channels)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=90));instrument()
    args.output.mkdir(parents=True,exist_ok=True)
    result={'status':'running','rank':rank,'hardware':torch.cuda.get_device_name(rank),'torch':torch.__version__,
            'nccl_version':torch.cuda.nccl.version(),
            'protocol':args.protocol,'channels':args.channels,'cases':[],
            'scope':'Unchanged current optimized_tp row_linear functions. Deterministic synthetic activations/weights at current TP2 MNK. No real linear-quality or full pipeline claim.'}
    rows=[r for r in json.loads(args.manifest.read_text())['GEMMs'] if r['component']=='dit' and r['partition']=='row']
    for index,row in enumerate(rows):
        torch.manual_seed(20261016+rank+index*99)
        m,n,k=[row[a] for a in ('M','N','K')];act=row['input_act']
        x=torch.randn((m,k*(2 if act=='swiglu' else 1)),device=rank)*.25
        w=torch.randint(-127,128,(n,k),device=rank,dtype=torch.int8);ws=torch.full((n,1),.001,device=rank)
        def api():
            torch.cuda.nvtx.range_push('dit/'+row['module']+'/API')
            try:return opt.row_linear(x,w,ws,dist,2,input_act=act)
            finally:torch.cuda.nvtx.range_pop()
        dist.barrier();start=time.perf_counter();actual=api();torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
        record={'target':row,'first_call_seconds':time.perf_counter()-start,
                'output_sha256':hashlib.sha256(actual.cpu().contiguous().numpy().tobytes()).hexdigest()}
        if args.profile:
            for _ in range(5):api()
            torch.cuda.synchronize();dist.barrier();torch.cuda.profiler.start()
            for _ in range(3):api()
            torch.cuda.synchronize();torch.cuda.profiler.stop()
        else:record['API']=measure(api,repeats=15)
        result['cases'].append(record);(args.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
        print('ROW_PROTOCOL',rank,args.protocol,args.channels,row['module'],'complete',flush=True)
        del actual,x,w,ws
    result['status']='complete';(args.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
