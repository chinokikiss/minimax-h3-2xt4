"""Physical two-T4 current linear paths, synthetic inputs at verified TP shapes.

This is a performance/core-equivalence fixture, not a quantization-quality
study of real linear activations. No checkpoint/model construction occurs.
"""
import argparse
from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
import comfy_kitchen as ck
from real_t4_attention_study import label,save
from tp2_timing import measure,warm_device
from patches import parallel_int8 as pi
from patches import optimized_tp as opt


def instrument():
    def wrapped(owner,name,tag):
        fn=getattr(owner,name)
        def call(*a,**kw):
            with label(tag):return fn(*a,**kw)
        setattr(owner,name,call)
    for name in ['_apply_input_act','_rotate_activation','quantize_int8_rowwise','quantize_and_rotate_rowwise',
                 'quantize_int8_rowwise_convrot64','quantize_int8_rowwise_convrot']:
        wrapped(ck.backends.cuda,name,'linear/'+name)
    wrapped(pi,'dequantize','linear/dequantize')
    original=opt.native_int_mm
    def mm(*a,**kw):
        with label('linear/INT8_GEMM_to_INT32'):return original(*a,**kw)
    pi.int_mm=mm
    for name in ['all_reduce','all_gather']:
        fn=getattr(dist,name)
        def call(*a,_fn=fn,_name=name,**kw):
            tensor=a[0];dtype=str(tensor.dtype) if isinstance(tensor,torch.Tensor) else 'list'
            with label('NCCL/'+_name+'/'+dtype+'/'+str(kw.get('op',''))):return _fn(*a,**kw)
        setattr(dist,name,call)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--profile',action='store_true');args=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261016+rank)
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120));instrument();warm_device();dist.barrier()
    args.output.mkdir(parents=True,exist_ok=True);target=args.output/f'rank{rank}.json'
    result={'status':'running','rank':rank,'hardware':torch.cuda.get_device_name(rank),'cases':[],
        'scope':'Actual optimized TP linear functions, simultaneous ranks and NCCL. Synthetic A/W at source-verified MNK; no real-activation quality claim.'}
    rows=json.loads(args.manifest.read_text())['GEMMs'];seen=set()
    for row in rows:
        if row.get('rank',rank)!=rank:continue
        m,n,k=[row[x] for x in ('M','N','K')];identity=(row['component'],m,n,k,row['partition'])
        if identity in seen:continue
        seen.add(identity);dtype=getattr(torch,row['input_dtype']);act=row['input_act'];isrow=row['partition']=='row'
        x=torch.randn((m,k*(2 if act=='swiglu' else 1)),device=rank,dtype=dtype)*.25
        w=torch.randint(-127,128,(n,k),dtype=torch.int8,device=rank);ws=torch.full((n,1),.001,device=rank)
        def api():
            with label(row['component']+'/'+row['module']+'/API'):
                if isrow:return opt.row_linear(x,w,ws,dist,2,input_act=act)
                # RMSNorm projection fusion needs real norm weights; baseline
                # isolates ConvRot+quant+GEMM, documents excluded RMSNorm.
                return ck.int8_linear(x,w,ws,None,dtype,convrot=True,convrot_groupsize=256,input_act=act if act=='swiglu' else None)
        dist.barrier();start=time.perf_counter();actual=api();torch.cuda.synchronize()
        record={'target':row,'first_call_seconds':time.perf_counter()-start,'excluded':'RMSNorm weight/residual inputs absent from synthetic GEMM fixture',
                'row_route':('scale_max' if opt.use_fallback_rotation(x,k*2,256) else 'full_feature_gather') if isrow else 'native_column',
                'output_finite':bool(torch.isfinite(actual).all())}
        assert record['output_finite']
        if args.profile:
            # Warm cuBLAS/NCCL before the measured capture.
            for _ in range(3):api()
            torch.cuda.synchronize();dist.barrier();torch.cuda.profiler.start()
            for _ in range(3):api()
            torch.cuda.synchronize();torch.cuda.profiler.stop()
        else:
            record['API']=measure(api,repeats=10)
            q=torch.randint(-127,128,(m,k),dtype=torch.int8,device=rank)
            native=pi.int_mm(q,w)
            # Independent PyTorch integer GEMM verifies the INT32 core.
            ref=pi.LEGACY_MM(q,w) if hasattr(pi,'LEGACY_MM') else opt.LEGACY_MM(q,w)
            record['INT32_core_exact']=torch.equal(native,ref);assert record['INT32_core_exact']
            record['raw_INT8_GEMM']=measure(lambda:pi.int_mm(q,w),repeats=10)
            if isrow:
                partial=native.clone()
                record['INT32_SUM']=measure(lambda:dist.all_reduce(partial,op=dist.ReduceOp.SUM),repeats=10)
                maximum=torch.ones((m,1),device=rank)
                record['FP32_MAX']=measure(lambda:dist.all_reduce(maximum,op=dist.ReduceOp.MAX),repeats=10)
            del q,native,ref
        record['status']='complete';result['cases'].append(record);save(target,result)
        print('TP2_GEMM',rank,row['component'],row['module'],m,n,k,'complete',flush=True)
        del x,w,ws,actual;torch.cuda.empty_cache()
    result['status']='complete';save(target,result);dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
