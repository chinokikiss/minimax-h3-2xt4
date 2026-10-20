"""Balanced paired FC2 confirmation with synchronized per-call Systems capture."""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import torch
import torch.distributed as dist
import tp2_tradeoffs as base
from patches import tilelang_transport_p2 as fused


def run(args,rank,result,target):
    fixture=next(r for r in json.loads((args.data/'real_transport_manifest.json').read_text())['records'] if r['name']=='fc2_l0_s0')
    path=args.data/fixture['file'];assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==fixture['sha256']
    data=torch.load(path,map_location='cpu',weights_only=True);k=7168;m=6814;n=5376
    x=torch.cat([p[:,rank*k:(rank+1)*k] for p in data['x'].chunk(2,-1)],-1).to(rank).contiguous()
    w=data['weights']['weight'][:,rank*k:(rank+1)*k].to(rank).contiguous();ws=data['weights']['weight_scale'].to(rank).contiguous();del data
    def quant():
        with base.label('linear/activation_rotation_quantization'):return base.swiglu.global_row_quantize(x,dist,2)
    q,s=quant();partial=base.native.native_int_mm(q,w);both=torch.empty(2*m,n,device='cuda',dtype=torch.int32);dist.all_gather_into_tensor(both,partial)
    parts=both.reshape(2,m,n);expected=base.pi.dequantize(parts[0]+parts[1],s,ws,torch.float32)
    variants=['int32','fp16','fp16_fused','int8_gather','int8_gather_fused']
    if args.round%2:variants.reverse()
    apis={};records={}
    for variant in variants:
        scheme=variant.removesuffix('_fused');is_fused=variant.endswith('_fused');name='communication/fc2_l0_s0/'+variant
        def api(scheme=scheme,is_fused=is_fused,name=name):
            with base.label(name+'/API'):
                aq,asc=quant()
                with base.label('linear/INT8_GEMM_to_INT32'):acc=base.native.native_int_mm(aq,w)
                del aq
                if is_fused:
                    with base.label('linear/encode_'+scheme):payload=fused.encode(acc,scheme,k)
                    del acc
                    with base.label('linear/collect_'+scheme):received=fused.collect(payload,scheme,dist)
                    if received is not payload:del payload
                    with base.label('linear/fused_reconstruction_dequantize'):return fused.finish(received,asc,ws,scheme,k)
                with base.label('linear/transport_'+scheme):reduced=base.transport.reduce(acc,dist,scheme,k)
                with base.label('linear/dequantize'):return base.pi.dequantize(reduced,asc,ws,torch.float32)
        actual=api();reference=base.pi.dequantize(base.transport_reference(parts,scheme,k),s,ws,torch.float32)
        error=None if base.byte_equal(actual,reference) else 'Independent codec/native-dequant mismatch'
        errors=base.agree(error);assert not any(errors),errors
        records[variant]=dict(family='communication/fc2_l0_s0',name=variant,status='gated',fixture=fixture,
            independent_contract_gate='bitwise_pass',output_sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),
            arithmetic_diagnostics=base.diagnostics(actual,expected),gpu=[],wall=[])
        apis[variant]=api;del actual,reference
    del q,s,partial,both,parts,expected;gc.collect();torch.cuda.empty_cache()
    base.warm_device();dist.barrier()
    for _ in range(10):
        for variant in variants:apis[variant]()
    torch.cuda.synchronize();dist.barrier()
    def gc_range(phase,info):
        if phase=='start':torch.cuda.nvtx.range_push('CPU_GC/generation'+str(info['generation']))
        else:torch.cuda.nvtx.range_pop()
    if args.profile:gc.callbacks.append(gc_range);torch.cuda.profiler.start()
    orders=[]
    try:
        for sample in range(30):
            offset=sample%len(variants);order=variants[offset:]+variants[:offset];orders.append(order)
            with base.label('paired/sample'+str(sample)):
                for variant in order:
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    start=time.perf_counter();a.record();apis[variant]();b.record();b.synchronize()
                    records[variant]['gpu'].append(a.elapsed_time(b));records[variant]['wall'].append((time.perf_counter()-start)*1000)
        if args.profile:
            with base.label('profile/rank_completion_barrier'):dist.barrier();torch.cuda.synchronize()
    finally:
        if args.profile:torch.cuda.profiler.stop();gc.callbacks.remove(gc_range)
    for variant in variants:
        record=records[variant];gpu=record.pop('gpu');wall=record.pop('wall')
        record.update(status='complete',API=dict(gpu_ms=gpu,wall_ms=wall,gpu_median_ms=statistics.median(gpu),wall_median_ms=statistics.median(wall),
            repeats=30,warmup_actual_calls=10,warmup_policy='Ten fixed calls per candidate, interleaved. One second local clock prewarm. Thirty blocks rotate candidate order; fresh process reverses base order.'),
            paired_native_minus_candidate_ms=[a-b for a,b in zip(records['int32']['API']['gpu_ms'] if 'API' in records['int32'] else records['int32']['gpu'],gpu)] if variant!='int32' else [0.]*30)
        result['cases'].append(record)
    result['sample_orders']=orders
    base.save(target,result);print('P2_FC2_PAIRED_CONFIRMATION',rank,'complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--round',type=int,default=0);parser.add_argument('--profile',action='store_true');args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert os.environ.get('NCCL_P2P_DISABLE')=='1' and os.environ.get('NCCL_MIN_NCHANNELS')==os.environ.get('NCCL_MAX_NCHANNELS')=='4' and 'NCCL_PROTO' not in os.environ
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=120))
    for method in ('all_reduce','all_gather_into_tensor'):
        original=getattr(dist,method)
        def wrapped(*a,_method=method,_original=original,**kw):
            with base.label('NCCL/'+_method+'/'+str(a[0].dtype)+'/'+str(kw.get('op',''))):return _original(*a,**kw)
        setattr(dist,method,wrapped)
    args.output.mkdir(parents=True,exist_ok=True);target=args.output/f'rank{rank}.json'
    result=dict(status='running',rank=rank,cases=[],hardware=torch.cuda.get_device_name(),torch=torch.__version__,scope='Additional paired FC2 stability confirmation; different scheduling from the original sequential screen.')
    run(args,rank,result,target);result['status']='complete';base.save(target,result);dist.barrier();dist.destroy_process_group()
