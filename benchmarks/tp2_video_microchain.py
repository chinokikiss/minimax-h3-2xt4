"""Exact active Video KV exchange + current Kitchen/Group32 attention, no model."""
import argparse
import hashlib
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
from attention_root_causes import metrics
from patches.parallel_vae import AlignedVideoDecode,aligned_ranges
from patches.tilelang_grouped_tp import prepare,execute


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--profile',action='store_true');args=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank));warm_device();dist.barrier()
    record=next(r for r in json.loads((args.data/'capture_manifest.json').read_text())['records'] if r['component']=='video')
    file=args.data/record['file'];assert hashlib.file_digest(file.open('rb'),'sha256').hexdigest()==record['sha256']
    data=torch.load(file,map_location='cpu',weights_only=True);ranges=aligned_ranges(1797);a,b=ranges[rank]
    q=data['q'][:,:,a:b].cuda().contiguous();k=data['k'].cuda().contiguous();v=data['v'].cuda().contiguous()
    local_k=k[:,:,a:b].contiguous();local_v=v[:,:,a:b].contiguous();sizes=[y-x for x,y in ranges]
    runner=object.__new__(AlignedVideoDecode);runner.buffers={};runner.comm=dist
    args.output.mkdir(parents=True,exist_ok=True);target=args.output/f'rank{rank}.json'
    result={'status':'running','rank':rank,'hardware':torch.cuda.get_device_name(rank),'q_shape':list(q.shape),'k_shape':list(k.shape),'cases':[],
        'scope':'Actual AlignedVideoDecode.exchange method, real QKV, current Kitchen int8_attention baseline. Excludes QKV producer, RoPE, output linear and full pipeline.'}
    expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    def exchange():
        with label('video/KV_stack_exchange_unpack'):
            return runner.exchange(torch.stack((local_k,local_v)),sizes,'kv',3).unbind(0)
    kk,vv=exchange();assert torch.equal(kk,k) and torch.equal(vv,v)
    for mode in ('current_kitchen','group32','group32_cached'):
        def attention(kk,vv):
            with label(mode+'/attention_API'):
                if mode=='current_kitchen':return ck.int8_attention(q,kk,vv)
                p=prepare(q,kk,vv,fused=True)
                return execute(p,query_offset=a,cached=mode.endswith('cached'))
        def chain():
            with label(mode+'/microchain'):
                kk,vv=exchange();out=attention(kk,vv)
                with label(mode+'/BND_finite_output'):return torch.nan_to_num(out.transpose(1,2).reshape(1,b-a,-1))
        dist.barrier();start=time.perf_counter();actual=attention(k,v);torch.cuda.synchronize()
        row={'candidate':mode,'first_call_seconds':time.perf_counter()-start,'native_SDPA_error':metrics(actual,expected)}
        assert torch.isfinite(actual).all()
        if args.profile:
            for _ in range(5):chain()
            torch.cuda.synchronize();dist.barrier();torch.cuda.profiler.start()
            for _ in range(3):chain()
            torch.cuda.synchronize();torch.cuda.profiler.stop()
        else:
            row['attention_API']=measure(lambda:attention(k,v),repeats=15)
            row['KV_exchange']=measure(exchange,repeats=15)
            row['microchain']=measure(chain,repeats=15)
        result['cases'].append(row);save(target,result);print('VIDEO_MICROCHAIN',rank,mode,'complete',flush=True)
    result['status']='complete';save(target,result);dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
