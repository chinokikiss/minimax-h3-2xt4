"""Bounded two-rank T4 operator study; never constructs an H3 model."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
import torch.nn.functional as F
from real_t4_attention_study import measure,label,save
from attention_root_causes import metrics
from patches.tilelang_grouped_tp import prepare,routes,execute,reference


def selected(p,ids):
    ids=torch.tensor(ids,device=p.qi.device);hk=p.ki.shape[1];h=p.qi.shape[1]
    kid=ids//(h//hk)
    fields={name:getattr(p,name)[:,idx].contiguous() for name,idx in
            [('q',ids),('qi',ids),('qs',ids),('ki',kid),('ks',kid),('v',kid),('kc',kid),('vc',kid)]}
    return replace(p,**fields)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--profile',action='store_true');ap.add_argument('--repeats',type=int,default=10);args=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261016)
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
    dist.init_process_group('nccl');args.output.mkdir(parents=True,exist_ok=True)
    result={'status':'running','rank':rank,'hardware':torch.cuda.get_device_name(rank),'torch':torch.__version__,
        'scope':'Simultaneous physical TP2 ranks; real captured QKV sliced by current production partition. Attention operator only, no model pipeline.',
        'layout_limit':'Captured BHND inputs; DiT V reconstructs packed local-QKV stride from TP source. Q/K and Video stride-sensitive producer fusions require a later capture at TP hooks.',
        'cases':[]}
    target=args.output/f'rank{rank}.json';save(target,result)
    records=json.loads((args.data/'capture_manifest.json').read_text())['records']
    for record in records:
        path=args.data/record['file'];assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==record['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True);component=record['component'];causal=component=='te'
        if component=='te':
            q=data['q'][:,rank*32:(rank+1)*32];k=data['k'][:,rank*4:(rank+1)*4];v=data['v'][:,rank*4:(rank+1)*4];offset=0
            assert q.shape==(1,32,353,128) and k.shape==(1,4,353,128)
        elif component=='dit':
            q,k,v=[data[x][:,rank*28:(rank+1)*28] for x in ('q','k','v')];offset=0
            assert q.shape==(1,28,6814,128)
        else:
            a,b=((0,1024),(1024,1797))[rank];q=data['q'][:,:,a:b];k=data['k'];v=data['v'];offset=a
            assert q.shape==(1,32,b-a,64) and k.shape==(1,32,1797,64)
        original=[x.to(rank).contiguous() for x in (q,k,v)];del data,q,k,v;gc.collect()
        if component=='dit':
            b,h,n,d=original[2].shape
            storage=torch.empty((b,n,3*h*d),dtype=original[2].dtype,device=rank)
            packed=storage.view(b,n,3,h,d)[:,:,2].permute(0,2,1,3);packed.copy_(original[2]);original[2]=packed
        shape={'q':list(original[0].shape),'k':list(original[1].shape),'dtype':str(original[0].dtype),'strides':[list(x.stride()) for x in original],'query_offset':offset}
        def output(x):return x.transpose(1,2).reshape(1,original[0].shape[2],-1).contiguous().to(original[0].dtype)
        def native():
            with label(component+'/native_sdpa_and_layout'):
                return output(F.scaled_dot_product_attention(*original,is_causal=causal,enable_gqa=component=='te'))
        dist.barrier();start=time.perf_counter();expected=native();torch.cuda.synchronize()
        native_row={'component':component,'candidate':'native','shape':shape,'first_call_seconds':time.perf_counter()-start}
        if not args.profile:native_row['timings']={'API':measure(native,repeats=args.repeats)}
        result['cases'].append(native_row);save(target,result)
        cases=[(m,prep,cache) for m in (['dense','drop','scalar'] if component=='dit' else ['dense'])
               for prep,cache in [(False,False),(True,False),(True,True)]]
        if args.profile:cases=[c for c in cases if c[1]]
        random.Random(20261016).shuffle(cases)  # identical order on both ranks
        controls={}
        for mode,fused,cached in cases:
            name=f'{component}/{mode}/'+('fused' if fused else 'torch')+('/cached' if cached else '/group_loads')
            def prep():
                with label(name+'/cast_quant_centroids_pack'):
                    return prepare(*(x.half() for x in original),fused=fused,sparse=mode!='dense')
            def routing(p):
                with label(name+'/routing'):return routes(p,874) if mode!='dense' else None
            def raw(p,r):
                with label(name+'/QK_softmax_PV'):return execute(p,r,mode,causal,offset,cached)
            def api():
                pp=prep();rr=routing(pp)
                with label(name+'/attention_and_output_layout'):return output(raw(pp,rr))
            dist.barrier();start=time.perf_counter();p=prep();r=routing(p);actual=raw(p,r);torch.cuda.synchronize()
            row={'component':component,'candidate':name,'shape':shape,'first_call_including_compile_seconds':time.perf_counter()-start}
            if not args.profile:
                # Compare valid operands only: legacy padding scales are zero,
                # fused padding scales are 1e-8; padded keys are always masked.
                base=prepare(*(x.half() for x in original),sparse=mode!='dense')
                match={a:torch.equal(getattr(p,a)[:,:,:n],getattr(base,a)[:,:,:n]) for a,n in
                       [('qi',p.queries),('qs',p.queries),('ki',p.keys),('ks',p.keys)]}
                if r is not None:match['routes']=torch.equal(r,routes(base,874))
                assert all(match.values()),match;row['operand_match']=match
                ids=[0,original[0].shape[1]//2,original[0].shape[1]-1]
                sp=selected(p,ids);sr=r[:,ids].contiguous() if r is not None else None
                ref=reference(sp,sr,mode,causal,offset,query_chunk=256)
                row['implementation_error']=metrics(actual[:,ids],ref)
                assert torch.isfinite(actual).all() and max(row['implementation_error']['relative_l2'])<.003,row['implementation_error']
                row['native_error']=metrics(actual,expected.reshape(1,p.queries,-1,actual.shape[-1]).transpose(1,2))
                if mode in controls:
                    row['control_error']=metrics(actual,controls[mode]);assert max(row['control_error']['relative_l2'])<.003
                else:controls[mode]=actual.clone()
                row['density']=r.float().mean().item() if r is not None else 1.
                if mode!='dense' and not fused:
                    full=execute(p,torch.ones_like(r),mode,False,0,cached)
                    dense=execute(p,None,'dense',False,0,cached)
                    row['density1_error']=metrics(full,dense)
                    assert max(row['density1_error']['relative_l2'])<1e-6
                    del full,dense
                row['timings']={'preparation':measure(prep,repeats=args.repeats),'raw':measure(lambda:raw(p,r),repeats=args.repeats),
                                'API':measure(api,repeats=args.repeats)}
                if r is not None:row['timings']['routing']=measure(lambda:routing(p),repeats=args.repeats)
                del base,sp,ref
            else:
                api();torch.cuda.synchronize();dist.barrier();torch.cuda.profiler.start()
                for _ in range(3):api()
                torch.cuda.synchronize();torch.cuda.profiler.stop()
            row['status']='complete';result['cases'].append(row);save(target,result)
            print('TP2_ATTENTION',rank,name,'complete',flush=True)
            del p,r,actual;gc.collect()
        del original,expected,controls;gc.collect();torch.cuda.empty_cache()
    result['status']='complete';save(target,result);dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
