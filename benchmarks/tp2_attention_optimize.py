"""Same-policy T4 attention optimization; no dense-relative quality study."""
import argparse
import datetime
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
from real_t4_attention_study import label,save
from tp2_timing import measure
from patches import tilelang_grouped_tp as old
from patches import tilelang_grouped_tp_optimized as opt


def configurations(component,mode):
    rows=[{'name':'frozen','baseline':True}]
    if component=='video':
        rows += [dict(name='q64_BND',tile=64,cache='group'),dict(name='q64_BND_finite',tile=64,cache='group',finite=True)]
    elif component=='te':
        rows += [dict(name='q32_BND',tile=32,cache='group'),dict(name='q64_Qreuse_BND',tile=64,cache='q')]
    elif mode=='dense':
        rows += [dict(name='q32_BND',tile=32,cache='group'),dict(name='q32_Qreuse_BND',tile=32,cache='q')]
    elif mode=='drop':
        rows += [dict(name='q32_pruned_BND',tile=32,cache='group'),dict(name='q32_tau0_BND',tile=32,cache='group',fast=True)]
    else:
        rows += [dict(name='q32_tau0_BND',tile=32,cache='group',fast=True),dict(name='q64_tau0_BND',tile=64,cache='group',fast=True)]
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--selection',type=Path);ap.add_argument('--round',type=int,default=0);ap.add_argument('--profile',action='store_true');args=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=90))
    selection=json.loads(args.selection.read_text()) if args.selection else None
    args.output.mkdir(parents=True,exist_ok=True);target=args.output/f'rank{rank}.json'
    result={'status':'running','rank':rank,'round':args.round,'hardware':torch.cuda.get_device_name(rank),'torch':torch.__version__,
        'scope':'Real captured TP2 partitions, simultaneous ranks. Same current dense/drop/scalar operator only; no sparse-versus-dense L1/L2 or generation-quality gate.',
        'cases':[]};save(target,result)
    records=json.loads((args.data/'capture_manifest.json').read_text())['records']
    for record in records:
        path=args.data/record['file'];assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==record['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True);component=record['component'];causal=component=='te'
        if component=='te':
            tensors=[data['q'][:,rank*32:(rank+1)*32],data['k'][:,rank*4:(rank+1)*4],data['v'][:,rank*4:(rank+1)*4]];offset=0
        elif component=='dit':tensors=[data[x][:,rank*28:(rank+1)*28] for x in ('q','k','v')];offset=0
        else:
            a,b=((0,1024),(1024,1797))[rank];tensors=[data['q'][:,:,a:b],data['k'],data['v']];offset=a
        original=[x.to(rank).contiguous() for x in tensors];del data,tensors;gc.collect()
        if component=='dit':
            b,h,n,d=original[2].shape;storage=torch.empty((b,n,3*h*d),device=rank,dtype=original[2].dtype)
            packed=storage.view(b,n,3,h,d)[:,:,2].permute(0,2,1,3);packed.copy_(original[2]);original[2]=packed
        for mode in (('dense','drop','scalar') if component=='dit' else ('dense',)):
            family=component+'/'+mode
            def layout(x):
                y=x.transpose(1,2).reshape(1,original[0].shape[2],-1).contiguous().to(original[0].dtype)
                return torch.nan_to_num(y) if component=='video' else y
            base=old.prepare(*(x.half() for x in original),fused=True,sparse=mode!='dense')
            br=old.routes(base,874) if mode!='dense' else None
            expected=layout(old.execute(base,br,mode,causal,offset,cached=component=='video'))
            assert torch.isfinite(expected).all()
            configs=configurations(component,mode)
            if selection:
                chosen=selection['winners'][family]['name'];configs=[c for c in configs if c['name'] in ('frozen',chosen)]
            random.Random(442+args.round).shuffle(configs)
            for cfg in configs:
                name=family+'/'+cfg['name'];baseline=cfg.get('baseline',False)
                def prep():
                    with label(name+'/preparation'):
                        half=[x.half() for x in original]
                        return old.prepare(*half,fused=True,sparse=mode!='dense') if baseline else opt.prepare(*half,fused=True,mode=mode,query_tile=cfg['tile'])
                def routing(p):
                    with label(name+'/routing'):
                        if mode=='dense':return None
                        return old.routes(p,874) if baseline else opt.routes(p,874,fast_tau0=cfg.get('fast',False))
                def raw(p,r):
                    with label(name+'/QK_softmax_PV_and_output'):
                        if baseline:return layout(old.execute(p,r,mode,causal,offset,cached=component=='video'))
                        y=opt.execute(p,r,causal,offset,cfg['cache'],layout='BND',output_dtype=original[0].dtype,finite_output=cfg.get('finite',False))
                        return torch.nan_to_num(y) if component=='video' and not cfg.get('finite',False) else y
                def api():
                    with label(name+'/API'):
                        p=prep();r=routing(p);return raw(p,r)
                row={'family':family,'name':cfg['name'],'config':cfg,'q_shape':list(original[0].shape),'k_shape':list(original[1].shape),
                     'input_strides':[list(x.stride()) for x in original]}
                dist.barrier();start=time.perf_counter();p=None;r=None;actual=None;error=None
                try:
                    p=prep();r=routing(p);actual=raw(p,r);torch.cuda.synchronize()
                    row['first_call_including_JIT_seconds']=time.perf_counter()-start
                    match={key:torch.equal(getattr(p,key)[:,:,:n],getattr(base,key)[:,:,:n]) for key,n in
                           [('qi',base.queries),('qs',base.queries),('ki',base.keys),('ks',base.keys)]}
                    if r is not None:match['routes']=torch.equal(r,br)
                    if mode=='scalar':match['KC']=torch.equal(p.kc,base.kc);match['VC']=torch.equal(p.vc,base.vc)
                    assert all(match.values()),match
                    assert torch.isfinite(actual).all()
                    # Implementation conformance to exactly the same sparse
                    # approximation; this is not a native/dense quality metric.
                    torch.testing.assert_close(actual,expected,rtol=.001,atol=.0002)
                    row['operand_identity']=match;row['same_operator_conformance']='pass';row['bitwise_output_equal']=torch.equal(actual,expected)
                except Exception as e:
                    error=type(e).__name__+': '+str(e);row['status']='failed';row['error']=error[:2400]
                errors=[None,None];dist.all_gather_object(errors,error)
                if any(errors):
                    row['rank_errors']=errors;row['status']='failed';result['cases'].append(row);save(target,result)
                    print('CANDIDATE_FAILED',rank,name,errors,flush=True);continue
                if args.profile:
                    for _ in range(5):api()
                    torch.cuda.synchronize();dist.barrier();torch.cuda.profiler.start()
                    for _ in range(3):api()
                    torch.cuda.synchronize();torch.cuda.profiler.stop()
                else:
                    row['API']=measure(api,warmup=10,repeats=15)
                row['status']='complete';result['cases'].append(row);save(target,result)
                print('OPERATOR_CANDIDATE',rank,name,'complete',flush=True)
                del p,r,actual;gc.collect()
            del base,br,expected
        del original;gc.collect();torch.cuda.empty_cache()
    result['status']='complete';save(target,result)
    collected=[None,None];dist.all_gather_object(collected,result)
    if rank==0 and not args.profile:
        families=sorted({r['family'] for r in result['cases']});winners={}
        for family in families:
            maps=[{r['name']:r for r in x['cases'] if r['family']==family and r['status']=='complete'} for x in collected]
            scores={name:max(m[name]['API']['gpu_median_ms'] for m in maps) for name in set(maps[0])&set(maps[1])}
            winner=min(scores,key=scores.get);winners[family]={'name':winner,'slower_rank_API_ms':scores[winner],'all_slower_rank_API_ms':scores}
        save(args.output/'selection.json',{'winners':winners,'scope':'Screening selection; requires independent-process confirmation. No generation quality inference.'})
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
