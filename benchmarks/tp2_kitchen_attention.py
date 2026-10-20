"""Current TP2 dense operators, Kitchen comparator, same-precision candidates."""
import argparse
from contextlib import contextmanager
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import torch
import torch.distributed as dist
import comfy_kitchen as ck
from patches import tilelang_grouped_tp_optimized as old, tilelang_grouped_tp_kitchen as new
from tp2_timing import measure


@contextmanager
def label(name):
    torch.cuda.nvtx.range_push(name)
    try:yield
    finally:torch.cuda.nvtx.range_pop()


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def identity(actual,expected):
    return {key:torch.equal(getattr(actual,key),getattr(expected,key)) for key in ('qi','ki','qs','ks','v','counts')}


def unit_gate(rank,output):
    rows=[];torch.manual_seed(20261017+rank)
    for dtype,d,n,nk,h,hk,tile in ((torch.float32,64,129,129,4,1,128),(torch.float16,128,97,65,2,2,64)):
        q=torch.randn(1,h,n,d,device=rank,dtype=dtype);k=torch.randn(1,hk,nk,d,device=rank,dtype=dtype)
        # Noncontiguous V mimics packed BND QKV with independent owned heads.
        storage=torch.randn(1,nk,3*hk*d,device=rank,dtype=dtype)
        v=storage.view(1,nk,3,hk,d)[:,:,2].permute(0,2,1,3)
        expected=old.prepare(q.half(),k.half(),v.half(),fused=True,query_tile=tile)
        actual=new.prepare(q,k,v,query_tile=tile);match=identity(actual,expected);assert all(match.values()),match
        rows.append(dict(dtype=str(dtype),D=d,Q=n,K=nk,query_tile=tile,identity=match))
    save(output/f'unit_rank{rank}.json',rows);print('KITCHEN_IDENTITY_GATE',rank,'pass',flush=True)


def dense_sample(q,k,v,offset,causal):
    idx=torch.linspace(0,q.shape[2]-1,min(32,q.shape[2]),device=q.device).long().unique()
    heads=torch.arange(q.shape[1],device=q.device)//(q.shape[1]//k.shape[1])
    scores=q.index_select(2,idx).float()@k.float().index_select(1,heads).transpose(-1,-2)/(q.shape[-1]**.5)
    if causal:scores.masked_fill_(torch.arange(k.shape[2],device=q.device)[None,:]>idx[:,None]+offset,-float('inf'))
    expected=scores.softmax(-1)@v.float().index_select(1,heads)
    return idx,expected


def dense_error(actual,reference):
    delta=actual.float()-reference
    return dict(per_head_relative_RMSE=(delta.square().sum((-2,-1))/reference.square().sum((-2,-1)).clamp_min(1e-30)).sqrt().flatten().tolist(),
                per_head_max_abs=delta.abs().amax((-2,-1)).flatten().tolist(),
                scope='32 evenly spaced real query positions per head, all keys, FP32 dense math with TF32 disabled; not full-output or generation quality.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path);ap.add_argument('--matrix',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--selection',type=Path);ap.add_argument('--round',type=int,default=0)
    ap.add_argument('--profile',action='store_true');ap.add_argument('--components',nargs='+',default=['te','dit','video']);a=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=90))
    a.output.mkdir(parents=True,exist_ok=True);target=a.output/f'rank{rank}.json'
    matrix=json.loads(a.matrix.read_text())['configurations'];selection=json.loads(a.selection.read_text()) if a.selection else None
    result=dict(status='running',rank=rank,hardware=torch.cuda.get_device_name(rank),torch=torch.__version__,cases=[],
        scope='Real native captures sliced to current TP2 partitions. Dense attention only; no H3 pipeline, KV exchange or sparse precision/policy study.')
    save(target,result)
    if not selection:unit_gate(rank,a.output)
    for record in json.loads((a.data/'capture_manifest.json').read_text())['records']:
        component=record['component']
        if component not in a.components:continue
        path=a.data/record['file'];assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==record['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        if component=='te':values=[data['q'][:,rank*32:(rank+1)*32],data['k'][:,rank*4:(rank+1)*4],data['v'][:,rank*4:(rank+1)*4]];offset=0
        elif component=='dit':values=[data[x][:,rank*28:(rank+1)*28] for x in ('q','k','v')];offset=0
        else:
            x,y=((0,1024),(1024,1797))[rank];values=[data['q'][:,:,x:y],data['k'],data['v']];offset=x
        original=[x.to(rank).contiguous() for x in values];del data,values;gc.collect()
        if component=='dit':
            b,h,n,d=original[2].shape;storage=torch.empty((b,n,3*h*d),device=rank,dtype=original[2].dtype)
            v=storage.view(b,n,3,h,d)[:,:,2].permute(0,2,1,3);v.copy_(original[2]);original[2]=v
        b,h,n,d=original[0].shape;causal=component=='te';base_cfg=next(c for c in matrix[component] if c.get('baseline'))
        base=old.prepare(*(x.half() for x in original),fused=True,query_tile=base_cfg['tile'])
        expected=old.execute(base,causal=causal,query_offset=offset,cache=base_cfg['cache'],layout='BND',output_dtype=original[0].dtype)
        if component=='video':expected=torch.nan_to_num(expected)
        indices,reference=dense_sample(*original,offset,causal)
        configs=list(matrix[component])
        if selection:
            chosen=selection['winners'][component]['name'];configs=[c for c in configs if c['name'] in ('current',chosen,'kitchen')]
        random.Random(20261017+a.round).shuffle(configs)
        for cfg in configs:
            name=component+'/'+cfg['name'];native=cfg.get('native',False)
            def prep():
                with label(name+'/preparation'):
                    return new.prepare(*original,query_tile=cfg['tile']) if cfg.get('fused') else old.prepare(*(x.half() for x in original),fused=True,query_tile=cfg['tile'])
            def raw(p):
                with label(name+'/QK_softmax_PV_and_output'):
                    function=new.execute if 'threads' in cfg else old.execute
                    kw=dict(threads=cfg['threads'],warp_policy=cfg['warp']) if 'threads' in cfg else {}
                    out=function(p,causal=causal,query_offset=offset,cache=cfg['cache'],layout='BND',output_dtype=original[0].dtype,**kw)
                    return torch.nan_to_num(out) if component=='video' else out
            def api():
                with label(name+'/API'):
                    if native:
                        out=ck.int8_attention(*original)
                        return torch.nan_to_num(out.transpose(1,2).reshape(b,n,h*d).contiguous())
                    return raw(prep())
            row=dict(component=component,name=cfg['name'],config=cfg,q_shape=list(original[0].shape),k_shape=list(original[1].shape),
                     input_strides=[list(x.stride()) for x in original]);error=None;p=None;actual=None
            dist.barrier();start=time.perf_counter()
            try:
                if not native:
                    p=prep();actual=raw(p)
                    own=old.prepare(*(x.half() for x in original),fused=True,query_tile=cfg['tile'])
                    row['operand_identity']=identity(p,own);assert all(row['operand_identity'].values()),row['operand_identity'];del own
                    torch.testing.assert_close(actual,expected,rtol=.001,atol=.0002)
                    row['bitwise_equal_to_current']=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
                else:actual=api()
                torch.cuda.synchronize();row['first_call_including_JIT_and_gate_seconds']=time.perf_counter()-start
                assert torch.isfinite(actual).all()
                row['output_sha256']=hashlib.sha256(actual.cpu().contiguous().numpy().tobytes()).hexdigest()
                row['dense_sample_error']=dense_error(actual.reshape(b,n,h,d).transpose(1,2).index_select(2,indices),reference)
            except Exception as e:error=(type(e).__name__+': '+str(e))[-3000:]
            errors=[None,None];dist.all_gather_object(errors,error)
            if any(errors):
                row.update(status='failed',errors=errors);result['cases'].append(row);save(target,result)
                print('KITCHEN_CANDIDATE_FAILED',rank,name,errors,flush=True);continue
            if a.profile:
                for _ in range(5):api()
                torch.cuda.synchronize();dist.barrier();torch.cuda.profiler.start()
                for _ in range(3):api()
                torch.cuda.synchronize();torch.cuda.profiler.stop()
            else:row['API']=measure(api,repeats=15)
            row['status']='complete';result['cases'].append(row);save(target,result)
            print('KITCHEN_CANDIDATE',rank,name,'complete',flush=True);del p,actual;gc.collect()
        del base,expected,original,reference;gc.collect();torch.cuda.empty_cache()
    result['status']='complete';save(target,result)
    all_results=[None,None];dist.all_gather_object(all_results,result)
    if rank==0 and not a.profile:
        winners={}
        for component in a.components:
            maps=[{r['name']:r for r in x['cases'] if r['component']==component and r['status']=='complete' and not r['config'].get('native')} for x in all_results]
            scores={name:max(m[name]['API']['gpu_median_ms'] for m in maps) for name in set(maps[0])&set(maps[1])}
            name=min(scores,key=scores.get);winners[component]=dict(name=name,slower_rank_API_ms=scores[name],all_slower_rank_API_ms=scores)
        save(a.output/'selection.json',dict(winners=winners,scope='Group32-only selection; Kitchen is a same-run comparator with different numerics.'))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
