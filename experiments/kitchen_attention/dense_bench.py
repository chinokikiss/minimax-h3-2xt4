"""Bounded SM75 dense candidates and promoted TE dispatch; synthetic inputs."""
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback

import torch

from .attention import Backend
from .matrix import Shape,bnd,fixture,measure


def variants(d,ck):
    result=['baseline','q64','narrow']
    if d==128 and ck==128:
        result+=['cache_q','pv_n8','q64_pv_n8','pv_n8_narrow','q64_pv_n8_narrow']
    return result


def save(path,state):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)


def screen(receipt,shape,variant):
    ck=128 if shape.d==128 and shape.nk>1024 else 64
    rows=[r for r in receipt['resources'] if r.get('head_dim')==shape.d and
        r.get('query_tile')==(64 if variant.startswith('q64') else 128) and
        r.get('key_tile')==ck and r.get('output_dtype')==('float16' if shape.dtype=='float16' else 'bfloat16') and
        not r.get('routed') and r.get('cache_q')==(variant=='cache_q') and
        r.get('pv_n8',False)==('pv_n8' in variant) and
        r.get('offset_bits',64)==(32 if 'narrow' in variant else 64)]
    return dict(allowed=variant=='baseline' or (len(rows)==1 and rows[0]['spill_free']),resources=rows)


def candidate_gates(backend,rank):
    result={v:dict(status='passed',cases=[]) for v in variants(128,128)}
    for d,n,nk,dtype in ((64,65,131,torch.float16),(128,129,257,torch.float16),
                         (128,129,1153,torch.float16),(128,65,1153,torch.float32)):
        torch.manual_seed(8421)
        q=torch.randn(1,4,n,d,device=rank,dtype=dtype)
        k=torch.randn(1,2,nk,d,device=rank,dtype=dtype);v=torch.randn_like(k)
        p=backend.quantize(q,k,v);expected=backend.execute(p)
        assert torch.equal(expected,backend.kitchen.int8_attention(q,k,v))
        for name in variants(d,p.cta_k):
            try:
                out=backend.execute(p,variant=name)
                assert torch.equal(out,expected),'Dense candidate changed output bytes'
                result[name]['cases'].append(dict(d=d,n=n,nk=nk,dtype=str(dtype),byte_equal=True))
            except Exception as error:
                result[name].update(status='rejected',error=type(error).__name__+': '+str(error))
                if any(s in str(error).lower() for s in ('illegal memory','device-side assert','misaligned address')):raise
        torch.cuda.synchronize()
    return result


def te_gates(backend,rank):
    from types import SimpleNamespace
    from unittest.mock import patch
    from patches import h3_operator_policy as policy
    os.environ['H3_KITCHEN_ATTENTION_LIBRARY']=backend.library
    block=SimpleNamespace(operator_policy=policy.H3OperatorPolicy())
    rows=[]
    for dtype in (torch.float16,torch.float32):
        torch.manual_seed(399)
        q=torch.randn(1,32,353,128,device=rank,dtype=dtype)*.2
        k=torch.randn(1,4,353,128,device=rank,dtype=dtype)*.2;v=torch.randn_like(k)*.2
        mask=torch.full((353,353),torch.finfo(dtype).min/4,device=rank,dtype=dtype).triu(1)
        for name in ('plain','padding','additive_bias','none'):
            current=mask.clone()
            if name=='padding':current[:,0]=torch.finfo(dtype).min/4
            if name=='additive_bias':current[12,0]=.125
            if name=='none':current=None
            expected=bnd(backend.kitchen.int8_attention(q,k,v,attn_mask=current))
            with patch.object(Backend,'attention',side_effect=AssertionError('Default TE must retain the pinned Kitchen mask API')):
                actual=policy.te_attention(block,q,k,v,current)
            torch.testing.assert_close(actual,expected,rtol=.02,atol=.015)
            assert torch.equal(actual,expected),'Default TE did not retain the pinned API bytes'
            rows.append(dict(kind=name,dtype=str(dtype),max_abs=float((actual-expected).abs().max()),byte_equal=torch.equal(actual,expected)))
    return rows


def phase_measure(fn,samples=5):
    for _ in range(2):out=fn();del out
    torch.cuda.synchronize();event=[];wall=[]
    for _ in range(samples):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();a.record();out=fn();b.record();b.synchronize()
        event.append(a.elapsed_time(b));wall.append((time.perf_counter()-start)*1000);del out
    return dict(event_ms=event,wall_ms=wall)


def profile_once(fn):
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as p:
        out=fn();torch.cuda.synchronize();del out
    rows={}
    for event in p.events():
        if event.device_type==torch.autograd.DeviceType.CUDA:
            row=rows.setdefault(event.name,dict(name=event.name,count=0,cuda_us=0))
            row['count']+=1;row['cuda_us']+=event.device_time_total
    return sorted(rows.values(),key=lambda r:r['cuda_us'],reverse=True)


def run(library,output,rank,deadline_seconds=520):
    output=Path(output);torch.cuda.set_device(rank);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability(rank)==(7,5) and 'T4' in torch.cuda.get_device_name(rank)
    backend=Backend(library);receipt=json.loads(Path(library).with_name('build_receipt.json').read_text())
    state=dict(status='running',rank=rank,gpu=torch.cuda.get_device_name(rank),uuid=str(torch.cuda.get_device_properties(rank).uuid),
        scope='Seeded synthetic complete-API dense/TE timing. No H3 model, generation or communication.',cases=[],errors=[])
    start=time.monotonic();deadline=start+deadline_seconds;save(output,state)
    try:
        from .qualify import gates
        state['regression_gates']=gates(backend,torch.device('cuda',rank),deadline)
        state['candidate_gates']=candidate_gates(backend,rank)
        state['te_gates']=te_gates(backend,rank);save(output,state)
        shapes=[Shape('dit_short_512',28,28,512,512,128,'float16',packed_v=True),
                Shape('dit_6814',28,28,6814,6814,128,'float16',packed_v=True),
                Shape('dit_long_51220',28,28,51220,51220,128,'float16',packed_v=True),
                Shape('video_short_d64',32,32,1024 if rank==0 else 773,1797,64,'float16')]
        for shape in shapes:
            q,k,v=fixture(shape,rank);ck=128 if shape.d==128 and shape.nk>1024 else 64
            cases={'kitchen_int8':lambda:bnd(backend.kitchen.int8_attention(q,k,v))};pruned=[]
            for variant in variants(shape.d,ck):
                resource=screen(receipt,shape,variant)
                gate=state['candidate_gates'][variant]
                if resource['allowed'] and gate['status']=='passed':
                    cases['native_'+variant]=lambda variant=variant:bnd(backend.attention(q,k,v,variant=variant))
                else:pruned.append(dict(variant=variant,gate=gate,resource=resource))
            row=dict(shape=asdict(shape),candidates=[],pruned=pruned);state['cases'].append(row);save(output,state)
            order=list(cases);random.Random(8529).shuffle(order)
            for rnd in (0,1):
                for name in order if rnd==0 else reversed(order):
                    print('DENSE_CASE',rank,shape.name,rnd,name,flush=True)
                    timing=measure(cases[name],shape,samples=7,deadline=deadline)
                    row['candidates'].append(dict(name=name,round=rnd,status='passed',**timing));save(output,state)
            p=backend.quantize(q,k,v)
            row['phases']={'quantize':phase_measure(lambda:backend.quantize(q,k,v))}
            for variant in ('baseline','narrow','pv_n8_narrow'):
                if 'native_'+variant in cases:
                    row['phases']['prepared_'+variant]=phase_measure(lambda variant=variant:backend.execute(p,variant=variant))
            prepared=backend.execute(p)
            row['phases']['bnd_copy']=phase_measure(lambda:bnd(prepared));del prepared,p
            if shape.name=='dit_6814':
                row['profiles']={}
                for name in ('kitchen_int8','native_baseline','native_narrow','native_pv_n8_narrow'):
                    if name not in cases:continue
                    try:row['profiles'][name]=dict(status='collected',kernels=profile_once(cases[name]))
                    except Exception as error:row['profiles'][name]=dict(status='unavailable',error=str(error))
            save(output,state)
            del cases,q,k,v;gc.collect();torch.cuda.empty_cache()
        # Measure the final direct default without mask classification.
        from types import SimpleNamespace
        from patches import h3_operator_policy as policy
        shape=Shape('te_default_causal_353',32,4,353,353,128,'float32',causal=True)
        q,k,v=fixture(shape,rank);block=SimpleNamespace(operator_policy=policy.H3OperatorPolicy())
        with torch.inference_mode():
            mask=torch.full((353,353),torch.finfo(q.dtype).min/4,device=rank,dtype=q.dtype).triu(1)
            cases={'kitchen_int8':lambda:bnd(backend.kitchen.int8_attention(q,k,v,attn_mask=mask)),
                   'native_causal_direct':lambda:bnd(backend.attention(q,k,v,is_causal=True)),
                   'te_kitchen_default':lambda:policy.te_attention(block,q,k,v,mask)}
            row=dict(shape=asdict(shape),candidates=[],pruned=[]);state['cases'].append(row)
            for rnd in (0,1):
                for name in cases if rnd==0 else reversed(cases):
                    row['candidates'].append(dict(name=name,round=rnd,status='passed',
                        **measure(cases[name],shape,samples=7,deadline=deadline)));save(output,state)
        state['status']='completed'
    except Exception:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        state['seconds']=time.monotonic()-start;save(output,state)
    return state
