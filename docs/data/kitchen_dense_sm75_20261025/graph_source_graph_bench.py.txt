"""Bounded dual-T4 probe; graph copies and independent output are timed."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import ModuleType

import torch


def save(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)


def worker(rank,output):
    # The frozen r5 capsule is reused unchanged. The graph and final TE adapter
    # are explicitly supplied, hash-verified supplemental source modules.
    from experiments.kitchen_attention.matrix import Shape,fixture,bnd,measure
    import comfy_kitchen as ck
    root=Path(__file__).parent
    graph=ModuleType('supplemental_graph');exec(compile((root/'graph.py').read_bytes(),str(root/'graph.py'),'exec'),graph.__dict__)
    te=ModuleType('supplemental_te');exec(compile((root/'kitchen_te_attention.py').read_bytes(),str(root/'kitchen_te_attention.py'),'exec'),te.__dict__)
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability(rank)==(7,5) and 'T4' in torch.cuda.get_device_name(rank)
    state=dict(status='running',rank=rank,gpu=torch.cuda.get_device_name(rank),uuid=str(torch.cuda.get_device_properties(rank).uuid),
        cases=[],gates=[],scope='Synthetic complete API; input copies, replay and independent output clone included. No H3 generation.')
    deadline=time.monotonic()+250
    try:
        with torch.inference_mode():
            # Non-square/GQA, packed-value geometry and changed input replay.
            for shape in [Shape('ragged_gqa',4,2,65,131,128,'float16'),
                          Shape('dit_short_512',28,28,512,512,128,'float16',packed_v=True),
                          Shape('dit_6814',28,28,6814,6814,128,'float16',packed_v=True),
                          Shape('video_d64',32,32,1024 if rank==0 else 773,1797,64,'float16')]:
                q,k,v=fixture(shape,rank)
                row=dict(shape=asdict(shape),candidates=[]);state['cases'].append(row)
                floor=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
                try:
                    plan=graph.KitchenGraphPlan(q,k,v)
                    row['plan']=dict(capture_ms=plan.capture_ms,persistent_allocated=torch.cuda.memory_allocated()-floor,
                        reserved_growth=torch.cuda.memory_reserved()-reserved)
                    expected=bnd(ck.int8_attention(q,k,v));actual=plan(q,k,v)
                    assert torch.equal(actual,expected),'Initial replay differs'
                    held=actual.clone();q.mul_(.875);k.add_(.03125);v.mul_(1.125)
                    second=plan(q,k,v);expected=bnd(ck.int8_attention(q,k,v))
                    assert torch.equal(second,expected),'Changed inputs were not replayed exactly'
                    assert torch.equal(actual,held),'A later replay overwrote an earlier result'
                    state['gates'].append(dict(shape=shape.name,byte_equal=True,changed_inputs=True,independent_output=True))
                    del actual,held,second,expected
                    cases={'kitchen_int8':lambda:bnd(ck.int8_attention(q,k,v)), 'kitchen_graph':lambda:plan(q,k,v)}
                    for rnd in (0,1):
                        for name in cases if rnd==0 else reversed(cases):
                            row['candidates'].append(dict(name=name,round=rnd,status='passed',
                                **measure(cases[name],shape,samples=11,deadline=deadline)))
                    del plan,cases
                except Exception:
                    row['status']='rejected';row['error']=traceback.format_exc()
                    if any(x in row['error'].lower() for x in ('illegal memory','device-side assert')):raise
                save(output,state)
                del q,k,v;gc.collect();torch.cuda.empty_cache()
            for dtype in ('float16','float32'):
                shape=Shape('te_direct_353_'+dtype,32,4,353,353,128,dtype,causal=True)
                q,k,v=fixture(shape,rank)
                mask=torch.full((353,353),torch.finfo(q.dtype).min/4,device=q.device,dtype=q.dtype).triu(1)
                row=dict(shape=asdict(shape),candidates=[]);state['cases'].append(row)
                expected=bnd(ck.int8_attention(q,k,v,attn_mask=mask));actual=te.attention(q,k,v,mask)
                assert torch.equal(expected,actual);del expected,actual
                cases={'kitchen_int8':lambda:bnd(ck.int8_attention(q,k,v,attn_mask=mask)),
                       'te_direct_default':lambda:te.attention(q,k,v,mask)}
                for rnd in (0,1):
                    for name in cases if rnd==0 else reversed(cases):
                        row['candidates'].append(dict(name=name,round=rnd,status='passed',
                            **measure(cases[name],shape,samples=11,deadline=deadline)))
                del cases,q,k,v,mask;gc.collect();torch.cuda.empty_cache();save(output,state)
        state['status']='completed'
    except Exception:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:save(output,state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.rank is not None:return worker(args.rank,args.output)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.device_count()==2
    state=dict(status='running',torch=torch.__version__,cuda=torch.version.cuda,
        packages={n:importlib.metadata.version(n) for n in ('comfy-kitchen','tilelang','numpy')},ranks=[])
    assert state['packages']['comfy-kitchen']=='0.2.37' and state['packages']['tilelang']=='0.1.13'
    procs=[];logs=[];start=time.monotonic()
    try:
        for rank in (0,1):
            log=(out/f'rank{rank}.log').open('w');logs.append(log)
            procs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__)), '--rank',str(rank),'--output',str(out/f'rank{rank}.json')],stdout=log,stderr=subprocess.STDOUT))
        for proc in procs:assert proc.wait(timeout=max(1,300-(time.monotonic()-start)))==0,'Graph worker failed'
        state['ranks']=[json.loads((out/f'rank{rank}.json').read_text()) for rank in (0,1)]
        state['status']='completed'
    finally:
        for proc in procs:
            if proc.poll() is None:proc.kill();proc.wait()
        for log in logs:log.close()
        state['seconds']=time.monotonic()-start;save(out/'batch_results.json',state)


if __name__=='__main__':main()
