"""Matched complete-API T4 comparisons; synthetic operator evidence only."""
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

import torch

from .attention import Backend,routing


@dataclass(frozen=True)
class Shape:
    name:str
    h:int
    hk:int
    n:int
    nk:int
    d:int
    dtype:str
    prefix:int=0
    causal:bool=False
    sparse:bool=False
    packed_v:bool=False


def shapes(rank):
    return [Shape('te_causal_353',32,4,353,353,128,'float32',causal=True),
        Shape('dit_short_512',28,28,512,512,128,'float16',64,sparse=True,packed_v=True),
        Shape('dit_6814',28,28,6814,6814,128,'float16',874,sparse=True,packed_v=True),
        Shape('dit_long_51220',28,28,51220,51220,128,'float16',874,sparse=True,packed_v=True),
        Shape('video_short_d64',32,32,1024 if rank==0 else 773,1797,64,'float16')]


def bnd(x):
    return x.transpose(1,2).reshape(x.shape[0],x.shape[2],-1) if x.ndim==4 else x


def fixture(shape,device):
    torch.manual_seed(20261025)
    dtype=getattr(torch,shape.dtype)
    q=torch.randn(1,shape.h,shape.n,shape.d,device=device,dtype=dtype)*.2
    k=torch.randn(1,shape.hk,shape.nk,shape.d,device=device,dtype=dtype)*.2
    v=torch.randn_like(k)*.2
    if shape.packed_v:
        storage=torch.empty((1,shape.nk,3*shape.h*shape.d),device=device,dtype=dtype)
        packed=storage.view(1,shape.nk,3,shape.h,shape.d)[:,:,2].permute(0,2,1,3)
        packed.copy_(v); v=packed
    return q,k,v


def candidate_screen(library,shape,name):
    receipt=json.loads(Path(library).with_name('build_receipt.json').read_text())
    assert receipt['artifact_sha256']==hashlib.sha256(Path(library).read_bytes()).hexdigest()
    key=128 if shape.d==128 and shape.nk>1024 else 64
    out='float16' if shape.dtype=='float16' else 'bfloat16'
    rows=[r for r in receipt['resources'] if r.get('head_dim')==shape.d
        and r.get('key_tile')==key and r.get('output_dtype')==out and not r.get('routed')
        and not r.get('pv_n8',False) and r.get('offset_bits',64)==64
        and r.get('query_tile')==(64 if name=='q64' else 128) and r.get('cache_q')==(name=='cache_q')]
    return dict(allowed=len(rows)==1 and rows[0]['spill_free'],resources=rows)


def functions(shape,q,k,v,backend):
    import comfy_kitchen as ck
    from patches import h3_operator_policy as policy
    from patches import tilelang_grouped_tp_kitchen_v2 as dense
    mask=torch.ones(shape.n,shape.nk,device=q.device,dtype=torch.bool).tril() if shape.causal else None
    cases={'kitchen_int8':lambda:bnd(ck.int8_attention(q,k,v,attn_mask=mask))}
    if shape.causal:
        block=SimpleNamespace(operator_policy=policy.H3OperatorPolicy(te_attention='tilelang_group32'))
        cases['tilelang_group32_dense']=lambda:bnd(policy.te_attention(block,q,k,v,mask))
    else:
        def group32():
            p=dense.prepare(q,k,v,query_tile=64)
            return dense.execute(p,cache='q' if shape.d==128 else 'group',layout='BND',
                output_dtype=q.dtype,threads=128,warp_policy='row' if shape.d==128 else 'square')
        cases['tilelang_group32_dense']=group32
    if shape.sparse:
        options={'minimax_h3_layout':SimpleNamespace(seq_len=shape.n,
            segments=[(0,shape.prefix,'text'),(shape.prefix,shape.n,'video')])}
        for name in ('tilelang_drop','tilelang_scalar'):
            block=SimpleNamespace(operator_policy=policy.H3OperatorPolicy(dit_attention=name))
            cases[name]=lambda block=block:policy.dit_attention(block,q,k,v,options)
    pruned=[]
    if backend is not None:
        cases['native_causal' if shape.causal else 'native_dense_baseline']=lambda:bnd(
            backend.attention(q,k,v,is_causal=shape.causal))
        if not shape.causal:
            for variant in ('q64','cache_q') if shape.d==128 and shape.nk>1024 else ('q64',):
                screen=candidate_screen(backend.library,shape,variant)
                if screen['allowed']:
                    cases['native_dense_'+variant]=lambda variant=variant:bnd(backend.attention(q,k,v,variant=variant))
                else:pruned.append(dict(name='native_dense_'+variant,reason='Spill/missing-specialization screen',**screen))
        if shape.sparse:
            for mode in ('drop','pooled'):
                cases['native_'+mode]=lambda mode=mode:bnd(backend.attention(q,k,v,mode=mode,prefix_tokens=shape.prefix))
    return cases,pruned


def check_layout(shape,out):
    expected=(1,shape.n,shape.h*shape.d)
    if out.shape!=expected or out.dtype!=getattr(torch,shape.dtype):
        raise AssertionError(f'Output boundary mismatch: {tuple(out.shape)}, {out.dtype}, expected {expected}/{shape.dtype}')
    if not torch.isfinite(out).all():raise AssertionError('Nonfinite output')


def measure(fn,shape,*,samples=5,warmups=2,deadline):
    if time.monotonic()>deadline:raise TimeoutError('Matrix deadline')
    tick=time.perf_counter(); out=fn(); torch.cuda.synchronize()
    first_ms=(time.perf_counter()-tick)*1000; check_layout(shape,out); del out
    for _ in range(warmups):out=fn(); del out
    torch.cuda.synchronize()
    floor=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    event=[];wall=[]
    for _ in range(samples):
        if time.monotonic()>deadline:raise TimeoutError('Matrix deadline')
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        tick=time.perf_counter();start.record();out=fn();end.record();end.synchronize()
        event.append(start.elapsed_time(end));wall.append((time.perf_counter()-tick)*1000)
        del out
    return dict(first_including_jit_ms=first_ms,event_ms=event,wall_ms=wall,
        event_median_ms=statistics.median(event),wall_median_ms=statistics.median(wall),
        baseline_allocated=floor,peak_allocated=torch.cuda.max_memory_allocated(),
        peak_temporary=torch.cuda.max_memory_allocated()-floor)


def run_matrix(library,output,rank,native_ok,deadline_seconds=780):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(rank);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==2 and torch.cuda.get_device_capability(rank)==(7,5)
    assert 'T4' in torch.cuda.get_device_name(rank)
    backend=Backend(library) if native_ok else None
    state=dict(status='running',rank=rank,gpu=torch.cuda.get_device_name(rank),
        uuid=str(torch.cuda.get_device_properties(rank).uuid),native_qualified=native_ok,
        input_kind='Seeded synthetic, matched raw Q/K/V across APIs; no model trajectory',
        output_boundary='BND / original input dtype for every backend',cases=[],errors=[])
    deadline=time.monotonic()+deadline_seconds
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(output)
    save()
    try:
        # Fixed tiny warmup outside the comparison, not a model execution.
        x=torch.randn(512,512,device=rank,dtype=torch.float16)
        stop=time.monotonic()+1
        while time.monotonic()<stop:torch.mm(x,x);torch.cuda.synchronize()
        del x
        for shape in shapes(rank):
            q,k,v=fixture(shape,rank)
            cases,pruned=functions(shape,q,k,v,backend)
            row=dict(shape=shape.__dict__,strides=[list(t.stride()) for t in (q,k,v)],
                candidates=[],pruned=pruned,scope='Complete API including preparation, routing, compaction, pooling, BND conversion')
            if shape.sparse:
                routes=routing(q,k,shape.prefix)
                row['retained_block_fraction']=float(routes.float().mean())
                counts=(shape.n-torch.arange(routes.shape[-1],device=rank)*64).clamp(0,64)
                row['retained_token_pair_fraction']=float((routes.float()*counts[None,None,:,None]*counts[None,None,None,:]).sum()/(shape.h*shape.n**2))
                workspace=shape.h*((shape.n+63)//64*64)*((shape.nk+63)//64)*4
                row['tilelang_scalar_centroid_score_workspace_bytes']=workspace
                row['tilelang_scalar_dispatch']='split' if workspace<=96*2**20 else 'inline'
                del routes
            state['cases'].append(row);save()
            order=list(cases);random.Random(8251).shuffle(order)
            failed=set()
            for round_id in (0,1):
                for name in order if round_id==0 else list(reversed(order)):
                    if name in failed:continue
                    print('MATRIX_CASE',rank,shape.name,round_id,name,flush=True)
                    try:
                        result=measure(cases[name],shape,deadline=deadline)
                        row['candidates'].append(dict(name=name,round=round_id,status='passed',**result))
                    except Exception as error:
                        failed.add(name)
                        failure=dict(shape=shape.name,name=name,round=round_id,error=type(error).__name__+': '+str(error))
                        row['candidates'].append(dict(status='failed',**failure));state['errors'].append(failure)
                        # Illegal access poisons this CUDA context; do not keep using it.
                        if any(s in str(error).lower() for s in ('illegal memory','device-side assert','misaligned address')):raise
                    save()
            del cases,q,k,v;gc.collect();torch.cuda.empty_cache()
        state['status']='completed' if not state['errors'] else 'completed_with_rejections'
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:save()
    return state
