"""Bounded, explicit physical-T4 operator gate and optional paired benchmark.

Never submits jobs, downloads weights, installs packages, or runs H3 generation.
Each invocation writes one durable result, including failure. Run per GPU.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import time
import traceback

import torch
import torch.nn.functional as F

from .attention import Backend,compact_reference,dequantized_operands,pooled_reference,token_mask,value_storage_indices
from .derive import verify_vendor


def error_metrics(actual,expected):
    delta=(actual.float()-expected.float())
    return dict(max_abs=float(delta.abs().max()),
        relative_l2=float(delta.norm()/expected.float().norm().clamp_min(1e-10)),
        byte_equal=torch.equal(actual,expected))


def gates(backend,device,deadline):
    rows=[]
    torch.manual_seed(8317)
    for d,n,nk,h,hk,dtype in ((64,65,131,4,2,torch.float16),
                             (128,129,257,4,2,torch.float16),
                             (128,65,1153,4,2,torch.float32)):
        if time.monotonic()>deadline: raise TimeoutError('Qualification deadline')
        q=torch.randn(1,h,n,d,device=device,dtype=dtype)
        k=torch.randn(1,hk,nk,d,device=device,dtype=dtype); v=torch.randn_like(k)
        default=backend.quantize(q,k,v)
        expected=backend.kitchen.int8_attention(q,k,v)
        baseline=backend.execute(default)
        # Same raw-input quantization AND same prequantized baseline are required.
        assert torch.equal(baseline,expected),'Native baseline differs from installed frozen Kitchen'
        dq,dk,dv=dequantized_operands(default)
        independent=F.scaled_dot_product_attention(dq,dk,dv,enable_gqa=True)
        torch.testing.assert_close(baseline.float(),independent,rtol=.02,atol=.015)
        rows.append(dict(kind='pinned_scale_producer_oracle',d=d,n=n,nk=nk,
                         **error_metrics(baseline,independent)))
        for variant in ('q64','cache_q') if d==128 and default.cta_k==128 else ('q64',):
            out=backend.execute(default,variant=variant)
            assert torch.equal(out,baseline),f'{variant} changed dense output bytes'
            rows.append(dict(kind='dense',d=d,n=n,nk=nk,variant=variant,**error_metrics(out,baseline)))
        p=backend.quantize(q,k,v,cta_k=64)
        routes=torch.rand(1,h,(n+63)//64,(nk+63)//64,device=device)>.4
        routes[...,0]=True
        # One genuinely empty query leaf exercises zero output behavior.
        routes[:,:,0,:]=False
        plan=backend.plan(routes,n,nk)
        reference_plan=compact_reference(routes.cpu(),n,nk)
        assert torch.equal(plan.counts.cpu(),reference_plan.counts)
        # Unused ID slots are unspecified; compare only initialized selected slots.
        active=torch.arange(plan.ids.shape[-1])[None,None,None,:]<reference_plan.counts[...,None]
        assert torch.equal(plan.ids.cpu()[active],reference_plan.ids[active])
        for causal,offset in ((False,0),(True,0),(True,1),(True,63)):
            out=backend.execute(p,plan=plan,is_causal=causal,query_offset=offset)
            mask=token_mask(reference_plan,is_causal=causal,query_offset=offset).to(device)
            expected=backend.kitchen.int8_attention_from_prequantized(replace(p,attn_mask=mask))
            torch.testing.assert_close(out,expected,rtol=.02,atol=.015)
            assert (out[:,:,0:64]==0).all(),'Empty route rows must be exactly zero'
            rows.append(dict(kind='drop',d=d,n=n,nk=nk,causal=causal,offset=offset,
                             **error_metrics(out,expected)))
        # Implicit dense causal path with arbitrary offset, without an NxN mask.
        for offset in (0,1,66):
            out=backend.execute(p,is_causal=True,query_offset=offset)
            mask=torch.arange(nk,device=device)[None,:]<=torch.arange(n,device=device)[:,None]+offset
            expected=backend.kitchen.int8_attention_from_prequantized(replace(p,attn_mask=mask[None,None].expand(1,h,n,nk)))
            torch.testing.assert_close(out,expected,rtol=.02,atol=.015)
            rows.append(dict(kind='causal',d=d,n=n,nk=nk,offset=offset,**error_metrics(out,expected)))
        # Fixed packed operands: changing masked future V cannot change earlier rows.
        changed_v=p.v.clone().reshape(1,hk,d,-1)
        locations=value_storage_indices(nk,p.q.device)
        changed_v[...,locations[31:]]=0
        changed=replace(p,v=changed_v.reshape_as(p.v))
        before=backend.execute(p,is_causal=True); after=backend.execute(changed,is_causal=True)
        assert torch.equal(before[:,:,:31],after[:,:,:31]),'Future packed V leaked through causal masking'
        # Pooled mode: density-one identity, mixed/empty fine state and ragged GQA.
        all_plan=backend.plan(torch.ones_like(routes),n,nk)
        pooled=backend.execute(p,plan=all_plan,pooled=True)
        assert torch.equal(pooled,backend.execute(p,plan=all_plan)),'Density-one pool changed output'
        qh,kh,vh=[t.cpu() for t in dequantized_operands(p)]
        expected,_=pooled_reference(qh,kh,vh,reference_plan,scale=p.attention_scale)
        out=backend.execute(p,plan=plan,pooled=True)
        torch.testing.assert_close(out.float().cpu(),expected,rtol=.03,atol=.02)
        rows.append(dict(kind='pooled',d=d,n=n,nk=nk,**error_metrics(out.cpu(),expected)))
        torch.cuda.synchronize(device)
    return rows


def paired_benchmark(backend,q,k,v,prefix,samples,deadline):
    cases={
        'kitchen':lambda:backend.kitchen.int8_attention(q,k,v),
        'native_baseline':lambda:backend.attention(q,k,v),
        'dense_q64':lambda:backend.attention(q,k,v,variant='q64'),
        'drop':lambda:backend.attention(q,k,v,mode='drop',prefix_tokens=prefix),
        'pooled':lambda:backend.attention(q,k,v,mode='pooled',prefix_tokens=prefix),
    }
    if q.shape[-1]==128 and k.shape[2]>1024:
        cases['dense_cache_q']=lambda:backend.attention(q,k,v,variant='cache_q')
    # Resource screen is bound to the exact library. Preserve baseline even if
    # it spills; withhold new spilling variants from performance selection.
    receipt_path=Path(backend.library).with_name('build_receipt.json')
    receipt=json.loads(receipt_path.read_text())
    assert receipt['artifact_sha256']==hashlib.sha256(Path(backend.library).read_bytes()).hexdigest()
    output_dtype='float16' if q.dtype==torch.float16 else 'bfloat16'
    key_tile=128 if q.shape[-1]==128 and k.shape[2]>1024 else 64
    pruned=[]
    for name in ('dense_q64','dense_cache_q'):
        if name not in cases: continue
        rows=[r for r in receipt['resources'] if r.get('head_dim')==q.shape[-1]
            and r.get('output_dtype')==output_dtype and r.get('key_tile')==key_tile
            and not r.get('pv_n8',False) and r.get('offset_bits',64)==64
            and not r.get('routed') and r.get('query_tile')==(64 if name=='dense_q64' else 128)
            and r.get('cache_q')==(name=='dense_cache_q')]
        if len(rows)!=1 or not rows[0]['spill_free']:
            pruned.append(dict(name=name,reason='CPU resource screen: spills or missing exact specialization',resources=rows))
            del cases[name]
    rows=[]
    order=list(cases); random.Random(718).shuffle(order)
    for round_id in (0,1):
        for name in order if round_id==0 else list(reversed(order)):
            fn=cases[name]
            if time.monotonic()>deadline: raise TimeoutError('Benchmark deadline')
            for _ in range(2): fn()
            torch.cuda.synchronize(q.device)
            torch.cuda.reset_peak_memory_stats(q.device)
            event_ms=[]; host_ms=[]
            for _ in range(samples):
                if time.monotonic()>deadline: raise TimeoutError('Benchmark deadline')
                start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                tick=time.perf_counter(); start.record(); out=fn(); end.record(); end.synchronize()
                host_ms.append((time.perf_counter()-tick)*1000); event_ms.append(start.elapsed_time(end))
                assert torch.isfinite(out).all(),'Nonfinite benchmark output'
                del out
            rows.append(dict(round=round_id,name=name,event_ms=event_ms,host_ms=host_ms,
                event_median_ms=statistics.median(event_ms),host_median_ms=statistics.median(host_ms),
                peak_allocated=torch.cuda.max_memory_allocated(q.device),
                scope='Complete API including quantization, routing/compaction/pooling and output cast'))
    return dict(cases=rows,pruned=pruned)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--tokens',type=int,default=6814,choices=(6814,51220))
    parser.add_argument('--capture',type=Path,help='Existing PRIVATE BHND Q/K/V capture; no download')
    parser.add_argument('--head-rank',type=int,choices=(0,1),default=0)
    parser.add_argument('--prefix-tokens',type=int,help='Actual packed layout prefix; required with capture benchmark')
    parser.add_argument('--samples',type=int,default=7)
    parser.add_argument('--deadline',type=int,default=240)
    args=parser.parse_args()
    if not 3<=args.samples<=15 or not 30<=args.deadline<=600:
        parser.error('Bounded samples 3..15 and deadline 30..600 seconds required')
    state=dict(status='started',scope='Operator-only qualification, no H3 pipeline',vendor=verify_vendor(),
        versions={name:importlib.metadata.version(name) for name in ('torch','comfy-kitchen')},
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    try:
        device=torch.device('cuda',args.device)
        torch.cuda.set_device(device)
        capability=torch.cuda.get_device_capability(device)
        assert capability==(7,5),'Physical SM75 required; no SM120 substitution'
        assert 'T4' in torch.cuda.get_device_name(device),'Physical Tesla T4 required'
        state.update(device=torch.cuda.get_device_name(device),capability=capability,cuda=torch.version.cuda)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False
        backend=Backend(args.library)
        deadline=started+args.deadline
        state['gates']=gates(backend,device,deadline)
        if args.benchmark:
            if args.capture:
                if args.prefix_tokens is None: raise ValueError('Real capture needs its actual prefix_tokens')
                data=torch.load(args.capture,map_location='cpu',weights_only=True)
                q,k,v=[data[name] for name in ('q','k','v')]
                assert q.ndim==4 and q.shape==k.shape==v.shape and q.shape[1]%2==0
                heads=q.shape[1]//2; sl=slice(args.head_rank*heads,(args.head_rank+1)*heads)
                q,k,v=[t[:,sl].contiguous().to(device) for t in (q,k,v)]
                prefix=args.prefix_tokens
                state.update(input_kind='private real capture head partition',
                    capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest())
            else:
                torch.manual_seed(310)
                q,k,v=[torch.randn(1,28,args.tokens,128,device=device,dtype=torch.float16) for _ in range(3)]
                prefix=args.prefix_tokens if args.prefix_tokens is not None else 874
                state['input_kind']='synthetic; not H3 trajectory performance evidence'
            state.update(shape=list(q.shape),prefix_tokens=prefix)
            state['benchmark']=paired_benchmark(backend,q,k,v,prefix,args.samples,deadline)
        state['status']='passed'
    except Exception:
        state.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        state['wall_seconds']=time.monotonic()-started
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n'); temporary.replace(args.output)


if __name__=='__main__':main()
