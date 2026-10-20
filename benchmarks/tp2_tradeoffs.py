"""Bounded real-T4 numerical-mode and same-route implementation experiments.

No model construction or generation. Approximate candidates are diagnostic,
not quality-qualified, even if their independent implementation gates pass.
"""
import argparse
from contextlib import contextmanager
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import torch
import torch.distributed as dist
import comfy_kitchen as ck
from patches import tilelang_transport_tradeoffs as transport,tilelang_attention_tradeoffs as attn
from patches import tilelang_grouped_tp_optimized as prep_sparse,tilelang_grouped_tp_kitchen as dense
from patches import tilelang_grouped_tp_kitchen_v2 as dense_v2
from patches import tilelang_row_quant as row,tilelang_swiglu_pad as swiglu,optimized_tp as native,parallel_int8 as pi
from tp2_timing import measure,warm_device


@contextmanager
def label(name):
    torch.cuda.nvtx.range_push(name)
    try:yield
    finally:torch.cuda.nvtx.range_pop()


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def agree(error):
    errors=[None,None];dist.all_gather_object(errors,error);return errors


def byte_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def diagnostics(a,b):
    """Arithmetic diagnostics only; never a generation or sparse-quality gate."""
    delta=a.float()-b.float();denom=b.float().square().sum().clamp_min(1e-30)
    return dict(relative_RMSE=float((delta.square().sum()/denom).sqrt()),max_abs=float(delta.abs().max()),
                finite=bool(torch.isfinite(a).all()),bitwise_equal=byte_equal(a,b),
                scope='Numerical-mode arithmetic diagnostic; not generation acceptance.')


def timed(fn,args):
    if args.profile:
        warm_device();dist.barrier()
        for _ in range(10):fn()
        torch.cuda.synchronize();dist.barrier();torch.cuda.profiler.start()
        for _ in range(3):fn()
        torch.cuda.synchronize()
        with label('profile/rank_completion_barrier'):dist.barrier();torch.cuda.synchronize()
        torch.cuda.profiler.stop();return None
    return measure(fn,warmup=10,repeats=15)


def transport_reference(parts,scheme,k):
    if scheme=='int32':return parts[0]+parts[1]
    if scheme=='fp16':
        shift=transport.shift_for(k)
        a=(parts[0].float()*(2.**-shift)).half();b=(parts[1].float()*(2.**-shift)).half()
        return ((a.float()+b.float()).half().float()*(2.**shift)).int()
    results=[]
    for x in parts:
        values=x.float().reshape(x.shape[0],-1,128)
        scales=(values.abs().amax(-1)*torch.tensor(1./127.,device=x.device)).clamp_min(1e-8)
        q=(values/scales[...,None]).round().clamp(-127,127).to(torch.int8)
        results.append((q.float()*scales[...,None]).reshape_as(x))
    return (results[0]+results[1]).round().int()


def communication(args,rank,result,target):
    manifest=json.loads(args.manifest.read_text())
    targets=[r for r in manifest['GEMMs'] if r['component']=='dit' and r['partition']=='row']
    if args.profile:targets=[r for r in targets if r['module']=='fc2']
    for spec in targets:
        m,n,k=(spec[x] for x in ('M','N','K'));act=spec['input_act'];module=spec['module']
        torch.manual_seed(20261018+rank+(91 if act else 0))
        x=torch.randn(m,k*(2 if act else 1),device='cuda')*.25
        w=torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8);ws=torch.full((n,1),.001,device='cuda')
        assert native.STREAM_ROWS==0 and native.use_fallback_rotation(x,k*2,256)
        def quantization():
            with label('linear/activation_rotation_quantization'):
                return swiglu.global_row_quantize(x,dist,2) if act else row.global_row_quantize(x,dist,2,256,None,'packed')
        q,s=quantization();partial=native.native_int_mm(q,w)
        gathered=torch.empty((m*2,n),device='cuda',dtype=torch.int32);dist.all_gather_into_tensor(gathered,partial)
        parts=gathered.reshape(2,m,n);expected=pi.dequantize(parts[0]+parts[1],s,ws,torch.float32)
        variants=['int32','fp16','int8_gather']
        if args.round%2:variants.reverse()
        for scheme in variants:
            name='communication/'+module+'/'+scheme;record=dict(family='communication/'+module,name=scheme,target=spec,approximate=scheme!='int32')
            def reduction(value):
                with label('linear/transport_'+scheme):return transport.reduce(value,dist,scheme,k)
            def api():
                with label(name+'/API'):
                    aq,asc=quantization()
                    with label('linear/INT8_GEMM_to_INT32'):acc=native.native_int_mm(aq,w)
                    reduced=reduction(acc)
                    with label('linear/dequantize'):return pi.dequantize(reduced,asc,ws,torch.float32)
            error=None;start=time.perf_counter()
            try:
                actual_sum=reduction(partial.clone());manual=transport_reference(parts,scheme,k)
                assert torch.equal(actual_sum,manual),'Distributed codec does not match independently specified reference'
                actual=api();assert torch.isfinite(actual).all()
                record.update(independent_codec_gate='pass',arithmetic_diagnostics=diagnostics(actual,expected),
                              output_sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),
                              first_call_with_JIT_and_gate_seconds=time.perf_counter()-start,
                              fp16_power_of_two_shift=transport.shift_for(k),
                              local_payload_bytes=m*n*(2 if scheme=='fp16' else 4) if scheme!='int8_gather' else m*(n//128)*132,
                              collective='all_gather_uint8' if scheme=='int8_gather' else 'all_reduce_SUM')
                if scheme=='int32':assert byte_equal(actual,expected)
            except Exception as e:error=type(e).__name__+': '+str(e)
            errors=agree(error)
            if any(errors):
                record.update(status='failed',errors=errors);result['cases'].append(record);save(target,result);continue
            del actual_sum,manual,actual
            if not args.profile:
                # Standalone transport includes a preserving clone for all schemes;
                # full API above creates its partial naturally and has no such clone.
                record['transport_with_preserving_clone']=measure(lambda:reduction(partial.clone()),repeats=15)
            record['API']=timed(api,args);record['status']='complete'
            result['cases'].append(record);save(target,result);print('TRADEOFF',rank,name,'complete',flush=True)
        del x,w,ws,q,s,partial,gathered,parts,expected;gc.collect();torch.cuda.empty_cache()


def pv_reference(p,vi,vs,causal=False):
    """Small independent PyTorch implementation of Group32 QK / block-local INT8 PV."""
    # torch.exp2 in the pinned cu130 wheel invokes Jiterator/NVRTC, whose
    # builtins library is absent on this Kaggle image. exp is precompiled.
    def exp2(x):return torch.exp(x*math.log(2.))
    b,h,n,d=p.qi.shape;hk=p.ki.shape[1];nk=p.ki.shape[2];mapping=torch.arange(h,device='cuda')//(h//hk)
    out=torch.zeros(b,h,n,d,device='cuda');denom=torch.zeros(b,h,n,device='cuda');maximum=torch.full_like(denom,-float('inf'))
    for kb in range(nk//64):
        logits=torch.zeros(b,h,n,64,device='cuda')
        for group in range(d//32):
            dots=p.qi[...,group*32:(group+1)*32].float()@p.ki[:,mapping,kb*64:(kb+1)*64,group*32:(group+1)*32].float().transpose(-1,-2)
            logits+=dots*p.qs[...,group,None]*p.ks[:,mapping,kb*64:(kb+1)*64,group].unsqueeze(-2)*(d**-.5*1.4426950408889634)
        valid=torch.arange(kb*64,(kb+1)*64,device='cuda')[None,:]<p.keys
        if causal:valid=valid & (torch.arange(kb*64,(kb+1)*64,device='cuda')[None,:]<=torch.arange(n,device='cuda')[:,None])
        logits.masked_fill_(~valid,-float('inf'));local=logits.amax(-1);updated=torch.maximum(maximum,local)
        # Fully masked blocks are skipped in the independent reference. Valid
        # query rows in supported causal tile geometry always have a valid key.
        probability=torch.nan_to_num(exp2(logits-local[...,None]))
        alpha=torch.nan_to_num(exp2(maximum-updated));beta=torch.nan_to_num(exp2(local-updated))
        qi=(probability*127).round().clamp(0,127)
        product=qi@vi[:,mapping,kb*64:(kb+1)*64].float()
        out=out*alpha[...,None]+product*vs[:,mapping,kb].unsqueeze(-2)*(1./127.)*beta[...,None]
        denom=denom*alpha+probability.sum(-1)*beta;maximum=updated
    return (out/denom[...,None]).half()[:,:,:p.queries].transpose(1,2).reshape(b,p.queries,h*d)


def attention_units(rank,output):
    records=[];torch.manual_seed(818+rank)
    for h,hk,n,nk,d,causal in [(2,1,65,65,64,False),(2,1,97,97,128,True)]:
        q=torch.randn(1,h,n,d,device='cuda')*.15;k=torch.randn(1,hk,nk,d,device='cuda')*.15
        v=torch.randn(1,hk,nk,d,device='cuda')*.2
        p=dense.prepare(q,k,v,query_tile=32);vi,vs=attn.value_quantizer(1,hk,p.v.shape[2],d)(p.v)
        values=p.v.float().reshape(1,hk,-1,64,d)
        scales=(values.abs().amax(-2)*torch.tensor(1./127.,device='cuda')).clamp_min(1e-8)
        quant=(values/scales.unsqueeze(-2)).round().clamp(-127,127).to(torch.int8).reshape_as(vi)
        assert byte_equal(vs,scales) and byte_equal(vi,quant),'V quantizer identity failure'
        actual=attn.execute(p,causal=causal,pv_int8=True,value=vi,scales=vs)
        expected=pv_reference(p,vi,vs,causal)
        torch.testing.assert_close(actual,expected,rtol=.005,atol=.002)
        records.append(dict(test='independent_INT8_PV',shape=[h,hk,n,nk,d],causal=causal,gate='pass',diagnostic=diagnostics(actual,expected)))
    for kind in ('empty','full','alternating','random'):
        route=torch.randint(0,2,(1,2,3,107),device='cuda',dtype=torch.int32)
        if kind=='empty':route.zero_()
        elif kind=='full':route.fill_(1)
        elif kind=='alternating':route[:]=torch.arange(107,device='cuda').remainder(2)
        indices,lengths=attn.route_compactor(1,2,3,107)(route)
        expected=torch.where(route.bool(),torch.arange(107,device='cuda'),107).sort(-1).values.int()
        assert torch.equal(indices,expected) and torch.equal(lengths,route.sum(-1).int())
        records.append(dict(test='sorted_compaction',pattern=kind,gate='pass'))
    save(output/f'unit_rank{rank}.json',records);print('TRADEOFF_UNITS',rank,'pass',flush=True)


def attention(args,rank,result,target):
    if args.round==0 and not args.profile:attention_units(rank,args.output)
    for fixture in json.loads((args.data/'capture_manifest.json').read_text())['records']:
        component=fixture['component'];path=args.data/fixture['file']
        if args.profile and component=='te':continue
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==fixture['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        if component=='te':values=[data['q'][:,rank*32:(rank+1)*32],data['k'][:,rank*4:(rank+1)*4],data['v'][:,rank*4:(rank+1)*4]];offset=0
        elif component=='dit':values=[data[x][:,rank*28:(rank+1)*28] for x in ('q','k','v')];offset=0
        else:
            a,b=((0,1024),(1024,1797))[rank];values=[data['q'][:,:,a:b],data['k'],data['v']];offset=a
        original=[v.to(rank).contiguous() for v in values];del values,data;gc.collect()
        shape_spec=next(x for x in json.loads(args.manifest.read_text())['attention'] if x['component']==component and x['rank']==rank)
        assert list(original[0].shape)==shape_spec['q'] and list(original[1].shape)==shape_spec['k'], 'Fixture no longer matches current TP2 shapes'
        if component=='dit':
            b,h,n,d=original[2].shape;storage=torch.empty((b,n,3*h*d),device='cuda',dtype=original[2].dtype)
            view=storage.view(b,n,3,h,d)[:,:,2].permute(0,2,1,3);view.copy_(original[2]);original[2]=view
        modes=('dense','drop','scalar') if component=='dit' else ('dense',)
        for mode in modes:
            if args.profile and mode=='scalar':continue
            family=component+'/'+mode
            tile,cache,threads,warp=(64,'q',256,'row') if component=='dit' and mode=='dense' else ((64,'group',128,'square') if component=='video' or mode=='scalar' else (32,'group',128,'square'))
            variants=['current','pv8']+(['compact'] if mode=='drop' else [])
            if args.profile and mode=='drop':variants=['current','compact']
            if args.round%2:variants.reverse()
            def prepare():
                if mode=='dense':
                    if component=='video':return dense_v2.prepare(*original,query_tile=tile,mean_mode='castpad')
                    return dense.prepare(*original,query_tile=tile)
                return prep_sparse.prepare(*(v.half() for v in original),fused=True,mode=mode,query_tile=tile)
            def route(p):return prep_sparse.routes(p,874,fast_tau0=True) if mode!='dense' else None
            def raw(p,r,variant):
                with label(family+'/'+variant+'/QK_softmax_PV_and_output'):
                    if variant=='current':out=dense.execute(p,r,causal=component=='te',query_offset=offset,cache=cache,layout='BND',output_dtype=original[0].dtype,threads=threads,warp_policy=warp)
                    else:
                        vi=vs=lengths=None
                        if variant=='pv8':
                            with label('attention/value_INT8_quantization'):vi,vs=attn.value_quantizer(p.v.shape[0],p.v.shape[1],p.v.shape[2],p.v.shape[3])(p.v)
                        if variant=='compact':
                            with label('attention/sorted_route_compaction'):r,lengths=attn.route_compactor(*r.shape)(r)
                        out=attn.execute(p,r,causal=component=='te',query_offset=offset,cache=cache,output_dtype=original[0].dtype,
                            threads=threads,warp_policy=warp,pv_int8=variant=='pv8',compact_drop=variant=='compact',value=vi,scales=vs,lengths=lengths)
                    return torch.nan_to_num(out) if component=='video' else out
            base=prepare();br=route(base);expected=raw(base,br,'current');assert torch.isfinite(expected).all()
            for variant in variants:
                name=family+'/'+variant;record=dict(family=family,name=variant,approximate=variant=='pv8',q_shape=list(original[0].shape),
                    k_shape=list(original[1].shape),input_strides=[list(x.stride()) for x in original],schedule=dict(tile=tile,cache=cache,threads=threads,warp=warp),
                    quality_status='not evaluated: requires full H3 generation',route_policy='unchanged tau0, leaf64, prefix874')
                def api():
                    with label(name+'/API'):
                        with label(name+'/preparation'):p=prepare()
                        with label(name+'/routing'):r=route(p)
                        return raw(p,r,variant)
                start=time.perf_counter();error=None
                try:
                    p=prepare();r=route(p)
                    identity={key:byte_equal(getattr(p,key),getattr(base,key)) for key in ('qi','ki','qs','ks','v','counts')}
                    if mode!='dense':identity.update(routes=byte_equal(r,br))
                    assert all(identity.values()),identity
                    actual=raw(p,r,variant);assert torch.isfinite(actual).all()
                    record.update(operand_identity=identity,bitwise_equal_to_same_mode=byte_equal(actual,expected),output_sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),first_call_with_JIT_and_gate_seconds=time.perf_counter()-start)
                    if variant=='compact':assert byte_equal(actual,expected),'Sorted compaction changed the current drop operator'
                    if mode=='dense':record['arithmetic_diagnostics']=diagnostics(actual,expected)
                    # Sparse quality is not judged by dense-relative output errors.
                except Exception as e:error=type(e).__name__+': '+str(e)
                errors=agree(error)
                if any(errors):
                    record.update(status='failed',errors=errors);result['cases'].append(record);save(target,result);continue
                del p,r,actual
                record['API']=timed(api,args);record['status']='complete';result['cases'].append(record);save(target,result)
                print('TRADEOFF',rank,name,'complete',flush=True)
            del base,br,expected
        del original;gc.collect();torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['communication','attention'],required=True)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--data',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--round',type=int,default=0);ap.add_argument('--profile',action='store_true');args=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert os.environ.get('NCCL_P2P_DISABLE')=='1' and os.environ.get('NCCL_MIN_NCHANNELS')==os.environ.get('NCCL_MAX_NCHANNELS')=='4' and 'NCCL_PROTO' not in os.environ
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=90))
    for method in ('all_reduce','all_gather_into_tensor'):
        original=getattr(dist,method)
        def wrapped(*a,_method=method,_original=original,**kw):
            with label('NCCL/'+_method+'/'+str(a[0].dtype)+'/'+str(kw.get('op',''))):return _original(*a,**kw)
        setattr(dist,method,wrapped)
    pi.int_mm=native.native_int_mm;args.output.mkdir(parents=True,exist_ok=True);target=args.output/f'rank{rank}.json'
    result=dict(status='running',rank=rank,kind=args.kind,hardware=torch.cuda.get_device_name(rank),torch=torch.__version__,cases=[],scope='Operator-only. Approximate-mode implementation/performance diagnostics, no generation acceptance.')
    save(target,result)
    (communication if args.kind=='communication' else attention)(args,rank,result,target)
    result['status']='complete';save(target,result);gathered=[None,None];dist.all_gather_object(gathered,result)
    if rank==0 and not args.profile:
        summary={}
        for family in sorted({r['family'] for r in result['cases']}):
            maps=[{c['name']:c for c in r['cases'] if c['family']==family and c['status']=='complete'} for r in gathered]
            common=set(maps[0])&set(maps[1]);scores={v:max(m[v]['API']['gpu_median_ms'] for m in maps) for v in common}
            base='int32' if args.kind=='communication' else 'current'
            summary[family]=dict(slower_rank_API_ms=scores,reduction_vs_baseline_pct={v:100*(1-ms/scores[base]) for v,ms in scores.items()} if base in scores else {})
        save(args.output/'summary.json',summary)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
