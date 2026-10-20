"""Small causal matrix: fixed Group32 QK, INT8 PV restoration and scalar summaries."""
import argparse
import datetime
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import torch
import torch.distributed as dist
import tp2_tradeoffs as base
from patches import tilelang_attention_p2 as candidate


# Independent Torch arithmetic, derived transparently from the validated
# reference; none of these expressions invoke a candidate kernel.
ref_source=inspect.getsource(base.pv_reference).replace('def pv_reference(p,vi,vs,causal=False):','def reference(p,vi,vs,causal=False,restore_unit=False,running_max=False):')
ref_source=ref_source.replace('        valid=torch.arange', '        valid=torch.arange')
ref_source=ref_source.replace('        # Fully masked blocks', '        if running_max:local=updated\n        # Fully masked blocks')
ref_source=ref_source.replace('out=out*alpha[...,None]+product*vs[:,mapping,kb].unsqueeze(-2)*(1./127.)*beta[...,None]',
                              'out=out*alpha[...,None]+(product*(1./127.) if restore_unit else product*vs[:,mapping,kb].unsqueeze(-2)*(1./127.)*beta[...,None])')
exec(compile(ref_source,__file__,'exec'),dict(torch=torch,math=__import__('math')),ref_namespace:={})
reference=ref_namespace['reference']


def centroid_scores(p,r=None,omitted=False):
    b,h,np,d=p.q.shape;hk=p.ki.shape[1]
    _,_,di,_=candidate.geometry(b,h,hk,p.queries,p.keys,d,str(p.qi.device))
    return candidate.centroid_scores(b,h,np,p.kc.shape[2],d,omitted=omitted)(p.q,p.kc,r if omitted else di)


def record_compilation(output,rank):
    """Archive actual Linux/T4 JIT sources when the pinned API exposes them."""
    import tilelang
    original=tilelang.compile;records=[];folder=output/'compiled'/f'rank{rank}';folder.mkdir(parents=True,exist_ok=True)
    def compile_recorded(fn,**kw):
        caller=inspect.currentframe().f_back.f_code.co_name
        result=original(fn,**kw);row=dict(factory=caller)
        try:
            source=result.get_kernel_source();key=hashlib.sha256(source.encode()).hexdigest();path=folder/(caller+'_'+key[:12]+'.cu');path.write_text(source)
            row.update(source=path.name,source_sha256=key)
            if hasattr(result,'export_ptx'):result.export_ptx(str(path.with_suffix('.ptx')))
            if hasattr(result,'resource_usage'):row['resources']=str(result.resource_usage)
        except Exception as error:row['export_error']=type(error).__name__+': '+str(error)
        records.append(row);base.save(folder/'manifest.json',records)
        return result
    tilelang.compile=compile_recorded


def units():
    records=[];torch.manual_seed(818)
    q=torch.randn(1,2,65,128,device='cuda')*.15;k=torch.randn_like(q)*.15;v=torch.randn_like(q)*.2
    p=base.dense.prepare(q,k,v,query_tile=32);vi,vs=candidate.value_quantizer(1,2,p.v.shape[2],128)(p.v)
    for restore_unit,running_max in ((False,False),(True,False),(False,True)):
        y=candidate.execute(p,pv_int8=True,value=vi,scales=vs,restore_unit=restore_unit,running_max=running_max)
        expected=reference(p,vi,vs,restore_unit=restore_unit,running_max=running_max)
        torch.testing.assert_close(y,expected,rtol=.005,atol=.002)
        records.append(dict(test='independent_PV',unit_restore=restore_unit,running_max=running_max,status='pass'))
    p=base.prep_sparse.prepare(q.half(),k.half(),v.half(),mode='scalar',fused=True,query_tile=64)
    scores=centroid_scores(p)
    expected=(p.q.float().unsqueeze(-2)*p.kc.float().unsqueeze(-3)).sum(-1)
    torch.testing.assert_close(scores,expected,rtol=1e-4,atol=1e-5)
    route=torch.zeros(1,2,2,2,device='cuda',dtype=torch.int32);route[...,0]=1
    omitted_scores=centroid_scores(p,route,True)
    torch.testing.assert_close(omitted_scores[...,1],scores[...,1],rtol=1e-4,atol=1e-5)
    for pv8,omitted in ((False,False),(True,False),(False,True),(True,True)):
        vi,vs=candidate.value_quantizer(1,2,p.v.shape[2],128)(p.v)
        old=base.attn.execute(p,route,pv_int8=pv8,value=vi,scales=vs)
        y=candidate.execute(p,route,pv_int8=pv8,value=vi,scales=vs,centroids=omitted_scores if omitted else scores)
        torch.testing.assert_close(y,old,rtol=.005,atol=.002)
        records.append(dict(test='same_route_scalar_split',pv8=pv8,omitted_only=omitted,status='pass',bitwise_equal=base.byte_equal(y,old)))
    return records


def run(args,rank,result,target):
    if args.round==0 and not args.profile and not args.restore_only and not args.scalar_profile_only:base.save(args.output/f'unit_rank{rank}.json',units())
    fixture=next(f for f in json.loads((args.data/'capture_manifest.json').read_text())['records'] if f['component']=='dit')
    path=args.data/fixture['file'];assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==fixture['sha256']
    data=torch.load(path,map_location='cpu',weights_only=True)
    original=[data[x][:,rank*28:(rank+1)*28].to(rank).contiguous() for x in ('q','k','v')];del data;gc.collect()
    b,h,n,d=original[2].shape;storage=torch.empty((b,n,3*h*d),device='cuda',dtype=original[2].dtype)
    view=storage.view(b,n,3,h,d)[:,:,2].permute(0,2,1,3);view.copy_(original[2]);original[2]=view
    for mode in (('scalar',) if args.scalar_profile_only else ('dense',) if args.restore_only else ('dense','scalar')):
        variants=['current','fp16_128','pv8_128','pv8_unit_restore'] if mode=='dense' else ['current','pv8','pooled_scores','pooled_omitted','pv8_pooled_omitted']
        if args.restore_only:variants=['current','pv8','pv8_unit_restore']
        if args.scalar_profile_only:variants=['current','pooled_scores','pooled_omitted']
        if args.round%2:variants.reverse()
        tile,cache,threads,warp=(64,'q',256,'row') if mode=='dense' else (64,'group',128,'square')
        def prepare():
            return base.dense.prepare(*original,query_tile=tile) if mode=='dense' else base.prep_sparse.prepare(*(v.half() for v in original),fused=True,mode='scalar',query_tile=tile)
        def routing(p):return base.prep_sparse.routes(p,874,fast_tau0=True) if mode=='scalar' else None
        p=prepare();routes=routing(p)
        vi,vs=candidate.value_quantizer(b,h,p.v.shape[2],d)(p.v)
        cs=centroid_scores(p) if mode=='scalar' else None
        omitted_cs=centroid_scores(p,routes,True) if mode=='scalar' else None
        expected=base.dense.execute(p,routes,cache=cache,layout='BND',output_dtype=torch.float32,threads=threads,warp_policy=warp)
        pv_expected=expected if args.scalar_profile_only else base.attn.execute(p,routes,cache=cache,output_dtype=torch.float32,threads=threads,warp_policy=warp,pv_int8=True,value=vi,scales=vs)
        for variant in variants:
            name='attention/'+mode+'/'+variant;pv8=variant.startswith('pv8');pooled='pooled_' in variant;omitted='omitted' in variant
            record=dict(family='attention/'+mode,name=variant,q_shape=list(original[0].shape),route_policy='unchanged tau0 leaf64 prefix874',
                diagnostic_only=variant=='pv8_unit_restore',approximate=pv8,quality_status='Unqualified: no full-model generation test.',
                schedule=dict(tile=tile,cache=cache,threads=128 if variant in ('pv8_128','fp16_128') else threads,warp=warp),
                additional_centroid_workspace_bytes=cs.numel()*4 if pooled else 0)
            def core(pp,rr,vv,ss,cc):
                with base.label(name+'/QK_softmax_PV'):
                    if variant in ('current','fp16_128'):return base.dense.execute(pp,rr,cache=cache,layout='BND',output_dtype=torch.float32,threads=128 if variant=='fp16_128' else threads,warp_policy=warp)
                    return candidate.execute(pp,rr,cache=cache,output_dtype=torch.float32,threads=128 if variant=='pv8_128' else threads,
                        warp_policy=warp,pv_int8=pv8,value=vv,scales=ss,centroids=cc if pooled else None,
                        restore_unit=variant=='pv8_unit_restore',running_max=variant=='pv8_running')
            def api():
                with base.label(name+'/API'):
                    with base.label('attention/preparation'):pp=prepare()
                    with base.label('attention/routing'):rr=routing(pp)
                    vv=ss=cc=None
                    if pv8:
                        with base.label('attention/value_INT8_quantization'):vv,ss=candidate.value_quantizer(b,h,pp.v.shape[2],d)(pp.v)
                    if pooled:
                        with base.label('attention/batched_centroid_scores'):cc=centroid_scores(pp,rr,omitted)
                    return core(pp,rr,vv,ss,cc)
            error=None;start=time.perf_counter()
            try:
                prepared=prepare();rr=routing(prepared)
                identity={key:base.byte_equal(getattr(prepared,key),getattr(p,key)) for key in ('qi','ki','qs','ks','v','counts')}
                if mode=='scalar':identity.update(routes=base.byte_equal(rr,routes))
                assert all(identity.values()),identity
                y=core(p,routes,vi,vs,omitted_cs if omitted else cs);assert torch.isfinite(y).all()
                if variant not in ('pv8_running','pv8_unit_restore'):
                    torch.testing.assert_close(y,pv_expected if pv8 else expected,rtol=.005,atol=.002)
                    record['same_method_implementation_gate']='pass'
                else:record['implementation_gate']='independent_small_reference_pass; unit_restore is diagnostic only'
                record.update(operand_identity=identity,output_sha256=hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest(),
                    same_method_bitwise_equal=base.byte_equal(y,pv_expected if pv8 else expected),first_call_with_JIT_and_gate_seconds=time.perf_counter()-start)
                if mode=='dense' and variant!='pv8_unit_restore':record['arithmetic_diagnostics']=base.diagnostics(y,expected)
                del prepared,rr,y
            except Exception as e:error=type(e).__name__+': '+str(e)
            errors=base.agree(error)
            if any(errors):
                record.update(status='failed',errors=errors);result['cases'].append(record);base.save(target,result);continue
            if not args.profile:
                # Core timing excludes V quantization and centroid preparation, preserving the same operands.
                record['prepared_main']=base.measure(lambda:core(p,routes,vi,vs,omitted_cs if omitted else cs),warmup=10,repeats=15)
            record['API']=base.timed(api,args);record['status']='complete';result['cases'].append(record);base.save(target,result)
            print('P2_ATTENTION',rank,mode,variant,'complete',flush=True)
        del p,routes,vi,vs,cs,omitted_cs,expected,pv_expected;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--round',type=int,default=0);ap.add_argument('--profile',action='store_true');ap.add_argument('--restore-only',action='store_true');ap.add_argument('--scalar-profile-only',action='store_true');args=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert os.environ.get('NCCL_P2P_DISABLE')=='1' and os.environ.get('NCCL_MIN_NCHANNELS')==os.environ.get('NCCL_MAX_NCHANNELS')=='4' and 'NCCL_PROTO' not in os.environ
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=120))
    args.output.mkdir(parents=True,exist_ok=True);record_compilation(args.output,rank)
    target=args.output/f'rank{rank}.json';result=dict(status='running',rank=rank,cases=[],hardware=torch.cuda.get_device_name(),torch=torch.__version__)
    base.save(target,result);run(args,rank,result,target);result['status']='complete';base.save(target,result)
    dist.barrier();dist.destroy_process_group()
