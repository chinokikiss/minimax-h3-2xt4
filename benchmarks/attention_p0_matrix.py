"""Controlled real-capture sequence/density replay; physical SM75 only."""
import argparse
import cProfile
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
from real_t4_attention_study import hardware,load,measure,label,raw_args,save
from patches.tilelang_sparse_attention import prepare,make_routes,get_kernel
from patches.tilelang_sage_preparation import prepare_sage_fused
from patches.tilelang_scalar_pool import scalar_pool_kernel


def summaries(p,k):
    b,h,np,d=p.q.shape;nb=np//64;n=p.tokens
    # One shared QK quantization for dense, dropping and pooling. Summaries are
    # additional sparse work, not silently charged to the dense baseline.
    kf=k.float();mean=kf.mean(-2,keepdim=True)
    kc=torch.nn.functional.pad(kf-mean,(0,0,0,np-n)).reshape(b,h,nb,64,d).sum(-2)/p.counts[None,None,:,None]
    vc=p.v.float().reshape(b,h,nb,64,d).sum(-2)/p.counts[None,None,:,None]
    return replace(p,kc=kc.half(),vc=vc.half())


def routes(p,target):
    nb=p.counts.numel();prefix=min(14,nb)
    qc=p.q.float().reshape(*p.q.shape[:2],nb,64,p.q.shape[-1]).sum(-2)/p.counts[None,None,:,None]
    score=qc@p.kc.float().transpose(-1,-2)
    ids=torch.arange(nb,device=score.device)
    mandatory=((ids[:,None]-ids[None,:]).abs()<=1)|(ids[None,:]<prefix)|(ids[:,None]<prefix)
    # Per-content-row budget solves global density after mandatory query rows.
    content=max(1,nb-prefix)
    budget=min(nb,max(prefix,math.ceil((target*nb*nb-prefix*nb)/content)))
    ranked=score.masked_fill(mandatory,float('inf'))
    selected=torch.zeros_like(score,dtype=torch.bool).scatter_(-1,ranked.topk(max(1,budget),dim=-1).indices,True)
    return (selected|mandatory).int().contiguous()


def invoke(p,r,kind):
    b,h,_,d=p.q.shape
    if kind=='scalar_pool':
        out=scalar_pool_kernel(b,h,p.tokens,d)(*raw_args(p,r)[:10])[0]
    else:out=get_kernel(b,h,p.tokens,d,False,False,sm=75)(*raw_args(p,r))[0]
    return out[:,:,:p.tokens]


def masked_reference(p,r,kind):
    # Streaming independent mathematics, including the per-leaf FP16 P operand.
    # Full output, one retained leaf at a time; no N×N materialization.
    from patches.tilelang_sparse_attention import reference
    return reference(replace(p,shift=torch.zeros(p.q.shape[:2]+(p.tokens,),device=p.q.device)),r,pooled=kind=='scalar_pool')[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--process',type=int,default=0);ap.add_argument('--profile',action='store_true')
    ap.add_argument('--native-only',action='store_true');ap.add_argument('--reference-heads',type=int,default=3)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    hw=hardware();assert len(hw['devices'])==2 and all(x['capability']==[7,5] for x in hw['devices'])
    records=[r for r in json.loads((args.data/'capture_manifest.json').read_text())['records'] if r['component']=='dit']
    assert records
    result={'status':'running','hardware':hw,'process_seed':args.process,'cases':[],
        'input_boundary':'original captured FP32 BHND CUDA -> FP16 output; casts, preparation and routing included in API',
        'length_semantics':'2048/4096 contiguous truncations of native tokens;6814 native. 8192/16384 skipped: no genuine captured sequences at those lengths.',
        'scope':'GPU0 real captured heads; shape recorded per case. No generation quality or TP2 throughput inference',
        'precision':'All matched modes use frozen fused Sage32 Q32/K64, FP32 token mean, half-away INT8 rounding, FP16 P/V, FP32 online softmax/PV accumulation',
        'counter_limit':'Prior actual ERR_NVGPUCTRPERM; no achieved occupancy/stall/TensorCore utilization claims'}
    for record in records:
        (fullq,fullk,fullv,_),heads=load(args.data,record)
        lengths=[fullq.shape[-2]] if args.profile or args.native_only else [2048,4096,fullq.shape[-2]]
        for n in lengths:
            original=[x[:,:,:n].contiguous() for x in [fullq,fullk,fullv]]
            half=[x.half() for x in original];p,dense=prepare_sage_fused(*half);s=summaries(p,half[1])
            def prep(kind):
                with label(kind+'/input_cast'):hh=[x.half() for x in original]
                with label(kind+'/QK_quantization'):current,dr=prepare_sage_fused(*hh)
                if kind!='sage32':
                    with label(kind+'/centroid_packing'):current=summaries(current,hh[1])
                return current,dr
            targets=[.62] if args.profile else [.25,.33,.5,.62,1.]
            specs=[('sage32',1.)]+[(kind,t) for t in targets for kind in ['drop','scalar_pool']]
            unique={}
            for kind,target in specs:
                rr=dense if kind=='sage32' else routes(s,target)
                key=(kind,hashlib.sha256(rr.cpu().numpy().tobytes()).hexdigest())
                if key in unique:unique[key][2].append(target)
                else:unique[key]=[kind,target,[target]]
            specs=list(unique.values())
            random.Random(301+args.process+n).shuffle(specs)
            for kind,target,requested_targets in specs:
                route=dense if kind=='sage32' else routes(s,target)
                actual_density=route.float().mean().item()
                row={'capture':record,'source_heads':heads,'shape':list(original[0].shape),'kind':kind,'requested_density':target,
                    'equivalent_requested_densities':requested_targets,'headwise_effective_density':route.float().mean((-1,-2)).cpu().tolist(),
                    'effective_density':actual_density,'mandatory_prefix_blocks':14,'mandatory_floor':1-((p.counts.numel()-14)/p.counts.numel())**2,
                    'tile':{'query':32,'key':64,'head_dim':original[0].shape[-1],'threads':128},'compiled_first_call_seconds':None}
                current=p if kind=='sage32' else s
                start=time.perf_counter();output=invoke(current,route,kind);torch.cuda.synchronize();row['compiled_first_call_seconds']=time.perf_counter()-start
                # Only once per process/kernel/length/route, before timings.
                if args.process==0 and not args.profile:
                    from attention_root_causes import metrics
                    ids=torch.linspace(0,current.q.shape[1]-1,min(args.reference_heads,current.q.shape[1]),device=current.q.device).round().long()
                    selected=replace(current,**{name:value[:,ids].contiguous() for name,value in current.__dict__.items()
                        if isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[1]==current.q.shape[1]})
                    expected=masked_reference(selected,route[:,ids].contiguous(),kind)
                    row['implementation_reference_heads']=ids.cpu().tolist()
                    row['implementation_error']=metrics(output[:,ids],expected)
                    assert max(row['implementation_error']['relative_l2'])<.003,(kind,n,row['implementation_error'])
                def api():
                    cur,dr=prep(kind)
                    if kind=='sage32':r=dr
                    else:
                        with label(kind+'/routing'):r=routes(cur,target)
                    with label(kind+'/QK_softmax_PV'):return invoke(cur,r,kind)
                if args.profile:
                    measure(api,repeats=3);torch.cuda.profiler.start()
                    with label('controlled/'+kind):
                        for _ in range(3):api()
                    torch.cuda.synchronize();torch.cuda.profiler.stop()
                    row['profiled_calls']=3
                else:
                    row['timings']={'raw':measure(lambda:invoke(current,route,kind)),
                        'cast_and_preparation':measure(lambda:prep(kind)),'API':measure(api)}
                    if kind!='sage32':row['timings']['routing']=measure(lambda:routes(s,target))
                result['cases'].append(row);save(args.output/'results.json',result)
                print('CASE',n,kind,target,round(actual_density,4),flush=True)
            if n==fullq.shape[-2] and not args.profile:
                # Legacy API causal ablation: identical original tensors, full
                # original prepare (including unused V8), and validation sync.
                legacy=prepare(*half);lr=make_routes(legacy,'sol',tau=0.)
                lr[...,:14]=1;lr[...,:14,:]=1
                from patches.tilelang_sparse_attention import execute
                def legacy_api():
                    with label('legacy/preparation'):pp=prepare(*[x.half() for x in original])
                    with label('legacy/routing'):
                        rr=make_routes(pp,'sol',tau=0.);rr[...,:14]=1;rr[...,:14,:]=1
                    with label('legacy/public_validation_kernel_LSE'):return execute(pp,rr,pooled=False)[0]
                measure(legacy_api,repeats=3)
                profile=cProfile.Profile();profile.enable()
                for _ in range(3):legacy_api()
                torch.cuda.synchronize();profile.disable();profile.dump_stats(str(args.output/('legacy_cpu_'+Path(record['file']).stem+'.prof')))
                result['legacy']={'density':lr.float().mean().item(),'API':measure(legacy_api),
                    'raw':measure(lambda:invoke(legacy,lr,'drop')),
                    'preparation':measure(lambda:prepare(*[x.half() for x in original]))}
                result.setdefault('legacy_cases',[]).append({'capture':record,**result['legacy']})
                save(args.output/'results.json',result)
    result['status']='complete';save(args.output/'results.json',result)


if __name__=='__main__':main()
