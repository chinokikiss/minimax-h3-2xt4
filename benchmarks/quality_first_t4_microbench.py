"""Bounded real-activation T4 diagnostics. No generation-quality rejection."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
import torch.nn.functional as F
from real_t4_attention_study import hardware,load,measure,label,save,raw_args
from attention_root_causes import metrics,probability_metrics,diagnose
from patches.tilelang_sparse_attention import prepare,make_routes,get_kernel,reference
from patches.tilelang_dense_preparation import prepare_dense,run_prevalidated_dense
from patches.tilelang_sage_qk import prepare_sage
from patches.tilelang_sage_preparation import prepare_sage_fused
from patches.tilelang_grouped_qk import prepare_grouped,execute_grouped,grouped_reference,grouped_kernel
from patches.tilelang_scalar_pool import scalar_pool_kernel


def qk_diagnostics(q,k,p,causal):
    n=q.shape[-2];d=q.shape[-1]
    ids=torch.linspace(0,n-1,min(64,n),device=q.device).round().long().unique()
    mask=torch.ones((len(ids),n),device=q.device,dtype=torch.bool)
    if causal:mask=torch.arange(n,device=q.device)[None,:]<=ids[:,None]
    mask=mask[None,None]
    # Compare centered candidate logits: a per-query K-mean shift cancels in softmax.
    z=(p.qi[:,:,ids].float()@p.ki[:,:,:n].float().transpose(-1,-2))
    z=z*p.qs[:,:,ids,None]*p.ks[:,:,None,:n]/math.sqrt(d)
    expected=q[:,:,ids].float()@k.float().transpose(-1,-2)/math.sqrt(d)
    return {'query_ids':ids.tolist(),'probability':probability_metrics(z.masked_fill(~mask,-float('inf')),
        expected.masked_fill(~mask,-float('inf')),mask)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=['sage_warp32','sage_warp32_fused','scalar_pool','drop'])
    parser.add_argument('--modes',help='Comma-separated focused follow-up modes')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    hw=hardware();assert len(hw['devices'])==2 and all(x['capability']==[7,5] for x in hw['devices']),hw
    records=json.loads((args.data/'capture_manifest.json').read_text())['records']
    result={'status':'running','hardware':hw,'primary_acceptance':'Generation quality; unmeasured in this operator stage',
            'scope':'Diagnostic heads, full native sequences. No full-pipeline throughput inference.',
            'counter_status':'ERR_NVGPUCTRPERM confirmed by previous actual probe','cases':[]}
    for record in records:
        (q,k,v,native),ids=load(args.data,record)
        causal=record['component']=='te';half=[x.half() for x in (q,k,v)]
        variants={
            'existing':lambda:prepare_dense(*half),
            'existing_pv8':lambda:prepare_dense(*half,int8_pv=True),
            'token_center_token_scale':lambda:prepare_sage(*half,q_block=1,k_block=1),
            'sage_warp32':lambda:prepare_sage(*half,q_block=32,k_block=64),
            'sage_block128':lambda:prepare_sage(*half,q_block=128,k_block=64),
            'sage_warp32_no_center':lambda:prepare_sage(*half,q_block=32,k_block=64,smoothing='none'),
            'sage_warp32_sol_center':lambda:prepare_sage(*half,q_block=32,k_block=64,smoothing='sol'),
            'sage_warp32_fp32_source':lambda:prepare_sage(q,k,v,q_block=32,k_block=64),
            'sage_warp32_fused':lambda:prepare_sage_fused(*half),
            'sage_cuda_warp32_contract':lambda:prepare_sage(*half,q_block=32,k_block=64,smoothing='token_native_dtype',rounding='cuda_rn'),
        }
        row={'capture':record,'head_ids':ids,'q_shape':list(q.shape),'modes':{},'quality_status':'not_measured'}
        fp16=F.scaled_dot_product_attention(*half,is_causal=causal)
        row['modes']['fp16_sdpa']={'versus_native':metrics(fp16,native),
            'timings':measure(lambda:F.scaled_dot_product_attention(*[x.half() for x in (q,k,v)],is_causal=causal)),
            'contract':'Torch FP16 SDPA reference; backend accumulation recorded by Systems, not assumed FP16 accumulation'}
        for name,prep in variants.items():
            if args.modes and name not in args.modes.split(','):continue
            p,routes=prep();pv8=name=='existing_pv8'
            operand_match=None
            if name=='sage_warp32_fused':
                baseline,_=prepare_sage(*half)
                n=q.shape[-2]
                operand_match={'q_int8_equal':torch.equal(p.qi[:,:,:n],baseline.qi[:,:,:n]),
                    'k_int8_equal':torch.equal(p.ki[:,:,:n],baseline.ki[:,:,:n]),
                    'q_scale_max_difference':(p.qs[:,:,:n]-baseline.qs[:,:,:n]).abs().max().item(),
                    'k_scale_max_difference':(p.ks[:,:,:n]-baseline.ks[:,:,:n]).abs().max().item()}
                assert operand_match['q_int8_equal'] and operand_match['k_int8_equal'],operand_match
            begin=time.perf_counter();kernel=get_kernel(*q.shape,False,pv8,sm=75,causal=causal)
            raw=lambda:kernel(*raw_args(p,routes))[0][:,:,:p.tokens]
            actual=raw();torch.cuda.synchronize();first=time.perf_counter()-begin
            # The dense fast path deliberately omits unused LSE reconstruction.
            # Its reference output needs no mean shift; compare centered LSE.
            if p.shift is None:p.shift=torch.zeros(q.shape[:-1],device=q.device)
            mathematical=reference(p,routes,pooled=False,int8_pv=pv8,causal=causal)
            implementation=metrics(actual,mathematical[0])
            assert max(implementation['relative_l2'])<.003,(name,'implementation mismatch',implementation)
            def api():
                with label(name+'/preparation'):current,r=prep()
                with label(name+'/QK_softmax_PV'):return run_prevalidated_dense(current,r,int8_pv=pv8,causal=causal)
            if args.profile:
                if name!=args.profile:continue
                measure(api,repeats=3);torch.cuda.profiler.start()
                for _ in range(3):api()
                torch.cuda.synchronize();torch.cuda.profiler.stop()
            entry={'versus_native':metrics(actual,native),'versus_fp16':metrics(actual,fp16),
                'implementation':implementation,'first_execution_compile_seconds':first,
                'qk':qk_diagnostics(q,k,p,causal),'quality_status':'pending_generation'}
            if operand_match:entry['preparation_contract']=operand_match
            if not args.profile:entry['timings']={n:measure(fn) for n,fn in [('raw',raw),('preparation',prep),('API',api)]}
            row['modes'][name]=entry;save(args.output/'results.json',result|{'cases':result['cases']+[row]})
            print('QUANT',record['file'],name,entry['versus_native']['relative_l2'],flush=True)
        if not args.profile and not args.modes:
            gp=prepare_grouped(*half);begin=time.perf_counter();grouped=execute_grouped(gp,causal)[0]
            torch.cuda.synchronize();first=time.perf_counter()-begin
            impl=metrics(grouped,grouped_reference(gp,causal));assert max(impl['relative_l2'])<.003,impl
            row['modes']['group32']={'versus_native':metrics(grouped,native),'versus_fp16':metrics(grouped,fp16),
                'implementation':impl,'first_execution_compile_seconds':first,
                'timings':{'raw':measure(lambda:grouped_kernel(*q.shape,causal)(gp.qi,gp.ki,gp.v,gp.qs,gp.ks)),
                    'API':measure(lambda:execute_grouped(prepare_grouped(*[x.half() for x in (q,k,v)]),causal)[0])}}
        if record['component']=='dit' and not args.modes:
            p=prepare(*half);row['routing_sweep']=[]
            for routing,parameter in [('sol',0.),('sol',.5),('sol',1.),('topk',.25),('topk',.5),('topk',.75)]:
                def route_fn():
                    r=make_routes(p,routing,tau=parameter,density=parameter if routing=='topk' else .5)
                    prefix=math.ceil(874/64);r[...,:prefix]=1;r[...,:prefix,:]=1
                    return r
                r=route_fn();sweep={'routing':routing,'parameter':parameter,'density':r.float().mean().item(),'variants':{}}
                if args.profile and (routing!='sol' or parameter!=0):continue
                if not args.profile:
                    sweep['routing_timing']=measure(route_fn)
                    sweep['diagnostics']=diagnose(q,k,v,p,r,queries=64)
                for correction in ['drop','original_pool','scalar_pool']:
                    if args.profile and correction!=args.profile:continue
                    begin=time.perf_counter()
                    kernel=scalar_pool_kernel(*q.shape) if correction=='scalar_pool' else get_kernel(*q.shape,pooled=correction=='original_pool',sm=75)
                    def invoke(current,route):
                        values=(current.q,current.qi,current.ki,current.v,current.qs,current.ks,current.kc,current.vc,current.counts,route)
                        if correction!='scalar_pool':values+=current.vi,current.vs
                        return kernel(*values)[0][:,:,:current.tokens]
                    raw=lambda:invoke(p,r)
                    actual=raw();torch.cuda.synchronize();first=time.perf_counter()-begin
                    mathematical=reference(p,r,pooled=correction!='drop')[0]
                    impl=metrics(actual,mathematical);assert max(impl['relative_l2'])<.003,(correction,impl)
                    def api():
                        with label(correction+'/preparation'):current=prepare(*[x.half() for x in (q,k,v)])
                        with label(correction+'/routing'):
                            route=make_routes(current,routing,tau=parameter,density=parameter if routing=='topk' else .5)
                            prefix=math.ceil(874/64);route[...,:prefix]=1;route[...,:prefix,:]=1
                        with label(correction+'/QK_softmax_PV'):return invoke(current,route)
                    entry={'versus_native':metrics(actual,native),'implementation':impl,'first_execution_compile_seconds':first}
                    if args.profile:
                        measure(api,repeats=3);torch.cuda.profiler.start()
                        for _ in range(3):api()
                        torch.cuda.synchronize();torch.cuda.profiler.stop()
                    else:entry['timings']={n:measure(fn) for n,fn in [('raw',raw),('API',api)]}
                    sweep['variants'][correction]=entry
                    print('CORRECTION',record['file'],routing,parameter,correction,flush=True)
                row['routing_sweep'].append(sweep)
        result['cases'].append(row);save(args.output/'results.json',result)
    result['status']='complete';save(args.output/'results.json',result)


if __name__=='__main__':main()
