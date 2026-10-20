"""Physical SM75 correctness matrix and untuned multi-approach timings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from patches.tilelang_sparse_attention import prepare,make_routes,execute,reference,get_kernel

# Prespecified implementation gates, independent of dense-model fidelity.
LIMITS={'rel_l2':0.002,'max_abs':0.015,'lse_max_abs':0.003}
MODES=[('dense_int8','dense',False,False,{}),
       ('sol_fp16_pv','sol',True,False,{}),
       ('sol_int8_pv','sol',True,True,{}),
       ('topk_12p5','topk',False,False,{}),
       ('local_global','local_global',False,False,{})]
SWEEP=[('sol_tau0','sol',True,False,{'tau':0.0}),
       ('sol_tau0p5','sol',True,False,{'tau':0.5}),
       ('topk_25','topk',False,False,{'density':0.25}),
       ('topk_50','topk',False,False,{'density':0.5})]


def metrics(out,ref):
    a,b=out.float(),ref.float()
    delta=a-b
    return dict(rel_l2=(delta.norm()/b.norm().clamp_min(1e-12)).item(),
                max_abs=delta.abs().max().item(),rms=delta.square().mean().sqrt().item(),
                cosine=F.cosine_similarity(a.flatten()[None],b.flatten()[None]).item())


def timing(fn,repeat=5):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    values=[]
    for _ in range(repeat):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();fn();end.record();end.synchronize();values.append(begin.elapsed_time(end))
    return {'median_ms':statistics.median(values),'samples_ms':values,'max_ms':max(values)}


def dense_reference(q,k,v,tile=128):
    # FP32 unquantized reference; peak score storage is BH*tile*N.
    outputs=[]
    for start in range(0,q.shape[2],tile):
        z=torch.matmul(q[:,:,start:start+tile].float(),k.float().transpose(-1,-2))/math.sqrt(q.shape[-1])
        outputs.append(torch.matmul(z.softmax(-1),v.float()))
    return torch.cat(outputs,dim=2).half()


def inputs(case,device):
    b,h,n,d=case['shape']
    torch.manual_seed(case['seed'])
    q=torch.randn(b,h,n,d,device=device,dtype=torch.float16)
    k=torch.randn_like(q);v=torch.randn_like(q)
    kind=case['kind']
    if kind=='zeros':q.zero_();k.zero_()
    elif kind=='constant_k':k[:]=3.0
    elif kind=='peaked':k=q.clone();q=q*4
    elif kind=='outliers':
        k+=8;k[:,:,::97,0]=80;q[...,0]*=4
    elif kind=='local_signal':
        # Shared within-leaf signatures; synthetic, not a captured model workload.
        pattern=torch.randn(b,h,(n+63)//64,d,device=device,dtype=torch.float16)*2
        q=pattern.repeat_interleave(64,dim=2)[:,:,:n]+q*0.2
        k=pattern.repeat_interleave(64,dim=2)[:,:,:n]+k*0.2
    elif kind=='strided':
        # Preserve BHND shape while making head/token strides non-contiguous.
        q=q.transpose(1,2).contiguous().transpose(1,2)
        k=k.transpose(1,2).contiguous().transpose(1,2)
        v=v.transpose(1,2).contiguous().transpose(1,2)
    return q,k,v


def main(output,quick=False):
    output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0')
    assert torch.cuda.get_device_capability(device)==(7,5)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_grad_enabled(False)
    cases=[dict(name='ragged65_d64',shape=[1,2,65,64],kind='random',seed=101),
           dict(name='ragged513_d128',shape=[1,2,513,128],kind='random',seed=102),
           dict(name='strided_b2h3_d64',shape=[2,3,256,64],kind='strided',seed=103),
           dict(name='uniform_zero_qk',shape=[1,2,513,128],kind='zeros',seed=104),
           dict(name='constant_k',shape=[1,2,513,128],kind='constant_k',seed=105),
           dict(name='peaked',shape=[1,2,513,128],kind='peaked',seed=106),
           dict(name='outliers',shape=[1,2,513,128],kind='outliers',seed=107),
           dict(name='local2048_h4',shape=[1,4,2048,128],kind='local_signal',seed=108),
           dict(name='random2048_h4',shape=[1,4,2048,128],kind='random',seed=109),
           dict(name='local8192_h28',shape=[1,28,8192,128],kind='local_signal',seed=110)]
    if quick:cases=cases[:2]
    result={'completed':False,'passed':False,'physical':True,'gpu':torch.cuda.get_device_name(0),
            'compute_capability':list(torch.cuda.get_device_capability(0)),
            'implementation_limits':LIMITS,'cases':[],
            'scope':'synthetic operator validation; no model-level quality certification; untuned timings',
            'source_sha256':{rel:hashlib.sha256((Path(__file__).resolve().parents[1]/rel).read_bytes()).hexdigest()
                             for rel in ['patches/tilelang_sparse_attention.py','benchmarks/tilelang_t4_attention.py']}}
    def save():
        (output/'results.json').write_text(json.dumps(result,indent=2))
    save()
    try:
        for case in cases:
            print('CASE',case['name'],flush=True)
            q,k,v=inputs(case,device)
            p=prepare(q,k,v)
            dense=dense_reference(q,k,v)
            prep_time=timing(lambda:prepare(q,k,v),3)
            baseline=None
            # PyTorch's hardware-selected FP16 SDPA is a measured baseline too.
            sdpa=F.scaled_dot_product_attention(q,k,v)
            result.setdefault('pytorch_baselines',[]).append(dict(case=case,
                vs_dense_fp32=metrics(sdpa,dense),forward=timing(lambda:F.scaled_dot_product_attention(q,k,v))))
            modes=MODES+(SWEEP if case['name'] in ['local2048_h4','random2048_h4'] else [])
            for mode,router,pooled,int8_pv,route_options in modes:
                print('MODE',case['name'],mode,flush=True)
                row=dict(case=case,mode=mode,status='running')
                result['cases'].append(row);save()
                started=time.monotonic()
                try:
                    routes=make_routes(p,router,**route_options)
                    kernel=get_kernel(*case['shape'],pooled,int8_pv)
                    compile_seconds=time.monotonic()-started
                    raw_args=(p.q,p.qi,p.ki,p.v,p.qs,p.ks,p.kc,p.vc,p.counts,routes,p.vi,p.vs)
                    out,lse=execute(p,routes,pooled,int8_pv);torch.cuda.synchronize()
                    ref,ref_lse=reference(p,routes,pooled,int8_pv)
                    impl=metrics(out,ref)
                    lse_error=(lse-ref_lse).abs().max().item()
                    passed=bool(torch.isfinite(out).all() and torch.isfinite(lse).all()
                                and impl['rel_l2']<=LIMITS['rel_l2']
                                and impl['max_abs']<=LIMITS['max_abs'] and lse_error<=LIMITS['lse_max_abs'])
                    if mode=='dense_int8':baseline=out.clone()
                    route_time=timing(lambda:make_routes(p,router,**route_options),3)
                    forward_time=timing(lambda:kernel(*raw_args))
                    def total():
                        current=prepare(q,k,v)
                        r=make_routes(current,router,**route_options)
                        return execute(current,r,pooled,int8_pv)
                    e2e_time=timing(total,3)
                    torch.cuda.reset_peak_memory_stats()
                    total();torch.cuda.synchronize()
                    stem=case['name']+'_'+('pv8' if int8_pv else ('pool' if pooled else 'selected'))
                    source=kernel.get_kernel_source()
                    (output/(stem+'.cu')).write_text(source)
                    # PTX confirms the real SM75 instruction families, not dispatch alone.
                    if case==cases[0]:
                        kernel.export_ptx(str(output/(stem+'.ptx')))
                    row.update(status='passed' if passed else 'numerical_failure',passed=passed,
                               implementation_error=impl,lse_max_abs=lse_error,
                               vs_dense_fp16=metrics(out,dense),vs_dense_int8=metrics(out,baseline),
                               selected_density=routes.float().mean().item(),route_options=route_options,
                               selected_count_min=routes.sum(-1).min().item(),selected_count_max=routes.sum(-1).max().item(),
                               preprocessing=prep_time,routing=route_time,forward=forward_time,end_to_end=e2e_time,
                               compile_seconds=compile_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               kernel_source_file=stem+'.cu')
                    print('RESULT',case['name'],mode,json.dumps({key:row[key] for key in ['passed','implementation_error','lse_max_abs','vs_dense_fp16','selected_density','forward']}),flush=True)
                except Exception:
                    row.update(status='error',passed=False,traceback=traceback.format_exc())
                    print(row['traceback'],flush=True)
                save()
            del p,q,k,v,dense,baseline
            torch.cuda.empty_cache()
        result['completed']=True
        result['passed']=all(row['passed'] for row in result['cases'])
    except Exception:
        result['fatal_traceback']=traceback.format_exc();print(result['fatal_traceback'],flush=True)
    finally:save()
    print('FINAL',json.dumps({'completed':result['completed'],'passed':result['passed'],'records':len(result['cases'])}),flush=True)
    if not result['passed']:raise RuntimeError('SM75 qualification failed; see results.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--quick',action='store_true')
    options=parser.parse_args();main(options.output,options.quick)
