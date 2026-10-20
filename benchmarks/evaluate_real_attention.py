"""Replay every captured head, holding native upstream activations fixed."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from patches.tilelang_sparse_attention import prepare,make_routes,execute,get_kernel


def error(a,b):
    delta=a.float()-b.float();bf=b.float()
    axes=(-2,-1)
    num=delta.square().sum(axes);den=bf.square().sum(axes)
    dot=(a.float()*bf).sum(axes)
    return {'squared_error':num.flatten().tolist(),'squared_reference':den.flatten().tolist(),
            'relative_l2':(num/den.clamp_min(1e-24)).sqrt().flatten().tolist(),
            'cosine':(dot/(a.float().square().sum(axes)*den).sqrt().clamp_min(1e-24)).flatten().tolist(),
            'max_absolute':delta.abs().amax(axes).flatten().tolist()}


def distribution(q,k,causal,queries=64):
    n=q.shape[2]
    ids=torch.linspace(0,n-1,min(n,queries),device=q.device).round().long().unique()
    z=q[:,:,ids].float()@k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])
    if causal:z=z.masked_fill(torch.arange(n,device=q.device)[None,:]>ids[:,None],float('-inf'))
    p=z.softmax(-1)
    entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)
    top=p.topk(min(64,n),dim=-1).values
    return {'sample_query_ids':ids.cpu().tolist(),'entropy_mean':entropy.mean(-1).flatten().tolist(),
            'effective_keys_mean':entropy.exp().mean(-1).flatten().tolist(),
            'top1_mass_mean':top[...,0].mean(-1).flatten().tolist(),
            'top8_mass_mean':top[...,:min(8,n)].sum(-1).mean(-1).flatten().tolist(),
            'top64_mass_mean':top.sum(-1).mean(-1).flatten().tolist()},p,ids


def evaluate(directory,head_chunk=8,only=None):
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((directory/'capture_manifest.json').read_text())
    result={'status':'running','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'head_chunk':head_chunk,'records':[],'source_sha256':hashlib.sha256((ROOT/'patches/tilelang_sparse_attention.py').read_bytes()).hexdigest(),
            'scope':'all captured native heads and query rows; operator replay, no propagated-error substitution'}
    path=directory/(f'attention_errors_{only}.json' if only else 'attention_errors.json')
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(path)
    save()
    modes=[('dense_int8_fp16_pv','dense',False,False,{}),('dense_int8_int8_pv','dense',False,True,{})]
    sparse=[('sol_tau0','sol',True,False,{'tau':0.}),('sol_tau0p5','sol',True,False,{'tau':.5}),
            ('sol_tau1','sol',True,False,{'tau':1.}),('sol_int8_pv_tau0','sol',True,True,{'tau':0.}),
            ('topk25','topk',False,False,{'density':.25}),('topk50','topk',False,False,{'density':.5}),
            ('local_global','local_global',False,False,{})]
    for record in manifest['records']:
        if only and record['component']!=only:continue
        file=directory/record['file']
        with file.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
        data=torch.load(file,map_location='cpu',weights_only=True)
        q,k,v,native=[data[name] for name in ['q','k','v','native_output']]
        causal=record['component']=='te'
        if causal:
            mask=data['mask']
            if mask is None:raise ValueError('Native TE mask missing')
            allowed=mask.reshape(-1,q.shape[2],q.shape[2])>=0
            assert (allowed==torch.ones(q.shape[2],q.shape[2],dtype=torch.bool).tril()).all(), 'Mask is not plain causal'
        elif data['mask'] is not None:raise ValueError('Unexpected masked non-TE attention')
        if data['scale'] is not None and abs(data['scale']-q.shape[-1]**-.5)>1e-9:raise ValueError('Nonstandard scale')
        row={'capture':record,'heads':q.shape[1],'modes':{},'distribution':[],'input_statistics':[]}
        result['records'].append(row)
        mode_rows=modes+(sparse if record['component']=='dit' else [])
        for start in range(0,q.shape[1],head_chunk):
            heads=torch.arange(start,min(start+head_chunk,q.shape[1]))
            group=q.shape[1]//k.shape[1]
            kh=heads//group
            qg=q[:,heads].cuda();kg=k[:,kh].cuda();vg=v[:,kh].cuda();ng=native[:,heads].cuda()
            stats,prob,ids=distribution(qg,kg,causal)
            row['distribution'].append({'head_ids':heads.tolist(),**stats})
            row['input_statistics'].append({'head_ids':heads.tolist(),
                'q_absmax':qg.abs().amax((-2,-1)).flatten().tolist(),
                'k_absmax':kg.abs().amax((-2,-1)).flatten().tolist(),
                'v_absmax':vg.abs().amax((-2,-1)).flatten().tolist()})
            cast=F.scaled_dot_product_attention(qg.half(),kg.half(),vg.half(),is_causal=causal)
            row['modes'].setdefault('fp16_cast',[]).append({'head_ids':heads.tolist(),'error':error(cast,ng)})
            p=prepare(qg.half(),kg.half(),vg.half())
            dense_int8=None
            for name,router,pooled,pv8,options in mode_rows:
                routes=make_routes(p,router,**options)
                if record['component']=='dit':
                    # Protect prefix conditioning, matching this workload's packed
                    # ordering (text + reference + target audio precede video).
                    prefix=math.ceil((qg.shape[2]-27*(352//16//2)*(640//16//2))/64)
                    prefix=max(0,min(prefix,routes.shape[-1]))
                    routes[...,:prefix]=1;routes[...,:prefix,:]=1
                started=time.perf_counter();out,lse=execute(p,routes,pooled,pv8,causal=causal)
                torch.cuda.synchronize()
                event0,event1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                event0.record();execute(p,routes,pooled,pv8,causal=causal);event1.record();event1.synchronize()
                selected=routes[:,:,ids//64].repeat_interleave(64,-1)[...,:qg.shape[2]].bool()
                omitted=(prob*~selected).sum(-1).mean(-1).flatten().tolist()
                metrics=error(out,ng)
                if name=='dense_int8_fp16_pv':dense_int8=out.clone()
                row['modes'].setdefault(name,[]).append({'head_ids':heads.tolist(),'error':metrics,
                    'vs_dense_int8':error(out,dense_int8),
                    'selected_density':routes.float().mean().item(),'true_unselected_mass_mean':omitted,
                    'api_warm_ms':event0.elapsed_time(event1),'cold_wall_seconds':time.perf_counter()-started})
                print('REPLAY',file.name,heads.tolist(),name,'rel_l2',max(metrics['relative_l2']),flush=True)
                del out,lse,routes
            del qg,kg,vg,ng,p,prob,cast,dense_int8
            torch.cuda.empty_cache();save()
        del data,q,k,v,native;gc.collect()
    result['status']='complete';save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=Path('D:/attention_eval/tokyo_ref2va_20261013'))
    parser.add_argument('--head-chunk',type=int,default=8);parser.add_argument('--only',choices=['te','dit','video'])
    a=parser.parse_args();evaluate(a.directory,a.head_chunk,a.only)
