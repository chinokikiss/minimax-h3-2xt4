"""Sampled real-distribution causal ablations on CPU; never performance data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
from attention_root_causes import metrics
from patches.tilelang_grouped_tp import prepare,routes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_grad_enabled(False);torch.set_num_threads(8)
    name='dit_layer25_call09.pt';p=args.data/name
    manifest=json.loads((args.data/'capture_manifest.json').read_text());record=next(r for r in manifest['records'] if r['file']==name)
    assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==record['sha256']
    data=torch.load(p,map_location='cpu',weights_only=True);q,k,v=[data[n] for n in ('q','k','v')]
    assert not q.is_cuda and q.shape==(1,56,6814,128)
    prepared=prepare(q.half(),k.half(),v.half(),sparse=True);r=routes(prepared,874)
    n=q.shape[2];counts=prepared.counts.float();nb=len(counts)
    result={'scope':'CPU-only accuracy diagnostics; all56 real DiT heads, deterministic32 uniform +32 largest-Q-norm query rows per head. Not GPU timings or full-trajectory quality.',
        'source_sha256':record['sha256'],'layer':25,'denoise_call':9,'heads':[]}
    for head in range(56):
        ids=torch.cat((torch.linspace(0,n-1,32).round().long(),q[0,head].square().sum(-1).topk(32).indices)).unique(sorted=True)
        qa=q[:,head:head+1,ids];ka=k[:,head:head+1];va=v[:,head:head+1]
        z=qa@ka.transpose(-1,-2)/math.sqrt(128);prob=z.softmax(-1);native=prob@va
        qh=(prepared.qi[:,head:head+1,ids].float().reshape(1,1,len(ids),4,32)*prepared.qs[:,head:head+1,ids,:,None]).reshape_as(qa)
        kh=(prepared.ki[:,head:head+1,:n].float().reshape(1,1,n,4,32)*prepared.ks[:,head:head+1,:n,:,None]).reshape_as(ka)
        zq=qh@kh.transpose(-1,-2)/math.sqrt(128);pq=zq.softmax(-1)
        selected=r[:,head:head+1,ids//64,:].bool();keep=selected.repeat_interleave(64,-1)[...,:n]
        native_drop=z.masked_fill(~keep,-float('inf')).softmax(-1)@va
        group_drop=zq.masked_fill(~keep,-float('inf')).softmax(-1)@va.half().float()
        summ=qa.half().float()@prepared.kc[:,head:head+1].float().transpose(-1,-2)/math.sqrt(128)+counts.log()
        scalar_z=torch.cat((zq.masked_fill(~keep,-float('inf')),summ.masked_fill(selected,-float('inf'))),-1)
        scalar_v=torch.cat((va.half().float(),prepared.vc[:,head:head+1].float()),-2)
        scalar=scalar_z.softmax(-1)@scalar_v
        pooled_probability=scalar_z.softmax(-1)[...,n:].sum(-1)
        leaf_logmass=torch.nn.functional.pad(zq,(0,nb*64-n),value=-float('inf')).reshape(1,1,len(ids),nb,64).logsumexp(-1)
        cast=(qa.half().float()@ka.half().float().transpose(-1,-2)/math.sqrt(128)).softmax(-1)@va.half().float()
        missing=(prob*~keep).sum(-1).flatten();delta=zq-z;delta-=delta.mean(-1,keepdim=True)
        row={'global_head':head,'rank':head//28,'local_head':head%28,'query_ids':ids.tolist(),
            'native_output_rms':native.square().mean().sqrt().item(),'density':r[:,head].float().mean().item(),
            'native_dropped_mass_mean':missing.mean().item(),'native_dropped_mass_p95':missing.quantile(.95).item(),'native_dropped_mass_max':missing.max().item(),
            'scalar_estimated_omitted_mass_mean':pooled_probability.mean().item(),
            'omitted_leaf_log_mass_gap_mean':(leaf_logmass-summ)[~selected].mean().item(),
            'group32_probability_TV':((pq-prob).abs().sum(-1)*.5).mean().item(),
            'group32_centered_logit_RMSE':delta.square().mean().sqrt().item(),
            'errors':{name:metrics(value,native) for name,value in [('cast_only',cast),('group32_dense',pq@va.half().float()),
                ('routing_only_native_QKV',native_drop),('group32_drop',group_drop),('group32_scalar',scalar)]}}
        result['heads'].append(row)
    args.output.write_text(json.dumps(result,indent=2))
    worst=sorted(result['heads'],key=lambda r:r['errors']['group32_drop']['relative_l2'][0],reverse=True)[:6]
    for row in worst:print('CPU_DIAGNOSTIC',row['global_head'],{k:round(v['relative_l2'][0],4) for k,v in row['errors'].items()},'omitted_mass',round(row['native_dropped_mass_mean'],4),'max',round(row['native_dropped_mass_max'],4),flush=True)


if __name__=='__main__':main()
