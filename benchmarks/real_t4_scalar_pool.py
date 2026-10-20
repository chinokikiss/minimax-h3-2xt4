"""Actual sparse-failure head: operator equivalence and repeated re-benchmark."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
from real_t4_attention_study import load,hardware,routes_for,measure,label,save
from attention_root_causes import metrics
from patches.tilelang_sparse_attention import prepare,execute,reference,get_kernel
from patches.tilelang_scalar_pool import execute_scalar_pool,scalar_pool_kernel

parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--profile',choices=['original','scalar']);parser.add_argument('--heads',type=int)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
hw=hardware();assert len(hw['devices'])==2 and all(g['capability']==[7,5] for g in hw['devices'])
records=json.loads((args.data/'capture_manifest.json').read_text())['records']
record=next((r for r in records if r['file']=='dit_layer25_call09.pt'),records[0])
(q,k,v,native),ids=load(args.data,record,heads=args.heads);p=prepare(q.half(),k.half(),v.half())
routes=routes_for(p,'sol_tau0','dit')
original,original_lse=execute(p,routes,True,False)
candidate,lse=execute_scalar_pool(p,routes)
expected,expected_lse=reference(p,routes,True,False)
implementation=metrics(candidate,expected);vs_original=metrics(candidate,original)
assert max(implementation['relative_l2'])<.002,implementation
assert (lse-expected_lse).abs().max().item()<.003
result={'status':'running','hardware':hw,'scope':'real DiT replay, full sequence; head slice recorded explicitly',
    'head_ids':ids,'q_shape':list(q.shape),
    'capture':record,'selected_density':routes.float().mean().item(),
    'original_error':metrics(original,native),'scalar_error':metrics(candidate,native),
    'implementation_error':implementation,'vs_original':vs_original}
base=get_kernel(*q.shape,True,False,sm=75)
modified=scalar_pool_kernel(*q.shape)
original_inputs=(p.q,p.qi,p.ki,p.v,p.qs,p.ks,p.kc,p.vc,p.counts,routes,p.vi,p.vs)
modified_inputs=original_inputs[:10]
def old_api():
    with label('original/preparation'):
        current=prepare(q.half(),k.half(),v.half())
    with label('original/routing'):selected=routes_for(current,'sol_tau0','dit')
    with label('original/pooled_attention'):return execute(current,selected,True,False)
def new_api():
    with label('scalar/preparation'):
        current=prepare(q.half(),k.half(),v.half())
    with label('scalar/routing'):selected=routes_for(current,'sol_tau0','dit')
    with label('scalar/pooled_attention'):return execute_scalar_pool(current,selected)
if args.profile:
    fn=old_api if args.profile=='original' else new_api
    measure(fn,repeats=1);torch.cuda.profiler.start()
    for _ in range(3):fn()
    torch.cuda.synchronize();torch.cuda.profiler.stop()
else:
    result['timings']={name:measure(fn,repeats=15) for name,fn in [('original_raw',lambda:base(*original_inputs)),
        ('scalar_raw',lambda:modified(*modified_inputs)),('original_API',old_api),('scalar_API',new_api)]}
result['status']='complete';save(args.output/'scalar_pool.json',result)
print(json.dumps(result,indent=2),flush=True)
