"""Distinguish real sparse approximation error from a pooled-kernel defect."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from patches.tilelang_sparse_attention import prepare,make_routes,execute,reference

if __name__=='__main__':
    output=Path('D:/attention_eval/tokyo_ref2va_20261013')
    errors=json.loads((output/'attention_errors.json').read_text())
    candidates=[]
    for row in errors['records']:
        if row['capture']['component']!='dit':continue
        for chunk in row['modes']['sol_tau0']:
            for head,value in zip(chunk['head_ids'],chunk['error']['relative_l2']):candidates.append((value,row['capture'],head))
    native_error,capture,head=max(candidates,key=lambda r:r[0])
    data=torch.load(output/capture['file'],map_location='cpu',weights_only=True)
    q,k,v=[data[name][:,head:head+1].half().cuda() for name in ('q','k','v')]
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    p=prepare(q,k,v);routes=make_routes(p,'sol',tau=0.)
    prefix=math.ceil((q.shape[2]-27*11*20)/64)
    routes[...,:prefix]=1;routes[...,:prefix,:]=1
    proof=json.loads((output/'real_kernel_validation.json').read_text())
    for pv8 in [False,True]:
        actual,lse=execute(p,routes,True,pv8)
        expected,el=reference(p,routes,True,pv8)
        error=((actual.float()-expected.float()).norm()/expected.float().norm()).item()
        le=(lse-el).abs().max().item()
        assert error<.002 and le<.003,(error,le)
        proof['cases'].append({'component':'dit','pooled':True,'capture':capture['file'],'head':head,'pv8':pv8,
                               'implementation_relative_l2':error,'lse_max_abs':le,'native_sparse_error':native_error})
        print('REAL_SPARSE_MATH',head,pv8,error,le,flush=True)
    (output/'real_kernel_validation.json').write_text(json.dumps(proof,indent=2))
