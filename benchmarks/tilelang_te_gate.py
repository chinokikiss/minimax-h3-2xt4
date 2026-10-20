"""Physical T4 gate for causal prefill and GQA head mapping."""
import argparse
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from patches.tilelang_sparse_attention import prepare,make_routes,execute,reference,get_kernel


def main(output):
    output.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.get_device_capability(0)==(7,5)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261013)
    result={'physical':True,'gpu':torch.cuda.get_device_name(),'passed':False,'cases':[]}
    try:
        for n,d,h,hkv in [(65,64,8,2),(129,128,8,2),(513,128,64,8)]:
            q=torch.randn(1,h,n,d,device='cuda',dtype=torch.float16)
            k=torch.randn(1,hkv,n,d,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
            p=prepare(q,k,v,enable_gqa=True);routes=make_routes(p,'dense')
            for pv8 in [False,True]:
                actual,lse=execute(p,routes,pooled=False,int8_pv=pv8,causal=True)
                expected,el=reference(p,routes,pooled=False,int8_pv=pv8,causal=True)
                error=(actual.float()-expected.float()).norm()/expected.float().norm()
                le=(lse-el).abs().max()
                passed=bool(error<0.002 and (actual-expected).abs().max()<0.015 and le<0.003)
                row={'n':n,'d':d,'heads':h,'kv_heads':hkv,'int8_pv':pv8,'relative_l2':error.item(),'lse_max_abs':le.item(),'passed':passed}
                result['cases'].append(row);print('TE_GATE',json.dumps(row),flush=True)
                if n==129:
                    kernel=get_kernel(1,h,n,d,False,pv8,causal=True)
                    kernel.export_ptx(str(output/f'causal_d128_pv8{int(pv8)}.ptx'))
        result['passed']=all(r['passed'] for r in result['cases'])
    except Exception:result['traceback']=traceback.format_exc();print(result['traceback'],flush=True)
    finally:(output/'te_gate.json').write_text(json.dumps(result,indent=2))
    assert result['passed'],result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
