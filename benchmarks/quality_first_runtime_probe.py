"""Actual T4 cu130 Kitchen execution + CUDA12.8 TileLang compatibility gate."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[os.environ['COMFY_ROOT'],str(ROOT),str(ROOT/'benchmarks')]
import torch
import comfy.quant_ops
import comfy_kitchen as ck
from patches.tilelang_sage_qk import prepare_sage
from patches.tilelang_sparse_attention import get_kernel,reference
from attention_root_causes import metrics

assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2))
torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(19)
nvcc=subprocess.check_output([os.environ['CUDACXX'],'--version'],text=True)
assert 'release 12.8' in nvcc,nvcc
q=torch.randn(1,1,65,64,device='cuda',dtype=torch.float16)
p,r=prepare_sage(q,q,q)
kernel=get_kernel(*q.shape,False,False,sm=75,causal=True)
actual=kernel(p.q,p.qi,p.ki,p.v,p.qs,p.ks,p.kc,p.vc,p.counts,r,p.vi,p.vs)[0][:,:,:65]
implementation=metrics(actual,reference(p,r,pooled=False,causal=True)[0]);assert implementation['relative_l2'][0]<.003,implementation
x=torch.randn(32,256,device='cuda',dtype=torch.float32)
w=torch.randint(-127,128,(128,256),device='cuda',dtype=torch.int8);scale=torch.tensor(.01,device='cuda')
kwargs=dict(x=x,weight=w,weight_scale=scale,bias=None,out_dtype=torch.float32,convrot=True,convrot_groupsize=256)
impl=ck.registry.get_implementation('int8_linear',kwargs=kwargs)
assert '.cuda' in impl.__module__,impl.__module__
linear=ck.int8_linear(**kwargs);torch.cuda.synchronize()
assert linear.shape==(32,128) and torch.isfinite(linear).all()
grouped={}
if os.environ.get('H3_GROUPED_GATE')=='1':
    from patches.tilelang_grouped_fused import prepare_grouped_fused,execute_grouped_fused,dequantized_reference
    from patches.tilelang_grouped_qk import prepare_grouped
    for group in [32,64]:
        gq=torch.randn(1,1,65,128,device='cuda',dtype=torch.float16)
        gp=prepare_grouped_fused(gq,gq,gq,group=group)
        go=execute_grouped_fused(gp,group=group)
        error=metrics(go,dequantized_reference(gp,group))
        assert error['relative_l2'][0]<.003,error
        grouped[str(group)]={'implementation':error}
        if group==32:
            old=prepare_grouped(gq,gq,gq)
            match={'QI':torch.equal(gp.qi[:,:,:65],old.qi[:,:,:65]),'KI':torch.equal(gp.ki[:,:,:65],old.ki[:,:,:65]),
                'QS':torch.equal(gp.qs[:,:,:65],old.qs[:,:,:65]),'KS':torch.equal(gp.ks[:,:,:65],old.ks[:,:,:65])}
            assert all(match.values()),match
            grouped[str(group)]['legacy_operands_equal']=match
Path(sys.argv[1]).write_text(json.dumps(dict(torch=torch.__version__,nvcc=nvcc,capabilities=[list(torch.cuda.get_device_capability(i)) for i in range(2)],
    kitchen_backends=ck.list_backends(),int8_linear_implementation=impl.__module__,tilelang_implementation=implementation,grouped=grouped),indent=2,default=str))
print('RUNTIME_COMPATIBILITY_PASS',torch.__version__,impl.__module__,flush=True)
