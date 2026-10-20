"""Numerical contract checks for removal of unused dense preparation work."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from patches.tilelang_sparse_attention import prepare
from patches.tilelang_dense_preparation import prepare_dense

torch.manual_seed(4083)
for n,d in [(65,64),(353,128),(129,128)]:
    q,k,v=[torch.randn(1,2,n,d).half() for _ in range(3)]
    base=prepare(q,k,v)
    for pv8 in [False,True]:
        actual,routes=prepare_dense(q,k,v,int8_pv=pv8,want_lse=True)
        for name in ['q','v','qi','ki','qs','ks','counts','shift']+(['vi','vs'] if pv8 else []):
            assert torch.equal(getattr(base,name),getattr(actual,name)),(n,d,pv8,name)
        assert bool(routes.eq(1).all())
q=torch.randn(1,8,65,128).half();k=torch.randn(1,2,65,128).half();v=torch.randn_like(k)
base=prepare(q,k,v,enable_gqa=True)
actual,_=prepare_dense(q,k,v,want_lse=True,enable_gqa=True)
for name in ['qi','ki','qs','ks','q','v','shift']:
    assert torch.equal(getattr(base,name),getattr(actual,name)),('GQA',name)
print('Dense preparation preserves quantized operands, counts and LSE shift exactly')
