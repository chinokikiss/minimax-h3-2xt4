"""CPU gates for GQA mapping, rectangular queries, causality and pooling."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from patches.tilelang_grouped_tp import prepare,reference
from patches.tilelang_grouped_qk import prepare_grouped,grouped_reference


def main():
    torch.manual_seed(20261016)
    q=torch.randn(1,4,65,64).half();k=torch.randn(1,2,65,64).half();v=torch.randn_like(k)
    p=prepare(q,k,v)
    old=prepare_grouped(q,k,v,enable_gqa=True)
    for causal in (False,True):
        actual=reference(p,causal=causal,query_chunk=17)
        expected=grouped_reference(old,causal=causal,query_chunk=17)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    # Quantization/centering must depend on all keys for a query partition.
    whole=reference(p,query_chunk=17)
    for a,b in ((0,32),(32,65)):
        part=prepare(q[:,:,a:b],k,v)
        assert torch.equal(part.ki,p.ki) and torch.equal(part.ks,p.ks)
        torch.testing.assert_close(reference(part),whole[:,:,a:b],rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(reference(part,causal=True,query_offset=a),reference(p,causal=True)[:,:,a:b],rtol=2e-5,atol=2e-6)
    sq=torch.zeros(1,2,129,64).half();sk=torch.randn_like(sq);sv=torch.ones_like(sq)
    sv[:,:,64:128]*=2;sv[:,:,128:]*=3
    sp=prepare(sq,sk,sv,sparse=True)
    r=torch.ones(1,2,3,3,dtype=torch.int32);r[:,:,:,1]=0
    dense=reference(sp)
    torch.testing.assert_close(reference(sp,r,'scalar'),dense,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(reference(sp,torch.ones_like(r),'drop'),dense)
    torch.testing.assert_close(reference(sp,r,'drop'),torch.full_like(dense,67/65),rtol=1e-6,atol=1e-6)
    print('PASS: GQA mapping, ragged leaf, rectangular global centering, causal offsets, drop, scalar pooling')


if __name__=='__main__':main()
