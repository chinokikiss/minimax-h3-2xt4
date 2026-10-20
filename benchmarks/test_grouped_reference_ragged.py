"""CPU regression: ragged V must stop at valid keys, including causal TE."""
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from patches.tilelang_grouped_qk import Grouped,grouped_reference
from patches.tilelang_grouped_fused import dequantized_reference

torch.set_num_threads(4);torch.manual_seed(2091)
for n in [65,353]:
    np=math.ceil(n/64)*64;d=128
    p=Grouped(torch.randint(-127,128,(1,1,np,d),dtype=torch.int8),torch.randint(-127,128,(1,1,np,d),dtype=torch.int8),
        torch.randn(1,1,np,d).half(),torch.rand(1,1,np,4)*.01,torch.rand(1,1,np,4)*.01,None,n)
    for causal in [False,True]:
        expected=grouped_reference(p,causal=causal,query_chunk=31)
        actual=dequantized_reference(p,32,causal)
        difference=(actual-expected).norm()/expected.norm()
        assert difference<1e-6,(n,causal,difference.item())
print('PASS ragged65/353, causal/noncausal references; CPU only, no performance measurement')
