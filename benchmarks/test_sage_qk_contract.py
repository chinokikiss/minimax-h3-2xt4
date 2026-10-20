"""CPU mathematical checks; these are not hardware performance evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from patches.tilelang_sage_qk import quant_blocks,prepare_sage

torch.manual_seed(19)
q=torch.randn(1,4,65,64);k=torch.randn(1,2,65,64);v=torch.randn_like(k)
p,r=prepare_sage(q,k,v,enable_gqa=True)
expanded=k.repeat_interleave(2,1)
mean=expanded.mean(-2,keepdim=True)
assert torch.allclose(p.shift,(q*mean).sum(-1)/8)
assert p.qi.shape==(1,4,128,64) and p.counts.tolist()==[64,1]
assert torch.equal(p.ki[:,0],p.ki[:,1])
assert (p.ki[:,:,65:]==0).all()
original=q@expanded.transpose(-1,-2)/8
centered=q@(expanded-mean).transpose(-1,-2)/8
assert torch.allclose(original.softmax(-1),centered.softmax(-1),atol=2e-6)
# Block maxima must include every channel and only the actual ragged tokens.
x=torch.tensor([[[[127.,.5,-.5],[1.,2.,3.],[4.,5.,6.]]]])
qi,scale=quant_blocks(x,2)
assert qi[0,0,0].tolist()==[127,1,-1]
assert scale[0,0,2]==6/127
assert torch.equal(scale[...,0],scale[...,1])
print('PASS Sage-inspired centering, ragged scales, rounding, GQA, softmax invariance')
