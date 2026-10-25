"""Explicit SM120 compile/arithmetic preview only, never target performance evidence."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
from h3_fp16_fc2_experiment import fused_kernel


def main():
    import comfy_kitchen as ck
    assert torch.cuda.get_device_capability()==(12,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False)
    x=torch.randn(3,1024,device='cuda',generator=torch.Generator(device='cuda').manual_seed(74)).half()*4
    x[0,0]=1000;x[0,512]=1000
    gate,up=x.float().chunk(2,-1);activated=torch.nn.functional.silu(gate)*up
    h=ck.backends.cuda._build_hadamard(256,device='cuda',dtype=torch.float32)
    reference=ck.backends.cuda._rotate_activation(activated,h,256)
    rotated,groups=fused_kernel(3,512,'float16','sm_120','stored')(x)
    torch.testing.assert_close(rotated,reference,rtol=2.e-4,atol=.02)
    maximum=groups.amax(-1,keepdim=True)
    unused,other=fused_kernel(3,512,'float16','sm_120','maximum')(x)
    assert unused.numel()==1 and torch.equal(groups,other)
    q,scale=fused_kernel(3,512,'float16','sm_120','recompute')(x,maximum)
    expected=(rotated/scale).round().clamp(-128,127).to(torch.int8)
    assert torch.equal(q,expected) and torch.isfinite(rotated).all()
    print(json.dumps(dict(scope='SM120 local compile/arithmetic only',finite=True,
        max_rotation_error=float((rotated-reference).abs().max()),quantized_exact_to_own_rotation=True,
        above_fp16_products=int((activated.abs()>65504).sum()))))


if __name__=='__main__':main()
