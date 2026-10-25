"""Small CUDA block comparison against preserved source; never a generation run."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from patches.tp_dit import TPDiTBlockINT8
from patches.h3_tensor_flow import TensorFlowPolicy,prepare_tp
from patches.h3_operator_policy import H3OperatorPolicy,configure_block
from patches.dit_column_linear import column_linear
import comfy_kitchen as ck


class SmallAdapter:
    def __init__(self,width,out):
        self.a=torch.randn(width,4,device='cuda')*.001
        self.b=torch.randn(4,out,device='cuda')*.001

    def add_to(self,x,out):
        return out+(x.float()@self.a@self.b).to(out.dtype)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(1023)
    spec=importlib.util.spec_from_file_location('h3_lifetime_reference_dit',args.baseline)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    result=dict(scope='Small block / nondefault stream local CUDA gates, not deployment memory evidence',
        device=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),cases=[])
    for mode in ('legacy','fp16'):
        for online_lora in (False,True):
            kwargs=dict(hidden=256,heads=2,head_dim=128,ffn=256,t_dim=8,world_size=1,device='cuda')
            actual=TPDiTBlockINT8(**kwargs);expected=reference.TPDiTBlockINT8(**kwargs)
            for name,param in actual.named_parameters():
                if param.dtype==torch.int8:param.random_(-127,128)
                elif name.endswith('weight_scale'):param.uniform_(.0001,.0003)
                elif 'adaln_proj' in name:param.normal_(0,.01)
            expected.load_state_dict(actual.state_dict())
            flow=TensorFlowPolicy(mode)
            for block in (actual,expected):
                prepare_tp(block,'dit',flow)
                configure_block(block,'dit',H3OperatorPolicy(dit_attention='native'))
            if online_lora:
                for name in ('qkv_proj','fc1'):
                    projection=getattr(actual,name)
                    adapter=SmallAdapter(projection.in_features,projection.out_features_per_rank)
                    projection.lora_adapter=getattr(expected,name).lora_adapter=adapter
            for rows in (33,513):
                x=torch.randn(rows,256,device='cuda',dtype=flow.dtype if flow.enabled else torch.float32)
                t=torch.randn(3,8,device='cuda');segments=[(0,rows,0)]
                with ck.registry.use_backend('eager'):
                    a=expected(x.clone(),t,segments);b=actual(x.clone(),t,segments)
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                unequal=int(torch.count_nonzero(a!=b))
                result['cases'].append(dict(mode=mode,online_lora=online_lora,rows=rows,
                    unequal_values=unequal,finite=finite))
                assert unequal==0 and finite
            del actual,expected
    # Churn storage on a nondefault stream after early releases. Both variants
    # use the same stream and synchronization occurs only at the test boundary.
    stream=torch.cuda.Stream();x=torch.randn(513,512,device='cuda',dtype=torch.float16)
    w=torch.randint(-127,128,(768,512),device='cuda',dtype=torch.int8);s=torch.rand(768,1,device='cuda')*.001
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream),ck.registry.use_backend('eager'):
        a=ck.int8_linear(x,w,s,None,torch.float16,convrot=True)
        b=column_linear(x,w,s)
        churn=torch.empty((513,768),device='cuda',dtype=torch.int32);churn.fill_(111)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(a,b)
    result.update(status='passed',nondefault_stream_exact=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print('DIT_SMALL_BLOCK_GATE_PASS',len(result['cases']),'cases and nondefault stream')


if __name__=='__main__':main()
