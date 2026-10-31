"""Local tiny-block wiring with two logical owners; not physical TP2 evidence."""
import json
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import os
from importlib.metadata import version
import hashlib
import inspect
from unittest.mock import patch
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks'),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI'))]
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from patches.h3_operator_policy import H3OperatorPolicy,configure_block,KitchenRefinerAttention
from patches import h3_operator_policy as dispatch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,
        default=Path('D:/attention_eval/h3_operator_enhancements_20261019/local_pipeline_wiring.json'))
    args=parser.parse_args();sys.argv=sys.argv[:1]
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    from patches import tp_dit as dit,tp_qwen3vl as te,optimized_tp
    comm=LogicalDualGPU(timeout=60);te.dist=dit.dist=comm
    isolate_cublas_workspaces();optimized_tp.enable()
    policy=H3OperatorPolicy()
    blocks=[configure_block(dit.TPDiTBlockINT8(hidden=256,heads=4,head_dim=64,ffn=512,
        rank=r,device='cuda:0'),'dit',policy) for r in (0,1)]
    torch.manual_seed(286)
    def initialize(block):
        for name,p in block.named_parameters():
            if p.dtype==torch.int8:p.copy_(torch.randint(-8,9,p.shape,device=p.device,dtype=p.dtype))
            elif name.endswith('weight_scale'):p.fill_(.002)
            elif 'norm' in name:p.fill_(1.)
            else:p.normal_(0,.01)
    initialize(blocks[0]);blocks[1].load_state_dict(blocks[0].state_dict())
    x=torch.randn(17,256,device='cuda');t=torch.randn(1,8,device='cuda')
    result=comm.run([lambda r=r:blocks[r](x.clone(),t,[(0,17,0)]) for r in (0,1)])
    torch.testing.assert_close(result[0],result[1],rtol=0,atol=0)
    assert torch.isfinite(result[0]).all()
    records=[dict(name='DiT_Kitchen_and_row_call_sites',status='pass')]
    del blocks,result
    te_policy=H3OperatorPolicy(te_attention='native')
    blocks=[configure_block(te.TPTransformerBlockINT8(hidden_size=256,intermediate_size=512,
        num_heads=4,num_kv_heads=2,head_dim=64,rank=r,device='cuda:0'),'te',te_policy) for r in (0,1)]
    for b in blocks:b.to('cuda:0')
    initialize(blocks[0]);blocks[1].load_state_dict(blocks[0].state_dict())
    x=torch.randn(1,17,256,device='cuda');mask=torch.ones(17,17,device='cuda',dtype=torch.bool).tril()
    rope=(torch.ones(1,1,17,64,device='cuda'),torch.zeros(1,1,17,32,device='cuda'),torch.zeros(1,1,17,32,device='cuda'))
    result=comm.run([lambda r=r:blocks[r](x.clone(),mask,rope) for r in (0,1)])
    torch.testing.assert_close(result[0],result[1],rtol=0,atol=0)
    assert torch.isfinite(result[0]).all()
    records.append(dict(name='TE_Native_GQA_mask_row_parts_call_sites',status='pass'))
    # Exercise the actual Native refiner producer and residency copy of its wrapper.
    import comfy.ops
    from comfy.ldm.minimax.model import Attention,rope_rotation_table
    from patches.weight_residency import copy_module
    base=Attention(256,4,64,1e-5,device='cuda',dtype=torch.float32,operations=comfy.ops.disable_weight_init).eval().requires_grad_(False)
    initialize(base);wrapper=copy_module(KitchenRefinerAttention(base),lambda z:z.clone())
    def native_sdpa(q,k,v,heads,**kwargs):
        q,k,v=(a.peek() if hasattr(a,'peek') else a for a in (q,k,v))
        return torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(1,17,256)
    for frequencies in (None,rope_rotation_table(torch.randn(17,32,device='cuda'),torch.float32)):
        refiner_x=torch.randn(17,256,device='cuda')
        with patch('comfy.ldm.minimax.model.optimized_attention',new=native_sdpa):
            expected=base(refiner_x,rope_freqs=frequencies)
        with patch('comfy_kitchen.int8_attention',side_effect=AssertionError('Refiner must use native SDPA')), \
                patch('torch.nn.functional.scaled_dot_product_attention',wraps=torch.nn.functional.scaled_dot_product_attention) as sdpa:
            y=wrapper(refiner_x,rope_freqs=frequencies)
            assert sdpa.call_count==1
        assert torch.equal(y,expected) and torch.isfinite(y).all()
    records.append(dict(name='Native_refiner_SDPA_legacy_wrapper_alias_after_residency_copy',status='pass'))
    # Current ComfyUI has inline input projections, unlike the pinned T4 source.
    # Compare its real zero-block Native forward before/after the TP bridge.
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from patches.windowed_tp import DiTPreprocessing
    cpu=MiniMaxH3Model(hidden_size=256,num_layers=0,token_refiner_num_layers=0,
        num_attention_heads=4,attention_head_dim=64,ffn_hidden_size=512,text_dim=256,
        time_embed_dim=8,adaln_curve_grid=1025,dtype=torch.float32,device='cpu',operations=comfy.ops.disable_weight_init)
    packed_preprocessing=hasattr(cpu,'_embed_and_pack')
    initialize(cpu)
    for buffer in cpu.buffers():
        if buffer.is_floating_point():buffer.fill_(.01)
    native=copy_module(cpu,lambda t:t.to('cuda'))
    sample=[torch.randn(1,24,1,4,4,device='cuda'),torch.randn(1,32,2,3,device='cuda')]
    context=torch.randn(1,3,256,device='cuda');timestep=torch.tensor([500.],device='cuda')
    reference=native._forward(sample,timestep,context,transformer_options={})
    owners=[];models=[]
    protected={'condition_proj','token_refiner','video_patch_proj','audio_patch_proj'}
    for rank in (0,1):
        m=copy_module(cpu,lambda t:t.clone());owner=DiTPreprocessing(m,rank,'cuda:0',comm,'tiny-inline')
        for name,child in list(m.named_children()):
            if name not in protected:setattr(m,name,copy_module(child,lambda t:t.to('cuda')))
        for name,value in list(m._buffers.items()):
            if value is not None:m._buffers[name]=value.to('cuda')
        models.append(m);owners.append(owner)
    class ProjectionWindow:
        calls=0
        @contextmanager
        def use(self,key):
            self.calls+=1
            group=nn.ModuleDict({n:owners[0].canonical[n] for n in ('video_patch_proj','audio_patch_proj')})
            yield copy_module(group,lambda t:t.to('cuda'))
    owners[0].window=ProjectionWindow()
    result=comm.run([lambda r=r:models[r]._forward(sample,timestep,context,transformer_options={}) for r in (0,1)])
    for output in result:
        for got,want in zip(output,reference):torch.testing.assert_close(got,want,rtol=0,atol=0)
    assert owners[0].window.calls==owners[0].project_uses==(1 if packed_preprocessing else 2)
    for owner in owners:owner.close()
    records.append(dict(name='Native_preprocessing_bridge_zero_block_forward',status='bitwise_pass',
        packed_preprocessing=packed_preprocessing,projection_leases=owners[0].project_uses))
    out=args.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(scope='Tiny seeded blocks on one SM120 GPU, two logical owners. No full pipeline or target performance claim.',
        torch=torch.__version__,comfy_kitchen=version('comfy-kitchen'),tilelang=version('tilelang'),hardware=torch.cuda.get_device_name(),
        native_model_source=inspect.getfile(MiniMaxH3Model),native_model_sha256=hashlib.sha256(Path(inspect.getfile(MiniMaxH3Model)).read_bytes()).hexdigest(),records=records),indent=2))
    print('H3_PIPELINE_WIRING_PASS',len(records),flush=True)


if __name__=='__main__':main()
