"""Numerical gauge contracts and ownership/copy boundaries; no model download."""
from pathlib import Path
from dataclasses import replace
import sys, unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1])]
import torch
from torch import nn
from patches.h3_tensor_flow import (TensorFlowPolicy,StableNorm,fold_linear,
    prepare_tp,prepare_te,prepare_video,modulated_norm,gated_add,_te_input,_te_output)
from patches.weight_residency import copy_module


class TinyTE(nn.Module):
    def __init__(self):
        super().__init__();self.embed_tokens=lambda x,out_dtype:x.to(out_dtype)
        self.tensor_flow=TensorFlowPolicy('fp16',4)
        self.register_forward_pre_hook(_te_input,with_kwargs=True)
        self.register_forward_hook(_te_output)
    def forward(self,x,embeds=None,deepstack_embeds=None,dtype=None):
        assert embeds.dtype==dtype==torch.float16
        return embeds+deepstack_embeds[0],None


class TensorFlowContracts(unittest.TestCase):
    def test_policy(self):
        self.assertFalse(TensorFlowPolicy().enabled)
        self.assertEqual(TensorFlowPolicy('bf16_reference',0).dtype,torch.bfloat16)
        for args in [('bad',4),('fp16',-1),('fp16',13),('fp16',True)]:
            with self.assertRaises(ValueError):TensorFlowPolicy(*args)

    def test_norms_and_affine_beta(self):
        torch.manual_seed(4);x=torch.randn(7,16)*1.e-4
        for norm in [nn.RMSNorm(16,eps=1.e-5),nn.LayerNorm(16,eps=1.e-5)]:
            if getattr(norm,'bias',None) is not None:norm.bias.data.fill_(.7)
            fixed=StableNorm(norm,16)
            torch.testing.assert_close(fixed(x/16),norm(x),rtol=1.e-6,atol=1.e-7)
            self.assertGreater(fixed.eps,0.)

    def test_dense_fold_bias_and_private_ownership(self):
        layer=nn.Linear(4,6);source=layer.weight.detach().clone();bias=layer.bias.detach().clone()
        x=torch.randn(8,4);expected=layer(x)/16
        fold_linear(layer,16)
        torch.testing.assert_close(layer(x),expected)
        torch.testing.assert_close(layer.weight,source/16)
        torch.testing.assert_close(layer.bias,bias/16)

    def test_int8_fold_preserves_bytes_and_post_b_order(self):
        layer=nn.Module();layer.weight=nn.Parameter(torch.randint(-127,128,(8,16),dtype=torch.int8),requires_grad=False)
        layer.weight_scale=nn.Parameter(torch.ones(8,1),requires_grad=False)
        term=nn.Module();term.scale=.5
        adapter=nn.Module();adapter.terms=nn.ModuleList([term]);layer.lora_adapter=adapter
        original=layer.weight.clone();fold_linear(layer,16)
        self.assertTrue(torch.equal(layer.weight,original));self.assertEqual(term.scale,.5/16)
        self.assertTrue(adapter.fp32_compute)
        torch.testing.assert_close(layer.weight_scale,torch.full((8,1),1/16))

    def test_te_deepstack_and_unscaled_export_survive_copy(self):
        model=TinyTE();clone=copy_module(model,lambda t:t.clone())
        x=torch.full((1,2,8),100000.);deep=torch.full_like(x,20000.)
        for candidate in [model,clone]:
            out,_=candidate(x,deepstack_embeds=[deep])
            self.assertEqual(out.dtype,torch.float32)
            self.assertTrue(torch.isfinite(out).all())
            torch.testing.assert_close(out,x+deep,rtol=.001,atol=0)

    def test_modulation_and_gate_per_token_rows(self):
        x=torch.randn(5,8);norm=nn.RMSNorm(8,eps=1.e-5)
        shift=torch.randn(3,8);scale=torch.randn(3,8);gate=torch.randn(3,8)
        ids=torch.tensor([2,0,1]);segments=[(0,2,0),(2,5,ids)]
        y=x/16;fixed=StableNorm(norm,16)
        h=modulated_norm(fixed,y,shift,scale,segments)
        ref=norm(x)
        for a,b,row in segments:ref[a:b]=ref[a:b]*(1+scale[row])+shift[row]
        torch.testing.assert_close(h,ref,rtol=1.e-6,atol=1.e-6)
        branch=torch.randn_like(x);expected=x.clone()
        for a,b,row in segments:expected[a:b]+=branch[a:b]*gate[row]
        actual=gated_add(y.clone(),gate,branch/16,segments)*16
        torch.testing.assert_close(actual,expected,rtol=1.e-6,atol=1.e-6)

    def test_legacy_is_noop(self):
        block=nn.Linear(2,3);before=block.weight.clone()
        self.assertIs(prepare_tp(block,'dit',TensorFlowPolicy()),block)
        self.assertTrue(torch.equal(block.weight,before));self.assertFalse(hasattr(block,'tensor_flow'))

    def test_video_default_does_not_rewrite_native_weights(self):
        model=nn.Module();model.decoder=nn.Linear(2,3)
        original=model.decoder.weight
        prepare_video(model,TensorFlowPolicy('fp16'))
        self.assertIs(model.decoder.weight,original)
        with self.assertRaises(ValueError):prepare_video(model,TensorFlowPolicy('fp16'))

    def test_flow_changes_residency_identity(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ComfyUI'))
        from patches.windowed_tp import tp_specs
        reader=SimpleNamespace(identity='checkpoint')
        with patch('patches.rank_local_int8.rank_block_bytes',return_value=1024):
            a=tp_specs(reader,'dit',0,1)[0].key
            reader.tensor_flow=TensorFlowPolicy('fp16')
            b=tp_specs(reader,'dit',0,1)[0].key
            reader.tensor_flow=replace(reader.tensor_flow,video_residual_exponent=4)
            c=tp_specs(reader,'dit',0,1)[0].key
        self.assertNotEqual(a,b);self.assertNotEqual(b,c)

    def test_native_lora_fp32_island_precedes_post_b_scale(self):
        from patches.h3_lora import NativeAlignedLoRA
        adapter=NativeAlignedLoRA([(torch.ones(1,4),torch.full((3,1),100000.),1/16)])
        adapter.fp32_compute=True
        x=torch.ones(2,4,dtype=torch.float16);out=torch.zeros(2,3,dtype=x.dtype)
        adapter.add_to(x,out)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out,torch.full_like(out,25000.))

    def test_runtime_defaults_and_bf16_guard(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ComfyUI'))
        from patches.comfy_tp_runtime import RuntimeConfig
        config=RuntimeConfig('te','dit')
        self.assertEqual(config.tensor_flow,TensorFlowPolicy('fp16',6))
        self.assertEqual(config.tensor_flow.vision_residual_exponent,0)
        self.assertEqual(config.tensor_flow.video_residual_exponent,0)
        legacy=replace(config,tensor_flow=TensorFlowPolicy('legacy'))
        self.assertNotEqual(config,legacy)
        self.assertFalse(legacy.tensor_flow.enabled)
        with self.assertRaises(ValueError):replace(config,tensor_flow=TensorFlowPolicy('bf16_reference',0))

    def test_loader_default_and_explicit_legacy_recreate_session(self):
        import importlib.util
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'ComfyUI'))
        spec=importlib.util.spec_from_file_location('h3_default_loader',root/'custom_nodes/minimax_h3_tp2/__init__.py')
        node=importlib.util.module_from_spec(spec);spec.loader.exec_module(node)
        schema=node.MiniMaxH3TP2Loader.INPUT_TYPES()
        self.assertEqual(schema['optional']['tensor_flow'][0][0],'fp16')
        self.assertEqual(schema['optional']['tensor_flow'][1]['default'],'fp16')
        # Old positional widget values must still map to their original fields;
        # missing appended fields receive the new defaults from ComfyUI.
        fields={**schema['required'],**schema['optional']}
        widgets=[name for name,value in fields.items() if name!='lora_stack']
        for name in ('video_minimax_h3_r2v_tp2_int8.json','video_minimax_h3_r2v_tp2_operators.json'):
            workflow=json.loads((root/'workflows'/name).read_text(encoding='utf-8'))
            loader=next(n for n in workflow['nodes'] if n['type']=='MiniMaxH3TP2Loader')
            for field,value in zip(widgets,loader['widgets_values']):
                if field not in ('text_encoder','diffusion_model'):
                    self.assertEqual(value,fields[field][1]['default'],field)
        def service(config):return SimpleNamespace(config=config,closed=False,close=Mock())
        with patch.object(node,'PhysicalTPService',side_effect=service) as create, \
             patch.object(node,'create_model',return_value='model'),patch.object(node,'create_clip',return_value='clip'):
            loader=node.MiniMaxH3TP2Loader()
            first=loader.load('te','dit')[2]
            self.assertEqual(first.config.tensor_flow.mode,'fp16')
            self.assertEqual(first.config.tensor_flow.divisor,64)
            self.assertIs(loader.load('te','dit')[2],first)
            legacy=loader.load('te','dit',tensor_flow='legacy')[2]
            first.close.assert_called_once()
            self.assertFalse(legacy.config.tensor_flow.enabled)
            self.assertEqual(create.call_count,2)

    def test_public_conditioning_stays_finite_and_preserves_native_layout(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ComfyUI'))
        from patches.comfy_sampler import create_model
        from patches.comfy_tp_runtime import RuntimeConfig
        calls=[]
        def call(command,context):
            calls.append((command,context))
            return torch.zeros(1,2,5376,dtype=torch.float16)
        service=SimpleNamespace(config=RuntimeConfig('te','dit',tensor_flow=TensorFlowPolicy('fp16')),call=call)
        model=create_model(service,device='cpu').model
        tags=torch.tensor([0,1]);context=torch.full((1,2,5120),100000.)
        out=model.extra_conds(cross_attn=context,device='cpu',seed=17,minimax_token_tags=tags,
            latent_shapes=[(1,24,1,4,4),(1,32,2)])
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][1].dtype,torch.float32)
        self.assertTrue(torch.isfinite(calls[0][1]).all())
        payload=out['minimax_payload'].cond
        self.assertIn('layout',payload);self.assertEqual(payload['seed'],17)
        self.assertTrue(torch.equal(payload['text_token_tags'],tags))

    def test_pinned_native_llama_deepstack_gauge_boundary(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ComfyUI'))
        import comfy.ops
        from comfy.text_encoders.llama import Llama2_,Qwen3VL_32BConfig
        with torch.device('meta'):
            text=Llama2_(Qwen3VL_32BConfig(num_hidden_layers=0),device='meta',dtype=torch.float16,
                ops=comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.float16))
        seen=[]
        class Block(nn.Module):
            def forward(self,x,**kwargs):seen.append(x.dtype);return x,None
        text.layers=nn.ModuleList([Block()]);text.norm=None
        model=nn.Module();model.model=text;model.visual=nn.Identity()
        prepare_te(model,TensorFlowPolicy('fp16'))
        embeds=torch.full((1,2,5120),100000.)
        deep=torch.full((1,5120),20000.)
        output,_=text(torch.zeros(1,2,dtype=torch.long),embeds=embeds,
            visual_pos_masks=torch.tensor([[True,False]]),deepstack_embeds=[deep])
        self.assertEqual(seen,[torch.float16]);self.assertTrue(torch.isfinite(output).all())
        expected=embeds.clone();expected[:,0]+=deep
        torch.testing.assert_close(output,expected,rtol=.001,atol=0)


if __name__=='__main__':unittest.main()
