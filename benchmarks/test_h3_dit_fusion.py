"""Production dispatch, numerical boundaries and lifetime tests; no GPU claims."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import weakref
import torch
from torch import nn

from patches import tilelang_dit_fusion as fusion,h3_tensor_flow as flow
from patches.h3_operator_policy import H3OperatorPolicy


class ProductionFusion(unittest.TestCase):
    def test_promoted_factories_match_measured_factories(self):
        root=Path(__file__).resolve().parents[1]
        def functions(path):
            return {n.name:ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        measured=functions(root/'benchmarks/dit_vram_kernels.py')
        promoted=functions(root/'patches/tilelang_dit_fusion.py')
        for name in ('norm_kernel','gate_kernel','row_gate_kernel'):
            self.assertEqual(promoted[name],measured[name])

    @torch.no_grad()
    def test_scaled_transport_consumes_fused_epilogue_after_the_existing_sum(self):
        from patches import tilelang_transport_p2 as codec,tilelang_swiglu_pad as quant,parallel_int8,optimized_tp
        value=torch.ones(2,3584);weight=torch.empty(5376,3584,dtype=torch.int8)
        codes=torch.ones(2,3584,dtype=torch.int8);scale=torch.ones(2,1)
        partial=torch.ones(2,5376,dtype=torch.int32);payload=partial.half()
        result=torch.ones(2,5376,dtype=torch.float16)
        comm=Mock();order=[]
        def collect(*args):order.append('sum');return payload
        def finish(p,a,w,k):
            self.assertEqual(order,['sum']);self.assertIs(p,payload)
            self.assertIs(a,scale);self.assertEqual(k,3584)
            order.append('fused_epilogue');return result
        with patch.object(codec,'_supported',return_value=True),patch.object(optimized_tp,'use_fallback_rotation',return_value=True), \
                patch.object(quant,'global_row_quantize',return_value=(codes,scale)), \
                patch.object(parallel_int8,'int_mm',return_value=partial),patch.object(codec,'encode',return_value=payload), \
                patch.object(codec,'collect',side_effect=collect),patch.object(codec,'finish',side_effect=AssertionError('Unfused epilogue')):
            actual=codec.row_linear(value,weight,torch.ones(5376,1),comm,2,finish_fn=finish)
        self.assertEqual(order,['sum','fused_epilogue'])
        self.assertTrue(torch.equal(actual,result))

    def test_defaults_switches_and_aot_are_consistent(self):
        from patches.h3_operator_aot import plan
        import custom_nodes.minimax_h3_tp2 as node
        policy=H3OperatorPolicy()
        widgets=list(node.MiniMaxH3TP2Loader.INPUT_TYPES()['optional'])
        self.assertEqual(widgets[-5:],['aot_precompile','dit_fused_norm','dit_fused_gate','dit_fused_row_gate','dit_prepared_attention'])
        for key in ('dit_fused_norm','dit_fused_gate','dit_fused_row_gate','dit_prepared_attention'):
            self.assertTrue(getattr(policy,key))
            self.assertTrue(node.MiniMaxH3TP2Loader.INPUT_TYPES()['optional'][key][1]['default'])
            self.assertNotEqual(policy,replace(policy,**{key:False}))
            with self.assertRaises(ValueError):H3OperatorPolicy(**{key:1})
        selected=[c for c in plan(59134,0,policy,norm_epsilon=1e-5/256) if c[0]=='tilelang_dit_fusion']
        self.assertEqual(len(selected),4)
        self.assertIn(('tilelang_dit_fusion','norm_kernel',(59134,5376,9,1e-5/256)),selected)
        disabled=replace(policy,dit_fused_norm=False,dit_fused_gate=False,dit_fused_row_gate=False)
        self.assertFalse(any(c[0]=='tilelang_dit_fusion' for c in plan(59134,0,disabled)))

    @torch.no_grad()
    def test_cpu_bf16_and_fp32_fallback_preserve_existing_arithmetic(self):
        segments=[(0,2,0),(2,5,1)]
        norm=nn.RMSNorm(32)
        for dtype in (torch.float16,torch.bfloat16,torch.float32):
            x=torch.randn(5,32,dtype=dtype);shift=torch.randn(3,32);scale=torch.randn(3,32)
            reference=flow.modulated_norm(norm,x,shift,scale,segments)
            self.assertTrue(torch.equal(flow.modulated_norm(norm,x,shift,scale,segments,fused=True),reference))
            gate=torch.randn(3,32);branch=torch.randn_like(x)
            self.assertTrue(torch.equal(flow.gated_add(x.clone(),gate,branch,segments,fused=True),
                                        flow.gated_add(x.clone(),gate,branch,segments)))

    def test_labels_reject_gaps_overlaps_and_unknown_modulation_rows(self):
        self.assertEqual(fusion.segment_key(5,[(0,2,7),(2,5,0)]),((0,2,7),(2,5,0)))
        for segments in ([(0,2,0)],[(0,3,0),(2,5,1)],[(0,5,9)]):
            self.assertIsNone(fusion.segment_key(5,segments))

    @torch.no_grad()
    def test_row_fusion_uses_the_selected_transport_and_fp16_branch_boundary(self):
        from patches import tilelang_transport_p2 as codec,tilelang_swiglu_quant as quant
        from patches.tilelang_transport_tradeoffs import shift_for
        rows=2;gate=torch.randn(9,5376);labels=torch.tensor([7,0],dtype=torch.int32)
        scale=torch.tensor([[.001],[.002]])
        weights=torch.full((5376,1),.005)
        payload=torch.full((rows,5376),.5,dtype=torch.float16)
        for input_act,width in ((None,3584),('swiglu',7168)):
            x=torch.randn(rows,5376,dtype=torch.float16)
            value=torch.randn(rows,width*(2 if input_act else 1),dtype=torch.float16)
            module=SimpleNamespace(operator_policy=H3OperatorPolicy(),world_size=2,convrot=True,
                convrot_groupsize=256,weight=torch.empty(5376,width,dtype=torch.int8),weight_scale=weights)
            restored=(payload.float()*2.**shift_for(width)).int()
            branch=((restored.float()*scale)*weights.T).half()
            expected=torch.addcmul(x.float(),branch.float(),gate[labels.long()]).half()
            def kernel(n,h,m,shift):
                def store(p,a,w,destination,g,l):
                    b=(((p.float()*2.**shift).int().float()*a)*w).half()
                    destination.copy_(torch.addcmul(destination.float(),b.float(),g[l.long()]).half())
                return store
            def run(source,*args,**kwargs):
                self.assertEqual(source.dtype,torch.float16 if input_act else torch.float32)
                if input_act:self.assertTrue(kwargs['scaled_transport'])
                return kwargs['finish_fn'](payload,scale,weights,width)
            with patch.object(fusion,'supported',return_value=True),patch.object(fusion,'labels_for',return_value=labels), \
                    patch.object(fusion,'row_gate_kernel',side_effect=kernel), \
                    patch.object(codec,'row_linear',side_effect=run) as ordinary, \
                    patch.object(quant,'row_linear',side_effect=run) as swiglu:
                result=fusion.row_gated(module,value,x,gate,[(0,1,7),(1,2,0)],object(),input_act)
            self.assertIs(result,x)
            self.assertTrue(torch.equal(result,expected))
            self.assertEqual(ordinary.call_count,0 if input_act else 1)
            self.assertEqual(swiglu.call_count,1 if input_act else 0)

    @torch.no_grad()
    def test_online_lora_and_int32_transport_refuse_fusion_before_collectives(self):
        for adapter,backend in ((object(),'tilelang_scaled_fp16'),(None,'tilelang_packed')):
            module=SimpleNamespace(operator_policy=H3OperatorPolicy(row_linear_backend=backend),lora_adapter=adapter)
            with patch('patches.tilelang_transport_p2.row_linear') as run:
                self.assertIsNone(fusion.row_gated(module,None,None,None,None,Mock()))
                run.assert_not_called()

    @torch.no_grad()
    def test_lora_addition_precedes_storage_cast_and_gate_on_fallback(self):
        from patches.tp_dit import RowParallelLinearINT8
        module=RowParallelLinearINT8(512,128,device='cpu')
        module.operator_policy=H3OperatorPolicy()
        class Adapter:
            def add_to(self,source,out,**kwargs):
                self_test.assertEqual(out.dtype,torch.float32)
                return out-50000.
        self_test=self;module.lora_adapter=Adapter()
        x=torch.ones(3,256,dtype=torch.float16)
        residual=torch.ones(3,128,dtype=torch.float16)
        gates=torch.full((3,128),.5)
        with patch('patches.h3_operator_policy.row_linear',return_value=torch.full((3,128),70000.)):
            result=module(x,residual=residual,gate=gates,segments=[(0,3,0)])
        self.assertTrue(torch.equal(result,torch.full_like(result,10000.)))
        self.assertTrue(torch.isfinite(result).all())

    @torch.no_grad()
    def test_prepared_attention_releases_qkv_before_execute_and_flattens_output(self):
        from patches.tp_dit import TPDiTBlockINT8
        block=TPDiTBlockINT8(hidden=32,heads=2,head_dim=16,ffn=32,t_dim=8,world_size=1,device='cpu')
        block.tensor_flow=flow.TensorFlowPolicy('fp16')
        block.operator_policy=H3OperatorPolicy()
        refs={}
        class Projection(nn.Module):
            def __init__(self,name,width):super().__init__();self.name=name;self.width=width
            def forward(self,value,**kwargs):
                if self.name=='out':self_test.assertEqual(tuple(value.shape),(5,32))
                if 'residual' in kwargs:return kwargs['residual']
                out=torch.ones(value.shape[0],self.width,dtype=torch.float16)
                if self.name=='qkv':refs['qkv']=weakref.ref(out)
                return out
        self_test=self
        block.qkv_proj=Projection('qkv',96);block.out_proj=Projection('out',32)
        block.fc1=Projection('fc1',64);block.fc2=Projection('fc2',32)
        def prepare(_,q,k,v,options=None):
            refs.update(q=weakref.ref(q),k=weakref.ref(k),v=weakref.ref(v))
            return object()
        def execute(prepared):
            for key in ('qkv','q','k','v'):self.assertIsNone(refs[key](),key)
            return torch.ones(1,5,32,dtype=torch.float16)
        with patch('patches.h3_operator_policy.prepare_dit_attention',new=prepare), \
                patch('patches.h3_operator_policy.execute_dit_attention',new=execute), \
                patch('patches.h3_operator_policy.dit_attention',side_effect=AssertionError('Wrong attention route')):
            result=block(torch.ones(5,32,dtype=torch.float16),torch.ones(3,8),[(0,5,0)])
        self.assertEqual(result.shape,(5,32))

    def test_production_never_imports_experiment_kernels(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('tilelang_dit_fusion','tp_dit','h3_tensor_flow','h3_operator_policy'):
            self.assertNotIn('from benchmarks', (root/'patches'/f'{name}.py').read_text())


if __name__=='__main__':unittest.main(verbosity=2)
