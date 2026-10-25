"""Production dispatch and storage boundaries for the promoted FP16-fix default."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from patches import h3_operator_policy as policy,tilelang_swiglu_quant as fused

ROOT=Path(__file__).resolve().parents[1]


class DefaultFusedRow(unittest.TestCase):
    def test_promoted_kernel_is_the_measured_kernel(self):
        def kernel(path):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fused_kernel'))
        self.assertEqual(kernel(ROOT/'patches/tilelang_swiglu_quant.py'),
            kernel(ROOT/'benchmarks/h3_fp16_fc2_experiment.py'))

    def test_dit_lora_is_added_in_fp32_before_one_output_store(self):
        from patches.tp_dit import RowParallelLinearINT8
        module=RowParallelLinearINT8(512,128,device='cpu')
        module.operator_policy=policy.H3OperatorPolicy()
        x=torch.full((3,512),1000.,dtype=torch.float16)
        class Adapter:
            def add_to(self,source,out,comm,input_act):
                self.dtypes=(source.dtype,out.dtype);self.calls=getattr(self,'calls',0)+1
                return out-50000.
        adapter=Adapter();module.lora_adapter=adapter
        def project(value,*args,mode,scaled_transport):
            self.assertEqual(value.dtype,torch.float16)
            self.assertEqual(mode,'fused_stored');self.assertTrue(scaled_transport)
            return torch.full((3,128),70000.,dtype=torch.float32)
        with patch.object(fused,'supported',return_value=True),patch.object(fused,'row_linear',side_effect=project):
            out=module(x,input_act='swiglu')
        self.assertEqual(adapter.dtypes,(torch.float32,torch.float32));self.assertEqual(adapter.calls,1)
        self.assertEqual(out.dtype,torch.float16)
        self.assertTrue(torch.equal(out,torch.full_like(out,20000.)))

    def test_te_parts_are_packed_before_product_and_store_fp16(self):
        from patches.tp_qwen3vl import RowParallelLinearINT8
        module=RowParallelLinearINT8(512,128)
        module.operator_policy=policy.H3OperatorPolicy()
        gate=torch.full((1,3,256),1000.,dtype=torch.float16)
        def project(value,*args,**kwargs):
            self.assertEqual(value.dtype,torch.float16)
            self.assertTrue(torch.equal(value,torch.cat((gate,gate),-1)))
            return torch.ones((1,3,128),dtype=torch.float32)
        with patch.object(fused,'supported',return_value=True),patch.object(fused,'row_linear',side_effect=project):
            out=module.forward_parts(gate,gate)
        self.assertEqual(out.dtype,torch.float16)

    def test_fallback_cast_precedes_swiglu(self):
        module=SimpleNamespace(operator_policy=policy.H3OperatorPolicy(),weight=torch.empty(128,256,dtype=torch.int8),
            weight_scale=torch.ones(128,1),world_size=2,convrot_groupsize=256)
        x=torch.full((3,512),1000.,dtype=torch.float16)
        def fallback(value,*args):
            self.assertEqual(value.dtype,torch.float32);self.assertEqual(args[-1],'swiglu')
            g,u=value.chunk(2,-1);self.assertTrue(torch.isfinite(torch.nn.functional.silu(g)*u).all())
            return torch.zeros(3,128)
        with patch.object(fused,'supported',return_value=False),patch.object(policy,'sm75',return_value=False):
            policy.row_linear(module,x,fallback,object(),'swiglu')

    def test_scaled_attention_output_row_accepts_fp16_storage(self):
        from patches import tilelang_transport_p2 as transport
        module=SimpleNamespace(operator_policy=policy.H3OperatorPolicy(),weight=None,weight_scale=None,
            world_size=2,convrot_groupsize=256)
        x=torch.ones(3,256,dtype=torch.float16)
        with patch.object(policy,'sm75',return_value=True),patch.object(transport,'row_linear',return_value='projected') as run:
            self.assertEqual(policy.row_linear(module,x,None,object()),'projected')
        self.assertEqual(run.call_args.args[0].dtype,torch.float32)

    def test_explicit_previous_policy_retains_its_dispatch(self):
        from patches import tilelang_swiglu_pad as previous
        module=SimpleNamespace(operator_policy=policy.H3OperatorPolicy(row_linear_backend='tilelang_packed',fc2_quantizer='owned_fp32'),
            weight=None,weight_scale=None,world_size=2,convrot_groupsize=256)
        x=torch.ones(3,512)
        with patch.object(policy,'sm75',return_value=True),patch.object(previous,'row_linear',return_value='old') as run:
            self.assertEqual(policy.row_linear(module,x,None,object(),'swiglu'),'old')
        self.assertIs(run.call_args.args[0],x)
        self.assertEqual(run.call_args.kwargs,dict(variant='packed',rotation_mode='owned_fp32'))
        self.assertNotEqual(module.operator_policy,replace(module.operator_policy,fc2_quantizer='fused_stored'))

    def test_core_gate_accepts_preserved_versions_and_rejects_drift(self):
        from patches import h3_operator_dependencies as dependencies
        with patch.object(dependencies,'version',side_effect=lambda n:{'comfy-kitchen':'0.2.37','tilelang':'0.1.13'}[n]):
            dependencies.validate_dependencies(policy.H3OperatorPolicy(),t4=True)
        with patch.object(dependencies,'version',side_effect=lambda n:{'comfy-kitchen':'0.2.37','tilelang':'0.1.15'}[n]):
            with self.assertRaises(RuntimeError):dependencies.validate_dependencies(policy.H3OperatorPolicy(),t4=True)

    def test_native_fp32_experiments_do_not_inherit_lossy_default(self):
        for name in ('run_h3_native_fp32_flow.py','run_h3_physical_fp16_flow.py',
                     'run_t4_fp16_lora_diagnosis.py','run_t4_fc2_fusion.py'):
            tree=ast.parse((ROOT/'benchmarks'/name).read_text(encoding='utf-8'))
            policies=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Name) and n.func.id=='H3OperatorPolicy']
            self.assertTrue(policies,name)
            for call in policies:
                kwargs={k.arg:ast.literal_eval(k.value) for k in call.keywords}
                self.assertEqual(kwargs['row_linear_backend'],'tilelang_packed',name)
                self.assertEqual(kwargs['fc2_quantizer'],'owned_fp32',name)


if __name__=='__main__':unittest.main()
