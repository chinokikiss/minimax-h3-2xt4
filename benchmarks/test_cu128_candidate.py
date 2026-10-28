"""CPU contracts: explicit alias is isolated from existing global dispatch."""
import unittest
from types import SimpleNamespace
from patches.kitchen_cu128_candidate import _register,ALIAS,Candidate


class CandidateContract(unittest.TestCase):
    def test_public_linear_default_dtype_is_preserved(self):
        import torch
        def public(x,out_dtype=None):return None
        def implementation(**kwargs):return kwargs['out_dtype']
        registry=SimpleNamespace(get_implementation=lambda *args,**kwargs:implementation)
        candidate=Candidate.__new__(Candidate)
        candidate.ck=SimpleNamespace(int8_linear=public,registry=registry)
        candidate.operations={'int8_linear'}
        self.assertIs(candidate.call('int8_linear',1),torch.bfloat16)
        self.assertIs(candidate.call('int8_linear',1,out_dtype=torch.float16),torch.float16)

    def test_alias_does_not_enable_global_cuda_or_change_automatic_selection(self):
        from comfy_kitchen.registry import BackendRegistry
        def eager(x):return ('eager',x)
        def cuda(x):return ('cuda',x)
        registry=BackendRegistry()
        backend=SimpleNamespace(int8_linear=cuda,_build_constraints=lambda:{'int8_linear':None})
        registry.register('eager',SimpleNamespace(int8_linear=eager),{'int8_linear':None})
        registry.register('cuda',backend,{'int8_linear':None});registry.disable('cuda')
        ck=SimpleNamespace(registry=registry,backends=SimpleNamespace(cuda=backend))
        _register(ck,('int8_linear',))
        self.assertTrue(registry.list_backends()['cuda']['disabled'])
        self.assertIs(registry.get_implementation('int8_linear'),eager)
        self.assertIs(registry.get_implementation('int8_linear',backend=ALIAS),cuda)
        with self.assertRaises(ValueError):_register(ck,('unknown',))

    def test_candidate_has_no_default_enable_hook(self):
        from unittest.mock import patch
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError):Candidate(('int8_linear',))

    def test_pinned_cuda_rms_rope_rejects_fp32_even_when_enabled(self):
        import torch
        import comfy_kitchen as ck
        constraint=ck.backends.cuda._build_constraints()['rms_rope_split_half']
        self.assertNotIn(torch.float32,constraint.params['q'].dtypes)
        self.assertIn(torch.float16,constraint.params['q'].dtypes)

    def test_native_conv_requires_its_separate_optin(self):
        from unittest.mock import Mock,patch
        import comfy.ldm.minimax.vae as native
        kernel=Mock()
        with patch.object(native.comfy.quant_ops,'ck',SimpleNamespace(fp16_conv3d=kernel)), \
                patch.object(native.comfy.ops,'_fp16_linear_wanted',return_value=False):
            self.assertIsNone(native._fp16_accum_conv(None,None,None,None,None))
        kernel.assert_not_called()


if __name__=='__main__':unittest.main()
