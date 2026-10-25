"""Reference mask/GQA/cast contracts, independent of model checkpoints."""
from pathlib import Path
import sys,unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'benchmarks')]
import torch
from torch.nn import functional as F
from fp32_ground_truth import query_chunked_sdpa
from fp32_ground_truth import dense_fp32


class ReferenceContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        for owner in (torch.backends.cuda.matmul,torch.backends.cudnn):
            original=owner.allow_tf32
            self.addCleanup(setattr,owner,'allow_tf32',original)
            owner.allow_tf32=False
    def check(self,**kwargs):
        q=torch.randn(1,4,17,16);k=torch.randn(1,2 if kwargs.get('enable_gqa') else 4,23,16);v=torch.randn_like(k)
        expected=F.scaled_dot_product_attention(q,k,v,**kwargs)
        actual=query_chunked_sdpa(F.scaled_dot_product_attention,q,k,v,query_chunk=5,**kwargs)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    def test_noncausal(self):self.check()
    def test_causal_global_offsets(self):self.check(is_causal=True)
    def test_gqa(self):self.check(enable_gqa=True)
    def test_boolean_mask(self):self.check(attn_mask=torch.rand(17,23)>.2)
    def test_additive_mask(self):self.check(attn_mask=torch.randn(1,1,17,23))
    def test_broadcast_mask(self):self.check(attn_mask=torch.randn(1,1,1,23))
    def test_reject_low_precision(self):
        q=torch.randn(1,2,3,8).half()
        with self.assertRaises(ValueError):query_chunked_sdpa(F.scaled_dot_product_attention,q,q,q)
    def test_dense_context_verifies_and_restores_flags(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ComfyUI'))
        model=torch.nn.Linear(8,4);model._full_precision_mm=False
        with dense_fp32(model) as proof:
            self.assertTrue(model._full_precision_mm)
            output=model(torch.randn(3,8))
        self.assertFalse(model._full_precision_mm)
        self.assertEqual(output.dtype,torch.float32)
        self.assertEqual(proof['dense_linear_calls'],1)
        self.assertEqual(proof['int8_linear_calls'],0)
    def test_dense_context_rejects_quantized_reference(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ComfyUI'))
        import comfy_kitchen as ck
        model=torch.nn.Linear(8,4)
        with dense_fp32(model):
            with self.assertRaisesRegex(ValueError,'INT8 activation GEMM'):ck.int8_linear(None,None,None)


if __name__=='__main__':unittest.main()
