import unittest
import torch
import torch.nn.functional as F
from .te_precision import fixture,mask_for,metrics,reference


class TEPrecisionOracle(unittest.TestCase):
    def test_explicit_GQA_matches_independent_SDPA(self):
        q,k,v=fixture(65,torch.float32,'cpu',831,h=4,hk=2)
        for kind in ('bool','additive','padding_bias'):
            mask=mask_for(65,kind,'cpu',q.dtype)
            expected=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True)
            actual=reference(q,k,v,mask)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)

    def test_reference_future_edits_and_truncation(self):
        q,k,v=fixture(65,torch.float32,'cpu',831,h=4,hk=2)
        for kind in ('bool','additive','padding_bias'):
            mask=mask_for(65,kind,'cpu',q.dtype);base=reference(q,k,v,mask)
            for index in range(3):
                changed=[x.clone() for x in (q,k,v)];changed[index][:,:,31:]*=32
                actual=reference(*changed,mask)
                torch.testing.assert_close(actual[:,:,:31],base[:,:,:31],rtol=0,atol=0)
            short=tuple(x[:,:,:31] for x in (q,k,v))
            actual=reference(*short,mask[...,:31,:31])
            torch.testing.assert_close(actual,base[:,:,:31],rtol=2e-5,atol=2e-6)

    def test_FP32_oracle_agrees_with_FP64(self):
        q,k,v=fixture(65,torch.float32,'cpu',731,h=4,hk=2)
        mask=mask_for(65,'bool','cpu',q.dtype)
        actual=reference(q,k,v,mask);expected=reference(q,k,v,mask,dtype=torch.float64)
        torch.testing.assert_close(actual.double(),expected,rtol=2e-5,atol=2e-6)

    def test_masks_have_a_valid_causal_key_per_row(self):
        for dtype in (torch.float16,torch.float32):
            for kind in ('bool','additive','padding_bias'):
                mask=mask_for(65,kind,'cpu',dtype)
                allowed=mask if mask.dtype==torch.bool else mask>torch.finfo(dtype).min/8
                self.assertTrue(allowed.any(-1).all())
                self.assertTrue((~allowed[...,torch.ones(65,65,dtype=torch.bool).triu(1)]).all())

    def test_error_metrics_report_changes_and_zero_reference(self):
        result=metrics(torch.zeros(3),torch.zeros(3))
        self.assertTrue(result['byte_equal'] and result['finite']);self.assertEqual(result['relative_l2'],0)
        result=metrics(torch.tensor([0.,2.,0.]),torch.tensor([0.,1.,0.]))
        self.assertFalse(result['byte_equal']);self.assertEqual(result['max_abs'],1)
        self.assertEqual(result['relative_l2'],1)


if __name__=='__main__':unittest.main()
