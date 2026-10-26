"""TE policy and mask-preservation contracts; CPU only."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import torch
from patches import kitchen_te_attention as te
from patches import h3_operator_policy as policy


class KitchenTE(unittest.TestCase):
    def setUp(self):te._mask_cache.clear()

    def test_pure_boolean_and_native_additive_triangle(self):
        allowed=torch.ones(7,7,dtype=torch.bool).tril()
        for dtype in (torch.float16,torch.float32,torch.bfloat16):
            for negative in (-float('inf'),torch.finfo(dtype).min/4):
                mask=torch.zeros(2,1,7,7,dtype=dtype).masked_fill(~allowed,negative)
                self.assertTrue(te.plain_causal_mask(mask,7,7))
        self.assertTrue(te.plain_causal_mask(allowed,7,7))

    def test_padding_bias_mutation_and_noncausal_masks_are_preserved(self):
        allowed=torch.ones(7,7,dtype=torch.bool).tril()
        self.assertTrue(te.plain_causal_mask(allowed,7,7))
        allowed[:,0]=False
        self.assertFalse(te.plain_causal_mask(allowed,7,7))
        for mask in (None,torch.ones(7,7,dtype=torch.bool),torch.zeros(7,7),
                     torch.zeros(7),torch.zeros(7,8)):
            self.assertFalse(te.plain_causal_mask(mask,7,7))
        bias=torch.zeros(7,7).masked_fill(torch.ones(7,7,dtype=torch.bool).triu(1),-float('inf'))
        bias[4,0]=.1
        self.assertFalse(te.plain_causal_mask(bias,7,7))

    def test_inference_tensor_mutations_are_not_cached(self):
        with torch.inference_mode():
            mask=torch.ones(7,7,dtype=torch.bool).tril()
            self.assertTrue(te.plain_causal_mask(mask,7,7))
            mask[3,0]=False
            self.assertFalse(te.plain_causal_mask(mask,7,7))

    def test_implicit_triangle_and_general_mask_dispatch(self):
        q,k,v=[torch.randn(1,2,7,64) for _ in range(3)]
        backend=Mock();backend.attention.return_value=q
        plain=torch.ones(7,7,dtype=torch.bool).tril()
        padded=plain.clone();padded[:,0]=False
        with patch.object(te,'_backend_for_device',return_value=backend), \
                patch('comfy_kitchen.int8_attention',return_value=q) as kitchen:
            out=te.attention(q,k,v,plain,use_native=True)
            self.assertEqual(tuple(out.shape),(1,7,128))
            backend.attention.assert_called_once_with(q,k,v,is_causal=True)
            for mask in (None,padded,torch.ones(3,2,7,7,dtype=torch.bool).tril()):
                te.attention(q,k,v,mask,use_native=True)
                self.assertIs(kitchen.call_args.kwargs['attn_mask'],mask)
            self.assertEqual(backend.attention.call_count,1)

    def test_default_never_classifies_or_synchronizes_the_mask(self):
        q,k,v=[torch.randn(1,2,7,64) for _ in range(3)]
        with torch.inference_mode():mask=torch.ones(7,7,dtype=torch.bool).tril()
        with patch.object(te,'_backend_for_device',side_effect=AssertionError('No native lookup in default dispatch')), \
                patch.object(te,'plain_causal_mask',side_effect=AssertionError('No default mask readback')), \
                patch('comfy_kitchen.int8_attention',return_value=q) as kitchen:
            out=te.attention(q,k,v,mask)
            kitchen.assert_called_once_with(q,k,v,attn_mask=mask)
            self.assertEqual(tuple(out.shape),(1,7,128))

    def test_default_policy_calls_kitchen_and_explicit_group32_survives(self):
        self.assertEqual(policy.H3OperatorPolicy().te_attention,'kitchen_causal')
        q=SimpleNamespace(is_cuda=True)
        block=SimpleNamespace(operator_policy=policy.H3OperatorPolicy())
        with patch.object(te,'attention',return_value='result') as call:
            self.assertEqual(policy.te_attention(block,q,'k','v','mask'),'result')
            call.assert_called_once_with(q,'k','v',mask='mask')
        block.operator_policy=policy.H3OperatorPolicy(te_attention='native')
        self.assertIsNone(policy.te_attention(block,q,'k','v','mask'))

    def test_narrow_addressing_rejects_overflow_before_allocation(self):
        from experiments.kitchen_attention.attention import Backend
        device=torch.device('cuda:0');n=2**20;nk=1153;d=128;h=32;hk=4;ck=128
        class Tensor:
            def __init__(self,shape,dtype):self.shape=shape;self.dtype=dtype;self.device=device
            def is_contiguous(self):return True
            def numel(self):
                import math
                return math.prod(self.shape)
        p=SimpleNamespace(q=Tensor((1,h,n,d),torch.int8),k=Tensor((1,hk,nk,d),torch.int8),
            v=Tensor((hk*d,(nk+ck-1)//ck*ck),torch.int8),
            q_scale=Tensor((1,h,(n+127)//128*32),torch.float32),
            k_scale=Tensor((1,hk,(nk+ck-1)//ck*4),torch.float32),
            v_scale=Tensor((hk*d,),torch.float32),attn_mask=None,cta_k=ck)
        with patch('experiments.kitchen_attention.attention._sm75'), \
                patch('torch.empty',side_effect=AssertionError('Overflow must fail before allocating GPU memory')):
            with self.assertRaisesRegex(ValueError,'32-bit addressing'):
                Backend.__new__(Backend).execute(p,variant='narrow')


if __name__=='__main__':unittest.main()
