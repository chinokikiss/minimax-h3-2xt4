"""CPU numerical/ownership/source gates; no GPU, build, network or installation."""
import ctypes
from dataclasses import replace
import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from . import attention as a
from .derive import ROOT, derive_header, verify_vendor


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(517)

    def test_routes_match_existing_policy_including_ragged_prefix(self):
        from patches.tilelang_grouped_tp_optimized import prepare,routes
        for n in (65,129,257):
            q,k,v=[torch.randn(1,2,n,128,dtype=torch.float16) for _ in range(3)]
            old=prepare(q,k,v,fused=False,mode='scalar',query_tile=64)
            for prefix in (0,1,64,n):
                self.assertTrue(torch.equal(a.routing(q,k,prefix),routes(old,prefix,fast_tau0=True)))

    def test_fp32_routing_retains_old_half_boundary(self):
        q,k=[torch.randn(1,2,129,64) for _ in range(2)]
        self.assertTrue(torch.equal(a.routing(q,k,64),a.routing(q.half(),k.half(),64)))

    def test_compaction_preserves_every_selected_block_in_order(self):
        routes=torch.rand(2,3,3,5)>.5
        plan=a.compact_reference(routes,129,257)
        for b in range(2):
            for h in range(3):
                for row in range(3):
                    expected=routes[b,h,row].nonzero().flatten().tolist()
                    count=int(plan.counts[b,h,row])
                    self.assertEqual(plan.ids[b,h,row,:count].tolist(),expected)
                    self.assertTrue((plan.ids[b,h,row,count:]==5).all())

    def test_empty_and_full_compaction(self):
        for full in (False,True):
            plan=a.compact_reference(torch.full((1,1,2,2),full),65,65)
            self.assertTrue((plan.counts==(2 if full else 0)).all())

    def test_causal_offset_and_ragged_token_mask(self):
        plan=a.compact_reference(torch.ones(1,4,2,3,dtype=torch.bool),65,131)
        for offset in (0,1,63,64,66,130):
            got=a.token_mask(plan,is_causal=True,query_offset=offset)
            expected=torch.arange(131)[None,:]<=torch.arange(65)[:,None]+offset
            self.assertTrue(torch.equal(got,expected.expand(1,4,65,131)))

    def test_future_tokens_have_no_probability_in_masked_reference(self):
        q,k,v=[torch.randn(1,2,65,64) for _ in range(3)]
        plan=a.compact_reference(torch.ones(1,2,2,2,dtype=torch.bool),65,65)
        mask=a.token_mask(plan,is_causal=True)
        expected=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        changed=v.clone(); changed[:,:,31:]+=1000
        result=F.scaled_dot_product_attention(q,k,changed,attn_mask=mask)
        self.assertTrue(torch.equal(expected[:,:,:31],result[:,:,:31]))

    def test_empty_rows_sdpa_reference_is_zero(self):
        plan=a.compact_reference(torch.zeros(1,1,2,2,dtype=torch.bool),65,65)
        q,k,v=[torch.randn(1,1,65,64) for _ in range(3)]
        out=F.scaled_dot_product_attention(q,k,v,attn_mask=a.token_mask(plan))
        self.assertTrue((out==0).all())

    def test_pooled_ragged_uniform_scores_preserve_counts(self):
        q=torch.zeros(1,2,65,64); k=torch.zeros_like(q); v=torch.randn_like(q)
        plan=a.compact_reference(torch.zeros(1,2,2,2,dtype=torch.bool),65,65)
        result,lse=a.pooled_reference(q,k,v,plan)
        torch.testing.assert_close(result,v.mean(-2,keepdim=True).expand_as(v))
        torch.testing.assert_close(lse,torch.full_like(lse,torch.tensor(65.).log()))

    def test_pooled_all_selected_reduces_to_dense_with_gqa(self):
        q=torch.randn(1,4,65,64); k=torch.randn(1,2,65,64); v=torch.randn_like(k)
        plan=a.compact_reference(torch.ones(1,4,2,2,dtype=torch.bool),65,65)
        got,_=a.pooled_reference(q,k,v,plan)
        expected=F.scaled_dot_product_attention(q,k,v,enable_gqa=True)
        torch.testing.assert_close(got,expected,rtol=2e-5,atol=2e-6)

    def test_pooled_mixed_routes_are_invariant_to_common_key_shift(self):
        q,k,v=[torch.randn(1,2,65,64) for _ in range(3)]
        routes=torch.tensor([[[[1,0],[0,1]],[[0,1],[1,0]]]],dtype=torch.int32)
        plan=a.compact_reference(routes,65,65)
        got,lse=a.pooled_reference(q,k,v,plan)
        shift=torch.randn(1,2,1,64)
        expected,new_lse=a.pooled_reference(q,k+shift,v,plan)
        torch.testing.assert_close(got,expected,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(new_lse-lse,(q*shift).sum(-1)/8,rtol=2e-5,atol=2e-6)

    def test_pinned_scale_layout_with_gqa_and_ragged_keys(self):
        for ck in (64,128):
            b,h,hk,n,nk,d=1,4,2,65,131,128
            p=SimpleNamespace(q=torch.ones(b,h,n,d,dtype=torch.int8),
                k=torch.ones(b,hk,nk,d,dtype=torch.int8),
                v=torch.ones(b*hk*d,((nk+ck-1)//ck)*ck,dtype=torch.int8),
                q_scale=torch.arange(b*h*((n+127)//128)*32).reshape(b,h,-1).float()+1,
                k_scale=torch.arange(b*hk*((nk+ck-1)//ck)*4).reshape(b,hk,-1).float()+1,
                v_scale=torch.arange(b*hk*d).float()+1,cta_k=ck)
            q,k,v=a.dequantized_operands(p)
            for token in (0,1,7,8,31,32,64):
                self.assertEqual(q[0,3,token,12],p.q_scale[0,3,token//32*8+token%8])
            for token in (0,3,4,63,64,127,130):
                # Independently enumerate the pinned producer: its four warps
                # write n=block*WARPK+j*8+warp*2+pair, sharing a scale.
                producer={}
                for block in range((nk+ck-1)//ck):
                    for warp in range(4):
                        for group in range(ck//8):
                            for pair in (0,1):
                                producer[block*ck+group*8+warp*2+pair]=block*4+warp
                self.assertEqual(k[0,1,token,12],p.k_scale[0,1,producer[token]])
            self.assertEqual(v[0,1,130,12],p.v_scale[d+12])

    def test_split_state_merge_handles_empty_fine_or_coarse(self):
        q,k,v=[torch.randn(1,1,65,64) for _ in range(3)]
        dense=q@k.transpose(-1,-2)/8
        for cut in (0,31,65):
            za,zb=dense[...,:cut],dense[...,cut:]
            la=za.logsumexp(-1); lb=zb.logsumexp(-1)
            oa=za.softmax(-1)@v[:,:,:cut]; ob=zb.softmax(-1)@v[:,:,cut:]
            top=torch.maximum(la,lb); wa=(la-top).exp(); wb=(lb-top).exp()
            got=(oa*wa[...,None]+ob*wb[...,None])/(wa+wb)[...,None]
            torch.testing.assert_close(got,dense.softmax(-1)@v,rtol=2e-5,atol=2e-6)

    def test_value_permutation_matches_independent_producer_table_at_ragged_lengths(self):
        # Direct table of the retained producer's inverse 16-lane permutation.
        # Distinct token values catch mistakes concealed by all-one V fixtures.
        producer=[0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]
        for nk in (65,67,79,84,131):
            kp=(nk+63)//64*64; d=64
            values=(torch.arange(kp)%127-63).to(torch.int8)
            storage=torch.empty(d,kp,dtype=torch.int8)
            for token in range(kp):
                storage[:,token//16*16+producer[token%16]]=values[token]
            p=SimpleNamespace(q=torch.zeros(1,1,1,d,dtype=torch.int8),
                k=torch.zeros(1,1,nk,d,dtype=torch.int8),v=storage,
                q_scale=torch.ones(1,1,32),k_scale=torch.ones(1,1,(nk+63)//64*4),
                v_scale=torch.ones(d),cta_k=64)
            _,_,actual=a.dequantized_operands(p)
            torch.testing.assert_close(actual[0,0],values[:nk,None].float().expand(nk,d),rtol=0,atol=0)

    def test_native_abi_sizes(self):
        self.assertEqual(ctypes.sizeof(a.AttentionArgs),144)
        self.assertEqual(ctypes.sizeof(a.PoolArgs),136)

    def test_frozen_vendor_and_deterministic_derivative(self):
        verify_vendor()
        with tempfile.TemporaryDirectory() as directory:
            first=Path(directory)/'a.cuh'; second=Path(directory)/'b.cuh'
            self.assertEqual(derive_header(first),derive_header(second))
            self.assertIn('H3_ROUTE',first.read_text())
        verify_vendor()

    def test_no_tilelang_import_in_new_backend(self):
        import ast
        for path in ROOT.glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    self.assertFalse(any(x.name.startswith('tilelang') for x in node.names))
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('tilelang'))

    def test_reject_bad_geometry_and_offset(self):
        with self.assertRaises(ValueError):a.compact_reference(torch.zeros(1,1,2,3),65,65)
        with self.assertRaises(ValueError):a.routing(torch.zeros(1,2,65,64),torch.zeros(1,1,65,64))
        with self.assertRaises(ValueError):a.routing(torch.zeros(1,1,65,64),torch.zeros(1,1,65,64),66)
        plan=a.compact_reference(torch.ones(1,1,2,2,dtype=torch.bool),65,65)
        with self.assertRaises(ValueError):a.token_mask(plan,is_causal=True,query_offset=-1)

    def test_runtime_imports_and_exact_pins_on_cpu(self):
        self.assertIsNotNone(a.runtime_contract(require_extension=False))


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
