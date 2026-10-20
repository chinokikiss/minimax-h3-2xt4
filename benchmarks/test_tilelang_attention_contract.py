"""CPU checks of preparation and independent numerical-reference contracts."""
import math
import unittest
import torch
from patches.tilelang_sparse_attention import prepare,make_routes,reference


class AttentionContract(unittest.TestCase):
    def setUp(self):torch.manual_seed(71)

    def tensors(self,n=129,d=64):
        return [torch.randn(1,2,n,d,dtype=torch.float16) for _ in range(3)]

    def test_quantized_dense_matches_direct_reconstructed_operands(self):
        q,k,v=self.tensors()
        p=prepare(q,k,v);routes=make_routes(p,'dense')
        actual,lse=reference(p,routes,False,round_p=False)
        qhat=p.qi[:,:,:129].float()*p.qs[:,:,:129,None]
        khat=p.ki[:,:,:129].float()*p.ks[:,:,:129,None]
        scores=qhat@khat.transpose(-1,-2)/math.sqrt(64)
        expected=(scores.softmax(-1)@v.float()).half()
        torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-4)
        torch.testing.assert_close(lse,scores.logsumexp(-1)+p.shift,rtol=1e-5,atol=2e-6)

    def test_ragged_uniform_pooling_preserves_token_counts(self):
        q,k,v=self.tensors(65,128);q.zero_();k.zero_()
        p=prepare(q,k,v)
        self.assertEqual(p.counts.tolist(),[64,1])
        routes=torch.zeros(1,2,2,2,dtype=torch.int32)
        actual,_=reference(p,routes,True,round_p=False)
        expected=v.float().mean(-2,keepdim=True).expand_as(v).half()
        torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-4)

    def test_all_selected_pool_mode_reduces_to_dense(self):
        q,k,v=self.tensors(65);p=prepare(q,k,v);routes=make_routes(p,'dense')
        for pv8 in [False,True]:
            a,la=reference(p,routes,False,pv8)
            b,lb=reference(p,routes,True,pv8)
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(la,lb))

    def test_single_block_int8_pv_scale_factorization(self):
        q,k,v=self.tensors(31);p=prepare(q,k,v)
        actual,_=reference(p,make_routes(p,'dense'),False,True)
        score=(p.qi[:,:,:31].float()@p.ki[:,:,:31].float().transpose(-1,-2))
        score*=p.qs[:,:,:31,None]*p.ks[:,:,None,:31]/8
        pe=(score-score.amax(-1,keepdim=True)).exp()
        pi=(pe*127+0.5).floor().clamp(0,127)
        expected=((pi@p.vi[:,:,:31].float())*p.vs[:,:,0,None,:]/127/pe.sum(-1,keepdim=True)).half()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_padding_and_mandatory_routes(self):
        q,k,v=self.tensors(513);p=prepare(q,k,v)
        self.assertTrue((p.qi[:,:,513:]==0).all())
        self.assertTrue((p.ki[:,:,513:]==0).all())
        self.assertTrue((p.vi[:,:,513:]==0).all())
        for kind in ['sol','topk','local_global']:
            r=make_routes(p,kind,density=0.01)
            self.assertTrue(r[:,:,:,0].bool().all())
            self.assertTrue(r.diagonal(dim1=-2,dim2=-1).bool().all())

    def test_reject_unsupported_inputs(self):
        q,k,v=self.tensors()
        with self.assertRaises(ValueError):prepare(q.float(),k.float(),v.float())
        with self.assertRaises(ValueError):prepare(q,k[:,:,:64],v)

    def test_causal_gqa_matches_explicit_dense_reconstruction(self):
        q=torch.randn(1,8,65,128,dtype=torch.float16)
        k=torch.randn(1,2,65,128,dtype=torch.float16);v=torch.randn_like(k)
        p=prepare(q,k,v,enable_gqa=True)
        actual,lse=reference(p,make_routes(p,'dense'),False,causal=True,round_p=False)
        z=(p.qi[:,:,:65].float()@p.ki[:,:,:65].float().transpose(-1,-2))
        z*=p.qs[:,:,:65,None]*p.ks[:,:,None,:65]/math.sqrt(128)
        z=z.masked_fill(torch.ones(65,65,dtype=torch.bool).triu(1),float('-inf'))
        expected=(z.softmax(-1)@v.repeat_interleave(4,1).float()).half()
        torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-4)
        torch.testing.assert_close(lse,z.logsumexp(-1)+p.shift,rtol=1e-5,atol=2e-6)
        with self.assertRaises(ValueError):prepare(q,k,v)


if __name__=='__main__':unittest.main()
