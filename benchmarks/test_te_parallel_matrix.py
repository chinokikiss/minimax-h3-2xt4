"""CPU gates for the TE comparison's reconstruction and communication ablation."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from te_parallel_matrix import BACKENDS, LENGTHS, CollectiveReplay, completed_lengths, fp32_attention, merge_shards, padding_masks, single_gpu_turn


class TEParallelContracts(unittest.TestCase):
    def test_additive_padding_control_matches_query_dtype(self):
        allowed=torch.ones(4,4,dtype=torch.bool).tril();allowed[:,-1]=False
        for dtype in (torch.float16,torch.float32):
            masks=padding_masks(allowed,dtype)
            self.assertIs(masks['boolean'],allowed)
            self.assertEqual(masks['additive'].dtype,dtype)
            self.assertTrue(torch.isfinite(masks['additive']).all())
            self.assertTrue(torch.equal(masks['additive']==0,allowed))

    def test_resume_requires_complete_valid_paired_points(self):
        from copy import deepcopy
        cases=[dict(tokens=128,backend=backend,topology=topology,block_vs_native_fp16_flow={'finite':True},
            attention_vs_fp32={'finite':True},replay_byte_exact_to_nccl=True,output_sha256=backend)
            for backend in BACKENDS for topology in ('single_gpu','tp2_nccl')]
        ranks=[dict(uuid=str(rank),device='Tesla T4',capability=[7,5],cases=deepcopy(cases)) for rank in (0,1)]
        self.assertEqual(completed_lengths(ranks),[128])
        ranks[1]['cases'].pop()
        self.assertEqual(completed_lengths(ranks),[])

    def test_inventory_and_global_native_threshold(self):
        from patches.h3_operator_policy import H3OperatorPolicy
        self.assertEqual(set(BACKENDS), {'native', 'kitchen_causal'})
        for name in BACKENDS:self.assertEqual(H3OperatorPolicy(te_attention=name).te_attention, name)
        self.assertTrue(all(n*32*128 >= 1024*128 for n in LENGTHS))

    def test_reconstruction_uses_output_rows_and_input_columns(self):
        a = {'q_proj.weight':torch.arange(6).reshape(2, 3), 'q_proj.weight_scale':torch.ones(2, 1),
             'o_proj.weight':torch.arange(6).reshape(3, 2), 'o_proj.weight_scale':torch.ones(3, 1),
             'q_norm.weight':torch.ones(2)}
        b = {k:v.clone() for k,v in a.items()};b['q_proj.weight'].add_(10);b['o_proj.weight'].add_(20)
        full = merge_shards(a, b)
        self.assertTrue(torch.equal(full['q_proj.weight'], torch.cat([a['q_proj.weight'],b['q_proj.weight']], 0)))
        self.assertTrue(torch.equal(full['o_proj.weight'], torch.cat([a['o_proj.weight'],b['o_proj.weight']], 1)))
        self.assertEqual(full['o_proj.weight_scale'].shape, (3, 1))
        b['q_norm.weight'].add_(1)
        with self.assertRaises(ValueError):merge_shards(a,b)

    def test_replay_restores_exact_collective_results_without_transport(self):
        calls = []
        def reduce(tensor, **kwargs):calls.append('sum');tensor.mul_(2)
        def gather(output, tensor, **kwargs):calls.append('gather');output.copy_(torch.cat([tensor,tensor+1]))
        comm = CollectiveReplay(SimpleNamespace(all_reduce=reduce, all_gather_into_tensor=gather))
        comm.begin('capture');value=torch.ones(2);comm.all_reduce(value, op='SUM')
        output=torch.empty(4);comm.all_gather_into_tensor(output,value);comm.finish()
        expected=output.clone();value.zero_();output.zero_()
        comm.begin('replay');comm.all_reduce(value,op='SUM');comm.all_gather_into_tensor(output,value);comm.finish()
        self.assertEqual(calls,['sum','gather']);self.assertTrue(torch.equal(output,expected))
        self.assertTrue(torch.equal(value,torch.full((2,),2.)))
        comm.begin('replay')
        with self.assertRaises(RuntimeError):comm.all_reduce(torch.ones(3),op='SUM')

    def test_partial_replay_is_rejected(self):
        comm=CollectiveReplay(SimpleNamespace(all_reduce=lambda tensor,**kwargs:None))
        comm.begin('capture');comm.all_reduce(torch.ones(1),op='MAX');comm.finish();comm.begin('replay')
        with self.assertRaises(RuntimeError):comm.finish()

    def test_single_gpu_controls_do_not_run_concurrently(self):
        barrier=threading.Barrier(2);events=[]
        def run(rank):
            with single_gpu_turn(rank, barrier.wait):events.extend([(rank,'start'),(rank,'end')])
        threads=[threading.Thread(target=run,args=(rank,)) for rank in (0,1)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(timeout=2)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(events,[(0,'start'),(0,'end'),(1,'start'),(1,'end')])

    def test_chunked_fp32_oracle_matches_independent_sdpa_with_gqa(self):
        generator=torch.Generator().manual_seed(20261027)
        q=torch.randn(1,4,9,8,generator=generator);k=torch.randn(1,2,9,8,generator=generator);v=torch.randn(1,2,9,8,generator=generator)
        mask=torch.ones(1,1,9,9,dtype=torch.bool).tril()
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True).transpose(1,2).reshape(1,9,32)
        self.assertTrue(torch.allclose(fp32_attention(q,k,v,mask),expected,atol=1e-6,rtol=1e-6))
        native_mask=torch.empty(9,9,dtype=torch.float16).fill_(torch.finfo(torch.float16).min/4).triu_(1)
        self.assertTrue(torch.allclose(fp32_attention(q,k,v,native_mask),expected,atol=1e-6,rtol=1e-6))


if __name__=='__main__':unittest.main(verbosity=2)
