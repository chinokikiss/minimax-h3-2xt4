"""CPU qualification for strict ablation dispatch and immutable platform lock."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from benchmarks.prepare_vae_cuda13 import requirements
from benchmarks.run_vae_dispatch_ablation import dispatch,CASES,real_tiles,summarize


class Cuda13Contract(unittest.TestCase):
    def test_factorial_attribution_reports_interaction(self):
        rows=[dict(batch=4,layer=0,linear=linear,attention=attention,gpu_median_ms=ms)
              for linear,attention,ms in [('eager','sdpa',10),('cuda','sdpa',6),
                  ('eager','kitchen',9),('cuda','kitchen',4),('fp16_native','sdpa',5)]]
        result=summarize(rows)[0]
        self.assertEqual(result['linear_cuda_saving_with_sdpa_ms'],4)
        self.assertEqual(result['attention_kitchen_saving_with_cuda_ms'],2)
        self.assertEqual(result['interaction_ms'],-1)
        self.assertEqual(result['cuda_int8_over_fp16_ratio'],.8)

    def test_lock_only_replaces_explicit_platform_with_hashes(self):
        lock=json.loads((Path(__file__).resolve().parents[1]/'vae_cuda13_lock.json').read_text())
        text=requirements(lock)
        self.assertIn('torch @ https://download-r2.pytorch.org/whl/cu130/',text)
        self.assertEqual(text.count('--hash=sha256:'),len(lock['packages']))
        lock['packages']['comfy-kitchen']=dict(url='https://files.pythonhosted.org/packages/anything',sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'Unexpected platform replacement'):
            requirements(lock)

    def test_attention_and_linear_are_independent_and_restore_on_failure(self):
        import comfy.quant_ops as quant
        import comfy_kitchen as ck
        before=quant.ck
        tensor=torch.ones((1,2,4,8),dtype=torch.float16)
        expected=torch.nn.functional.scaled_dot_product_attention(tensor,tensor,tensor)
        for linear,attention in CASES:
            row={}
            def kernel(**kwargs):return kwargs['x']+1
            with patch.object(ck.registry,'get_implementation',return_value=kernel) as lookup, \
                    patch.object(ck,'int8_attention',return_value=tensor+2) as kitchen:
                with dispatch(linear,attention,row):
                    value=quant.ck.int8_linear(tensor,torch.ones(8,8,dtype=torch.int8),torch.tensor(1.0),out_dtype=torch.float16)
                    self.assertTrue(torch.equal(value,tensor+1))
                    result=quant.ck.int8_attention(tensor,tensor,tensor)
                    self.assertTrue(torch.equal(result,expected if attention=='sdpa' else tensor+2))
                    self.assertEqual(lookup.call_args.kwargs['backend'],linear)
                    self.assertEqual(kitchen.call_count,int(attention=='kitchen'))
            self.assertIs(quant.ck,before)
        with self.assertRaisesRegex(RuntimeError,'strict backend failed'):
            with patch.object(ck.registry,'get_implementation',side_effect=RuntimeError('strict backend failed')):
                with dispatch('cuda','sdpa',{}):
                    quant.ck.int8_linear(tensor,torch.ones(8,8,dtype=torch.int8),torch.tensor(1.0))
        self.assertIs(quant.ck,before)

    def test_real_tile_plan_matches_prior_native_geometry(self):
        # The shell uses the frozen Native architecture and normalization;
        # no trained tensor or GPU operation is needed for its tile plan.
        from types import SimpleNamespace
        reader=SimpleNamespace(header={})
        tiles=real_tiles(reader,torch.zeros(1,24,107,34,60))
        self.assertEqual(list(tiles[4].shape),[4,24,7,16,16])
        self.assertEqual(list(tiles[1].shape),[1,24,7,16,16])
        from comfy.ldm.minimax.vae import LATENTS_MEAN
        self.assertTrue(torch.equal(tiles[1][0,:,0,0,0],torch.tensor(LATENTS_MEAN).half()))


if __name__=='__main__':unittest.main()
