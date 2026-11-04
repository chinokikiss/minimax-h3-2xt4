"""CPU-only frozen-input admission for the full PTD v2 0.9MP 10s all-KC E2E."""
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.ptd_v2_long_kc_contract import from_previous, validate
from benchmarks.ptd_e2e_contract import RELEASE

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ComfyUI'))
PRIOR=ROOT/'docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102/request.json'

class LongPTDKCContract(unittest.TestCase):
    def setUp(self):
        self.previous=json.loads(PRIOR.read_text(encoding='utf-8'))
        self.plan=from_previous(self.previous)

    def test_exact_replay_and_geometry(self):
        p=self.plan
        self.assertEqual((p['width'],p['height'],p['frames'],p['seed']),(1280,704,243,20261020))
        self.assertEqual(p['prompt'],self.previous['prompt'])
        self.assertEqual(p['reference_images'],self.previous['reference_images'])
        self.assertNotIn('turbo_lora',p)
        self.assertEqual((p['video_shift'],p['audio_shift']),(12.,3.))
        self.assertEqual(p['operators']['dit_attention'],'kitchen')
        self.assertEqual(p['operators']['dit_attention_schedule'],'')
        self.assertEqual(validate(p)['generated_video_tokens'],63360)

    def test_fail_closed_on_changed_identity_and_sol(self):
        for changed in (dict(width=1344),dict(seed=1),dict(video_shift=6.),
                dict(sampler='euler'),dict(steps=2),
                dict(operators=self.plan['operators']|dict(dit_attention='sol')),
                dict(operators=self.plan['operators']|dict(dit_attention_schedule='1-8:sol@1.0')),
                dict(turbo_lora=self.previous['turbo_lora'])):
            with self.subTest(changed=list(changed)),self.assertRaises(ValueError):
                validate(self.plan|changed)

    def test_bank_v2_requires_all_fifty_layers_and_eight_hops(self):
        from benchmarks.run_ptd_v2_long_kc_e2e import validate_bank
        one=dict(shards={str(i):i for i in range(100)})
        bank=SimpleNamespace(layers=50,hops=[(None,one)]*8,
            recipe=dict(ptd_checkpoint_sha256=RELEASE['sha256'],
                video_shift=12.,audio_shift=3.,pruned_adaln_approximation=True),
            digest=('metadata','demo'),recipe_sha='recipe')
        self.assertEqual(validate_bank(bank)['shards_by_hop'],[100]*8)
        with self.assertRaises(ValueError):
            validate_bank(SimpleNamespace(**(vars(bank)|dict(layers=2))))

    def test_real_v2_manifest_recipe_and_checkpoint_guard(self):
        from benchmarks.run_ptd_v2_long_kc_e2e import validate_bank
        manifest=(ROOT/'docs/data/h3_ptd_v2_long_kc_e2e_20261104/gpu_final_v1'
                  /'h3_ptd_v2_09mp_10s_kc_e2e_20261104/trial/bank_v2.json')
        if not manifest.is_file():
            self.skipTest('Kaggle v1 Bank v2 manifest fixture not available')
        data=json.loads(manifest.read_text(encoding='utf-8'))
        recipe=data['recipe']
        self.assertEqual(recipe['ptd_checkpoint_sha256'],RELEASE['sha256'])
        bank=SimpleNamespace(layers=recipe['layers'],
            hops=[(None, {'shards':row['shards']}) for row in data['hops']],
            recipe=recipe,digest='bank-v2-fixture',recipe_sha=data['recipe_sha256'])
        self.assertEqual(validate_bank(bank)['shards_by_hop'],[100]*8)
        variants=(recipe|{'ptd_checkpoint_sha256':'0'*64},
            {k:v for k,v in recipe.items() if k!='ptd_checkpoint_sha256'})
        for bad_recipe in variants:
            with self.subTest(keys=list(bad_recipe)),self.assertRaises(ValueError):
                validate_bank(SimpleNamespace(**(vars(bank)|{'recipe':bad_recipe})))

    def test_parallel_video_decode_strategy_is_accepted(self):
        source=(ROOT/'benchmarks/run_ptd_v2_long_kc_e2e.py').read_text(encoding='utf-8')
        checks=[node for node in ast.walk(ast.parse(source))
                if isinstance(node,ast.If) and isinstance(node.test,ast.Compare)
                and "decoders['video'].last_strategy" in ast.unparse(node.test)]
        self.assertEqual(len(checks),1)
        self.assertIn('native_temporal_pairs',ast.literal_eval(checks[0].test.comparators[0]))
        self.assertIn('diagnostic_warnings',ast.unparse(checks[0]))

    def test_cuda13_is_experiment_only(self):
        from patches.h3_operator_dependencies import compiler_qualification
        keys=('H3_PTD_2NFE_ALLOW_UNVALIDATED_CUDA13',
              'H3_TURBO_15S_ALLOW_UNVALIDATED_CUDA13',
              'H3_PTD_E2E_ALLOW_UNVALIDATED_CUDA13')
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError):
                compiler_qualification('release 13.0',torch_cuda='13.0')
            with patch.dict(os.environ,{keys[2]:'1'}):
                self.assertEqual(compiler_qualification('release 13.0',torch_cuda='13.0'),
                                 'experimental_cuda_13_0_ptd_full_e2e_only')
                self.assertEqual(compiler_qualification('release 12.8',torch_cuda='13.0'),
                                 'validated_cuda_12_8')

if __name__=='__main__':unittest.main(verbosity=2)
