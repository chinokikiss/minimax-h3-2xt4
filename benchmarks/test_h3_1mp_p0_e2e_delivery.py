"""CPU source, payload, and workload preflight for the one-shot 1MP H3 T4 E2E.

This does not execute CUDA, contact Kaggle, or claim a completed GPU test.
"""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102'
JOB=EVIDENCE/'gpu_job'
OLD=ROOT/'docs/data/h3_t4_fp16_512_sol_all_8nfe_10s_20261101'


class OneMegapixelE2EPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding=json.loads((JOB/'source_binding.json').read_text(encoding='utf-8'))
        cls.meta=json.loads((JOB/'kernel-metadata.json').read_text(encoding='utf-8'))
        cls.notebook=json.loads((JOB/'experiment.ipynb').read_text(encoding='utf-8'))
        cls.plan=cls.binding['plan']
        cls.code=''.join(cls.notebook['cells'][0]['source'])

    def test_previous_completed_baseline_and_canvas(self):
        old=json.loads((OLD/'request.json').read_text(encoding='utf-8'))
        p=self.plan
        from benchmarks.turbo_challenge_contract import validate
        geometry=validate(p)
        self.assertEqual((p['width'],p['height']),(1344,768))
        self.assertEqual(geometry['video_latent_shape'],[1,24,72,48,84])
        self.assertEqual(geometry['generated_video_tokens'],72576)
        self.assertEqual(p['workload_contract'],'turbo_1mp_10s')
        for key in ('frames','fps','steps','sampler','scheduler','cfg',
                    'video_shift','audio_shift','seed','prompt',
                    'prompt_sha256','reference_images','turbo_lora',
                    'offline_lora','disk_baked_lora'):
            with self.subTest(field=key):self.assertEqual(p[key],old[key])

    def test_fp16_512_and_all_nfe_sol(self):
        p=self.plan
        self.assertEqual(p['residual_exponent'],9)
        self.assertIs(p['debug'],False)
        self.assertEqual(p['steps'],8)
        self.assertEqual(p['operators']['te_attention'],'kitchen_causal')
        self.assertEqual(p['operators']['dit_attention_schedule'],'1-8:sol@1.0')
        self.assertEqual(p['operators']['video_attention'],'kitchen')
        self.assertTrue(p['operators']['sol_topology_v2'])
        self.assertFalse(p['operators']['sol_collect_route_stats'])
        self.assertEqual(self.binding['attention_selection'],[['sol',1.0]]*8)
        self.assertEqual(self.binding['expected_sol_per_rank'],400)
        self.assertEqual(self.binding['expected_kitchen_dit_per_rank'],0)
        self.assertTrue(p['offline_lora'] and p['disk_baked_lora'])
        from patches.h3_operator_policy import H3OperatorPolicy
        policy=H3OperatorPolicy()
        for name in ('dit_p0_2_lifetime','dit_p0_3_lifetime',
                     'dit_p0_a1_turing','dit_p0_b_inplace','dit_p0_c_local_rotation'):
            with self.subTest(default=name):self.assertIs(getattr(policy,name),True)

    def test_overlay_is_closed_and_bytes_match(self):
        from benchmarks.source_overlay_closure import assert_closed
        binding=self.binding
        self.assertEqual(binding['baseline_commit'],
            'd7c7528515d2a7961de5e6f961232d81d21cdc8b')
        names=set(binding['overlay_hashes'])
        self.assertEqual(set(binding['closure'])-names,set())
        for name in ('patches/tilelang_swiglu_inplace.py',
                     'patches/tilelang_swiglu_quant.py',
                     'patches/tilelang_row_quant.py',
                     'patches/tilelang_dit_fusion.py',
                     'patches/h3_operator_policy.py',
                     'patches/sol_sm75_backend.py',
                     'patches/dit_kitchen_turing_a1.py'):
            self.assertIn(name,names)
        for name,expected in binding['overlay_hashes'].items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected)
        inplace=(ROOT/'patches/tilelang_swiglu_inplace.py').read_text(encoding='utf-8')
        p0c=(ROOT/'patches/tilelang_row_quant.py').read_text(encoding='utf-8')
        self.assertNotIn('from benchmarks',inplace)
        self.assertIn('or x.shape[0]<=0',p0c)
        self.assertNotIn('x.shape[0] not in (128,6814,59134,65166)',p0c)

    def test_notebook_and_secure_reference_staging(self):
        meta=self.meta
        self.assertEqual(meta['id'],'pamcks/h3-1mp-fp16-512-sol-v2-8nfe-10s-p0-20261102')
        self.assertEqual(re.sub(r'[^a-z0-9]+','-',meta['title'].lower()).strip('-'),
                         meta['id'].split('/')[-1])
        self.assertTrue(meta['is_private'] and meta['enable_gpu'])
        self.assertEqual(meta['machine_shape'],'NvidiaTeslaT4')
        self.assertEqual(meta['dataset_sources'],[])
        self.assertEqual(meta['kernel_sources'],[])
        self.assertEqual(len(self.notebook['cells']),1)
        self.assertLess((JOB/'experiment.ipynb').stat().st_size,900000)
        ast.parse(self.code)
        anchor='worker.write_bytes(base64.b64decode('
        self.assertEqual(self.code.count(anchor),2)
        begin=self.code.rfind(anchor)+len(anchor)
        end=self.code.index('))',begin)
        worker=base64.b64decode(ast.literal_eval(self.code[begin:end]),validate=True).decode('utf-8')
        self.assertIn('H3_REFERENCE_STAGED',worker)
        self.assertIn('H3_FP16_512_SOL_V2_ALL_8_NFE_1MP_10S_P0_DEFAULTS',worker)
        self.assertIn('h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/batch_results.json',self.code)
        self.assertIn("['status']=='complete'",self.code)
        self.assertEqual(worker.count('H3_REFERENCE_STAGED'),1)
        self.assertFalse(re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_-]{20,})',
            self.code))
        self.assertEqual(self.binding['gpu_count'],2)
        self.assertIs(self.binding['status_polling'],False)
        self.assertIs(self.binding['live_log_polling'],False)
        self.assertEqual(self.binding['final_collection_count'],0)

    def test_old_gpu_e2e_is_a_complete_distinct_job(self):
        receipt=json.loads((OLD/'gpu_final_v2/h3_t4_fp16_512_sol_v2_all_8nfe_10s/batch_results.json')
            .read_text(encoding='utf-8'))
        self.assertEqual(receipt['status'],'complete')
        self.assertEqual(receipt['completed_steps'],8)
        self.assertEqual((receipt['request']['width'],receipt['request']['height']),(1216,704))


if __name__=='__main__':
    unittest.main(verbosity=2)
