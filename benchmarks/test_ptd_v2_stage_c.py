"""CPU-only Stage C plan admission and fail-closed regression tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.plan_ptd_v2_stage_c import build_plan, EXPECTED_PTD_SHA256


class StageCPlan(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipe = dict(layers=50, hops=8, steps=32,
                           ptd_checkpoint_sha256=EXPECTED_PTD_SHA256)
        self.recipe_sha = hashlib.sha256(json.dumps(
            self.recipe, sort_keys=True).encode()).hexdigest()
        self.result = dict(status='complete', layers=50, hop=0, shards=100,
                           verified_merge_targets=208, ptd_sha256=EXPECTED_PTD_SHA256,
                           recipe_sha256=self.recipe_sha,
                           stages={f'stage_{i}':dict(status='complete') for i in range(6)},
                           local_part_validation=dict(
                               status='verified', part='hop_00', payload_files=202,
                               layers=50, hop=0, recipe_sha256=self.recipe_sha,
                               payload_bytes=21179555374))
        self.preflight = dict(status='passed', release_sha256_checked=True,
                              physical_baker_ready=True, issues=[],
                              checkpoint_sha256=EXPECTED_PTD_SHA256)
        self.shared = dict(recipe=self.recipe, recipe_sha256=self.recipe_sha,
                           aux_bytes=58150408, native_header_bytes=150860,
                           block_plan_bytes=391820)
        self.hop0 = dict(recipe_sha256=self.recipe_sha,
                         shards={f'dit.block{i:02d}.rank{r}':{}
                                 for i in range(50) for r in (0, 1)},
                         verified_targets=[f'target_{i}' for i in range(208)])
        self._save()

    def _save(self):
        for name, value in [('result', self.result), ('source_preflight',self.preflight),
                            ('shared_receipt',self.shared), ('hop_00_receipt',self.hop0)]:
            (self.root / (name+'.json')).write_text(json.dumps(value), encoding='utf-8')

    def test_stage_b_sources_are_qualified_without_fake_remote_pass(self):
        result = build_plan(self.root)
        self.assertEqual(result['status'], 'source_qualified_publication_not_started')
        self.assertFalse(result['safe_to_upload_now'])
        self.assertEqual(result['estimated_eight_hop_plus_shared_bytes'],
                         8*21179555374+58150408+150860+391820)
        self.assertEqual(len(result['part_order']), 9)
        self.assertEqual(result['release_requirements']['remote_readback_proofs'], 0)

    def test_rejects_incomplete_and_tampered_evidence(self):
        scenarios = [
            ('stage',lambda:self.result['stages']['stage_0'].update(status='failed')),
            ('hash',lambda:self.preflight.update(checkpoint_sha256='0'*64)),
            ('recipe',lambda:self.shared['recipe'].update(steps=31)),
            ('shards',lambda:self.hop0['shards'].pop('dit.block49.rank1')),
            ('proof',lambda:self.result['local_part_validation'].update(payload_files=1)),
        ]
        for name, mutate in scenarios:
            with self.subTest(name=name):
                original = [copy.deepcopy(x) for x in
                            (self.result,self.preflight,self.shared,self.hop0)]
                mutate()
                self._save()
                with self.assertRaises(ValueError):
                    build_plan(self.root)
                self.result,self.preflight,self.shared,self.hop0 = original
                self._save()


if __name__ == '__main__':
    unittest.main()
