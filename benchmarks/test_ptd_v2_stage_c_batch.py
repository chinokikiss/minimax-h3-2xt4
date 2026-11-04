"""CPU-only regressions for bounded Stage C Hop lifecycle / receipt-only admission."""
import ast
import base64
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.test_ptd_v2_bake import BoundedParts, EmptySyntheticPTD
from patches.ptd.bake_v2 import bake_hop_part
from patches.ptd.publish_v2 import admit_baked_part, prepare_private_kaggle_part
from benchmarks.run_ptd_v2_stage_c_batch_t4 import retire_owned_hop
from benchmarks.run_ptd_v2_stage_c_publish_t4 import dataset_id


class StageCBoundedLifecycle(unittest.TestCase):
    def setUp(self):
        fixture = BoundedParts('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root

    def baked_staged_hop(self):
        scratch = self.root / 'scratch'
        parts = scratch / 'parts'
        staged = scratch / 'publish_staging'
        parts.mkdir(parents=True)
        staged.mkdir()
        with patch('patches.ptd.bake_v2.PTDCheckpoint', EmptySyntheticPTD):
            receipt = bake_hop_part(
                self.fixture.fixture.source.descriptor, self.fixture.heads,
                1, parts / 'hop_01', devices=('cpu',),
                layers=1, allow_fixture=True, prefetch_layers=True)
        admitted = admit_baked_part(parts / 'hop_01', 'hop_01')
        prepare_private_kaggle_part(
            parts / 'hop_01', 'hop_01', staged / 'hop_01',
            dataset_id=dataset_id('hop_01'), license_name='other',
            verified_proof=admitted)
        return scratch, admitted

    def test_retire_only_exact_owned_hardlinked_hop_after_matching_ack(self):
        scratch, proof = self.baked_staged_hop()
        unrelated = scratch / 'base'
        unrelated.mkdir()
        (unrelated / 'KEEP').write_text('base', encoding='utf-8')
        ack = dict(dataset_id=dataset_id('hop_01'),
                   receipt_sha256=proof['receipt_sha256'],
                   status='create_request_accepted_processing_unverified')
        removed = retire_owned_hop(scratch, 1, proof['receipt_sha256'], ack)
        self.assertTrue(removed['retired'])
        self.assertEqual(removed['remote_sha256'], 'pending')
        self.assertFalse((scratch / 'parts' / 'hop_01').exists())
        self.assertFalse((scratch / 'publish_staging' / 'hop_01').exists())
        self.assertEqual((unrelated / 'KEEP').read_text(), 'base')

    def test_wrong_identity_or_missing_ack_preserves_every_file(self):
        scratch, proof = self.baked_staged_hop()
        fake = dict(dataset_id=dataset_id('hop_02'),
                    receipt_sha256=proof['receipt_sha256'],
                    status='create_request_accepted_processing_unverified')
        with self.assertRaisesRegex(ValueError, 'matching'):
            retire_owned_hop(scratch, 1, proof['receipt_sha256'], fake)
        fake['dataset_id'] = dataset_id('hop_01')
        fake['status'] = 'upload_in_progress'
        with self.assertRaisesRegex(ValueError, 'matching'):
            retire_owned_hop(scratch, 1, proof['receipt_sha256'], fake)
        self.assertTrue((scratch / 'parts' / 'hop_01').is_dir())
        self.assertTrue((scratch / 'publish_staging' / 'hop_01').is_dir())

    def test_batch_builder_is_private_bounded_and_no_implicit_overwrite(self):
        from benchmarks.build_ptd_v2_stage_c_batch_job import build, DEFAULT_BASE
        output = self.root / 'batch_notebook'
        with self.assertRaisesRegex(ValueError, 'reclamation'):
            build(DEFAULT_BASE, output, hops=(3, 4, 5))
        meta = build(DEFAULT_BASE, output, hops=(3, 4, 5, 6, 7),
                     retire_after_upload_ack=True)
        self.assertEqual(meta['hops'], [3, 4, 5, 6, 7])
        self.assertEqual(meta['replace_hops'], [])
        self.assertEqual(len(meta['dataset_ids']), 5)
        self.assertTrue(meta['retire_after_upload_ack'])
        nb = json.loads((output / 'experiment.ipynb').read_text(encoding='utf-8'))
        code = ''.join(nb['cells'][0]['source'])
        ast.parse(code)
        match = re.findall(r"worker.write_bytes\(base64.b64decode\('([^']+)'\)\)", code)
        self.assertTrue(match)
        worker = base64.b64decode(match[-1]).decode('utf-8')
        self.assertIn('--retire-after-upload-ack', worker)
        self.assertIn('--hops', worker)
        self.assertTrue(json.loads((output/'kernel-metadata.json').read_text())['is_private'])

    def test_builder_replacement_subset_is_explicit(self):
        from benchmarks.build_ptd_v2_stage_c_batch_job import build, DEFAULT_BASE
        meta = build(DEFAULT_BASE, self.root/'replace_notebook',
                     hops=(1, 2), replace_hops=(1,), retire_after_upload_ack=False)
        self.assertEqual(meta['replace_hops'], [1])
        self.assertFalse(meta['retire_after_upload_ack'])


if __name__ == '__main__':
    unittest.main()
