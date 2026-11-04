"""Regression gates for the pinned PTD v2 source/Baker preflight."""
import math
import unittest
from unittest.mock import patch

from benchmarks import preflight_ptd_v2_bake as preflight
from benchmarks.test_ptd_v2_bake import BoundedParts, EmptySyntheticPTD
from patches.tp2_bundle import sha


class V2BakerPreflight(unittest.TestCase):
    def setUp(self):
        holder = BoundedParts('runTest')
        holder.setUp()
        self.addCleanup(holder.doCleanups)
        self.holder = holder
        self.root = holder.root
        self.checkpoint = holder.heads
        self.base = holder.fixture.source.descriptor
        self.scratch = self.root
        self.fake_release = dict(preflight.RELEASE, bytes=self.checkpoint.stat().st_size,
                                 sha256=sha(self.checkpoint))

    def _inspect(self, **kwargs):
        with (patch.object(preflight, 'RELEASE', self.fake_release),
              patch.object(preflight, 'PTDCheckpoint', EmptySyntheticPTD)):
            return preflight.inspect_inputs(self.base, self.checkpoint, self.scratch,
                                            layers=1, allow_fixture=True, **kwargs)

    def test_metadata_requires_explicit_sha_and_never_qualifies_fixture(self):
        row = self._inspect()
        self.assertEqual(row['status'], 'blocked')
        self.assertFalse(row['source_ready'])
        self.assertIn('Pinned PTD SHA256 not checked', ' '.join(row['issues']))
        self.assertEqual(row['cuda_gate'], 'not_checked')
        self.assertIsNotNone(row['scratch'])
        self.assertTrue(row['scratch']['enough_for_one_hop'])

    def test_real_hash_validation_and_shared_scratch_formula(self):
        row = self._inspect(verify_ptd_sha256=True)
        self.assertTrue(row['release_sha256_checked'])
        self.assertEqual(row['checkpoint_sha256'], self.fake_release['sha256'])
        self.assertEqual(row['issues'], [])
        self.assertFalse(row['source_ready'], 'Synthetic fixtures cannot become a production release')
        budget = row['scratch']
        self.assertEqual(budget['minimum_free_bytes'],
                         math.ceil(1.15 * (budget['base_block_bytes']
                                           + budget['overlay_fp32_bytes'])) + 2**30)

    def test_source_only_and_physical_ready_are_distinct(self):
        # The metadata-admission function is mocked; this exercises gate
        # composition, not a synthetic claim of genuine production readiness.
        with (patch.object(preflight, 'RELEASE', self.fake_release),
              patch.object(preflight, 'PTDCheckpoint', EmptySyntheticPTD),
              patch.object(preflight, '_verify_inputs')):
            source = preflight.inspect_inputs(self.base, self.checkpoint, self.scratch,
                                              layers=50, verify_ptd_sha256=True)
            self.assertEqual(source['status'], 'source_qualified')
            self.assertTrue(source['source_ready'])
            self.assertFalse(source['physical_baker_ready'])
            self.assertTrue(source['pending'])
            with patch.object(preflight, '_devices_ok') as device_check:
                physical = preflight.inspect_inputs(
                    self.base, self.checkpoint, self.scratch, layers=50,
                    verify_ptd_sha256=True, check_t4=True)
                device_check.assert_called_once()
            self.assertEqual(physical['status'], 'passed')
            self.assertTrue(physical['physical_baker_ready'])

    def test_mismatched_release_sha_rejected(self):
        wrong = dict(self.fake_release, sha256='0' * 64)
        with (patch.object(preflight, 'RELEASE', wrong),
              patch.object(preflight, 'PTDCheckpoint', EmptySyntheticPTD)):
            row = preflight.inspect_inputs(self.base, self.checkpoint, self.scratch,
                                           layers=1, allow_fixture=False,
                                           verify_ptd_sha256=True)
        self.assertFalse(row['source_ready'])
        self.assertIn('SHA256 differs', ' '.join(row['issues']))

    def test_invalid_full_release_size_blocks_production(self):
        row = preflight.inspect_inputs(self.base, self.checkpoint, self.scratch,
                                       layers=50, verify_ptd_sha256=False)
        self.assertFalse(row['source_ready'])
        self.assertIn('size differs', ' '.join(row['issues']))

    def test_missing_source_has_no_implicit_download(self):
        row = preflight.inspect_inputs(self.base, self.root / 'not-downloaded.safetensors',
                                       self.scratch, layers=50, check_t4=False)
        self.assertFalse(row['physical_baker_ready'])
        self.assertIn('checkpoint is missing', ' '.join(row['issues']))
        self.assertFalse((self.root / 'not-downloaded.safetensors').exists())

    def test_insufficient_scratch_rejects_without_allocating(self):
        with patch.object(preflight.shutil, 'disk_usage', return_value=type('DU', (), {'free': 7})()):
            row = self._inspect(verify_ptd_sha256=True)
        self.assertFalse(row['scratch']['enough_for_one_hop'])
        self.assertIn('Scratch free space', ' '.join(row['issues']))

    def test_cuda_is_not_probed_without_explicit_flag(self):
        with patch.object(preflight, '_devices_ok', side_effect=AssertionError('probed')) as gate:
            self._inspect(verify_ptd_sha256=True)
            gate.assert_not_called()
        with patch.object(preflight, '_devices_ok', side_effect=RuntimeError('No T4')) as gate:
            row = self._inspect(verify_ptd_sha256=True, check_t4=True)
            gate.assert_called_once_with(('cuda:0', 'cuda:1'), False)
            self.assertEqual(row['cuda_gate'], 'blocked')
            self.assertIn('No T4', ' '.join(row['issues']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
