"""Independent mounted transport verifier parity and tamper tests."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import benchmarks.test_ptd_v2_bake as bake_fixture
from benchmarks.verify_ptd_v2_stage_c_remote import verify
from benchmarks.build_ptd_v2_stage_c_readback_job import build
from patches.ptd.publish_v2 import verify_part, check_readback_proof
from patches.tp2_bundle import sha


class StageCRemoteReadback(unittest.TestCase):
    def setUp(self):
        fixture = bake_fixture.BoundedParts('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture._bake()

    def test_readback_delivery_private_cpu_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            left=Path(temp)/'left'
            right=Path(temp)/'right'
            a=build(left,part='hop_00',version=1,receipt_sha256='a'*64,
                    recipe_sha256='b'*64)
            b=build(right,part='hop_00',version=1,receipt_sha256='a'*64,
                    recipe_sha256='b'*64)
            self.assertEqual(a['notebook_sha256'],b['notebook_sha256'])
            self.assertEqual(a['provider_version_authentication'], 'required_before_submission')
            meta=json.loads((left/'kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private'])
            self.assertFalse(meta['enable_gpu'] or meta['enable_internet'])
            self.assertEqual(meta['dataset_sources'],['pamcks/h3-ptd-v2-hop-00-20261103'])
            script=''.join(json.loads((left/'experiment.ipynb').read_text())['cells'][0]['source'])
            ast.parse(script)
            self.assertIn('/kaggle/input/datasets/pamcks/h3-ptd-v2-hop-00-20261103',script)

    def test_ioab_readback_uses_isolated_private_experimental_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            for part in ('hop_01', 'hop_02'):
                output = Path(temp) / part
                receipt = 'a' * 64 if part == 'hop_01' else 'b' * 64
                binding = build(output, part=part, version=1,
                                receipt_sha256=receipt, recipe_sha256='c' * 64,
                                ioab=True)
                handle = f'pamcks/h3-ptd-v2-{part.replace("_", "-")}-ioab-20261103'
                metadata = json.loads((output / 'kernel-metadata.json').read_text())
                self.assertEqual(metadata['dataset_sources'], [handle])
                self.assertEqual(metadata['id'],
                                 f'pamcks/h3-ptd-v2-readback-{part.replace("_", "-")}-ioab-20261103')
                self.assertTrue(metadata['is_private'])
                self.assertFalse(metadata['enable_gpu'] or metadata['enable_internet'])
                self.assertEqual(binding['attached_dataset'], handle)
                self.assertEqual(binding['operator_claimed_version_pin'], f'{handle}/versions/1')
                self.assertEqual(binding['provider_version_authentication'], 'required_before_submission')
                self.assertEqual(binding['status'], 'prepared_not_submitted')
                script = ''.join(json.loads((output / 'experiment.ipynb').read_text())['cells'][0]['source'])
                ast.parse(script)
                self.assertIn('/kaggle/input/datasets/' + handle, script)
                self.assertIn("state['status']", script)
            with self.assertRaisesRegex(ValueError, 'IOAB Dataset exists only'):
                build(Path(temp) / 'shared_ioab', part='shared', version=1,
                      receipt_sha256='a' * 64, recipe_sha256='c' * 64, ioab=True)

    def test_shared_and_hop_byte_proofs_match_authoritative_verifier(self):
        for part in ('shared', 'hop_00'):
            root = self.root / part
            source_sha = sha(root / 'receipt.json')
            recipe = json.loads((root / 'receipt.json').read_text())
            actual = verify(root, part, source_sha, recipe['recipe_sha256'],
                            f'pamcks/h3-ptd-v2-{part}/versions/1', expected_layers=1)
            reference = verify_part(root, part,
                                    expected_receipt_sha256=source_sha,
                                    published_identity=f'pamcks/h3-ptd-v2-{part}/versions/1')
            for key in ('part', 'receipt_sha256', 'recipe_sha256', 'payload_files',
                        'payload_bytes', 'payload_tree_sha256'):
                self.assertEqual(actual[key], reference[key], key)
            self.assertEqual(actual['status'], 'verified')
            proof = self.root / f'{part}.remote.json'
            proof.write_text(json.dumps(actual), encoding='utf-8')
            check_readback_proof(proof, root, part, source_sha,
                                 recipe['recipe_sha256'], expected_layers=1)

    def test_tampered_payload_rejected(self):
        part = self.root / 'hop_00'
        receipt = json.loads((part / 'receipt.json').read_text())
        source_sha = sha(part / 'receipt.json')
        shard = part / 'dit.block00.rank0.safetensors'
        data = shard.read_bytes()
        shard.write_bytes(data[:-1] + bytes((data[-1] ^ 1,)))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            verify(part, 'hop_00', source_sha, receipt['recipe_sha256'],
                   'pamcks/h3-ptd-v2-hop-00/versions/1', expected_layers=1)

    def test_unversioned_identity_and_wrong_source_rejected(self):
        part = self.root / 'shared'
        receipt = json.loads((part / 'receipt.json').read_text())
        with self.assertRaisesRegex(ValueError, 'versions/N'):
            verify(part, 'shared', sha(part / 'receipt.json'),
                   receipt['recipe_sha256'], 'pamcks/h3-ptd-v2-shared',
                   expected_layers=1)
        with self.assertRaisesRegex(ValueError, 'receipt SHA256'):
            verify(part, 'shared', '0' * 64, receipt['recipe_sha256'],
                   'pamcks/h3-ptd-v2-shared/versions/1', expected_layers=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
