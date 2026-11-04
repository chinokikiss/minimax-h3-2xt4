"""Offline readback proofs and nine-part atomic publication gates for PTD v2."""
import json
import os
from pathlib import Path
import unittest

import benchmarks.test_ptd_v2_bake as bake_fixture
from patches.ptd.bake_v2 import assemble_bank_v2
from patches.ptd.bank_v2 import PTDBankV2
from patches.ptd.publish_v2 import (verify_part, check_readback_proof,
                                    prepare_private_kaggle_part)
from patches.tp2_disk_bake import atomic_json
from patches.tp2_bundle import sha


class ReadbackPublication(unittest.TestCase):
    def setUp(self):
        fixture = bake_fixture.BoundedParts('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture._bake()
        self.shared = self.root / 'shared'
        self.hops = [self.root / f'hop_{hop:02d}' for hop in range(8)]

    def _proofs(self):
        files = []
        for part, root in [('shared', self.shared)] + [
                (f'hop_{i:02d}', r) for i, r in enumerate(self.hops)]:
            value = verify_part(root, part, expected_receipt_sha256=sha(root / 'receipt.json'),
                                published_identity=f'owner/ptd-{part}@v1')
            proof = self.root / (part + '.proof.json')
            atomic_json(proof, value)
            self.assertEqual(value['status'], 'verified')
            self.assertGreater(value['payload_bytes'], 0)
            files.append(proof)
        return files

    def test_nine_verified_mounts_can_publish_fixture_manifest(self):
        proofs = self._proofs()
        manifest = assemble_bank_v2(self.root / 'bank_v2.json', self.shared,
                                    self.hops, allow_fixture=True, readback_proofs=proofs)
        bank = PTDBankV2(manifest, allow_fixture=True)
        self.assertEqual(bank.reader(7).hop, 7)
        self.assertEqual(json.loads(manifest.read_text())['parts']['hop_07']['identity'],
                         'owner/ptd-hop_07@v1')

    def test_tampered_remote_payload_is_rejected_by_full_sha256(self):
        part = self.hops[0]
        shard = part / 'dit.block00.rank0.safetensors'
        before = shard.stat()
        with shard.open('r+b') as stream:
            stream.seek(-1, os.SEEK_END)
            b = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes((b[0] ^ 1,)))
        os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            verify_part(part, 'hop_00')

    def test_mismatched_readback_receipt_blocks_final_manifest(self):
        proofs = self._proofs()
        value = json.loads(proofs[4].read_text())
        value['receipt_sha256'] = 'invalid'
        atomic_json(proofs[4], value)
        manifest = self.root / 'bank_v2.json'
        with self.assertRaisesRegex(ValueError, 'matching remote readback'):
            assemble_bank_v2(manifest, self.shared, self.hops,
                             allow_fixture=True, readback_proofs=proofs)
        self.assertFalse(manifest.exists())

    def test_shared_schema_tamper_fails_receipt_pinned_sha256(self):
        header = self.shared / 'native_header_v2.json'
        before = header.stat()
        with header.open('r+b') as stream:
            stream.seek(-1, os.SEEK_END)
            end = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(b'X' if end != b'X' else b'Y')
        os.utime(header, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            verify_part(self.shared, 'shared')

    def test_pinned_dataset_identity_and_recipe_layer_gate(self):
        valid_id = 'pamcks/ptd-hop-00/versions/2'
        proof = verify_part(self.hops[0], 'hop_00',
                            expected_receipt_sha256=sha(self.hops[0] / 'receipt.json'),
                            published_identity=valid_id)
        path = self.root / 'pinned.proof.json'
        atomic_json(path, proof)
        accepted = check_readback_proof(
            path, self.hops[0], 'hop_00', sha(self.hops[0] / 'receipt.json'),
            proof['recipe_sha256'], expected_layers=1)
        self.assertEqual(accepted['published_identity'], valid_id)
        with self.assertRaisesRegex(ValueError, 'layer count'):
            check_readback_proof(path, self.hops[0], 'hop_00',
                                 sha(self.hops[0] / 'receipt.json'),
                                 proof['recipe_sha256'], expected_layers=50)
        proof['source_receipt_sha256'] = ''
        atomic_json(path, proof)
        with self.assertRaisesRegex(ValueError, 'independently saved source receipt'):
            check_readback_proof(path, self.hops[0], 'hop_00',
                                 sha(self.hops[0] / 'receipt.json'), proof['recipe_sha256'])
        proof['source_receipt_sha256'] = sha(self.hops[0] / 'receipt.json')
        proof['published_identity'] = 'pamcks/ptd-hop-00'
        atomic_json(path, proof)
        with self.assertRaisesRegex(ValueError, 'version/commit-pinned'):
            check_readback_proof(path, self.hops[0], 'hop_00',
                                 sha(self.hops[0] / 'receipt.json'), proof['recipe_sha256'])

    def test_kaggle_private_staging_is_zero_copy_and_nonoverwriting(self):
        original = sorted(p.name for p in self.hops[0].iterdir())
        staged = self.root / 'kaggle-stage-hop00'
        result = prepare_private_kaggle_part(
            self.hops[0], 'hop_00', staged, dataset_id='pamcks/h3-ptd-hop00-sha',
            license_name='other')
        self.assertEqual(result['status'], 'staged_not_uploaded')
        self.assertNotIn('-u', result['upload_command'])
        self.assertEqual(result['upload_command'][-2:], ['-r', 'tar'])
        self.assertEqual(sorted(p.name for p in staged.iterdir()),
                         ['dataset-metadata.json', 'payload'])
        self.assertEqual(original, sorted(p.name for p in self.hops[0].iterdir()))
        meta = json.loads((staged / 'dataset-metadata.json').read_text())
        self.assertEqual(meta['id'], 'pamcks/h3-ptd-hop00-sha')
        shard = 'dit.block00.rank0.safetensors'
        self.assertEqual((staged / 'payload' / shard).stat().st_ino,
                         (self.hops[0] / shard).stat().st_ino)
        self.assertEqual(verify_part(staged / 'payload', 'hop_00')['receipt_sha256'],
                         sha(self.hops[0] / 'receipt.json'))
        with self.assertRaises(FileExistsError):
            prepare_private_kaggle_part(self.hops[0], 'hop_00', staged,
                                        dataset_id='pamcks/another-slug',
                                        license_name='other')

    def test_kaggle_staging_refuses_surplus_files_or_unversioned_handles(self):
        with self.assertRaisesRegex(ValueError, 'unversioned Kaggle'):
            prepare_private_kaggle_part(self.hops[0], 'hop_00',
                                        self.root / 'not-started',
                                        dataset_id='pamcks/hop/versions/1', license_name='other')
        extra = self.hops[0] / 'unrelated-secret.txt'
        extra.write_text('not part of a verified upload', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unexpected part-root'):
            prepare_private_kaggle_part(self.hops[0], 'hop_00',
                                        self.root / 'not-started',
                                        dataset_id='pamcks/safe-slug', license_name='other')
        self.assertFalse((self.root / 'not-started').exists())

    def test_assembly_rejects_duplicate_mounts(self):
        with self.assertRaisesRegex(ValueError, 'distinct immutable part mounts'):
            assemble_bank_v2(self.root / 'bank_v2.json', self.shared,
                             [self.hops[0]] * 8, allow_fixture=True)

    def test_production_manifest_requires_readback_proofs(self):
        with self.assertRaisesRegex(ValueError, 'nine remote readback proofs'):
            assemble_bank_v2(self.root / 'bank_v2.json', self.shared, self.hops)
        self.assertFalse((self.root / 'bank_v2.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
