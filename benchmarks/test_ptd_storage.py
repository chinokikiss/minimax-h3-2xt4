"""Scratch faults must stop before models are downloaded or banks published."""
import errno
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from benchmarks.ptd_storage import capacity_probe, failure_evidence


class ScratchAdmission(unittest.TestCase):
    def test_allocation_failure_is_recorded_and_owned_probe_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); receipt = root/'receipt.json'
            original = OSError(errno.ENOSPC, 'injected filesystem allocation limit')
            with patch('os.posix_fallocate', create=True, side_effect=original):
                with self.assertRaises(OSError) as raised: capacity_probe(root, 16384, receipt)
            self.assertIs(raised.exception, original)
            self.assertEqual(json.loads(receipt.read_text())['errno'], errno.ENOSPC)
            self.assertFalse(list(root.glob('.h3_capacity_*')))

    def test_fsync_eio_is_not_bypassed_or_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); receipt = root/'receipt.json'
            def allocate(fd, offset, size): os.ftruncate(fd, size)
            stat = SimpleNamespace(st_size=16384, st_blocks=32)
            with patch('os.posix_fallocate', create=True, side_effect=allocate), \
                    patch('os.fstat', return_value=stat), \
                    patch('os.fsync', side_effect=OSError(errno.EIO, 'injected fsync failure')) as sync:
                with self.assertRaises(OSError): capacity_probe(root, 16384, receipt)
            self.assertEqual(sync.call_count, 1)
            self.assertEqual(json.loads(receipt.read_text())['errno'], errno.EIO)
            self.assertFalse(list(root.glob('.h3_capacity_*')))

    def test_sparse_allocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('os.posix_fallocate', create=True), \
                    patch('os.fstat', return_value=SimpleNamespace(st_size=16384, st_blocks=0)):
                with self.assertRaisesRegex(OSError, 'non-sparse'): capacity_probe(root, 16384, root/'receipt.json')

    @unittest.skipUnless(hasattr(os, 'posix_fallocate'), 'Linux allocation check, not available on Windows')
    def test_real_small_allocation_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = capacity_probe(root, 16384, root/'receipt.json')
            self.assertEqual(result['status'], 'passed')
            self.assertTrue(result['write_sync_readback'])
            self.assertFalse(list(root.glob('.h3_capacity_*')))

    def test_failure_evidence_preserves_receipts_without_weight_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'scratch'; hop = root/'eight_hop_tp2.partial/hop_00'
            hop.mkdir(parents=True)
            (hop/'shard.json').write_text('{"native_verified":true}')
            (hop/'shard.json.partial').write_text('{')
            (hop/'shard.safetensors').write_bytes(b'private weight payload')
            output = Path(directory)/'evidence'
            result = failure_evidence(root, output)
            self.assertEqual(result['receipts'], 2)
            self.assertEqual(result['groups']['eight_hop_tp2.partial']['files'], 3)
            with zipfile.ZipFile(output/'partial_bank_receipts.zip') as archive:
                self.assertEqual(len(archive.namelist()), 2)
                self.assertFalse(any(n.endswith('.safetensors') for n in archive.namelist()))


if __name__ == '__main__': unittest.main(verbosity=2)
