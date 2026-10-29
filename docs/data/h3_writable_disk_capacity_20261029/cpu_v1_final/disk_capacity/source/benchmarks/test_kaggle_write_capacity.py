"""Verify writeback accounting and cleanup without filling a local disk."""
import errno
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from benchmarks import kaggle_write_capacity as probe


class CapacityContracts(unittest.TestCase):
    def test_mount_parser_decodes_escaped_paths_and_keeps_ro(self):
        rows = probe.mounts('1 0 0:1 / / rw - overlay overlay rw\n'
                            '2 1 8:1 / /mount\\040name ro - ext4 /dev/sda ro\n')
        self.assertEqual(rows[1]['mount'], '/mount name')
        self.assertEqual(rows[1]['options'], 'ro')

    def test_disk_aliases_are_deduplicated_and_overlay_is_last(self):
        def row(path, device, filesystem):
            return dict(path=path, st_dev=device, disk_candidate=True,
                        mount=dict(filesystem=filesystem, source=filesystem))
        rows = [row('/tmp', 1, 'overlay'), row('/kaggle/temp', 1, 'overlay'),
                row('/kaggle/working', 2, 'ext4'), row('/kaggle/huggingface', 2, 'ext4')]
        result = probe.select_disks(rows, 2)
        self.assertEqual([r['base'] for r in result], ['/kaggle/working', '/tmp'])
        self.assertEqual(len(result[0]['aliases']), 2)

    def test_real_full_write_readback_and_cleanup_preserve_unrelated_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            untouched = root/'unrelated.bin'
            untouched.write_bytes(b'preserve me')
            result = probe.measure(root, root/'result.json', cap_bytes=24576,
                                   chunk_bytes=8192, report_headroom=0)
            self.assertEqual(result['verified_synced_bytes'], 24576)
            self.assertEqual(result['stop_reason'], 'configured_byte_cap')
            self.assertTrue(all(c['synced'] and c['readback_verified'] for c in result['completed_chunks']))
            self.assertEqual(result['cleanup_errors'], [])
            self.assertFalse(list(root.glob('h3_write_capacity_*')))
            self.assertEqual(untouched.read_bytes(), b'preserve me')

    def test_failed_sync_file_is_excluded_from_durable_capacity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = probe.write_chunk
            calls = 0
            def write(path, size, deadline):
                nonlocal calls
                calls += 1
                if calls == 2:
                    path.write_bytes(b'partial')
                    raise OSError(errno.ENOSPC, 'injected backing storage exhaustion')
                return original(path, size, deadline)
            with patch.object(probe, 'write_chunk', side_effect=write):
                result = probe.measure(root, root/'result.json', cap_bytes=24576,
                                       chunk_bytes=8192, report_headroom=0)
            self.assertEqual(result['verified_synced_bytes'], 8192)
            self.assertEqual(result['unqualified_last_file_bytes'], 7)
            self.assertEqual(result['status'], 'observed_boundary')
            self.assertFalse(result['exact_capacity'])
            self.assertFalse(list(root.glob('h3_write_capacity_*')))

    def test_eio_is_preserved_as_writeback_fault_not_exact_capacity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(probe, 'write_chunk', side_effect=OSError(errno.EIO, 'injected sync fault')):
                result = probe.measure(root, root/'result.json', cap_bytes=16384,
                                       chunk_bytes=8192, report_headroom=0)
            self.assertEqual(result['status'], 'writeback_error')
            self.assertEqual(result['verified_synced_bytes'], 0)
            self.assertEqual(json.loads((root/'result.json').read_text())['errno'], errno.EIO)

    def test_output_headroom_stops_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(probe, 'write_chunk') as write:
                result = probe.measure(root, root/'result.json', cap_bytes=8192,
                                       chunk_bytes=8192, report_headroom=probe.usage(root)['free_bytes'])
            self.assertEqual(result['stop_reason'], 'report_filesystem_headroom')
            self.assertEqual(result['verified_synced_bytes'], 0)
            write.assert_not_called()

    def test_deadline_aborts_before_data_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'probe.bin'
            with self.assertRaises(TimeoutError):
                probe.write_chunk(path, 8192, time.monotonic()-1)
            self.assertEqual(path.stat().st_size, 0)

    def test_fsync_failure_propagates_before_readback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'probe.bin'
            with patch.object(probe.os, 'fsync', side_effect=OSError(errno.EIO, 'sync fault')) as sync:
                with self.assertRaises(OSError):
                    probe.write_chunk(path, 8192, time.monotonic()+10)
            self.assertEqual(sync.call_count, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
