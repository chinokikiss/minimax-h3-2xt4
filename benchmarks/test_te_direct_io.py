"""Portable adapter tests; Linux support must be established on the job filesystem."""
import ctypes
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import te_direct_io as adapter
from benchmarks.te_direct_io import DirectFile


class DirectIOAdapter(unittest.TestCase):
    def fixture(self, path, *, fail=False, limit=None):
        calls = []
        def opener(name, flags):
            return os.open(name, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
        def reader(fd, buffers, offset):
            target = buffers[0]
            self.assertEqual(offset % 4096, 0)
            self.assertEqual(len(target) % 4096, 0)
            self.assertEqual(ctypes.addressof(ctypes.c_char.from_buffer(target)) % 4096, 0)
            calls.append((offset, len(target)))
            if fail:
                raise OSError('direct I/O unsupported')
            os.lseek(fd, offset, os.SEEK_SET)
            data = os.read(fd, min(len(target), limit or len(target)))
            target[:len(data)] = data
            return len(data)
        return DirectFile(path, chunk_bytes=8192, opener=opener, reader=reader,
                          alignment=4096), calls

    def test_unaligned_offsets_lengths_chunk_boundaries_and_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'unaligned.bin'
            raw = bytes(range(251))*103
            path.write_bytes(raw)
            file, calls = self.fixture(path)
            with file:
                for offset, size in [(0, 8), (8, 1337), (4093, 17001), (len(raw)-37, 4096)]:
                    self.assertEqual(file.pread(size, offset), raw[offset:offset+size])
                self.assertEqual(file.pread(10, len(raw)), b'')
                file.seek(-91, os.SEEK_END)
                self.assertEqual(file.read(1000), raw[-91:])
                self.assertEqual(file.tell(), len(raw))
            self.assertTrue(calls)
            self.assertTrue(file.closed)

    def test_hash_file_digest_uses_bounded_readinto_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'hash.bin'
            raw = os.urandom(67013)
            path.write_bytes(raw)
            file, calls = self.fixture(path)
            with file:
                self.assertEqual(hashlib.file_digest(file, 'sha256').hexdigest(),
                                 hashlib.sha256(raw).hexdigest())
            self.assertLessEqual(max(size for _, size in calls), 8192)

    def test_short_aligned_reads_make_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'short.bin'
            raw = os.urandom(20003)
            path.write_bytes(raw)
            file, _ = self.fixture(path, limit=4096)
            with file:
                self.assertEqual(file.read(30000), raw)

    def test_direct_error_is_not_silently_replaced_with_buffered_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'failure.bin'
            path.write_bytes(b'x'*8192)
            file, _ = self.fixture(path, fail=True)
            with file, self.assertRaises(OSError):
                file.read(8)

    def test_checkpoint_ranges_and_both_hashes_use_the_adapter_without_math_changes(self):
        import torch
        from safetensors.torch import save_file
        from patches import windowed_tp as wt, tp2_bundle as bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'te.block00.rank0.safetensors'
            expected = {'w':torch.arange(120, dtype=torch.int8).reshape(12,10),
                        's':torch.arange(12, dtype=torch.float32)}
            save_file(expected, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            module = torch.nn.Linear(7,9)
            native_tensor_hashes = bundle.state_hashes(module)
            audit = adapter.IOAudit(root/'audit.jsonl')
            test = self
            # Inject syscalls without claiming local Windows O_DIRECT support.
            def fake_direct(name, **kwargs):
                def opened(path, flags):
                    return os.open(path, os.O_RDONLY | getattr(os,'O_BINARY',0))
                def read(fd, buffers, offset):
                    target = buffers[0]
                    test.assertEqual(offset % 4096, 0)
                    test.assertEqual(ctypes.addressof(ctypes.c_char.from_buffer(target)) % 4096, 0)
                    os.lseek(fd, offset, os.SEEK_SET)
                    raw = os.read(fd, len(target))
                    target[:len(raw)] = raw
                    return len(raw)
                return DirectFile(name, opener=opened, reader=read, alignment=4096, **kwargs)
            originals = (Path.open, wt.os, wt.CheckpointReader.read_tensor, bundle.sha, bundle.state_hashes)
            try:
                with patch.object(adapter, 'DirectFile', fake_direct):
                    adapter.install(root, 'direct', audit)
                    reader = wt.CheckpointReader(path)
                    self.assertTrue(torch.equal(reader.read_tensor('w'), expected['w']))
                    self.assertTrue(torch.equal(reader.read_tensor('w', rows=[(1,4)], columns=(2,7)), expected['w'][1:4,2:7]))
                    self.assertTrue(torch.equal(reader.read_tensor('s'), expected['s']))
                    self.assertEqual(bundle.sha(path), digest)
                    self.assertEqual(bundle.state_hashes(module), native_tensor_hashes)
            finally:
                Path.open, wt.os, wt.CheckpointReader.read_tensor, bundle.sha, bundle.state_hashes = originals
                audit.close()
            import json
            rows = [json.loads(line) for line in (root/'audit.jsonl').read_text().splitlines()]
            self.assertTrue(any(r['kind'] == 'direct_syscall' and r['category'] == 'file_sha256' for r in rows))
            self.assertEqual(sum(r['kind'] == 'tensor_sha256' for r in rows), 1)
            self.assertEqual(sum(r['kind'] == 'file_sha256' for r in rows), 1)


if __name__ == '__main__':
    unittest.main()
