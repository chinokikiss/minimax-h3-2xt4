"""CPU-only contract tests for mounted PTD throughput metrics."""
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from benchmarks.probe_ptd_mounted_io_throughput import (
    UniqueRegions, _percentile, measure,
)


@contextmanager
def platform_pread():
    if hasattr(os, 'pread'):
        yield
        return
    lock = threading.Lock()
    def compat(fd, size, offset):
        with lock:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, size)
    with patch.object(os, 'pread', compat, create=True):
        yield


class MountedThroughputTests(unittest.TestCase):
    def test_percentile_uses_interpolated_observations(self):
        self.assertEqual(_percentile([10., 20., 30., 40.], .1), 13.)
        self.assertEqual(_percentile([10., 20., 30., 40.], .5), 25.)
        self.assertEqual(_percentile([10., 20., 30., 40.], .9), 37.)
        self.assertIsNone(_percentile([], .1))

    def test_unique_regions_never_repeat_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            files = [(Path(directory) / f"{i}.data", 4096) for i in range(3)]
            cursor = UniqueRegions(files)
            first = cursor.segments(5500, 1024)
            second = cursor.segments(5500, 1024)
            spans = [(str(p), start + j) for p, start, size in first + second
                     for j in range(size)]
            self.assertEqual(len(spans), 11000)
            self.assertEqual(len(spans), len(set(spans)))

    def test_probe_reports_windowed_mbps_and_keeps_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for i in range(4):
                path = root / f"shard{i}.safetensors"
                path.write_bytes(b"x" * (16 * 1024))
                sources.append((path, path.stat().st_size))
            output = root / "metrics.json"
            with platform_pread():
                report = measure(sources, profiles=(1, 2), windows=2,
                                 window_bytes=4096, chunk_bytes=1024,
                                 target_mbps=1e-6, output=output)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["readings"]), 4)
            self.assertEqual(sum(r["application_read_bytes"]
                                 for r in report["readings"]), 16384)
            self.assertTrue(all(r["MBps"] > 0 for r in report["readings"]))
            self.assertTrue(all(r["samples"] == 2 for r in report["summary"]))
            self.assertTrue(report["current_1_worker_passes"])
            self.assertIn("not Kaggle download", output.read_text())

    def test_insufficient_unique_bytes_fail_before_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small"
            path.write_bytes(b"x" * 1024)
            with platform_pread(), self.assertRaisesRegex(ValueError, "Insufficient unique"):
                measure([(path, 1024)], profiles=(1, 4), windows=2,
                        window_bytes=1024, chunk_bytes=512)


if __name__ == "__main__":
    unittest.main()
