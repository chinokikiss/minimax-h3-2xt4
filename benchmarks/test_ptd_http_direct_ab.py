"""CPU-only, network-free tests for HTTP-direct vs GCSFuse Kaggle job."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from benchmarks import probe_ptd_http_direct_ab as probe
from benchmarks import build_ptd_http_direct_ab_job as builder


class TestHTTPDirectAB(unittest.TestCase):
    def test_disjoint_selection_and_early_stop(self):
        probe.selftest()
        self.assertEqual(
            probe.early_stop_reason({"status": "timeout"}, 90, 0),
            "hard_timeout_skip_higher_workers")
        self.assertEqual(
            probe.early_stop_reason({"status": "blocked_space"}, 90, 0),
            "failed_or_auth_blocked_skip_higher_workers")
        self.assertIsNone(
            probe.early_stop_reason({"status": "complete"}, 90, 1))

    def test_network_rate_is_diagnostic(self):
        samples = [(i, i * 123_000_000) for i in range(10)]
        result = probe.summarize_network(samples)
        self.assertTrue(result["p10_qualified"])
        self.assertAlmostEqual(result["p10_MBps"], 123.0)
        self.assertEqual(result["seconds_sampled"], 9)
        self.assertIn("other_traffic", result["scope"])

    def test_http_local_path_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "owned"
            root.mkdir()
            target = root / "data"
            target.write_bytes(b"abc")
            with mock.patch.object(probe.mounted, "mount_type",
                                   return_value={"filesystem": "overlay"}):
                observed, _ = probe.ensure_local_path(target, root)
                self.assertEqual(observed, target.resolve())
                with self.assertRaises(RuntimeError):
                    probe.ensure_local_path(Path(tmp) / "elsewhere", root)
            with mock.patch.object(probe.mounted, "mount_type",
                                   return_value={"filesystem": "fuse.gcsfuse"}):
                with self.assertRaises(RuntimeError):
                    probe.ensure_local_path(target, root)

    def test_frozen_cpu_kernel_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "job"
            receipt = builder.build(output)
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            experiment = (output / "experiment.py").read_text()
            ast.parse(experiment)
            self.assertFalse(metadata["enable_gpu"])
            self.assertTrue(metadata["enable_internet"])
            self.assertTrue(metadata["is_private"])
            self.assertEqual(metadata["dataset_sources"], builder.SYNTH + builder.REAL)
            self.assertEqual(receipt["outer_supervisor_timeout_seconds"], 420)
            self.assertEqual(receipt["HTTP_real_case_deadlines_seconds"]["16"], 55)
            self.assertIn("DISABLE_KAGGLE_CACHE", experiment)
            self.assertEqual(hashlib.sha256(
                (output / "experiment.py").read_bytes()).hexdigest(),
                receipt["notebook_sha256"])
            self.assertEqual(len(receipt["frozen_sources_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
