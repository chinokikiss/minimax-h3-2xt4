"""CPU-only checks for adaptive PTD mount-IO v2 scheduling."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks import probe_ptd_fs_mount_max_v2 as probe
from benchmarks import build_ptd_fs_mount_max_v2_job as builder


class TestAdaptiveFSMountV2(unittest.TestCase):
    def test_unique_ranges_and_confirmation(self):
        probe.selftest()

    def test_early_stop_trigger(self):
        self.assertEqual(
            probe.early_stop_reason({"status": "timeout"}, 100, 0),
            "pilot_timeout")
        self.assertEqual(
            probe.early_stop_reason({"status": "failed"}, 100, 0),
            "pilot_failed")
        self.assertEqual(
            probe.early_stop_reason({"status": "complete"}, 100, 2),
            "two_consecutive_below_30pct_of_best")
        self.assertIsNone(
            probe.early_stop_reason({"status": "complete"}, 100, 1))

    def test_frozen_kernel_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cpu_v2"
            receipt = builder.build(out)
            meta = json.loads((out / "kernel-metadata.json").read_text())
            self.assertEqual(meta["dataset_sources"], builder.SYNTH + builder.REAL)
            self.assertFalse(meta["enable_gpu"])
            self.assertFalse(meta["enable_internet"])
            self.assertEqual(receipt["per_case_deadlines_seconds"],
                             {"synthetic": 45, "pilot": 90, "confirm": 150})
            self.assertTrue(receipt["early_stop_after_pilot_timeout"])
            self.assertEqual(receipt["outer_supervisor_deadline_seconds"], 900)
            data = (out / "experiment.py").read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), receipt["script_sha256"])


if __name__ == "__main__":
    unittest.main()
