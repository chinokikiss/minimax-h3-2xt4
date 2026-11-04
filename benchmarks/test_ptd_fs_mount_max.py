"""CPU-only contract tests for the PTD synthetic/public-private and real-Hop IO sweep."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import build_ptd_fs_mount_max_job as builder
from benchmarks import probe_ptd_fs_mount_max as probe


class TestPTDFSMountMax(unittest.TestCase):
    def test_nonoverlapping_plan(self):
        probe.selftest()

    def test_flat_and_payload_layout_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "flat"
            nested = root / "nested"
            flat.mkdir()
            (nested / "payload").mkdir(parents=True)
            (flat / "manifest.json").write_text("{}")
            (nested / "payload" / "manifest.json").write_text("{}")
            with patch.object(probe, "candidate_roots", return_value=[flat, nested]):
                self.assertEqual(probe.locate("any/example", "manifest.json"), flat)
            with patch.object(probe, "candidate_roots", return_value=[flat / "unused", nested, nested / "payload"]):
                self.assertEqual(probe.locate("any/example", "manifest.json"), nested / "payload")

    def test_builder_binds_all_eight_mounts_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = builder.build(Path(tmp) / "job")
            self.assertEqual(result["real_hop_rank_streams"], 12)
            self.assertEqual(result["real_worker_profiles"], [1, 2, 4, 8, 16, 32])
            meta = json.loads((Path(tmp) / "job" / "kernel-metadata.json").read_text())
            self.assertEqual(meta["dataset_sources"], builder.SYNTH + builder.REAL)
            self.assertFalse(meta["enable_gpu"])
            self.assertFalse(meta["enable_internet"])
            script = (Path(tmp) / "job" / "experiment.py").read_bytes()
            self.assertEqual(hashlib.sha256(script).hexdigest(), result["script_sha256"])


if __name__ == "__main__":
    unittest.main()
