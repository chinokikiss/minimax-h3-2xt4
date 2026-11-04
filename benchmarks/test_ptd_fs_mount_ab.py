"""CPU-only contracts for the two-phase synthetic Kaggle mount A/B experiment."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks import probe_ptd_fs_mount_ab as probe
from benchmarks import build_ptd_fs_mount_ab_job as builder


class TestPTDFilesystemMountAB(unittest.TestCase):
    def test_unique_unrepeated_read_ranges(self):
        probe.selftest()

    def test_synthetic_payload_shared_across_public_private_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_shards, original_bytes = probe.SHARDS, probe.SHARD_BYTES
            try:
                probe.SHARDS = 4
                probe.SHARD_BYTES = 4 * probe.MIB
                source, manifest = probe.prepare_source(Path(temporary) / "test")
                private = probe.staging(Path(temporary) / "test", source, "private")
                public = probe.staging(Path(temporary) / "test", source, "public")
                self.assertEqual(manifest["bytes"], 16 * probe.MIB)
                self.assertEqual(
                    hashlib.sha256((private / "payload" / "manifest.json").read_bytes()).hexdigest(),
                    hashlib.sha256((public / "payload" / "manifest.json").read_bytes()).hexdigest(),
                )
                for shard in manifest["shards"]:
                    a = private / "payload" / shard["name"]
                    b = public / "payload" / shard["name"]
                    self.assertTrue(a.samefile(b))
                    self.assertEqual(a.stat().st_size, shard["bytes"])
                priv_meta = json.loads((private / "dataset-metadata.json").read_text())
                pub_meta = json.loads((public / "dataset-metadata.json").read_text())
                self.assertNotEqual(priv_meta["id"], pub_meta["id"])
                self.assertEqual(priv_meta["licenses"], pub_meta["licenses"])
            finally:
                probe.SHARDS = original_shards
                probe.SHARD_BYTES = original_bytes

    def test_publisher_and_reader_metadata_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            pub = builder.build("publish", Path(temporary) / "publish")
            reader = builder.build("read", Path(temporary) / "read")
            self.assertEqual(pub["source_sha256"], reader["source_sha256"])
            self.assertEqual(pub["dataset_sources"], [])
            self.assertEqual(reader["dataset_sources"], list(probe.PUBLISH_HANDLES.values()))
            for phase in ("publish", "read"):
                root = Path(temporary) / phase
                m = json.loads((root / "kernel-metadata.json").read_text())
                self.assertFalse(m["enable_gpu"])
                self.assertTrue(m["is_private"])
                self.assertEqual(m["enable_internet"], phase == "publish")
            script = (Path(temporary) / "publish" / "experiment.py").read_text()
            self.assertNotIn("KAGGLE_KEY=", script)
            self.assertNotIn("HF_TOKEN=", script)


if __name__ == "__main__":
    unittest.main()
