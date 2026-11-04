"""Local synthetic and frozen-notebook regression for PTD preadv stripe study."""
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from benchmarks import build_ptd_preadv_stripe_cpu_job as builder
from benchmarks import probe_ptd_preadv_stripe_cpu as probe


class TestPreadvStripeCPU(unittest.TestCase):
    def test_coverage_and_nonoverlap(self):
        payload_bytes = 77 * probe.MIB + 499
        ranges = [(0, 55 * probe.MIB, "large"),
                  (55 * probe.MIB, payload_bytes, "small")]
        for mode, stripe in (("tensor_pread", 8), ("tensor_preadv", 8),
                             ("stripe_preadv", 8), ("stripe_preadv", 4)):
            chunks = probe.tasks_for(payload_bytes, ranges, mode, stripe)
            self.assertEqual(chunks[0][0], 0)
            self.assertEqual(sum(size for _, size in chunks), payload_bytes)
            cursor = 0
            for offset, size in chunks:
                self.assertEqual(offset, cursor)
                self.assertGreater(size, 0)
                cursor += size
            self.assertEqual(cursor, payload_bytes)

    def test_unique_immutable_shards_and_overlay_cases(self):
        found = {
            hop: {"blocks": {layer: {"stem": f"b{layer}", "path": f"h{hop}-{layer}"}
                             for layer in range(22)},
                  "overlay": {"stem": "aux_overlay", "path": f"h{hop}-aux"}}
            for hop in probe.HOPS
        }
        cases = probe.case_plans(found)
        self.assertEqual(len(cases), 17)
        self.assertEqual(sum(c["kind"] == "block" for c in cases), 11)
        self.assertEqual(sum(c["kind"] == "overlay" for c in cases), 6)
        files = [file["path"] for case in cases for file in case["files"]]
        self.assertEqual(len(files), len(set(files)))
        self.assertEqual({c["workers"] for c in cases}, {1, 2, 4, 8})
        self.assertEqual({c["stripe_MiB"] for c in cases}, {4, 8})

    def test_safetensors_header_plan_rejects_hole(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "data.safetensors"
            correct = {"one": {"dtype": "U8", "shape": [4],
                               "data_offsets": [0, 4]},
                       "two": {"dtype": "U8", "shape": [3],
                               "data_offsets": [4, 7]}}
            for data, expected in ((correct, False),
                                   ({**correct, "two": {"dtype": "U8", "shape": [3],
                                       "data_offsets": [5, 8]}}, True)):
                header = json.dumps(data).encode()
                file.write_bytes(struct.pack("<Q", len(header)) + header +
                                 (b"12345678" if expected else b"1234567"))
                if expected:
                    with self.assertRaises(ValueError):
                        probe.safetensors_plan(file)
                else:
                    recorded_header, payload, ranges, signature = probe.safetensors_plan(file)
                    self.assertEqual(recorded_header, struct.pack("<Q", len(header)) + header)
                    self.assertEqual(payload, 7)
                    self.assertEqual(len(ranges), 2)
                    self.assertEqual(signature[0], file.stat().st_size)

    def test_frozen_private_cpu_job(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "notebook"
            receipt = builder.build(destination)
            script = (destination / "experiment.py").read_bytes()
            ast.parse(script)
            meta = json.loads((destination / "kernel-metadata.json").read_text())
            self.assertEqual(meta["id"], builder.KERNEL_ID)
            self.assertEqual(meta["dataset_sources"], builder.HOPS)
            self.assertFalse(meta["enable_gpu"])
            self.assertFalse(meta["enable_internet"])
            self.assertTrue(meta["is_private"])
            self.assertEqual(receipt["block_unique_shards"], 22)
            self.assertEqual(receipt["overlay_unique_hops"], 6)
            self.assertEqual(hashlib.sha256(script).hexdigest(),
                             receipt["notebook_sha256"])

    @unittest.skipUnless(hasattr(os, "pread") and hasattr(os, "preadv"),
                         "Linux os.preadv required")
    def test_linux_synthetic_end_to_end(self):
        self.assertTrue(probe.synthetic_selftest())


if __name__ == "__main__":
    unittest.main()
