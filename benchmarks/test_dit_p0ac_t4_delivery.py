"""Read-only source-bound delivery tests for the new P0-A1/P0-C CPU/T4 notebooks."""
import ast
import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "docs/data/h3_dit_p0ac_t4_20261102"
E2E = ROOT / "docs/data/h3_t4_fp16_512_sol_all_8nfe_10s_20261101/gpu_job_v2"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def extract(mode):
    directory = JOB / (mode + "_job")
    notebook = (directory / "experiment.ipynb").read_bytes()
    candidate = json.loads(notebook)
    code = "".join(candidate["cells"][0]["source"])
    anchor = "worker.write_bytes(base64.b64decode("
    assert code.count(anchor) == 2
    at = code.rfind(anchor) + len(anchor)
    end = code.index("))", at)
    worker = base64.b64decode(ast.literal_eval(code[at:end]), validate=True).decode()
    tree = ast.parse(worker)
    assignments = {
        node.targets[0].id: node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    manifest = ast.literal_eval(assignments["manifest"])
    expression = assignments["raw"]
    compressed = base64.b85decode(ast.literal_eval(expression.args[0]))
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(compressed)), mode="r:") as archive:
        files = {entry.name: archive.extractfile(entry).read() for entry in archive.getmembers()}
    return directory, notebook, code, worker, manifest, files, compressed


class Delivery(unittest.TestCase):
    def test_frozen_source_coverage_and_no_archived_p01(self):
        prior = json.loads((E2E / "source_binding.json").read_text(encoding="utf-8"))
        expected_base = set(prior["overlay_hashes"])
        self.assertEqual(len(expected_base), 90)
        for mode in ("cpu", "gpu"):
            with self.subTest(mode=mode):
                directory, notebook, code, worker, manifest, files, compressed = extract(mode)
                self.assertEqual(set(files), set(manifest))
                self.assertEqual(len(files), 99)
                self.assertTrue(expected_base.issubset(files))
                for name, blob in files.items():
                    self.assertEqual(len(blob), manifest[name]["bytes"], name)
                    self.assertEqual(digest(blob), manifest[name]["sha256"], name)
                    if name.endswith(".py"):
                        ast.parse(blob, filename=name)
                for name in (
                    "patches/h3_operator_policy.py", "patches/tp_dit.py",
                    "patches/dit_column_linear.py", "patches/tilelang_row_quant.py",
                    "patches/tilelang_transport_p2.py", "patches/tilelang_swiglu_quant.py",
                    "benchmarks/dit_p0ac_t4_candidates.py",
                    "benchmarks/run_dit_p0ac_case.py",
                    "benchmarks/run_dit_p0ac_batch.py",
                    "benchmarks/test_dit_p0ac_t4_preflight.py",
                ):
                    self.assertIn(name, files)
                    self.assertEqual(files[name], (ROOT / name).read_bytes())
                for removed in ("_vram_p0_1", "_vram_p0_1_inplace", "inplace_quant"):
                    self.assertNotIn(removed, files["patches/tp_dit.py"].decode())
                self.assertIn("run_dit_p0ac_batch.py", worker)
                self.assertIn("os.execve(sys.executable,argv,env)", worker)
                self.assertIn("ready_for_device_gate", code)
                binding = json.loads((directory / "source_binding.json").read_text())
                self.assertEqual(binding["source_overlay_sha256"], digest(compressed))
                self.assertEqual(binding["mode"], mode)
                self.assertEqual(binding["submission_count"], 0)
                self.assertFalse(binding["full_pipeline"])

    def test_independent_private_t4_metadata(self):
        cpu = json.loads((JOB / "cpu_job/kernel-metadata.json").read_text())
        gpu = json.loads((JOB / "gpu_job/kernel-metadata.json").read_text())
        for metadata in (cpu, gpu):
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_tpu"])
            self.assertEqual(metadata["docker_image_pinning_type"], "original")
        self.assertFalse(cpu["enable_gpu"])
        self.assertTrue(gpu["enable_gpu"])
        self.assertEqual(gpu["machine_shape"], "NvidiaTeslaT4")
        self.assertNotEqual(cpu["id"], gpu["id"])
        self.assertNotIn("v3", gpu["id"])

    def test_immutable_source_snapshot_and_receipts(self):
        manifest = json.loads((JOB / "build_manifest.json").read_text())
        self.assertEqual(manifest["status"], "prepared_not_submitted")
        self.assertEqual(manifest["source_files"], 99)
        self.assertEqual(manifest["previous_e2e_overlay_files"], 90)
        self.assertEqual(manifest["compressed_bytes"], len(extract("gpu")[-1]))
        self.assertTrue(manifest["no_status_polling"])
        self.assertTrue(manifest["no_live_log_polling"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
