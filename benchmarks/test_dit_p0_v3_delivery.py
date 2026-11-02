"""Prove that historical V3 notebooks embed their frozen, verified Sol-v2 sources.

The V2 T4 job failed before kernels because the frozen bootstrap kept an old
h3_operator_policy.py. These tests inspect the exact nested notebook payload,
not just a manifest written by the local builder.
"""
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
EVIDENCE = ROOT / "docs/data/h3_dit_p0_lifetime_20261102_v3"
E2E = ROOT / "docs/data/h3_t4_fp16_512_sol_all_8nfe_10s_20261101/gpu_job_v2"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def code_file(mode):
    folder = EVIDENCE / ("cpu_job" if mode == "cpu" else "gpu_job")
    notebook = json.loads((folder / "experiment.ipynb").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][0]["source"])
    anchor = "worker.write_bytes(base64.b64decode("
    assert code.count(anchor) == 2, code.count(anchor)
    start = code.rfind(anchor) + len(anchor)
    end = code.index("))", start)
    wrapped = ast.literal_eval(code[start:end])
    worker = base64.b64decode(wrapped, validate=True).decode("utf-8")
    return code, worker

def payload(mode):
    code, worker = code_file(mode)
    tree = ast.parse(worker)
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    manifest = ast.literal_eval(assignments["manifest"])
    expression = assignments["raw"]
    assert isinstance(expression, ast.Call) and expression.func.attr == "b85decode"
    compressed = base64.b85decode(ast.literal_eval(expression.args[0]))
    buffer = io.BytesIO(lzma.decompress(compressed))
    with tarfile.open(fileobj=buffer, mode="r:") as archive:
        files = {member.name: archive.extractfile(member).read()
                 for member in archive.getmembers()}
    return code, worker, manifest, files, compressed

class Delivery(unittest.TestCase):
    def test_both_notebooks_have_complete_verified_sol_closure(self):
        proven = json.loads((E2E / "source_binding.json").read_text(encoding="utf-8"))
        proven_names = set(proven["overlay_hashes"])
        self.assertEqual(len(proven_names), 90)
        for mode in ("cpu", "gpu"):
            with self.subTest(mode=mode):
                code, worker, manifest, files, compressed = payload(mode)
                self.assertEqual(set(files), set(manifest))
                self.assertTrue(proven_names.issubset(files))
                self.assertEqual(len(files), 97)
                for name, raw in files.items():
                    self.assertEqual(sha(raw), manifest[name]["sha256"], name)
                    self.assertEqual(len(raw), manifest[name]["bytes"], name)
                for name in (
                    "patches/h3_operator_policy.py",
                    "patches/sol_attention.py",
                    "patches/sol_sm75_backend.py",
                    "patches/h3_tensor_flow.py",
                    "patches/tp_dit.py",
                    "patches/dit_column_linear.py",
                    "patches/tilelang_dit_fusion.py",
                    "patches/tilelang_transport_p2.py",
                    "patches/tilelang_row_quant.py",
                    "patches/tilelang_swiglu_quant.py",
                ):
                    # Historical V3 payloads must not be forced to track later
                    # production changes (including the P0-2/P0-3 promotion).
                    self.assertIn(name, files)
                policy = files["patches/h3_operator_policy.py"].decode("utf-8")
                self.assertIn("'sol'", policy)
                self.assertIn("dit_prepared_attention", policy)
                self.assertIn("os.execve(sys.executable,argv,env)", worker)
                self.assertIn("'--cpu-preflight'", worker)
                self.assertIn("source.joinpath('patches/h3_operator_policy.py')", worker)
                self.assertIn("for name in retired:", worker)
                self.assertIn("if target.exists():target.unlink()", worker)
                self.assertIn("ready_for_device_gate", code)

    def test_no_submitted_v2_source_was_overwritten(self):
        original = EVIDENCE.parent / "h3_dit_p0_lifetime_20261102_v2"
        self.assertTrue((original / "gpu_submission.json").is_file())
        self.assertTrue((original / "gpu_failure.json").is_file())
        self.assertNotEqual(
            sha((original / "gpu_job/experiment.ipynb").read_bytes()),
            sha((EVIDENCE / "gpu_job/experiment.ipynb").read_bytes()),
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
