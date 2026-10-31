"""Offline source/payload contract for real H3 Sol cross-backend capture job."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"docs/data/h3_real_sol_comfy_capture_20261031"
JOB=DATA/"gpu_job"

def sha(x):return hashlib.sha256(Path(x).read_bytes()).hexdigest()

class RealH3CrossBackendCaptureContract(unittest.TestCase):
    def test_real_source_interception(self):
        code=(ROOT/"benchmarks/real_h3_cross_backend_capture.py").read_text()
        ast.parse(code)
        self.assertIn("original.microgate=capture",code)
        self.assertIn("qs,ks,vs=[x[:,0:1].contiguous()",code)
        self.assertIn("65166,128",code)
        self.assertIn("torch.save(cpu,saved)",code)
        self.assertIn("sol.attention(qs,ks,vs,",code)
        self.assertNotIn("torch.randn(",code)
        self.assertIn("layout=options.get(\"minimax_h3_layout\")",code)
        runner=(ROOT/"benchmarks/run_real_h3_cross_backend_batch.py").read_text()
        self.assertIn("benchmarks/real_h3_cross_backend_capture.py",runner)
        self.assertNotIn("benchmarks/real_h3_attention_microgate.py",runner)
    def test_unchanged_real_h3_workload(self):
        base=json.loads((ROOT/"docs/data/h3_turbo_085mp_10s_20261029/gpu_v2_final/turbo_085mp_e2e/retry_request.json").read_text())
        plan=json.loads((DATA/"request.json").read_text())
        for field in ["width","height","frames","fps","steps","seed","sampler","scheduler",
                      "cfg","video_shift","audio_shift","prompt_sha256","reference_images","turbo_lora"]:
            self.assertEqual(plan[field],base[field],field)
        self.assertEqual((plan["width"],plan["height"],plan["frames"],plan["steps"]), (1216,704,243,8))
    def test_binding_and_old_source(self):
        b=json.loads((JOB/"source_binding.json").read_text())
        self.assertEqual(b["gpu_count"],2)
        self.assertFalse(b["status_polling"] or b["live_log_polling"] or b["full_pipeline_authorized"])
        hashes=b["overlay_hashes"]
        self.assertIn("benchmarks/real_h3_cross_backend_capture.py",hashes)
        self.assertIn("benchmarks/run_real_h3_cross_backend_batch.py",hashes)
        for name,digest in hashes.items():
            self.assertEqual(sha(ROOT/name),digest,name)
        restore=json.loads((ROOT/"docs/data/h3_sol_v2_active_source_identity.json").read_text())
        for path,row in restore["paths"].items():
            self.assertEqual(hashes[path],row["restored_pre_math_sha256"])
    def test_one_new_notebook_and_encoded_worker(self):
        meta=json.loads((JOB/"kernel-metadata.json").read_text())
        receipt=json.loads((DATA/"build_receipt.json").read_text())
        self.assertEqual(meta["id"],"pamcks/h3-real-sol-comfy-capture-20261031")
        self.assertEqual(meta["title"],"H3 Real Sol Comfy Capture 20261031")
        self.assertNotIn("id_no",meta)
        self.assertEqual(meta["code_file"],"experiment.ipynb")
        self.assertTrue(meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"])
        self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(meta["dataset_sources"],[])
        self.assertEqual(meta["kernel_sources"],[])
        notebook=JOB/"experiment.ipynb"
        self.assertLess(notebook.stat().st_size,900000)
        self.assertEqual(sha(notebook),receipt["notebook_sha256"])
        n=json.loads(notebook.read_text())
        self.assertEqual(len(n["cells"]),1)
        code="".join(n["cells"][0]["source"])
        ast.parse(code)
        self.assertIn("real_h3_sol_comfy_capture/batch_results.json",code)
        embedded=[base64.b64decode(n.value).decode()
                for n in ast.walk(ast.parse(code))
                if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith("aW1w")]
        worker=next(s for s in embedded if "H3_SOL_WORKSPACE_POOL" in s)
        self.assertIn("benchmarks/run_real_h3_cross_backend_batch.py",worker)
        self.assertIn("real_h3_sol_comfy_capture",worker)
        self.assertIn("'/tmp/h3_real_sol_comfy_capture_20261031'",worker)
        self.assertIn("H3_SOL_WORKSPACE_POOL",worker)
    def test_local_replay_requirements(self):
        x=ROOT/"benchmarks/replay_real_h3_comfy_sol.py"
        code=x.read_text()
        ast.parse(code)
        self.assertIn("ck.sol_attn(q,k,v,",code)
        self.assertIn("ck.sol_attn_is_available",code)
        self.assertIn("payload_sha256",code)
        self.assertIn("matched_dense_prefix_tau1_extra0",code)
        self.assertIn("comfy_h3_default_sinks_tau1_extra256",code)
        self.assertIn("comfy_ui_default_tau1p3_extra256",code)
if __name__=="__main__":
    unittest.main(verbosity=2)
