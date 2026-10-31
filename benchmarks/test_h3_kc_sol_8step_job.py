"""No-GPU preflight for 0.85MP long E2E KC1-3/Sol4-8/Video-KC job."""
import ast,base64,hashlib,json,pathlib,unittest
from patches.attention_schedule import parse_attention_schedule,select_attention_policy
from benchmarks.turbo_challenge_contract import validate
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT/"docs/data/h3_turbo_085mp_10s_20261029/gpu_v2_final/turbo_085mp_e2e/retry_request.json"
EVID=ROOT/"docs/data/h3_kc_sol_8step_long_e2e_20261031"
JOB=EVID/"gpu_job"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

class HybridQualification(unittest.TestCase):
    def test_source_provenance(self):
        bind=json.loads((JOB/"source_binding.json").read_text())
        restore=json.loads((ROOT/"docs/data/h3_sol_v2_active_source_identity.json").read_text())
        hashes=bind["overlay_hashes"]
        self.assertGreaterEqual(len(hashes),50)
        for name,digest in hashes.items():
            self.assertEqual(sha(ROOT/name),digest,name)
        for name,row in restore["paths"].items():
            self.assertEqual(hashes[name],row["restored_pre_math_sha256"],name)
        self.assertEqual(bind["overlay_baseline_commit"],bind["baseline_source_binding"]["publication"]["commit"])
    def test_input_identity(self):
        source=json.loads(BASE.read_text())
        plan=json.loads((EVID/"request.json").read_text())
        validate(plan)
        self.assertEqual({k:v for k,v in plan.items() if k not in ("operators","source_note","name")},
                         {k:v for k,v in source.items() if k not in ("operators","source_note","name")})
        self.assertEqual((plan["width"],plan["height"],plan["frames"],plan["steps"],plan["seed"]),
            (1216,704,243,8,20261020))
        self.assertEqual(plan["operators"]["video_attention"],"kitchen")
        self.assertEqual(plan["operators"]["te_attention"],"kitchen_causal")
        self.assertEqual(plan["operators"]["sol_tau"],1.)
        self.assertTrue(plan["operators"]["sol_topology_v2"])
        self.assertTrue(plan["operators"]["sol_sync_free_lut"])
    def test_nfe_dispatch(self):
        op=json.loads((EVID/"request.json").read_text())["operators"]
        parsed=parse_attention_schedule(op["dit_attention_schedule"])
        got=[select_attention_policy(parsed,n,op["dit_attention"],op["sol_tau"]) for n in range(1,9)]
        self.assertEqual(got,[("kitchen",None)]*3+[("sol",1.)]*5)
        self.assertEqual(len(got),8)
        self.assertEqual(sum(mode=="kitchen" for mode,_ in got)*50,150)
        self.assertEqual(sum(mode=="sol" for mode,_ in got)*50,250)
    def test_submission_notebook(self):
        bind=json.loads((JOB/"source_binding.json").read_text())
        meta=json.loads((JOB/"kernel-metadata.json").read_text())
        receipt=json.loads((EVID/"build_receipt.json").read_text())
        self.assertEqual(meta["id"],"pamcks/h3-kc-sol-8step-long-e2e-20261031")
        self.assertEqual(meta["code_file"],"experiment.ipynb")
        self.assertTrue(meta["is_private"] and meta["enable_gpu"])
        # This full E2E downloads source/weights/reference images, unlike zero-weight gates.
        self.assertTrue(meta["enable_internet"])
        self.assertEqual(meta["dataset_sources"],[])
        self.assertEqual(meta["kernel_sources"],[])
        notebook=JOB/"experiment.ipynb"
        self.assertLess(notebook.stat().st_size,900000)
        self.assertEqual(sha(notebook),receipt["notebook_sha256"])
        nb=json.loads(notebook.read_text())
        self.assertEqual(len(nb["cells"]),1)
        code="".join(nb["cells"][0]["source"])
        ast.parse(code)
        self.assertIn("h3_kc_sol_8step_long_e2e/batch_results.json",code)
        # The watchdog stores its worker as a base64 literal in the notebook.
        candidates=[base64.b64decode(x.value).decode() for x in ast.walk(ast.parse(code))
             if isinstance(x,ast.Constant) and isinstance(x.value,str) and x.value.startswith('aW1w')]
        worker=next(x for x in candidates if 'H3_SOL_WORKSPACE_POOL' in x)
        self.assertNotIn("dit_attention_schedule='2-7:sol@1.0'",worker)
        self.assertIn("4-8:sol@1.0",worker)
        self.assertIn("video_attention': 'kitchen'",worker)
        self.assertFalse(bind["status_polling"] or bind["live_log_polling"])
        self.assertTrue(bind["full_pipeline_authorized"])
        self.assertEqual(bind["gpu_count"],2)

if __name__=="__main__":unittest.main(verbosity=2)
