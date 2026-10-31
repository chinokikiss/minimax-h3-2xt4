"""CPU-only preflight for independent all-Kitchen 0.85MP / 10s long E2E."""
import ast
import base64
import hashlib
import json
import unittest
from pathlib import Path

from benchmarks.turbo_challenge_contract import validate
from patches.attention_schedule import parse_attention_schedule,select_attention_policy

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"docs/data/h3_turbo_085mp_10s_20261029/gpu_v2_final/turbo_085mp_e2e/retry_request.json"
OLD=ROOT/"docs/data/h3_kc_sol_8step_long_e2e_20261031"
EVID=ROOT/"docs/data/h3_kc_all_8step_long_e2e_20261031"
JOB=EVID/"gpu_job"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class KitchenAllLongE2E(unittest.TestCase):
    def test_input_identity(self):
        old=json.loads(BASE.read_text(encoding="utf-8"))
        plan=json.loads((EVID/"request.json").read_text(encoding="utf-8"))
        validate(plan)
        exempt={"name","operators","source_note"}
        self.assertEqual({k:v for k,v in old.items() if k not in exempt},
                         {k:v for k,v in plan.items() if k not in exempt})
        self.assertEqual((plan["width"],plan["height"],plan["frames"],plan["steps"],plan["seed"]),
                         (1216,704,243,8,20261020))
        self.assertEqual(plan["turbo_lora"],old["turbo_lora"])
        self.assertEqual(plan["reference_images"],old["reference_images"])

    def test_just_one_attention_change_from_previous_hybrid(self):
        hybrid=json.loads((OLD/"request.json").read_text())
        dense=json.loads((EVID/"request.json").read_text())
        self.assertEqual({k:v for k,v in dense.items() if k not in {"name","source_note","operators"}},
                         {k:v for k,v in hybrid.items() if k not in {"name","source_note","operators"}})
        a=dict(dense["operators"])
        b=dict(hybrid["operators"])
        self.assertEqual(a.pop("dit_attention_schedule"),"")
        self.assertEqual(b.pop("dit_attention_schedule"),"1-3:kitchen,4-8:sol@1.0")
        self.assertEqual(a,b)

    def test_all_eight_nfe(self):
        o=json.loads((EVID/"request.json").read_text())["operators"]
        self.assertEqual(o["dit_attention"],"kitchen")
        self.assertEqual(o["dit_attention_schedule"],"")
        self.assertEqual(o["video_attention"],"kitchen")
        self.assertEqual(o["te_attention"],"kitchen_causal")
        self.assertEqual([select_attention_policy(parse_attention_schedule(o["dit_attention_schedule"]),step,
                         o["dit_attention"],o["sol_tau"]) for step in range(1,9)],
                         [("kitchen",None)]*8)

    def test_unchanged_frozen_source(self):
        a=json.loads((OLD/"gpu_job/source_binding.json").read_text())
        b=json.loads((JOB/"source_binding.json").read_text())
        self.assertEqual(a["overlay_hashes"],b["overlay_hashes"])
        self.assertEqual(a["overlay_archive_sha256"],b["overlay_archive_sha256"])
        self.assertEqual(a["baseline_result_sha256"],b["baseline_result_sha256"])
        self.assertEqual(a["baseline_request_sha256"],b["baseline_request_sha256"])
        self.assertGreater(len(b["overlay_hashes"]),50)
        for path,digest in b["overlay_hashes"].items():
            self.assertEqual(sha(ROOT/path),digest,path)
        self.assertFalse(b["status_polling"] or b["live_log_polling"])
        self.assertEqual(b["gpu_count"],2)

    def test_metadata_and_encoded_execution(self):
        meta=json.loads((JOB/"kernel-metadata.json").read_text())
        b=json.loads((JOB/"source_binding.json").read_text())
        receipt=json.loads((EVID/"build_receipt.json").read_text())
        self.assertNotIn("id_no",meta)
        self.assertEqual(meta["id"],"pamcks/h3-kc-all-e2e-20261031")
        self.assertEqual(meta["title"],"H3 KC All E2E 20261031")
        self.assertEqual(meta["code_file"],"experiment.ipynb")
        self.assertTrue(meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"])
        self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
        self.assertFalse(meta["dataset_sources"] or meta["kernel_sources"])
        nb=(JOB/"experiment.ipynb")
        self.assertLess(nb.stat().st_size,900000)
        self.assertEqual(sha(nb),receipt["notebook_sha256"])
        d=json.loads(nb.read_text())
        self.assertEqual(len(d["cells"]),1)
        code="".join(d["cells"][0]["source"])
        ast.parse(code)
        self.assertIn("h3_kc_all_8step_long_e2e/batch_results.json",code)
        decoded=[base64.b64decode(x.value).decode() for x in ast.walk(ast.parse(code))
            if isinstance(x,ast.Constant) and isinstance(x.value,str) and x.value.startswith("aW1w")]
        worker=next(x for x in decoded if "H3_SOL_WORKSPACE_POOL" in x)
        self.assertIn("'dit_attention_schedule': ''",worker)
        self.assertIn("'dit_attention': 'kitchen'",worker)
        self.assertIn("'video_attention': 'kitchen'",worker)
        self.assertIn("'te_attention': 'kitchen_causal'",worker)
        self.assertNotIn("'4-8:sol@1.0'",worker)
        self.assertIn("'/tmp/h3_kc_all_8step_long_20261031'",worker)
        self.assertIn("'H3_SOL_WORKSPACE_POOL']='0'",worker)
        self.assertTrue(b["full_pipeline_authorized"])

if __name__=="__main__":unittest.main(verbosity=2)
