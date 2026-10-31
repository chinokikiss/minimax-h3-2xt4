"""No-GPU acceptance tests for 0.85 MP 8-NFE Kitchen/Sol alternate-seed job.

This job changes only seed in runtime inputs; the one-source-line contract
allowlist extension exists *inside the notebook overlay*, not in the repo.
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
from benchmarks.turbo_challenge_contract import validate as pinned_validate
from patches.attention_schedule import parse_attention_schedule,select_attention_policy

ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT/"docs/data/h3_kc_sol_8step_long_e2e_20261031"
BASE=ROOT/"docs/data/h3_kc_sol_seed987654321_long_e2e_20261031"
JOB=BASE/"gpu_job"
PATH="benchmarks/turbo_challenge_contract.py"
def sha(b):return hashlib.sha256(b).hexdigest()

class AlternateSeedHybridPreflight(unittest.TestCase):
    def test_only_seed_and_descriptions_change(self):
        old=json.loads((PREV/"request.json").read_text())
        new=json.loads((BASE/"request.json").read_text())
        self.assertEqual(old["seed"],20261020)
        self.assertEqual(new["seed"],987654321)
        for field,value in old.items():
            if field not in ("seed","name","source_note"):
                self.assertEqual(new[field],value,field)
        self.assertEqual(set(new),set(old))
        self.assertEqual((new["width"],new["height"],new["frames"],new["steps"]), (1216,704,243,8))
        self.assertEqual(new["operators"]["dit_attention_schedule"],"1-3:kitchen,4-8:sol@1.0")
        self.assertEqual(new["operators"]["video_attention"],"kitchen")
        # Original repository contract must *still* reject the new seed.
        with self.assertRaisesRegex(ValueError,"Unauthorized Turbo challenge seed"):
            pinned_validate(new)
        pinned_validate({**new,"seed":20261020})

    def test_policy_exact_all_eight_nfe(self):
        plan=json.loads((BASE/"request.json").read_text())
        o=plan["operators"]
        scheduled=parse_attention_schedule(o["dit_attention_schedule"])
        self.assertEqual([select_attention_policy(scheduled,i,o["dit_attention"],o["sol_tau"]) for i in range(1,9)],
                         [("kitchen",None)]*3+[("sol",1.)]*5)
        self.assertEqual(o["te_attention"],"kitchen_causal")
        self.assertEqual(o["video_attention"],"kitchen")
        self.assertFalse(o["sol_collect_route_stats"])

    def test_source_closure_only_one_line_deviates_from_hybrid(self):
        previous=json.loads((PREV/"gpu_job/source_binding.json").read_text())
        binding=json.loads((JOB/"source_binding.json").read_text())
        a=previous["overlay_hashes"];b=binding["overlay_hashes"]
        self.assertEqual(set(a),set(b))
        self.assertEqual([name for name in sorted(a) if a[name]!=b[name]],[PATH])
        original=(ROOT/PATH).read_bytes()
        self.assertIn(b"SEEDS=frozenset((20261020,20261021))",original)
        replacement=original.replace(b"SEEDS=frozenset((20261020,20261021))",
                                     b"SEEDS=frozenset((20261020,20261021,987654321))")
        self.assertEqual(b[PATH],sha(replacement))
        self.assertEqual(a[PATH],sha(original))
        self.assertEqual(binding["isolated_seed_contract_override"]["original_sha256"],a[PATH])
        self.assertEqual(binding["isolated_seed_contract_override"]["overlay_sha256"],b[PATH])
        self.assertEqual(binding["overlay_baseline_commit"],previous["overlay_baseline_commit"])
        self.assertEqual(binding["baseline_request_sha256"],previous["baseline_request_sha256"])
        self.assertEqual(binding["baseline_result_sha256"],previous["baseline_result_sha256"])
        for name,digest in b.items():
            if name!=PATH:self.assertEqual(sha((ROOT/name).read_bytes()),digest,name)
        restore=json.loads((ROOT/"docs/data/h3_sol_v2_active_source_identity.json").read_text())
        for name,row in restore["paths"].items():
            self.assertEqual(b[name],row["restored_pre_math_sha256"],name)

    def test_notebook_source_artifact_contract(self):
        meta=json.loads((JOB/"kernel-metadata.json").read_text())
        binding=json.loads((JOB/"source_binding.json").read_text())
        receipt=json.loads((BASE/"build_receipt.json").read_text())
        self.assertEqual(meta["id"],"pamcks/h3-kc-sol-seed987654321-e2e")
        self.assertNotIn("id_no",meta)
        self.assertEqual(meta["title"],"H3 KC Sol Seed987654321 E2E")
        self.assertEqual(meta["code_file"],"experiment.ipynb")
        self.assertTrue(meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"])
        self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
        self.assertFalse(meta["dataset_sources"] or meta["kernel_sources"])
        self.assertEqual(binding["gpu_count"],2)
        self.assertTrue(binding["full_pipeline_authorized"])
        self.assertFalse(binding["status_polling"] or binding["live_log_polling"])
        n=JOB/"experiment.ipynb"
        self.assertLess(n.stat().st_size,900000)
        self.assertEqual(sha(n.read_bytes()),receipt["notebook_sha256"])
        code="".join(json.loads(n.read_text())["cells"][0]["source"])
        ast.parse(code)
        self.assertIn("h3_kc_sol_seed987654321_long_e2e/batch_results.json",code)
        tokens=[base64.b64decode(v.value).decode() for v in ast.walk(ast.parse(code))
                if isinstance(v,ast.Constant) and isinstance(v.value,str) and v.value.startswith("aW1w")]
        worker=next(s for s in tokens if "H3_SOL_WORKSPACE_POOL" in s)
        self.assertIn("987654321",worker)
        self.assertIn("1-3:kitchen,4-8:sol@1.0",worker)
        self.assertIn("'/tmp/h3_kc_sol_seed987654321_long_20261031'",worker)
        self.assertIn("['H3_SOL_WORKSPACE_POOL']='0'",worker)
        self.assertIn("['H3_SOL_WORKSPACE_POOL']='0'",worker)
        self.assertIn("['H3_SOL_LUT_BITMAP']='0'",worker)

    def test_overlay_archive_actual_contract(self):
        n=json.loads((JOB/"experiment.ipynb").read_text())
        code="".join(n["cells"][0]["source"])
        tokens=[base64.b64decode(v.value).decode() for v in ast.walk(ast.parse(code))
                if isinstance(v,ast.Constant) and isinstance(v.value,str) and v.value.startswith("aW1w")]
        worker=next(s for s in tokens if "H3_SOL_WORKSPACE_POOL" in s)
        source=ast.parse(worker)
        # The builder substitutes the packed b85 literal into a fixed assignment.
        assigned=[a for a in ast.walk(source)
                  if isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="archive_raw" for t in a.targets)]
        self.assertEqual(len(assigned),1)
        expr=assigned[0].value
        self.assertIsInstance(expr,ast.Call)
        self.assertIsInstance(expr.args[0],ast.Constant)
        blob=base64.b85decode(expr.args[0].value)
        binding=json.loads((JOB/"source_binding.json").read_text())
        self.assertEqual(sha(blob),binding["overlay_archive_sha256"])
        with tarfile.open(fileobj=io.BytesIO(lzma.decompress(blob)),mode="r:") as tar:
            f=tar.extractfile(PATH)
            self.assertIsNotNone(f)
            payload=f.read()
        self.assertIn(b"987654321",payload)
        self.assertEqual(sha(payload),binding["overlay_hashes"][PATH])
        self.assertEqual((ROOT/PATH).read_bytes().count(b"987654321"),0)

if __name__=="__main__":unittest.main(verbosity=2)
