"""Submit one independent 2xT4 all-Kitchen E2E; no GPU status polling."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/"docs/data/h3_kc_all_8step_long_e2e_20261031"
JOB=BASE/"gpu_job"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    from benchmarks.turbo_challenge_contract import validate
    from patches.attention_schedule import parse_attention_schedule,select_attention_policy
    meta=json.loads((JOB/"kernel-metadata.json").read_text())
    binding=json.loads((JOB/"source_binding.json").read_text())
    plan=json.loads((BASE/"request.json").read_text())
    build=json.loads((BASE/"build_receipt.json").read_text())
    assert meta["id"]=="pamcks/h3-kc-all-e2e-20261031"
    assert meta["title"]=="H3 KC All E2E 20261031"
    assert "id_no" not in meta
    assert meta["code_file"]=="experiment.ipynb"
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert not meta["dataset_sources"] and not meta["kernel_sources"]
    assert sha(JOB/"experiment.ipynb")==build["notebook_sha256"]
    assert binding["full_pipeline_authorized"] and binding["gpu_count"]==2 and binding["zero_mounts"]
    assert not binding["status_polling"] and not binding["live_log_polling"]
    assert validate(plan)["output_megapixels"]==0.856064
    o=plan["operators"]
    assert o["video_attention"]==o["dit_attention"]=="kitchen"
    assert o["te_attention"]=="kitchen_causal" and o["dit_attention_schedule"]==""
    expected=[("kitchen",None)]*8
    actual=[select_attention_policy(parse_attention_schedule(o["dit_attention_schedule"]),step,
                                    o["dit_attention"],o["sol_tau"]) for step in range(1,9)]
    assert actual==expected
    receipt=BASE/"gpu_submission.json"
    state=dict(status="submitting",kernel=meta["id"],requested_gpu_count=2,
        submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        notebook_sha256=build["notebook_sha256"],
        source_binding_sha256=sha(JOB/"source_binding.json"),
        result_relative="h3_kc_all_8step_long_e2e/batch_results.json",
        final_collection_count=0,status_polling=False,live_log_polling=False)
    with receipt.open("x",encoding="utf-8") as stream:
        json.dump(state,stream,indent=2)
    try:
        proc=subprocess.run([sys.executable,str(ROOT/"benchmarks/kaggle_tls12_runner.py"),
            "--cli","kernels","push","-p",str(JOB)],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        response=proc.stdout
        (BASE/"gpu_submission_cli.log").write_text(response,encoding="utf-8")
        m=re.search(r"Kernel version (\d+) successfully pushed",response)
        state.update(status="accepted" if proc.returncode==0 and m else "acceptance_uncertain",
            returncode=proc.returncode,version=int(m.group(1)) if m else None,
            response=response[-2500:])
        u=re.search(r"https://www\.kaggle\.com/(?:code/)?[\w-]+/[\w-]+",response)
        if u:state["url"]=u.group(0)
    except BaseException as exc:
        state.update(status="acceptance_uncertain",error=repr(exc))
        raise
    finally:
        receipt.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in state.items() if k!="response"},indent=2),flush=True)
    if state["status"]!="accepted":
        raise RuntimeError("Kaggle submission outcome uncertain; no auto-retry")
if __name__=="__main__":main()
