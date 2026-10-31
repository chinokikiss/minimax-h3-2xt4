"""One-shot Kaggle push of real-H3 post-RoPE QKV/Sol-v2 capture for Comfy replay.

Two Tesla T4 GPUs are requested; no status polling, no implicit retry.
"""
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
BASE=ROOT/"docs/data/h3_real_sol_comfy_capture_20261031"
JOB=BASE/"gpu_job"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    meta=json.loads((JOB/"kernel-metadata.json").read_text())
    bind=json.loads((JOB/"source_binding.json").read_text())
    receipt=json.loads((BASE/"build_receipt.json").read_text())
    plan=json.loads((BASE/"request.json").read_text())
    assert meta["id"]=="pamcks/h3-real-sol-comfy-capture-20261031"
    assert "id_no" not in meta
    assert meta["code_file"]=="experiment.ipynb"
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert not meta["dataset_sources"] and not meta["kernel_sources"]
    assert bind["gpu_count"]==2 and not bind["full_pipeline_authorized"]
    assert not bind["status_polling"] and not bind["live_log_polling"]
    assert sha(JOB/"experiment.ipynb")==receipt["notebook_sha256"]
    assert plan["seed"]==20261020 and (plan["width"],plan["height"],plan["frames"],plan["steps"])==(1216,704,243,8)
    assert plan["operators"]["sol_tau"]==1.
    assert "benchmarks/real_h3_cross_backend_capture.py" in bind["overlay_hashes"]
    for name,expected in bind["overlay_hashes"].items():
        assert sha(ROOT/name)==expected,name
    prev=json.loads((ROOT/"docs/data/h3_sol_v2_active_source_identity.json").read_text())
    for name,row in prev["paths"].items():
        assert bind["overlay_hashes"][name]==row["restored_pre_math_sha256"]
    destination=BASE/"gpu_submission.json"
    state=dict(kernel=meta["id"],status="submitting",
       submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
       gpu_count_requested=2,
       notebook_sha256=receipt["notebook_sha256"],
       source_binding_sha256=sha(JOB/"source_binding.json"),
       output_relative="real_h3_sol_comfy_capture/generation/rank_results",
       status_polling=False,live_log_polling=False,final_collection_count=0)
    with destination.open("x",encoding="utf-8") as stream:
        json.dump(state,stream,indent=2)
    try:
        proc=subprocess.run([sys.executable,str(ROOT/"benchmarks/kaggle_tls12_runner.py"),
             "--cli","kernels","push","-p",str(JOB)],cwd=ROOT,
             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
             text=True,timeout=180)
        output=proc.stdout
        (BASE/"gpu_submission_cli.log").write_text(output)
        match=re.search(r"Kernel version (\d+) successfully pushed",output)
        state.update(status="accepted" if proc.returncode==0 and match else "acceptance_uncertain",
             returncode=proc.returncode,version=int(match.group(1)) if match else None,
             response=output[-2500:])
        u=re.search(r"https://www\.kaggle\.com/(?:code/)?[\w-]+/[\w-]+",output)
        if u:state["url"]=u.group(0)
    except BaseException as exc:
        state.update(status="acceptance_uncertain",error=repr(exc))
        raise
    finally:
        destination.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in state.items() if k!="response"},indent=2),flush=True)
    if state["status"]!="accepted":
        raise RuntimeError("Kaggle acceptance uncertain; do not resubmit automatically")
if __name__=="__main__":
    main()
