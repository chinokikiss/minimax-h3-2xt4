"""Submit independent H3 KC1-3 / Sol4-8 / VideoVAE-KC E2E, seed 987654321.

No Kaggle status/log polling or auto retries.
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
PREV=ROOT/"docs/data/h3_kc_sol_8step_long_e2e_20261031"
BASE=ROOT/"docs/data/h3_kc_sol_seed987654321_long_e2e_20261031"
JOB=BASE/"gpu_job"
CONTRACT="benchmarks/turbo_challenge_contract.py"
def sha(x):return hashlib.sha256(x).hexdigest()

def main():
    from benchmarks.turbo_challenge_contract import validate as original_validate
    from patches.attention_schedule import parse_attention_schedule,select_attention_policy
    meta=json.loads((JOB/"kernel-metadata.json").read_text(encoding="utf-8"))
    bind=json.loads((JOB/"source_binding.json").read_text(encoding="utf-8"))
    plan=json.loads((BASE/"request.json").read_text(encoding="utf-8"))
    old=json.loads((PREV/"request.json").read_text(encoding="utf-8"))
    build=json.loads((BASE/"build_receipt.json").read_text(encoding="utf-8"))
    oldbind=json.loads((PREV/"gpu_job/source_binding.json").read_text(encoding="utf-8"))
    assert meta["id"]=="pamcks/h3-kc-sol-seed987654321-e2e"
    assert "id_no" not in meta
    assert meta["title"]=="H3 KC Sol Seed987654321 E2E"
    assert meta["code_file"]=="experiment.ipynb"
    assert meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert not meta["dataset_sources"] and not meta["kernel_sources"]
    assert sha((JOB/"experiment.ipynb").read_bytes())==build["notebook_sha256"]
    assert bind["gpu_count"]==2 and bind["full_pipeline_authorized"] and bind["zero_mounts"]
    assert not bind["status_polling"] and not bind["live_log_polling"]
    assert plan["seed"]==987654321 and old["seed"]==20261020
    assert set(plan)==set(old)
    assert all(plan[k]==old[k] for k in old if k not in ("seed","name","source_note"))
    assert original_validate({**plan,"seed":20261020})["output_megapixels"]==0.856064
    ops=plan["operators"]
    assert ops["video_attention"]=="kitchen" and ops["te_attention"]=="kitchen_causal"
    assert ops["dit_attention_schedule"]=="1-3:kitchen,4-8:sol@1.0"
    seq=[select_attention_policy(parse_attention_schedule(ops["dit_attention_schedule"]),n,
         ops["dit_attention"],ops["sol_tau"]) for n in range(1,9)]
    assert seq==[("kitchen",None)]*3+[("sol",1.0)]*5
    hashes=bind["overlay_hashes"]
    oldhashes=oldbind["overlay_hashes"]
    assert set(oldhashes)==set(hashes)
    assert [k for k in oldhashes if oldhashes[k]!=hashes[k]]==[CONTRACT]
    orig=(ROOT/CONTRACT).read_bytes()
    oldrow=b'SEEDS=frozenset((20261020,20261021))'
    newrow=b'SEEDS=frozenset((20261020,20261021,987654321))'
    assert orig.count(oldrow)==1
    assert hashes[CONTRACT]==sha(orig.replace(oldrow,newrow))
    for k,h in hashes.items():
        if k!=CONTRACT:assert sha((ROOT/k).read_bytes())==h,k
    restored=json.loads((ROOT/"docs/data/h3_sol_v2_active_source_identity.json").read_text())
    for k,rec in restored["paths"].items():
        assert hashes[k]==rec["restored_pre_math_sha256"],k
    destination=BASE/"gpu_submission.json"
    state=dict(status="submitting",kernel=meta["id"],requested_gpu_count=2,
        submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        seed=987654321,previous_seed=20261020,
        notebook_sha256=build["notebook_sha256"],
        source_binding_sha256=sha((JOB/"source_binding.json").read_bytes()),
        result_relative="h3_kc_sol_seed987654321_long_e2e/batch_results.json",
        final_collection_count=0,status_polling=False,live_log_polling=False)
    with destination.open("x",encoding="utf-8") as stream:json.dump(state,stream,indent=2)
    try:
        proc=subprocess.run([sys.executable,str(ROOT/"benchmarks/kaggle_tls12_runner.py"),
            "--cli","kernels","push","-p",str(JOB)],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        response=proc.stdout
        (BASE/"gpu_submission_cli.log").write_text(response,encoding="utf-8")
        version=re.search(r"Kernel version (\d+) successfully pushed",response)
        provider_error=bool(re.search(r"(?im)^\s*(?:Kernel push error:|Error:)",response))
        accepted=(proc.returncode==0 and version is not None and not provider_error)
        state.update(status="accepted" if accepted else "acceptance_uncertain",
            returncode=proc.returncode,version=int(version.group(1)) if version else None,
            response=response[-2500:])
        url=re.search(r"https://www\.kaggle\.com/(?:code/)?[\w-]+/[\w-]+",response)
        if url:state["url"]=url.group(0)
    except BaseException as exc:
        state.update(status="acceptance_uncertain",error=repr(exc))
        raise
    finally:
        destination.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in state.items() if k!="response"},indent=2),flush=True)
    if state["status"]!="accepted":raise RuntimeError("Provider acceptance uncertain; do not automatically retry")

if __name__=="__main__":main()
