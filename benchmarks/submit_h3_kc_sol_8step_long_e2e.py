"""One-shot source-locked 2xT4 long E2E push (NFE1-3 KC, NFE4-8 Sol, VAE KC)."""
from __future__ import annotations
import datetime as dt,hashlib,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/"docs/data/h3_kc_sol_8step_long_e2e_20261031"
JOB=BASE/"gpu_job"
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    from benchmarks.turbo_challenge_contract import validate
    from patches.attention_schedule import parse_attention_schedule,select_attention_policy
    meta=json.loads((JOB/"kernel-metadata.json").read_text())
    binding=json.loads((JOB/"source_binding.json").read_text())
    plan=json.loads((BASE/"request.json").read_text())
    build=json.loads((BASE/"build_receipt.json").read_text())
    assert meta["id"]=="pamcks/h3-kc-sol-8step-long-e2e-20261031"
    assert meta["code_file"]=="experiment.ipynb" and meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert not meta["dataset_sources"] and not meta["kernel_sources"]
    assert sha((JOB/"experiment.ipynb").read_bytes())==build["notebook_sha256"]
    assert binding["gpu_count"]==2 and binding["full_pipeline_authorized"] and binding["zero_mounts"]
    assert not binding["status_polling"] and not binding["live_log_polling"]
    assert plan["operators"]["video_attention"]=="kitchen"
    assert plan["operators"]["te_attention"]=="kitchen_causal"
    assert plan["operators"]["sol_tau"]==1.0
    validate(plan)
    scheduled=parse_attention_schedule(plan["operators"]["dit_attention_schedule"])
    got=[select_attention_policy(scheduled,n,plan["operators"]["dit_attention"],plan["operators"]["sol_tau"]) for n in range(1,9)]
    assert got==[("kitchen",None)]*3+[("sol",1.)]*5
    restored=json.loads((ROOT/"docs/data/h3_sol_v2_active_source_identity.json").read_text())
    for path,rec in restored["paths"].items():
        assert binding["overlay_hashes"][path]==rec["restored_pre_math_sha256"],path
        assert sha((ROOT/path).read_bytes())==rec["restored_pre_math_sha256"],path
    receipt=BASE/"gpu_submission.json"
    state=dict(status="submitting",kernel=meta["id"],requested_gpu_count=2,
         submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         notebook_sha256=build["notebook_sha256"],
         source_binding_sha256=sha((JOB/"source_binding.json").read_bytes()),
         result_relative="h3_kc_sol_8step_long_e2e/batch_results.json",
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
            response=response[-2000:])
        u=re.search(r"https://www\.kaggle\.com/(?:code/)?[\w-]+/[\w-]+",response)
        if u:state["url"]=u.group(0)
    except BaseException as e:
        state.update(status="acceptance_uncertain",error=repr(e))
        raise
    finally:
        receipt.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in state.items() if k!="response"},indent=2),flush=True)
    if state["status"]!="accepted":raise RuntimeError("Uncertain Kaggle provider acceptance: do not auto-resubmit")
if __name__=="__main__":main()
