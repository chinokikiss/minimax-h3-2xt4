"""One-shot submission of an already source-locked Sol bitmap T4 microgate.

No status polling, kernel cancellation, live logging, or resubmission.
Persist an exclusive receipt before contacting Kaggle to prevent duplicates
when the provider response is ambiguous.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence",required=True,type=Path)
    args=parser.parse_args()
    evidence=args.evidence.resolve()
    job=evidence/"gpu_job"
    binding=json.loads((job/"source_binding.json").read_text(encoding="utf-8"))
    metadata=json.loads((job/"kernel-metadata.json").read_text(encoding="utf-8"))
    assert binding.get("bitmap_ab") is True
    assert binding.get("full_pipeline") is False
    assert binding.get("internet") is False
    assert metadata["id"]=="pamcks/h3-sol-bitmap-ab-20261031"
    assert metadata["is_private"] and metadata["enable_gpu"]
    assert metadata["dataset_sources"]==metadata["kernel_sources"]==[]
    nb=job/"experiment.ipynb"
    assert hashlib.sha256(nb.read_bytes()).hexdigest()==binding["notebook_sha256"]
    root=Path(__file__).resolve().parents[1]
    mismatches=[
        name for name,h in binding["source_hashes"].items()
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h
    ]
    if mismatches: raise RuntimeError(f"Frozen source diverged: {mismatches}")
    receipt=evidence/"gpu_submission.json"
    state=dict(kernel=metadata["id"],status="submitting",
               submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               source_binding_sha256=hashlib.sha256(
                   (job/"source_binding.json").read_bytes()).hexdigest(),
               notebook_sha256=binding["notebook_sha256"],
               status_polling=False,live_log_polling=False,final_collection_count=0)
    with receipt.open("x",encoding="utf-8") as f:
        json.dump(state,f,indent=2)
    command=[sys.executable,str(root/"benchmarks/kaggle_tls12_runner.py"),
             "--cli","kernels","push","-p",str(job)]
    try:
        proc=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=180)
        output=proc.stdout+proc.stderr
        (evidence/"gpu_submission_cli.log").write_text(output,encoding="utf-8")
        version=re.search(r"Kernel version (\d+)",output)
        provider_error=bool(re.search(r"(?im)^\s*(?:Kernel push error:|Error:)",output))
        accepted=proc.returncode==0 and version is not None and not provider_error
        state.update(status="accepted" if accepted else "acceptance_uncertain",
                     returncode=proc.returncode,provider_error=provider_error,
                     response=output[-4000:])
        if version:state["version"]=int(version.group(1))
        uri=re.search(r"https://www\.kaggle\.com/code/[\w-]+/[\w-]+",output)
        if uri:state["url"]=uri.group(0)
    except BaseException as e:
        state.update(status="acceptance_uncertain",exception=repr(e))
        raise
    finally:
        receipt.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in state.items() if k not in
                      ("response","exception")},indent=2))
    if state["status"]!="accepted":
        raise RuntimeError("Provider acceptance unconfirmed: do not retry automatically")


if __name__=="__main__":
    main()
