"""Build an immutable v2 retry of the 10s H3 Kaggle E2E notebook.

The only behavior change from v1 is staging the exact shrine reference from
the previously published, commit-pinned public image mirror before the batch
runner starts. This script never submits or polls Kaggle.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/data/h3_t4_fp16_512_sol_all_8nfe_10s_20261101"
V1 = EVIDENCE / "gpu_job"
V2 = EVIDENCE / "gpu_job_v2"
IMAGE = ROOT / "bench_assets/public_reference_images/cd4cbfd7137b80e85e64ada86ffd93e512b874de84cb4291fc0278b747461e03.png"
EXPECTED_SHA = "cd4cbfd7137b80e85e64ada86ffd93e512b874de84cb4291fc0278b747461e03"
EXPECTED_BYTES = 7703256
MIRROR = (
    "https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/"
    "225da31d025d510bb152683f53cf6f0a9843e0d2/"
    "bench_assets/public_reference_images/" + EXPECTED_SHA + ".png"
)

# This is inserted only into the one-shot Kaggle worker, after the verified
# source overlay and before run_h3_l1_turbo_batch.py. The frozen image identity
# and production code remain untouched.
REFERENCE_STAGE = r"""
# Benchmark-only reference delivery. Do not alter l1_assets_identity.json.
import requests,time
_rows=json.loads((source/'l1_assets_identity.json').read_text())['images']
_matches=[r for r in _rows if r['file']=='reference_shrine.png']
assert len(_matches)==1
_ref=_matches[0]
assert (_ref['sha256'],_ref['bytes'],_ref['size'])==(SHRINE_SHA,SHRINE_BYTES,[4096,2048])
_target=source/'bench_assets/l1_current/reference_shrine.png'
_target.parent.mkdir(parents=True,exist_ok=True)
if not _target.is_file():
    for _attempt in range(3):
        try:
            _response=requests.get(SHRINE_MIRROR,
                headers={'User-Agent':'H3-REF2VA-evaluation/1.0'},
                timeout=(20,90))
            _response.raise_for_status()
            _data=_response.content
            break
        except requests.RequestException as _error:
            _status=getattr(getattr(_error,'response',None),'status_code',None)
            if _attempt==2 or (_status is not None and _status not in (408,429,500,502,503,504)):
                raise
            _delay=(5,15)[_attempt]
            print('H3_REFERENCE_MIRROR_RETRY',_attempt+1,_status,_delay,flush=True)
            time.sleep(_delay)
    if len(_data)!=SHRINE_BYTES or hashlib.sha256(_data).hexdigest()!=SHRINE_SHA:
        raise ValueError('Immutable shrine mirror has different bytes')
    _target.write_bytes(_data)
if _target.stat().st_size!=SHRINE_BYTES or hashlib.sha256(_target.read_bytes()).hexdigest()!=SHRINE_SHA:
    raise ValueError('Staged shrine reference has wrong bytes')
(out/'reference_delivery_v2.json').write_text(json.dumps(dict(
    mode='commit_pinned_github_mirror',local_source_sha256=SHRINE_SHA,
    source_url=SHRINE_MIRROR,bytes=SHRINE_BYTES,verified=True),indent=2))
print('H3_REFERENCE_STAGED',SHRINE_SHA,SHRINE_BYTES,flush=True)
""".replace("SHRINE_SHA", repr(EXPECTED_SHA)).replace(
    "SHRINE_BYTES", repr(EXPECTED_BYTES)).replace("SHRINE_MIRROR", repr(MIRROR))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if V2.exists():
        raise FileExistsError("Preserve previously prepared retry, refusing overwrite: " + str(V2))
    local = IMAGE.read_bytes()
    assert len(local) == EXPECTED_BYTES and sha(local) == EXPECTED_SHA
    original_receipt = json.loads((EVIDENCE / "gpu_submission.json").read_text(encoding="utf-8"))
    assert original_receipt["version"] == 1 and original_receipt["status"] == "accepted"
    original = (V1 / "experiment.ipynb").read_bytes()
    assert sha(original) == original_receipt["notebook_sha256"]
    notebook = json.loads(original)
    assert len(notebook["cells"]) == 1
    code = "".join(notebook["cells"][0]["source"])

    # Preserve the setup, watchdog and final-output assertion byte for byte.
    anchor = "worker.write_bytes(base64.b64decode("
    assert code.count(anchor) == 2  # frozen setup supervisor + E2E supervisor
    p = code.rfind(anchor) + len(anchor)
    q = code.index("))", p)
    payload = ast.literal_eval(code[p:q])
    worker = base64.b64decode(payload, validate=True).decode("utf-8")
    needle = "(out/'request.json').write_text("
    assert worker.count(needle) == 1
    assert worker.count("os.execve(sys.executable,argv,env)") == 1
    assert "H3_REFERENCE_STAGED" not in worker
    new_worker = worker.replace(needle, REFERENCE_STAGE + needle, 1)
    ast.parse(new_worker)
    new_payload = base64.b64encode(new_worker.encode("utf-8")).decode("ascii")
    new_code = code[:p] + repr(new_payload) + code[q:]
    ast.parse(new_code)
    assert new_code.count("T4_SUPERVISOR notebook_cell_started") == 2
    assert new_code.count("['status']=='complete'") == 1
    notebook["cells"][0]["source"] = new_code.splitlines(keepends=True)
    output = json.dumps(notebook).encode("utf-8")
    assert len(output) < 900_000, "Notebook exceeds Kaggle notebook envelope"

    meta = json.loads((V1 / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == original_receipt["kernel"] and meta["enable_gpu"]
    assert meta["is_private"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert not meta["dataset_sources"] and not meta["kernel_sources"]

    old_binding = json.loads((V1 / "source_binding.json").read_text(encoding="utf-8"))
    assert old_binding["benchmark_debug"] is False and old_binding["runtime_debug"] is False
    assert old_binding["plan"]["steps"] == 8
    assert old_binding["plan"]["residual_exponent"] == 9
    assert old_binding["plan"]["operators"]["dit_attention_schedule"] == "1-8:sol@1.0"
    assert old_binding["attention_selection"] == [["sol",1.0]]*8
    binding = dict(old_binding)
    binding["retry_of_version"] = 1
    binding["retry_reason"] = "Wikimedia Commons HTTP 429 before model downloads"
    binding["reference_delivery_v2"] = dict(
        mode="benchmark_only_pre_stage",
        origin_local=str(IMAGE.relative_to(ROOT)).replace("\\", "/"),
        published_mirror=MIRROR,
        bytes=EXPECTED_BYTES,
        sha256=EXPECTED_SHA,
        original_identity_preserved=True,
        inference_source_overlay_unchanged=True,
        original_notebook_sha256=sha(original),
        original_worker_sha256=sha(worker.encode("utf-8")),
        retry_worker_sha256=sha(new_worker.encode("utf-8")),
    )
    receipt = dict(status="ready", version=2, kernel=meta["id"],
        notebook_bytes=len(output), notebook_sha256=sha(output),
        v1_notebook_sha256=sha(original),
        worker_sha256=sha(new_worker.encode("utf-8")),
        source_overlay_sha256=old_binding["overlay_sha256"],
        changed_scope="reference delivery only", model_operator_changes=0,
        debug=False, residual_exponent=9, nfe=8,
        shrine_sha256=EXPECTED_SHA, shrine_bytes=EXPECTED_BYTES,
        shrine_mirror=MIRROR, status_polling=False, live_log_polling=False,
        submission_count=0)
    V2.mkdir(parents=True, exist_ok=False)
    (V2 / "experiment.ipynb").write_bytes(output)
    (V2 / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (V2 / "source_binding.json").write_text(json.dumps(binding, indent=2), encoding="utf-8")
    (EVIDENCE / "build_receipt_v2.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
