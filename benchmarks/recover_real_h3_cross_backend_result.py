"""Offline recovery of a valid Real-H3 capture from a failed Kaggle summary stage.

Never changes Kaggle v1's original failure record or invents quality metrics.
Requires both rank payloads and hashes and a complete SM120 replay.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVID=ROOT/"docs/data/h3_real_sol_comfy_capture_20261031"
META=EVID/"failed_v1_artifacts/real_h3_sol_comfy_capture"
PAYLOADS=EVID/"failed_v1_payloads/real_h3_sol_comfy_capture/generation/rank_results"
REPLAY=EVID/"local_comfy_replay/result.json"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    source=json.loads((META/"generation/result.json").read_text(encoding="utf-8"))
    assert source["status"]=="failed"
    assert source["error"]=="KeyError: 'summary'"
    assert "old_rel_l2" in source["traceback"]
    replay=json.loads(REPLAY.read_text(encoding="utf-8"))
    assert replay["status"]=="complete"
    assert sorted(replay["ranks"])==["0","1"]
    ranks=[]
    for rank in (0,1):
        name=f"rank{rank}.json"
        row=json.loads((META/"generation/rank_results"/name).read_text())
        local=json.loads((PAYLOADS/name).read_text())
        assert local==row and row["status"]=="complete" and row["nfe"]==1
        payload=PAYLOADS/row["payload_file"]
        assert payload.stat().st_size==row["payload_bytes"]
        assert sha(payload)==row["payload_sha256"]
        assert replay["ranks"][str(rank)]["status"]=="complete"
        assert replay["ranks"][str(rank)]["source_tensor_sha256"]==row["tensor_sha256"]
        assert all(x["status"]=="complete" for x in replay["ranks"][str(rank)]["cases"].values())
        ranks.append({"rank":rank,"input_shape":row["dims"],"file":row["payload_file"],
            "sha256":row["payload_sha256"],"bytes":row["payload_bytes"],
            "t4_sol_output_sha256":row["tensor_sha256"]["sol_v2"]})
    recovery=dict(status="recovered_complete",
        scope="actual T4 QKV capture + restored Sol-v2 + SM120 ComfyUI Sol replay, not full E2E",
        original_kaggle_status="failed",
        original_failure="Post-capture legacy summary assembler indexed rank['summary'] despite replacement capture schema",
        reason_gpu_rerun_unnecessary="both QKV/sol outputs were produced before summary exception and verified",
        physical_t4_capture=ranks,
        local_replay_status=replay["status"],
        local_replay_environment=replay["replay_env"],
        original_result_sha256=sha(META/"generation/result.json"),
        local_replay_sha256=sha(REPLAY),
        source_result=str((META/"generation/result.json").relative_to(ROOT)),
        local_replay=str(REPLAY.relative_to(ROOT)),
        quality_limitations="One real first-DiT attention head per rank; low-level Sol not chunked H3 producer; cannot infer full video semantic consistency.")
    target=EVID/"recovered_v1_result.json"
    target.write_text(json.dumps(recovery,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":recovery["status"],"payload_count":len(ranks),
        "sha256_valid":True,"local_replay_complete":True,"output":str(target.relative_to(ROOT))},indent=2))
if __name__=="__main__":main()
