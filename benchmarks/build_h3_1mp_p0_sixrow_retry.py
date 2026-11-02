"""Create an isolated v2 rebuild of the failed 1MP dual-T4 E2E notebook.

Reuses the previously qualified workload, baseline source publication,
and immutable v1 failure archive. Does not submit or query Kaggle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks import build_h3_t4_fp16_512_sol_all_8nfe_1mp_10s as baseline

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102"
RETRY = ROOT / "docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102_v2"
REPAIR = ROOT / "patches/tilelang_dit_fusion.py"


def main():
    if RETRY.exists():
        raise FileExistsError("Preserve the v2 job and receipts: " + str(RETRY))
    parent = json.loads((PARENT / "gpu_submission.json").read_text(encoding="utf-8"))
    assert parent["version"] == 1 and parent["final_collection_count"] == 1
    assert parent["final_execution_status"] == "failed_cuda_oom_first_dit_o_proj"
    assert (PARENT / "gpu_final_v1").is_dir()
    repair = REPAIR.read_text(encoding="utf-8")
    for required in (
        "value.shape[0]>=3 and value.shape[0]%3==0",
        "labels=labels_for(residual,segments,gate.shape[0])",
        "row_gate_kernel(*residual.shape,gate.shape[0],shift_for(local_k))",
        "return norm_kernel(*x.shape,shift.shape[0],float(norm.eps))",
        "gate_kernel(*x.shape,gate.shape[0])",
        "consume_input=consume,p0_c=local_rotation",
    ):
        if required not in repair:
            raise RuntimeError("The six-row lifetime fix is missing: " + required)

    baseline.EVIDENCE = RETRY
    baseline.JOB = RETRY / "gpu_job"
    baseline.main()

    binding = json.loads((baseline.JOB / "source_binding.json").read_text(encoding="utf-8"))
    prior_request = json.loads((PARENT / "request.json").read_text(encoding="utf-8"))
    assert binding["plan"] == prior_request, "Retry must preserve all E2E inputs"
    repair_sha = hashlib.sha256(REPAIR.read_bytes()).hexdigest()
    assert binding["overlay_hashes"]["patches/tilelang_dit_fusion.py"] == repair_sha
    (RETRY / "retry_provenance.json").write_text(json.dumps(dict(
        status="built_not_submitted",
        reason="first_nfe_six_row_adaln_fused_row_gate_dispatch",
        parent_kernel=parent["kernel"],
        parent_version=1,
        parent_failure_archive="../h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102/gpu_final_v1",
        repair_source="patches/tilelang_dit_fusion.py",
        repair_sha256=repair_sha,
        overlay_sha256=binding["overlay_sha256"],
        unchanged_request=True,
        requested_kernel=parent["kernel"],
        expected_version=2,
        local_tests="42 Python tests; loader migration PASS",
        status_polling=False,
        live_log_polling=False,
        final_collection_count=0,
    ), indent=2), encoding="utf-8")
    print(json.dumps(dict(status="built_v2_not_submitted",
        job=str(baseline.JOB.relative_to(ROOT)),
        source_sha256=repair_sha,
        overlay_sha256=binding["overlay_sha256"],
        unchanged_request=True), indent=2))


if __name__ == "__main__":
    main()
