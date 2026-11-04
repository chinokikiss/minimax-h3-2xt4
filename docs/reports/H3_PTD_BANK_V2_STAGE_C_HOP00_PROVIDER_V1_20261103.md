# MiniMax H3 PTD Bank v2 — Stage C Hop00 receipt and provider identity (2026-11-03)

## Remote SHA256 verification — Shared PASS, Hop00 running (2026-11-03)

The user explicitly requested full per-file SHA256. Two private, CPU-only mounted-readback Kaggle Kernels were submitted exactly once each: Shared `https://www.kaggle.com/code/pamcks/h3-ptd-v2-readback-shared-20261103` version 1; Hop00 `https://www.kaggle.com/code/pamcks/h3-ptd-v2-readback-hop-00-20261103` version 1. They pin the authenticated provider Dataset `/versions/1` identities and pre-upload source receipt hashes, and use the independently parity-tested `standalone-stage-c-mount-sha256-v1` verifier. Submission evidence: `docs/data/h3_ptd_v2_stage_c_20261103/readback_submission_shared_v1.json` and `readback_submission_hop_00_v1.json`.

**Shared remote mounted readback PASS:** `docs/data/h3_ptd_v2_stage_c_20261103/readback_final_shared_v1/h3_ptd_v2_readback_shared/result.json` reports `passed_remote_sha256`, mounted `/kaggle/input/datasets/pamcks/h3-ptd-v2-shared-20261103/payload`; all 3 files and 58,693,088 bytes match source receipt `e9e843e4665b320fde52bfe8ffbe6f3a73644a0cdfb1fe3ccc68f3a715234285` and payload-tree SHA `a26310d9eafb44c964bd684ad3943d2f2a19ec247d9b116d0110e08e849bee54`. Production-format proof `docs/data/h3_ptd_v2_stage_c_20261103/readback_final_shared_v1/h3_ptd_v2_readback_shared/part_readback_proof.json` preserved. This resolves source-versus-mounted byte integrity despite the provider's earlier 561,811-byte metadata figure.

**Hop00 remote mounted readback still PENDING:** a final output collection did not yet return files for the 21,179,555,374-byte, 202-file CPU job; do not call it failed or passed. No Kaggle status/list/live-log polling, no automatic reruns, and no Hop01 GPU submission until Hop00 SHA256 proof passes. Full nine-part production Bank remains unassembled.


## Verified result

Kaggle GPU Kernel [pamcks/h3-ptd-v2-stage-c-hop-00-20261103](https://www.kaggle.com/code/pamcks/h3-ptd-v2-stage-c-hop-00-20261103), **version 1**, finished its Stage C worker with **exit code 0** after **1,363.084 s**. Its `result.json` reports `status=upload_requests_accepted_remote_readback_pending`; all 11 stages are complete, including 50-layer Shared/Hop00 re-bake, local SHA256 and two private Dataset-create requests. The runtime used two physically distinct Tesla T4 UUIDs.

The final GPU files were retrieved into:
`docs/data/h3_ptd_v2_stage_c_20261103/gpu_final_hop00_v1/`.

Both source receipt files have been **independently re-hashed on Windows** and match the GPU's earlier local part proofs. Both sources bind to the exact recipe SHA256
`c8b832fb86d00233a7cee437f73c5de02fa9509b58954fef55671a3012bc3716`.

| Part | Independent receipt SHA256 | Local payload files | Local payload bytes | Local payload tree SHA256 |
| --- | --- | ---: | ---: | --- |
| Shared | `e9e843e4665b320fde52bfe8ffbe6f3a73644a0cdfb1fe3ccc68f3a715234285` | 3 | 58,693,088 | `a26310d9eafb44c964bd684ad3943d2f2a19ec247d9b116d0110e08e849bee54` |
| Hop00 | `e58edab7fba104ffdadb25cf59b9460ccd1d0a7312758e222757d151547ba6f8` | 202 | 21,179,555,374 | `f8920b05ecacab749b03902517e5bedc7eae556381366e03a0cd05a5f6a3c33c` |

Hop00 contains all **100** rank shard entries plus the rank sidecars and auxiliary overlay.

## Authenticated Kaggle Dataset identity

A separate one-time **authenticated exact-dataset `ApiGetDatasetRequest → get_dataset`** call was made for **each exact handle**. No Kaggle dataset list, status, or live-log polling was used.

| Part | Kaggle Dataset ID | Owner | Private | Current version | Version count | Immutable intended pin |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Shared | 13049779 | pamcks | true | 1 | 1 | `pamcks/h3-ptd-v2-shared-20261103/versions/1` |
| Hop00 | 13049897 | pamcks | true | 1 | 1 | `pamcks/h3-ptd-v2-hop-00-20261103/versions/1` |

The provider's `total_bytes` field was **561,811** for Shared and **21,180,239,999** for Hop00. Shared's returned value is **not equal to** its 58,693,088-byte source payload. Provider size-field semantics are not proven; do not infer remote completeness from metadata. The separately mounted readback remains mandatory.

Authentic identity evidence:
- `docs/data/h3_ptd_v2_stage_c_20261103/provider_identity_shared_v1.json`
- `docs/data/h3_ptd_v2_stage_c_20261103/provider_identity_hop_00_v1.json`
- `docs/data/h3_ptd_v2_stage_c_20261103/provider_metadata_shared_v1/dataset-metadata.json`

## Next immutable gate (not executed yet)

Two private, CPU-only Kaggle mounted-readback notebooks are locally built and **NOT submitted**:
- `docs/data/h3_ptd_v2_stage_c_20261103/readback_shared_v1_local/`, Notebook SHA256 `4ed6a23c7f0b2a2540cc47d0db6c77db6bbf5c5ab6f96fb350b89e4a5cb8542c`
- `docs/data/h3_ptd_v2_stage_c_20261103/readback_hop_00_v1_local/`, Notebook SHA256 `8d3d202c3d839296a5503c37202c7315c9b830fe790e86456c501e256e2b5fc4`

Each input is the exact new private Dataset, and the notebook streams complete remote payload SHA256 against the independently preserved pre-upload receipt and recipe. Only after both readbacks pass and their remote proofs are archived may Hop01 be submitted. Version pin claims in the proof are caller-provided and must be checked against the independently captured provider identity above.

**Do not claim production Bank v2 completion, remote model SHA256, or E2E inference from the provider version and create acknowledgments alone.**
