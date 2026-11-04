# H3 PTD v2 Stage C Dual-Hop 01/02 — Serial Dataset Upload v2

Date: 2026-11-03

## Result

**Kaggle private Kernel version 2 accepted for execution**. Push exited 0; link:
https://www.kaggle.com/code/pamcks/h3-ptd-v2-stage-c-dual-hop-0102-ioab-20261103

No GPU result or private Dataset create receipt has been obtained for v2. The final remote version pins and mounted per-file SHA256 are pending.

## Prior v1 failure

Both independent physical T4 workers completed Hop01 and Hop02 and local SHA256 (202 files / 21,179,555,374 bytes per Hop); two concurrent Kaggle Dataset create calls both failed without ACK after approximately 189.5 seconds. There was no remote readback, and the baked payload was ephemeral. The user reports neither target Dataset was created.

## Version 2 changes

- GPU0 produces all of Hop01 and GPU1 all of Hop02, concurrently, keeping one-layer CPU input prefetch and durable output writing.
- Local verification and hardlink staging overlap in two uploader threads, without another full-payload hash scan during staging.
- Kaggle Dataset create requests are serialized: Hop01 must return an accepted create response before Hop02 can enter `dataset_create_new`.
- If Hop01 create fails (or Hop01 bake fails), Hop02 create is skipped. Hop02 local baking/proof can still complete.
- Exception diagnostics retain the Python exception type, optional HTTP status and errno, never raw SDK response bodies, signed URLs, or credentials.
- No automatic retry, no status/list/live-log polling, and no promotion into production Bank v2 on a submission ACK.

## Local admission

- 28/28 local CPU regression tests PASS.
- Exact new Notebook frozen-overlay cold import PASS, 90 files, baseline commit `d7c7528515d2a7961de5e6f961232d81d21cdc8b`.
- Notebook SHA256: `f530d8141100ebb25a5ad5e2b106b21f3ea644a031fec3d05a85c61210c6fe87`
- Overlay SHA256: `9f30b343e03e39d2a659950a54058afdd8e091525b1bbe0d7e531a1f79ef7e4a`
- Frozen job: `docs/data/h3_ptd_v2_stage_c_20261103/gpu_job_dual_hop_ioab_v3_local/`
- Submission receipt: `docs/data/h3_ptd_v2_stage_c_20261103/gpu_submission_dual_hop_ioab_v2.json`

## Next verification gate

After final Kaggle artifacts are available, inspect `result.json`, `hop_01_local_proof.json`, `hop_02_local_proof.json`, and upload stage errors/ACKs. Only then authenticate both private Dataset versions and independently SHA256 the mounted payloads. Do not treat local SHA or SDK ACK as remote integrity proof.
