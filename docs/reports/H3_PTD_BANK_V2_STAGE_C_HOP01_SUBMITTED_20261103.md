# Stage C Hop01 dual-T4 publication — submission receipt (2026-11-03)

The user's request was to upload Hop01–07. Following the established per-hop publication/readback gate, Hop01 was the first independently submitted job. Shared and Hop00 already passed separately mounted remote SHA256, providing prior admission.

- Kernel: `pamcks/h3-ptd-v2-stage-c-hop-01-20261103`, version **1**; Kaggle push exit code **0**.
- URL: https://www.kaggle.com/code/pamcks/h3-ptd-v2-stage-c-hop-01-20261103
- Local immutable candidate: `docs/data/h3_ptd_v2_stage_c_20261103/gpu_job_hop01_v1_local/`.
- Notebook SHA256: `e4a1ede3180616bf67b2f99f2d08967cd9210398a0fb49457412a5eb774282fd` (verified immediately before push).
- Frozen-source overlay: 89 files, SHA256 `33339f4a85fc41f0d7704addd8f3207dd158930e9f12b2b0b9cf2499875ea19e`; cold import PASS immediately before push.
- Explicit intent: one full 50-layer physical dual-T4 Hop01 bake followed by **one private Dataset create request** to `pamcks/h3-ptd-v2-hop-01-20261103` using `dir_mode=tar`.
- Submission evidence: `docs/data/h3_ptd_v2_stage_c_20261103/gpu_submission_hop01_v1.json`.

**Current external status:** GPU result, source receipt, private Dataset creation ACK, authenticated `/versions/N` and mounted SHA256 readback are all **pending**. A successful Kernel push is not a model-weight publication proof. Hop02–07 candidates remain locally built/verified but **not submitted**, preserving the established sequential gate. No Kaggle status/list/live-log polling and no automatic retries.
