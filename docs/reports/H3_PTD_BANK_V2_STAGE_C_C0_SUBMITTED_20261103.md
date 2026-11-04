# H3 PTD Bank v2 — Stage C C0 Private Dataset Probe (2026-11-03)

## Stage C continuation — readback tooling and seven Hop candidates (2026-11-03)

The user explicitly requested completing the full nine-part publication and verification sequence. One final-output collection of Kaggle GPU Hop00 Kernel v1 returned exit 0 **without files**; local `gpu_final_hop00_v1/` contains no final artifacts. Do not interpret this as a completed or failed GPU job, and do not poll Kernel status/list/live logs or retry. Hop00 GPU outcome, two requested private Dataset create ACKs, provider version identities and model payload remote SHA256 remain **pending**.

Implemented `benchmarks/verify_ptd_v2_stage_c_remote.py` (CPU-only, standard library), `benchmarks/build_ptd_v2_stage_c_readback_job.py` (deterministic private Kaggle CPU Notebook builder) and `benchmarks/test_ptd_v2_stage_c_remote.py`. New tests **4/4 PASS**: standalone verifier matches authoritative `verify_part` file count, byte count and payload-tree SHA on fixture Shared/Hop00, rejects tampering and unversioned identities, and builds byte-identical private CPU Notebook deliveries. Each future readback Notebook requires a real source receipt SHA256, Stage B recipe SHA256, authenticated private Dataset version, and distinct mounted input path. **No production readback job submitted yet.** Version pins passed to verifier are operator-provided; they require a separate authenticated provider metadata proof, and Kaggle dataset source mount itself is not an authenticated version claim.

Generated unsubmitted immutable Hop01–07 GPU candidates, one per hop, at `docs/data/h3_ptd_v2_stage_c_20261103/gpu_job_hop01_v1_local/` through `gpu_job_hop07_v1_local/`. Seven of seven separate frozen-overlay cold imports PASS, each 89 files and overlay SHA256 `33339f4a85fc41f0d7704addd8f3207dd158930e9f12b2b0b9cf2499875ea19e`. **Do not submit Hop01** until Hop00 model payload and independently mounted Shared/Hop00 readback are qualified; no final Bank assembly yet.


## Latest result — C0 v3 PASS; Stage C Hop00 submitted (2026-11-03)

User authorized one v3 CPU readback retry. Kaggle accepted `pamcks/h3-ptd-v2-c0-tar-readback-20261103` Kernel version **3**; local Notebook SHA256 `d929e1df923201f8a2242fb70616413f0ae674f31504b2424c84469b945dd66b`. Final CPU artifact `docs/data/h3_ptd_v2_c0_tar_probe_20261103/readback_final_v3/h3_ptd_v2_c0_tar_readback/result.json` reports `passed_direct_payload_mount`. Exact mounted path `/kaggle/input/datasets/pamcks/h3-ptd-v2-c0-tar-probe-20261103`, actual payload `payload/`, expected receipt `payload/receipt.json`, **202/202** synthetic SHA256 pass, **26,476,544** synthetic bytes. Earlier v1/v2 failed because they did not search the nested Kaggle mount convention; remote provider ZIP proof was separate. CPU experiment is synthetic-only and does not establish large-model upload performance.

Following C0 PASS and the pre-existing authorization to complete Stage C, **one** Hop00 real dual-T4 private publication Notebook was submitted successfully as Kernel version **1**: https://www.kaggle.com/code/pamcks/h3-ptd-v2-stage-c-hop-00-20261103. Immutable submission evidence `docs/data/h3_ptd_v2_stage_c_20261103/gpu_submission_hop00_v1.json`, Notebook SHA256 `5926bd21d30ab80d82c257cbf8e8418199835e2ba8ef16b01e1d6f2682c06448`, 89-file overlay cold import PASS and 16/16 CPU regressions PASS. **Notebook execution result PENDING.** Intended output requests private Datasets `pamcks/h3-ptd-v2-shared-20261103` and `pamcks/h3-ptd-v2-hop-00-20261103`; no create ACK, data availability, actual Kaggle `/versions/N` pin, or mounted per-file model SHA256 is proven yet. No status/list/live-log polling, no automatic retries. Do not launch Hop01 before explicit Hop00 outcome and independent remote readback qualification. Stage C nine-part Bank assembly is pending.


## Historical result — C0 CPU readback v2 (resolved by v3 PASS; 2026-11-03)

The user authorized **one** corrected CPU mounted-readback retry. Kaggle accepted Kernel v2 at https://www.kaggle.com/code/pamcks/h3-ptd-v2-c0-tar-readback-20261103 (exit 0; local submission receipt `docs/data/h3_ptd_v2_c0_tar_probe_20261103/readback_submission_v2.json`). Final output collected **once**: `docs/data/h3_ptd_v2_c0_tar_probe_20261103/readback_final_v2/h3_ptd_v2_c0_tar_readback/result.json` reports `status=failed`, `error_type=FileNotFoundError`, `mounted_slugs=['datasets']`. The v2 script searched `/kaggle/input/SLUG` and the immediate children of `/kaggle/input`; it did **not** descend into `/kaggle/input/datasets/OWNER/SLUG`. Thus neither the existence nor nonexistence of a nested mount is established. No mounted SHA256 executed; do not conflate prior provider-ZIP download proof with Notebook mounting.

Kaggle community reporting identifies `/kaggle/input/datasets/USERNAME/DATASET` as a possible Dataset mount layout (https://www.kaggle.com/questions-and-answers/684771), consistent with the only top-level `datasets` directory observed. This is a **hypothesis, not validated C0 data**. Prepared a **CPU-only v3 candidate, NOT submitted**, that explicitly probes the flat and owner-nested layouts, records bounded nested directory names and then verifies the expected 202 synthetic SHA256s: `docs/data/h3_ptd_v2_c0_tar_probe_20261103/readback_job_v3_local/` notebook SHA256 `d929e1df923201f8a2242fb70616413f0ae674f31504b2424c84469b945dd66b`.

Latest **Stage C Hop00 GPU candidate v4 remains local-only, NOT submitted**: `docs/data/h3_ptd_v2_stage_c_20261103/gpu_job_hop00_v4_local/`, source-closure **89** files; frozen-source cold imports PASS, combined 16/16 tests PASS, deterministic notebook SHA256 `5926bd21d30ab80d82c257cbf8e8418199835e2ba8ef16b01e1d6f2682c06448`. The Kaggle GPU source overlay now includes a separately pinned, locally hashed official MiniMax H3 LICENSE (`MiniMaxAI/MiniMax-H3` at `42ed227ee7df40d41602854ae760620d6eb651fe`); the previous code incorrectly tried to fetch that LICENSE using a revision belonging to the separate `Comfy-Org/MiniMax-H3` source repository. The fix avoids that cross-repository revision error and embeds the offline license asset.

**Historical v2 gate:** at this earlier point, further submission required explicit authorization. That authorization was subsequently granted for v3 CPU readback, which PASSED; Hop00 GPU was then submitted once. Remote model bytes and all nine production part readback proofs remain pending. All prior C0 v1/v2 artifacts and local candidate versions remain preserved.

## External status

User confirmed Kaggle private Dataset publication under pamcks, sufficient capacity, and H3 derivative-weight hosting authorization.

**C0 synthetic tar layout probe submitted once:** https://www.kaggle.com/code/pamcks/h3-ptd-v2-c0-private-tar-probe-20261103

- Kaggle kernel version: 1; push exit code: 0.
- Dataset requested: pamcks/h3-ptd-v2-c0-tar-probe-20261103.
- Scope: 203 synthetic payload files (~25 MiB), private API create with dir_mode=tar; no model weights or embedded credentials.
- Push warned that title and requested slug differed. The returned Kernel URL is preserved; no re-push.
- Final C0 upload Notebook result collected once: API status Ok; request accepted; payload 26,476,544 bytes, 203 files; Dataset URL https://www.kaggle.com/datasets/pamcks/h3-ptd-v2-c0-tar-probe-20261103.
- Separate CPU mounted readback was submitted once: https://www.kaggle.com/code/pamcks/h3-ptd-v2-c0-tar-readback-20261103 (kernel v1, push exit 0). Its final output was collected once: FAILED before hashing with FileNotFoundError because expected /kaggle/input/... mount directory was absent.
- One-time provider Dataset metadata returned datasetId 13049152, owner pamcks and isPrivate=true.
- One-time Kaggle Dataset ZIP readback downloaded 204 entries containing payload/receipt.json, 202 synthetic files, and README. Independent SHA256 verified every synthetic file (26,476,544 bytes), the receipt, and the preserved payload/ structure. Evidence: docs/data/h3_ptd_v2_c0_tar_probe_20261103/download_readback_v1.json. This is NOT mounted NFS proof.
- One-time provider Kernel source/metadata pull returned 403 Forbidden on GetKernel. No alternate access was attempted.
- The v2 readback candidate was explicitly submitted as Kernel version 2 and FAILED due to absent flat mount; its observed `/kaggle/input` entry was `datasets`. A v3 nested-path-aware candidate is locally prepared, NOT submitted pending new explicit authorization.
- No Kaggle status/list/live-log polling.

## Local implementation

- benchmarks/build_ptd_v2_c0_tar_probe.py: synthetic private upload probe builder.
- benchmarks/build_ptd_v2_c0_readback.py: CPU mounted readback candidate (built; not submitted).
- benchmarks/run_ptd_v2_stage_c_publish_t4.py: independent physical dual-T4 one-hop Baker and private-create publisher. Hop 00 emits Shared and Hop 00; other hops emit only themselves.
- benchmarks/build_ptd_v2_stage_c_job.py: frozen-source, dependency-closed Notebook builder; stable gzip timestamps.
- benchmarks/verify_ptd_v2_delivery.py: extended to support the new Stage C Notebook overlay.
- benchmarks/test_ptd_v2_stage_c_delivery.py: private-only, closure and byte-reproducibility tests.

Stage C GPU publisher retains qualified Base/PTD sources, verifies the exact Stage B recipe c8b832fb86d00233a7cee437f73c5de02fa9509b58954fef55671a3012bc3716, requires two distinct T4 GPUs, attaches source-pinned MiniMax H3 LICENSE and modified-derivative NOTICE, and sets public=False. It does not poll external status or confuse upload acknowledgment with remote byte proof.

## Local validation

- Frozen-commit cold import with 88-file overlay PASS.
- Combined Stage C and existing A/B delivery/publication tests: 16/16 PASS.
- Latest deterministic Stage C builder test: 3/3 PASS, repeat builds byte-identical.
- The initial Stage C Hop 00 candidate is local only and was not submitted. Rebuild after final title alignment edit before use.

## Next gate

The C0 readback v1 and explicitly authorized v2 both failed to reach mounted SHA256; v2 revealed a top-level `datasets` directory without inspecting its nested contents. Direct provider-download SHA256 and tar hierarchy PASS, but Notebook NFS mount remains unqualified. Stop before the large Stage C Hop 00 GPU job. Obtain explicit authorization for a new v3 CPU nested-path readback submission; never auto-retry. Only after mount SHA256 passes should Stage C Hop 00 Bake/Publish be submitted. Current local GPU candidate is `gpu_job_hop00_v4_local`: frozen-commit cold import passed all 89 overlay files, notebook SHA256 `5926bd21d30ab80d82c257cbf8e8418199835e2ba8ef16b01e1d6f2682c06448`. Stage C model part publication and remote readback have not yet started.
