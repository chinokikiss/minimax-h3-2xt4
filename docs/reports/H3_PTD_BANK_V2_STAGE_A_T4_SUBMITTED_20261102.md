# H3 PTD Bank v2 Stage A — Physical T4 Submission (2026-11-02)

## Submission status

**Kaggle push accepted; GPU execution is not yet verified.** This report records one source-bound private Stage A submission, not a completed CUDA benchmark, full-model production qualification, or PTD E2E acceptance.

| Field | Value |
| --- | --- |
| Submission | `kaggle kernels push`, one attempt |
| Acknowledgement | `Kernel version 1 successfully pushed.` |
| CLI exit code | `0` |
| Reported version | `1` |
| Returned page | https://www.kaggle.com/code/pamcks/h3-ptd-bank-v2-stage-a-two-block-t4-20261102 |
| Requested metadata ID | `pamcks/h3-ptd-v2-stage-a-t4-20261102` |
| Provider identity caveat | Kaggle reported a title-derived slug different from the requested metadata ID. Use the returned URL, not an inferred URL. |
| Execution / final output | **Not collected; do not claim pass** |
| Kaggle status / log polling | None |

The original submission source is preserved in `docs/data/h3_ptd_v2_stage_a_t4_20261102/gpu_job_v2/`. A complete acknowledgement record is stored in `docs/data/h3_ptd_v2_stage_a_t4_20261102/gpu_submission.json`. There was no repeat push to repair the title/slug warning.

## Deliberately small test contract

- Exactly two different physical Tesla T4 devices, SM75, verified by UUID.
- Real trained PTD **two-block range subset** at the immutable released commit, with per-range source provenance. This explicitly does **not** verify the complete 11.11 GB release SHA256 and cannot be deployed.
- Four Native DiT rank shards for source blocks 0 and 1; no base shard downloads for blocks 2–49.
- One shared v2 part and eight **two-block** PTD Hop parts, assembled as a fixture Bank v2. Production admission must reject that manifest.
- Exactly one eight-Hop tiny-latent TP2 runtime smoke. Its seeded synthetic tensors do not constitute video generation. No TE execution, VAE execution, 1 MP input or 10-second output.
- The ordinary TP2 Loader, production `allow_fixture=False` behavior and full 50-layer paths remain unchanged.

## Coverage and observability

1. Frozen notebook environment from the previously qualified physical-T4 source, with a reviewed 20-file PTD v2 overlay.
2. Base and trained PTD range provenance; local source SHA256 and memory/disk capacity telemetry.
3. Per-hop Bake timing and two-rank shard coverage, Native independent merge checks, full receipt-pinned local SHA256 readback.
4. Bank v2 schema admission, production-fixture rejection and eight-hop execution via the unchanged TP2 operator window.
5. GPU memory sampled by `nvidia-smi`, per-hop selection / denoiser timing, finite-state checks, final latent hashes, backend dispatch counters and failure traces.

The pre-submission local conda `comfyui` regression suite passed **82/82** tests. Notebook AST, private T4 metadata, zero Dataset/kernel mounts, selected layer count, trained-range download gate and SHA256 of all 20 reviewed overlay files were checked.

Notebook SHA256: `720f7478ee0e9c714de878baad0cec12f23d74d05e630a2f3643bc9dc2f03dd6`.

Overlay SHA256: `423df5238ec464480724b74ebab779c715d1db42190855944d3bbf1f3ae7ab53`.

## Next evidence gate

After a user-reported completion event, perform **one** final-output collection for the returned version-pinned Kaggle kernel. Inspect `result.json`, trained-source provenance, the timing table, memory samples, local part proofs, smoke outputs and any exception. Until then, actual GPU pass/fail, runtime precision, speed and memory remain unknown.

Do not interpret a two-layer Stage A pass as evidence of the 50-layer single-hop Baker, full eight-hop Dataset publication, mounted NFS throughput, or production PTD visual/audio quality. Do not poll Kaggle status/list/live logs or automatically retry.
