# H3 KC1–3 / Sol4–8 / Video VAE Kitchen — Alternate Seed 987654321

**Submission date:** 2026-10-31
**Kaggle:** https://www.kaggle.com/code/pamcks/h3-kc-sol-seed987654321-e2e
**Notebook version:** 1
**Submission:** **ACCEPTED** (exit code 0). This is a new Notebook entity with no stale `id_no`.
**Execution result:** **NOT YET COLLECTED**; no GPU status/log polling or final outputs retrieval was performed.
**Requested GPU:** Kaggle 2× Tesla T4 SM75 (the notebook must validate actual GPU allocation on execution).

## Experiment contract

A controlled alternate-seed replay of the previous accepted 0.85MP, 8-NFE long E2E hybrid request:

| Key | Previous | New |
|---|---|---|
| `seed` | 20261020 | **987654321** |
| Input geometry | 1216×704, 243 native frames | Unchanged |
| Export | 240 frames / 24 FPS / 10 s | Unchanged |
| NFE / sampler / scheduler | 8 / Euler / Simple | Unchanged |
| CFG / video shift / audio shift | 1.0 / 6.0 / 3.0 | Unchanged |
| DiT NFE 1–3 | Kitchen INT8 | Unchanged |
| DiT NFE 4–8 | Restored pre-math-fix Sol v2, τ=1 | Unchanged |
| TE | Kitchen Causal | Unchanged |
| Video VAE | Kitchen INT8 | Unchanged |
| Audio VAE | Native | Unchanged |
| Prompt / 2 reference images | Original anime shrine / shrine maiden / white fox | Unchanged, including hashes |
| Turbo LoRA | Pinned `lightx2v/Minimax-h3-Turbo` regular 8-step, offline baked | Unchanged |
| Base TP2 checkpoint | `cnmds/minimax-h3-tp2-int8-561a6211` | Unchanged |

The previous submission is recorded in `docs/data/h3_kc_sol_8step_long_e2e_20261031`. The new plan differs only in `seed`, plus the non-semantic `name` and `source_note` labels. Changing the integer by a large magnitude yields a different PRNG seed, but numerical distance between seeds **does not guarantee** a correspondingly large visual or semantic change.

## Narrow, source-locked seed allowlist override

The existing `benchmarks/turbo_challenge_contract.py` only accepts `20261020` or `20261021` by design. To support the user's high-distance seed **without altering any checked-in code**, the new builder packages a one-line, in-memory variant of that contract in its source overlay:

```diff
-SEEDS=frozenset((20261020,20261021))
+SEEDS=frozenset((20261020,20261021,987654321))
```

This is the **only source-file hash difference** versus the previous accepted hybrid job. All runtime math, attention kernels, model weights and artifact identity constraints are unchanged. The native Sol source is verified against the exact pre-math-restore manifest. The modified contract is included in the Notebook tar archive and verified by local tests against its recorded SHA256.

The seed is applied consistently to the full runtime request (video and audio sampling where relevant). No new checkpoint or LoRA is introduced.

## Local preflight

- **35/35 tests passed**: source closure, exact input-diff, all 8 NFE policy dispatch, old Sol restoration, job-local contract archive bytes, Notebook metadata / worker code, and immutable request/binding.
- Notebook size: **838,092 bytes**, beneath the known 900,000-byte envelope.
- Frozen source overlay contains original source files except the one-line seed allowlist amendment.
- `H3_SOL_FUSED_KC_THRESHOLD=0`, `H3_SOL_WORKSPACE_POOL=0`, `H3_SOL_LUT_BITMAP=0`.
- **Expected** (not yet observed): 150 Kitchen + 250 Sol DiT Attention calls per rank (50 blocks × 3/5 NFEs), plus Video VAE Kitchen.
- Source/hash checks fail closed if the underlying restored Sol source changes.

## Provider submission receipt

- Time: **2026-10-31T06:18:49.291984+00:00**
- Kernel: `pamcks/h3-kc-sol-seed987654321-e2e`
- Version: **1**.
- Notebook SHA256: `a9cff3db6bc213b349aa2bfe7d657a078d6011f8390f0437a25c0eba2d8ca083`
- Source binding SHA256: `5f74c4176326ff2d67a75dc0122caa43db1641420908af9d6c88b07f4fa69658`

## Artifacts

- Notebook: `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/gpu_job/experiment.ipynb`
- Request: `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/request.json`
- Source binding: `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/gpu_job/source_binding.json`
- Build receipt: `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/build_receipt.json`
- Submit receipt: `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/gpu_submission.json`
- Provider CLI receipt: `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/gpu_submission_cli.log`
- Preflight: `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/preflight.log`
- Builder: `benchmarks/build_h3_kc_sol_seed987654321_long_e2e.py`
- Unit tests: `benchmarks/test_h3_kc_sol_seed987654321_job.py`
- Submission script: `benchmarks/submit_h3_kc_sol_seed987654321_long_e2e.py`

**No automated polling, log inspection, retry, or real-time status claim after acceptance.** GPU completion, video/media artifacts, per-NFE attention dispatch counters, and elapsed times require explicit final collection later.
