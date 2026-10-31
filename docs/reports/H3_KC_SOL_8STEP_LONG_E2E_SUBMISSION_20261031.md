# H3 Long E2E — KC1–3, Sol4–8, VAE KC — 2026-10-31

## Submission

- Kaggle provider accepted **one GPU submission**, reporting **version 4** with exit code 0.
- Returned provider link: https://www.kaggle.com/pamcks/h3-kc1-3-sol4-8-vae-kc-long-e2e-20261031
- Intended metadata slug: `pamcks/h3-kc-sol-8step-long-e2e-20261031`
- Submission UTC: `2026-10-31T04:01:24.847778+00:00`
- **IMPORTANT metadata caveat:** the cloned historical baseline metadata retained `id_no: 141520451`. This numeric ID matches the old `h3-turbo-0-85mp-10s-t4-20261029` Kaggle notebook. Kaggle warned the title does not resolve to the specified ID, and its returned URL uses the **title-derived slug**, not the intended `id` slug. The acceptance as **version 4** strongly suggests updating an existing Kaggle notebook instead of creating a fresh independent notebook; however, this **is** a newly accepted GPU run. Do not treat it as an independent-new-notebook proof or automatically submit again. The previously completed baseline result files and frozen source remain preserved locally.
- Status/list/live-log polling: **not performed**. Actual GPU allocation, completion and runtime dispatch remain unverified.
- Provider receipt: `docs/data/h3_kc_sol_8step_long_e2e_20261031/gpu_submission.json`
- CLI response: `docs/data/h3_kc_sol_8step_long_e2e_20261031/gpu_submission_cli.log`
- Source-lock and exact request: `docs/data/h3_kc_sol_8step_long_e2e_20261031/gpu_job/source_binding.json` and `request.json`

## Reused E2E workload

- Exact completed previous **regular 0.85 MP / 10 s / 8 NFE** input, seed `20261020`, 1216×704, 243 native frames, 240 frames exported at 24 FPS.
- Same anime shrine / shrine maiden / fox prompt, two reference images, pinned regular 8-step `lightx2v/Minimax-h3-Turbo` adapter, persisted offline TP2 baking, model bundle, Euler/Simple, CFG 1, video shift 6, audio shift 3.
- Previous successful input source: `docs/data/h3_turbo_085mp_10s_20261029/gpu_v2_final/turbo_085mp_e2e/retry_request.json`.
- Previous complete generation: matching `.../generation/result.json`.
- Frozen source and environment bootstrap: `docs/data/h3_turbo_085mp_10s_20261029/provider_v2_source/`, using source-closed overlay.

## Requested attention schedule

- TE: `kitchen_causal`.
- DiT: `1-3:kitchen,4-8:sol@1.0`; default `kitchen`. Expected 50 attention operators per NFE on each rank: **150 Kitchen + 250 Sol** if all 8 denoiser calls complete.
- Sol: restored **pre-math-fix** Sol v2; topology v2 on, temporal radius 1, local token radius 64, protected frame boundaries, sync-free LUT, route stats disabled, `tau=1.0`, KC threshold fusion disabled, workspace pool off, bitmap LUT off.
- Video VAE: `kitchen` (KC INT8). Audio VAE: native.
- No model, prompt, reference, LoRA, seed, sampler, geometry or shift changes.

## Preflight

- **40/40** unit/preflight checks passed locally, including old Sol file-hash restoration, overlay closure, exact input identity, schedule, metadata and notebook AST.
- `experiment.ipynb`: 837,976 bytes, SHA256 `0bab712e0b337dd751c3212beccd1a3c4bfa2243fdee0fdc48eaf42f46a0babc`.
- Source binding SHA256: `b9badce8b2a9e3f5a5adb714ae55eed7600c8ffb94e73c873bb58034038c03f5`.
- 85 source-locked overlay files; five restored Sol core hashes match the pre-math-fix manifest.
- The full E2E must access external model/reference/source downloads, so `enable_internet=true` is retained.
- No additional setup GPU job was submitted.

**Result:** provider submission **accepted**, not a completed E2E validation. Await user-triggered follow-up rather than polling.
