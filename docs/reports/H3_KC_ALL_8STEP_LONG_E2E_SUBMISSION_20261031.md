# H3 Kitchen INT8 All E2E — Long 0.85 MP / 10s / 8 NFE — Submission

**Date:** 2026-10-31
**Kaggle:** https://www.kaggle.com/code/pamcks/h3-kc-all-e2e-20261031
**Provider submission:** ACCEPTED, **version 1**. Submission UTC 2026-10-31T05:29:43.713661+00:00.
**GPU:** One Kaggle notebook request for two Tesla T4 GPUs (SM75). Actual device count validated on runtime by the frozen benchmark, not yet proven here.
**Status:** Submitted only, **no run completion claim**, no provider status polling or log polling.

## Exact inherited workload

- Frozen completed baseline: `docs/data/h3_turbo_085mp_10s_20261029/gpu_v2_final/turbo_085mp_e2e/retry_request.json`.
- MiniMax-H3 Ref2VA; `1216x704` (0.856064 MP); 243 native frames (10.125 s); export 240 frames / 10 s at 24 fps.
- Regular 8-step Turbo LoRA `lightx2v/Minimax-h3-Turbo`, `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`, revision `30f485ed874aa44c8183348ad361c5eabad318da`, strength 1.0, offline-baked TP2 disk checkpoint.
- Euler / simple; CFG 1.0; video shift 6; audio shift 3; seed 20261020.
- Original exact Japanese shrine, purple-haired shrine maiden and white fox spirit prompt, and two reference images, all SHA256/pins preserved.

## Requested full-KC operator policy

- TE: `kitchen_causal`.
- DiT: `dit_attention='kitchen'` with **empty** `dit_attention_schedule`, so **all eight NFE steps** use Kitchen INT8; no Sol dispatch expected.
- Video VAE: `video_attention='kitchen'`, native temporal-pair Video VAE decode.
- Audio VAE: Native.
- Sol-specific source remains in the identical source-locked overlay as the prior hybrid run for strict source comparability, **but should not execute**. KC threshold fusion, Workspace Pool and Bitmap LUT remain disabled in the frozen process environment.
- Expected DiT attention counters on each rank at completion: **8 × 50 = 400 Kitchen calls; zero Sol calls**, subject to actual execution validation.
- Compared with the previous hybrid `1-3:kitchen,4-8:sol@1.0` / Video VAE Kitchen job, only the DiT attention schedule changed. All other operator policy fields and external workload input keys are identical.

## Preflight and provenance

- Local tests: **35/35 PASS**, including source hash closure, unchanged external inputs, identical overlay archive vs previous hybrid, and nested notebook worker policy.
- Notebook: `docs/data/h3_kc_all_8step_long_e2e_20261031/gpu_job/experiment.ipynb`.
- Notebook SHA256: `e5e367f3acbb59ee1aad1871b1f17ce2d45b653fc3f2e189f461cebf2c9f13c7`.
- Source binding SHA256: `4b1144e9ebac54720166be9d02a111a0ac8203f15ee7db4d24c251ce98e2de66`.
- Source overlay is inherited, frozen and hash verified against the preceding KC/Sol hybrid E2E. Pre-math-fix Sol source is preserved; the main production source was not modified.
- A prior Notebook metadata `id_no` was explicitly excluded to prevent Kaggle from modifying the historical notebook. Provider returned **v1** with matching canonical slug.
- Baseline source bootstrap preserves no dataset/kernel mounts, pinned CUDA/PyTorch/Kitchen/TileLang versions, model/LoRA integrity validations and download/bake pipeline.

## Receipt

- `docs/data/h3_kc_all_8step_long_e2e_20261031/gpu_submission.json`
- `docs/data/h3_kc_all_8step_long_e2e_20261031/gpu_submission_cli.log`
- `docs/data/h3_kc_all_8step_long_e2e_20261031/request.json`
- `docs/data/h3_kc_all_8step_long_e2e_20261031/gpu_job/source_binding.json`
- `docs/data/h3_kc_all_8step_long_e2e_20261031/preflight.log`

**No status polling, live log polling, automatic collection or resubmission.** Actual quality, elapsed wall time, frame decode completion and GPU dispatch counters await the user's later explicit follow-up.
