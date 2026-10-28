# Sanitized T4 results, 2026-10-28

`summary.json` contains source capsule identities, actual DiT timing/allocator
acceptance, and VAE ablation aggregates. `vae_corrected_attribution.json`
contains all 30 case measurements with actual Chrome kernel-category counts
and CPU-launch-correlated phases. GPU annotation ranges are excluded.

No activation values, model tensors, credentials or raw Nsight environment
tables are included. Raw final outputs remain under the corresponding
`D:/attention_eval` experiment directories. The report explains the fixture,
observer overhead and block-versus-pipeline limitations:
[results](../../reports/H3_DIT_PAIR_VAE_CUDA13_RESULTS_20261028.md).
