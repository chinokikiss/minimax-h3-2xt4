# H3 PTD V2 Overlay — tensor_preadv ×1 vs HTTP ×4 (2026-11-04)

Status: **Kaggle CPU Kernel v1 submitted, execution/measurements pending**. No GPU used and no production PTD runtime was changed.

- Actual Kaggle URL: https://www.kaggle.com/code/pamcks/h3-ptd-overlay-preadv1-vs-http4-cpu-20261104
- Submission CLI exit: 0; provider response: `Kernel version 1 successfully pushed.`
- Planned ID differed from title-derived actual URL; preserved this discrepancy in the source-binding/submission record. No retry was attempted.
- Frozen notebook SHA256: `f182f44191d4ab24da0f6eeade08d07a544ca98787e159220ffa5b48cab7261e`.

## Fairness and measurement

Every method reads **the same eight published immutable Hop00–07 `aux_overlay.safetensors` files**, each verified by the receipt SHA256. One `tensor_preadv` reader loads four Overlay files serially; four KaggleHub HTTP workers download four Overlay files concurrently. Both arms therefore process **four identical-sized full Overlay files per case**, rather than comparing differently sized shards.

| Case | Method | Hops |
| --- | --- | --- |
| phase1_mount_A | Mounted `tensor_preadv`, one worker | 02, 03, 04, 05 |
| phase1_http4_B | Four concurrent HTTP file downloads | 06, 07, 00, 01 |
| phase2_http4_A | Four concurrent HTTP file downloads | 02, 03, 04, 05 |
| phase2_mount_B | Mounted `tensor_preadv`, one worker | 06, 07, 00, 01 |

Recorded metrics:
1. Mounted application wall throughput and isolated `tensor_preadv` payload throughput.
2. HTTP completed payload bytes divided by four-worker end-to-end download time.
3. HTTP output is placed in dedicated **instance-disk `/kaggle/temp`**; subsequent full-file one-reader disk-to-RAM throughput and integrity are measured separately.
4. Combined HTTP download plus disk-to-RAM wall throughput; checksum times are separately recorded.

Caveats: Kaggle's FUSE cache cannot be reliably cold-reset. The initial A/B uses disjoint first-touch file groups; the second pass crosses over source groups. A completed Kaggle Kernel push is not a successful benchmark result. No status/list/live-log polling or automatic retry is permitted.

## Evidence

- Frozen job: `docs/data/h3_ptd_v2_stage_c_20261103/overlay_http4_vs_preadv_v1_local/`
- Local case-plan self-test: **PASS**
- Submit receipt: `submission_v1.json`
- Await `/kaggle/working/h3_ptd_overlay_preadv_vs_http4_cpu_20261104/measurement/result.json` before giving real throughput figures.
