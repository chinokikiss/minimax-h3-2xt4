# Physical two-T4 PTD queue comparison — final evidence

Collected on 2026-10-29 (Asia/Shanghai). Kaggle notebook version 2 completed all four ABBA cases and all 19 PTD contracts. Hardware: two distinct Tesla T4 (SM75), native Torch 2.11.0+cu128, Kitchen 0.2.37, TileLang 0.1.13.

- `summary.json`: independently checked hashes, identities, complete worker spans, derived timings and memory metrics.
- `ptd_result.json.gz`: lossless compressed original result. Its uncompressed SHA256 is recorded in the summary; raw original is retained on D:.
- Case directories: original AOT receipts and durable complete worker reports.
- Collection: one final artifact collection, with one bounded transport-only resume after SSL unexpected EOF. Ten already downloaded files were excluded. Two output-download CLI invocations, no status/list/live-log polling, no new submission. Signed download URLs and CLI error logs are excluded from this public evidence.

Timing is the sum of parent hop-selection and denoiser RPC durations. It excludes download/bake, AOT preparation, service creation/configuration and teardown; it is not full-pipeline latency. The fixture uses two trained blocks, both refiners and complete heads for all eight experts, with small seeded 24-token inputs. Each method has two repetitions. No full-model speedup or generation-quality claim follows.
