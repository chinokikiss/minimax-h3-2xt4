# H3 PTD Bank v2 — Stage B Physical Dual-T4 Hop 0 (2026-11-02)

## Submission outcome

**Submitted and accepted by Kaggle, version 1. GPU runtime result not yet collected.**

- Kernel: `pamcks/h3-ptd-v2-hop0-bake-t4-20261102`
- Provider URL: https://www.kaggle.com/code/pamcks/h3-ptd-v2-hop0-bake-t4-20261102
- Notebook SHA256: `6f29d0c1746521fb68ae0cd77db0d78989c1468dd3a638bb47c1fc7fd26ed231`
- Overlay SHA256: `39e4c5b99ec37f3f83a38272bf2544566d4262244259ffb18ddc9064ea0ba6f1` (87 files)
- Pinned frozen commit: `d7c7528515d2a7961de5e6f961232d81d21cdc8b`
- Source delivery cold-import check: PASS (87 files)
- Targeted local unit tests: 29/29 PASS
- `kaggle kernels push`: one attempt, exit code 0; kernel version 1
- Kaggle timeout: 10,800 seconds; bounded worker budget: 9,000 seconds
- Execution status, output, per-stage timing and physical T4 memory: **not collected**. Do not infer a GPU pass.

## Stage B test contract

One complete, original trained PTD release (11,112,893,024 bytes) and full Native DiT Base are staged and validated on **two distinct physical Tesla T4 GPUs**. The worker verifies the complete release SHA256 and the 50-layer PTD schema and scratch budget before baking. It creates a **single Hop 00** with 50 DiT layers, 2 Rank shards per layer (100 total), plus a shared auxiliary part. It checks merged-target coverage, recipe identity, local hop payload/receipt SHA256, GPU sampled VRAM and scratch space. No user model/video inference, other seven hops, remote Dataset publishing or production Bank assembly is authorized.

The current fast PTD v2 merge path retains the composed FP32 matrix for exported Native INT8 byte checks without repeating base dequant or LoRA GEMM. This is **not independent A/B formula verification** by default; set `PTD_V2_INDEPENDENT_VERIFY=1` in a separate audit run to restore exhaustive independent per-target verification. The test runner was corrected to describe this distinction accurately.

## Prerequisite evidence

Stage A Kaggle version 2 completed all 14 stages on two distinct T4 GPUs, baked all 8 two-layer hops, validated 9 local parts, assembled a fixture Bank, rejected it in production mode, and completed finite 8-hop tiny-latent inference. Its physical GPU result is a **two-layer fixture**, not qualification for the full 50-layer Stage B test.

## Evidence locations

- Stage B candidate: `docs/data/h3_ptd_v2_hop0_t4_20261102/gpu_job_v3_local/`
- Immutable Stage B acceptance receipt: `docs/data/h3_ptd_v2_hop0_t4_20261102/gpu_submission_v1.json`
- Stage A collected GPU results: `docs/data/h3_ptd_v2_stage_a_t4_20261102/gpu_final_v2/`

Do not automatically resubmit, poll or claim Stage B PASS without the provider's final output.
