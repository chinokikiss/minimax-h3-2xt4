# H3 PTD v2 Hop0 Bake T4 — Stage B Final Result (2026-11-02)

**Result: PASS for full 50-layer, single-Hop-0 physical dual-T4 Bake qualification only.**

## Source evidence

- Kaggle kernel: `pamcks/h3-ptd-v2-hop0-bake-t4-20261102`, version 1; URL https://www.kaggle.com/code/pamcks/h3-ptd-v2-hop0-bake-t4-20261102
- Collected Kaggle outputs: `docs/data/h3_ptd_v2_hop0_t4_20261102/gpu_final_v1/`
- Authoritative result: `ptd_v2_hop0_bake_t4_20261102/result.json` -> `status=complete`
- Worker supervision: `ptd_v2_hop0_supervision/supervision.json` -> `exit_code=0`, `seconds=884.236`
- Source preflight: `source_preflight.json` -> `status=passed`, `source_ready=true`, `physical_baker_ready=true`, `issues=[]`
- Receipt files: `hop_00_receipt.json`, `shared_receipt.json`
- Notebook binding SHA256: `6f29d0c1746521fb68ae0cd77db0d78989c1468dd3a638bb47c1fc7fd26ed231` (submission record)

## Hardware and source

- Two different physical NVIDIA Tesla T4s (SM75); PyTorch 2.11.0+cu128; CUDA 12.8; TileLang 0.1.13; Comfy Kitchen 0.2.37.
- Full Native DiT base: 50 layers; source file size 20,970,379,616 bytes; source SHA256 `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779`.
- Complete pinned PTD 8-step checkpoint: 11,112,893,024 bytes; full source file SHA256 verified and matches `6796802b9150110a3cd7f66891c9e6190502a432c30b0fa1aa299724cfc15cc3`.
- PTD schema: 50 adapted blocks, 2 adapted refiners, 8 available hops and 32 PDD intervals. Explicit pruned-AdaLN approximation allowed. Zero ordinary LoRAs.
- Scratch preflight minimum available gate: required 26,895,315,658 bytes; available 1,092,314,185,728 bytes; passed. Tar/upload archive headroom was not part of this estimate.

## Stage timings

| Stage | Time, seconds | Status |
|---|---:|---|
| stage_full_base | 252.418 | complete |
| download_pinned_full_ptd | 129.090 | complete |
| production_source_preflight | 35.064 | complete |
| bake_shared | 35.176 | complete |
| bake_hop_00 | 346.361 | complete |
| verify_local_part_sha256 | 82.512 | complete |
| **Sum of timed stages** | **880.622** | **6/6 PASS** |

Start: `2026-11-02T15:12:23.006605+00:00`; finish: `2026-11-02T15:27:05.755379+00:00`; observed end-to-end result interval **882.749 s**; worker supervision **884.236 s**. These are different measurement spans. They exclude unspecified provider startup/mounting.

## Output and proofs

- 50 complete DiT layers, **100/100 rank shards** (`dit.block00.rank0` through `dit.block49.rank1`).
- **208 verified merge targets**: 200 DiT linear targets + 8 token-refiner targets. Four DiT linear targets per layer; four per refiner.
- All 100 rank shard safetensors are **196,348,072 bytes** each in the receipt. One Hop-0 auxiliary overlay has **1,544,161,056 bytes**.
- Local Hop-0 payload includes **202 files** totaling **21,179,555,374 bytes** (19.73 GiB); `verify_part` returned `status=verified` and checked receipt-pinned file hashes.
- Shared and Hop-0 recipe digests match: `c8b832fb86d00233a7cee437f73c5de02fa9509b58954fef55671a3012bc3716`.
- Hop-0 local payload tree SHA256: `9414def1ad33ac8b0e1f22bdb09654d9a6a30b1b9bc8be2bfdf1061f7deec2f1`.
- Shared part receipt complete: auxiliary SHA256 `31d75f4ac268eb191aebdf2251977fa10a8b066ebf68cd465e4326f2468d7279`.

## Resource telemetry

- NVML sampled peak GPU 0: **3965 MiB** (3.87 GiB); GPU 1: **3727 MiB** (3.64 GiB).
- **841** GPU memory samples; **0** sampling errors. Sampled peaks are not absolute instantaneous peak guarantees.
- Local scratch final free space: **1,071,074,586,624 bytes**, after retaining local intermediate artifacts.

## Verification semantics and limits

**Important:** `independent_native_reference=false` in this run. The updated PTD v2 default performs original Native full-width merge once, retains the composed FP32 reference, and checks the quantized output bytes from that same reference. It does **not** execute the separate independent A/B formula per target. Strict independent reference checking remains opt-in through `PTD_V2_INDEPENDENT_VERIFY=1`.

This run **did not** bake Hops 1–7, assemble the full eight-hop production Bank, publish/read back any Dataset, run PTD inference, or measure video/audio quality. Full Hop-0 weights and output artifacts were in Kaggle disposable local scratch; only **result/preflight/receipt/log evidence** was exported and downloaded locally. This is a local part readback proof, **not remote Dataset durability evidence**.

**Decision:** Stage B physical full-50-layer Hop-0 local Bake and hash-check gate **PASS**. Full eight-hop immutable publication, remote readback, production Bank assembly and end-to-end quality remain separate outstanding gates. Do not automatically submit another notebook or publish the payload.
