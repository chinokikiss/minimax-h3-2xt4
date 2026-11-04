# H3 PTD Bank v2 Stage C — independent dual-T4 Hop01+Hop02 I/O overlap

**Date:** 2026-11-03
**Status:** Kaggle private GPU Kernel **version 1 accepted**; **physical Bake, private Dataset create acknowledgments, actual timings and separately mounted remote SHA256 all pending**. This document does not claim a speedup.

## Test target

- Kaggle private Notebook: https://www.kaggle.com/code/pamcks/h3-ptd-v2-stage-c-dual-hop-0102-ioab-20261103
- Kernel ID: `pamcks/h3-ptd-v2-stage-c-dual-hop-0102-ioab-20261103`
- Embedded frozen-source overlay: **90 files**, `0ebff2eb3b266d9fcb7d2f13f9ef54997db748db26685b4e38b29d473f34e3ce`.
- Notebook SHA256: `6618bba34f5509a6c05a89cb71fa12aad6ac2832252f5d69a022b2e45416bd75`.
- Parent frozen commit: `d7c7528515d2a7961de5e6f961232d81d21cdc8b`.
- Frozen cold import: PASS. Local regressions: **39 PASS**. Kaggle CLI push exit **0** with `Kernel version 1 successfully pushed.`
- Dedicated, new private Dataset create targets: `pamcks/h3-ptd-v2-hop-01-ioab-20261103` and `pamcks/h3-ptd-v2-hop-02-ioab-20261103`. Existing production Hop01/Hop02 Dataset IDs and Kernel jobs were **not** overwritten.

## Architecture change (opt-in; ordinary Stage C unchanged)

1. Source Base staging and pinned PTD download run in parallel from disjoint paths. `download_on_t4` already streams and checks the full pinned PTD SHA256. Its exact inode/mtime/ctime/size fingerprint is passed to the read-only preflight, eliminating a redundant checkpoint-wide second read without weakening the normal preflight default. The recipe is constructed once from authenticated Base/PTD digest and schema, and checked against the qualified Hop00 recipe digest `c8b832fb86d00233a7cee437f73c5de02fa9509b58954fef55671a3012bc3716`.
2. GPU **0 produces all 50 layers of Hop01**, GPU **1 produces all 50 layers of Hop02**. Each independent worker reconstructs full Native projection width and emits the **two TP2 rank shards** per layer. This is *not* simultaneous inter-device TP2 GEMM.
3. Inside each worker, independent CPU prefetch reads and validates one next layer in a background thread; current-layer GPU merge/quantize and previous-layer durable safetensors/manifest writes are overlapped with bounded ownership. Base rank-state and output shard proofs are preserved.
4. Each completed Hop is fully verified exactly once with local per-file SHA256 and an inode/ctime/size source snapshot. Zero-copy hardlink staging reuses that verified proof if—and only if—the source identity is unchanged. On Windows, where st_ctime is creation time rather than inode-change time, cached-snapshot checks conservatively fall back to content hashes; physical trial is Linux.
5. A two-thread private Kaggle uploader starts on each Hop's completion independently of the other GPU's Bake. SDK calls are private *create* (never update), no automatic retry, no status/list/live-log polling. The complete receipt is required before a single Dataset can begin uploading; this is overlap **across** Hops, not speculative partial Dataset upload.
6. Upload ACK is **not** private Dataset version identity or remote byte correctness. A separately version-pinned, independently mounted SHA256 readback is still required before Bank v2 assembly. Do not substitute these experimental Dataset parts into the nine-part production Bank without explicit acceptance and remote proof.

## Measured Hop00 reference and pending comparison

Hop00 is a historical **one-Hop, two-T4 layer-partitioned** Stage C run. The numbers below are measured in the downloaded Hop00 `result.json`, not estimates.

| Component | Hop00 measured | New Hop01/Hop02 IOAB |
| --- | ---: | --- |
| Base staging | 231.614 s | Pending; overlaps PTD download |
| PTD download | 137.336 s | Pending; overlaps Base staging |
| Full preflight | 33.028 s | Pending; reuses download-time pinned PTD SHA |
| Hop Bake | 317.250 s (one Hop, two GPUs split layers) | Pending; two independent full Hop workers |
| Local Hop full SHA256 | 82.141 s | Pending; one scan per Hop |
| Hop private staging + SDK upload | 485.992 s | Pending; avoids second full local staging hash |
| Shared part Bake + verify + upload | 42.082 s | Excluded; Shared was already independently validated |
| Cold-start **one-Hop equivalent** (excluding Shared) | **1,288.442 s** | Not yet measured |
| Estimated two sequential Hop00-style jobs | **2,576.885 s** | Compare against measured dual-Hop wall after completion |

The 2,576.885-second figure is **two times measured Hop00 non-Shared stages**, an *extrapolated sequential expectation*, **not** historical measurement of Hop01+02. The analysis should report both overall throughput and per-stage times, and can never claim remote completion merely from successful upload API responses.

## Evidence, continuation

- Submission receipt: `docs/data/h3_ptd_v2_stage_c_20261103/gpu_submission_dual_hop_ioab_v1.json`.
- Source-bound Notebook: `docs/data/h3_ptd_v2_stage_c_20261103/gpu_job_dual_hop_ioab_v2_local/`.
- Experiment executor: `benchmarks/run_ptd_v2_stage_c_dual_hop_t4.py`.
- Comparison tool: `benchmarks/compare_ptd_v2_stage_c_ioab.py`; run only after the user reports the Notebook has finished and final output has been explicitly collected, against the existing archived Hop00 result path.
- No Kernel status polling, no automatic download or resubmission, no production Dataset overwrite, no model Bank promotion.
