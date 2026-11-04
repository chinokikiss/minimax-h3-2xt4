# Stage C IOAB Hop01/02 — remote proof gate (2026-11-03)

## Observed, independently checked

GPU notebook `pamcks/h3-ptd-v2-stage-c-dual-hop-0102-ioab-20261103` version 2 completed with both local bakes and both **serial Dataset-create ACKs**; its final `result.json` still correctly says `upload_requests_accepted_remote_readback_pending`.

| Part | Experimental private Dataset handle | Source receipt SHA256 (rehashed on local host) | Local payload-tree SHA256 |
| --- | --- | --- | --- |
| Hop01 | `pamcks/h3-ptd-v2-hop-01-ioab-20261103` | `9cae93ba1c8004bf7a0bd30a04b4b2ac0f127cc7f84ea0d734bc2b54d6106387` | `aa51f0a0d6cc9e3b91f9852864c1a9674473634b11f8f28a7d185f66e183718e` |
| Hop02 | `pamcks/h3-ptd-v2-hop-02-ioab-20261103` | `e504858cdf63053563454db7a282bcc485ce7aa759d8268060fbfe8de729a20e` | `6bf72338f4e9100fb4fff724feaf2cab4b75385f7961d889a876dae5ec85ab70` |

Both locally verified 202 payload files and 21,179,555,374 bytes (50 layers; 100 rank shards); recipe `c8b832fb86d00233a7cee437f73c5de02fa9509b58954fef55671a3012bc3716`.

These are **local**, not independently mounted remote SHA256 proofs. Kaggle `dataset_create_new` ACK is not proof that a final private Dataset version is readable.

## Independent provider identity gate — BLOCKED

The exact authenticated private Dataset metadata API returned HTTP **403 Forbidden** for these two IOAB Dataset IDs in the preceding retrieval. No provider-authenticated version number, version count, or independent private flag has been obtained. We did not assume that `/versions/1` was authenticated, repeat a denied request by another route, list/poll Kaggle jobs, or submit an unqualified remote-readback notebook.

## Readback support completed locally

`benchmarks/build_ptd_v2_stage_c_readback_job.py` now supports the explicit `--ioab` switch for **only** `hop_01` and `hop_02`, preserving normal Dataset defaults. It targets separate private CPU readback Kernel IDs `pamcks/h3-ptd-v2-readback-hop-01-ioab-20261103` and `pamcks/h3-ptd-v2-readback-hop-02-ioab-20261103`, mounting exactly the IOAB Dataset handles. The embedded readback verifier checks the original receipt SHA256 and independently hashes all 202 files in the mounted `payload/` directory. It still labels the version identity as *operator-pinned*, not provider-authenticated.

Five focused Stage C readback tests **PASS**, including both IOAB bindings, private CPU-only metadata, deterministic source, and tamper detection. No persistent CPU readback submission was generated because the provider version gate is still unsatisfied.

## Next steps to finish

1. Obtain separately authenticated **private** metadata for the two exact IOAB Dataset handles, including the actual `/versions/N` pins; require owner `pamcks` and verify these are not similarly named non-IOAB production Datasets.
2. Generate two immutable CPU-only readback notebooks with `--ioab`, the authenticated version numbers, the exact source receipt hashes above, and the pinned recipe.
3. Submit each once (no GPU); after the owner confirms completion, collect only final artifacts. Require `status=passed_remote_sha256`, 202 files, 21,179,555,374 bytes, the matching source receipt SHA256, and the matching local payload-tree SHA256 *for each Hop independently*.
4. Only after both provider identity and mounted bytes pass, mark the IOAB Hop01/02 as remotely qualified; do not promote the rest of the nine-part Bank ahead of later parts.

This report is an explicit **pending/blocker record**, not a remote proof or Bank-v2 promotion.
