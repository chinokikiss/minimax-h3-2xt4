# H3 PTD Bank v2 — Stage C Entry / Publication Gate (2026-11-02)

## Current status

**Stage C source admission PASS; Dataset publication and remote readback have NOT started.** This report is based on the physical Stage B Kaggle v1 output, not a new full-size GPU run.

- Source evidence: `docs/data/h3_ptd_v2_hop0_t4_20261102/gpu_final_v1/ptd_v2_hop0_bake_t4_20261102/`
- Current source-bound plan: `docs/data/h3_ptd_v2_hop0_t4_20261102/stage_c_plan_v1.json`
- Deterministic local plan generator: `benchmarks/plan_ptd_v2_stage_c.py`
- CPU tests: `benchmarks/test_ptd_v2_stage_c.py` and `benchmarks/test_ptd_v2_publication.py`; **11/11 PASS**.
- Production 50-layer source recipe SHA256: `c8b832fb86d00233a7cee437f73c5de02fa9509b58954fef55671a3012bc3716`.
- PTD full-release source SHA256: `6796802b9150110a3cd7f66891c9e6190502a432c30b0fa1aa299724cfc15cc3`.

## Why the Hop 0 Bake must run again

The successful Stage B notebook wrote its 21,179,555,374-byte Hop 0 payload to Kaggle disposable local scratch and exported only result receipts and telemetry. There is no persistent Hop 0 Dataset or locally downloaded set of weights. Stage C must **re-bake Hop 0 and shared** under a publication-enabled run; Stage B does **not** imply the part is currently available for assembly.

## Storage projection (non-quota evidence)

Stage B Hop 0 was exactly 21,179,555,374 bytes of receipt-pinned payload. Shared part's nominal payload was 58,693,088 bytes. Assuming equivalent per-hop shard/overlay geometry:

```
8 × 21,179,555,374 + 58,693,088 = 169,495,136,080 bytes
≈ 169.50 GB decimal ≈ 157.86 GiB
```

This **excludes** Dataset metadata, receipts, tar archives, retained versions, existing private Dataset usage, and server-side processing overhead. It is not evidence of actual available Kaggle quota or large-payload transfer throughput.

## Stage C recommended execution sequence

1. **C0: publication provider admission.** Decide Kaggle private 9-collection cohort or immutable Hugging Face private commit(s); check actual account quota, dataset size/processing limits, credential availability, and the license/derivative redistribution policy. Do not infer authorization from the Stage B local Bake alone. Previously successful 16 MiB native Kaggle API private upload did not establish 21 GiB reliability or any remotely verified hash. Verify a representative small-to-large end-to-end private upload/readback first.
2. **C1: re-bake and durably publish Shared + Hop 00.** Stage full pinned Base+PTD; re-bake Shared and Hop 00 using exactly the recipe above, run `verify_part` on source scratch, save source receipt SHA256 separately, prepare zero-copy Kaggle staging under `payload/` on the same volume (if Kaggle), create a **new** private Dataset identity without `-u`, confirm processing, and **independently read back actual mounted remote bytes** to generate a version-pinned proof. Do not delete scratch before that gate. Do not alter the existing Stage B or Stage A receipts.
3. **C2: remaining Hops 01..07.** One Hop at a time: bake → local proof → immutable private publication → processing-complete check → separately mounted remote readback SHA256 → record `owner/slug/versions/N` → only then reclaim owned scratch. Do not use repeated Dataset versions as an append primitive; each must be independently complete.
4. **C3: production Bank assembly.** Mount **9 distinct** immutable remote parts (shared + 8 hops), validate all 9 independently preserved source receipt hashes against their remote readback proofs, and invoke `assemble_bank_v2` with nine proofs and `allow_fixture=False`. Only on passing all 50-layer/100-shard/recipe gates publish the small global manifest last.
5. **C4: full PTD runtime qualification**, separately from Bank publication: 8-Hop real full-model execution, numeric finite/hash/reference checks, performance/VRAM/host memory, and eventually TE/VAE E2E and quality. A successful C3 alone is not inference-quality certification.

## Existing code supports the gates but does not authenticate a remote publish

- `patches/ptd/publish_v2.py::prepare_private_kaggle_part` performs SHA checks and zero-copy hardlink staging but **never uploads**; Kaggle CLI should use `datasets create -r tar` and never `-u` for public visibility.
- `verify_part` streams receipt-pinned per-file SHA256 on a separately mounted **remote** part and returns a proof. `check_readback_proof` requires a pinned Dataset identity and saved source receipt SHA256.
- `assemble_bank_v2` requires 9 distinct mounted parts and 9 remote proofs in production mode; no partial Bank publication.
- **Known limitation:** a readback proof can contain a *claimed* published identity; the current validator does not authenticate the provider's dataset metadata. Production publication needs an external version/owner/visibility confirmation before treating the proof as authoritative.

## Operations performed this turn

- Read Stage B real GPU receipts, existing publication code and published CPU upload probe.
- Added a source-bound Stage C plan generator and fail-closed regression tests.
- Executed 11/11 publication/Stage C tests successfully.
- Emitted immutable local `stage_c_plan_v1.json` with `status=source_qualified_publication_not_started`.
- **No** GPU bake, Kaggle Kernel push, Dataset upload, dataset deletion, remote download, or Bank assembly occurred.

**Next gating decision**: select and authorize the private persistence destination and license, and verify account quota/large payload upload. Do not claim Stage C execution PASS based on the source-only admission.
