The reusable TE/DiT TP2 offline bundle is published and verified.

- Public, ungated HF repository: [cnmds/minimax-h3-tp2-int8-561a6211](https://huggingface.co/cnmds/minimax-h3-tp2-int8-561a6211).
- Pin revision: `d3e6a4b131056d2c626a01acf6ba48123a885783`.
- 200 shards, 41.009 GiB: 50 TE layers and 50 DiT layers, both ranks.
- Bundle SHA256: `87b2815f270f1637bd4ddf1b5e78c581e6226c5fdce4695376e4426bfe22076a`.

Kaggle revision 6 performed the conversion and all weight uploads. Every shard
passed the frozen Native tensor-byte proof, strict offline reload, and serialized
file SHA256/size check. HF object availability and all final commit file identities
were verified. This inherits the previously executed Native proof; it does not
claim another Native model execution or new dual-T4 performance evidence.

Setup took 13.80 minutes; conversion and upload took 40.73 minutes. Total Kaggle
time was 54.53 minutes. The final metadata commit initially hit a connection
timeout; retrying only that step completed publication.

The repository was made public and ungated on 2026-10-12 at the user's request.
Anonymous API access and all 200 shard HEAD identities were verified against
the immutable publication; see `semantic_results/offline_public_access.json`.
Reuse requires no `HF_TOKEN` or Kaggle Secret. The causal audit's public loader
explicitly uses `token=False` and downloads weights on Kaggle. Use the immutable
revision, not `main`:

```bash
cd /kaggle/working/pipeline_audit_source
python benchmarks/offline_checkpoint_store.py \
  --directory /tmp/rank_local_tp2 \
  --plan semantic_results/offline_conversion_plan.json \
  --checkpoints /kaggle/working/pipeline_audit_source/checkpoint_paths.json \
  --repo cnmds/minimax-h3-tp2-int8-561a6211 \
  --reuse-revision d3e6a4b131056d2c626a01acf6ba48123a885783
```

`checkpoint_paths.json` must identify the pinned original TE and DiT files. The
reuse path validates source identities, configuration, bundle completeness,
file checksums and Native tensor hashes, then rebinds strict manifests to the
current host. It never reconverts missing shards during reuse: it fails instead.
The bundle contains TP2 transformer blocks; Native startup/vision, model-global
components and VAEs still use their original checkpoints. Pin ComfyUI and the
qualified loader sources used by this project.

The `HF_token` Secret service failed in earlier Kaggle runs. Revision 6 bypassed
it with expiring, object-scoped upload URLs. Public reuse now avoids that Secret
service entirely; no repeat conversion is needed.

Proof: `semantic_results/kaggle_hf_publication_v6/publication.json`, all 200
manifests, `upload_receipts.json`, and `semantic_results/kaggle_hf_finalize_v6.log`.
New physical measurements must pass a Native output comparison before being
accepted. Public availability is a transport change, not performance evidence.
