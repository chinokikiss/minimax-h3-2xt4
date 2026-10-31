# Production file checks

As requested on 2026-10-30, production inference admits local artifacts using
metadata and structural checks. It does not recompute or verify SHA256 for
TE/DiT shards, auxiliary checkpoints, persisted LoRA/PTD weights, manifests,
receipts, imported source files or compiled Kitchen libraries.

## What is checked

- Regular-file identity: byte size, nanosecond modification/change times,
  device and inode/file identifier, as provided by the platform.
- The identity is captured on first admission and checked on subsequent use.
  JSON reads and checkpoint reads also check for changes during consumption.
- Safetensors headers must fit inside the file. Tensor dtype, shape, byte
  length and non-overlapping contiguous ranges must cover the payload exactly.
- Rank tensor names, shapes and dtypes must match the retained schema. Strict
  module loading, layer/rank coverage, quantization metadata, bake completion
  and cross-manifest provenance labels remain checked.
- Known baked-shard and PTD-overlay byte sizes are checked against receipts.
  Parent/worker PTD admission compares the manifest's metadata identity.
- Native libraries retain target, compiler, successful-build and nonempty-file
  checks. New build receipts include binary byte size. Existing receipts remain
  usable with session-local metadata admission. Imported module paths and
  dependency/API version checks remain unchanged.

Root identities are checked during reads. Full admitted-shard metadata is
checked at layer preparation boundaries; there is no repeated payload scan.
The normal buffered reader remains the default.

## Runtime and tool boundaries

`RankRuntime` calls `open_checkpoint(..., verify_weights=False)` for both TE and
DiT. Production PTD adapters likewise select `verify_payloads=False`.
The PTD Bake node's existing-bank path also uses metadata checks, including a
size check for an explicitly supplied local checkpoint. Creating a new bank
still invokes the offline baker and its publication proofs.
The legacy `verify_dit_weights` runtime/loader input is accepted but normalized
to false and removed from the UI. The PTD adapter's legacy
`expected_bank_sha256` string input is also accepted for compatibility but no
longer verifies a hash; it is removed from the UI.

Low-level `BundleReader`, `DiskBakedReader`, `PTDBank` and `open_checkpoint`
retain verified defaults for offline baking and integrity tools. Explicit
source qualification (`launch_h3_legacy --check`), package/source capsule
qualification, downloads/publication and offline bake proofs retain their
existing content verification. Historical benchmark sources and results are
not retroactively changed. Cache-key and newly generated build/AOT receipt
hashes are identities, not runtime content verification; they do not scan
model weights during inference.

## Trust and compatibility

The first local artifact is trusted. Metadata guards detect ordinary
replacement, truncation and timestamp changes during a session; they cannot
prove initial contents or detect every same-size corruption with preserved
metadata. `ctime` has different semantics on Windows and Linux. No
cryptographic-integrity claim is made by the runtime.

Existing descriptors and bake layouts need no conversion, weights and operator
arithmetic are unchanged, and old graphs remain loadable. A changed admitted
file requires reloading the session rather than silently reusing stale weights.

## Validation

`benchmarks/test_runtime_metadata.py` forbids SHA256/file/tensor hashing while
loading both TE and DiT fixtures, all eight PTD hops and their auxiliaries,
legacy rank blocks, source files and compiled-library fixtures. It checks exact
tensor equality and exercises mtime changes, same-size/mtime file replacement,
truncation, malformed headers, wrong tensor dtypes and parent/worker identity
disagreement. Offline integrity, baking and existing loading tests are run
separately. No full H3 generation or T4 speedup is claimed for this change.
