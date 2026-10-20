# Video VAE query-parallel fallback — checked 2026-10-19

**Deployment choice:** use the pinned local ComfyUI and retain the legacy
attention implementation. Its trained Query Parallel path passes a byte-exact
local logical-rank check. See [runtime setup and validation](H3_LEGACY_RUNTIME.md).
The newer-source fallback findings below are historical diagnostics.

With the current local `ComfyUI-master`, `decode_mode=auto` selects single-device
Native Video VAE decode. Query Parallel is not executing on that source version.
This is an attention API compatibility fallback, occurring before worker memory
admission or any query/KV collective.

## Root cause

The qualified pinned source has:

```python
Attention.forward(x, rotary_pos_emb, pre_norm, residual, residual_scale)
```

Its attention module owns fused pre-normalization and the residual epilogue,
has a nonpersistent unit `qk_norm_scale`, and selects Kitchen INT8 attention for
quantized QKV weights using a VAE-local availability flag.

The current source has:

```python
Attention.forward(x, rotary_pos_emb=None)
```

Pre-normalization/residual addition moved to `TransformerBlock.forward`; Q/K
normalization and RoPE are separate calls. The VAE-local Kitchen flag and fused
unit-scale buffer are absent in all 36 blocks. Attention dispatch now uses the
module's global `optimized_attention` selection.

Kitchen attention **is available** in the checked installation, and all 36
loaded QKV weights remain `QuantizedTensor`. The fallback is not caused by lost
INT8 weights or an unavailable Kitchen build. Merely importing/reintroducing
the old flag would leave the call signature and normalization interface wrong.

Restoring query parallelism on this API requires adapting the attention wrapper
to the new normalization/residual boundaries and selecting/validating the
desired attention backend. No such numerical/partitioning change was made in
this check; the qualified old API remains supported.

## Executed check

Actual trained Video VAE, FP16 `[1,24,1,16,16]` latents, all 36 decoder layers,
256×256 output. Its 261 query tokens have a valid 128-aligned split, so this
does not depend on the tiny-query native-only rule.

| Public call | Native output comparison | SDPA calls | Kitchen attention calls | Worker commands |
|---|---|---:|---:|---|
| `decode` | byte-exact | 36 | 0 | `decode_start`, `decode_complete` |
| `decode_tiled` | byte-exact | 36 | 0 | `decode_start`, `decode_complete` |

There were no `vae_prepare`, `video_pixels`, or `vae_release` calls. Native
public batch/tile planning stayed active, and `_decode_pixels` was not replaced.
The observed backend was PyTorch SDPA in this fresh local process. A ComfyUI
process launched with another global attention selection can use that backend;
this check does not assert another running process's startup flags.

`query_partitioning_status()` now records the exact flag/signature/buffer/weight
checks. `ParallelDecodeHandoff.last_query_partitioning` exposes that status,
and `last_fallback`/the log identify the failing conditions. The boolean helper
remains import-compatible. Unknown signatures fail closed.

The legacy pinned source was also checked locally: the compatibility gate
passes, all 36 nonpersistent QK scale buffers are ones, and explicit Native
fallback matches ordinary Native decode byte-for-byte. The check found that
meta-device staging needed to restore those unit buffers after `to_empty`;
that restoration is fixed. Query-parallel NCCL execution was not rerun here.

## Evidence and limits

Files in `D:/attention_eval/video_query_fallback_20261019/`:

- `check.json`: trained current-source checks, kernel-call counts, command
  traces, compatibility status and source hashes.
- `legacy_native.json`: pinned-source compatibility and Native fallback check.
- Unit logs: six compatibility-gate tests and thirteen loading/lifecycle tests.

All GPU work here used one RTX 5060 Ti SM120. No T4 job, full H3 generation,
physical dual-GPU timing or speedup measurement was performed.

```powershell
python benchmarks/test_video_query_fallback.py
python benchmarks/check_video_query_fallback.py
```
