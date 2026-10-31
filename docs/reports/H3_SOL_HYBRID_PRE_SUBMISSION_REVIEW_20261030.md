# MiniMax-H3 Sol Hybrid Pre-Submission Review

**Date:** 2026-10-30  
**Scope:** Current local working tree after Kaggle v3 failure review  
**Submission state:** **NOT SUBMITTED**  
**Target:** 2 x Tesla T4 / SM75 / Kaggle  
**Baseline commit:** `d7c7528515d2a7961de5e6f961232d81d21cdc8b`

## 1. Requested E2E policy

```text
TE                 = kitchen_causal

DiT NFE1           = kitchen
DiT NFE2-NFE7      = sol@1.0
DiT NFE8           = kitchen

Video VAE          = sol
Global Sol tau     = 1.0
```

The replay workload remains the completed 0.856 MP / 10 s / 8-NFE regular-Turbo workload. The dry-build request differs from the baseline request only in `name`, `operators`, and `source_note`.

## 2. Review result

The local source is ready for a **physical T4 execution-semantic preflight**, but it is **not yet qualified for the full E2E submission**.

No additional P0 CPU/static blocker was found after the fixes below. The remaining P0/P1 blockers are physical-SM75 numerical/runtime qualification items, not known local API/closure failures.

## 3. Sol native stride / ABI review

### Q/K input

Production Sol uses the fused preprocessing path. The real tensor strides are passed explicitly into the native Q/K preprocessing kernel. The kernel reads Q/K through those strides and writes packed Q8/K8 into contiguous BHLD buffers.

This is compatible with real H3 transpose views such as:

```text
shape  = [B, H, L, D]
stride ~= (3584, 128, 3584, 1)
```

provided the innermost dimension remains contiguous.

### V input

The All-INT8 V quantizer receives `stride(0)`, `stride(1)`, and `stride(2)` explicitly. It reads the original FP16 V tensor through those strides, then emits the native packed V layout `[B,H,D,padded_N]`.

The native V quantizer also validates base/stride alignment and fails closed instead of silently accepting an unsupported layout.

### Output

The previously identified silent output-layout bug is locally fixed. The native kernel writes contiguous BHLD, and Python now allocates an explicit contiguous:

```python
torch.empty((batch, heads, length, head_dim), ...)
```

instead of `torch.empty_like(v)`.

### Remaining physical gate

Static ABI review is closed. Physical T4 qualification is still required for:

- real transpose-view numerical execution;
- actual pointer/alignment behavior on real H3 tensors;
- current protected-route density;
- output sanity against a qualified reference/tolerance.

## 4. DiT schedule and condition protection

The current runtime uses 4-field schedule records:

```text
(start_nfe, end_nfe, backend, tau)
```

and `prepare_dit` checks Sol usage through `schedule_uses_backend()`; the old 3-tuple unpack failure class is removed.

For the real long-sequence layout, the exact-protected prefix stops at the start of generated audio:

```text
text + references + conditioning = protected
generated audio                  = Sol-eligible
generated video                  = Sol-eligible
```

For the previous 65,166-token layout this boundary is 4,164 tokens, before Q64 block rounding.

The native route protection is bidirectional:

```text
all queries -> protected keys = exact
protected queries -> all keys = exact
```

## 5. Video VAE review

The Sol D64 adapter remains:

```text
Q64 -> pad D128, multiply Q by sqrt(2)
K64 -> pad D128
V64 -> pad D128
Sol D128
slice output[..., :64]
```

This restores the original D64 score scale under the fixed D128 kernel scale.

The five VAE global suffix tokens are protected bidirectionally.

### Temporal remote path

The current `native_temporal_pairs` design does **not** perform approximate cross-time attention partitioning.

Each worker independently decodes one complete Native temporal clip. The parent retains the Native padding, chunking, overlap, blending, and final output-write semantics. `TemporalClipPairs` only caches/prefetches the adjacent decoded clip.

Existing CPU contracts verify bit-exact equality against the Native planner for the covered fixtures.

Physical T4 Video VAE Sol quality, VRAM, and throughput remain unqualified.

## 6. Mixed-NFE observer review

The current long-E2E runner is fail-closed for the requested mixed policy.

For each rank it derives expected dispatch from the schedule and requires:

```text
Kitchen DiT = 2 NFE x 50 blocks = 100 calls
Sol DiT     = 6 NFE x 50 blocks = 300 calls
```

It also requires:

- no counted DiT attention outside its denoise command;
- observed Video VAE Sol calls;
- recorded DiT Sol AOT receipt;
- recorded Video VAE Sol AOT receipt;
- `native_temporal_pairs` with no fallback for the requested Video VAE Sol run.

The generic `ref2va_e2e_run.py` observer is less strict by itself, but the current long-E2E entry point adds the required hybrid assertions.

## 7. Source-overlay closure review

The old v1-v3 overlay was not dependency-closed. The current builder now recursively closes changed/new internal Python dependencies against the frozen baseline and explicitly includes dynamic AOT/native dependencies.

Current compact closure:

```text
changed/new dependency nodes = 43
overlay files                = 85
Python overlay files         = 42
```

### Packaging fixes made in this review

1. Native/vendor source trees now retain all regular files, including:
   - `STAR7_LICENSE`
   - third-party `LICENSE`
   - `NOTICE`
   - manifests and headers

2. `experiments/kitchen_attention/` is no longer copied wholesale.
   - Python modules remain governed by recursive import closure.
   - Only `experiments/kitchen_attention/native/` is force-included as the non-AST runtime source dependency.
   - Benchmark/delivery/test-only experiment files are not unnecessarily embedded.

3. Notebook archive extraction no longer depends on `tarfile.extractall(filter='data')`.
   - The notebook extracts only builder-created regular files.
   - Absolute paths and `..` components are rejected explicitly.
   - This avoids Python patch-level compatibility dependence.

## 8. Local gates

Final current-tree results:

```text
Focused Sol/closure/hybrid contracts     17/17 PASS
Full requested local regression          67/67 PASS
H3 operator integration                  PASS (3 record groups)
Overlay py_compile                       42/42 PASS
git diff --check                         PASS
```

The `git diff --check` command reports existing LF/CRLF conversion warnings only; no whitespace error was reported.

## 9. Final dry-build

Prepared locally without `--submit`:

```text
D:\attention_eval\h3_sol_hybrid_085mp_10s_20261030\review_post_audit_compact\gpu_job
```

Build identity:

```text
notebook bytes  = 819,320
notebook SHA256 = 91ff9b28957f20779b164998769cce1da1f077b72094bf7eb547956747b29006

overlay archive bytes  = 150,580
overlay archive SHA256 = 57b22ee686b04f60a4a5eca1aa13bc79e394d981d85262018b6e79ce01206886
```

The notebook remains below the current 900 KB hard build gate with substantially more margin than the earlier 871,704-byte build.

## 10. Remaining blockers before full E2E

### Required physical T4 preflight

There is no current integrated preflight covering all of the following in one real-H3 path:

1. `RankRuntime` startup from the dependency-closed overlay;
2. Kitchen/Sol AOT startup;
3. at least NFE1 Kitchen -> NFE2 Sol dispatch;
4. real H3 transpose-view Q/K/V through production Sol;
5. real condition-layout protection;
6. one real DiT block/short denoise path;
7. Video VAE Sol D64 adapter on the production temporal remote path.

The existing `build_sol_all_int8_t4_job.py` is an isolated synthetic operator benchmark and does not close this gate.

### Still unqualified

- real H3 hybrid trajectory drift;
- final generation quality;
- reference adherence;
- temporal consistency;
- Video VAE Sol visual quality;
- Video VAE Sol peak memory;
- current protected-route density and actual E2E speedup.

## 11. Submission decision

**Do not submit the full 0.856 MP / 10 s E2E yet.**

Recommended next sequence:

1. build a small source-bound physical-T4 execution-semantic preflight for the seven items above;
2. run it once;
3. inspect final artifacts after the user reports completion/failure;
4. only if it passes, submit the prepared full hybrid E2E once;
5. after submission, do not poll status or live logs; wait for the user's manual completion/failure report.

No commit, push, reset, clean, or unrelated working-tree rewrite was performed by this review.
