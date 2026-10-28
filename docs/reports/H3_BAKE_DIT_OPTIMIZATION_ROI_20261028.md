# Offline bake and DiT optimization priorities from actual T4 profiles

A dedicated **FP32 fused RMS-RoPE can directly replace the DiT eager fallback**.
The pinned Kitchen kernel rejects FP32 Q/K; enabling its registry cannot add
that capability. An H3-owned implementation must be explicitly selected for the
qualified shape and preserve FP32 arithmetic. This needs no dependency upgrade
or global Kitchen override. No new operator or GPU experiment is implemented
by this analysis.

The evidence comes from the completed two-T4 offline-bake profile, the two
59,134-token DiT calls under cu128, and the cu128 dispatch probe. The bake and
DiT measurements include instrumentation overhead. DiT uses trained offline-baked
weights with synthetic conditioning and seeded noise; generation quality is
outside their scope. Raw results and submitted sources remain unchanged.

## Offline LoRA baking: optimize the host pipeline first

The measured bake spans **324.306 seconds** in the parent. The two layer-parallel
workers overlap for almost their entire **272.082/272.120-second** spans. They
start about 13.1 seconds after the parent begins and finish about 39.1 seconds
before it returns. Adding worker times would double-count parallel execution.

| Worker phase | GPU0 worker wall seconds | GPU1 worker wall seconds |
| --- | ---: | ---: |
| Base file SHA256, exclusive | 51.237 | 52.772 |
| Base + output tensor-state SHA256, exclusive | 54.290 | 56.224 |
| Output file/manifest SHA256, exclusive | 28.698 | 27.898 |
| All hashing above | **134.225** | **136.894** |
| `fsync`, exclusive | 48.514 | 46.510 |
| Production merge, inclusive | 32.704 | 32.685 |
| Independent verification, inclusive | 39.433 | 39.917 |
| Checkpoint reads, exclusive | 23.982 | 22.338 |
| Safetensors serialization/write, exclusive | 4.332 | 4.311 |

Inclusive merge/verification numbers contain nested reads and operations and
must not be summed with those children. CUDA event spans around those phases
also contain host/dependency gaps; they are not pure kernel work. Only the first
merge is captured as a CUDA trace per worker, so its approximately 125 ms of
actual kernels cannot be extrapolated to all matrices.

Hashing occupies **49.3–50.3%** of each worker span. Tensor-state hashing and
output hashing have thread CPU time almost equal to wall time: this is substantial
CPU work, not simply slow SSD reads. In contrast, `fsync` has only about
0.76–0.78 seconds of thread CPU time despite 46–49 seconds of wall time.
Approximately **46.2 seconds per worker** is payload-file durability; metadata
syncs account for only 2.30/0.34 seconds. Optimizing JSON alone has little ROI.

The current disk baker already assigns whole layers to GPUs and merges each
full-width matrix once before splitting both owners. The measured recipe has
200 block targets plus eight auxiliary targets; all 208 independent Native
checks and all 100 persisted shard-state proofs passed. Reintroducing per-rank
full-width merging would regress this design.

Recommended order:

1. **Pipeline reading/admission, GPU bake/verification, and writing/hash/sync.**
   A bounded CPU writer queue can hash and persist the preceding layer while a
   GPU processes the next. Keep per-file sync before publishing its completed
   manifest and preserve final atomic publication. Bound outstanding CPU states
   to one or two layers and size the CPU pool against actual available cores;
   the existing hash threads already consume substantial CPU. This exposes
   overlap rather than assuming all hashing or disk work disappears.
2. **Reduce repeated traversal without weakening proofs.** The base file digest,
   tensor-state digests and output digest serve different purposes. Preserve
   them, but combine admissible reads and hashing where possible and reuse
   validated immutable identities within the job. `state_hashes` already uses
   a CPU memoryview rather than a full `tobytes()` copy. Changing digest
   algorithms or replacing checksums with size/mtime is not an equivalent fix.
   A writer that hashes exact serialized bytes while emitting them would remove
   a reread, but requires serialization/identity validation before adoption.
3. **Shorten the parent-only tail.** Eight auxiliary targets are currently
   baked after both layer workers finish, followed by auxiliary validation,
   hashing, sync and publication. After the workers finish, measured exclusive
   parent work includes about 12.92 seconds of file hashing, 6.40 seconds of
   sync, and 7.68 seconds of inclusive merge/verification. Schedule auxiliary
   work into the worker plan or overlap its preparation, with explicit single
   ownership of the auxiliary file. Retain the final completion barrier.
4. **Make cache reuse cheap and explicit.** Persisted recipe-based reuse already
   exists. Reuse an admitted reader across requests instead of reconstructing
   and rehashing the same large auxiliary artifact unnecessarily. Preserve
   recipe/source/LoRA-strength binding and mutation detection. This improves
   repeat startup, not the measured first-bake critical path.

Keep independent Native verification for new bake outputs. Removing it would
change the qualification contract. Faster LoRA GEMMs, more GPU workers, or
metadata batching alone are lower priorities than the measured host pipeline.
The workers finish within 23 ms of each other, so layer-load balancing has little
visible headroom in this particular recipe. New overlap speedups remain unmeasured.

## DiT: separate cold start, memory peaks and steady-state work

The first/second denoiser RPCs take **229.158/176.148 seconds**. First-call
compilation scopes total about 45.7 seconds per rank and disappear in the second
call. Prepare an exact-shape operator warmup/cache, bound to source, compiler,
core versions and SM75, before claiming a steady-state improvement. Rank times
overlap; their compilation totals must not be added.

The warmed traced windows have 98.3–98.8% kernel/copy coverage and only
2.14–2.94 seconds of idle gaps. This is activity coverage, not hardware occupancy.
The existing lookahead is already feeding the profiled run effectively. Increasing
it can increase the observed 4.39–4.57 GiB resident-weight overlap without much
idle time left to hide. Keep the policy and measure changes with a controlled
lookahead comparison before increasing residency.

| Warm-call GPU work | Rank0 seconds | Rank1 seconds |
| --- | ---: | ---: |
| Kitchen attention | 68.298 | 64.661 |
| Column INT8 GEMMs | 40.833 | 28.351 |
| Row INT8 GEMMs | 13.570 | 8.421 |
| QK RMSNorm/RoPE | 4.659 | 4.339 |
| Modulated norm + gated residual | 9.518 | 9.401 |
| NCCL kernels, including waits | 15.280 | 38.115 |

Rank0 has approximately 17.6 seconds more GEMM work while rank1 spends about
22.8 seconds more in NCCL kernels. This is consistent with a straggler/arrival
imbalance, but does not identify its cause. Check clocks/throttling, kernel
selection/layout, prefetch interference and arrival times at matching collectives
before redesigning transport. NCCL duration is not pure PCIe transfer cost.

Attention and GEMMs dominate steady-state work. Dedicated RMS-RoPE targets
an important memory peak, but its direct measured GPU-time budget is only
**2.49–2.67% of the warm traced window**. Eliminating that phase's GPU work
entirely would remove only 4.34–4.66 seconds per rank from these records; real
fusion still costs time. Indirect allocator, host-launch or scheduling effects
must be measured, not added as assumed savings.

## Dedicated FP32 RMS-RoPE: the precise replacement

Current DiT behavior is:

```text
FP16 packed QKV
  -> Q/K FP32 copies
  -> FP32 RMSNorm (per head, D=128)
  -> FP32 split-half RoPE
  -> FP16 Q/K storage
  -> existing attention
```

An explicit H3 fused operator can perform those same FP32 calculations on SM75.
It must accept the real strides, per-rank 28 heads, head dimension 128, separate
Q/K gains, norm epsilon, split-half frequency matrix, and actual partial-rotation
contract. Norm reduction is across the full head; partial rotation affects only
the rotary prefix. Preserve multiplication/FMA boundaries and final storage
rounding as closely as required by validation. Changing reduction order can
still change FP32 results; fusion alone does not prove byte identity.

Two interfaces have different memory ROI:

- **Direct FP32-input/output replacement:** removes Kitchen's unsupported-dtype
  fallback and eager intermediate norm/rotation passes. Existing Q/K FP32
  promotions and returned FP32 planes still consume memory.
- **FP16 storage with FP32 fused arithmetic:** load Q/K directly from packed
  FP16 QKV, compute squares/reduction/rsqrt/gains/rotation in FP32, then write
  FP16 once at the existing final cast boundary. This also removes the full
  FP32 promotion/output planes. FP16 storage does not mean FP16 normalization.
  Start with separate output buffers; overwriting Q/K in the packed allocation
  needs an additional alias/consumer audit and must preserve V.

At S=59,134 and 28 heads/rank, one full FP32 Q or K plane is
**847,745,024 bytes / 0.790 GiB**. The two explicit promotions cost **1.579 GiB**.
Warm peak records show five such FP32-sized allocations simultaneously, plus
the **1.184 GiB** packed FP16 QKV allocation. The direct-storage fused interface
avoids the explicit promotions and further full-sized norm/rotation temporaries;
the FP16 Q/K output pair itself still costs 0.790 GiB unless a separately
qualified in-place path is used. A new peak cannot be obtained by subtracting
all these sizes from 11 GiB: lifetimes, allocator reuse, weight residency and
another phase can become limiting.

Implement it behind an explicit H3 function call with the current eager path
as an unqualified-shape fallback. Do not modify Kitchen's constraints to claim
FP32 support, change its version, or downcast norm arithmetic to fit the old
kernel. The cu128 probe's FP16 VAE RoPE speedup is evidence for that different
operator; it is not a measured speedup for the proposed DiT FP32 kernel.

## Remaining memory work, in priority order

1. **FP32 RMS-RoPE fusion:** target the observed warm RoPE peak around
   10.86–11.03 GiB, using the storage/arithmetic distinction above.
2. **Fuse gated residual and modulated norm.** The cold rank0 peak is at FC2
   end, 11.078 GiB. Its live set includes the 1.579 GiB FC1 output, a 1.184 GiB
   FP32 row result, and three roughly 1.09 GiB FP32 residual temporaries.
   A dedicated gated-add kernel can read current residual/branch/gates and
   store FP16 directly while doing the arithmetic in FP32. A later fused
   transport-epilogue/gated-add could avoid the full FP32 row result too.
   Preserve the existing scaled-transport reconstruction and FP32 rounding
   sequence. Norm/modulation and residual ranges account for 6,600 kernels per
   rank across 50 blocks and about 9.4–9.5 seconds of GPU work.
3. **Release FC1 at the quantization boundary.** The caller already deletes it
   after FC2 returns; adding the same `del` again does nothing. Split FC2 into
   prepare/quantize and GEMM/reduce/epilogue stages so the original FC1 buffer
   can be released immediately after its final quantization read. Audit aliases,
   stream dependencies and observer retention. Offline LoRA avoids an online
   adapter consumer keeping that buffer alive.
4. **Bound full INT32 column accumulators if they become the next peak.** FC1
   uses `[59134,14336]` INT32 accumulation, theoretically **3.158 GiB** before
   its FP16 epilogue. Token-row tiling can bound workspace while keeping each
   GEMM's full K and the same row quantization domain. GEMM dispatch may change,
   so revalidate exact INT32/FP16 results and throughput. Direct GEMM epilogue
   output is another candidate but needs its own rounding comparison. The
   existing TileLang epilogue is already integrated; do not count it as new.
5. **Stream FC1 into SwiGLU/quantization only if needed.** This can remove the
   full FC1 activation but is a larger change: paired gate/up ownership,
   ConvRot groups, row maxima, stored-SwiGLU rounding and FC2 consumption must
   remain consistent. Try the simpler lifetime/fusion changes first.

Large recorded host residual times include intrusive shape/lifetime observers.
The current histories are complete, but an observer-off run is needed before
attributing those host times to the production operator or promising their removal.
Keep one uninstrumented latency control and a separate diagnostic capture with
the same fixed inputs. Validate real captured Q/K, short/tail and 59,134-token
shapes, finiteness/overflow and FP16 output differences against the existing
FP16_fix eager FP32-island path. Keep Native FP32 tensor flow as the semantic
ground-truth comparison before promoting a fused kernel; that does not mean
converting the weights or every pipeline operator to FP32.

## Evidence and scope

The extracted measurements and source artifact hashes are in
[`evidence.json`](../data/h3_optimization_roi_20261028/evidence.json).
Related reports:
[DiT cold/warm trace and lifetime results](H3_DIT_PAIR_VAE_CUDA13_RESULTS_20261028.md),
[cu128 dispatch results](H3_CU128_DISPATCH_AUDIT_20261028.md).
Raw bake profiles are under
`D:/attention_eval/h3_vae_precision_bake_20261028/gpu_outputs/vae_bake/bake_profile`;
raw DiT analysis is under
`D:/attention_eval/h3_dit_pair_retry_20261028/gpu_outputs/dit_pair/profile`.

All proposed post-change memory and speed gains remain unmeasured. No Kaggle
interaction, dependency change, production default change or full H3 generation
was required for this analysis.
