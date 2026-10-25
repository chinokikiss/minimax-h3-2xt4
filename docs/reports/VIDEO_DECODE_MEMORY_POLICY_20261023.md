# Video decode: separate memory policy and weight-free parent

The Video Query Parallel handoff now uses its own configurable **2-GiB VRAM
reserve** and a parent planner with **zero learned parameters**. Native public
tiling, temporal overlap, normalization, blending, output conversion and buffer
writes are retained. Trained parent weights are offloaded to CPU before worker
admission and remain available for Native fallback. The DiT reserve and all
attention, quantization and transport operators are unchanged.

This source has **local component qualification**, including saved long-sequence
sampler latents, rather than new physical-T4 performance qualification. The
previous physical E2E run used the old admission policy and completed through
Native fallback; it is not evidence that the revised policy executed on T4.

## Why the previous run declined

In [the completed 2×T4 E2E](H3_L1_LIFETIME_E2E_20261023.md), the parent first
loaded a complete decoder on GPU0. Both worker replicas then required:

`free >= decoder payload + 2 * Native activation estimate + DiT reserve`.

At 896×512/362 frames, this is approximately 2.282 + 1.538 + 9 = **12.820 GiB
free per rank**. A sample 73 ms before refusal shows GPU0 using **3.179 GiB**.
The inherited reserve and extra parent replica explain the likely GPU0 refusal,
although the old combined RAM/VRAM message did not record the exact individual
gate values. No Query Parallel raw decode started in that run.

## Implementation and compatibility

- `RuntimeConfig.video_decode_reserve_bytes` defaults to `2 * GiB`. The new
  optional loader widget `video_decode_reserve_gib` is appended after existing
  widgets, preserving older positional workflow values. Changing the policy
  participates in normal session configuration equality/recreation.
- Worker admission retains the existing two-times activation allowance, actual
  decoder payload and shared host staging lease. The example free-memory
  threshold becomes **5.820 GiB**. This is a policy threshold, not measured
  peak memory, and the 2-GiB reserve is not a universal OOM guarantee.
- `patches/video_decode_planner.py` derives a small planner from the frozen
  Native Video VAE class without constructing learned layers. It copies Native
  planner settings and normalization buffers, and retains only decoder output
  channel metadata. Learned raw decoding dispatches to the existing workers.
- A scoped swap replaces only this VAE's model, patcher and cached size. Native
  public `decode`/`decode_tiled` continues to load a real CoreModelPatcher, but
  that patcher contains no decoder weights. Global model management and frozen
  ComfyUI source are not patched.
- Before the swap, the trained parent patcher is unloaded. A parent that cannot
  offload declines before worker preparation. On admission denial, the original
  model/patcher/size is restored before Native fallback. Decode errors propagate
  after worker release and parent restoration; no fallback is attempted after
  partial collective decoding.
- Each rank records free/total VRAM, payload, activation estimate/allowance,
  dedicated reserve, required free bytes and host staging size. GPU refusal,
  host refusal and peer refusal are distinguished. Host refusal includes a
  shared-budget snapshot; GPU-skipped host admission is recorded as `null`.
- Native override, incompatible attention contracts, patched models and Audio
  VAE retain their prior routes. Audio remains single-GPU Native FP32.

The parent **CPU** decoder, its loading I/O and conservative host lease remain.
Retained TE/DiT caches or simultaneous construction leases can still prevent
host admission. Removing those costs is a separate follow-up, not an implicit
increase to the shared host budget in this change.

## Qualification

The local conda runtime and frozen ComfyUI are retained, with comfy-kitchen
0.2.37 and TileLang 0.1.13. The focused CPU suite passes **57 tests**, covering
independent reserve configuration, loader reuse, CPU/GPU/peer admission
refusal, lease rollback, parent restoration, no-parameter planning, public
tiling, legacy API checks, reference loading and Native-only audio.

The trained Video VAE check uses **two logical CUDA ranks on the local RTX
5060 Ti**. Only transport and stream-private benchmark scratch are substituted;
no TE, DiT sampler or Audio VAE is run. Both new worker preparation and actual
public Video decode are exercised.

| Input / check | Native comparison |
|---|---|
| Single frame, 261 query tokens | Byte-exact, finite |
| Batch of two single-frame inputs | Byte-exact, finite |
| Seven-token temporal input, 1,797 queries | Byte-exact, finite |
| 31 temporal tokens, 512×512, repeated tiles, 103 output frames | Byte-exact, finite |
| Public `decode_tiled` | Byte-exact, finite |
| Saved E2E video latent `[1,24,107,32,56]`, 896×512/362 output frames | Byte-exact, finite |
| Forced host refusal → Native fallback | Byte-exact |
| Injected decode failure → cleanup → next decode | Error propagated; recovered output byte-exact |

The benchmark checks that the parent planner has zero learned parameter bytes
and the trained parent has zero GPU parameter bytes at every `vae_prepare` and
`video_pixels` dispatch. Replica models are released after public calls, and
the final shared host reservation returns to zero. Source hashes are captured
before the checks and verified unchanged before accepting the result.

Raw evidence: `D:/attention_eval/video_decode_memory_20261023/local_final/`.
The [aggregate record](../data/video_decode_memory_policy_20261023.json) binds
source hashes, the 57-test log, saved input digest and returned local checks.

Physical-T4 admission, warmed timing and memory remain unmeasured for this new
source. The next hardware test should replay the saved Video latents with
retained-cache admission accounting; it should run only the decoder, with the
same frozen source/core pins, bounded execution and one final artifact retrieval.
