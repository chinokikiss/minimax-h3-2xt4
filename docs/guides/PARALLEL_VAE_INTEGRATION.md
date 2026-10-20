# Default parallel VAE inference

The TP2 ComfyUI workflow now defaults to **Native-aligned Query Partitioning
for video** and **FP32 Channel Parallel for audio**. TE/DiT, conditioning,
samplers and reference encoding keep their existing implementations. Independent
cropped video slabs are not enabled.

## Use

Open `workflows/video_minimax_h3_r2v_tp2_int8.json` after restarting ComfyUI with
the linked `custom_nodes/minimax_h3_tp2` directory. Both existing VAE loaders
are now `MiniMaxH3ParallelVAELoader` nodes, connected to the same persistent
TP2 session. Select the INT8 video and FP32 audio files in the VAE directory;
the decoder ranks read the exact same path that constructed the Native VAE.
The loader's default `decode_mode` is `auto`. Set either loader to `native`
to retain strict Native decoding for that model.

`workflows/video_minimax_h3_r2v_native_int8.json` remains the separate fully
Native fallback. Physical execution uses Linux, two CUDA GPUs and NCCL;
the existing single-GPU local qualification harness does not provide physical
dual-GPU performance evidence.

## Dispatch and lifetimes

- `patches/parallel_vae.py` extracts the physically qualified aligned video
  algorithm into inference code: Native prefix projection once, contiguous
  query ownership aligned to 128-token scale groups, global RoPE coordinates,
  global KV per layer, and a final projected-output join. All 36 trained blocks
  participate. Very short sequences that cannot form two aligned partitions
  execute Native raw decoding on rank zero.
- The parent calls the original public ComfyUI VAE. Video replaces only
  `_decode_pixels`, preserving normalization, temporal chunking, spatial tiles,
  blending, trimming, finalization and output buffers. Audio replaces only the
  model's `decode`; public layout/output conversion remains Native. Encoding
  methods and reference latents are untouched.
- Existing persistent rank processes perform these operations, with NCCL KV
  AllGather/broadcast for video and a waveform AllGather for audio. There is
  no second worker service or benchmark-module import in inference.
- Decoder-only checkpoint loading excludes unused encoder parameters. Shared
  HostBudget covers checkpoint staging and construction. Parent weights are
  loaded before the VRAM admission check so their cost is already visible.
  Both ranks agree on admission with a MIN collective before starting decode.
  Construction occurs outside inference_mode: a locally reproduced
  Module.to behavior otherwise stripped the QuantizedTensor wrapper and
  changed Native operator dispatch. Decode executes in inference_mode.
- VAE replicas stay resident across the raw calls within one public decode,
  then are fenced and released. The temporary parent hook is restored even
  on failure. RAM/VRAM admission denial uses Native before raw collectives;
  failures after decoding begins propagate through existing worker abort
  handling. No partial output is silently accepted.
- The session lock spans preparation through output completion and cleanup.
  Parent TF32/cuDNN flags are mirrored for this stage and restored in the
  workers afterward. CPU-only next-request TE lookahead starts during public
  decoding; completion notification follows replica release.

The parent retains its Native VAE weights while workers hold decoder replicas.
This deliberate integration cost preserves Native model management and reference
encoding. Admission is conservative: weight payload plus twice the Native decode
memory estimate plus the existing reserve. Under memory pressure, `auto` can use
Native instead. This is not a proof of OOM freedom for arbitrary external
allocations. Allocating another model concurrently is outside this qualification.

## Completed local qualification

`benchmarks/qualify_parallel_vae_pipeline.py` exercised trained checkpoints on
the 5060 Ti with two logical ranks, the actual RankRuntime VAE dispatch,
bounded empty host caches and full Native public ComfyUI decode.

| Check | Result |
| --- | --- |
| Default workflow graph | Both VAE loaders use auto and share the TP2 session; graph links validated |
| Very short video query sequence | Bit-exact public pixels through the Native raw control |
| Video single frame, 256px | Bit-exact public pixels |
| Video batch 2, 256px | Bit-exact public pixels |
| Video seven-token temporal input | Bit-exact public pixels; 22 output frames |
| Video 31-token input, 512x512 | Bit-exact public pixels; 103 output frames; Native repeated chunks and spatial tile batching |
| Video explicit Native override | Bit-exact; no parallel preparation |
| Public decode_tiled entry point | Bit-exact public pixels |
| Keyword decode/decode_tiled calls | Bit-exact public pixels |
| One-rank host admission denial | Coordinated Native fallback; no raw collective decode |
| Injected raw-decode failure | Error propagated; hook restored; both replicas released |
| Next decode after that failure | Bit-exact public pixels |
| Audio 5/10/20 seconds | Exact against sequential mono; relative L2 1.280/1.237/1.205e-5 against Native |
| Audio Native override | Bit-exact |

The suite completes with zero shared host reservations. Peak managed
reservation is 10,337,744,256 bytes, below its 16 GiB limit. This reservation
does not include the parent VAE, outputs or total process RSS. No TE/DiT model
or speculative startup weights were loaded in this focused suite.

Audio remains an **explicitly accepted approximation** under the user's
instruction to enable Channel Parallel after reviewing its rounding cause.
The previous 1e-5 correctness gate is not changed: all three audio comparisons
still fail it, and the JSON records `original_strict_gate_pass: false`.
No claim of Native bit-exactness or perceptual inaudibility is made.

Evidence: `semantic_results/parallel_vae_pipeline.json` and its log, including
tested source hashes. Previous raw-decoder physical evidence remains in
[the dual-T4 video report](../reports/VAE_KV_T4_RESULTS.md); the audio explanation is in
[the channel error diagnosis](../reports/AUDIO_CHANNEL_ERROR_DIAGNOSIS.md).

The new integrated path has **not yet been qualified or timed on physical
T4s**, nor tested on a complete HTTP-queued sampler-produced Ref2VA request.
The suite uses fixture latents and empty TE/DiT caches, not an integrated
near-maximum-memory residency audit. CPU parent/worker packets, repeated decoder
loading, retained parent weights, memory admission and next-request cache
contention may reduce the existing component speed advantage. No new
end-to-end speedup is claimed.
