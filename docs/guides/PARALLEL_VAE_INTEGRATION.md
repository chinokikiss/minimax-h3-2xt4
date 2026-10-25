# Video query parallelism and Native single-GPU Audio VAE

The TP2 ComfyUI workflow now defaults to **Native-aligned Query Partitioning
for video** and **Native single-GPU FP32 decoding for audio**. TE/DiT, conditioning,
samplers and reference encoding keep their existing implementations. Independent
cropped video slabs are not enabled.

## Use

Open `workflows/video_minimax_h3_r2v_tp2_int8.json` after restarting ComfyUI with
the linked `custom_nodes/minimax_h3_tp2` directory. Both existing VAE loaders
are now `MiniMaxH3ParallelVAELoader` nodes, connected to the same persistent
TP2 session. Select the INT8 video and FP32 audio files in the VAE directory;
the video decoder ranks read the exact same path that constructed the Native VAE.
Video defaults to `decode_mode=auto`; `native` disables video query partitioning.
Audio is always Native single-GPU, including older workflows saved with `auto`.
The shared loader node name is retained for workflow compatibility. Both audio
channels stay together in the original Native decode call; channel, temporal
and hybrid audio parallelism are no longer runtime options.

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
- The parent calls the original public ComfyUI VAE through a temporary,
  weight-free Native planner. Its `_decode_pixels` dispatches to workers,
  preserving normalization, temporal chunking, spatial tiles,
  blending, trimming, finalization and output buffers. Audio does not replace
  model methods; public layout, chunking and output conversion stay Native. Encoding
  methods and reference latents are untouched.
- Existing persistent rank processes perform these operations, with NCCL KV
  AllGather/broadcast for video. No audio tensors are sent to workers. There is
  no second worker service or benchmark-module import in inference.
- Decoder-only checkpoint loading excludes unused encoder parameters. Shared
  HostBudget covers checkpoint staging and construction. Trained parent weights
  are offloaded before admission and remain on CPU for Native fallback. Only
  planner metadata and copied normalization buffers can load on the parent GPU.
  Both ranks agree on admission with a MIN collective before starting decode.
  Construction occurs outside inference_mode: a locally reproduced
  Module.to behavior otherwise stripped the QuantizedTensor wrapper and
  changed Native operator dispatch. Decode executes in inference_mode.
- VAE replicas stay resident across the raw calls within one public decode,
  then are fenced and released. The original parent model/patcher/size is restored even
  on failure. RAM/VRAM admission denial uses Native before raw collectives;
  failures after decoding begins propagate through existing worker abort
  handling. No partial output is silently accepted.
- The session lock spans preparation through output completion and cleanup.
  Parent TF32/cuDNN flags are mirrored for this stage and restored in the
  workers afterward. CPU-only next-request TE lookahead starts during public
  decoding; completion notification follows replica release.

Video admission uses weight payload plus twice the Native decode memory estimate
plus **`video_decode_reserve_bytes`**, default **2 GiB**. The loader appends the
optional `video_decode_reserve_gib` widget; older saved widget positions and the
DiT `reserve_bytes` setting are preserved. The former parent GPU decoder copy
is absent during Query Parallel execution. Its CPU weights and host lease are
retained, so this change does not remove the parent CPU loading cost.

Each rank records free/total VRAM, decoder payload, activation estimate,
decode reserve, required headroom and host admission. Host refusal, local VRAM
refusal and peer refusal have distinct reasons. A parent that cannot offload
uses Native before worker preparation. Under memory pressure, `auto` can use
Native instead. This is not a proof of OOM freedom for arbitrary external
allocations. Allocating another model concurrently is outside this qualification.

The [2026-10-23 memory-policy qualification](../reports/VIDEO_DECODE_MEMORY_POLICY_20261023.md)
records current local trained checks and the separate physical-T4 validation boundary.

Audio keeps the staged encoder-first/decoder-last loader, Native FP32 dtype and
CPU next-request TE lookahead notifications. Worker `vae_prepare` rejects audio
before allocation or collectives, and the obsolete `audio_channels` command is
rejected. Historical audio parallel helper modules/results are retained solely
for old benchmark provenance; the runtime does not import them.

## Current audio regression check — 2026-10-21

The trained FP32 Audio VAE was checked locally with identical seeded stereo
latents `[1,32,2,16]`. Both `auto` and `native` return exactly the same finite
FP32 waveform `[1,12800,2]` as Native single-GPU decoding (maximum absolute
difference 0). Only `decode_start`/`decode_complete` scheduling notifications
are sent; no audio weights/latents or decoder replicas go to TP workers.
Historical audio parallel helper modules were not imported.

The focused regression suite passes 85 tests, including public stereo/tiling
delegation, staged decoder preparation and rejection of obsolete audio worker
commands. The existing Video query gate passes 36 Native / 72 logical-parallel
attention calls. This is local component validation, not a full H3 run or a
performance measurement. [Evidence](../data/audio_native_only_20261021.json).
Raw files/source snapshots: `D:\attention_eval\audio_native_only_20261021`.

## Historical local qualification before audio parallelism was removed

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

Audio was an **explicitly accepted approximation** under the earlier
instruction to enable Channel Parallel after reviewing its rounding cause.
The previous 1e-5 correctness gate is not changed: all three audio comparisons
still fail it, and the JSON records `original_strict_gate_pass: false`.
No claim of Native bit-exactness or perceptual inaudibility is made.
These historical channel-parallel results do not describe the current Native
audio path. The current regression gate requires exact Native output and no
audio-worker preparation/dispatch, rather than accepting a rounding tolerance.

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
