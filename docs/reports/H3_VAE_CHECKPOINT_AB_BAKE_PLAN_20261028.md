# Video VAE checkpoint replay and offline LoRA bake profile

Status: prepared for CPU delivery qualification, then physical 2×T4 execution.
No new T4 results or visual-quality conclusion are available yet.

## Scope and controls

Replay the user-specified `video_latents.pt` from the 2026-10-26 Temporal
experiment. This is the older 362-frame, 960×544 workload at 24 FPS, **not** the
243-frame anime generation. The `[1,24,107,34,60]` FP32 latent tensor is finite;
its SHA256 is `8fb3296f9dfc00641698e1a1aa26acb7f0fa98675b89b597dc95f10e64a9fa64`.
The original file SHA256 is
`08a10be3efdda85ea6415c832eb2654101da5d5fabe54b6e34002071ee2dde25`.

| Case | Checkpoint in Comfy-Org/MiniMax-H3 | Whole file bytes |
|---|---|---:|
| INT8 ConvRot | vae/minimax_h3_video_vae_int8_convrot.safetensors | 2,811,065,184 |
| FP16 | vae/minimax_h3_video_vae_fp16.safetensors | 5,207,808,496 |

Both files are bound to immutable revision
`e5eb578a89295337b8ff433a035929ce0279e0b6`, with full file hashes in the
experiment manifest. Both use the frozen Native decoder, FP16-fix tensor flow,
Temporal clip pairing and identical native normalization, blending and export
settings. Native attention dispatch is preserved and observed on the constructed
weights. INT8 and FP16 checkpoints can select different attention paths, so this
is a **checkpoint-level comparison**, not an isolated weight-bit-width ablation.
Worker tile input shapes, successful calls and OOM attempts are recorded; any
geometry difference must be disclosed with the visual results.

Only one checkpoint's replicas are resident at a time. The benchmark's host
admission budget is 24 GiB for both cases, allowing the larger FP16 replicas;
the production default is unchanged. A declined parallel admission or fallback
fails the case rather than being reported as a successful Temporal replay.

Deliver two clearly labelled MP4s (libx264 CRF 12, 24 FPS) and matched lossless
PNG frames at 0, 48, 96, 144, 192, 240, 288, 336 and 361. All frames must be
finite and the output shape must be `[1,362,544,960,3]`. User visual assessment
remains the quality criterion. No TE/DiT sampling or Audio VAE execution occurs.
Decode wall time includes first-load/compilation work; worker CUDA spans and
prepare events are retained separately. This is not a steady-state speed test.

## Offline baking profile

Use the public pre-sliced TP2 base and the same 8-step Turbo adapter used in
the completed anime run: `lightx2v/Minimax-h3-Turbo`, revision
`30f485ed874aa44c8183348ad361c5eabad318da`,
`minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`, strength 1.0.
Assign alternating whole layers to the two GPU worker processes, preserving
the production disk baker and its independent Native verification.

The observer records nested wall-clock and thread CPU spans for:

- Source reads, input shard validation, file and tensor hashing.
- Production LoRA merging, dequantization, Native LoRA application and ConvRot requantization.
- Independent Native numerical verification.
- TP2 splitting, safetensors serialization, fsync and atomic JSON publication.
- Parent coordination and worker totals on a shared monotonic timeline.

CUDA events measure inclusive spans, including dependencies and GPU idle time;
they are not summed kernel execution times. Nested CPU exclusive wall time
subtracts instrumented child scopes. Worker processes overlap, so their totals
must not be added to produce the end-to-end critical path. Thread CPU time
does not measure all native library threads. Profiling overhead is included;
this run does not establish a speedup over the earlier 434.87-second bake.

A minimal kernel probe checks CUPTI capture, explicit export/flush and worker
process exit before bake attribution. Retain one first-production-merge CUDA
trace per worker and phase events for the entire bake. Missing kernel events
are a profiling blocker, not evidence of GPU idleness. Require all 100 shard
tensor-state hashes and the baked auxiliary hash to match the prior physical
T4 bake, with all 208 targets independently verified.

## Delivery and reproducibility

Extend the last T4-qualified source distribution
`1973087713b2627b03135908f0d118485fb12bfa61918238cfe99139e0354de3`.
All production patch bytes, frozen ComfyUI source and approved wheel pins are
preserved. Only benchmark helpers and immutable manifests are overlaid. Local
checks use the existing conda `comfyui` interpreter; they provide CPU contract
evidence only. Kaggle retains its native Torch family and exact core pins,
including Kitchen 0.2.37 and TileLang 0.1.13.

The exact 20 MB latent file was submitted as a **private** Kaggle Dataset,
`pamcks/h3-vae-ab-latents-20261028`. No HF credential is embedded in either
notebook. Source, public checkpoints and approved packages use immutable
GitHub/HF revisions and hashed package delivery. Large model datasets and
mounted previous notebook outputs are unnecessary.

`benchmarks/submit_vae_bake.py` creates one acknowledged submission per phase.
The CPU-only notebook qualifies dependencies, headers, observer transparency
and the mounted latent hashes. The GPU notebook requires both matching CPU
receipts, checks two distinct physical SM75 T4 GPUs, and runs the two decoder
cases followed by one fresh persisted bake. Setup/execution deadlines are
600/4800 seconds, within a 5400-second GPU notebook bound. Preserve partial
videos, traces, logs and receipts on failure. No status/list/live-log polling or
automatic duplicate resubmission is used; collect final artifacts on request.
