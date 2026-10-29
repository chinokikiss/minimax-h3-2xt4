# Regular Turbo 0.9 MP / ten-second retry — 2026-10-29

Kaggle accepted **version 1** of [H3 Turbo 0.9MP 10s T4 20261029](https://www.kaggle.com/code/pamcks/h3-turbo-0-9mp-10s-t4-20261029).
Final artifacts confirm **OOM after the first DiT denoiser call**, before the
first sampler step completed. No CPU smoke job or test suite was run for
this retry, following the user's explicit instruction.

## Collected failure

One final collection retrieved 26 artifacts, with source binding and download
hashes recorded. Offline baking completed in **346.690 s**. The full E2E
subprocess ran **492.064 s** before failure; reference/TE preparation completed.
The result records one finite FP16 video/audio denoiser output pair and zero
completed sampler steps. Decode was not reached and no final media was produced.

Unlike the 1 MP failure inside attention-output quantization, this failure was
in the parent sampler: `model_sampling.calculate_denoised`, at
`model_input - model_output * sigma`. It requested **24 MiB**, with **4.75 MiB**
free on GPU 0. The error reported a separate process holding **14.01 GiB**;
the failing process had about **554 MiB** including non-PyTorch allocations.
This identifies cross-process device pressure at sampler handoff, rather than
establishing that another DiT kernel failed or that allocator fragmentation
caused the failure. No allocator or residency-policy changes were made here.

The user selected a subsequent **1216×704 (0.856064 MP)** retry, skipping another
CPU smoke run. Raw evidence is retained under
`docs/data/h3_turbo_09mp_10s_20261029/gpu_v1_final/`.

## Workload

| Item | Retry |
|---|---|
| Canvas | **1280 × 704**, **901,120 pixels / 0.901120 MP** |
| Native generation | 243 frames, 24 FPS, 10.125 s |
| Export | First 240 frames and 320,000 samples at 32 kHz: 10.000 s |
| Native video latent | `[1,24,72,44,80]` |
| Generated-video tokens | **63,360**, versus 72,576 at 1 MP; actual joint packed length remains execution-derived |
| Sampling | Same eight-step regular Turbo, Euler / Simple, CFG 1 |
| Shifts / seed | Video 6, Audio 3; seed 20261020 |
| Inputs | Same anime prompt and ordered character/shrine references |
| Weights | Same immutable pre-sliced base and eight-step Turbo LoRA |
| Baking | One dual-GPU offline TP2 bank persisted to instance disk; no online LoRA or PTD |
| Runtime | Same qualified production operators, trusted-bake loading, FP16-fix, Temporal Video and Native single-GPU Audio VAE |

Only requested geometry and experiment labels change. Noise identities and
reference token geometry can change with the canvas; they are captured rather
than claimed equal to the failed run. No allocator tuning, lower-step fallback,
operator replacement or dependency update is included. This is an attempted
memory-fit retry, not evidence that 0.9 MP fits before execution completes.

The failed 1 MP run completed baking and TE, then OOM'd while rotating the
attention output for quantization in the first DiT call. The failed artifacts
remain separate and unchanged. The new instance must bake its own bank because
the earlier job's `/tmp` bank was not uploaded or persisted across instances.

## Qualified source reuse and explicit benchmark overlay

The retry fetches the **same CPU-qualified capsule**:
`21abe7dd91f284eed9626bb018c6abf66adf1c066b2f76662d72ce53d46d41a1`,
published at `d7c7528515d2a7961de5e6f961232d81d21cdc8b`. Its successful CPU
setup/API receipt is reused without changing or fabricating its capsule hash.
The normal exact-pinned GPU bootstrap remains; it does not execute a new CPU
smoke suite.

After that setup, the experiment installs **one benchmark-only file overlay**:
`benchmarks/turbo_challenge_contract.py`. It permits the two explicitly
authorized canvases and validates decoded output against the selected canvas.
The historical contract-family identifier `turbo_1mp_10s` stays compatible with
the qualified runner; the actual request name, dimensions, output path and
Kaggle title identify the 0.9 MP retry.

Before applying the overlay, the worker hashes every original capsule source
file against its qualified manifest. It then verifies and installs the exact
content-hashed control module from commit
`938dc7d5746256058433a1eb0e72a5e539874645`. It writes `source_overlay.json`
containing the original/new hashes and the complete effective source map.
The altered request is written separately; production patch files, Native
ComfyUI, dependencies and numeric kernels remain byte-exact. Syntax/configuration
and embedded-payload hash checks are preparation checks, not a CPU smoke run.

The notebook is private, has zero dataset/output mounts and contains no
credentials. It reuses the exact original character bytes and immutable shrine
mirror. The bounded single GPU submission retains the 7,200-second execution
watchdog and 8,100-second provider limit. No status/live-log polling or automatic
duplicate job is configured.

Kaggle canonicalized the requested `09mp` slug to `0-9mp`; the receipt records
the acknowledged canonical URL for future artifact collection. This did not
cause a second submission. Future builder metadata uses the canonical spelling.

Local evidence: `D:\attention_eval\h3_turbo_09mp_10s_20261029\`.
Public receipts: `docs/data/h3_turbo_09mp_10s_20261029/`.
