# VAE INT8 versus FP16: isolated CUDA 13 attribution

## Two separate investigations

The DiT pair failed on an HF CDN HTTP 503 before inference. Its retry retains
the validated native CUDA 12.8 stack. CUDA 13 is exclusively a VAE experiment
to investigate the earlier INT8 ConvRot versus FP16 decoder timing gap.

The previous two-T4 checkpoint replay measured 246.577 seconds for INT8 and
142.857 seconds for FP16. It changed checkpoint and Native attention dispatch
together, ran once per checkpoint, and included loading. It cannot identify
the cost of attention or linear layers independently.

The frozen `comfy/quant_ops.py` disables Kitchen's CUDA registry backend when
`torch.version.cuda < 13`. The prior job used Torch 2.11.0+cu128. Its saved
Nsight metadata records driver 580.178.04. The guard affects registry operations,
including linears and auxiliary fusion; it does not imply attention itself
falls back, and does not govern DiT's direct custom GEMMs.

## Environment change and gates

Use official Torch 2.11.0+cu130, torchvision 0.26.0+cu130 and torchaudio
2.11.0+cu130 in a separate `/kaggle/temp/h3_vae_cu130_overlay`. All CUDA runtime
dependencies are enumerated in a versioned, SHA-256 locked wheel manifest.
No dependency resolver runs during installation. Package sources are the
official PyTorch wheel index and PyPI. The Kaggle base environment and local
conda environment remain untouched.

Kitchen 0.2.37, TileLang 0.1.13 and all frozen ComfyUI/production patch bytes
remain unchanged. The validated system nvcc 12.8 compiler is retained; this
experiment upgrades Torch's runtime, not the system toolkit or driver. No
new TileLang kernel is compiled for the VAE attribution.

CPU qualification installs the hashed stack and checks actual imports,
signatures, core/source identity, compiled SM75 architecture flags, replay
contracts and the private saved latent's hash. Only after this succeeds may
the T4 job be submitted. Before large checkpoint downloads, the T4 job checks
two distinct physical SM75 devices, driver >=580, strict CUDA INT8 linear and
Kitchen attention execution, finite outputs, and nonempty CUPTI traces on
both devices. Unsupported wheels, APIs, drivers or capture failures stop the
experiment and are reported as blockers; the frozen guard is never bypassed.

## Controlled matrix

For the same INT8 ConvRot weights and captured block input:

| Linear implementation | Attention implementation | Purpose |
| --- | --- | --- |
| Strict eager | Native Torch SDPA | Common counterfactual |
| Strict CUDA | Native Torch SDPA | Isolate linear dispatch |
| Strict eager | Kitchen INT8 | Isolate attention dispatch |
| Strict CUDA | Kitchen INT8 | Combined deployment candidate |

A fifth case uses the FP16 checkpoint with Native attention and the identical
captured input as a performance anchor. It is not a precision ground truth.
Other registry operations retain the same automatic CUDA 13 dispatch across
all factorial cases. Consequently, within-runtime deltas isolate linears and
attention, but do not measure every auxiliary operation changed by the CUDA
12.8-to-13 guard. Do not extrapolate block speedups to full decode wall time.

Explicit `registry.get_implementation(..., backend=...)` validates constraints
and rejects unsupported calls instead of silently falling back. Receipts name
the actual selected implementation. Contexts restore attention flags and
module references on success or failure; production defaults are untouched.

## Inputs, shapes and measurement

Reuse the private Dataset `pamcks/h3-vae-ab-latents-20261028`, containing the
exact user-specified saved tensor `[1,24,107,34,60]`. No HF token or new large
Dataset is required. Both checkpoints are pinned to the same previously
validated Comfy-Org revision and full file hashes.

Native normalization, temporal chunk geometry and spatial indices select
actual first-clip tiles. The prior decoder used tile batches `[4,24,7,16,16]`
and `[1,24,7,16,16]`. Replay these on GPU0 and GPU1 respectively, independently;
this is not a new Temporal-versus-Query throughput measurement.

One real forward per tile batch captures pre-block inputs and RoPE at layers
0, 17 and 35. Every case reuses these inputs. Token length is 1797 including
five suffix tokens; width 2048; Q/K/V `[B,32,1797,64]`; QKV output width 6144;
FC1 output width 16384 and post-SwiGLU FC2 input width 8192.

Each case has three warmup calls, seven repeated CUDA-event/wall measurements,
and one separate CPU/CUDA profiler call. Case order rotates by layer/batch.
Loading, capture, cold/cache transition and profiler overhead remain separate
from warm measurements. Shared input hashes detect accidental mutation.

Traces name QKV, attention output, FC1, FC2 and attention call sites, retain
individual GPU kernel names/counts/work durations, and reject empty CUDA
captures. Inclusive parent/child durations overlap and must not be summed.
Peak allocated VRAM is recorded per case. All outputs must remain finite.

Counterfactual reports calculate linear savings at fixed attention, attention
savings at fixed linears, their interaction, and the combined INT8/FP16 block
latency ratio. A final claim requires these measurements and corresponding
kernel evidence. No full video generation, semantic-quality verdict, GPU
occupancy estimate or memory-stall counter claim is produced by this replay.

## Reproduction and current status

`lock_vae_cuda13.py` resolves official wheel metadata without installing
anything locally. `build_vae_cuda13_delivery.py` extends the immutable source;
`build_vae_cuda13_job.py --mode cpu` qualifies it. GPU mode requires both the
matching native/core CPU receipt and the matching CUDA13 CPU receipt.
Submissions and final collection are one-shot, without status/log polling.

Local harness tests pass. Linux CUDA13 installation/import qualification and
physical T4 results remain pending until the corresponding receipts exist.

Evidence root: `D:/attention_eval/h3_vae_cuda13_20261028`.

Sources: [official Torch version matrix](https://pytorch.org/get-started/previous-versions/),
[NVIDIA CUDA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
