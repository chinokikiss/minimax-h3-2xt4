# DiT pair: download failure and isolated recovery

The first `h3-dit-pair-59134-t4-20261028` job failed during public weight
staging, before offline baking or either denoiser call. It is not a DiT kernel
failure. Its final artifacts were retrieved once on 2026-10-28.

## Evidence

The preserved `dit_pair/batch_results.json` records two distinct Tesla T4
devices (SM75), Torch 2.11.0+cu128, Kitchen 0.2.37 and TileLang 0.1.13.
The minimal worker CUDA capture/flush/exit gate passed in 12.745 seconds.
Hugging Face's public CDN then returned HTTP 503 while retrieving
`dit.block00.rank0.safetensors`. No completed denoiser timing or tensor
lifetime attribution can be inferred from this job.

The saved Nsight SQLite system metadata records driver 580.178.04. Do not
publish the original system environment table: Nsight can capture inherited
platform credentials. Only sanitized fields belong in reports.

## Recovery

The benchmark downloader now makes at most three attempts per transient
transport failure, waiting 2 and 5 seconds. Recoverable HTTP codes are
408/429/500/502/503/504; network timeouts/disconnects are also recoverable.
A retry refreshes HF download metadata. Authentication failures, malformed
ranges, incorrect lengths and checksum failures remain fatal. Successful
objects retain their immutable revision, size and SHA-256 checks. Auxiliary
range downloads use the same bounded recovery.

Three CPU regression tests reproduce the observed 503, exhausted recovery,
permanent failure and corrupt-object rejection. Together with the four pair
contract/trace tests, all seven pass against the frozen local CPU source.

The retry retains native Torch 2.11.0+cu128 / CUDA 12.8, the original two-call
59134-token fixture, offline persisted LoRA, the capture gate and strict
trace/lifetime acceptance rules. New CPU qualification and submission receipts
are separate from the failed job. Production patches are byte-preserved.

## Separate VAE investigation

CUDA 13 is an independent experiment for native VAE Kitchen linear dispatch.
The frozen ComfyUI CUDA guard disables that registry backend under CUDA 12.8;
DiT's custom GEMM path does not use that registry. Do not attribute the failed
download to the guard, or change the DiT runtime to test the VAE hypothesis.
The VAE experiment must cross linear and attention backends independently,
and qualify its isolated CUDA 13 stack on CPU before GPU submission.

Local raw evidence: `D:/attention_eval/h3_dit_pair_59134_20261028/gpu_outputs`.
Recovery evidence: `D:/attention_eval/h3_dit_pair_retry_20261028`.
