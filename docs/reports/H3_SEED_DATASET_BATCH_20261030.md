# Seed repeat and dataset I/O studies — 2026-10-30

Kaggle accepted **three version-1 submissions**: exactly one GPU notebook and
two CPU notebooks. **Execution results are pending.** No status/list/live-log
polling, automatic resubmission or additional H3 CPU smoke job is configured.

| Notebook | Scope | Provider execution limit |
|---|---|---:|
| [H3 Turbo Seed 20261021 T4 20261030](https://www.kaggle.com/code/pamcks/h3-turbo-seed-20261021-t4-20261030) | Same physical two-T4 E2E; change only sampling seed | 8,100 s |
| [H3 Dataset Read Sweep CPU 20261030](https://www.kaggle.com/code/pamcks/h3-dataset-read-sweep-cpu-20261030) | Six mounted public binary datasets; concurrency/aggregation/cache/memory measurements | 1,260 s |
| [H3 Dataset Upload Study CPU 20261030](https://www.kaggle.com/code/pamcks/h3-dataset-upload-study-cpu-20261030) | Native CLI/API/KaggleHub capability and small private synthetic uploads | 840 s |

## GPU: change the seed, retain the operators

The baseline is the [completed 0.856 MP, ten-second E2E](H3_TURBO_085MP_10S_E2E_20261030.md),
whose user review reported semantic/temporal defects. This new run uses
**seed 20261021 instead of 20261020**. It retains 1216×704, native 243 frames /
240 exported frames at 24 FPS, eight-step regular Turbo, Euler/Simple, CFG 1,
Video Shift 6 and Audio Shift 3, the exact prompt and two references, immutable
base/adapter revisions and persisted offline TP2 baking. TE remains Kitchen
causal INT8; DiT and both decoder strategies are unchanged.

This tests whether the observed defects recur with another noise sample.
It does **not** isolate TE causality or qualify the model's distribution from
two videos. Future output collection should compare conditioning hashes and
confirm that noise hashes changed. Review umbrella persistence, animal identity
and entrance, vertical motion, reference preservation and behavior after two
seconds, rather than treating finite tensors as a quality pass.

The exact previously CPU-qualified runtime/core capsule is reused:
`21abe7dd91f284eed9626bb018c6abf66adf1c066b2f76662d72ce53d46d41a1`.
The benchmark-only contract explicitly allows the new seed; bounded HF download
retry handling is retained. Both overlays are independently hash-bound to source
commit `4f1f599eea823bafc5aa10e738974d5e9b036580`. The old CPU receipt is not
relabeled as qualification of changed runtime code. Production runtime,
operators, frozen ComfyUI and Kitchen 0.2.37 / TileLang 0.1.13 remain unchanged;
Kaggle's matching native Torch family is recorded at execution.

Private character-image bytes remain only in the private notebook; its full
notebook is excluded from public receipts. The GPU job has zero dataset/output
mounts and no embedded credentials. A new instance rebakes its bank; no previous
instance's `/tmp` bank is assumed to survive.

## CPU read sweep: more datasets versus more readers

Six distinct public datasets are deliberately mounted for this CPU-only proxy:
[512](https://www.kaggle.com/datasets/cdeotte/melanoma-512x512),
[768](https://www.kaggle.com/datasets/cdeotte/melanoma-768x768),
[1024](https://www.kaggle.com/datasets/cdeotte/melanoma-1024x1024),
[384](https://www.kaggle.com/datasets/cdeotte/melanoma-384x384),
[256](https://www.kaggle.com/datasets/cdeotte/melanoma-256x256), and
[128](https://www.kaggle.com/datasets/cdeotte/melanoma-128x128).
They are treated as opaque TFRecord byte streams; images are not decoded or
analyzed. No medical inference is performed and no dataset payload is republished.

| Control | Implementation |
|---|---|
| Dataset aggregation | Prefix cohorts of **1, 2, 4 and 6 datasets**, plus individual single-reader controls for the other five datasets |
| Reader concurrency | **1, 2, 4, 8, 16, 32, 64** total readers, not that count per dataset |
| Comparable work | Same **12 s / 4 GiB** maximum per trial, stopping at the first limit or corpus exhaustion |
| Repeat/order control | Two main measurements per cohort/concurrency, deterministic randomized order; report per-concurrency medians |
| Work assignment | Unique extents within each trial; readers rotate across dataset queues; record per-dataset bytes to expose unequal consumption |
| Data memory | One reusable **1 MiB** `readinto` buffer per worker: **64 MiB maximum payload buffers**, no payload-bearing futures or whole-file reads |
| Memory guard | Stop if process RSS rises more than **256 MiB** above the trial baseline; record RSS and cgroup current/stat/pressure |
| Cache controls | Advisory `DONTNEED` after consumed extents; separate normal-cache controls at 1 and 64 readers |
| Provenance | File sizes and first-4-KiB hashes, filesystem device/mount information and CPU quota |

The reader uses mounted filesystem I/O, so its 64-thread sweep does not issue
repeated Kaggle API downloads. It studies whether aggregate **mounted read**
throughput increases when more datasets are active. It is not a measurement of
CLI/CDN network-download throughput or T4 storage. Mount preparation happens
before the cell and is outside the worker timer.

Six dataset handles do not imply six physical storage devices or independent
rate limits. They may share backend/cache resources. DONTNEED is advisory;
neither first-pass reads nor the hint establishes physically cold storage.
Repeated access can warm remote, filesystem and host caches. The payload/RSS
guard bounds the reader's own memory, but kernel file-cache usage still needs
the recorded cgroup counters. Different corpus sizes, short trials and shared
host load limit extrapolation to a universal upper bound or model-loading speed.

The worker has a 1,150 s watchdog and a 1,020 s trial-scheduling deadline.
Source installation, H3 imports and GPU allocation are absent.

## CPU upload study: native clients and authentication

The study records the actually installed Kaggle/KaggleHub/SDK versions and
function signatures without upgrading or installing clients. It examines:

1. **Kaggle CLI:** one `datasets create` request, using its private default.
2. **Kaggle Python API:** one `dataset_create_new(..., public=False)` request.
3. **KaggleHub:** one version upload to a private dataset whose CLI/API creation
   was acknowledged in this run; no unverified-default public creation.

The official [CLI dataset documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md)
describes private-by-default creation, and the
[KaggleHub documentation](https://github.com/Kaggle/kagglehub#upload-dataset)
describes dataset and version upload. These references describe capabilities;
the notebook records which installed APIs and credentials actually work.

Each operation uses a **16 MiB synthetic random file**, generated directly on
disk in 1 MiB increments. No H3 weights, activations or previous media are
uploaded. Requests are time-bounded to 180 s and child RSS is guarded at 512 MiB.
The small payload is an API/authentication/memory-behavior probe, not a large
bake-upload throughput estimate. Created private probe datasets are retained;
no automatic deletion, status checks or processing-state polling is attempted.

Authentication uses only the notebook's native environment or already attached
`KAGGLE_API_TOKEN` / username-key Secrets. **Local Kaggle credentials are not
embedded or transferred.** Missing authentication, unavailable native clients
or insufficient permissions are explicit blockers. A request acknowledgement
is not proof that processing completed or that round-trip bytes were verified;
no success is claimed before final evidence is collected.

## Preparation, evidence and collection

Local checks exercised seed allowlisting and small-file read accounting, buffer
bounds and multi-dataset byte totals. They performed no model/GPU/network
execution. Static preparation verified all three scripts/notebooks, source
hashes, GPU-versus-CPU flags, private metadata, six read mounts and no embedded
credential values. Each submission has an exclusive durable receipt, accepted
version and provider URL. Only one GPU and two CPU submissions were made.

Sources: `benchmarks/build_turbo_09mp_retry.py`,
`benchmarks/build_dataset_cpu_studies.py`,
`benchmarks/kaggle_dataset_read_sweep.py`, and
`benchmarks/kaggle_dataset_upload_study.py`.

Public receipts:
[`docs/data/h3_seed_dataset_batch_20261030/`](../data/h3_seed_dataset_batch_20261030/).
Private local preparation and final-collection directories:
`D:\attention_eval\h3_seed_dataset_batch_20261030\`.
No automatic status check or final-output collection has been scheduled.
