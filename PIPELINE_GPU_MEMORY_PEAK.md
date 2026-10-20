# H3 full-pipeline GPU memory peak

The highest observed device-memory use is in **DiT denoising / transformer
execution**, not TE, video VAE or audio VAE, for the existing full-request dual-T4
captures. This is a reanalysis of physical evidence, with no inference changes
or additional GPU job.

Scope: 512x512, 124 frames, 50 TE/DiT layers, four actual NFEs, Native reference
encoding, Native-aligned query video VAE and Channel Parallel audio VAE. Eighteen
captures contain 55 measured requests: 14 timing captures/41 requests, three
Kineto captures/12 requests, and one separately labelled diagnostic capture/two
requests. These historical full-request runs precede the current Online LoRA
and operator experiments. Their peaks must not be presented as a fresh complete
pipeline measurement of those changes.

## Stage location: unprofiled device usage

Representative **Offline+P1+P2 timing capture**, maxima across its four measured
requests. NVML board-used memory includes all processes, allocator cache,
libraries and communication state; it is not live activation memory.

| Full-request stage | GPU 0 peak, GiB | GPU 1 peak, GiB |
|---|---:|---:|
| Reference encoding + TE | 4.845 | 4.724 |
| **DiT sampler, all four NFEs** | **14.212** | **13.845** |
| Video VAE decode | 6.398 | 3.341 |
| Audio VAE decode | 4.523 | 1.200 |

Across all 14 unprofiled captures, the corresponding stage maxima are
6.642/6.331 GiB for reference+TE (including optional idle residency),
**14.214/13.847 GiB for DiT**, 6.400/3.341 GiB for video decode, and
4.921/1.200 GiB for audio decode. These maxima come from different requests;
they are not simultaneous and must not be added.

Source: `kaggle_causal_resume_output/causal_audit_results/continuous/offline_p1_p2/timing/result.json`,
`host_monitor.samples` correlated with `requests[].stages` by monotonic time.
The aggregate also includes continuous P0/P1 residency, TE critical-path and
idle-residency timing captures. Cold/warmup and between-request samples are
retained separately in the derived evidence; the table excludes them.

## Correlated block locations

The matching Kineto capture reaches 14.237/13.857 GiB board use. The first sample
at each GPU's maximum falls inside these GPU stream intervals:

| GPU | Request | Request-relative sample time | Active DiT interval | Monotonic interval, ns |
|---|---:|---:|---|---|
| 0 | 0 | 241.064857 s | NFE 3, block 0 | 10879420444637–10880498694637 |
| 1 | 0 | 182.789463 s | NFE 1, block 47 | 10820391369219–10821953556719 |

NFE numbering is one-based; block numbering is zero-based. These locate
**samples on the high-residency plateau**, not unique peak-generating blocks or
allocation sites. Allocator reservations can persist after the allocating
operator. CPU preparation/enqueue runs ahead: rank1's preceding CPU-ready
observation already names block49 while GPU computation is in block47. GPU
intervals, not the latest CPU label, are used above.

Evidence: matching `kineto/result.json` and
`kineto/batch_r{0,1}_causal.json.gz`, `gpu_intervals/block_compute_stream`.

## What consumes the memory

In that same physical capture:

- DiT live/pending weight-window accounting near the sampled peaks is
  **8.963 / 8.960 GiB**, against a configured 9 GiB DiT window. It includes
  window-managed entries/pending retirement, not every model allocation.
- Worker PyTorch batch allocated maxima are **11.484 / 11.500 GiB**. These
  counters include live tensors, weights and PyTorch-managed workspace.
- Worker reserved maxima are **13.338 / 13.191 GiB**. Reservation includes
  reusable inactive allocator blocks; allocated memory is a subset, not an
  additional quantity to add to reserved memory.
- NVML's **14.237 / 13.857 GiB** covers the entire device, including the parent
  process and non-PyTorch memory. Parent/worker/rank independent maxima cannot
  be summed to reconstruct this point in time.

Therefore the full-device peak reflects a large retained DiT weight working set
plus execution buffers and allocator/communication state. It cannot be reduced
to the isolated FC1 B output alone. Retaining weights is intentional NFE reuse;
shrinking that residency may exchange memory for more H2D loading and latency.

## Exact operator: established boundary and remaining limit

The full-request observers record NVML samples, batch allocator maxima and
pre-block memory, but Kineto was captured with `profile_memory=False`.
Consequently they **do not identify which operator created the maximum live
allocation within a full DiT block**. Their observed median sample spacing is
136–274 ms, despite a nominal 50 ms monitor sleep; brief transients can be missed.

Separately, the current isolated S8192 Row Partial study establishes:

| Isolated FFN variant | Largest temporary peak/rank | Measured location |
|---|---:|---|
| Production Row Partial baseline | 1348 MiB | Column-LoRA FC1, full B output |
| Benchmark-only Selective B | 1344 MiB | FC2 global FP32 rotation/quantization |

Selected FC1 itself peaks at 938.03 MiB because the unchanged base INT8 operator
allocates two output buffers. These are **temporary deltas above a different
resident baseline**, at S8192, and cannot establish the exact operator at the
historical full-request peak. Selective B remains benchmark-only.

The confirmed answer is **DiT transformer execution with near-full weight
residency**. Exact current-pipeline allocation-site attribution still requires
a narrowly scoped physical memory diagnostic using the current source/LoRA
configuration: reset/read allocator peaks around attention, FC1 and FC2 within
the actual request, retain per-block/NFE labels, and collect an allocator
snapshot only for the winning interval. Such a diagnostic must be separate from
accepted latency timing. No new run or production hook is implied by this audit.

## Reproduction and retained evidence

`benchmarks/analyze_pipeline_memory.py` reuses existing result files/directories,
requires successful physical captures, binds source JSONs with SHA256, separates
warmup/measured stages, and optionally correlates local compressed causal traces.
The output is `semantic_results/pipeline_memory_peak_audit.json` (about 144 KiB).
It retains per-capture stage maxima, exact sample times, batch allocator fields
and correlated block intervals, without copying traces or checkpoints.

Example: pass the continuous causal-results directory with
`--output semantic_results/pipeline_memory_peak_audit.json`. Add the continuous
residency/TE and idle-result directories to reproduce the aggregate scope.
