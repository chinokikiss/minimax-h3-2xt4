# REF2VA end-to-end results — Kaggle dual T4, version 2

**S1 and S2 completed. Their output tensors and final MP4s are byte-identical. L1 failed during denoising and is incomplete. OOM investigation and further runs are deferred at the user's request.**

Hardware: two physical Tesla T4 GPUs, SM75; PyTorch2.11.0+cu130, comfy-kitchen0.2.37, TileLang0.1.15. Notebook: https://www.kaggle.com/code/ocnsai/h3-ref2va-end-to-end, version2.

## Completed workload and latency

Both short runs generated **864×480, 124 frames at24fps**, with20 actual denoising evaluations, identical reference/prompt/seed and sampler settings. The actual DiT sequence was **16,374 tokens**, including570 TE conditioning tokens,405 encoded-reference tokens,414 generated-audio tokens and14,985 generated-video tokens.

| Metric | S1: idle disabled | S2: bounded idle enabled |
|---|---:|---:|
| Background preload | 0 s | 35.25 s |
| Foreground trigger → MP4/WAV ready | **933.19 s** | **912.19 s** |
| Preload + foreground | **933.19 s** | **947.44 s** |
| VAE construction | 9.34 s | 11.33 s |
| Reference encoding + TE | 149.49 s | 129.69 s |
| Sampler,20 evaluations | 717.07 s | 713.15 s |
| Video decode | 48.77 s | 49.24 s |
| Audio decode | 2.59 s | 2.67 s |
| MP4/WAV export | 5.69 s | 6.00 s |

Downloads and TP-service setup are outside foreground latency. Preload is measured from its start/end timestamps, including the initial control call. Foreground includes lightweight observation overhead. Stage totals have small gaps from input preparation and snapshot/control work.

**Observed result:** S2's foreground was21.00 s shorter (2.25%), while preload plus foreground was14.25 s longer (1.53%). Idle residency is useful here for moving work into an already available idle interval; this pair does not show a lower total preparation-plus-generation time. These are single cold requests, not repeated distributions or a statistically established speedup.

## Residency and memory

S2 admitted and reused17 TE prefix blocks per rank, totaling **3.863 GiB/rank**, below the4 GiB cap. Its saved counters report17 idle GPU hits and4,147,638,784 bytes of avoided **foreground** transformer H2D per rank. Total transformer transfer volume including preload was unchanged:12,198,937,600 bytes/rank. Startup/vision transfers are separate from these transformer-window counters.

TE foreground loading stalls were reduced, not eliminated: rank0's recorded TE host-future waits fell from73.29 to60.69 s; its CUDA-event block-acquisition spans fell from72.97 to60.31 s. Host waits can overlap GPU work; neither quantity should be added to execution time as an independent serial stage. The TE attention API spans also differed between runs, so the full19.80 s TE-stage difference cannot be attributed solely to prefix reuse.

| Memory metric | S1 GPU0 / GPU1 | S2 GPU0 / GPU1 |
|---|---:|---:|
| Rank allocator peak allocated | 8.824 / 8.791 GiB | 8.824 / 8.791 GiB |
| Rank allocator peak reserved | 12.748 / 12.584 GiB | 12.896 / 12.584 GiB |
| Sampled NVML board-used peak | 13.700 / 13.249 GiB | 13.849 / 13.249 GiB |

Allocator values describe individual rank processes; NVML includes the other contexts and libraries on each device. Do not add their independent peaks. Sampled process-tree RSS sums peaked at22.50/22.42 GiB for S1/S2 and may count shared mappings more than once. Cgroup usage reached approximately30 GiB, including Linux file cache and kernel charges; it is not equivalent to anonymous working-set memory. Short runs completed without OOM.

## Correctness, dispatch and media

- S1/S2 conditioning, video/audio latents, decoded pixels and waveform had identical shapes, dtypes and SHA256 hashes; all output tensors were finite.
- Both MP4s have SHA256 `2c550037f10858ebdb7a04fda42fc8a12c1c86df82582ee6bdbe4d93c6bc0388`.
- Both downloaded MP4s were fully decoded locally:124 frames,864×480. Their decoded tensor shape was `[1,124,480,864,3]`; the exporter correctly handles the batch dimension.
- Both ranks executed50 TE Group32 attention calls. S1/S2 used the configured kitchen attention path for DiT and the existing parallel VAE interfaces.
- S1/S2 byte equality validates the idle-policy comparison. It is **not** a fresh dense Native accuracy comparison, and visual/temporal quality remains for user review.

Downloaded files:

- S1: `semantic_results/ref2va_e2e_v2/ref2va_e2e_results/S1/video.mp4` and `audio.wav`.
- S2: `semantic_results/ref2va_e2e_v2/ref2va_e2e_results/S2/video.mp4` and `audio.wav`.

## L1: incomplete

The requested864×480/362-frame configuration reached **46,447 actual packed tokens**, including1,096 TE conditioning tokens,810 image-reference tokens,1,206 audio tokens and43,335 video tokens. This matches the user's decision to retain the requested geometry/duration instead of forcing100k tokens.

Reference encoding + TE completed in161.05 s. The first denoising evaluation did not complete: a QKV LoRA B projection requested a3.72 GiB allocation and failed. Consequently there is **no L1 end-to-end latency, completed video, or successful memory-safety result**. The job ended with ERROR because of L1; it does not invalidate the separately completed S1/S2 outputs. Further OOM work is deferred; no new GPU job is being submitted for it.

## Evidence and limits

Raw results, per-rank/per-NFE timings, bounded loading/copy/cache counters, compressed memory samples, logs and download checksums are under `semantic_results/ref2va_e2e_v2/`. `compact_summary.json` and `RESULTS.md` provide generated summaries; `ref2va_e2e_results/s1_s2_correctness.json` contains the output-comparison receipt.

Separate cold TileLang/CUDA cache directories were used for S1/S2. First-use compilation/module-loading time remains in these cold request timings. CPU file loading uses monotonic/thread clocks; CUDA events measure GPU timeline spans that can include dependencies or idle time. Attention/LoRA API spans are not kernel-only timings or causal no-LoRA ablations. Full Nsight/Kineto kernel traces and a repeated warm-run distribution were not collected. No claim is made that all mandatory profiling dimensions or the long stress test passed.
