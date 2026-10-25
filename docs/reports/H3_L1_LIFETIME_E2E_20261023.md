# Long-sequence E2E retry after DiT lifetime validation

The user explicitly authorized repeating the previous full long-sequence H3
test after the [lifetime-only T4 validation](DIT_TEMPORARY_LIFETIMES_20261023.md).
The earlier operator-only restriction does not apply to this specific request.

**Completed on actual 2×T4:**
[H3 L1 Current Offline E2E](https://www.kaggle.com/code/pamcks/h3-l1-current-offline-e2e)
**version 3**, submitted on 2026-10-23 at 15:16 China time, completed all four
sampling steps and exported video with audio. The one-shot final artifact
collection completed at **16:42:45 China time**, returning 27 files. All 27
size/SHA256 checks and the submitted source binding passed. No status/list or
live-log polling was used. Generation quality remains pending review.

Preparation preserves the prior L1 request exactly: **896×512, 362 frames,
24 fps, four LCM/simple steps, CFG 1, seed 20261020**, the same superhero /
mechanical-dragon prompt and the same two reference images. Native packing is
expected to produce the previous **51,220 tokens**. Offline DMAD strength 1
uses the same immutable LoRA revision and 208 independently verified targets;
no online LoRA attaches during inference.

The runtime is the source already validated on actual 2×T4: FP16-fix,
Group32 TE attention, the prior L1 scalar sparse DiT override, fused-stored
FC2 and scaled-FP16 transport. Video VAE remains demand-loaded with its
validated attention/query fallback; Audio VAE remains native single-GPU FP32.
The 6 GiB rank weight cap, 9 GiB reserve and disabled idle residency are
unchanged. This retry adds no new candidate operator or precision setting.

Only the two existing hash-verified reference-image files were added to the
immutable delivery archive. All **105 previous capsule files**, including
runtime, runners, frozen source, request and core setup, remain byte-identical
to the T4-qualified lifetime source. This prevents image fetching from a moving
branch during the new run. Models and LoRA still use immutable HF revisions
and exact file hashes, with no credentials or Kaggle input mounts.

The new 107-file capsule SHA256 is
`8481ff336f0573030bcf59c35aba06fee4a3d33123a4cbded9db011498a8d1d1`.
It is published at GitHub commit
`e46a6c382c33e5b763bbc18f68659a03be816efe`.
The separate CPU qualification, `pamcks/h3-l1-lifetime-cpu` version 1, passed
**31 tests** and the frozen API/source gates in 109.14 seconds. Its matching
receipt qualifies the submitted 107-file source. This is environment/source
qualification, not an E2E success claim.

The GPU notebook retains a **240-second setup** watchdog and **4,800-second
execution** watchdog, with the existing 1,800-second bake and 2,400-second
generation caps inside the execution budget. Provider queue/startup is outside
these code deadlines. Sampling latents are saved before decoding, so a later
VAE/export failure does not discard completed denoising outputs.

There is no Kaggle status/list/live-log polling. A local one-shot collector
can retrieve final artifacts once after the bounded window. Its exclusive
claim prevents a timer/manual collection race; cancellation is local and
makes no Kaggle request. Returned source bindings and prior conditioning/noise
hashes are checked separately from sampling/decode success and quality review.

## Returned execution evidence

The returned hardware gate identifies two distinct Tesla T4 UUIDs, both SM75.
The actual stack is Kaggle-native Torch **2.11.0+cu128**, CUDA/nvcc **12.8**,
comfy-kitchen **0.2.37** and TileLang **0.1.13**. The returned capsule hash is
the submitted `8481ff33…a8d1d1` value. This validates this Kaggle-native profile;
it does not qualify a different Linux Torch stack.

Offline baking completed in **352.03 s** with **208 native-independent target
checks**. Both inference ranks report no online LoRA attached. The exact prior
conditioning and both initial-noise SHA256 values match. Both ranks report
**51,220 tokens**, with local Q shape **[1, 28, 51220, 128]**. Each completed
50 DiT blocks on each of four denoiser calls.

The backend counters on each rank show 50 TE Group32 calls, 200 DiT sparse
calls, 250 fused SwiGLU/quantization calls and 500 scaled-FP16 epilogues. The
preserved L1 DiT scalar policy was exercised; this run does not change or
qualify the normal Kitchen DiT default. Token Refiner remains Native SDPA.

| Stage / boundary | Elapsed seconds |
|---|---:|
| Direct-pip/source setup, before execution | 73.65 |
| Offline LoRA bake, before generation | 352.03 |
| Reference encoding + TE | 148.94 |
| Sampling, all four steps | 953.17 |
| Video decode | 392.28 |
| Audio decode | 3.33 |
| File export | 16.83 |
| Foreground request through export | 1,515.38 |
| Generation subprocess, including startup/cleanup | 1,560.13 |
| Full cold execution, including model delivery/bake/generation | 2,634.63 |

These boundaries overlap hierarchically and must not be added together. The
four rank-0 denoiser GPU command spans were **263.33, 223.59, 223.95 and
224.05 s**. The first call includes cold initialization/compilation effects;
the later calls use different denoising states and sparse routes. This is one
E2E viability run, not repeated warmed performance benchmarking or a measured
speedup against the previous run, which failed before its first step finished.

## Memory and fallback behavior

| DiT worker measurement | Rank 0 | Rank 1 |
|---|---:|---:|
| Peak allocated, GiB | 11.634 | 10.511 |
| Peak reserved, GiB | 13.945 | 14.305 |
| Sampled device-wide NVML peak during sampling, GiB | 14.995 | 14.980 |
| Mean sampled GPU utilization during sampling, % | 93.44 | 95.01 |

Allocated/reserved values describe worker PyTorch allocators. NVML samples
include device/context/other process usage and were sampled about every 0.5 s;
they are not an exact instantaneous allocation trace. The previous synthetic
50-block 10.53-GiB lifetime result must not be substituted for this E2E rank-0
11.63-GiB peak. No OOM occurred with the unchanged 6-GiB weight cap and 9-GiB
reserve. The successful full trajectory demonstrates that the former first-FC1
OOM is avoided for this fixed offline-LoRA workload, without establishing
headroom for larger shapes or concurrent requests.

Video Query Parallel **declined admission** with the recorded reason
`RAM/VRAM admission declined on a rank`; decode completed using the **native
fallback**. During that stage, sampled GPU utilization averaged 96.42% on
device 0 and 0% on device 1. This proves the fallback works in this workload;
it does not validate parallel Video decode. Video decoding accounts for about
26% of foreground time and is a remaining single-GPU cost. Audio used native
single-GPU decoding as configured. Only the Video Encoder was loaded before
sampling (360,747,328 bytes read); the Audio Encoder remained unloaded. Video
and Audio Decoders were demand-loaded after sampling.

All recorded denoiser outputs, sampler latent/x0 summaries, final latents,
decoded pixels and waveform are finite. The saved decode-input checkpoint is
preserved. Local container inspection confirms **362 H.264 frames at 896×512,
24 fps**, lasting **15.083 s**, plus an embedded AAC track. The separate stereo
WAV lasts **15.075 s**. Container validity and finite tensors do not establish
visual semantics, temporal consistency, audio quality or equality to a dense
Native baseline; those remain pending review, including behavior after 2 s.

Raw outputs under the evidence root are
`gpu_output_v1/light_execution/e2e/L1/{result.json,video.mp4,audio.wav,decode_inputs.pt,memory_samples.json.gz}`.
No Nsight capture or new full-pipeline comparison was performed in this retry.

Evidence root: `D:/attention_eval/h3_l1_lifetime_e2e_20261023/`.
The [submission record](../data/h3_l1_lifetime_e2e_20261023.json) binds the
request, source, CPU gate, accepted GPU version and one-shot collector.
The copied collector script and PID are preserved under that evidence root;
an exclusive claim prevents automatic duplicate artifact retrieval. Queue
delay beyond the collection window is reported as unavailable outputs, never
inferred as a successful run. The collector does not submit or retry jobs.
