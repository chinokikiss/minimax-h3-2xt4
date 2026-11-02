# H3 1 MP FP16/512 SOL-v2 E2E on Dual Tesla T4 — Final v2 Results (2026-11-02)

## Verdict

**PASS: full 1.032192 MP, 10-second, eight-NFE E2E completed on two physically distinct Tesla T4 GPUs.** Offline regular Turbo LoRA bake, all six application stages, 8/8 denoising calls, video decode, audio decode, and media export completed successfully. The earlier v1 first-DiT-O-Proj CUDA OOM did not recur.

**Fallback verdict, with evidence boundaries:** Every monitored DiT attention dispatch used **SOL-v2 INT8** (400/400 expected calls per rank, 50/50 per NFE); the monitored DiT Kitchen fallback count was **zero** on both ranks. Both decoder records have `fallback: null`. Scaled-FP16 row-epilogue and fused-SwiGLU implementation calls were observed. However, this frozen E2E does **not** count individual `row_gated`, P0-2, P0-3, P0-B, or P0-C fast-path hits versus guarded fallback. Thus **zero fallback is proven for the instrumented DiT attention and decode selections, but a blanket zero-fallback claim for every lifetime optimization is not independently demonstrated**. No explicit scaled-FP16 fallback warning or CUDA OOM appears in the collected E2E log.

This distinction matters: the enabled policy flags and backend wrapper counts are not sufficient to certify P0-C's local rotation on every call. The existing `current/scaled_fp16_epilogue` observer wraps both `tilelang_transport_p2.finish` and `tilelang_dit_fusion.finish_row`; it does not distinguish the fused row-gate path from the ordinary supported scaled-FP16 path.

## Frozen job and provenance

- Kaggle notebook: [pamcks/h3-1mp-fp16-512-sol-v2-8nfe-10s-p0-20261102](https://www.kaggle.com/code/pamcks/h3-1mp-fp16-512-sol-v2-8nfe-10s-p0-20261102), **version 2**.
- Root receipt: `docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102_v2/gpu_submission.json`.
- Final evidence: `docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102_v2/gpu_final_v2/` (28 files, 50,412,627 bytes).
- Principal E2E directory inside that archive: `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/`.
- Both devices report `Tesla T4`, compute capability `(7,5)`, with **distinct GPU UUIDs**. Runtime: PyTorch `2.11.0+cu128`, CUDA `12.8`, Kitchen `0.2.37`, TileLang `0.1.13`.
- The v2 source overlay's SHA256 matches the current working-tree copies of `tilelang_dit_fusion.py`, `tilelang_transport_p2.py`, `tilelang_row_quant.py`, `tilelang_swiglu_quant.py`, `tilelang_swiglu_inplace.py`, and `h3_operator_policy.py`. The six-row repair's `tilelang_dit_fusion.py` SHA256 is `f3d8a5afa2237692d7ee0f5c418df50348b595529f1ded7d7f194a2ac8c64669`.
- Workload: 1344 × 768 = **1,032,192 pixels**, 243 native video frames, 240 exported frames at 24 FPS, 10 seconds, 8 Euler/Simple steps, CFG=1, residual exponent 9 (**FP16 / 512**), offline regular Turbo LoRA, no online LoRA, DiT SOL-v2 tau 1.0 on NFE 1–8. DiT sequence tokens **S=78,310**.
- Offline bake verified 208 native targets; `inference_remerge=false` and `online_lora=false`.

The user completion signal authorized exactly one version-bound final-output collection. The first transfer partially succeeded but terminated with a Kaggle CDN TLS EOF; a bounded second transfer fetched only missing v2 files. These were **two transfer attempts within the same single logical collection**, not Kaggle status polling, not a new notebook submission, and not a second execution.

## E2E completion and complete timing ledger

The final `batch_results.json` reports `status=complete`, `completed_steps=8`, and successful offline-bake and full-E2E command exits (both zero). The notebook supervisor reports `worker_exited`, `exit_code=0`. The following tables separate independent measurement scopes to avoid adding nested intervals twice.

### 1. E2E application stages (additive)

Durations are calculated directly from `generation/result.json::stages` `start_ns/end_ns` pairs. These six entries are sequential application stages and may be summed.

| Application stage | Exact elapsed (s) | Minutes | Share of six-stage total |
| --- | ---: | ---: | ---: |
| `vae_metadata` | 0.541 | 0.009 | 0.03% |
| `reference_and_TE` (reference-image processing + text encoder) | **148.695** | 2.478 | 7.96% |
| `sampler` (refiner, preparation, eight NFE and host-side work) | **1,581.495** | 26.358 | **84.62%** |
| `video_decode` (Video VAE) | **112.492** | 1.875 | 6.02% |
| `audio_decode` (Audio VAE) | 2.898 | 0.048 | 0.16% |
| `file_export` (video/audio encoding and writing) | 22.828 | 0.380 | 1.22% |
| **Sum of all six stages** | **1,868.950** | **31.149** | **100%** |

The foreground timing is 1,869.402 s; the measured difference between this enclosing interval and the six stage intervals is **0.453 s**. Stage durations are independent of the separately recorded offline bake and do not include notebook environment setup.

### 2. Sampler internal service-call breakdown

These are `generation/result.json::service_calls` durations (`end_ns - start_ns`). The service call includes both ranks, inter-rank communication, orchestration and waiting; this is **wall-clock service latency, not isolated GPU kernel time**.

| Service operation | Calls | Total duration (s) |
| --- | ---: | ---: |
| `begin_sample` | 1 | 1.217 |
| `refine` (first call) | 1 | 6.792 |
| `prepare_dit` (including AOT/kernel preparation) | 1 | **54.709** |
| `refine` (second call) | 1 | 0.531 |
| `denoise` NFE 1–8 | 8 | **1,515.231** |
| `end_sample` | 1 | 0.755 |
| Application gaps and sampling orchestration, by difference | — | **2.261** |
| **Whole sampler stage** | — | **1,581.495** |

All 8 denoising service calls account for **95.81%** of the sampler stage; preparation accounts for **3.46%**. The 2.261 s remainder is an arithmetic difference between stage duration and the non-overlapping listed calls, not a distinct instrumented kernel.

### 3. Exact NFE wall-clock service latency and rank command enqueue spans

The first numeric column below is the authoritative end-to-end `denoise` service-call duration. Per-rank values are the worker-observed `REF2VA_RANK_COMMAND` spans `enqueue_end_ns - start_ns`; they are provided only for dispatch diagnostics and **must not be summed across GPUs or interpreted as synchronized per-card GPU execution time**.

| NFE | Full service wall time (s) | Rank 0 enqueue span (s) | Rank 1 enqueue span (s) |
| --- | ---: | ---: | ---: |
| 1 | **209.054** | 204.118 | 201.559 |
| 2 | 187.098 | 141.165 | 173.813 |
| 3 | 185.184 | 168.844 | 167.308 |
| 4 | 187.299 | 140.829 | 186.056 |
| 5 | 187.267 | 168.972 | 178.771 |
| 6 | 186.421 | 140.162 | 171.313 |
| 7 | 186.727 | 168.894 | 166.395 |
| 8 | 186.181 | 140.273 | 159.238 |
| **Sum (service wall time only)** | **1,515.231** | — | — |

NFE 2–8 average **186.597 s per NFE**. NFE 1 is **22.457 s** slower than this subsequent-step mean. The log contains additional TileLang compilation records during initial sampling, but no attribution trace uniquely partitions NFE 1 latency into compile, compute, transport or waiting; the difference must not be reported as a pure compile tax.

### 4. Reference / TE and decoder service details

- **Reference + text encoder stage:** 148.695 s total. Its measured `te` service call occupies **144.123 s**; the **4.572 s** remainder covers other stage work and gaps, without an independent split into reference image preprocessing versus CPU work.
- **Video VAE stage:** 112.492 s total. `vae_prepare` = **10.844 s**, `decode_start` = **0.661 s**, seven `video_temporal_pair` service calls total **96.040 s** with individual times **13.703, 14.676, 14.436, 13.446, 12.943, 13.149, 13.687 s**; `vae_release` = **0.319 s**. The remaining ~4.628 s is stage-local work and gaps. These are distinct from the later media encoding/export stage.
- **Audio VAE stage:** 2.898 s. The measured `decode_start` and `decode_complete` service notifications are each ~0.001 s; most audio decode work is outside those trivial notifications and has no further instrumented subdivision.
- **Media file export:** **22.828 s**, separate from the Video VAE and Audio VAE stages.

### 5. Outer notebook and command scopes (not additive to the stage table)

| Timing scope | Observed | Relationship |
| --- | ---: | --- |
| Offline disk-bake command | **336.092 s** (5m 36.09s) | Separate preparatory command; 208 native targets verified |
| E2E foreground interval | **1,869.402 s** (31m 9.40s) | Encloses the six application stages |
| Full E2E command | **1,918.628 s** (31m 58.63s) | Encloses foreground and other command overhead; difference **49.226 s** |
| Notebook worker supervisor elapsed | **2,814.325 s** (46m 54.33s) | Includes commands and notebook setup/orchestration; difference from the two command durations **559.605 s**, not independently attributed |

**Never add the 1,868.950 s stage sum to the 1,918.628 s E2E command or the 2,814.325 s supervisor duration:** they overlap in the same execution hierarchy. Likewise, the eight NFE durations belong *inside* the 1,581.495 s sampler stage.

This run uses a larger sequence and newer kernels than the archived 2026-11-01 1216×704 baseline (1,490.008 s); it is **not a controlled speed-regression or speedup comparison**.

## Per-card CUDA and NVML memory

DiT command peaks come from `REF2VA_RANK_COMMAND` / `ranks[*].benchmark.commands` in the final E2E evidence; these are per-rank PyTorch allocator metrics. The NVML numbers are approximately 0.5-second device-memory samples from `generation/memory_samples.json.gz`, not PyTorch allocated memory.

| Measurement | Rank 0 / GPU 0 | Rank 1 / GPU 1 |
| --- | ---: | ---: |
| Maximum PyTorch **allocated**, all DiT NFE | **12.605559 GiB** | **12.369410 GiB** |
| Maximum PyTorch **reserved**, all DiT NFE | 13.705078 GiB | **14.306641 GiB** |
| First NFE PyTorch allocated | **12.605559 GiB** | 12.368757 GiB |
| NFE 2–8 PyTorch allocated | 11.635498 GiB | 12.369410 GiB |
| NVML sampled **device used** maximum | 14.939 GiB | 14.982 GiB |
| NVML sampled **process used** maximum | 14.496 GiB | 14.539 GiB |

The global allocated peak occurred on GPU 0 in NFE 1. The subsequent steady per-step allocations are lower on GPU 0. The sampled device usage is extremely close to the provisioned memory; do not confuse device usage, process usage, reserved memory, and allocated memory. Both GPUs completed without a CUDA allocation failure. The cgroup sampling records zero OOM kills.

## Actual backend dispatch and fallback audit

| Component | Requested / expected | Physical evidence | Verdict |
| --- | --- | --- | --- |
| DiT attention | SOL-v2 on all eight NFE, 50 blocks per NFE, per GPU | `DiT/sol_all_int8=400` per GPU; `attention_dispatch.sol.total=400` and every step =50; `attention_dispatch.kitchen.total=0` on each GPU | **No DiT attention fallback observed; all 800 dispatched calls executed SOL** |
| TE attention | Kitchen causal | `TE/kitchen_causal=50` per GPU | Expected TE backend; **not a DiT fallback** |
| Generic Kitchen INT8 | Used in TE/other non-DiT sites | `kitchen/int8_attention=2066` per GPU, while DiT Kitchen count remains zero | Expected unrelated calls; not proof of DiT fallback |
| Row transport and fused epilogue | Scaled-FP16 row reduction | `current/scaled_fp16_epilogue=900` per GPU | Actual optimized implementation executed; does not isolate fused row-gate hits |
| SwiGLU ConvRot quantization | `fused_stored` | `current/fused_swiglu_quant=450` per GPU | Actual optimized preparation executed; does not separately count P0-3 and P0-B |
| P0-2 / P0-C O-Proj | Enabled, inference-only owned tensors, SM75, TP2 | Source overlay and flags match; six-row table guard is repaired; E2E proceeds through all steps | **Eligible and successful; per-call P0-2/P0-C hit count not recorded** |
| P0-3 / P0-B FC2 | Enabled, `fused_stored`, SM75, TP2 | Fused preparation observed; current code has guarded lifetime and in-place branches | **Per-call lifetime/in-place hit count not recorded** |
| Video VAE | Temporal pair decode; Kitchen video attention | `strategy=native_temporal_pairs`, `fallback=null`, `attention_backend=kitchen` | No recorded decode-strategy fallback; expected Kitchen |
| Audio VAE | Native | `strategy=native`, `fallback=null` | No recorded decode-strategy fallback; expected Native |

The source-level AdaLN repair accepts modulation table row counts that are positive multiples of three (including six), with bounded segment labels and unchanged exact-Tensor/SM75/LoRA/geometry safety checks. This removes the identified v1 six-row rejection. The first-NFE CUDA OOM did not recur, but no runtime counter uniquely proves **every** O-Proj hit `_local_o_proj_p0c`; that narrower claim requires a future per-branch counter or trace. No new GPU job was submitted for this report.

The stored `sol_collect_route_stats=false` is separate from the actual dispatch counters. It prevents per-token SOL route-statistics analysis, **not** the per-block SOL-versus-Kitchen verification above.

## Numerical checks and media integrity

- All **16/16** video/audio denoiser outputs (two per NFE) are finite.
- All **32/32** sampled `denoised` and `latent` summaries (four per step) are finite; combined **48/48** intermediate finite checks pass.
- The four final `outputs` summaries (video latent, audio latent, decoded pixels and waveform) are finite.
- Native decoded shapes: video `[1,243,768,1344,3]`; stereo waveform `[1,324000,2]`.
- Exported shapes: 240 frames and 320,000 stereo audio samples for exactly ten seconds.
- `generation/video.mp4`: **17,506,425 bytes**; FFprobe reports 1344×768, 24/1 FPS, 240 frames, 10.000 seconds, plus a 32 kHz stereo audio stream. Full FFmpeg demux/decode to the null muxer exited **0**, with no reported decoding error.
- `generation/audio.wav`: **1,280,044 bytes**; 32 kHz, stereo, 16-bit PCM, 320,000 frames, 10.000 seconds, normalized absolute PCM peak 0.126984, RMS 0.022204, **zero clipped PCM samples**.
- `decode_inputs.pt` preserves the actual generated latents for independent decode investigations.
- The saved `quality_acceptance` field is **“Pending visual/audio review; no dense Native comparison or generation-equivalence claim”**. Finite values, successful file decode, and no audio clipping do **not** establish perceptual quality or numerical equivalence to a Native reference.

## Source and instrumentation caveat

The backend call observers in `benchmarks/run_h3_l1_turbo_e2e.py::CurrentRank._install_observers` wrap:

- `patches.tilelang_swiglu_quant.prepare` as `current/fused_swiglu_quant`;
- `patches.tilelang_transport_p2.finish` and `patches.tilelang_dit_fusion.finish_row` under the same `current/scaled_fp16_epilogue` name.

There are no dedicated incrementing counters around `row_gated` eligibility, `tilelang_row_quant._local_o_proj_p0c`, the P0-2 `consume_input` branch, `tilelang_swiglu_inplace.prepare`, or the P0-3 `set_()` branch. Future qualification can add these counters, with stage/rank/NFE labels, to explicitly certify 0 guarded fallbacks without relying on flags or caller-level inference. **Do not infer an unmeasured optimization's savings from global E2E memory alone**.

## Primary archived evidence

1. `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/batch_results.json`: completion, actual hardware, command timings, bake provenance.
2. `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/generation/result.json`: configuration, 8 NFE, stage timings, 48 finite checks, backend and attention dispatch counters, decoder strategies, final digests.
3. `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/full_e2e.log`: per-rank command peak allocations/reservations and per-step completions.
4. `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/generation/memory_samples.json.gz`: 3,717 sampled NVML/host records.
5. `h3_t4_fp16_512_sol_v2_1mp_10s_supervision/supervision.json`: bounded worker success and elapsed time.
6. `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/source_overlay.json`: exact delivered source hashes.
7. `h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/generation/video.mp4`, `audio.wav`, `decode_inputs.pt`: actual generated deliverables.

**Disposition:** Record this as the first complete 1 MP, 8-NFE, 10-second FP16/512 SOL-v2 dual-T4 E2E after the six-row AdaLN repair. Preserve the failed v1 evidence separately. This report and receipt only collect/analyze existing artifacts; no inference source, model, or Kaggle job is modified.
