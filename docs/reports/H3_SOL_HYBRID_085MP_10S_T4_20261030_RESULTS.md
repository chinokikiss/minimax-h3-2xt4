# H3 Sol Hybrid 0.85 MP / 10 s — Physical 2×T4 E2E Results

**Date:** 2026-10-30  
**Kernel:** `pamcks/h3-sol-hybrid-085mp-10s-t4-20261030`  
**Kernel version:** 4  
**Status:** **COMPLETE**

## 1. Scope

This run is the production source-bound 2×Tesla T4 end-to-end validation for the hybrid attention policy:

- Text Encoder: `kitchen_causal`
- DiT NFE 1: Kitchen INT8
- DiT NFE 2–7: Sol All-INT8, `tau=1.0`
- DiT NFE 8: Kitchen INT8
- Video VAE attention: Sol All-INT8, `tau=1.0`
- Audio VAE decode: Native single-GPU

This report records execution results only. It does not claim visual/audio quality equivalence against a dense Native reference.

## 2. Fixed workload

| Field | Value |
|---|---:|
| Resolution | 1216 × 704 |
| Pixels/frame | 856,064 |
| Frames decoded | 243 |
| Exported frames | 240 |
| FPS | 24 |
| Export duration | 10.0 s |
| Sampler | Euler |
| Scheduler | Simple |
| NFE | 8 |
| Seed | 20261020 |
| Video shift | 6.0 |
| Audio shift | 3.0 |
| Turbo LoRA | `lightx2v/Minimax-h3-Turbo` |
| LoRA file | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` |
| LoRA revision | `30f485ed874aa44c8183348ad361c5eabad318da` |
| LoRA strength | 1.0 |
| Verified LoRA targets | 208 |

## 3. Completion summary

The batch and generation both completed successfully.

- Batch status: `complete`
- Generation status: `complete`
- Completed sampler steps: **8 / 8**
- Completed denoiser calls: **8 / 8**
- Video decode strategy: **`native_temporal_pairs`**
- Video decode fallback: **none**
- Audio decode strategy: **Native**
- All recorded final tensors: **finite**

Decoded shapes:

- Video pixels: `[1, 243, 704, 1216, 3]`
- Audio waveform: `[1, 324000, 2]`

Export shapes:

- Video: `[1, 240, 704, 1216, 3]`
- Audio: `[1, 320000, 2]`

Exported media payloads:

- Video: **19,767,959 bytes**
- Audio: **1,280,044 bytes**

## 4. E2E timing

The full E2E worker completed in **1549.299 s = 25 min 49.3 s**.

The measured foreground generation interval was **1497.299 s = 24 min 57.3 s**. The difference includes worker/setup overhead outside the foreground interval.

| Stage | Time |
|---|---:|
| Service/model setup | 11.963 s |
| VAE metadata | 0.462 s |
| Reference processing + TE | 181.249 s |
| 8-step sampler | 1191.284 s |
| Video decode | 100.943 s |
| Audio decode | 2.973 s |
| File export | 19.845 s |

The separately measured offline disk bake took **340.140 s = 5 min 40.1 s** and is not included in the 25 min 49.3 s full-E2E worker duration.

## 5. DiT attention dispatch proof

The runtime recorded the same exact dispatch on both ranks.

### Rank 0

| NFE | Kitchen calls | Sol calls |
|---:|---:|---:|
| 1 | 50 | 0 |
| 2 | 0 | 50 |
| 3 | 0 | 50 |
| 4 | 0 | 50 |
| 5 | 0 | 50 |
| 6 | 0 | 50 |
| 7 | 0 | 50 |
| 8 | 50 | 0 |
| **Total** | **100** | **300** |

- Kitchen calls outside denoise: **0**
- Sol calls outside denoise: **0**

### Rank 1

Rank 1 recorded the identical **100 Kitchen / 300 Sol** distribution with zero calls outside denoise.

Therefore the requested hybrid schedule was executed exactly:

`Kitchen → Sol → Sol → Sol → Sol → Sol → Sol → Kitchen`

No Native DiT attention call was expected or accepted by the E2E observer.

## 6. Video VAE Sol proof

Both ranks recorded:

- Video VAE backend: `sol`
- Video decode strategy: `native_temporal_pairs`
- Fallback: `null`
- Observed `VAE/sol_all_int8` calls: **2016 per rank**
- Video Sol AOT receipt: **present**
- Sol native library: `/root/.cache/minimax_h3/sol_sm75/723b60b97966dde4af0e70e0/h3_sol_all_int8_sm75.so`

This confirms that the production temporal-pair decode path executed with Sol rather than silently falling back to Native or Kitchen.

## 7. Sol AOT and runtime environment

Both ranks recorded successful DiT Sol AOT preparation for the actual sequence length:

- DiT rows: **65,166**
- Architecture: **SM75**
- CUDA compiler: **CUDA 12.8 / nvcc 12.8.93**
- PyTorch: **2.11.0+cu128**
- TileLang: **0.1.13**
- `sol_precompiled`: **true**
- `complete_default_custom_coverage`: **true**

DiT AOT preparation took approximately **51.30 s per rank**. The Sol native library build/load contribution inside that preparation was approximately **11.35 s on rank 0** and **11.17 s on rank 1**.

The Video VAE reused the prepared Sol library and recorded successful Video AOT preparation on both ranks.

## 8. Actual packed layout

The runtime recorded one 65,166-token H3 packed layout:

| Segment | Range | Tokens |
|---|---:|---:|
| Text | 0–2528 | 2528 |
| Reference image 1 | 2528–3344 | 816 |
| Reference image 2 | 3344–4164 | 820 |
| Generated audio | 4164–4974 | 810 |
| Generated video | 4974–65166 | 60192 |

DiT query shape per rank:

`[1, 28, 65166, 128]`

Under the current production `protected_prefix()` contract, this recorded layout resolves the exact-protected conditioning prefix to **4164 tokens**: text plus both reference-image segments. Generated audio and video remain outside the protected prefix.

## 9. VRAM observations

Maximum recorded denoise memory across the two ranks:

- Peak allocated: **11.22 GiB**
- Peak reserved: **14.29 GiB**

The temporal Video VAE commands recorded approximately:

- Peak allocated: **2.99 GiB per rank**
- Peak reserved: **3.41 GiB per rank**

The TE phase reached approximately:

- Peak allocated: **2.86 GiB**
- Peak reserved: **4.81 GiB**

These are runtime allocator observations from this run, not total board-memory or system-RAM measurements.

## 10. Offline bake

The persisted TP2 DiT bake completed successfully:

- Persistent baked bytes: **21,237,118,728**
- Verified Native targets: **208**
- Disk baked: **true**
- Inference re-merge: **false**
- Online LoRA: **false**
- Bake receipt SHA-256: `a2f452fdb4b7c29bae52efd94599e57d94b4b9164e553fc1932c1dcdfcaed3fb`

The job released **19,634,812,800 bytes** of job-owned base DiT downloads after the persistent bake.

## 11. Reproducibility hashes

Final tensor hashes:

| Tensor | SHA-256 |
|---|---|
| Video latent | `cfd9ef443ba926647935056c57c43f2d3f1c3abaf928bd87171ee918d66d4def` |
| Audio latent | `234b9dd4bf66ca0164af95fc99393953a35e7ee7f6c4f935e98d53223580800b` |
| Decoded pixels | `acc396456ae0048d34c3b47ad178a04f05b49aa75cebc097383670d503aa61c9` |
| Decoded waveform | `4eb91a65ef1127c379bdd5f2153aa602a1ee3a5ad8ec494266dd45f3a26a874e` |

All four recorded tensors were finite.

## 12. Result

**Execution acceptance: PASS.**

This run establishes, on physical 2×Tesla T4 hardware and the source-bound production runtime, that:

1. the complete 8-NFE H3 request finishes;
2. the requested per-NFE Kitchen/Sol DiT schedule executes exactly on both ranks;
3. Sol All-INT8 runs on the real 65,166-token DiT workload;
4. Sol AOT completes on both ranks;
5. the Video VAE executes through the production `native_temporal_pairs` path with Sol and no fallback;
6. the requested 243-frame video and audio decode complete with finite outputs;
7. the 10-second export completes.

**Quality acceptance remains pending visual/audio review.** This run did not execute a dense Native generation-equivalence comparison.

## 13. Artifact collection note

The first all-output Kaggle collection attempt encountered a transient TLS EOF from `kaggleusercontent.com` after downloading two setup files. A subsequent targeted collection of result artifacts succeeded and retrieved:

- `batch_results.json`
- `full_e2e.log`
- `generation/result.json`
- `offline_disk_bake_receipt.json`
- `request_config.json`

The TLS interruption was an artifact-download issue after kernel completion and is not an E2E execution failure.
