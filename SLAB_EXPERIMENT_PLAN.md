# Two-slab Video VAE: findings and new experiment

**October 11 local diagnosis:** shared KV alone leaves substantial positional
error. Shared KV plus global RoPE reduces error to about 1.6%, where repacking
queries changes Native INT8 query-scale groups. A separate candidate preserving
Native query blocks and exchanging KV every layer restores bit-exact final
outputs on both saved full trained workloads. Physical performance is pending;
the current independent-slab implementation remains approximate. See
[controlled diagnosis and implementation direction](SLAB_KV_DIAGNOSIS.md).

The trained local simulation passes spatial partitioning, two-stream native decoding and stitching. Independent slabs remain an **approximation**: cropping removes global attention context and renormalizes width positions. Overlap reduces error but cannot restore the full computation.

## Completed local execution

Hardware: RTX 5060 Ti 16GB, Torch 2.12.1+cu132, comfy-kitchen 0.2.37, pinned ComfyUI d49e888586dd8ae012c0667b33466b815fee07f7. Both ranks use cuda:0, separate threads/streams and **independently checkpoint-loaded full 36-block decoder replicas**.

Checkpoint: D:\minimaxH3\minimax_h3_video_vae_int8_convrot.safetensors; SHA256 `52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e`. No model download was needed for the trained local validation.

| Check | Result |
|---|---|
| Independent per-column geometry oracle, odd/even widths and invalid plans | 367 passed |
| Trained native two-stream decode and stitching, including B=2 / latent T=3 | 13 cases passed |
| Rank outputs against sequential native slab decodes | Every rank bit-exact |
| Independent sequential stitch against concurrent stitch | Worst relative L2 1.53e-6, maximum absolute difference 0.0004883 (FP16 blend rounding) |
| Entire latent width on both ranks | Final output bit-exact against full decode |
| Overlapping identical coordinate ramps | Exact reconstruction |
| Tiny native CUDA trace | INT8 Tensor Core GEMM and INT8 attention; no FP32 SGEMM |

Evidence: [trained_local_semantics.json](slab_results/trained_local_semantics.json). The test body, including checkpoint hashing, took about 12 seconds. Earlier [local_semantics.json](slab_results/local_semantics.json) is a reduced initialized fixture and is superseded for trained quality claims.

The two benchmark workloads use **saved original latent tensors**, hash-checked against the completed T4 run. Regenerating the larger tensor with the same seed on another runtime did not reproduce its bytes. Saved full raw outputs were retrieved without rerunning the earlier experiments; see [manifest.json](slab_results/saved_references/manifest.json).

## Geometry and output scope

Overlap is the **total shared output width**, not a halo of that size on each side. A 64-pixel overlap is four latent columns. Temporal axes are preserved. Ranges below are half-open.

| Full latent [B,C,T,H,W] | Overlap | Left / right slab | Global shared output columns | Final raw output |
|---|---:|---|---|---|
| [1,24,1,48,48] | 64 | [1,24,1,48,26] / same | [352,416) | [1,3,4,768,768] |
| [1,24,1,48,48] | 128 | [1,24,1,48,28] / same | [320,448) | [1,3,4,768,768] |
| [1,24,2,48,84] | 64 | [1,24,2,48,44] / same | [640,704) | [1,3,8,768,1344] |
| [1,24,2,48,84] | 128 | [1,24,2,48,46] / same | [608,736) | [1,3,8,768,1344] |

The optional overlap-128 / center-fade-32 variant uses the same slab shapes, blends [368,400) or [656,688), and discards outer halo outputs. It was tested locally. These outputs are `_decode_pixels` raw tensors, without public-VAE normalization, temporal padding/cropping or RGB conversion.

## Trained approximation findings

Relative L2 against **local full trained native decoding of the saved inputs**:

| Workload | 64 / full fade | 128 / full fade | 128 / center 32 | Historical trained 384px tiles |
|---|---:|---:|---:|---:|
| 768x768, latent T=1 | 11.356% | 9.990% | 10.063% | 24.3% |
| 768x1344, latent T=2 | 8.121% | 7.548% | 7.577% | 21.7% |

Slabs substantially reduce this numerical error versus the old small tiles, but **every slab variant fails the original 1% full-decoder gate**. The historical tile column comes from the T4 runtime; it is a contextual cross-runtime comparison. The physical slab run will compare against the original saved T4 full outputs. No perceptual or production quality threshold has been established.

A larger overlap helps these inputs; the narrow center fade adds no numerical benefit here. Use full decoding by default when it fits. Treat slabs as an optional speed/memory/quality tradeoff. For deployment, the next quality experiment must use generated latents with motion, thin structures, texture and objects crossing the center seam; inspect decoded frames and flicker. Random-noise benchmark latents cannot establish visual acceptability. If quality remains insufficient, a position-control ablation can isolate coordinate renormalization from lost context; global width coordinates alone cannot restore omitted attention.

## Earlier standalone physical attempt

Submitted: [MiniMax H3 Slab Only](https://www.kaggle.com/code/chinokiki9/minimax-h3-slab-only), version 1, 2026-10-08. Submission was made only after the trained local simulations passed and source fingerprints matched. The old component benchmark instances are complete; this is one new slab-only job.

The run measures only **64 and 128 full-fade slabs**, both workloads, one warmup and five synchronized samples per case. It first checks native INT8 kernels on both physical T4s and a tiny full-width replica/transport control. Each workload then passes a sequential-slab comparison before timing; the full-decoder quality gate is reported separately and is expected to fail. No old native, tile, audio, TE or DiT timing sweep is repeated.

- Setup/download cap: 300 seconds; only the pinned 2.8GB video model is downloaded.
- Experiment cap: 600 seconds including model loading and validation.
- Kaggle session cap: 900 seconds; no automatic restart or replacement instance.
- Timing scope: CPU latent to both GPUs, concurrent raw decode, one output copy, stitch, result on GPU 0.
- Measurements: samples/latency, peak allocated/reserved VRAM per GPU, 200ms utilization samples, input/decode/copy/stitch stages, transferred output bytes, MAE/L2/cosine/max error, seam-band/gradient/temporal errors.

The output-copy duration is a PyTorch cross-device-copy measurement. Physical routing must be observed before labeling it PCIe wire time. Local same-device copy times have no two-GPU transport meaning.

Reuse historical raw timings: native / fastest measured two-GPU 384px tiles were **518.77 / 1092.41 ms** at 768x768 and **2474.32 / 1542.08 ms** at 768x1344. Ratios against the new run are exploratory cross-allocation comparisons. The larger slab case must beat **1542 ms**, not only native's 2474 ms, to improve the existing fastest measured method. Report speed, memory and quality together.

**The physical attempt ended during historical artifact selection**, before dependency installation, model downloads or GPU tests. Version 1 selected by basename and did not find a unique mounted tensor. The old run contains repeated basenames across rounds; the packager now selects the exact manifest round-0/suite path. All four selections pass on a simulated three-round mount; the real mount remains unverified. The corrected package is prepared but **not resubmitted** under the no-retry limit. That job is stopped and no T4 speedup was established by that attempt; the later combined run is recorded below. Evidence: [slab_manifest.json](slab_results/physical_t4/slab_manifest.json) and [mount_selection.json](slab_results/mount_selection.json). The harness persists a result after each case and failures on exit. Packaging: [build_slab_notebook.py](build_slab_notebook.py); execution: [retest_slab.py](benchmarks/retest_slab.py).

## Reproduction

```powershell
& C:/ProgramData/miniconda3/envs/comfyui/python.exe benchmarks/test_slab_semantics.py --checkpoint D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors --output slab_results/trained_local_semantics.json
& C:/ProgramData/miniconda3/envs/comfyui/python.exe benchmarks/simulate_experiments.py --trained-model-dir D:/minimaxH3
```

These commands validate semantics rather than rerunning historical timing sweeps. A single 5060 Ti cannot test physical GPU-1 extension state, separate-device memory or NCCL; complete TE/DiT integration also remains outside these local component checks. See [semantic scope and repairs](SEMANTIC_VALIDATION.md).

## Subsequent authorized combined benchmark

The user later explicitly requested a dual-T4 benchmark of repaired TE, DiT and video VAE. The new `minimax-h3-repaired-dual-t4` job includes these 64/128 slab modes, with newly measured full raw outputs as same-allocation quality references. It embeds original video input bytes and no longer depends on a historical output mount. The failed first slab-only attempt remains recorded above. See [current combined run](REPAIRED_T4_REPORT.md).

## Completed physical slab findings

The later authorized combined benchmark completed all four slab cases on two real T4s, with native INT8 kernels confirmed on both devices and a passing full-width replica/transport control. Overlap 64/128 measured **234.0/261.2ms** at 768x768 and **974.7/1022.3ms** at 768x1344. Full trained decode L2 is **11.395%/10.012%** and **8.114%/7.526%**, respectively. The matching sequential-slab gate passes, while every case still fails the original strict full-decoder quality gate.

Same-allocation native raw timings were 451.9/2190.3ms with GPU-resident input; slab timings include CPU transfers. Report wall ratios with that boundary difference. Historical best-tile comparisons span allocations. The combined job is stopped and no replacement was launched. [Complete physical measurements, memory, utilization and copy/stitch durations](REPAIRED_T4_REPORT.md).
