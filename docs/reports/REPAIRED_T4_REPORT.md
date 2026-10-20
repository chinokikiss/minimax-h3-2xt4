# Repaired dual-T4 benchmark findings — 2026-10-08

The physical run is finished. **TE and DiT TP/SP passed correctness, but were slower than native on every tested workload. Video TP failed correctness before timing. Two-slab video decoding passed scheduling checks and was faster, with substantial approximation error.**

One dual-T4 allocation took **369.6 seconds (6m10s), including 197.9 seconds of setup**. TE, DiT and slabs completed; the scheduler reports ERROR because video TP was rejected. The job is stopped. No replacement allocation was launched. Audio was excluded as requested.

## TE and DiT results

These are **actual trained block-0 measurements**, not full conditioning, diffusion or generation latencies. Inputs are portable uniform synthetic activations, rounded to FP16; DiT computes in supported FP32 with the checkpoint timestep embedding and nontrivial segment/RoPE arguments. S37 controls also completed; all rows and samples are in the linked evidence.

| Component / sequence | Native ms | TP=2 ms | SP=2 ms | Correctness |
|---|---:|---:|---:|---|
| TE S512 | 21.51 | 26.93 | 28.76 | Stored rank-0 TP/SP outputs bit-exact |
| TE S2048 | 68.65 | 99.91 | 104.76 | Stored rank-0 TP/SP outputs bit-exact |
| DiT S1024 | 40.82 | 64.75 | 77.93 | Stored rank-0 TP/SP outputs bit-exact |
| DiT S4096 | 271.26 | 304.84 | 331.84 | Stored rank-0 TP/SP outputs bit-exact |

All **12 recorded TP/SP comparisons**, including S37, are bit-exact against native before/after timing. Each distributed rank passed the declared numerical gate; rank-1 bit equality was not separately archived. The gate was relative L2 <=1e-5, cosine >=0.999999 and maximum absolute error <=0.005.

Peak allocated VRAM shows why sharded weights do not establish lower whole-workload memory. At TE S2048, native uses 0.861GiB, TP 0.784GiB per GPU and SP about 0.794GiB per GPU. At DiT S4096, native uses **1.556GiB**, TP **2.466GiB per GPU** and SP **2.508GiB per GPU**. Full activation gathers and quantization buffers can outweigh the saved block weights.

**Action:** retain native computation with the previously verified layer placement for capacity; do not adopt these repaired TP/SP implementations for speed on the tested workloads. The historical 50-block placement results remain separate evidence. Complete embeddings/vision/refiner/head integration is still unverified. If further TP/SP optimization is needed, target the wide activation gathers: investigate group-aligned local rotation with globally agreed quantization scales, preserving native activation/rounding dispatch. Revalidate exact outputs before another timing experiment.

## Video native and slab results

Native measures the complete trained 36-block **raw `_decode_pixels`** decoder. The two original saved latents are reused: [1,24,1,48,48] and [1,24,2,48,84], producing raw [1,3,4,768,768] and [1,3,8,768,1344] outputs. Public-VAE normalization, padding/cropping and RGB conversion are excluded.

| Workload | Native ms | Slab 64 ms | Slab 64 full L2 | Slab 128 ms | Slab 128 full L2 |
|---|---:|---:|---:|---:|---:|
| 768x768, latent T=1 | 451.92 | 234.04 | 11.395% | 261.20 | 10.012% |
| 768x1344, latent T=2 | 2190.29 | 974.73 | 8.114% | 1022.29 | 7.526% |

Every slab case passes the matching sequential-slab scheduling gate. Both physical devices pass native INT8 Tensor Core GEMM/attention probes, with no FP32 SGEMM observed. The full-width physical replica/transport control passes. Independent blend-weight construction introduces at most 0.0004883 absolute difference; this is separate from the much larger full-decoder approximation error.

Native/slab wall-time ratios are about **1.93x / 1.73x** at 768x768 and **2.25x / 2.14x** at 768x1344 for overlaps 64/128. These are same-allocation wall ratios with a timing-boundary difference: native input is already GPU-resident, whereas slabs include CPU input transfers. They are conservative comparisons, not an identical-boundary end-to-end speedup measurement. Historical fastest two-GPU 384px tile ratios are **4.67x / 4.18x** for the smaller case and **1.58x / 1.51x** for the larger; those span allocations and are exploratory.

| Workload / overlap | Latent slab widths | Peak allocated GiB, GPU 0 / 1 | Mean GPU utilization %, 0 / 1 | Output copy ms | Stitch ms |
|---|---|---|---|---:|---:|
| 768x768 / 64 | 26 / 26 | 2.738 / 2.737 | 79.7 / 76.0 | 0.857 | 0.340 |
| 768x768 / 128 | 28 / 28 | 2.744 / 2.743 | 80.4 / 75.5 | 0.945 | 0.514 |
| 768x1344 / 64 | 44 / 44 | 2.964 / 2.964 | 90.6 / 87.4 | 2.763 | 0.938 |
| 768x1344 / 128 | 46 / 46 | 2.979 / 2.978 | 90.6 / 88.5 | 2.870 | 1.239 |

The native peaks are 2.800GiB and 3.240GiB on GPU 0. Slabs slightly reduce the per-device peak, while two replicas increase aggregate memory. Utilization averages include validation, warmup and timing windows; they are not isolated kernel occupancy. Output copies transfer about 7.3–7.9MiB or 24.8–25.9MiB. These durations measure PyTorch cross-device copies; physical routing was not classified, so they are not established PCIe wire times. Copy/stitch durations are small relative to the measured decode wall times.

**Action:** use native decoding when full-context fidelity is required. Slabs are a useful optional approximation candidate, numerically better than historical 384px tiles (24.3% / 21.7% L2), but **all four fail the original full-decoder 1% quality gate**. Overlap 64 is fastest; overlap 128 improves numerical quality for roughly 12% / 5% more latency on the smaller/larger workloads. Evaluate generated latents with motion and objects crossing the center seam, including frame/flicker inspection, before selecting a production quality policy. Random benchmark latents do not establish visual acceptability.

## Video TP failure and repair status

The first video TP comparison, at 768x1344 latent T=2, produced **1.5776% relative L2**, cosine 0.9998756, maximum absolute error 0.2869 and mean absolute error 0.01085. This fails both the new strict execution gate and the original 1% video gate. **No TP timing was accepted. Video SP was not attempted after its shared TP core failed.**

The same target inputs passed on the 5060 Ti. That establishes a hardware/runtime-sensitive discrepancy, not a proved root cause. Single-GPU simulation cannot certify T4 dispatch or physical execution.

A concrete implementation difference was corrected after this run: the custom video block used out-of-place RMS/RoPE, while pinned native inference selects the in-place operator and preserves QKV view layout. The updated code again passes all 20 exact-input comparisons bit-exact locally, but **has not been revalidated on T4**. Do not call the physical discrepancy repaired or enable video TP/SP on the basis of that local result.

**Next diagnostic:** on one informative T4 workload, compare the first divergent block's QKV projection, RMS/RoPE, INT8 attention, output projection, gate/up projection and residual outputs against native, recording kernel/layout/device context. Isolate the first difference before rerunning a timing sweep or relaxing any tolerance. No replacement instance was launched for this investigation.

## Credibility, boundaries and evidence

One allocation; fixed native/TP/SP order; one warmup and five synchronized samples per accepted row. **24 timing rows / 120 samples** were retained. Bootstrap intervals describe within-process variation, not independent-session uncertainty. TE native block calls include mask/RoPE construction, while TP/SP prepare those arguments outside timing: avoid precise pure-block optimization ratios from these unequal boundaries. TE TP/SP remain slower even against that larger native scope. A future comparison should hoist native setup too, if close timing differences affect a decision.

The range downloader fetched only the ~875MB of TE/DiT checkpoint tensors required for block 0 and verified the assembled files against locally validated SHA256. Only the full 2.8GB video checkpoint was downloaded. Original video input bytes and checkpoint fingerprints are preserved. The abandoned private upload created no dataset used by this run. Pinned runtime: two Tesla T4s, Torch 2.11.0+cu130, comfy-kitchen 0.2.37, comfy-aimdo 0.5.5, ComfyUI d49e888586dd8ae012c0667b33466b815fee07f7; NCCL P2P/IB were disabled, matching the prior transport configuration.

Source fingerprints were matched to the downloaded executed-source archive; sample medians and gate/validity records were checked. Saved output tensors and full kernel traces were not independently audited. The archive preserves the pre-run source/proof; the current local simulation artifact reflects the later RMS/RoPE dispatch correction.

Evidence: complete measured table (`kaggle_repaired_output/MEASURED_RESULTS.md`, local-only evidence), samples/metrics (`kaggle_repaired_output/aggregate.json`, local-only evidence), audit (`kaggle_repaired_output/audit.json`, local-only evidence), execution manifest (`kaggle_repaired_output/repaired_manifest.json`, local-only evidence), latest local checks (`semantic_results/repaired_t4_inputs.json`, local-only evidence), and [Kaggle version 1](https://www.kaggle.com/code/chinokiki9/minimax-h3-repaired-dual-t4).
