# Actual local dual-rank validation — 2026-10-08

The experiments were executed on the RTX 5060 Ti with two independent logical ranks, CUDA streams and model replicas. The earlier source/AST blocking framework was removed after the user clarified the request. Its `preflight.json`, `guard_tests.json` and entry-guard artifacts describe superseded code, not the current implementation.

## Results and scope

| Execution | Result | What was actually exercised |
|---|---|---|
| `benchmarks/simulate_experiments.py --trained-model-dir D:\minimaxH3` | **51 comparisons passed; every comparison bit-exact**, about 15 seconds of test execution | Native initialized TE/DiT graphs, actual checkpoint dimensions, trained TE/DiT block 0, full trained 36-block raw video decoder, reduced native PP wrappers |
| `benchmarks/test_slab_semantics.py --checkpoint D:\minimaxH3\minimax_h3_video_vae_int8_convrot.safetensors` | **13 trained cases and 367 independent CPU geometry cases passed**, about 12 seconds including checkpoint hashing | Independently loaded full video replicas, odd widths, B=2, temporal axes, zero/full overlap, center fades, saved original workload inputs |
| `benchmarks/simulate_vae_scheduling.py` | **24 checks passed** | Actual audio/video scheduling wrappers driven by coordinate decoders; not trained audio reconstruction or trained tile quality |

Evidence: [logical_dual_gpu.json](semantic_results/logical_dual_gpu.json), [trained_local_semantics.json](slab_results/trained_local_semantics.json), [vae_scheduling.json](semantic_results/vae_scheduling.json).

The transport simulator exchanges **different rank tensors** through real gather, scatter and sum operations; it does not repeat a rank's input or stub attention/matrix multiplication. Threads rendezvous in collective order. Native CUDA kernels execute on cuda:0. Native scratch buffers are isolated by stream to represent separate devices' private workspaces. TF32 is disabled in the TE/DiT simulation to avoid relying on an arithmetic feature unavailable on T4. Source fingerprints accompany the results.

The trained video checkpoint SHA256 is `52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e` (2,811,065,184 bytes), matching the pinned checkpoint. Models in D:\minimaxH3 were read without modification.

## Defects repaired through execution

- **TE:** restored causal/padding masks and native multimodal rotary embeddings; preserved GQA and normalization layout. Native SDPA kernel selection depends on the full query size: choosing a different backend for each smaller head shard changed rounding and subsequent INT8 quantization. The TP path now preserves that native decision.
- **DiT:** fixed packed QKV and gate/up selection, native rotary application, FP32 AdaLN/normalization, fused SwiGLU quantization and global segment indices crossing a sequence boundary. Removed the nonrotated fallback.
- **Video:** restored native INT8 attention, head-interleaved QKV and component-aware gate/up packing. SP preserves exactly four register tokens and one native zero token, with uneven shards rather than extra attended padding. Output projections now use the native fused INT8 operator, preserving quantization, bias and residual rounding through all 36 trained blocks.
- **Native PP wrappers:** preserved the TE tuple/intermediate/cache/DeepStack contracts and current-device contexts; preserved DiT FP32 placement and tensor-valued segment indices. Reduced native stacks exercise these paths locally.
- **Tile planners:** reject invalid overlaps and sizes rather than entering a zero-step loop or silently replacing explicit zero with defaults.

The video output projections are now **replicated**, with head activations gathered before the native output GEMM. TE/DiT SP also gathers native activations before the corrected TP block. These choices spend communication/compute to preserve the computation. Old TP/SP timings and claimed INT8 payload savings cannot be attributed to the repaired algorithms; performance must be measured separately if it later matters.

One simulation-harness issue was also corrected: creating/moving quantized models inside global inference mode dequantized the native reference. References are now constructed outside inference mode, and their native quantized dispatch is retained.

## Slab correctness versus quality

Each rank's trained raw slab decode is bit-exact against independent sequential native execution. Independently computed cross-fade weights differ by at most **0.0004883 absolute / 1.53e-6 relative L2** across all 13 cases. The full-width control is bit-exact against full decoding. Native INT8 Tensor Core GEMM and INT8 attention were traced; no FP32 SGEMM was observed. Kernel recognition was checked against both the new local trace and the saved T4 profile.

Correct independent slabs still omit global attention context and renormalize width positions. Against the locally executed full trained decoder, saved-input relative L2 is **11.36% / 8.12%** for overlap 64 and **9.99% / 7.55%** for overlap 128 at 768x768 / 768x1344. All fail the original full-decoder 1% gate. This is a documented approximation, not a native-equivalent decoder. See [the slab plan](SLAB_EXPERIMENT_PLAN.md).

## Remaining limits and cluster scope

A single GPU cannot validate GPU-1 extension state, physical transfer behavior, NCCL, separate-device memory peaks or T4 latency. Trained TE/DiT checks cover block 0, not all 50 trained layers or full conditioning/generation. PP checks use reduced stacks; complete vision/packing/head integration and production replica constructors remain unverified. Audio coordinate tests do not overturn the historical trained channel-split numerical failure. Visible seam quality needs representative generated latents.

Only the **new slab method** was submitted after the trained local checks passed. The private `chinokiki9/minimax-h3-slab-only` job uses two T4s, original saved tensors, overlap 64/128, one warmup and five samples per case. It has a 300-second setup limit, a 600-second experiment limit and a 900-second Kaggle session limit, without automatic retry. Both physical kernel probes and a full-width replica/control comparison must pass before timing. No old audio, TE, DiT, native-video or tile timing sweep was relaunched. Cluster status/results are recorded separately in the slab plan.

The submitted version 1 stopped during historical artifact selection (`Expected exactly one mounted historical tensor`), before model downloads or decoder tests. Its filename-only selection did not identify a unique file; the completed run contains repeated basenames across rounds. The packager now selects the manifest's exact round-0 path. All four selections passed a simulated mount with three rounds, while the real mount remains unverified. No replacement was launched. Evidence: [slab_manifest.json](slab_results/physical_t4/slab_manifest.json) and [mount_selection.json](slab_results/mount_selection.json).

## Newly authorized repaired dual-T4 benchmark

After explicit authorization to benchmark TE, DiT and video VAE, the exact target workloads were executed locally again: **20 comparisons, all bit-exact**, including TE S2048, DiT S4096 and the complete video decoder at both large geometries. [Current input/source evidence](semantic_results/repaired_t4_inputs.json). This includes the new raw-video worker option and corrected explicit FP32 DiT worker input dtype. The previous 51-case artifact refers to its earlier source revision; the current package checks the new evidence fingerprints. A single bounded dual-T4 job is now submitted; see [execution scope/status](REPAIRED_T4_REPORT.md).

## Physical findings from the authorized run

The combined job is finished and stopped. TE/DiT TP/SP passed the physical gate, with all 12 stored rank-0 comparisons bit-exact; each rank passed the numerical predicate. Both methods were slower than native at all tested sequence lengths. Video TP failed before timing at the larger workload (1.5776% L2), so video SP was not attempted. The full native video and all four physical slab cases completed. Slab execution matches sequential slabs, while full-decoder approximation errors remain 7.53–11.40%. [Physical results and limitations](REPAIRED_T4_REPORT.md).

After the physical failure, the video implementation was aligned with native inference's in-place RMS/RoPE dispatch; its previous out-of-place operation used a different output layout. All 20 exact-input local comparisons pass again, but this change has not been verified on T4 and the physical discrepancy remains unresolved. The executed source archive retains the previous revision. Do not treat the local simulation as a universal hardware correctness guarantee.
