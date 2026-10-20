# Benchmark findings and actionable repair plan

**Completed repaired dual-T4 benchmark (2026-10-08):** TE/DiT TP/SP now passed all 12 stored comparisons bit-exact but were slower than native on all tested block workloads. Video TP failed before timing (1.5776% L2); SP was not attempted after the shared core failed. Physical slabs passed execution checks, taking 0.234/0.261s at 768x768 and 0.975/1.022s at 768x1344 for overlaps 64/128, with 7.53–11.40% full-decoder L2 error. The job is stopped; no replacement was launched. [New measured findings, boundaries and actions](REPAIRED_T4_REPORT.md). The sections below retain the earlier benchmark review and local repair history.


**2026-10-08 repair update:** Actual two-rank execution on the local 5060 Ti now passes 51 comparisons, all bit-exact, including trained TE/DiT block 0 and complete trained 36-block video decoding. Causal/padding/rotary, packed-component, segment-index and native INT8 rounding/dispatch defects were repaired. Video output projections are replicated; TE/DiT SP communication also changed. Historical timing tables below describe the old code and do not measure these repairs. Full TE/DiT integration and physical collective behavior remain unverified. [Detailed repairs and scope](SEMANTIC_VALIDATION.md).

**New slab findings:** 13 trained scheduling cases and 367 geometry cases passed. Full-decode relative L2 at 768x768 / 768x1344 is 11.36% / 8.12% with overlap 64 and 9.99% / 7.55% with overlap 128. Every candidate fails the original 1% quality gate. The single new slab-only T4 job stopped during historical artifact selection. Exact round-0 selection is now prepared and tested on a simulated mount; no replacement was submitted. No new T4 timings exist. [Findings and next steps](SLAB_EXPERIMENT_PLAN.md).


## Deployment decision

Use native layer placement as the integration target for TE/DiT, and keep native single-GPU video decoding as the default. The existing TE/DiT TP/SP implementations fail numerical correctness; current video TP/SP is slower than native decoding. Channel-split audio is a promising optional optimization with very small numerical differences. Tiling should remain an explicitly approximate memory-saving mode.

Both benchmark jobs finished. No additional GPU experiments were started for this analysis. Their completed artifacts supersede the earlier report based only on live logs.

The selected artifacts contain **246 measured timing rows and 2,460 raw timing samples**. Audits passed for timing arithmetic, declared gate decisions, input hashes, archived harness hashes, checkpoint revision, runtime pins and hardware. Output metrics below are those recorded by the benchmark; output tensors were not independently recomputed.

## Findings and their implications

Times below are medians of three process medians. Every process used three warmups and ten synchronized timed calls. Single-block, 50-block and full-decoder timings have different scopes.

| Component | Evidence | Interpretation and decision |
|---|---|---|
| Native 50-block TE/DiT placement | All **30** comparisons pass and report bit-identical outputs. TE S512: **0.970 s**; DiT S8192: **50.863 s**. | Viable native block placement with a corrected CUDA device context. Integrate this first. No measured CPU-offload baseline supports a speedup claim. |
| TE TP/SP | All **36** comparisons fail the L2 gate; relative L2 **2.01–2.71%**. TP is faster per block; SP is slower. | Attention semantics differ. Restore native causal/rotary behavior before accepting performance results. |
| DiT TP/SP | All **24** comparisons fail; relative L2 **0.971–1.084**, cosine **0.160–0.259**. | Substantial mismatch. Repair packed projection sharding, modulation, positions and segment coordinates. |
| Full video TP/SP | Native is faster on **all five workloads**. Largest case: native **14.060 s**, TP **30.592 s**, SP **29.583 s**. All **30** comparisons fail the engineering gate. | Approximately **2.18×/2.10× slower** on the largest case. Keep native decoding; fix the FP32 GEMM fallback before communication tuning. |
| Audio channel split | Observed timing ratio **1.71–1.92×** versus native. All **9** strict gates fail: relative L2 **1.33–1.95×10⁻⁵**, max absolute error ≤ **1.59×10⁻⁴**. | The zero-error claim is false, but the discrepancy is tiny compared with TE/DiT errors. Retain as an experimental option pending targeted quality validation. |
| Parallel tiled decode | All **27** comparisons with the matching one-GPU tile path report bit equality. Largest video case with 384-pixel tiles: **1.914×** faster than the same tiles on one GPU. | The tested parallel scheduling preserves tiled computation. Tiling itself changes full-context outputs: video L2 **11.3–44.6%**, audio **5.47–5.80%**. Expose the approximation explicitly. |

The engineering gates do not establish perceptual quality. High cosine similarity does not excuse an incorrect causal mask. Conversely, a tiny audio difference failing a strict gate does not establish audible degradation.

## Specific repairs

### 1. Integrate native placement with explicit device guards

The original stack attempts failed with:

```text
BufferError: Can't export tensors on a different CUDA device index.
Expected: 1. Current device: 0.
```

An activation transfer to GPU 1 does not switch the current CUDA device. The corrected harness guards every native block call with `with torch.cuda.device(x.device)`; all 50-block comparisons then pass.

Apply scoped device guards to stage execution, final normalization and heads in [pp_qwen3vl.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/patches/pp_qwen3vl.py>) and [pp_dit.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/patches/pp_dit.py>). Restore the caller's device afterward. Preserve native masks, multimodal positions, rotary tables, conditioning, intermediate outputs and output-device expectations.

The measured harness calls native blocks directly. It does **not** validate the repository's complete PP loader or full generation integration. Preserve the native model forward path and change placement boundaries rather than rewriting attention.

For the tested stacks, peak allocated memory was about **11.9 GiB per GPU** for TE and up to **11.6 GiB per GPU** for DiT. Embeddings, vision, packing/refiner/heads and other generation components were excluded. Balance the split around these extra allocations; 25/25 is a starting point. Release TE before loading DiT, then release the required DiT allocation before loading VAEs. The complete TE and DiT checkpoints cannot both stay resident on two T4s.

For a single request, stage 0 and stage 1 execute sequentially. The expected benefit is keeping weights resident and avoiding offload, rather than halving compute time. The current implementation does not demonstrate overlapped pipeline throughput.

**Acceptance:** one integrated forward with real conditioning, correct device placement, native-equivalent outputs and measured memory headroom. Measure the actual CPU-offload path separately before advertising a placement speedup.

### 2. Repair DiT semantics before optimizing its communication

Concrete defects appear in [tp_dit.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/patches/tp_dit.py:250>) and [sp_int8_dit.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/patches/sp_int8_dit.py:332>):

- **Packed QKV:** native DiT stores `[Q_all; K_all; V_all]`. A contiguous half of the combined matrix does not contain a rank's Q, K and V. Split components, select the same head range from each, then concatenate `[Q_rank; K_rank; V_rank]`. Apply the same selection to scales and match output-projection columns to those heads.
- **Packed SwiGLU:** split `[gate_all; up_all]` before selecting FFN rows. Concatenate `[gate_rank; up_rank]`, and select matching FC2 columns.
- **Curve conditioning:** the checkpoint's curve path uses `apply_silu=False` and FP32 AdaLN projections. Remove unconditional SiLU on curve coordinates and preserve that FP32 island. Read these settings from the model configuration.
- **Rotary positions and normalization:** actually consume the supplied rotary tables and match native normalization epsilon, `1e-5` for this DiT configuration.
- **SP modality segments:** intersect global intervals with each rank's shard, then subtract the shard offset. Clipping global indices to local length does not implement this translation.
- **Compute precision:** upstream MiniMax-H3 supports BF16/FP32; existing proposals use FP16. Establish matching supported precision first. Evaluate any later precision reduction separately.

For example, slicing the first half of `[Q0,Q1,Q2,Q3,K0,K1,K2,K3,V0,V1,V2,V3]` and dividing it into three produces Q rows as K and K rows as V. This layout bug exists independently of quantization or PCIe overhead.

These defects are supported by source inspection. The experiment does not isolate each defect's contribution to the total error. The video VAE has a different, head-interleaved QKV layout; do not apply the DiT rule indiscriminately to video weights.

**Acceptance:** CPU checks reconstruct packed components from shards exactly. Then one checkpoint block with nontrivial positions and multiple modality segments must pass the unchanged numerical gate. Stop immediately on a mismatch.

### 3. Restore TE causal and rotary attention

[tp_qwen3vl.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/patches/tp_qwen3vl.py:170>) accepts `attention_mask` and `freqs_cis` without implementing their native behavior. Its fallback attention uses `is_causal=False`. The SP path has the same omission in [sp_int8_qwen3vl.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/patches/sp_int8_qwen3vl.py>).

Apply native Qwen3-VL rotary positions to Q/K, preserve GQA head correspondence, and propagate causal/padding masks through every attention backend. SP must use global positions across shards. A simple causal flag does not cover arbitrary multimodal/padding masks.

**Acceptance:** changing future tokens leaves earlier-token outputs unchanged under a causal mask; nonzero rotary positions agree with native output. Then apply the block L2/cosine gate. The current TP speed ratio cannot be accepted while these checks fail.

### 4. Fix video kernel dispatch and SP padding

The profiler identifies an actual implementation bottleneck:

- Native decoding uses SM75 INT8 Tensor Core GEMMs.
- TP and SP each execute **144 `volta_sgemm_128x128_tn` kernels per profiled rank**. Their projection helper falls back to floating-point GEMM when the requested `torch.ops.comfy_kitchen.cublas_gemm_int8` operator is absent.
- Per-rank collectives are **72 AllReduce** calls for TP, versus **145 AllGather + 72 ReduceScatter** calls for SP.

Repair dispatch in [tp_video_vae.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/patches/tp_video_vae.py:117>) and corresponding SP helpers. Use the supported comfy-kitchen quantized-linear interface with correct ConvRot metadata, or a supported quantized-activation GEMM adapter. Report fallback explicitly. Validate quantization, scales, layout and dtype; replacing an operator name alone is insufficient.

SP pads the largest native sequence from **28,229** to **28,230** tokens. Its extra token participates in attention. Mask the additional key consistently or support uneven shards. Dropping that output token afterward does not undo its influence on attention.

**Acceptance:** one block matches native output, kernel identity confirms intended INT8 GEMM, and an odd-token case agrees with native attention. Only then time one full decode. Kernel-duration sums can overlap; they do not prove a communication-overhead percentage or that PCIe dominates.

### 5. Validate audio approximation and select tiling configurations

Channel splitting changes the stereo decoder's batch shape. Different convolution execution/accumulation is a plausible explanation for the small audio discrepancy, but it has not been isolated. Compare intermediate decoder outputs before changing tolerances. Remove the zero-error assertion from [multi_gpu_audio_vae.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/patches/multi_gpu_audio_vae.py:206>).

Retain the original strict gate and its failure result. If approximate audio is acceptable, define a new prospective policy using generated latents, waveform errors, channel integrity and listening checks. Do not retroactively turn this run into a strict pass.

For tiling, matching one-GPU tiles establishes scheduling equivalence, not global-decoder equivalence. Global attention loses context across tiles; more overlap can reduce seams but does not generally restore omitted attention context. Check spatial seams and temporal consistency on generated latents.

On the largest raw-video workload, 384-pixel two-GPU tiles take **1.542 s** versus **2.474 s** for full raw decoding, with relative L2 **21.7%**. The 256-pixel mode is slower than full raw decoding on every tested workload; two-GPU 384-pixel tiles are also slower than full raw decoding on the smaller workloads. Use full decoding when it fits and these measurements favor it. Keep tiling as an opt-in memory/quality tradeoff.

The tiled wrapper calls `_decode_pixels`; public VAE decoding also performs normalization, temporal padding/cropping and RGB conversion. Do not present their ratio as equivalent end-to-end decode performance.

The video wrapper's default replica construction uses `deepcopy`, which does not support the tested quantized tensors. The harness loaded replicas separately from the checkpoint. Add a replica factory or explicit checkpoint reload in [multi_gpu_video_vae.py](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/patches/multi_gpu_video_vae.py:66>), and verify that the normal constructor can load the real checkpoint. The harness workaround is not a production constructor fix.

## Work order and time efficiency

Actual local dual-rank execution replaced the earlier AST blocking framework. Known TP/SP defects were repaired and 51 numerical comparisons pass bit-exact. The shared timer still validates outputs before warmups and after timing; those checks do not substitute for execution. See [semantic evidence and remaining scope](SEMANTIC_VALIDATION.md). Historical diagnoses and priorities below should be read alongside the repair update above.

| Priority | Deliverable | Completion criterion |
|---|---|---|
| P0 | Correct documentation and deployment defaults | Withdraw synthetic measured rankings, 30× speedup, zero-error and universal no-P2P claims. Link actual measurements and their scope. |
| P0 | Integrate native TE/DiT placement | Device guards, component lifecycle and one real-conditioning forward pass; auxiliary modules fit with headroom. |
| P1 | Investigate optional channel-split audio | Small discrepancy localized; explicit quality policy passed on representative outputs. |
| P2 | Repair TE/DiT TP/SP if additional compute parallelism is needed | Targeted sharding, modulation, mask and rotary checks pass before timing. |
| P2 | Restore video INT8 dispatch and padding semantics | Correctness and kernel identity pass on one informative case; native video remains the default meanwhile. |

Use CPU checks for layout/indexing. Future GPU verification should exercise one affected component and one informative workload first, stop on the first correctness failure or backend fallback, and have a hard runtime budget including setup and downloads. Repeat full timing sweeps only after correctness passes and timing uncertainty affects a concrete decision. No further GPU work is authorized by this report.

## Credibility and limits

### Added proposal: two-slab raw video decode

The [2-Slab proposal](2-Slab Video VAE.md) was simulated using independently loaded full trained 36-block replicas on two cuda:0 streams. Every rank decode matches sequential native execution exactly; stitch differences from independently calculated blend weights remain below 0.0004883 absolute and 1.53e-6 relative L2. The full-width control is bit-exact. Kernel traces confirm native INT8 GEMM and attention.

This validates the method's scheduling, not full-context equivalence. Cropping loses global attention and changes width positions. The trained overlap-128 errors are 9.99% / 7.55%, versus historical 384px tile errors of 24.3% / 21.7%, but all slabs fail the original strict quality gate. Use full decoding by default; representative generated latents and visual/temporal inspection are required before deployment.

One new slab-only job was submitted after local validation. It reused historical inputs/timings, selected only 64/128 slab modes, and had a 5-minute setup / 10-minute experiment / 15-minute session cap. It stopped during historical artifact selection before model downloads or GPU tests. The packager now selects exact round-0 paths, tested on a simulated three-round mount; the real mount remains unverified and no replacement was launched. No physical performance claim is available. Details and evidence are in [SLAB_EXPERIMENT_PLAN.md](SLAB_EXPERIMENT_PLAN.md).

### Existing benchmark evidence

The original TE/DiT strategy tables multiply block timings by assumed factors (`0.54`/`0.53`, then `0.96`), synthesize communication costs and use fixed memory figures. DiT synthetic weights/scales were uninitialized. The old video accuracy comparison used different random inputs for native and distributed paths. Those reports do not establish their strategy rankings or generation speedups.

The rerun improves the evidence through real checkpoints, shared inputs, actual NCCL execution and raw synchronized samples. However, fixtures are synthetic, three rounds share one allocation per suite, and bootstrap intervals describe within-process variation. Fifty-block placement excludes embeddings/vision, DiT packing/refiner/heads, sampling and VAEs. Its measured TE S512 time is 0.970 s; the old 0.093 s claim is not validated for this scope, and no measured CPU-offload baseline supports 30×.

Peer-access eligibility is **true** in both allocations. NCCL P2P/IB were explicitly disabled for these tests; those flags do not establish the transport used by PyTorch PP boundary copies. Production topology and transport must be observed.

Evidence: [complete measured table](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/kaggle_retest_output/final/MEASURED_RESULTS.md>), [aggregate metrics and provenance](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/kaggle_retest_output/final/aggregate_results.json>), [artifact audit](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/kaggle_retest_output/final/combined_audit.json>) and [selection/exclusions](<C:/Users/Administrator/Documents/project/2xT4 minimax h3/kaggle_retest_output/final/selection.json>).

Pinned runtime: two physical T4s per allocation; PyTorch `2.11.0+cu130`; comfy-kitchen `0.2.37`; comfy-aimdo `0.5.5`; ComfyUI `d49e888586dd8ae012c0667b33466b815fee07f7`; checkpoint revision `e5eb578a89295337b8ff433a035929ce0279e0b6`. CUDA 12.8 diagnostics are excluded because ComfyUI disabled native CUDA kernels. The original component allocation's FP16 DiT and missing-device-guard stacks are excluded; corrected DiT/stack data use the separate FP32/device-guard allocation. Exact source archives and environments are retained.

Core audits cover selected suite JSON and timing arithmetic. Profile summaries were inspected for kernel identity, collective counts and token shapes; full Chrome traces and saved output tensors were not independently audited. No complete generation speedup, perceptual equivalence or integrated model memory claim is established.
