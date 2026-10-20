# Native ComfyUI sampler integration

**2026-10-11 VAE update:** the default TP2 workflow now uses Native-aligned video Query Partitioning and audio Channel Parallel through `MiniMaxH3ParallelVAELoader`. Public decode planning and reference encoding remain Native. Per-model Native overrides are available. Video public outputs pass the focused local qualification; audio preserves the small diagnosed rounding error. See [current VAE integration and limits](PARALLEL_VAE_INTEGRATION.md). The qualification results below describe the earlier Native-VAE milestone and do not validate the newly integrated parallel VAE path on physical T4s.

Implemented for the validated INT8 MiniMax-H3 checkpoints and single-request latency. ComfyUI owns conditioning metadata, guider, sampler, audiovisual latent packing, schedules and output conversion. Two persistent workers execute Native model entry points with the validated TP blocks and a bounded weight window. This establishes the synchronous integration; it does not establish an end-to-end speedup.

## Files and use

- `custom_nodes/minimax_h3_tp2`: `MiniMaxH3TP2Loader` returns standard MODEL and CLIP objects plus a shared worker session. `MiniMaxH3ParallelVAELoader` is the default video/audio loader; `MiniMaxH3PostDecodePrefetch` remains available for older Native-only graphs.
- `patches/comfy_sampler.py`: Native CLIP and MiniMax ModelBase adapters and an OUTER_SAMPLE lifecycle wrapper. BasicGuider/CFGGuider, SamplerCustomAdvanced, scheduler and ModelSampling nodes continue through ComfyUI.
- `patches/comfy_tp_runtime.py`: persistent NCCL rank processes, checkpoint loading, shared 16 GiB host budget, worker acknowledgements and cancellation/error cleanup.
- `workflows/video_minimax_h3_r2v_tp2_int8.json`: separate derivative of the supplied workflow, preserving its sampler, schedules, seed and reference graph while selecting the qualified INT8 checkpoint stack.
- `workflows/video_minimax_h3_r2v_native_int8.json`: standard Native loaders with the same INT8 stack, for fallback. The original Heretic/NVFP4/FP16 workflow has not been changed.

Link the node directory into the target ComfyUI `custom_nodes` directory rather than copying its `__init__.py` alone. It resolves its repository path and imports `patches`. On Linux, for example:

```sh
ln -s /path/to/project/custom_nodes/minimax_h3_tp2 /path/to/ComfyUI/custom_nodes/minimax_h3_tp2
```

Restart ComfyUI, open the TP2 workflow, and set the TE/DiT loader's absolute checkpoint paths. Put the video/audio VAE files in ComfyUI's configured VAE directory. The TP2 loader requires two physical CUDA GPUs with NCCL; the one-GPU qualification harness is a separate correctness tool. Use the Native fallback workflow on the single 5060 Ti.

On Windows, create a directory junction instead of copying the node files, using the actual project and ComfyUI paths. Physical execution is qualified on Linux/NCCL; a Windows installation without an NCCL backend cannot execute this two-worker service. Neither this request nor the qualification installs or starts a second ComfyUI server on the desktop.

The qualification environment uses ComfyUI `d49e888586dd8ae012c0667b33466b815fee07f7`, kitchen 0.2.37, TE FP16 storage with Native FP32 conditioning activations, and DiT FP32 inference. An independently installed ComfyUI version or another checkpoint format requires qualification. The supplied ComfyUI-master directory has no Git checkout identity, so it has not been substituted for the pinned validation environment.

## Lifetimes and memory

1. Native MiniMax tokenizer and SDClipModel preserve reference labels/order, image/video preprocessing, vision tower, multimodal RoPE, DeepStack features and modality tags. Vision executes with transient GPU weights and is released before decoder streaming. Both workers currently execute vision; removing that redundant work is a future measured optimization.
2. A real OUTER_SAMPLE wrapper opens the worker sampling session before Native condition refinement. Each actual denoiser call appends its projection/transformer uses to the residency window. The schedule does not infer NFE from the nominal step count: guidance and multi-evaluation samplers can make additional calls.
3. Rank zero performs Native text refinement once per condition and broadcasts it. Rank one never loads duplicate refiner/projection backing. GPU entries remain reusable between calls, subject to the live-byte and free-memory guards. All GPU weights are released at sample completion or interruption; CPU cache workers remain alive.
4. The default VAE loaders now dispatch parallel raw decoding inside Native public conversion, with CPU startup lookahead during decoding and bounded post-decode TE block prefetch. Decoder replicas are released after each public decode. This supersedes the earlier Native-only VAE wrapper; see the current VAE integration document for exact scope and memory fallback.

Default limits: 16 GiB shared host reservations, 3 GiB pageable cache per rank, 6 GiB pinned staging allowance per rank, 6 GiB GPU weight window per rank, plus an activation/free-memory reserve. The per-rank allowances share the single 16 GiB host limit; they cannot all be filled independently. Host reservations cover the runtime's canonical auxiliaries and estimated construction peaks as well as caches/staging. They are not total RSS; ComfyUI, Native VAEs, outputs, libraries and allocator caches also consume memory. Available-RAM admission provides additional headroom. The local harness uses 2 GiB TE and 4 GiB DiT windows because both logical ranks share one device.

Pinned staging is charged at twice the tensor payload, bounding PyTorch's per-tensor power-of-two allocator rounding. Stage teardown fences consuming work and purges inactive pinned blocks, so old cached allocations do not compete with the next model's CPU construction. The planned physical PyTorch 2.11 build exposes the same `_host_emptyCache` binding used locally; see its [CUDA binding source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/cuda/Module.cpp). This private binding is version-sensitive; unqualified Torch upgrades need a new memory lifecycle check.

Native model management sees proxy weights, so worker VRAM is governed by this runtime's guards rather than ComfyUI's proxy model-size estimate. Starting another TP2 configuration closes the preceding session. Parallel requests on a session are serialized. The node's loader output remains cached by ComfyUI across ordinary queued runs.

## Executed findings

The expanded two-layer local run passes eleven bit-exact comparisons: image conditioning, image/video conditioning, negative conditioning, audiovisual res_multistep, guided Heun, a complete request after sampler interruption, Native ComfyUI video decode/handoff, and TE after decode. It executes trained weights and the full Native vision and video VAE. The shared reservation peak is 13,658,772,480 bytes (12.72 GiB), with zero admission denials and zero reservations after close. Evidence: `semantic_results/comfy_sampler_2/result.json`.

The complete 50-layer TE/DiT local run passes seven bit-exact comparisons: the three conditioning outputs, both audiovisual res_multistep output streams, Native video decode and subsequent TE conditioning. Its managed reservation peak is 17,111,486,208 bytes (15.94 GiB), below the 16 GiB ceiling. It records 526 admission deferrals and releases all reservations at close. These deferrals are backpressure, not proof of negligible memory pressure: the full-depth path operates close to the ceiling. Both local runs use the same frozen source. Evidence: `semantic_results/comfy_sampler_50/result.json`.

Integration defects were fixed:

- FP32 conditioning requires Native RMSNorm weight casting. Short prompts also exposed the missing explicit masked-GQA fallback at the global/local head-count dispatch boundary. The corrected TE preserves both Native decisions.
- Loading the refiner/projections on rank one duplicated unused CPU backing. Loading only its output/curve/RoPE auxiliaries removes that cost. Native bound-method cycles are broken at stage teardown before reservations are released.
- Keeping vision projection storage alive competed with the long decoder's retained weight window. Transient vision staging now ends before decoder execution. The TE construction bound uses the inspected Native FP16 loading path; DiT retains its larger bound for FP16-to-FP32 expansion.
- The recursive module-copy helper retained its tensor/module memo through a Python closure cycle. Evicted GPU weights and retired pinned buffers could survive until cyclic GC, outside the scheduler's accounting. Clearing both memos immediately after copying makes physical lifetime follow the owning modules/events. A full-depth failure with zero live window weights and only 81 MB driver-free memory led to this diagnosis.

Native reference generation runs in a separate process and exits before TP verification. Full-depth reference loading previously left a large CPU allocator high-water footprint competing with the persistent workers and post-decode next request. Process isolation removes benchmark-only memory pressure without relaxing runtime RAM admission. The Native reference tensors retain matching source fingerprints.

Physical sampler qualification **passed** in [Kaggle version 1](https://www.kaggle.com/code/chinokiki9/minimax-h3-comfyui-sampler-gate), on two Tesla T4 GPUs (15,636,037,632 bytes each), PyTorch 2.11.0+cu130/CUDA 13.0. All eleven comparisons are bit-exact: zero relative L2 and maximum absolute error. Both ranks execute eleven denoiser calls across the ordinary, guided, interrupted and recovery samples. Native callback behavior and token tags also match. Both ranks finish with no active sample or GPU model phase; after service close, managed reservations are zero. The peak is 12.72 GiB with zero admission deferrals.

| Qualification | Trained TE/DiT depth | Comparisons | Outcome | Peak managed host reservation |
| --- | --- | --- | --- | --- |
| Local 5060 Ti, logical TP2 | 50/50 | 7 | All bit-exact | 15.94 GiB; 526 admission deferrals |
| Local 5060 Ti, logical TP2, guidance/cancellation | 2/2 | 11 | All bit-exact | 12.72 GiB; zero deferrals |
| Physical 2× T4, persistent NCCL workers | 2/2 | 11 | All bit-exact | 12.72 GiB; zero deferrals |

All three runs include the full Native vision tower and video VAE. The physical gate uses two trained TE/DiT layers to isolate process transport and NCCL lifecycle behavior; it is not a physical full-depth generation test or a latency benchmark. Source fingerprints match both local runs and the current runtime. NCCL logs confirm two ranks and the selected Socket transport; P2P and IB are disabled consistently with the preceding qualification. Initialization logs are not per-collective timing traces.

Setup took 302.68 seconds and the complete reference/verification CUDA qualification took 132.58 seconds, for 435.26 seconds total. These are job durations, not model or generation latency. The job completed without retry within its 720-second setup/480-second qualification budgets. Evidence: `kaggle_comfy_sampler_output/sampler_results/result.json`, `hardware.json`, `manifest.json` and both `nccl.*.log` files. `comfy_sampler_control.py --download` collects the completed session evidence. Local timing is not physical dual-T4 performance evidence.

## Limits and next optimization

The initial parent/worker transport uses synchronized CPU packets with preserved dtypes and Native PackedLayout metadata. Per-call D2H/H2D and CPU serialization are included in actual execution but no transport-overlap gain is claimed. The separate asynchronous ActivationHandoff API is not enabled here.

Sampler callback interruption releases the local session. Physical worker interruption, failure or deadline expiry terminates both ranks so an old NCCL collective cannot leak into the next request; reload the loader/session afterward. Detected weight patches/hooks, ControlNet, additional model wrappers and unrecognized worker payloads fail explicitly. Online LoRA is supported through `MiniMaxH3TP2LoRAStack` connected to the TP2 loader. Standard parameter-patching LoRA loaders remain unsupported because the proxy does not expose real checkpoint parameters. The loader offers optional `lora_row_partial_reduce`, disabled by default; Concat A is deprecated. See [current Row Partial implementation](ROW_PARTIAL_MEMORY.md). Native model_sampling object patches remain on the parent ModelBase.

Independent cropped video slabs remain disabled. Native-aligned video queries and audio channel parallelism are now enabled in the default derivative workflow; the audio rounding difference is explicitly accepted rather than described as passing its old strict gate. The five-second generation quality comparison, representative 8k integrated latency/profile, larger full DiT sequences, abrupt physical worker-death recovery, redundant vision removal and asynchronous transport remain distinct qualification/optimization work. Reuse accepted component timings; do not extrapolate this compact sampler check to production speed or quality.

The supplied workflow's non-loader nodes are unchanged, and its graph links are validated. The harness executes the actual Native sampler APIs, not the complete HTTP prompt queue, media loaders or SaveVideo/mux nodes. A complete queued five-second workflow is the next deployment acceptance test. Then profile one representative S8192 request before changing transport: quantify per-call CPU packet overhead, recurrent weight reloads and memory-admission stalls. Optimize the dominant measured cost and repeat only its affected correctness gate.
