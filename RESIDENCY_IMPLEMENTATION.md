# Request-wide weight residency: implementation and local evidence

Date: 2026-10-09. Target: single-request latency. Status: executable residency core and Native TE/DiT adapters validated locally; production ComfyUI worker/sampler integration and physical dual-T4 qualification remain.

## Implemented behavior

`patches/weight_residency.py` implements a rank-private window over the ordered weight uses of one request. The order can span TE, repeated DiT evaluations, and video decoding. A CPU preparation worker reads/builds checkpoint-backed modules; a dedicated CUDA stream transfers their physical weights. Compute waits on a ready event. Weights and copy sources remain owned until their respective CUDA consumers finish.

The window preserves INT8 payloads, tensor-valued layout metadata, original dtype and operator dispatch. It copies these fields explicitly because the installed kitchen `QuantizedTensor.to` implementation does not propagate `non_blocking` to its payload transfer. Immutable weights are released against their checkpoint backing, without a GPU-to-CPU weight copy.

Admission uses separate live-weight and pinned-preparation byte limits, plus a free-memory guard for activations. The guard includes reusable caching-allocator storage and excludes allocations awaiting side-stream completion. These are weight/preparation limits, not a cap on total VRAM or process RAM: embeddings, heads, activations, allocator fragmentation and transient CPU construction require additional headroom.

The policy retains weights with future uses when capacity permits. Under pressure it evicts the inactive entry whose next use is furthest away. Speculative loading cannot displace weights needed sooner. Final uses release entries, allowing lookahead to reach the next model before the current model finishes. This does not imply whole TE/DiT/VAE models can coexist.

Fixed lookahead is available. Optional adaptation uses observed CPU preparation and CUDA copy/compute durations, adjusting lookahead one group at a time between the configured bounds. Byte and free-memory guards take precedence. This initial policy has passed correctness tests; its latency benefit has not been measured on T4.

`ActivationHandoff` performs actual copies, including separate destination storage on the same GPU. Producer, copy and consumer events establish dependencies. It retains source owners through copy completion, records destination storage on the consumer stream, and bounds outstanding source ownership. The cross-device branch still requires physical testing.

`patches/windowed_tp.py` attaches the validated TE/DiT TP blocks to Native model shells. Native outer execution retains masks, RoPE, packing, AdaLN and output heads. The adapter currently targets prefill and batch-one Ref2VA, with the validated attention dispatch; KV-cache decoding and custom attention require separate qualification.

DiT preprocessing uses Native text refinement once per request on rank zero and broadcasts the result. Per-denoising-call input projection has a separate weight group, so projecting latents does not repeatedly load the text refiner. Reference image/video packing and conditioning tags remain in the Native entry point.

## Defects found and repaired during actual CUDA execution

| Finding | Change | Evidence |
| --- | --- | --- |
| Driver-only free-memory accounting falsely rejected S16384 after Native attention populated the allocator | Include reusable reserved storage, excluding active/pending blocks | DiT S16384 subsequently passed |
| Two logical S32768 ranks exhausted the 5060 Ti while retaining large, already-consumed attention/rotation intermediates | Release those Python tensor references after their last consumer; preserve operation order | Both S32768 ranks passed bit-exact after the change |
| Constructing/moving the Native video comparator under inference mode materialized FP16 weights, changing INT8 dispatch and creating an unfair mismatch | Construct the comparator outside inference mode, under no-grad; verify the physical INT8 payload exists | Raw and public video comparisons then passed bit-exact |
| Module copying replaced weak-referenceable hook registries with plain dictionaries | Preserve container types when mapping tensor fields | Native module hooks and full entry points execute successfully |
| A bounded handoff queue could discard its source owner before waiting for the oldest copy | Retain the owner through event completion | Eight queued-copy backpressure cases passed |

These fixes do not establish the root cause of the historical overlapping PP failure, and do not qualify the old PP scheduler. The video comparator repair also does not change the historical slab approximation errors.

## Executed validation

All listed runs use trained checkpoint weights and actual CUDA operators/copies on the RTX 5060 Ti. Two logical ranks share that physical GPU. Numerical references execute Native with equivalent inputs and dispatch. Local times are not dual-T4 performance evidence.

| Artifact | Executed scope | Result |
| --- | --- | --- |
| `semantic_results/weight_residency.json` | 12 delayed-copy cases; three fixed/adaptive window fixtures; two trained TE blocks at S512/2048/4096 on both ranks; two trained DiT blocks at S8192/16384/32768 on both ranks, each evaluated twice; trained 36-layer video decoder raw/public compact paths | 29 comparisons, all bit-exact |
| `semantic_results/weight_residency_transport.json` | Current handoff implementation: 12 delayed copies, eight queued-copy backpressure cases, three fixed/adaptive window fixtures | 23 comparisons, all bit-exact |
| `semantic_results/windowed_full_50_video.json` | Complete 50-layer Native TE entry point, once-per-request text refinement, complete 50-layer Native audiovisual DiT entry point with references, and 36-layer public Native video decode, on both ranks through the shared adaptive window | Six comparisons, all bit-exact |

The full-entry fixture uses TE S512 and compact packed DiT S68. It passes TE output into Native text refinement, then applies a deterministic latent update before video decode. It is a composed entry-point test, not a production sampler or five-second generation. Vision/reference encoders are represented by precomputed fixture conditioning. Video tiling is disabled at the compact geometry in both paths.

The trained DiT block tests assert one load per shard across two repeated evaluations when the working set fits. They do not prove all 50 DiT shards remain resident at production sequence lengths. The full-entry test contains one denoising evaluation; its no-reload result is not evidence of sampler-loop persistence.

In the full-entry run, measured peak live/pending physical weights were 3235.04 MiB and 2666.15 MiB for logical ranks 0/1 against a 4096 MiB limit each. Peak admitted pinned preparation was 4466.01 MiB and 3877.09 MiB against a 6144 MiB limit each. These exclude other allocations and are not physical T4 VRAM measurements. Both windows adapted from lookahead two to four.

Artifacts include source hashes. The shape matrix predates the final handoff ownership adjustment and preprocessing adapter; its exact residency implementation is retained in `semantic_results/residency_validated_source/weight_residency.py`. Current transport and full-entry artifacts validate the later implementation within their stated scopes. Failed intermediate artifacts remain available; the passing files do not erase those failures.

## Reproduce locally

Use the project's ComfyUI Python environment and the four checkpoint paths documented in the integration plan. The scripts add the pinned local dependencies themselves.

```powershell
& 'C:\ProgramData\miniconda3\envs\comfyui\python.exe' benchmarks/simulate_weight_residency.py --transport-only --output semantic_results/weight_residency_transport.json
& 'C:\ProgramData\miniconda3\envs\comfyui\python.exe' benchmarks/simulate_weight_residency.py --long-matrix --output semantic_results/weight_residency.json
& 'C:\ProgramData\miniconda3\envs\comfyui\python.exe' benchmarks/simulate_windowed_full.py --layers 50 --adaptive --include-video
```

These commands overwrite their named result files. Preserve earlier artifacts before rerunning if historical source hashes need to remain available.

## Next implementation and qualification

The next-request host cache is now implemented and locally validated: persistent pageable weights, independent disk/H2D lookahead, process-shared admission, foreground priority, and TE preloading after video decode. See [host cache implementation and physical gate status](HOST_CACHE_IMPLEMENTATION.md) for current evidence and remaining ComfyUI integration work. This extends the first milestone; it does not retroactively change the source versions recorded in earlier artifacts.

1. Connect this API to long-lived rank workers and ComfyUI's actual conditioning/sampler lifecycle. Preserve vision/deepstack, guidance branches, Native model-management behavior and request cancellation. Construct the complete ordered use plan for the actual number of denoising evaluations. Release persistent model-shell allocations at stage boundaries as well as window-owned blocks.
2. Validate that integration locally with changing denoising inputs and repeated requests, then freeze the source for one bounded dual-T4 correctness job. Exercise real cross-device handoff and NCCL on both ranks; stop at the first failed gate and retain partial artifacts. Do not restart the broad historical benchmark sweep.
3. After physical correctness, compare the integrated synchronous TP path, cache retention, and asynchronous window using identical work and timing boundaries. Measure cold/warm loading, full sampler latency, H2D stalls, communication, peak allocated/reserved VRAM and host memory. Attribute improvements against real Native caching/preprocessing behavior.

Native public video and Native audio remain the defaults. Two-slab video and audio channel parallelism retain their historical failed qualification gates and are not enabled by this residency work. No cluster job was launched for this local implementation milestone, and no end-to-end speedup is claimed.
