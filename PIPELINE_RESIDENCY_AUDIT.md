# Full MiniMax-H3 lookahead and cache residency audit

The repaired physical 2xT4 run completed. Cold and warm outputs are bit-exact against Native. Adaptive streaming recorded 2.1% lower cold latency and 3.0% lower warm latency than the saved synchronous run, across separate Kaggle jobs. Those observations do not establish a statistically reliable speedup. The implementation does not yet demonstrate substantial transfer hiding or minimum single-request latency: only about 35 ms of weight H2D overlapped computation in the profiled DiT evaluation, while roughly 44 s remained without recorded GPU activity.

## Accepted physical comparison

| Measurement | Sync cold | Adaptive cold | Sync warm | Adaptive warm |
| --- | ---: | ---: | ---: | ---: |
| End-to-end | 455.404 s | 445.678 s | 437.717 s | 424.579 s |
| Startup-inclusive | 459.845 s | 449.254 s | 437.717 s | 424.579 s |
| VAE construction | 5.788 s | 10.251 s | Retained | Retained |
| Reference encoding + TE | 138.581 s | 137.481 s | 124.823 s | 123.488 s |
| Four-evaluation sampler | 276.595 s | 263.335 s | 279.462 s | 267.202 s |
| Video decode | 32.700 s | 32.889 s | 32.346 s | 32.771 s |
| Audio decode | 1.738 s | 1.719 s | 1.084 s | 1.114 s |
| Native comparisons | Five bit-exact | Five bit-exact | Five bit-exact | Five bit-exact |

Both jobs used actual Tesla T4 devices, Torch 2.11.0+cu130, all 50 trained TE/DiT layers, Native vision/DeepStack/reference encoding, Native audiovisual res_multistep sampling and complete Native video/audio VAEs. The request is 512x512, 124 frames at 24 fps, CFG=1, seed 1729, with identical positive/negative conditioning and a controlled RGB reference. Shapes: conditioning [1,279,5120]; video latent [1,24,37,32,32]; audio latent [1,32,2,207]; pixels [1,124,512,512,3]; waveform [1,165600,2] at 32 kHz. Reference tokens are included through the Native frontend.

The physical Native reference is preserved from the earlier completed run; its source/artifact hashes were verified. Only two qualified memory-policy methods changed. Model/operator sources and request shapes match. Native timing is diagnostic, not a competitive Native latency baseline. Performance evidence compares synchronous versus adaptive TP2. The previously studied 2-Slab/channel-parallel VAEs are not active in this audit.

The baseline has depth one, current-stream H2D and a consumer-stream fence before leases; CPU preparation can still overlap previously queued GPU work. The candidate uses a separate copy stream and adaptive depth. Both use 3 GiB host caches per rank, a shared 24 GiB host cap, 9 GiB DiT weight cap, 4.5 GiB DiT activation/headroom reserve, and separate TE limits. Validation and profiler work are excluded from request timing. Warm reruns TE without graph caching, retains the parent VAEs and gets CPU prefetch lead time during validation. Storage coldness is unverified. Video/audio container export and GUI execution are outside the boundary. One observation per temperature, different Kaggle jobs, and variable VAE construction costs limit causal claims; the earlier correct sync cold run was 440.247 s.

## Audit verdicts

| Area | Verdict | Actual evidence |
| --- | --- | --- |
| Native correctness | PASS for tested workload | All five comparisons have max absolute error zero for both temperatures and strategies. Four actual denoiser evaluations per request. |
| Adaptive lookahead | PARTIAL | Recorded depth changes 2->3->4 and later decreases in DiT. EMAs alter depth, but include enqueue/peer waits. A single preparation producer limits throughput. |
| Deadline and cross-model scheduling | FAIL | Deadlines are next-use indices, not wall-clock readiness deadlines. No global TE->DiT->VAE GPU prefetch. Next-request TE layer prefetch starts after video decode. |
| Cache efficiency | PARTIAL; DiT reuse PASS | Both strategies load 55 DiT blocks per rank for 200 uses: 72.5% of reloads avoided. Host demand hits are 3/105 cold and 9/105 warm. Native full blocks are read/prepared independently on both ranks. GPU windows drop at end_sample. |
| Async and memory safety | PARTIAL; repaired 512 case PASS | Ready/last-use events and record_stream fence lifetimes. Both physical temperatures finish without observed OOM, zero admission denials and zero final reservations. Higher-resolution/transient-buffer guarantees remain untested. Parent and allocator ownership exceed the managed ledger's scope. |
| Transfer contention | PARTIAL | Weight H2D overlaps compute by only 2.2%/4.2% of weight-copy activity; no measured H2D/NCCL overlap in this trace. This provides little exposure to the high-contention scenario the design intends to exploit. |
| End-to-end benchmark | PARTIAL | Actual full-depth cold/warm comparison and traces completed; cross-job N=1 results support exploratory differences, not reliable speedup or optimality. |

Worker packets pass through CPU and CUDA completion fences before acknowledgement. Asynchronous inter-model result delivery and persistent inter-request GPU residency are not implemented.

## Memory and cache evidence

| Job-wide peak or final metric | Synchronous | Adaptive |
| --- | ---: | ---: |
| System-used RAM | 23.423 GiB | 23.602 GiB |
| Minimum available RAM | 7.924 GiB | 7.745 GiB |
| Process-tree RSS | 22.241 GiB | 22.425 GiB |
| Managed host reservation peak | 15.643 GiB | 20.158 GiB |
| NVML board-used peak, ranks 0 / 1 | 14.044 / 13.946 GiB | 14.933 / 14.911 GiB |
| Worker CUDA reserved peak, ranks 0 / 1 | About 13.252 / 13.252 GiB | About 14.199 / 14.262 GiB |
| Final managed reservations / denials | 0 / 0 | 0 / 0 |

Adaptive NVML board capacity is 15 GiB per GPU; CUDA-accessible capacity is 14.5625 GiB. Board-used memory includes allocator reservations. Admission also considers the worker's reusable inactive CUDA cache; low driver-free memory alone does not mean no usable headroom. Independent parent/worker peaks must not be added. Peaks include the extra diagnostic invocation. The test reaches near-full reserved VRAM, but leaves substantial system RAM available: near-maximum safe host-cache utilization is not established.

Increasing cache capacity to 6 GiB per rank failed local Native DiT startup admission and was rejected before physical submission. It is not a proven physical T4 limit. The result shows why retained caches cannot outrank foreground construction/transient-memory admission.

Managed H2D and logical checkpoint reads were unchanged by adaptive depth: TE 11.361 GiB per rank; DiT 13.039/10.057 GiB per rank. Cold logical reads were 47.346/45.857 GiB, warm 45.529/44.039 GiB. Logical reads are not physical disk traffic; cold misses in a one-pass model are not automatically cache thrashing. DiT GPU reuse is already effective. Adaptive TE evicts 49 once-used blocks per rank and performs one foreground allocator reclamation per rank per request, costing 0.31-0.36 s. Retaining completed single-pass TE blocks provides no reuse for this workload.

## Profiler and loading bottlenecks

Traces cover one extra full-depth, one-evaluation DiT invocation, excluded from cold/warm timing. Detailed TE/VAE traces and a four-evaluation steady-state trace are absent. Kernel classification uses names; GEMM includes Native Hadamard rotation, and ambiguous quantization kernels remain in other. Sums are GPU activity, not wall time or SM occupancy.

| Activity | Sync rank 0 / 1 | Adaptive rank 0 / 1 |
| --- | ---: | ---: |
| Attention | 36.345 / 30.995 s | 35.521 / 33.560 s |
| GEMM | 6.894 / 5.414 s | 6.448 / 6.833 s |
| Identifiable quantization/layout | 1.249 / 1.155 s | 1.216 / 1.293 s |
| NCCL kernels, including peer waiting | 5.546 / 22.220 s | 4.821 / 16.122 s |
| Weight H2D activity | 1.998 / 1.308 s | 1.556 / 0.823 s |
| Weight H2D/compute overlap | 0 / 0 | 0.035 / 0.035 s |
| No recorded kernel/copy/memset activity | 45.432 / 45.458 s | 44.087 / 44.211 s |
| Kernel launches | 5,246 / 5,164 | 5,246 / 5,164 |

Removing consumer-stream fences changes where waiting appears: CPU cudaStreamSynchronize totals fall from about 51.8 s to below 0.1 s, but inclusive CPU weight-lease scopes rise from about 46 s to about 94 s. Those async lease waits overlap GPU execution. Their increase is not an increase in exposed loading stalls. Cold DiT CUDA lease gaps remain about 54.5 s baseline versus 55.7 s adaptive on rank 0; TE gaps remain about 109 s. Lease gaps include dispatch/preparation/copy waits and are not pure PCIe time. The GPU-activity-free trace span is the stronger evidence that substantial loading/dispatch stalls remain.

NCCL uses SHM/direct/direct with P2P and IB disabled identically in both jobs. Rank 1 broadcast activity remains about 10 s, including rank 0 startup/preprocessing wait; collective kernel time is not wire time. This comparison does not establish optimal transport settings. No significant H2D/NCCL coexecution was observed, so causal interference under deeper overlap remains unverified.

NVML sampled PCIe estimates include all traffic and protocol/polling effects. For the cold sampler, estimated RX/TX GiB were sync 114.3/110.3 on rank 0 and 329.6/124.1 on rank 1; adaptive 166.9/109.6 and 158.5/115.3. Weight payload bytes are identical despite those shifts, so bus estimates cannot establish redundant weight loading or payload savings. Full per-stage estimates are saved separately. Torch pinned-memory counters reported impossible current/peak values and are rejected as physical RAM residency evidence.

## Repairs and validation

The prior physical adaptive run failed at final TE position 49: zero live window weights, a required 243,978,752-byte block, and insufficient apparent memory against the 1.5 GiB reserve. A trained-Qwen local reproducer showed zero allocated tensors but 40 MiB active in deferred record_stream allocator state. Foreground-only empty_cache retry in ResidencyWindow._room drains that state, preserves event fencing and rechecks capacity. The physical rerun exercised the repair and completed both requests. Speculative admission stays nonblocking.

The failed run also retained 9.292 GiB in the managed host ledger after workers were dead. PhysicalTPService.abort now resets service-owned accounting only after all workers have been reaped and replaces a lock whose owner may have been killed. A real spawned-worker test terminated a lock holder and verified zero reservations and accessible diagnostics.

Both current-source, full-50-layer local cold/warm qualifications passed before submission. Local logical ranks share the 5060 Ti, use smaller GPU windows and a 1 GiB available-RAM reserve versus 3 GiB physically; they qualify semantics/lifetimes, not dual-T4 performance or capacity. Earlier parent-model reclamation, unrelated-TE-tail eviction, directly owned Native refiner backing and fenced pinned-cache cleanup remain in place. Source checks permit only the two AST-scoped memory-policy changes when reusing Native reference outputs.

## Top three remaining opportunities

1. **Directly construct the INT8 shards each rank owns.** Remove full Native block construction and redundant payload reads/copies while preserving packed QKV/gate ordering, scales, replicated norms/AdaLN, dtype conversions and operator dispatch. This has the largest measured opportunity: about 109 s TE readiness gap and 55 s DiT lease gap per rank in the cold request. These are ceilings on the relevant critical-path waits, not promised savings or additive rank costs. Use bounded tiled reads or a validated shard format; naive per-row disk reads could add syscall overhead.
2. **Preload/reuse Native TE startup and vision CPU backing across decode.** Video decode provides about 32.8 s of lead time, versus the present 1.7 s audio interval. The layer prefix already benefits from validation lead time in this benchmark; additional gains require startup/vision work or real consecutive-arrival measurements. Foreground VAE/DiT allocations must have priority under one shared, byte-accurate transient budget. Expected benefit is bounded by needed preparation and the decode interval; it applies to the next request.
3. **Separate CPU preparation horizon from GPU residency and staging.** Queue readiness by predicted completion deadlines, retain DiT reuse, retire single-use TE weights promptly, and reuse bounded pinned/device buffers. Adaptive allocation counts rose from 56/58 to 175/231 cudaMalloc calls in the trace, without substantial copy overlap. Immediate measured reclaim cost is only 0.31-0.36 s per rank/request; the larger value is reducing memory churn and making earlier preparation possible. A deeper GPU window alone has little demonstrated latency upside here.

Keep the repaired implementation as a qualified candidate for the tested 512 workload, and preserve the synchronous configuration as an ablation. Prioritize the loader and CPU ownership/scheduling work before expanding residency. Further performance claims require Native validation and actual hardware evidence for the changed implementation. Higher resolutions, longer sampling schedules, production CFG variants, additional references and full allocation-lifetime profiling remain outside this bounded experiment.

## Evidence

The focused physical continuation is `chinokiki9/minimax-h3-pipeline-tp-audit` version 3. It completed with 708.916 s setup and 1,033.085 s adaptive work, with no automatic retries. Baseline is continuation version 2; Native reference is `chinokiki9/minimax-h3-pipeline-residency-audit` version 3.

Combined results, original traces and explicit cross-job provenance are in `kaggle_pipeline_audit_output/comparison`: `analysis.json`, `comparison_provenance.json`, `cpu_annotations.json`, `stage_pcie_summary.json`, and both strategies' rank traces/results. Original job evidence remains in `kaggle_pipeline_audit_output/tp_v2/pipeline_audit_results` and `tp_v3/pipeline_audit_results`. Repair qualification evidence is `semantic_results/async_allocator_reproducer.json`, `abort_budget_ownership.json`, and `pipeline_audit_depth4_local/{sync,adaptive}_result.json`.
