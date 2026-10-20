# Prescribed MiniMax-H3 TE / DiT benchmark

Use TE **S512, S2048, S4096** and DiT **S8192, S16384, S32768**. The optional TE S8192 / DiT S65536 stress cases are excluded. No workflow token counting or private reference media is needed.

DiT S means the **total native packed sequence**, including target video/audio, conditioning, reference image and reference video tokens. Reproducible fixture values realize these exact totals through Native `PackedLayout`, preprocessing, modality tags, AdaLN and RoPE. These shapes are not estimates of the saved workflow's sequence or pixel-resolution performance. TE uses the trained INT8 checkpoint's text decoder; the workflow's separate NVFP4/vision stack is outside this comparison.

| Experiment | Workload | Purpose |
|---|---|---|
| Native / legacy TP / optimized TP block ablation | All three lengths for each model; additional TE true B2 at S512 and two independent DiT requests at S8192 | Quantify the causes of legacy TP overhead and targeted changes |
| Native / optimized TP2 / balanced PP2 full latency | TE at all three lengths; prioritize DiT S8192, then bounded larger capacity/time probes | Complete 50 trained layers and native entry/output operators |
| Independent-request throughput | TE S512 cohorts M4/M8; DiT S8192 M4, microbatch 1 | Compare identical requests, including pipeline fill/drain and scheduling |

Full TE includes embeddings and 50 decoder layers. Full DiT is one complete native audiovisual Ref2VA denoiser step with both output heads; vision/VAE encoding and the sampling loop are excluded. Required DiT preprocessing weights are staged from immutable CPU parameters; transfer, synchronization and TP's single preprocessing broadcast are inside every strategy's timed boundary. Peak temporary increments therefore include staged weights, not only activation buffers.

Physical execution uses **2× Tesla T4**, the same pinned ComfyUI/kitchen/runtime and full checkpoints verified by SHA256. The local 5060 Ti simulates two ranks/streams only to check correctness. No local timing is performance evidence. Local coverage combines complete 50-layer TP at compact native inputs, trained-block checks at every prescribed length, and native-entry PP with two distinct trained layers and repeated independent requests. Physical complete-model pre/post gates remain required.

Block runs use at least three warmups and a common 0.5-second warmup minimum, then nine synchronized samples. Full latency uses one warmup and three samples; throughput uses three complete cohorts. Pre/post gates require relative L2 ≤ 1e-5, cosine ≥ 0.999999 and maximum absolute error ≤ 0.005, with both TP ranks agreeing. Bit equality is reported separately. Sample counts and within-session uncertainty will be explicit; token-length choices alone do not establish statistical confidence.

Profiler passes are separate from ordinary timing, with tensor-shape retention disabled to preserve memory headroom. Archive both TP ranks, per-device kernel groups, NCCL transport logs, CUDA Events, allocation/reservation peaks and pipeline stage/completion events. Completion intervals use chronological order. Report measured fill/drain and kernel-free intervals separately from the ideal `1/(M+1)` bubble fraction; a four-request cohort does not establish asymptotic steady-state throughput.

Run each shape/strategy in a fresh process and preserve successful rows immediately. Complete the primary latency comparisons and block matrix before expensive throughput/larger full-model probes. No automatic retries. The allocation has a 20-minute setup limit and a 90-minute benchmark ceiling, with shorter per-job limits. These are ceilings, not an ETA. Capacity failures and time-capped cases produce no accepted speedup or extrapolated full-model latency.
