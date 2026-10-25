# One DiT denoiser call on 2×T4 — results and profiling limits

The one-call experiment **completed on two distinct Tesla T4 GPUs (SM75)**.
All **36 output artifacts are downloaded**. The first transfer broke mid-file;
the user authorized resuming it. Existing complete files were preserved. The
source capsule binding and decompressed SQLite SHA256 are verified. No Kaggle
status/list/live-log polling or additional GPU run was performed.

The frozen API gates passed 48 local and 48 Kaggle CPU tests. GPU versions are
Torch 2.11.0+cu128, CUDA 12.8, Kitchen 0.2.37 and TileLang 0.1.13.

## Completed single-call measurements

| Measurement | Rank 0 | Rank 1 |
|---|---:|---:|
| DiT calls / trained attention blocks | 1 / 50 | 1 / 50 |
| CUDA-event command elapsed | 161.015 s | 161.016 s |
| Host enqueue span | 154.604 s | 154.615 s |
| Peak PyTorch allocated VRAM | 10.175 GiB | 10.510 GiB |
| Peak PyTorch reserved VRAM | 13.877 GiB | 14.283 GiB |
| Separate one-block warmup | 36.878 s | 46.646 s |

The parent denoise service call took **161.094 s**, including worker completion;
the Native sampler wrapper took **167.282 s**, including auxiliary refinement
and other wrapper work. The host enqueue span ends before GPU completion and
must not be reported as completed-call latency. Conditioning and both initial
noise hashes match the prior E2E exactly. Both returned sampler outputs are
finite; this is not an accuracy-baseline or generation-quality comparison.

## Phase evidence and profiler limits

The table below contains **sums of nested CUDA stream event intervals**, in
seconds across all blocks on each rank. These include dependency waits and
host launch gaps. Rows overlap and cannot be added as exclusive kernel time.

| Observed interval | Rank 0 | Rank 1 |
|---|---:|---:|
| Column INT8 GEMMs | 54.020 | 47.924 |
| Kitchen attention | 49.752 | 47.264 |
| Row INT8 GEMMs | 9.111 | 5.621 |
| Scaled-FP16 SUM transport | 12.679 | 16.162 |
| FC2 fused SwiGLU/ConvRot/quantization | 2.221 | 5.951 |
| Modulated normalization | 4.710 | 4.736 |
| Gated residual addition | 4.043 | 4.058 |
| Transport encode | 0.654 | 0.659 |
| Transport fused epilogue | 0.784 | 0.785 |

These observations prioritize GEMM, attention and collective/wait analysis in
the next profiling pass, but do not prove their exclusive critical-path shares.
The host records also contain large normalization/residual thread-CPU spans;
without valid worker kernel/launch correlations, their CPU-versus-GPU waiting
mechanism is unresolved. They must not be translated into GPU kernel durations.

**Nsight Systems coverage is incomplete.** The raw report has both rank NVTX
denoiser ranges and semantic scopes, but **zero CUDA kernel events from either
rank process**. Its 44 CUDA kernels belong to the parent process. The returned
kernel/API summary CSVs therefore cannot describe DiT GPU execution or its GPU
utilization. This is an actual coverage failure, not a zero-cost DiT result.
The installed Systems 2025.1.1 also reports that driver 13.0 is newer than its
supported driver 12.8 trace libraries; that warning alone does not prove why
worker events are missing. Worker termination before CUPTI flush is another
source-based hypothesis: runtime `close()` sends close and then calls `abort()`.
Neither hypothesis is presented as a verified root cause. Before another
capture, test an explicit worker-side profiler flush before the close reply and
require worker CUDA coverage as a benchmark gate, without changing production
runtime behavior or silently updating the frozen dependencies.

**Nsight Compute hardware counters are blocked by `ERR_NVGPUCTRPERM`.** No
memory-stall, occupancy, achieved-bandwidth or Tensor Core counter claims are
available. This run is one instrumented NFE, not a repeated steady-state or
concurrent-request benchmark.

Machine-readable phase, memory, input-hash, hardware and coverage evidence:
`D:/attention_eval/dit_single_profile_20261023/report_v1/verified_results.json`.
All raw artifacts remain under `gpu_output_v1/` in that evidence directory.

## Fixed workload and execution boundary

- Same L1 prompt and two hash-frozen images, 896×512, 362 frames, seed 20261020,
  CFG 1, video/audio shifts 12/2. Native packing must report **51,220 tokens**.
- Current production operators: Kitchen DiT attention, FP16-fix, fused-stored
  FC2 and scaled-FP16 SUM transport; 50 blocks, 28 heads/rank, head dimension
  128. The former E2E's optional scalar sparse override is not selected here.
- Offline DMAD strength 1, same immutable base/LoRA revisions and 208 native
  independent bake checks; no online LoRA attached during profiling.
- Prior E2E outputs did not retain initial conditioning. Real TE conditioning
  and reference-image latents are prepared once outside the profiled process.
  The temporary input file remains in private job scratch and is deleted after
  profiling; returned artifacts contain hashes/shapes/summaries, not activations.
- A separate process profiles the first interval of the original Native LCM
  schedule: sigmas `[1.0, 0.972972989…]`. The Native guider/sampler is preserved,
  but the interval contains exactly **one** denoiser evaluation. Each rank must
  report one call and 50 Kitchen attention invocations.
- One representative trained block warms exact operator shapes before that
  call. Warmup is reported separately; no second full DiT warmup call is run.
  Fresh auxiliary loading, text refinement and any remaining first-use work
  are retained as separate phases. One instrumented call does not establish
  a warmed-run latency distribution or steady-state throughput.
- No Video/Audio decoder, audio checkpoint download, full sampling trajectory,
  media export or generation-quality experiment is run.

## Evidence collected

The benchmark observers are isolated under `benchmarks/`; they invoke original
operators once without modifying scheduling or arithmetic. They provide NVTX
ranges, host wall/thread CPU spans, CUDA stream elapsed intervals, per-block
loading/wait/compute intervals, checkpoint reads, backend counts and NVML/RAM
samples on both ranks. Categories include QKV, modulated norms, QK normalization
and RoPE, Kitchen attention, attention output projection, FC1, FC2 fused
SwiGLU/quantization, INT8 GEMM, transport encoding/SUM/fused epilogue and residuals.

When installed, Nsight Systems captures the complete process tree with
CUDA/NVTX/OSRT events; the denoiser ranges identify the measured call. The raw
`.nsys-rep`, compressed SQLite export and kernel/API/memory/NVTX summary CSVs
are retained. The CLI process-tree controls follow the
[NVIDIA Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
Nested stream elapsed spans include launch gaps and dependencies and must not
be summed as exclusive kernel busy time. A fused attention kernel is not
reported as separate measured QK/softmax/PV kernels.

The installed Nsight Compute executable gets one small permission probe.
`ERR_NVGPUCTRPERM` is recorded as a hardware-counter blocker, following
[NVIDIA's explanation](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute).
That probe is not an actual-DiT counter measurement. A separate Compute replay
would exceed this one-NFE Systems-capture scope, so no memory-stall counters
will be invented if they are unavailable. Missing or failed profilers are
explicitly recorded; the DiT call is never automatically repeated to retry them.

## Reproducibility and delivery

The frozen ComfyUI legacy API, Kitchen 0.2.37 and TileLang 0.1.13 are preserved.
Kaggle's approved native Torch/CUDA family remains in place. Core packages are
installed only as the exact approved differences in an isolated overlay. The
source archive uses immutable GitHub commit
`41a900987933a375802b7386f5728ca3815160f3`, archive SHA256
`6128e10d8919b0036589bcf7e36a82bc284665336562ad500ae80a74787b2f47`,
capsule SHA256
`fc8e621c0e34e7e3dd09e3cba596685a4dda7c5994021d1dc99c81c3fc6cb596`.
Prior submitted sources/results and the user's working index remain untouched.

The job uses zero dataset/kernel-output mounts. CPU preflight preceded the
bounded dual-T4 submission, whose setup and execution caps are **240 s** and
**2,400 s**. Provider queue/startup is outside those code deadlines. The CPU
image contained neither Nsight tool; GPU-image availability is checked by the
execution job rather than inferred from that CPU result.
Setup, delivery, baking, conditioning, warmup and
profiled execution have separate receipts/timing boundaries. There is no Kaggle
status/list/live-log polling: each stage has one final artifact collection.

Evidence root: `D:/attention_eval/dit_single_profile_20261023/`.
The [aggregate record](../data/dit_single_call_profile_20261023.json) records
submission and final-collection state. The measurements above are hardware-verified; exclusive kernel attribution remains blocked.
