# Video VAE: physical dual-T4 KV profiling

The **Native-aligned query candidate is bit-exact and faster than Native on all
three raw-decoder workloads**. The largest fixture falls from 18.026 s to
9.877 s (1.825×, 45.2% less time). Independent cropped slabs remain incorrect.
Recommend the candidate for integration qualification; do not deploy cropped
slabs or describe this as a measured full-pipeline speedup.

Notebook: [completed version 1](https://www.kaggle.com/code/chinokiki9/minimax-h3-vae-kv-profile).
Two physical Tesla T4s, 15 GiB each, driver 580.178.04, torch 2.11.0+cu130,
comfy-kitchen 0.2.37, NCCL 2.28.9, pinned ComfyUI
`d49e888586dd8ae012c0667b33466b815fee07f7`. Full trained 36-block checkpoint.
The job completed inside its cap, without OOM. Download recovery required fixing
a concurrent path-resolution issue in the local collector; the GPU job was not
rerun. All 21 cases and 42 compressed profiler traces were recovered and validated.

## Measured comparison

Median wall time in milliseconds, one warmup and three synchronized samples.
Identical input bytes and fresh Native references on the same allocation.
Timing includes prefix handoff and output join; GPU-resident latent to complete
GPU-resident raw pixels on rank zero. Model loading and tracing are excluded.

| Approach | 768×768 / 4 raw frames | 768×1344 / 8 raw frames | 512×512 / 124 raw frames | Physical correctness |
|---|---:|---:|---:|---|
| Native: one active GPU | 446.6 | 2,190.2 | 18,025.6 | Reference |
| Independent slabs | 249.2 | 1,038.4 | 9,024.7 | FAIL: 10.01%, 7.53%, 5.00% L2 error |
| Global RoPE only | 250.4 | 1,062.4 | 9,188.8 | FAIL: 5.93%, 4.89%, 3.76% |
| Shared KV + global RoPE | 352.0 | 1,534.9 | 12,390.2 | FAIL: 1.64%, 1.59%, 1.61% |
| Duplicated full-query control | 436.9 | 2,060.3 | 16,713.5 | Bit-exact on all three |
| Prequantized query-block control | 473.9 | 1,902.1 | 12,457.1 | Bit-exact on all three |
| **Native-aligned query candidate** | **321.7** | **1,481.2** | **9,876.8** | **Bit-exact on all three** |

Only correctness-passing approaches can earn speedup claims. The duplicated
full-query control's 2.2% advantage on the smallest fixture is within the Native
sample spread and does not establish an advantage.

| Workload | Native sample range, ms | Candidate sample range, ms | Candidate / Native speedup | Candidate peak allocated GiB, GPU0 / GPU1 |
|---|---:|---:|---:|---:|
| 768×768 / 4 raw frames | 436.6–450.3 | 318.8–321.9 | 1.388× | 2.783 / 2.757 |
| 768×1344 / 8 raw frames | 2,188.0–2,196.6 | 1,466.6–1,487.8 | 1.479× | 3.131 / 3.107 |
| 512×512 / 124 raw frames | 17,911.7–18,166.1 | 9,800.1–9,884.4 | 1.825× | 4.611 / 4.532 |

Native's active GPU peaks at 2.800, 3.295 and 5.219 GiB respectively. The harness
also loads a replica on Native's idle GPU1; that allocation is not necessary for
single-GPU Native operation. The candidate replicates model weights, so lower
per-device memory does not mean lower aggregate memory than single-GPU Native.
For the largest candidate, peak reserved memory is 5.646 / 5.377 GiB; reusable
transport buffers account for 0.641 GiB on each rank, included in allocated peaks.

## Why the candidate wins

It computes the Native prefix once, partitions the flattened sequence at a
128-query boundary, and retains global positions and Native query scale groups.
Each rank executes only its owned queries and FFN rows, against freshly gathered
global KV in every layer. The five suffix tokens are owned once. There is no
overlap/fade, full-Q gather, or attention-output gather inside transformer blocks.
Final projected patch outputs are joined once. This is global-attention token
partitioning, not two independent spatial crops.

The full-query control computes all queries' attention twice. The query-block
control avoids that duplication but gathers Q and attention outputs every layer,
prequantizes full Q/K/V on each rank, and keeps duplicated overlap FFN rows.
Removing those costs improves the largest case from 12.457 s to 9.877 s,
**20.7% less time** than the exact query-block control.

The exact candidate costs 0.852 s (9.4%) more than the incorrect independent-slab
fixture on the largest case. That difference includes restoring omitted global
attention, KV communication and changing overlap work; it is not pure
communication overhead. The 1.825× Native comparison is the valid comparison.

## Bottleneck evidence

Separate traced decode; CUDA-event stages below are summed across 36 layers on
rank zero. Rank one's costs are similar. These are profiled costs, not the three
untraced timing samples. Nested transformer totals must not be added again.

| Largest candidate stage | Time, s |
|---|---:|
| Native INT8 attention API, including its Q/K/V quantization | 5.083 |
| Native FFN | 1.926 |
| KV packing + collective + layout | 1.698 |
| QKV projection + RMS/RoPE | 0.745 |
| Attention output projection | 0.243 |
| Prefix projection + handoff | 0.011 |
| Final projection + join | 0.045 |

Actual kernel events, excluding overlapping profiler annotations, show:

| GPU0 kernel category | Sum, s |
|---|---:|
| Attention kernel | 4.841 |
| GEMM kernels | 1.889 |
| NCCL kernels | 1.312 |
| Kernels named quantization/ConvRot | 0.665 |
| RMS/RoPE kernel | 0.223 |

These categories overlap conceptually with the stage table; do not add the tables
together. Fused dequantization is included in GEMM, and some normalization/layout
work remains in other kernels. The candidate has 956 / 987 recorded kernel
events on GPU0 / GPU1 in the traced decode. Summed annotated profiler rows would
double-count GPU work and are deliberately not used as a kernel breakdown.

The topology is PHB, with no NVLink. Direct GPU P2P is disabled by the inherited
benchmark policy. NCCL logs confirm **SHM/direct/direct** between the local ranks;
the Socket network plugin initialization is not evidence of TCP GPU collectives.
AllGather uses `RING_LL` in these traces. Actual wire-byte counters were not
collected. The ideal aggregate FP16 KV payload is 8.720 GiB over all 36 layers,
plus padding/protocol costs; this must not be reported as observed PCIe traffic.

Recorded blank gaps between GPU kernel/copy events are only about 2.7 / 2.5 ms
within each rank's profiled activity span. NCCL kernels can include waiting or
spinning, so this does not establish high SM occupancy or eliminate communication
stalls. KV completion remains a dependency before complete global attention.

## Limits that matter for integration

**This benchmarks `_decode_pixels`, not public `decode` or the sampler pipeline.**
The 124-frame fixture means raw output geometry. Native's public VAE applies
latent normalization, temporal chunking, overlap blending, frame trimming,
pixel finalization and output-buffer transfers. With the default public settings,
the same T=31 input is processed as six 7-token temporal clips after one padding
token, and yields 103 public frames, not 124. This was verified from the pinned
Native frame-planning functions without running another GPU benchmark.

Integration must replace raw decoding **inside the existing Native chunk plan**.
Using one global T=31 call to replace six Native calls changes semantics. The
9.877 s result must not be used as the VAE cost of an actual five-second pipeline.
Actual per-clip shapes, tile batches and sampler-produced latent distributions
still need qualification, including public outputs and output-buffer lifetimes.

All inputs are random latent fixtures. Only three repeated-decode timing samples,
one allocation and fixed approach order were used. This establishes a strong
initial speed advantage on the tested raw workloads, not cross-session confidence
intervals, visual-quality evidence, or full-pipeline performance. Local qualification
used torch 2.12.1 and a 5060 Ti; the cropped-prefix query controls fail the long
local case despite passing the long T4 case. Their cross-runtime divergence is
unresolved. The aligned candidate passes both environments.

## Actionable recommendation

Use Native-aligned query ownership as the preferred VAE candidate. Keep Native
as fallback. First qualify its hook into the unmodified public temporal/spatial
plan on sampler-produced latents and batch variation, then measure whole Video
Decode and full-request latency with the residency manager active. No production
VAE patch or running P0/P1/P2 payload was changed by this profiling job.

Ranked remaining optimization opportunities within the candidate:

1. **KV transport.** Profile `NCCL_PROTO=Simple` against the observed LL path at
   these exact message sizes; probe CUDA peer access before testing P2P. Then
   consider packed INT8 KV transport only with Native global centering, anchor,
   scales and layout preserved. The observed NCCL budget is about 1.31 s; reducing
   it by half would save about 0.66 s (6.6%) *if* no other overhead replaces it.
   This is a sensitivity estimate, not a measured or promised gain.
2. **Packing/layout buffers.** Exchange writes directly into the final canonical
   KV layout where feasible; remove stack/transpose/pad/cat copies and avoid
   copying the locally owned KV back through a full gathered buffer. The measured
   KV stage exceeds total NCCL kernel time by about 0.39 s. That is an approximate
   optimization budget, not a clean attribution or a guaranteed saving.
3. **Short-call balance and launch costs.** The smallest case uses 1,280 / 1,029
   owned queries; a nearer balanced 128-aligned split is 1,152 / 1,157. Test this
   with Native byte comparisons, then assess buffer reuse and CUDA graph capture
   only if the actual public clip profiles show launch overhead. This matters
   more for Native's repeated shorter clips than for the balanced long fixture;
   no numeric full-request gain is established.

Attention is still the largest absolute cost (4.84 s of attention kernels), but
changing its algorithm, softmax order or quantization to reduce work risks the
Native correctness contract. It is not presented as a safe free speedup.

Evidence: authoritative results (`kaggle_vae_kv_profile_output/vae_kv_profile/results.json`, local-only evidence),
validated analysis and hashes (`kaggle_vae_kv_profile_output/vae_kv_profile/analysis.json`, local-only evidence),
GPU0 trace (`kaggle_vae_kv_profile_output/vae_kv_profile/512x512_124frames_aligned_rank0.json.gz`, local-only evidence),
GPU1 trace (`kaggle_vae_kv_profile_output/vae_kv_profile/512x512_124frames_aligned_rank1.json.gz`, local-only evidence),
CUDA-event stages (`kaggle_vae_kv_profile_output/vae_kv_profile/512x512_124frames_aligned_rank0.stages.json`, local-only evidence).
