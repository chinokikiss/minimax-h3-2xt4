# Physical dual-T4 VAE profiling

**Completed:** all 21 physical cases and 42 traces were recovered and validated.
The aligned candidate is bit-exact and achieves 1.388× / 1.479× / 1.825× raw
decoder speedups. See [results, bottlenecks and public-VAE limits](VAE_KV_T4_RESULTS.md).
The 124-frame workload is raw geometry; Native public temporal decoding uses a
different chunk/frame plan and still requires integration qualification.

Submitted notebook: https://www.kaggle.com/code/chinokiki9/minimax-h3-vae-kv-profile

Version 1 requests NvidiaTeslaT4 (two physical GPUs). The existing P0/P1/P2
notebook remains independent. Setup is capped at 900 seconds, profiling at
1800 seconds, and the whole notebook at 2700 seconds. No automatic resubmission.
All model bytes are downloaded on Kaggle; the submitted payload contains source,
qualification evidence and two small saved latent tensors only.

## Comparison

Native, independent slabs, global RoPE only, shared KV/global RoPE,
duplicated full-query attention, prequantized query-block control, and disjoint
Native-aligned query ownership. All execute the trained 36-block decoder.

Timing starts with GPU-resident input and ends with complete GPU-resident raw
pixels on rank zero. It includes candidate prefix handoff and final output join.
Model loading, correctness CPU copies and profiler iterations are excluded.
One warmup, three synchronized wall/CUDA-event samples per workload/mode. A
separate profiler iteration writes compressed Chrome traces and operator tables
on each rank, plus CUDA-event stage breakdowns. NCCL initialization/graph/
collective logs and nvidia-smi topology are saved. The same inherited transport
policy as earlier experiments disables direct P2P and IB; logs identify the
actual transport. Traffic formulas below are not observed wire-byte counters.

The three workloads are the two hash-verified historical latent inputs,
`[1,24,1,48,48]` and `[1,24,2,48,84]`, plus a seeded synthetic
`[1,24,31,32,32]` latent, decoding to 512×512 and 124 frames (approximately
five seconds at 25 fps). The latter uses realistic geometry, not sampler-produced
latent values. Public VAE normalization/cropping and pipeline end-to-end latency
are outside this test.

## Qualification

The final local run completed 21 mode/workload cases on two logical ranks of a
5060 Ti. The candidate is Native bit-exact on all three workloads. The two
query-layout controls are exact on the short fixtures but fail the longer case
at 1.608% relative L2. Their cropped prefix path remains a diagnostic, and its
long-sequence divergence needs localization before any deployment. No failing
control earns an accepted speedup. Physical correctness is checked again
against fresh Native outputs on the same allocation.

The harness loads checkpoints outside inference mode, preserving QuantizedTensor
and Native INT8 attention dispatch, and runs *all* decodes inside inference mode.
A discarded harness trial loaded inside inference mode, changed the parameter
type and Native dispatch, and was terminated locally. Its measurements are not
qualification evidence. Only the final source hash is packaged.

## Cost hypotheses to measure

| Workload | Tokens including five suffix tokens | Ideal aggregate FP16 KV payload over 36 layers | Global attention work / independent 128px-overlap slabs |
|---|---:|---:|---:|
| 768×768, latent T=1 | 2,309 | 0.634 GiB | 1.465× |
| 768×1344, latent T=2 | 8,069 | 2.216 GiB | 1.666× |
| 512×512, latent T=31 | 31,749 | 8.720 GiB | 1.280× |

The attention column compares pair counts, not total decoder time. Independent
slabs omit context and cannot establish a valid speedup. The exact candidate
does Native's total global attention work across two ranks and removes duplicated
overlap projection/FFN rows (14.4%, 8.7%, and 20.0% fewer than these slab cases).
Global RoPE alone needs no additional collective. Full-query controls duplicate
attention or add Q/attention-output gathers. Query-block control adds roughly
another KV-sized aggregate payload for its Q plus output exchanges.

The static-shape candidate uses one fused K/V allgather per layer, with known
ownership sizes and reusable buffers; no per-layer size exchange or host
`.item()` synchronization. Padding, layout conversion and collective buffering
increase costs beyond the ideal traffic formula. Its full KV tensors require
about 18, 63 and 248 MiB per rank per layer respectively, not all 36 simultaneously;
these figures exclude local activations, transport buffers and model replicas.
Native weights are replicated on both GPUs. Actual peak allocated/reserved VRAM
and reusable buffer sizes are measured per rank.

The stage timing aggregates are nested: transformer-block time includes attention
and FFN; KV stage includes packing, collective and layout conversion. Do not add
overlapping totals. Profiler/NCCL traces separate their kernels and stalls.

Three samples give an initial cost profile, not a statistically robust speedup
claim. Recommendations require physical correctness and an observed improvement
larger than timing spread. No physical performance result is claimed at submission.

Evidence: `semantic_results/vae_kv_profile_local/results.json`,
`kaggle_vae_kv_profile/build_manifest.json`, and the submitted frozen notebook.
The evidence watcher downloads only `vae_kv_profile/` results/traces after completion.
