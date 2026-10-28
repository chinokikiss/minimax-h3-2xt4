# Two-T4 results: DiT cold/warm calls and VAE CUDA13 attribution

Both jobs completed. Final outputs were collected once per job without
status/list/live-log polling: 61 DiT files and 66 VAE files. Raw outputs,
traces and submitted archives remain preserved under `D:/attention_eval`.

These are separate experiments. DiT retains Torch 2.11.0+cu128 / CUDA 12.8;
VAE uses the explicitly authorized Torch 2.11.0+cu130 overlay. Both retain
Kitchen 0.2.37, TileLang 0.1.13 and the frozen ComfyUI source. Neither job
runs the complete H3 pipeline.

## VAE: CUDA linears reverse the sampled-block performance result

All 30 real-input replay cases passed on two physical Tesla T4 GPUs, SM75,
driver 580.178.04. The CUDA registry backend is available and enabled. Strict
lookup records `comfy_kitchen.backends.cuda.int8_linear` in the CUDA cases
and `comfy_kitchen.backends.eager.quantization.int8_linear` in eager cases.
Actual CUDA linear and attention execution, finite outputs and CUDA capture
passed before checkpoint downloads.

Native first-clip tiles from the saved real H3 latent produce block inputs
`[B,1797,2048]`. Layers 0, 17 and 35 are replayed with identical captured
inputs across the factorial. Each case has three warmup calls, seven timed
calls and a separate diagnostic CUDA profile. B=4 runs on GPU0; B=1 runs on
GPU1. They are independent deployment tile shapes, not a scaling comparison.

The following numbers are medians of the three sampled layers' warm medians,
in milliseconds per decoder transformer block:

| Linear implementation / attention | B=1 | B=4 |
| --- | ---: | ---: |
| INT8 eager / SDPA | 40.30 | 166.86 |
| INT8 CUDA / SDPA | 10.71 | 50.12 |
| INT8 eager / Kitchen | 39.83 | 162.11 |
| INT8 CUDA / Kitchen | **9.51** | **45.80** |
| FP16 checkpoint / Native SDPA | 14.52 | 63.05 |

At fixed Kitchen attention, CUDA linears deliver 4.19x and 3.54x block speedups
for B=1 and B=4 respectively. CUDA INT8 plus Kitchen attention reduces latency
by 34.5% and 27.4% against the FP16 block anchors. Those anchors use different
checkpoint weights and are not accuracy references.

The within-layer controlled effects are more informative than subtracting
medians across layers. Switching eager to CUDA linears saves 29.5–30.8 ms
at B=1 and 115.4–120.5 ms at B=4 with Kitchen attention unchanged. Switching
SDPA to Kitchen while holding CUDA linears fixed saves 0.68–1.62 ms and
2.55–5.88 ms respectively. Linear dispatch is the dominant performance cause
in these sampled blocks; attention is a smaller, beneficial component.

### Kernel evidence

Representative B=4, layer 17, Kitchen attention held fixed. Times below are
summed *actual kernel work* from one separate diagnostic capture, not the
seven-run warm medians above:

| Call site | Eager linear kernels / ms | CUDA linear kernels / ms |
| --- | ---: | ---: |
| QKV including norm/quantization | 24 / 27.107 | 2 / 5.525 |
| Attention output projection | 24 / 11.219 | 4 / 3.578 |
| FC1 including norm/quantization | 29 / 69.890 | 2 / 11.271 |
| FC2 including SwiGLU/quantization | 26 / 41.418 | 7 / 12.770 |
| Kitchen attention | 4 / 8.181 | 4 / 7.905 |
| Other norm/RoPE/layout | 3 / 2.994 | 3 / 3.105 |
| Entire block | **110 / 160.807** | **22 / 44.155** |

Eager dispatch launches four legacy `cutlass1x::...Igemm...` kernels totaling
86.199 ms, plus 55.586 ms of elementwise/layout kernels. CUDA dispatch selects
SM75 INT8 TensorOp/visitor GEMMs totaling 21.638 ms, fused ConvRot quantization,
and substantially fewer separate tensor passes. The runtime guard had therefore
prevented both the optimized GEMM selection and surrounding fusion. This is
not solely an attention-backend difference or a quantization launch count issue.

The captured B=4 peak allocated memory is approximately 3.75 GiB for eager
INT8, 2.89 GiB for CUDA INT8 and 4.94 GiB for the FP16 anchor. These include
the resident whole decoder and replay buffers; they are not standalone operator
workspace sizes or full-pipeline VRAM estimates.

### Corrected profiler bookkeeping

The submitted helper counted every CUDA-device event as a kernel. PyTorch
also returns synthetic `gpu_user_annotation` events on that device; adding
their durations double-counts work. Some extension launches also lack an
External id, which makes direct `key_averages()` phase totals incomplete.

The offline analyzer now reads Chrome events of category `kernel` only,
separates copies/memsets, and assigns kernels using runtime/driver launch
correlations and enclosing CPU call-site ranges. All 30 traces have complete
launch attribution. Six GPU annotation ranges are excluded in the representative
captures above. Raw traces and the original results remain unchanged. Two
CPU regression tests cover annotation exclusion and uncorrelated extension
launches. This correction does not change warmed latency or VRAM measurements.

### Scope of the VAE conclusion

The experiment strongly supports CUDA13-enabled linear dispatch as the fix
for the block-level INT8 slowdown. It does not establish a new full-decoder
speedup, quality equivalence, numerical error bound or complete-video result.
Other registry operations have the same automatic CUDA13 dispatch within the
factorial; their CUDA12.8-to-13 change is not isolated here. No production
environment or operator default has been changed from this experiment.

## DiT: both 59134-token calls completed with full trace coverage

The retry recovered from the previous HF download failure. The offline persisted
LoRA bake setup command took 337.863 seconds, including process startup, with
its native target/state proofs. This setup is outside both denoiser calls.
The performance fixture uses real trained weights and Native token layout,
with seeded latent noise and synthetic conditioning/reference activations.
It is not a replay of a complete generation trajectory or real prompt encoding.

Runtime policy: Kitchen DiT attention, `cuda_int8_tilelang` column linears,
`fused_stored` FC2 quantization, scaled FP16 transport and FP16_fix. The tested
window has CPU lookahead 16, GPU lookahead 2 and adaptive lookahead enabled;
weight/pin budgets are 6 GiB each, CPU weight cache 3 GiB, host budget 16 GiB.
The baked descriptor has no inference-time LoRA merge.

| Measurement | First call | Second call |
| --- | ---: | ---: |
| Parent denoiser RPC | 229.158 s | 176.148 s |
| Rank0 traced window | 227.546 s | 174.499 s |
| Rank1 traced window | 227.551 s | 174.495 s |
| Rank0 GPU idle gaps | 55.925 s | 2.136 s |
| Rank1 GPU idle gaps | 55.714 s | 2.935 s |
| Rank0 compute kernel union | 156.654 s | 156.414 s |
| Rank1 compute kernel union | 128.354 s | 132.777 s |
| Rank0 traced kernel/copy coverage | 75.42% | 98.78% |
| Rank1 traced kernel/copy coverage | 75.52% | 98.32% |

The second call's RPC latency is 23.1% lower. Both video and audio denoiser
outputs are finite and byte-identical between calls. Both calls have all 50
trained blocks correlated to GPU kernels on both ranks. Worker CUDA checkpoint
flush/restart/close proofs pass, both processes exit cleanly, and the Nsight
report plus compressed SQLite are complete.

### Why the first call is slower; what lookahead actually achieves

The first call records 11 TileLang compilation scopes per rank, totaling
45.684 and 45.743 seconds of inclusive wall time. The second call records no
compilation scopes. The longest first-call GPU gaps coincide with compilation;
rank0's longest is 8.708 seconds at the first block's column epilogue. Rank0
useful compute kernel time is almost unchanged between calls. These observations
identify cold compilation as the main visible source of the first-call gap,
alongside cold weight preparation and synchronization. Parallel rank compilation
times must not be added, and overlapping preparation cannot be summed into an
exclusive latency decomposition.

On the warm call, the existing lookahead/window configuration sustains traced
GPU kernel/copy activity over 98% of the call window. The GPU is not idle for
most of this profiled call. The largest remaining initial gap is 0.962 seconds
on rank0 and 2.061 seconds on rank1 while acquiring/preparing the first FC1
weights. This is an observed configuration result, not a causal lookahead-off
comparison or an occupancy/utilization counter measurement.

Warm rank0 major kernel work: Kitchen attention 68.298 seconds; column INT8
GEMMs 40.833 seconds; row INT8 GEMMs 13.570 seconds. Rank1's corresponding
values are 64.661, 28.351 and 8.421 seconds. NCCL kernels occupy 15.280 seconds
on rank0 and 38.115 seconds on rank1. The rank asymmetry indicates that NCCL
kernel time includes arrival/wait imbalance; it must not be labeled pure PCIe
transfer time. Hardware occupancy, memory-stall counters and exact imbalance
causes were not measured in this Systems-only capture.

### Lifetime and measurement limits

All four lifetime histories are complete: 27227–28130 events each, no unknown
free events, no history/marker truncation and no OOMs. Allocator lifetime peaks:

| Rank / call | Peak allocated | Phase | Resident weights | Activations/temporaries |
| --- | ---: | --- | ---: | ---: |
| 0 / first | 11.078 GiB | Block0 FC2 end | 3.164 GiB | 7.902 GiB |
| 0 / second | 10.855 GiB | Block21 RoPE begin | 4.393 GiB | 6.446 GiB |
| 1 / first | 11.033 GiB | Block24 RoPE begin | 4.572 GiB | 6.459 GiB |
| 1 / second | 11.034 GiB | Block24 RoPE begin | 4.572 GiB | 6.446 GiB |

The first rank0 peak includes the 1.579 GiB `[59134,14336]` FP16 FC1 output
still live at FC2 end. Warm peaks also involve a 1.184 GiB QKV allocation and
prefetched weight residency. Reducing FC1 alone will not eliminate the warm
QKV/residency peak. Reserved memory is larger than allocated memory and should
not be substituted for these lifetime totals.

Every call contains the same intrusive lifetime/shape/NVTX observers. Lifetime
export costs approximately 1.45–1.58 seconds per rank/call; that is only export
cost, not the total observer overhead. Large recorded host residual times
cannot be treated as uninstrumented operator cost without a separate observer-off
control. These timings are not production latency claims. Gauge fields such
as EMA and historical peaks must not be interpreted as additive counter deltas.

## Next optimization decisions supported by these results

1. VAE: retain the CUDA13 candidate for a future decoder/quality validation;
   prioritize the verified CUDA linear path over attention replacement.
2. DiT cold start: qualified ahead-of-time/warm compilation is the clearest
   measured target, while keeping cold and steady-state latency separate.
3. DiT memory: target both FC1/FC2 overlap and QKV/weight residency overlap;
   do not increase lookahead simply to address the now-small warm idle gaps.
4. Before acting on host residual totals, measure observer-off overhead and
   rank execution/clock differences. Do not infer a transport-only bottleneck
   from NCCL kernel duration alone.

No new GPU run, dependency change, precision promotion or full generation was
performed while collecting and analyzing these results.

Data: [sanitized summary](../data/h3_dit_pair_vae_cuda13_20261028/summary.json),
[corrected VAE trace attribution](../data/h3_dit_pair_vae_cuda13_20261028/vae_corrected_attribution.json).
Raw DiT evidence: `D:/attention_eval/h3_dit_pair_retry_20261028/gpu_outputs`.
Raw VAE evidence: `D:/attention_eval/h3_vae_cuda13_20261028/gpu_outputs`.
