# SwiGLU + TP2 padding fusion

2026-10-17. Implemented in `patches/tilelang_swiglu_pad.py`, with local operator
validation and offline SM75 compilation. No H3 generation, production hook
installation, package upgrade or new Kaggle GPU submission. The queued
attention experiment and its source-hashed test items remain unchanged.

## Actual TP2 boundary

The fused operation computes FP32 `silu(gate) * up` and writes it directly into
the owning rank's columns of a full-width FP32 tensor, filling all other
columns with positive zero. That tensor feeds the existing native ConvRot
matrix multiplication with its original global-width geometry.

| Component | Rows | Local activation width | Gate/up storage | Padded width |
|---|---:|---:|---|---:|
| TE down projection | 353 | 12800 | Separate gate and up tensors | 25600 |
| DiT FC2 | 6814 | 7168 | Strided halves of packed `[gate | up]`, row stride14336 | 14336 |

Rank0 writes the first half; rank1 writes the second half. The TileLang kernel
reads both storage layouts directly. It does not concatenate TE inputs or
materialize either DiT half. One launch replaces eager SwiGLU plus ownership
padding: four eager kernels, or three with the earlier packed-padding candidate.
The fused output remains full-width, preserving downstream rotation rounding.

The numerical expression uses `expf`, explicit rounded FP32 addition/division
and multiplication. It retains the FP32 SiLU result before multiplying up.
No approximate sigmoid, exp2 rewrite, reduced-precision activation or changed
INT8 quantizer is introduced. Unowned columns are explicitly written as zero;
they are not obtained by multiplying possibly nonfinite values by a mask.

## Implementation and opt-in entry points

- `swiglu_pad(gate, up, rank=..., world_size=2)` handles separate tensors or
  views with contiguous final channels.
- `swiglu_pad_packed(x, rank=..., world_size=2)` accepts packed DiT input.
- `row_linear(..., input_act='swiglu', variant='packed')` is the DiT FC2
  integration candidate.
- `row_linear_parts(gate, up, ...)` is the TE down-projection candidate.

Both integration paths retain full-width FP32 ConvRot, global FP32 row MAX,
the existing external-max row quantizer, native INT8 GEMM, INT32 SUM before
dequantization, bias and residual behavior. The wrappers select fusion only
for the FP32 fallback-rotation path with multiple ranks and packed variant;
other routes fall back. The standalone fused kernel supports target SM75 and
explicit local SM120 previews. No existing forward method or global hook is
automatically replaced: TE needs the parts entry point before materializing
its activation, and DiT needs the opt-in row-linear function at its FC2 call.

## Numerical validation

All41 edge tests pass bitwise against eager PyTorch, covering packed/separate
storage, both TP2 ranks, odd widths, leading dimensions/world-size3, signed
zero, preserved subnormal outputs, large values, NaN and both infinities.
Inputs remain unchanged. The strengthened subnormal suite is retained under
`edge_confirmation` separately from full-size measurements.

All four full-size padding cases match byte for byte. Full TE and DiT row
arithmetic also passes on both logical ranks: global-MAX inputs, quantized Q,
scales, local INT32 GEMM partials and final outputs with bias/residual are
identical. No L1/L2 threshold is used to accept a different activation.

Those chain tests run sequentially on one local GPU. Collective results are
precomputed from both ranks, and the mock transport checks each actual local
MAX/INT32 SUM contribution before supplying the global result. This validates
the operator arithmetic and collective order; it does not measure NCCL or
two-GPU latency. No communication simulation timing is reported.

Local Kitchen0.2.33 lacks two private helpers/signatures used by the pinned
0.2.37 code, so the benchmark process supplies the matching activation
signature and16-byte vector-argument normalization. The frozen row-quantizer
dependency is copied and retargeted to SM120 only for this process. On SM120,
Kitchen also chooses a different DiT rotation dispatch because of its shared
memory capacity. The DiT arithmetic gate explicitly selects the source-audited
SM75 full-width rotation fixture; that override is outside all timed padding
functions, does not fake device capability, and is restored afterward. TE
does not need the dispatch override. These adapters are recorded in JSON;
the shared environment and original sources are unchanged.

## Local performance preview

Hardware: RTX5060Ti SM120; Torch2.12.1+cu132, TileLang0.1.13 and Kitchen0.2.33.
Synthetic inputs use the actual TP2 shapes. Each function warms up five calls,
then collects15 CUDA-event and synchronized wall samples. A reverse-order
second round runs without compilation. JIT is outside timing. Values below
are the slower logical partition's median, not actual TP2 request latency.

| Component / round | Eager stage ms | Previous packed-padding stage ms | Fused stage ms | Reduction vs packed |
|---|---:|---:|---:|---:|
| TE first | 0.3323 | 0.2791 | 0.2314 | 17.1% |
| TE reverse repeat | 0.3505 | 0.3019 | 0.2572 | 14.8% |
| DiT first | 4.7491 | 4.2962 | 2.2900 | 46.7% |
| DiT reverse repeat | 5.0080 | 4.5889 | 2.5329 | 44.8% |

These are **SwiGLU plus padding stage** measurements. They exclude rotation,
quantization, GEMM and communication, and are not FC2/down-projection or
pipeline speedups. The reverse repeat is in the same process; this is not an
independent T4 allocation. Local drift is visible in TE and in the second
DiT partition; no stable single-number T4 forecast is justified.

The activation intermediate eliminated by the fused path is logically
17.24MiB for TE and186.32MiB for DiT per rank. This is the tensor's size,
not a measured pipeline-VRAM reduction. PyTorch peak-allocation counters are
retained in raw JSON but do not support claiming that same reduction versus
every baseline/cache state. The full padded rotation input remains necessary.

## SM75 status and reproduction

All four target/rank sources lower and compile offline to SM75 cubins:
17/19 registers per thread for DiT and20 for TE, with no reported spills,
stack or barriers. This is Windows nvcc13.3 compilation, not execution with
the pinned Linux nvcc12.8/PyTorch2.11 T4 stack. Actual T4 activation equality
and latency remain to be measured before enabling production hooks.

Commands:

```powershell
python benchmarks/lower_swiglu_padding.py --output D:/attention_eval/operator_optimization_tp2_20261017/swiglu_padding/offline
python benchmarks/compile_grouped_tp_offline.py --input D:/attention_eval/operator_optimization_tp2_20261017/swiglu_padding/offline
python benchmarks/swiglu_padding.py --output D:/attention_eval/operator_optimization_tp2_20261017/swiglu_padding/local_confirmed
```

Evidence root:
`D:/attention_eval/operator_optimization_tp2_20261017/swiglu_padding`.
Qualified results are in `local_confirmed`, `edge_confirmation`, `offline`
and `summary.json`. Earlier diagnostic attempts/logs are retained separately
and excluded from the reported timing table. Source hashes and archived runner
copies are recorded alongside the evidence. All current TP2 manifest hashes,
first-pass attention source hashes and queued second-pass attention source
hashes still match.

## Physical T4 validation submitted 2026-10-18

Kaggle: [h3-tp2-swiglu-padding, version 1](https://www.kaggle.com/code/chinokiki9/h3-tp2-swiglu-padding).
Provider runtime cap: 600 seconds; internal deadline: 560 seconds. Initial
provider status was RUNNING. GPU allocation and results require artifact checks.
Harness: `benchmarks/tp2_swiglu_padding.py`; builder:
`build_tp2_swiglu_padding_job.py`. Evidence and immutable submission snapshots:
`D:/attention_eval/operator_optimization_tp2_20261018/swiglu_padding`.
No datasets, model weights or full H3 pipeline are used.

The new source check detected changes in `tp_dit.py`, `optimized_tp.py` and
`comfy_tp_runtime.py` since the local preview. A refreshed CPU/meta schema audit
confirmed all attention and GEMM shapes are unchanged. Today's default remains
`scalemax`, `STREAM_ROWS=0`, with native INT32 reduction. Optional streamed
rotation is outside this fusion experiment. Prior manifests are preserved.

Both physical GPUs must identify as T4/SM75. The harness has no architecture
retargeting, simulated transport, private-helper adapters or forced rotation
dispatch. It tests separate TE gate/up and strided halves of packed DiT FC1 at
the audited TP2 sizes. Gates compare bytes for padding, INT8 payload, scales,
local/global MAX and INT32-SUM contributions, and final bias/residual outputs.
Every recorded reduction executes real NCCL. Each GPU also runs 41 edge cases.

Three variants are screened: current eager padding, previous TileLang packed
padding after eager SwiGLU, and fused SwiGLU/padding. Measurements cover padding
alone, quantization including MAX, and the complete row-linear API including
INT32 SUM. Each phase uses 10 warm-up calls and 15 measured calls per rank;
compilation and gates precede timing. A fresh process repeats packed versus
fused in reverse order. Summary values use the slower rank's median; CUDA-event
and wall samples are retained separately. PyTorch allocation counters are not
full-pipeline VRAM measurements.

Nsight Systems captures three full-API calls for each component/variant, with
NVTX labels for activation, padding, rotation, maxima, quantization, GEMM,
collectives and dequantization. A separately labelled completion barrier keeps
both ranks in capture until their calls finish. GPU telemetry is sampled every
500 ms. Restricted Nsight Compute counters are not retried in this submission.
Production hooks remain disabled pending actual T4 results.

## Completed physical T4 results

Version 1 completed on **two Tesla T4 SM75 GPUs**. All stages succeeded:
dependency setup 13.09 s, pinned PyTorch setup 119.83 s, initial screen 98.44 s,
fresh-process repeat 27.34 s, and Nsight Systems 45.08 s. The internal job
finished at approximately 319 s, within the 600 s provider cap. Runtime:
PyTorch 2.11.0+cu130, Kitchen 0.2.37, TileLang 0.1.15, nvcc 12.8.93,
NCCL 2.28.9; P2P disabled and four NCCL channels, matching the previous TP2
operator tests. No full H3 generation was run.

All **82 edge gates** and **28 full-size component/rank/variant records** passed
bitwise checks across screen, repeat and profiling processes. These include
padding, quantized activations, scales, pre/post real MAX and INT32 SUM tensors,
and final outputs with bias/residual. Replicated output hashes agree across
the two ranks. Submitted source hashes and the refreshed manifest match the
downloaded artifacts. These are synthetic-input arithmetic/conformance results;
they are not real-activation distribution or generation-quality measurements.

Values below are the slower rank's CUDA-event median in milliseconds. The
baseline is **previous TileLang packed padding after eager SwiGLU**. Both rounds
use the same inputs and weights; the repeat reverses candidate order in a fresh
process on the same GPU allocation, rather than an independent host allocation.

| Component / phase | Screen packed | Screen fused | Reduction | Repeat packed | Repeat fused | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| TE SwiGLU + padding | 0.6533 | 0.3781 | 42.13% | 0.6529 | 0.3705 | 43.26% |
| TE quantization incl. MAX | 1.7932 | 1.5768 | 12.07% | 1.8431 | 1.5974 | 13.33% |
| TE full row API | 4.5056 | 4.2553 | 5.56% | 4.5773 | 4.3308 | 5.39% |
| DiT SwiGLU + padding | 6.6519 | 3.4189 | 48.60% | 6.6298 | 3.4089 | 48.58% |
| DiT quantization incl. MAX | 19.1820 | 16.3902 | 14.55% | 20.2510 | 17.1295 | 15.41% |
| DiT full row API | 61.9068 | 59.3110 | 4.19% | 62.4556 | 59.3248 | 5.01% |

Synchronized wall medians corroborate the direction. Repeat full-row wall
latencies are TE 4.6180 → 4.3714 ms (5.34%) and DiT 62.5537 → 59.4114 ms
(5.02%). The full-row harness includes replicated bias and a residual addcmul
to exercise those epilogues; current TE/DiT call sites do not pass those
optional arguments into row_linear. Therefore these full-API timings are for
the documented operator harness, not exact in-pipeline projection latency.

The current eager production preparation route is also included in screen
JSON as contextual evidence. Its difference from fused includes the earlier
external-max/packing changes, so the entire gain is not attributed to this
SwiGLU fusion. The table isolates fusion relative to the previous packed path.

## Profiler evidence and remaining bottleneck

Four Nsight Systems reports were downloaded and exported locally to SQLite.
All **256 kernel records** have process-qualified NVTX attribution. Each report
contains three full-API calls on each GPU. SwiGLU plus packing changes from
**three launches to one**, reducing the complete tested API from **11 to 9
kernel launches per rank per call**. Completion barriers are outside API ranges.
Actual T4 launch records show 20 registers/thread for TE and 17/19 for DiT
rank 0/1, with 128 threads and no shared-memory or local-memory allocation
reported. These launch records do not measure achieved occupancy or stalls.

DiT rank 0 mean kernel durations per profiled call illustrate the limit:

| Phase | Packed ms | Fused ms |
|---|---:|---:|
| SwiGLU + padding | 4.0150 + 2.6310 = 6.6460 | 3.3740 |
| Full-width FP32 rotation | 7.2935 | 7.2491 |
| Local maximum | 0.7297 | 0.7288 |
| External-max INT8 quantization | 1.0199 | 1.0195 |
| FP32 MAX collective | 0.1266 | 0.5746 |
| INT8 GEMM → INT32 | 7.8045 | 7.6108 |
| INT32 SUM collective | 35.3857 | 35.5508 |
| Native dequantization | 1.1868 | 1.1878 |
| Residual addcmul | 1.7762 | 1.7809 |

Rank 1 confirms the same broad DiT result: packing 2.6305 ms plus SwiGLU
4.0061 ms becomes a 3.3448 ms fused kernel; INT32 SUM remains about 35.4 ms.
The SUM handles a 6814×5376 INT32 tensor (139.74 MiB per rank). Its kernel span
is approximately 60% of the fused API's measured latency, explaining why a
roughly 49% padding-stage reduction yields only 4–5% for the whole row operator.
NCCL kernel durations include peer waiting and protocol execution; they are
not a measurement of pure PCIe transfer bandwidth. Rotation and GEMM also
remain material costs. No rotation geometry or reduction datatype was changed.

TE's short trace shows substantial rank imbalance: in the fused report,
rank 1's MAX span is 2.7872 ms while rank 0's is 0.0283 ms; their rotation and
GEMM durations also differ markedly. This trace supports the launch reduction
but is not used to estimate stable TE phase latency. The warmed-up screen
and fresh-process CUDA/wall samples supply the performance conclusion.

Nsight Compute hardware counters were not retried because the preceding Kaggle
probe reported restricted counter access. No measured memory-stall or achieved
SM-occupancy claim is made. GPU telemetry and all profiler evidence are retained.

Both components qualify as useful optional fusion candidates: consistent
full-operator improvements and exact tested arithmetic. The operator already
provides packed DiT and separate TE entry points. Production call-site
integration remains a separate change. Further optimization should target
the unchanged INT32 collective and full-width rotation/GEMM costs; the evidence
does not justify expecting another large whole-operator gain from padding alone.

Evidence: `D:/attention_eval/operator_optimization_tp2_20261018/swiglu_padding/v1/swiglu_padding`.
Key files: `validated_summary.json`, `wall_summary.json`, `profile_summary.json`,
`trace_attribution.json`, four `.nsys-rep`/`.sqlite` pairs, per-rank raw JSON,
`telemetry.csv`, and `status.json`. The parent `v1/download_manifest.json`
records all downloaded artifact hashes. Reproduce CPU checks with:

```powershell
python benchmarks/summarize_tp2_swiglu.py D:/attention_eval/operator_optimization_tp2_20261018/swiglu_padding/v1/swiglu_padding
```
