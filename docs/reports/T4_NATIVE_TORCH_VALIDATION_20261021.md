# Native Kaggle Torch, preserved H3 core: T4 follow-up — 2026-10-21

**Current outcome, 2026-10-22:** three bounded physical dual-T4 batches supplied
current-stack operator/component evidence. Refiner, lifetimes, residency,
four-site transport, layer-bake identity, staged VAE lifecycle and Native Audio
passed their stated component scopes. The suite is **not an overall numerical
qualification**: the FP16 FC2 fallback has a demonstrated SwiGLU range failure,
and the 261-token Video query split has a repeatable byte-identity regression.

A controlled FP32 FC2-path ablation reduced the layer-49 stress error from
6.498 to 0.01512 relative to Native FP32; it remains a benchmark candidate.
Native Video fallback is byte-exact on the failing case. These are actionable
next integration/diagnosis targets, with no change to production defaults,
local conda, upstream source or core library versions. See the final sections
for controlled results and explicit remaining limits.

The user authorized Kaggle's native Torch and matching platform packages while
preserving core libraries and the legacy attention API. The local conda
environment, its lock, and the frozen ComfyUI archive were not changed. The
earlier strict Linux CUDA resolver conflict is preserved in
[its historical report](T4_FOLLOWUP_PREFLIGHT_20261021.md); its four proposed
constraint edits were not applied.

## Environment and execution contract

Kitchen remains **0.2.37**, TileLang **0.1.13**, and ComfyUI remains
`d49e888586dd8ae012c0667b33466b815fee07f7`. Exact core wheels were prepared and
hashed on CPU for Python 3.11, 3.12 and 3.13. The GPU batch selects the wheel set
matching its native interpreter and installs offline into an isolated overlay.
It records the actual Torch/Python/CUDA versions and verifies two physical T4
devices with capability 7.5 before running operators. Native Torch, Torchvision,
Torchaudio and NumPy remain with Kaggle's platform; UI-only Comfy packages are
excluded from this operator profile.

No full H3 generation, sampler, new sparse quality assessment, upstream clone,
dependency upgrade, GPU-time wheel resolution or compiler installation is part
of the batch. No Kaggle status/list/live-log polling is used. The batch has a
25-minute internal budget and bounded subprocesses; final evidence is collected
once by explicit version.

The production service's historical TileLang 0.1.15 / nvcc 12.8 gate is retained.
Isolated TileLang 0.1.13 operator qualification does not silently relax that gate.
The service lifecycle test uses a Native TE/row arithmetic policy to isolate
ownership from the separately tested operator ABI.

## CPU preparation evidence

[CPU preflight](https://www.kaggle.com/code/pamcks/h3-native-torch-cpu-preflight),
version 4, returned `cpu_ready`: Python **3.13.15**, native CPU Torch
**2.11.0+cpu**, Kitchen **0.2.37** and TileLang **0.1.13** imports succeeded. The
legacy Video attention signature remained
`forward(x, rotary_pos_emb, pre_norm, residual, residual_scale)`. Kitchen's
`int8_attention(q, k, v, *, scale=None, attn_mask=None)` signature was recorded.
These CPU checks do not qualify CUDA extensions or T4 arithmetic.

The real Linux safetensors reader gate passed six cases: native `os.pread`,
short reads, seek/readinto fallback, concurrent descriptors, fallback after a
partial pread and rejection of a truncated range. This is correctness evidence,
not an SSD performance comparison.

[CPU asset preparation](https://www.kaggle.com/code/pamcks/h3-trained-component-cpu-assets),
version 1, returned `ready`: seven selected component/input files totaling
**5,663,282,805 bytes**. Trained tensors were assembled from immutable publisher
revisions using CPU range reads, then checked against the locally prepared
component SHA256 values. Only a compressed retained input was uploaded from the
local machine; full TE/DiT checkpoints were not uploaded.

Sources:

- Comfy-Org/MiniMax-H3 revision `e5eb578a89295337b8ff433a035929ce0279e0b6`:
  trained TE block 0, DiT blocks 0/24/49, refiner attention, Video VAE and Audio VAE.
- ZhengmingYu/DMAD revision `439136a1dd4f435b8137a084f73d558d2cecda2a`:
  matching LoRA tensors for the selected DiT blocks.
- Retained first-block input: reconstructed from actual Native conditioning,
  reference and initial noise at **512×512, 90 frames, seed 20261013**. It is not
  the earlier requested 640×352 trajectory. Layers 24/49 reuse this input as a
  stress probe; it is not their own trajectory. TE block input and standalone
  attention Q/K/V are seeded synthetic shape probes. Missing historical QKV
  captures are not relabeled as real distributions.

## Physical batch and measurement boundaries

[T4 validation batch](https://www.kaggle.com/code/pamcks/h3-native-torch-t4-validation),
version 1, was submitted with `NvidiaTeslaT4`, no internet, and the reviewed CPU
core/assets outputs attached. Notebook completion alone is not a test pass:
`batch_results.json`, rank receipts and stage logs determine each outcome.

The Native FP32 activation flow is the numerical reference. Its current INT8
weights, GEMMs, attention operators and mixed precision islands remain active;
there is no forced all-FP32 implementation. SM75 never executes a BF16 baseline.
Kitchen may retain internal BF16 storage/conversion in that Native path; a
BF16 output type in a kernel name is not a BF16 residual-flow baseline or a
claim of BF16 Tensor Core execution on T4.
The default FP16 residual stream is folded by 64; the late-layer stress probe
also tests unscaled FP16 and a divisor of 16. Storage casts are plain dtype casts,
so observed overflow is not hidden by clamp/sanitization.

First-call compilation and model loading are reported separately. Complete API
measurements include preparation, communication and epilogues. Timings use two
warmups and repeated synchronized samples; TP2 row comparisons use the maximum
of corresponding rank samples. Profiling occurs after timed warm runs. Repeated
output, rank agreement and unchanged INT8 weight bytes are checked where
applicable. The TE lifetime ablation removes only explicit `del` statements in
a benchmark copy and checks output identity.

Operator dimensions are TP2-local: TE causal GQA Q `[1,32,353,128]`, KV
`[1,4,353,128]`; DiT Q/K/V `[1,28,6814,128]`. Row projection M is 353 for TE and
6814 for DiT, with each site's current sharded K/N. Four-site transport tests
compare Native INT32 SUM to power-of-two scaled FP16 SUM plus the fused
epilogue. Sparse drop/scalar pooling remain optional; their checks cover
execution and finite outputs, not semantic or generation-distribution fidelity.

Three-layer offline LoRA baking compares serial and two physical GPU workers,
including fresh worker startup and file I/O, and verifies output hashes against
the independently verified warm result. Its 2:1 layer partition is not a
50-layer throughput prediction. Adaptive residency uses physical copy events,
an actual FP16 residual workspace and controlled admission thresholds; it is not
a full TE throughput test. VAE tests use trained bounded inputs and real service
workers; they cover phase ownership, idle encoder/TE coexistence, Query Parallel
and Native fallback, pressure release and single-GPU Native Audio decoding.

Nsight Systems is used only if preinstalled. Hardware counter access is not
assumed; the historical Nsight Compute permission error is not bypassed.
PyTorch trace kernel-duration sums cannot establish memory-stall counters,
critical-path latency or GPU utilization.

## Evidence locations

Local root: `D:\attention_eval\t4_native_torch_20261021`.

- `cpu_output_v4/native_profile/preflight_receipt.json`: successful CPU profile,
  package/API inventory and core wheel archive identities.
- `cpu_output_v4/native_profile/linux_pread.json`: six Linux reader checks.
- `assets_cpu_output/component_assets/assets_receipt.json`: selected trained
  file identities and source revisions.
- `t4_batch_v1/source_binding.json`: submitted per-file source hashes and the
  exact reviewed CPU/assets receipt hashes.
- `t4_output_v1/download_manifest.json`: version-bound downloaded evidence
  filenames, byte counts and SHA256 values.

The physical results below come from downloaded artifacts only.
Full trajectory quality, frontend folding qualification, full pipeline
concurrency and sparse semantic acceptance remain outside this bounded batch.

## Physical results: batch version 1

The device gate passed on **two distinct Tesla T4 SM75 devices**, using native
**Torch 2.11.0+cu128**, CUDA **12.8**, Python **3.13.15**, preinstalled nvcc
**12.8.93**, Kitchen **0.2.37** and TileLang **0.1.13**. Kitchen's CUDA extension
and legacy attention API were available. The frozen source archive SHA256 is
`97cb93348993c7ff9a1390680ca4509d2880712707ae5bb249a1f4b2e8a92ebc`.
The submitted runtime/benchmark capsule SHA256 is
`a9bdc10cc7d3f930563a30942dfab82e80799d9f43c79369c39367ab43880c3c`.
The native image logged NCCL `2.31.2+cuda13.3`; it was not replaced. Actual
physical collectives passed. Those facts describe this mixed native image,
not a newly constructed exact replica of local Windows CUDA libraries.

The internal batch finished in **537.3 s**, including **47.5 s** of offline
overlay installation. Attention, row transport, adaptive residency, TE/DiT flow
execution and the serial/dual-device bake completed. Refiner and Video component
identity checks failed, so overall status is **partial_or_failed**. A successful
notebook export is not reported as overall qualification.

### FP16 range, accuracy and actual dispatch

Relative L2 below is a dense component diagnostic against the existing Native
FP32 activation flow. It is not sparse quality or generation acceptance.
The retained full DiT component has **7,895 tokens**; standalone operator probes
use the separate TP2 deployment target of 6,814 tokens.

| Trained component/input | Divisor-64 relative L2 | Nonfinite output elements | Interpretation |
|---|---:|---:|---|
| TE block 0, synthetic S353 | 0.001368 | 0 | Bounded component difference; no real-TE trajectory claim |
| DiT block 0, retained S7895 | 0.002768 | 0 | Actual first-block input only |
| DiT block 24, reused S513 stress | 0.03243 | 0 | Larger late-layer sensitivity; not its native input distribution |
| DiT block 49 + LoRA, reused S513 stress | **6.498** | 0 | **Finiteness passes, accuracy does not qualify** |

For the same layer-49 LoRA stress input, unscaled FP16 produced **1,581,592**
nonfinite output elements and divisor 16 produced **3,885**. Divisor 64 prevented
those output nonfinites but did not restore fidelity. Its physical output max
was 3,352,576 versus Native FP32's 977,014.875. The observed casts expose overflow
directly; no test-only clamp was introduced. This rejects any blanket claim that
the current default is numerically qualified merely because its outputs are
finite. No production default was changed on the basis of this stress test.

Both observed TE runs called the masked TileLang Group32 attention kernel with
FP16 and FP32 activation flows. FP32 TE/DiT called TileLang row preparation and
SwiGLU padding; FP32 DiT also called `_quantize_owned`. The FP16 cases instead
called `optimized_tp.global_row_quantize` four times across the two observed
forwards, and never called the FP32-only SwiGLU padding/owned quantizer kernels.
Consequently the old FP32 fusion speedups do not transfer to the default simply
by selecting the same policy string.

Rank-0 steady peak allocation for the full DiT block fell from **2.894 GiB** to
**1.372 GiB**. Its measured medians were 222.35 ms and 215.89 ms, respectively;
the small timing difference is not a robust pipeline speedup. Several initial
steady samples still drifted after the fixed two warmups; GPU clocks/thermal
state were not locked or sampled. Seven samples and raw traces are preserved;
these numbers support a compatibility/bottleneck screen, not a fine-grained
optimization ranking.

The TE explicit-deletion ablation was byte-exact. In rank 0, keeping the old
temporaries increased peak allocation from 400,222,208 to 407,660,544 bytes:
**7.09 MiB** more for this single S353 block. No full-model saving is extrapolated.

### Standalone attention and four-site transport

Complete API medians below use the maximum of corresponding device/rank samples.
Attention Q/K/V are synthetic; sparse routing behavior is not a real-distribution
speedup result.

| Standalone attention | Shape/dtype | Median ms |
|---|---|---:|
| TE TileLang Group32 causal GQA | S353, FP32 | 1.032 |
| TE TileLang Group32 causal GQA | S353, FP16 | 1.083 |
| DiT Kitchen | S6814, FP16 | 22.712 |
| DiT TileLang dense | S6814, FP16 | 60.437 |
| DiT TileLang drop | S6814, FP16 | 38.456 |
| DiT TileLang scalar | S6814, FP16 | 46.783 |

TE Group32 matched its independent prepared-Q/K reference within the declared
operator tolerance. DiT checks verify execution, shape and finiteness only;
none are promoted to a sparse semantic pass. Kitchen remains the DiT default.

All four FP32-eligible row sites matched the independent Native INT32 reference
byte-for-byte. Scaled FP16 transport + fused epilogue matched its separately
computed codec reference byte-for-byte; it is still lossy relative to INT32 SUM.
The full-row control uses Native rotation preparation to isolate transport.

| Site, local M×K→N | Packed/INT32 ms | Scaled FP16 ms | Complete API ratio |
|---|---:|---:|---:|
| DiT O, 6814×3584→5376 | 75.399 | 58.071 | 1.298× |
| DiT FC2, 6814×7168→5376 | 56.973 | 42.927 | 1.327× |
| TE O, 353×4096→5120 | 3.285 | 2.389 | 1.375× |
| TE down, 353×12800→5120 | 6.160 | 5.131 | 1.201× |

Transport relative L2 was approximately **0.000306** on these seeded cases.
This is not an adversarial cancellation sweep or generation-quality acceptance.
It remains optional; the FP32 eligibility guard prevents it from accelerating
the default FP16 row flow as currently implemented.

### Trace evidence and bottleneck implications

The post-warm PyTorch trace contains one full DiT block per flow and rank.
Rank-0 kernel-duration sums were:

| Kernel category | Native FP32 ms | FP16-fix ms |
|---|---:|---:|
| Two large INT32 SUM collectives | 81.589 | 80.628 |
| All recorded NCCL collectives, including waits | 83.945 | 85.092 |
| CUTLASS GEMMs | 47.847 | 44.853 |
| Fused Kitchen attention | 25.620 | 26.214 |
| Other GEMMs, including rotation | 17.897 | 3.852 |
| Norm/pointwise/copy category | 27.348 | 46.296 |
| All kernel-duration sum | 220.340 | 215.452 |
| Recorded kernel launches | 91 | 146 |

Grouping is based on kernel names. Generic `main_kernel` launches remain
unattributed in the machine record; they are not silently counted as QK or PV.
Collective kernel durations include waiting and cannot be read as pure PCIe
transfer time. Categories can overlap on streams; sums are not critical-path
decomposition or hardware utilization counters.

The concrete limitation is visible: halving residual storage does **not** halve
the large INT32 SUM payloads. At S7895 each row accumulator is
169,774,080 bytes, and both attention-output and FC2 projections reduce one.
FP16 rotation GEMMs become cheaper, but casts and FP32 norm/residual islands add
launches and work. That explains the weak complete-block timing change more
directly than attention-only latency. Future work should measure FP16-compatible
transport/epilogue integration and fuse the demonstrated cast/norm boundaries
after the accuracy failure is understood. No speculative memory-stall or
occupancy claim is made.

Nsight Systems was actually available and captured TE successfully; the report
is `te_nsys.nsys-rep` in the version-1 provider output. The narrow follow-up
exports statistics from that existing capture. Its whole-process scope includes
loading, warmups and observations, so it cannot be treated as one steady TE
forward. Nsight Compute counters remain uncollected: historical probes were
denied permission, and current permissions were not retested. No counter-based
stall/utilization conclusions are supported.

### Residency, LoRA baking and failed component checks

Both physical residency windows passed depth **2→1→2** under controlled headroom
thresholds while a real `[6814,5376]` FP16 workspace coexisted. Copies, events and
eviction were exercised. This proves admission behavior for bounded fixtures,
not target TE throughput under production pressure.

The trained three-layer bake used both T4 UUIDs and produced byte-identical
artifacts across serial/parallel and verified-warm/measured runs. The base file
was unchanged. Median production `convert` time, including I/O and fresh worker
startup, was **12.265 s serial** versus **12.848 s parallel**: **0.955×**, slightly
slower on this small 2:1 partition. Peak VRAM/RSS were not sampled; that part of
T06's requested characterization remains open. There is no full-model bake
throughput claim.

The refiner passed its no-RoPE expression comparison, then failed the RoPE case:
the harness-created attention parameters still had `requires_grad=True`, which
Kitchen's inference-only in-place RoPE correctly rejected. The production
frontend already calls `eval().requires_grad_(False)`. The harness was corrected
to use that construction; no library/API change was made.

The VAE service demonstrated encoder + one TE-layer coexistence on each worker,
byte-exact staged encoding and byte-exact short-query decoding. The 261-token
split-query output then failed identity and stopped the original stage. Both
worker TE prefixes had 243,978,752 live bytes; parent preload reported no errors;
the final host-budget reservation was zero. Later temporal/fallback/Audio tests
were not executed in this original failed stage, and are not marked passed.

A separate eight-minute-capped
[component follow-up](https://www.kaggle.com/code/pamcks/h3-t4-component-followup),
version 1, repeats the failed components, preserves construction outside
inference mode, records Video mismatch diagnostics instead of stopping before
the remaining cases, and probes FP32 FC2/row preparation islands for the
layer-49 LoRA stress failure. It reuses the reviewed successful operator gates
and existing Systems capture rather than repeating the broad screen or bake.

## Component follow-up results and current qualification

The component follow-up used another two distinct T4 devices with the same
native Torch 2.11.0+cu128, preserved Kitchen/TileLang, compiler and unchanged
production source hashes. It finished in **220.9 s**, including **44.9 s** of
offline overlay installation. Its source/result binding names the exact
original batch receipt. Overall status remains **partial_or_failed**.

- **Refiner:** both trained S353 comparisons, with and without RoPE, are now
  byte-exact against the independent Native SDPA expression. The fix was to the
  harness's parameter inference flags; the production wrapper/API was unchanged.
- **Lifecycle:** parent Video encoder + TE preload, staged Video/Audio encoding,
  cancellation/pressure release and final host-budget cleanup passed on the two
  physical devices. This closes the bounded T07 lifecycle scope, not the larger
  model-switch/interruption matrix.
- **Audio:** both `auto` and `native` returned byte-exact Native single-GPU FP32
  stereo output `[1,12800,2]`, with no Audio worker decode commands.
- **Video:** short and temporal queries, and controlled admission-decline Native
  fallback, were byte-exact. The temporal result was `[1,9,256,256,3]`. The
  **261-token split still failed byte identity**: 33,668 of 196,608 output
  elements differed, relative L2 **0.0001382**, maximum absolute difference
  **0.0004473** in normalized pixels. Both Native and parallel runs were
  individually repeatable. Staged Native was byte-exact to ordinary Native.
  Legacy signatures, quantized QKV, unit QK scales and Kitchen availability all
  passed the compatibility contract. Thus this is a shape-dependent numerical
  regression of the split path, not the earlier attention-signature mismatch.
  The first differing internal primitive is not yet localized; no visual-quality
  or byte-exact qualification is granted to this split. Native decode remains
  the verified fallback on this stack.

The new FP16 stage diagnostic failed before producing its first comparison:
its captured CPU tensors were inference tensors, and the harness tried to
rescale them in place outside inference mode. This is a benchmark error, not
evidence about the proposed FP32 preparation island. The helper now rescales
out of place; a CPU regression test reproduces the inference-tensor condition
and verifies that capture storage is not mutated. On **2026-10-22**, a separate
four-minute-capped
[stage diagnostic](https://www.kaggle.com/code/pamcks/h3-t4-fp16-stage-diagnosis)
was submitted with only that corrected probe. Components and baking were not
repeated. Its collected results appear in the final section below.

The existing Systems capture was successfully analyzed, and its raw 4,014,364-byte
report plus three CSV summaries were collected. Over the **whole instrumented
TE process**, `cudaMemcpyAsync` accounted for 1.053 s of CUDA API time across
702 calls; `cudaLaunchKernel` accounted for 1.041 s across 5,434 calls. Recorded
device-to-host copies totaled 0.784 s, host-to-device 0.212 s and device-to-device
0.012 s. This process deliberately loads/copies weights and observes/hashes
outputs. Those copy/API totals include diagnostic overhead and **must not be
attributed to a steady production TE request**. The capture confirms the
presence of those operations and collective kernels; it is not a replacement
for a range-marked production critical-path profile.

Raw evidence is in
`D:\attention_eval\t4_native_torch_20261021\component_followup_output_v1\t4_validation`.
The reproducible merged record is
[t4_native_torch_validation_20261021.json](../data/t4_native_torch_validation_20261021.json).
`summarize_t4_native_validation.py --followup ...` verifies both download
manifests, the original-receipt binding, physical device identities, stack
versions and unchanged runtime patch hashes before merging successful scopes.

| Dimension | Current result |
|---|---|
| T01 FP16-fix/folding | Physical probes executed; FC2/SwiGLU stress failure, with and without LoRA; targeted mitigation tested; full trajectory/frontend scope open |
| T02 FP16 dispatch | Actual TileLang attention and guarded row/SwiGLU fallbacks verified |
| T03 scaled-FP16 row backend | Four-site physical codec/integration pass; optional FP32-eligible path only |
| T04 refiner/TE lifetimes | Bounded trained component pass, including byte-exact deletion ablation |
| T05 adaptive lookahead | Physical bounded admission/event pass; no production throughput claim |
| T06 layer-parallel bake | Byte-exact physical pass; small-case throughput slightly worse; peak VRAM/RSS unmeasured |
| T07 staged VAE + idle preload | Bounded physical lifecycle pass |
| T08 Video/Audio regression | Audio and Native fallback pass; 261-token Video split identity fails |
| T09 positional reader | Six real Linux CPU/descriptor cases pass |

Full-pipeline generation, sparse semantics/distribution, approximate transport
generation acceptance, concurrency and hardware-counter profiling remain open
under the existing scope. The local conda lock, frozen source, operator defaults
and historical production dependency gate were not changed to manufacture a pass.

## Corrected stage diagnosis — 2026-10-22

The corrected diagnostic completed on another two actual T4s, using the same
native Torch/core profile and unchanged production patch hashes, in **124.3 s**
including offline setup. Its batch status is `passed`; its numerical status is
deliberately `completed_diagnostic_not_quality_acceptance`. Six controlled
cases ran per rank on trained DiT layer 49 with the same reused S513 input.
The previous failed diagnostic and its logs remain preserved.

| Layer-49 stress case | Relative L2 to its matching Native FP32 flow |
|---|---:|
| FP16-fix /64, no LoRA | **6.42419** |
| FP16-fix /64, with LoRA | **6.49837** |
| FP16-fix /64 + FP32 FC2 path, with LoRA | **0.0151191** |
| FP16-fix /64 + FP32 attention-output and FC2 rows, with LoRA | **0.0151725** |

This is **not a LoRA-specific failure**. For the ordinary FP16-fix + LoRA case,
rank 0's stage differences were 0 at AdaLN, 0.000292 at QKV input, 0.003638 at
QKV output, 0.04202 at attention output, 0.02741 at FC1 output/FC2 input, then
**6.67489 at FC2 output**. The catastrophic increase occurs inside FC2 rather
than being present in its packed gate/up input.

The directly measured SwiGLU products explain the range hazard. With LoRA,
rank 0's FP32 product maximum was **375,732**, with **483** values exceeding
65,504; rank 1's maximum was **238,007.25**, with **485** values exceeding the
limit. Converting those products to FP16 produced exactly those nonfinite
counts. Similar counts occur without LoRA. The SM75 FP16 fallback in
`optimized_tp.global_row_quantize` eagerly applies Kitchen's
`silu(gate).mul_(up)` in its input dtype before rotation/quantization. A residual
divisor does not bound this internal product: RMS normalization restores the
branch's scale, and FC1 is not scaled down by the residual gauge.

The tested mitigation passes packed FC2 inputs to the existing complete row
operator in FP32, including its preparation, reduction, epilogue and LoRA
addition, then stores the result back into the FP16 residual flow. It retains
the original INT8 weights and Native INT32 SUM. This reduces output relative L2
by approximately **430×** on this stress probe, to about **1.51%**, while widening
the attention-output row too provides no further benefit. This localizes the
highest-value precision island to FC2.

The ablation widens the **whole FC2 operator**, not only the pointwise product.
It therefore establishes a range failure and an effective operator-level
mitigation, but does not independently apportion remaining error among FP16
SwiGLU storage, rotation/quantization, epilogue rounding and base/LoRA addition.
The remaining attention-stage deviation also survives the mitigation. There
is no claim of full trajectory fidelity, recovered generation quality, or a
measured speedup: this diagnostic records stage values, not performance timing.

The candidate is implemented **in the benchmark only**; it has not silently
changed the production FP16 default. The immediate integration candidate is a
targeted FP32 FC2 preparation/operator island followed by the FP16 storage cast,
with separate memory/steady-time validation. Increasing the residual divisor
alone is not a demonstrated solution. Keep the 261-token Video query split
unqualified and use its verified Native fallback when byte identity is required
on this native-Torch stack.

Diagnostic artifacts:
`D:\attention_eval\t4_native_torch_20261021\stage_diagnosis_output_20261022_v1`.
The merged JSON verifies all three version-bound downloads and their source,
device and prior-result bindings. Reproduce the offline aggregation with:

```powershell
& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/summarize_t4_native_validation.py `
  --data D:\attention_eval\t4_native_torch_20261021\t4_output_v1 `
  --cpu D:\attention_eval\t4_native_torch_20261021\cpu_output_v4 `
  --followup D:\attention_eval\t4_native_torch_20261021\component_followup_output_v1 `
  --diagnosis D:\attention_eval\t4_native_torch_20261021\stage_diagnosis_output_20261022_v1 `
  --output docs/data/t4_native_torch_validation_20261021.json
```

Local regression verification: **32 focused tests passed**, covering frozen
setup, the authorized native profile, Linux constraint boundaries, tensor-flow
contracts, Native Audio policy, evidence acceptance and the inference-capture
repair. No CLI/API/browser status polling, full H3 run or local SM120 substitute
benchmark was used for these physical conclusions.
