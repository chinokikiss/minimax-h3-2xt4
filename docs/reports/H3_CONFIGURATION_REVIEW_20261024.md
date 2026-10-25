# H3 TP2 configuration review — 2026-10-24

The best supported balance is **FP16-fix /64, dense Kitchen DiT attention,
Group32 TE attention, FP32 stored SwiGLU/quantization, scaled-FP16 base SUM,
direct INT8 + TileLang DiT columns, and bounded weight streaming**. Enable
lifecycle-aware prefetch and host-pressure admission. For supported short/medium
online LoRAs, enable selected FC1 and QKV B. For long sequences, use TP2
bundle-offline LoRA; long-sequence online-LoRA tests and retries are cancelled.

The supported choices below are now loader/runtime defaults where applicable;
this is not a measured global optimum. This revision incorporates the collected actual-T4
results and cancellation in the
[T4 recovery report](H3_REVIEW_T4_RECOVERY_20261024.md), superseding this
review's earlier “T4 pending” recommendations. Version1 completed ten operator
cases per owner before its final LoRA repeat OOM; version2 completed the isolated
remaining timing case but its combined profiling replay failed. All48 recovery
artifacts were collected. No new Kaggle request or hardware run was made for
this document update, and no complete profiler attribution is claimed.
The [machine-readable recommendation](../data/h3_configuration_review_20261024.json)
contains the loader settings, workload-specific overrides, all55 table entries,
source hashes and validation scope.

## Default promotion and control cleanup

The loader now defaults to offline LoRA, direct INT8 + TileLang columns,
lifecycle prefetch, host-pressure admission and adaptive lookahead. The sampling
reserve defaults to the retained long-shape9GiB and the command deadline to
1500s. Online remains explicit for fitting short/medium requests, with owned
QKV B on by default inside its geometry guard. Runtime-only bare configurations
do not infer offline baking: the loader derives `bundle_offline_lora` from
offline mode plus a TP2 JSON descriptor. Ordinary safetensors offline input
still invokes the existing whole-checkpoint converter; prefer bundles for the
bounded long-sequence path.

Removed from production configuration and dispatch:

- The one-value `tp_quantization` and base `swiglu_reduction` selectors: global
  row MAX and fallback Native INT32 SUM are fixed contracts.
- Deprecated `lora_concat_a` plumbing. Old false-valued Python loader arguments
  are accepted at the compatibility boundary; true is rejected.
- `swiglu_padding=False` and its unfused policy branch. Padding/FP32 fusion
  staging is fixed on; the Native reference backend remains available.
- The `selective_b` master switch. Supported online FC1 owned B is fixed on;
  unsupported geometry still uses full B. Offline has no online B projection.
- `checkpoint_read_chunk_mib`/`column_read_chunk_bytes` and the full-row scratch
  branch. Column slices read only owned source ranges with pread/seek fallback.
- The production `tilelang_exact` selector. The direct path won the tested
  shapes; preallocated remains the fallback. The epilogue-only comparator is
  retained solely in the private historical benchmark path.
- Rejected coalesced/extra-worker/priority TE loading code in `patches`: moved
  to `benchmarks/historical_te_loading.py`, outside production imports. Earlier
  submitted archives and experiment evidence remain unchanged.

All attention operators and selectors remain available, including Scalar,
dropping, TileLang dense and Native. Recompute, INT32 transport, Native FP32
flow, full-input LoRA gathering and memory/scheduling bounds remain useful
controls; a slower control is not deleted when it serves accuracy, capacity
or lifecycle needs. Idle preload remains optional because interactive gaps
can benefit from it.

Both maintained workflows and the existing local ComfyUI TP2 workflow now use
configuration version2. A browser extension remaps older positional widgets by
name so removed controls do not shift attention, dtype or LoRA values. Old
explicit selections are preserved except retired controls; old epilogue-only
selection maps to the direct path. Restart ComfyUI and reload the page.

## Recommended operating profile

Use the existing verified TP2 base descriptors/shards. The following profile
is for long-sequence deployment with adapters, including the retained
51,220-token workload. Offline mode here means bounded bundle block merging,
not caching a complete baked checkpoint:

```python
loader_options = dict(
    lora_mode="offline",                # TP2 JSON input; long-sequence policy
    lora_row_partial_reduce=True,        # inactive offline
    lora_selective_qkv_b=True,            # inactive offline; enabled if online
    te_attention="tilelang_group32",
    dit_attention="kitchen",
    row_linear_backend="tilelang_scaled_fp16",
    fc2_quantizer="fused_stored",
    column_linear_backend="cuda_int8_tilelang",  # measured SM75 DiT columns
    tensor_flow="fp16",
    residual_exponent=6,
    vision_residual_exponent=0,
    video_residual_exponent=0,
    gpu_weight_window_gib=6.0,
    ram_cache_per_rank_gib=3.0,
    dit_reserve_gib=9.0,                 # retained 51,220-token profile
    video_decode_reserve_gib=2.0,
    prefetch_lifecycle=True,
    memory_pressure_aware=True,          # Linux/cgroup host
    adaptive_lookahead=True,             # bounded depth 1–2
    idle_residency=False,                # continuous jobs; see interactive case
    idle_weight_gib=4.0,
    scalar_score_workspace_mib=96,       # inactive with Kitchen
    command_timeout_seconds=1500,
    rank_checkpoint_dir="",             # TP2 descriptor already supplies shards
)
```

For short/medium requests that fit online LoRA, override `lora_mode="online"`
with `lora_selective_qkv_b=True` and `lora_row_partial_reduce=True` retained.
Supported FC1 owned B is always enabled. Owned QKV B has actual-T4 byte/timing support
at261 and4096 rows with the guarded rank128 geometry. These are tested
fixtures, not a universal4096-token admission threshold or an E2E capacity
guarantee. Unsupported factors retain the ordinary path. Select the sampling
reserve for that workload: this profile's9GiB value is historical long-shape
headroom, not a validated short-request optimum. Without adapters, the LoRA
switches provide no benefit.

Keep runtime-only `layers=50`, `cpu_lookahead=16`, `gpu_lookahead=2`,
`cross_request_te=True`, `cross_model_cpu=True`, `decode_te_blocks=2`,
`pinned_bytes=6 GiB` per rank and **one shared** `host_bytes=16 GiB`.
Leave `te_weight_bytes=None`. Fallback SUM is fixed Native INT32; deprecated
Concat A is removed. Use the post-decode handoff for repeated
requests. Video `decode_mode='auto'` is the throughput-oriented choice when
small shape-dependent FP16 rounding is acceptable; Audio stays Native on one
GPU. Exact Video output identity requires `decode_mode='native'` until the
known final-projection geometry case is qualified.

The9GiB sampling reserve comes from completed long-sequence runs; it is not a
universal optimum for shorter clips. Current code also uses this same reserve
for the TE window despite the UI's `dit_` name. A lower shape-specific value
may retain more weights, but cannot be selected responsibly from a single
global VRAM peak. VRAM headroom is checked before weight admission, not
reserved as an allocated9GiB buffer. The6GiB weight number is a ceiling, not
a guarantee that6GiB remains resident beside large activations.

Long-sequence online LoRA is **excluded from the recommended operating profile**
and its further tests/profiling retries are cancelled by the user. The repaired
isolated M51220 timing case passed byte checks, but the combined replay still
OOMed at LoRA; neither establishes whole-request admission. Use bundle-offline
for this workload despite its known CPU preparation cost. A smaller weight
window cannot remove an operator's irreducible activation peak. No complete
baked checkpoint is recommended or produced by this review.

For interactive use with meaningful think time, set `idle_residency=True`
and keep the4GiB idle cap; this also enables lifecycle scheduling. It preloads
the reference-selected VAE encoder as well as TE, subject to admission and
cancellation. For a continuous queue or a one-shot job, leave it false.

## What the physical evidence supports

### Collected T4 operator results

The recovery used two actual Tesla T4 SM75 GPUs, native Torch2.11.0+cu128 /
CUDA12.8, comfy-kitchen0.2.37 and TileLang0.1.13. Timings below are the median
of five aligned samples, taking the slower owner for each sample. Compilation,
loading and correctness checks are outside the steady-state timings. Inputs
were seeded shape fixtures with real trained base shards, not captured
real-distribution activations or a generated H3 trajectory.

| DiT column | Rows | Original ms | Preallocated ms | `tilelang_exact` ms | `cuda_int8_tilelang` ms |
|---|---:|---:|---:|---:|---:|
| QKV, K5376/N10752 | 261 | 8.088 | 8.023 | 7.684 | 1.554 |
| QKV, K5376/N10752 | 4096 | 41.884 | 41.241 | 36.375 | 14.928 |
| QKV, K5376/N10752 | 51220 | 533.289 | 520.692 | 466.791 | 232.461 |
| FC1, K5376/N14336 | 261 | 13.500 | 13.468 | 12.613 | 1.560 |
| FC1, K5376/N14336 | 4096 | 56.869 | 56.476 | 50.053 | 19.578 |
| FC1, K5376/N14336 | 51220 | 742.564 | 736.233 | 668.215 | 338.258 |

All four variants matched the reference as raw storage bytes on both owners
for these fixtures. The direct path is the supported performance recommendation
for the aligned SM75 DiT QKV/FC1 route, rather than an unmeasured candidate.
Its large-shape speedups versus original are about2.29× and2.20×; versus
preallocated, about2.24× and2.18×. Small-M warmup/clock variation is visible in
the retained samples; five measurements do not establish p95 or locked-clock
repeatability.

| M51220 incremental operator peak,GiB | Original | Preallocated | `tilelang_exact` | `cuda_int8_tilelang` |
|---|---:|---:|---:|---:|
| QKV | 3.803 | 3.579 | 3.079 | 3.078 |
| FC1 | 4.811 | 4.604 | 4.104 | 4.104 |

Peaks exclude already-live inputs/weights and take the larger owner value.
They are not full-block or request peaks. Keep `preallocated` as the exact
allocation fallback/control. `tilelang_exact` is now a historical benchmark
control rather than a production selection.
The direct route is guarded to inference/eager execution on SM75 with weight
shapes `(10752,5376)` or `(14336,5376)`. Other shapes/devices use preallocated;
other selected backends, compiled calls and autograd retain the public API.
This is not a global replacement for all TE/GEMM operators.

| Helper | Rows | Original ms | Candidate ms | Recommendation |
|---|---:|---:|---:|---|
| TE Q/K/V preparation | 1182 | 9.626 | 7.619 shared | Retain shared preparation on the supported eager route. |
| TE gate/up preparation | 1182 | 38.553 | 36.578 shared | Retain; this is not a new attention selector. |
| Online QKV LoRA B | 261 | 1.701 | 1.374 owned | Enable the guarded short/medium online option. |
| Online QKV LoRA B | 4096 | 27.148 | 20.842 owned | Same; not long-sequence online admission. |

These helper outputs were also byte-exact on both owners. Owned B reduced
incremental peak from43.538 to32.243MiB at261 rows and674.000 to506.000MiB
at4096 rows. This removes the previous SM120-only evidence limitation for
these T4 cases; it does not establish generation quality across arbitrary
adapters or inputs.

Version1's last full-B warm-up/validation repeat needed a4.10GiB FP32 output
while retaining a1.026GiB GPU oracle. Moving the oracle to CPU repaired that
benchmark lifetime defect. Version2's isolated timing case then completed
byte-exactly, while its combined profiling replay OOMed at LoRA after the
column/TE ranges. The minimal worker trace gate passed, but the full replay
did not complete capture/flush/exit coverage. Therefore **no accepted complete
profiler trace, bottleneck attribution, or E2E speedup follows from this batch**.
It also does not prove that profiler overhead caused the later OOM. Further
long-sequence online-LoRA tests/retries are cancelled, not pending.

Sources: [T4 recovery and cancellation](H3_REVIEW_T4_RECOVERY_20261024.md),
[retained version1 samples](../data/h3_review_t4_partial_20261024.json),
[recovery result/collection index](../data/h3_review_t4_recovery_20261024.json).

### Existing numerical-flow and scheduling evidence

| Decision | Evidence | Practical implication |
|---|---|---|
| Kitchen over Scalar Sparse at the actual long shape | At51,220 tokens, attention GPU duration sums were48.085/50.796s for Kitchen and113.000/118.109s for Scalar. Both had10.175/10.510GiB allocated peaks. | Scalar costs about2.3× attention GPU time here and has no measured request-peak saving. The approximately1000s cold calls were dominated by CPU bundle preparation, so their near-equal duration is not an attention tie. |
| Stored fusion plus scaled transport | At DiT0/S6814, corrected-control row41.500ms became27.727ms; complete block265.846ms became253.446ms. | A useful measured combination, with a smaller block benefit than its row speedup. Fixed-order noisy samples are not a universal1.50× guarantee. |
| Recompute only for memory pressure | Same component: stored+scaled27.727ms and about1116MiB row working set; recompute+scaled35.577ms and about930MiB. | About186MiB saved at that fixture, with about28% higher row latency. Do not linearly extrapolate this peak difference to51,220 tokens. |
| FP16-fix with FP32 numerical islands | Against Native FP32 tensor flow with its original INT8 operators, bounded fused+scaled block relative errors were0.00136683,0.00276694 and0.01515993 for TE0, DiT0 and reused DiT49+LoRA stress. No nonfinites in that corrected batch. | Much more appropriate than unscaled FP16, which overflowed. These are component diagnostics, not generation acceptance. Transport rounding adds more row error than fused butterfly summation in these fixtures. |
| Lifecycle prefetch and host-pressure admission | Historical dual-T4 continuous P0+P1 request344.897s versus baseline355.480s;0 OOM and Native output checks pass on that workload. Most observed gain was outside TE, and cgroup memory still approached30GiB. | Reasonable bounded scheduling choices; no precision cost. Old workload/software and retired Audio parallelism prevent treating these numbers as current-pipeline gains. |
| Idle preload depends on request gaps | Historical standalone15s-gap pilot TE98.685→81.108s; integrated prompt-start cancellation differed from the pilot. Zero-gap cases submitted no idle copies. | Enable for interactive readiness, not as a universal throughput improvement or as a way to shorten the current decode. |
| Bundle-offline has a large cold preparation cost | Current cold bundle preparation medians about20.2s/block; around954–957s consumer-wait sums versus about0.9s H2D per GPU. | It removes inference LoRA activations and remains the long-sequence capacity policy, but remerges evicted blocks on CPU. Increasing GPU lookahead does not fix its producer throughput. |

Sources: [current long-shape trace](BUNDLE_SCALAR_EXPLICIT_FLUSH_20261024.md),
[current-stack fusion/transport](FP16_FC2_FUSION_VIDEO_SPLIT_20261022.md),
[Native FP32 flow reference](H3_NATIVE_FP32_FLOW_20261021.md),
[historical lifecycle/pressure tests](TE_RESIDENCY_P01_RESULTS.md),
[idle pilot limitations](TE_IDLE_RESIDENCY_RESULTS.md).

## Every toggle in the supplied table

“Good” means a useful choice for the stated objective, not that every possible
workload is faster. “Conditional” means the code/evidence identifies a specific
use. “Control” means a diagnostic or resource bound rather than an accelerator.
“Inactive” describes a combination, not a request to delete the feature.

### LoRA

| Switch | Recommendation | Judgment and reasoning |
|---|---|---|
| `lora_mode` | `offline` on TP2 bundles for long sequences; `online` for fitting short/medium requests | Conditional. Long-sequence online testing is cancelled, not a pending promotion. Online avoids base rebaking/requantization and cold full-matrix CPU merges, but adds per-NFE projections, communication and large outputs. Offline removes that online work but changes the INT8 weight approximation. Use a TP2 JSON descriptor for bounded CPU block merges; safetensors offline instead invokes a whole-checkpoint cached bake and is not the recommended long-sequence delivery path. |
| `bundle_offline_lora` | Let the loader derive it | Internal, not an independent UI knob. It becomes true for offline mode plus a `.json` DiT input. False for online mode. No adapters means no merge benefit. Do not manually enable it on an ordinary checkpoint. |
| `lora_row_partial_reduce` | `True` online | Good engineering tradeoff; approximate summation order. A rank128 hidden SUM replaces the large activation gather: O localK3584/rank128 =28, FC2 localK7168/rank128 =56 logical element ratios for one adapter. These are payload ratios, not measured speedups. Stacked adapters add separate hidden SUM launches. False retains the full-input gather for strict ordering comparisons. Inactive offline. |
| `selective_b` | Fixed on; removed from UI/policy | Guarded online rank128 FC1 owned B is always enabled in the pipeline. Unsupported geometries retain full B. Offline has no online adapter. The low-level store still supports explicit benchmark controls. |
| `lora_selective_qkv_b` | Default `True` for supported online LoRA; inactive offline | Good within its rank128 DiT QKV guard. Actual T4 M261/M4096 fixtures are byte-exact and faster with lower incremental peaks. Long-sequence offline remains the policy; isolated recovery success does not qualify a full online request. Unsupported geometries fall back. |
| `lora_concat_a` | Removed | Deprecated field and store plumbing are gone. The loader compatibility boundary accepts old False only; True is rejected. |

### Attention, GEMM and numerical flow

| Switch | Recommendation | Judgment and reasoning |
|---|---|---|
| `te_attention` | `tilelang_group32` | Good balanced dense INT8 choice, preserving the TE mask/GQA contract. `native` is a useful quality/debug control, not an established speed winner. The Group32 TE path currently targets SM75 and falls back elsewhere. |
| `dit_attention` | `kitchen` | Best measured choice for this long workload. `tilelang_scalar` is a measured regression in attention time here; retain it as an optional sparsity method. `tilelang_drop` has no current long-shape win and sacrifices omitted attention. Historical S6814 current-stack Kitchen22.712ms beat drop38.456ms, scalar46.783ms and TileLang dense60.437ms. These synthetic-attention figures do not establish real-distribution quality. `native` is a dense reference option. |
| `scalar_score_workspace_mib` | Leave96; inactive in the recommended mode | Not a general VRAM reservation. Used only by Scalar mode. At B1/H28/N51220, Npad51264 and801 leaves require4,598,995,968bytes, about4385.94MiB, for this score path. Every exposed cap0–512 therefore selects the same inline path. At S6814 the78.3MiB workspace fits96 and can accelerate Scalar. Zero chooses inline summaries, not “disable sparse attention.” |
| `row_linear_backend` | `tilelang_scaled_fp16` | Good speed/accuracy tradeoff. Original INT8 GEMM is retained; the large INT32 partial is power-of-two encoded for FP16 SUM and a fused FP32 epilogue. It is lossy relative to exact INT32 SUM, especially under cancellation. `tilelang_packed` keeps INT32 SUM and is the sensible accuracy rollback. `native` retains the gather-oriented fallback; it is not automatically safer with FP16 activation flow. |
| `fc2_quantizer` | `fused_stored`; `fused_recompute` only if memory requires it | Stored is the measured speed choice and preserves FP32 SwiGLU/rotation/quantization islands. Recompute trades work for a smaller rotated-activation lifetime. `owned_fp32` is an older useful preparation control on the packed DiT path. With scaled transport and no fused producer, `native` and `owned_fp32` both enter the same default rotation preparation: the current scaled API does not receive this selector. |
| `swiglu_padding` | Fixed on; disable branch removed | Disabling lost fusion and could multiply in FP16 before the FP32 island. There is no policy switch or unfused transport branch. Native reference execution remains available. |
| `column_linear_backend` | Default `cuda_int8_tilelang`; `preallocated` fallback | Best measured complete-API column path at M261/M4096/M51220, raw-byte exact on both owners in these fixtures. The aligned SM75 DiT guard remains; no global TE/GEMM replacement or E2E attribution. Epilogue-only selection is removed from production and retained as a private benchmark comparator. |
| `swiglu_reduction` | Removed; Native INT32 fallback SUM hard-coded | This formerly one-value runtime field never overrode policy-selected scaled FP16 transport. The optimized initializer accepts old constant benchmark arguments only; there is no alternate reduction branch. |
| `tp_quantization` | Removed; global-row `scalemax` fixed | No deployment alternative remains. Fixed/gather quantization selection was removed from the optimized initializer; independent rank scales would change the operator. |
| `tensor_flow` | `fp16` | Good principal VRAM reduction with residual gauges and FP32 islands. `legacy` is the original Native FP32 tensor flow, retaining original INT8 operators; it does not mean converting all weights/GEMMs to FP32. `bf16_reference` is rejected by deployment RuntimeConfig on T4. |
| `residual_exponent` | `6` | Keep the tested divisor64. Zero reintroduces observed overflow; increasing it toward12 is not an accuracy improvement and can lose small values. Power-of-two scaling provides range headroom, not extra mantissa bits. |
| `vision_residual_exponent` | `0` | Good Native-preserving choice. Extra scaling has no established need/performance benefit here. |
| `video_residual_exponent` | `0` | Good Native-preserving choice. Do not extend TE/DiT gauge assumptions to the decoder without separate evidence. |

### Memory budgets and scheduling

| Switch | Recommendation | Judgment and reasoning |
|---|---|---|
| `gpu_weight_window_gib` | `6.0` ceiling; lower if the actual request needs it | Useful cap, not preallocation and not total VRAM. Larger values improve repeated-NFE reuse only while activations still fit.10GiB is not a blanket speed setting. If one operator is itself too large, eviction alone will not fix it. |
| `ram_cache_per_rank_gib` | `3.0` on the roughly30GiB Kaggle host | Useful CPU cache.4GiB may reduce some rereads/remerges, but consumes file-cache/construction headroom. Historical cgroup pressure argues against maximizing it blindly. This does not directly reduce VRAM. |
| `pinned_bytes` | Retain6GiB/rank cap | Host staging bound, not6GiB eagerly pinned and not a GPU allocation. With GPU horizon2, normal live staging is much smaller; retired copies and large supported specs still need room. The shared host admission cap remains authoritative. Reducing it is a RAM experiment, not a demonstrated operator speedup. |
| `host_bytes` | `16GiB` shared across both ranks | Good bounded admission. Includes managed pageable/preparation/staging charges. Do not count it as16GiB per rank or add it to every cache limit as independent memory. A larger value can worsen file refaults/host pressure. |
| `dit_reserve_gib` | `9.0` for the retained51,220-token workload | Necessary shape-specific headroom, not a throughput booster. Current implementation also forwards it to TE, so its UI label is narrower than its actual scope.0–1GiB is not an established safe long-sequence setting.9GiB on every small workload can be unnecessarily conservative; separate TE/DiT activation guards would allow better placement. |
| `video_decode_reserve_gib` | `2.0` | Keep separate from sampling. Copying the9GiB sampling reserve here can needlessly reject query-parallel admission and force slow single-GPU decoding. Zero can under-budget transients. |
| `te_weight_bytes` | `None`; explicit override only for stage tuning | None inherits the common weight cap. Setting the same number again is ineffective. It changes TE weight capacity, not its shared activation reserve. |
| `idle_weight_gib` | `4.0` when idle is enabled | Conditional. Useful preloading cap; inactive with idle false.8GiB may just evict other useful state or fail headroom admission. |
| `idle_reserve_bytes` | Keep4GiB | Conditional idle-stage guard. Not an allocated buffer and not a current-request speed setting. Inactive when idle is off. |
| `adaptive_lookahead` | `True` for the variable/long-shape profile | Reasonable bounded memory policy, not a claimed latency win. Current implementation chooses GPU depth1–2 and also adapts CPU horizon using observed preparation/copy/compute times. Controlled current-stack T4 admission2→1→2 passed. Fixed depth2 already retains byte/free-memory guards. |
| `prefetch_lifecycle` | `True` | Good reuse-aware scheduling choice. Prevents useful prefix/cache state from being discarded unnecessarily. Independent of idle; enabling it alone does not start idle GPU speculation. |
| `idle_residency` | False for continuous/one-shot; True for interactive gaps | Conditional, with no arithmetic change. Preload TE and reference-selected VAE encoders only when idle admission allows. It moves work outside the next request; it cannot accelerate the first request or guarantee whole-cycle throughput. Loader automatically enables lifecycle with it. |
| `idle_delay_seconds` | `.25` | Debounce/control, not model math. Zero can speculate on rapid submissions; a large delay wastes useful think time. Inactive with idle off. |
| `memory_pressure_aware` | `True` on the Linux/cgroup host | Good bounded-pressure choice, especially with two CPU preparation streams. Can reduce speculation/concurrency to protect RAM/file cache. It is not a solution for duplicated CPU LoRA work. Missing cgroup telemetry limits its pressure information on other platforms. |
| `cpu_lookahead` | `16` | Queue horizon, not a16-worker pool. Cache/host admission bounds actual preparation.32 cannot turn a slow serial producer into a faster producer;1 reduces overlap opportunity. |
| `gpu_lookahead` | `2` maximum | Good short transfer horizon.1 is a pressure fallback; current adaptive mode chooses depth within1–2. Weight retention across NFEs is separate from this speculative depth. |
| `cross_request_te` | `True` for repeated requests | Good readiness optimization. Starts CPU TE startup work during decode when the handoff is connected. No benefit for an isolated final request; do not confuse it with GPU idle residency. |
| `cross_model_cpu` | `True` | Useful CPU DiT prefetch at known TE completion. Bounded by the same host cache; does not preload both complete models on GPU. Can be disabled for a measured CPU/I/O-contention diagnosis. |
| `layers` | `50` | Model definition, not a safe optimization knob. Reducing it skips trained transformer layers and changes the model. Small values are for debugging/component tests. |
| `decode_te_blocks` | `2` | Good bounded prefix for lifecycle scheduling;1–4 is not a universal higher-is-better scale. It affects this decode-stage TE-block prefetch only when cross-request scheduling, lifecycle and handoff all run. |

### I/O, VAE, launch and environment

| Switch | Recommendation | Judgment and reasoning |
|---|---|---|
| `command_timeout_seconds` | `1500` for cold/long requests | Reliability control, no speed or precision benefit.300s may abort valid cold work; increasing the deadline does not fix a1000s CPU bottleneck. Keep a separate bounded experiment watchdog. |
| `rank_checkpoint_dir` | Empty for TP2 descriptor input; verified directory for ordinary checkpoint input | Good optional storage representation. TP2 readers' `load_rank` takes precedence, so adding a directory then is ineffective. Never point at base-only shards for a separately baked checkpoint: provenance gates must match. |
| `checkpoint_read_chunk_mib` | Removed; owned-range reads fixed | The full-row scratch implementation could roughly double half-column source traffic and was bypassed by main TP2 full-block loads. Scratch budgeting and runtime wiring are removed; positional short-read handling and seek fallback remain. |
| `decode_mode` | Video `auto` for permitted rounding; Native for exact output control | Auto can use both GPUs to reduce attention query work, but replicates decoder weights and still communicates K/V. Admission/API fallback is useful, not a numerical equivalence test. Actual261-token mismatch was first localized to M261→M256 final dense projection geometry, with max valid-frame error0.0004472733; attention/MLP outputs matched. New parent-weight removal has CPU tests; current long-T4 public decoding remains unqualified. Audio always Native. |
| `MiniMaxH3PostDecodePrefetch` | Connect for repeated requests using a plain VAE | Good lifecycle hook. `ParallelDecodeHandoff` already subclasses `DecodeHandoff` and emits decode notifications; the service's once-per-decode guard makes a second wrapper unnecessary there. A one-shot request has no following TE to benefit. |
| `launch_h3_legacy.py --preview` | Omit for deployment | Diagnostic mode; relaxes the launcher's hardware gate. Not a faster one-GPU implementation of TP2. |
| `launch_h3_legacy.py --isolated-ui` | Omit normally | Diagnostic plugin isolation. Can change startup/node availability, not the attention/GEMM critical path. |
| `launch_h3_legacy.py --check` | Preflight only | Imports/checks then exits; not an inference optimization. |
| `launch_h3_legacy.py --comfy-root` | Frozen repository `ComfyUI` | Source binding, not a performance knob. Do not choose sibling ComfyUI-master or upstream HEAD. |
| `CUDACXX` / `CUDA_HOME` / `CUDA_PATH` | Validated CUDA12.8 nvcc for SM75 workers | Compiler/ABI requirement. Not three independent acceleration switches, and not a request to change native Torch's driver/runtime family. |
| `NCCL_P2P_DISABLE` | `1` | Fixed measured topology in TP2 workers. Parent value0 is overwritten. Do not claim a tested P2P speedup. |
| `NCCL_IB_DISABLE` | Retain launcher value`1` | Single-host transport setup. Changing it without usable IB is not an H3 optimization. Launcher enforces it; a direct service caller is a different entry path. |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | Both`4` | Fixed in workers. Parent-only changes are overwritten and ineffective; any real change needs a new transport benchmark. |
| `NCCL_PROTO` | Unset | Worker and launcher clear it. Setting a parent protocol does not change the measured worker behavior. |
| `H3_RUNTIME_LOCK` | Approved local snapshot or authorized Kaggle native-platform profile | Provenance/ABI control, not an accelerator. Bypassing it permits a different experiment, not a free speed improvement. |
| `te_loading.install(mode)` | Removed from production | Coalesced/extra-worker policies failed promotion; the priority mode was an alias. The code lives only in `benchmarks/historical_te_loading.py` to preserve historical experiment reproducibility. Production always uses the original reader/cache. |

The post-decode wrapper conclusion is about duplicate notifications through the
existing service guard; it does not remove the handoff needed by a plain VAE.
Similarly, “inactive” settings may remain useful for another supported mode.

## Negative optimizations and ineffective combinations

The clearest measured bad fit is **Scalar Sparse for this long shape**. Keep
it for sparse-method research or a separately validated distribution; do not
select it expecting the nominal block sparsity to translate into lower latency.
Drop and TileLang dense also lack a performance win over Kitchen in the
available current-stack comparison, but their long-shape behavior is not
proven merely by extrapolating the S6814 screen.

Other avoidable costs are:

1. **Treating bundle-offline as a speed optimization:** CPU rebaking dominated
   the measured cold call. Keep it for long-sequence capacity; that cost is
   not a reason to resume cancelled online long-sequence tests. For fitting
   short/medium online requests, full-gather LoRA is also expensive when the
   partial-reduction contract is acceptable.
2. **Recompute when stored already fits:** pays extra compute for a memory
   saving the request does not need. Conversely, stored can be the wrong
   choice when its rotated buffer causes OOM.
3. **Disabling SwiGLU fusion/padding in FP16:** loses useful fusion and can
   move multiplication before the FP32 safety island. “Native” does not make
   that arithmetic equivalent to Native FP32 flow.
4. **Maximizing every cache/depth:** extra VRAM displaces activations; extra
   anonymous RAM competes with file cache; deeper queues cannot repair a
   slower producer. Reserve0 and reserve12 are not two levels of speed.
5. **Full-row chunked reads on an I/O-limited half-column workload:** fewer
   system calls can cost twice the source bytes. On already-contiguous TP2
   blocks, the option is mostly irrelevant to the main load path.
6. **Idle preload in a continuously occupied queue:** no useful idle window;
   moving work or retaining weights adds no guaranteed request-cycle gain.
7. **Unconditional extra TE preparation workers:** historical promotion
   failed; they can contend for memory/storage and amplify the current full
   CPU LoRA work. They do not solve its algorithmic duplication.

One-value quantization/SUM controls, Concat A, master Selective B, SwiGLU
padding disable and full-row scratch reads have now been removed as described
above. Derived `bundle_offline_lora` remains internal. Development launch flags
and enforced NCCL settings are diagnostics/environment contracts, not exposed
performance choices. Actual fallbacks, numerical controls, every attention
operator, recompute and per-stage memory bounds remain available.

## Supported changes and remaining limits

1. Recommend **`cuda_int8_tilelang` columns now** within the existing SM75
   guards, using the collected byte checks, repeated complete-API timings and
   incremental peaks. This is now the production policy/loader default.
2. Recommend guarded **owned QKV B for short/medium online rank128 LoRAs**
   with guarded FC1 owned B fixed on. Keep whole-request capacity separate from an
   isolated operator result. Long sequences use bundle-offline; online
   long-sequence tests and profiling retries remain cancelled.
3. Retain the existing bundle-offline CPU bottleneck finding. Its duplicated
   full-matrix merge remains outside the prior requested optimization scope;
   this review does not implement it or describe offline as a fast path.
4. Separate TE and DiT activation-reserve policy before claiming smaller
   workload-specific reserves are safe. Query-parallel Video already has its
   separate2GiB guard. Existing window/lookahead settings remain bounded
   starting points, not a newly measured scheduling optimum.
5. Keep profiling **blocked/incomplete** in the evidence record. Partial
   column/TE ranges do not qualify the failed combined replay for performance
   attribution. No new experiment, automatic retry or full-pipeline run is
   scheduled by this update.

The right approximation budget is spent on the small, bounded dense-kernel
and transport changes already exercised. It is not spent on unvalidated
sparsity semantics, smaller layer counts, or removing FP32 stability islands.
No claimed operator gain is added arithmetically into an E2E speedup.
