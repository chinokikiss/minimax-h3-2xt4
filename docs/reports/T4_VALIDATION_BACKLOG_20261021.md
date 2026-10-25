# What still lacks real-T4 validation — 2026-10-21

> This is the historical pre-execution backlog. The user subsequently authorized
> native Kaggle Torch with preserved core libraries, and physical follow-up
> experiments ran. See [current results and remaining limits](T4_NATIVE_TORCH_VALIDATION_20261021.md)
> instead of treating the missing-result rows below as today's completion state.

This is a read-only audit of retained reports, local result manifests and current source. No experiment, upload, package installation, full H3 generation or GPU job was launched. “Missing” means no qualifying result was found in the inspected evidence; it does not prove that an inaccessible remote run never occurred.

There are two different gaps: **new work with only local evidence**, and **old T4-tested operators whose current source/dtype/environment is not qualified**. They must not be combined into “nothing has been tested on T4.”

## Prerequisite: current frozen Linux/T4 stack remains unqualified

The frozen local environment is Torch **2.12.1+cu132**, Kitchen **0.2.37**, TileLang **0.1.13**, using pinned ComfyUI `d49e888586dd8ae012c0667b33466b815fee07f7`. Historical T4 operator jobs commonly used Torch **2.11.0+cu130**, TileLang **0.1.15** and nvcc **12.8**. Their results remain valid for their saved snapshots, but do not qualify the current stack.

The retained Linux metadata audit identifies seven Linux-only Torch requirements not covered by the Windows snapshot. No explicit qualified Linux platform lock/wheelhouse is recorded here. The current [operator dependency gate](../../patches/h3_operator_dependencies.py) also still requires historical TileLang 0.1.15 on T4 and nvcc 12.8; it intentionally does not treat frozen TileLang 0.1.13 as qualified. Preserve versions rather than installing the historical versions to bypass this gate.

Resolve package/platform/compiler preparation using CPU-only preflight first. Do not allocate a T4 to install or troubleshoot dependencies. Sources: [frozen runtime](../../runtime/README.md), [runtime qualification limits](../guides/H3_LEGACY_RUNTIME.md).

## 1. Current implementations needing a T4 follow-up

| ID / priority | Experiment or change | Evidence already available | Actual remaining target check |
|---|---|---|---|
| T01 — first | **FP16-fix residual flow and weight folding** | Local trained component checks, 38 configurations against the existing Native FP32 flow; no T4 | SM75 finite/overflow/underflow behavior and state/output fidelity for current TE/DiT shapes, including LoRA. Keep INT8 operators in the reference; do not replace them with dense FP32 |
| T02 — first | **Actual FP16 dispatch through Group32 / packed rows / SwiGLU / owned-FP32 FC2** | The FP32 variants have historical T4 results; local SM120 uses native fallbacks | Trace which backend actually runs with FP16 activations, validate it and measure complete APIs. Several row/SwiGLU/transport helpers have FP32 guards, so earlier FP32 speedups cannot be assumed to apply |
| T03 — second | **`tilelang_scaled_fp16` integrated row backend** | P2 codec/fused epilogues were tested on T4 for historical DiT FP32 boundaries; the later four-site row integration has local M=17 logical-rank gates | Physical NCCL + current wrappers for TE O/down and DiT O/FC2, independent codec reference, cancellation-sensitive inputs and realistic row counts. The wrapper's FP32 eligibility also needs to be reported under the default FP16 flow |
| T04 — second | **Native Token Refiner SDPA wrapper and explicit TE attention/MLP lifetime deletion** | CPU/local wiring and lifetime checks; native SDPA itself is not a new untested primitive | Current integrated SM75 dispatch, output correctness and peak-memory/lifetime behavior. No new target performance result for these particular changes is recorded |
| T05 — second | **Current `adaptive_lookahead` option and its 1–2-group admission behavior** | Local fixture/ownership tests; older adaptive streaming has historical physical results | Current flag/wiring, current memory policy and FP16 workspace coexistence on two devices. The older adaptive run does not establish a stable speedup for this flag/version |
| T06 — second | **Whole-layer offline LoRA baking on two physical GPUs** | Serial/spawned CPU checks and two spawned CUDA workers sharing one 5060 Ti; exact sampled weight/scale bytes | Two distinct T4 devices, layer ownership, output artifact identity, peak VRAM/RAM and baking throughput. Older offline-LoRA checkpoints/generation results do not validate this newer layer-parallel baker |
| T07 — second | **Encoder-first/decoder-last VAE loading and idle VAE-encoder + TE preload** | Trained local Video/Audio component checks; local cancellation, pressure and host-budget fixtures | Actual parent/worker coexistence, Native model-management lifetime, decoder loading and cancellation on dual T4. Keep Audio Native single-GPU; no full H3 request is required for a bounded component gate |
| T08 — regression | **Current pinned Video query/fallback path and Native-only Audio wrapper** | Latest local Video gate: 36 Native / 72 logical-parallel calls. Trained local Audio output is exact for both auto/native; no audio-worker dispatch | SM75 regression of the current source/legacy API, staged Native fallback and single-GPU Audio decode. Video query parallelism and Native audio both have historical T4 evidence; this is current-integration requalification, not a new algorithm |
| T09 — CPU-first | **Current positional `os.pread` reader and fallback** | CPU positional-read fixtures and Windows fallback; earlier T4 I/O studies used their own saved readers | Native Linux file descriptors, short reads, current range layout and target-filesystem behavior. This is primarily a CPU/OS/I/O experiment, not a reason to consume T4 time |

Evidence and explicit scope limits:

- T01/T02: [Native FP32-flow comparison](H3_NATIVE_FP32_FLOW_20261021.md), [residual-fix study](FP16_RESIDUAL_FIX_LOCAL_20261020.md); `D:\attention_eval\h3_native_fp32_flow_20261021` and `fp16_residual_fix_20261020`.
- T03/T04/T05/T09: [five enhancements](../guides/H3_OPERATOR_ENHANCEMENTS.md); `D:\attention_eval\h3_operator_enhancements_20261019\validation.json` explicitly says no T4/NCCL run. [P2 transport scope](TP2_P2_OPTIMIZATION_REPORT.md) is the earlier, narrower physical evidence.
- T05: [older residency results](PIPELINE_RESIDENCY_AUDIT.md) already contain a physical adaptive run; its small, cross-job latency difference is not a current flag qualification or a robust speedup claim.
- T06/T07: [loading-stage validation](../guides/H3_LOADING_STAGES.md); `D:\attention_eval\h3_loading_stages_20261019\validation.json`, `local_layer_bake.json`, `trained_vae.json` explicitly identify the local/single-device scope.
- T08: [Video fallback check](VIDEO_VAE_QUERY_FALLBACK_CHECK.md), [current VAE behavior](../guides/PARALLEL_VAE_INTEGRATION.md), [latest Audio result](../data/audio_native_only_20261021.json), [pipeline review limits](H3_PIPELINE_REVIEW.md).

These rows overlap intentionally: for example T01's FP16 numerics and T02's dispatch/performance can share one bounded operator job after preflight. They are validation dimensions, not nine unrelated GPU submissions.

## 2. Designs and CPU studies that are not yet T4 experiment candidates

| Work | Present status | Next useful step |
|---|---|---|
| **TE compressed-weight streaming** | RAM-resident CPU codec screening only. No integrated compressed loader or measured target read-plus-decode experiment | Pre-fold artifact/schema verification, equivalent RAW-pack control and bounded CPU-only I/O replay before any GPU integration |
| **Bounded stage overlap / former zero-bubble architecture** | Revised proposal, not an implemented cross-stage scheduler | One merged CPU deadline plan and ownership/pressure tests; do not submit the design as if implemented |

Sources: [local codec study](TE_COMPRESSION_LOCAL_20261020.md), [compressed-weight proposal](../design/H3_TE_COMPRESSED_WEIGHT_STREAMING_PROPOSAL.md), [stage-overlap design](../design/ZERO_BUBBLE_STREAMING_ARCHITECTURE.md).

## 3. Older “pending” notes that have been superseded

These should **not** be queued again just because an early design/report says “local only” or “not executed.” Current-stack requalification is still a separate issue.

| Family | Retained physical T4 evidence |
|---|---|
| Kitchen-inspired Group32 v1/v2, dense/drop/scalar mappings | [Group32 T4](TP2_KITCHEN_GROUP32_RESULTS.md), [v2](TP2_KITCHEN_GROUP32_V2_RESULTS.md), [operator pass 2](TP2_OPERATOR_PASS2.md) |
| SwiGLU + Padding fusion | Later physical screen/repeat/Nsight evidence in [SwiGLU fusion](SWIGLU_PADDING_FUSION.md); the introductory local-only paragraph describes an earlier stage |
| Packed row-linear fusion | [Physical row fusion](TP2_ROW_FUSION_RESULTS.md) |
| NCCL automatic vs Simple and four-channel configuration | [Operator pass 2](TP2_OPERATOR_PASS2.md), including complete row-API corroboration. This later experiment covers the protocol question; it does not establish that the older `tp2_nccl_microbench.py` entry point itself was executed |
| P2 scaled-FP16/packed-INT8 transport, fused epilogues, INT8 PV, scalar omitted scores | [P2 report](TP2_P2_OPTIMIZATION_REPORT.md); saved final verification records 28 Systems reports and 70 fresh-process identity checks. Notebook COMPLETE is not enough by itself; internal failed/rejected stages are retained |
| Selective B / Row Partial / Native-ordered online LoRA repair and storage | [Selective B](SELECTIVE_B_FEASIBILITY.md), [Row Partial](ROW_PARTIAL_MEMORY.md), [online LoRA](ONLINE_LORA_BYTE_EXACT_OPTIMIZATION.md) |
| Owned-FP32 FC2 quantizer/lifetime optimization | [FC2 report](FC2_QUANTIZER_MEMORY.md), [T4 v1](../data/fc2_quantizer_t4_v1_summary.json), [integrated S32768 v2](../data/fc2_quantizer_t4_v2_summary.json) |
| Initial TP2 operator integration | Saved `D:\attention_eval\h3_operator_integration_20261018\integration_results.json`: actual two Tesla T4s, Torch 2.11, provider/source binding. It explicitly lists post-target changes and does not qualify later edits |

Additional P2 scope extensions remain untested: the real O-projection call-0/call-19 endpoints were local-only, outside the published T4 fixture subset. Broader layer/step/shape coverage is follow-up work, not evidence that the original P2 operator study never ran.

## 4. Attempted but blocked, or deliberately deferred

- **Nsight Compute hardware counters:** historical T4 probes returned `ERR_NVGPUCTRPERM`. Systems traces exist; achieved occupancy, memory stalls and Tensor Core utilization are not thereby measured. Resolve permissions/tool access before spending another GPU session on the same probe.
- **Sparse semantics/distribution, FP16 full-trajectory quality, approximate transport generation acceptance:** component accuracy/performance cannot establish these. Existing older quality clips do not qualify the current FP16/default/integration revision. Full H3 is currently prohibited; keep these deferred rather than substituting operator L1/L2 for sparse semantic validation.
- **Large idle-residency matrix, physical model-switch/interruption cases:** [idle report](TE_IDLE_RESIDENCY_RESULTS.md) lists the larger matrix and physical switch/interruption gates as pending. The standalone physical idle pilot already completed, but its ComfyUI lifecycle equivalence was partial. Do not launch the roughly 80-request matrix under the current budget/scope.
- **Current integrated single/concurrent-request latency, throughput and peak memory:** not established by the latest local checks. Current service sampling is serialized; concurrency would require a separate supported execution/ownership plan, not simply more benchmark clients.

## 5. Efficient validation order

1. CPU-only frozen Linux/platform/compiler preflight, with no version substitutions and no T4 setup job.
2. One short **FP16-fix + actual dispatch** operator matrix, reusing existing trained fixtures and current TP2 dimensions. Record source/fixture identity, real kernel selection, correctness and ranges, warm samples, and loading/compilation separately. Label synthetic TE inputs and reused late-layer stress inputs honestly.
3. The four-site row backend/collective checks and refiner/TE memory-lifetime checks, sharing fixtures where appropriate. No sampler or full pipeline.
4. Separately bounded dual-device bake, staged VAE/idle ownership and current Native audio/video component regressions.
5. CPU compression/I/O replay and merged-planner tests; GPU work only after a concrete implementation and a CPU gate exist.

This is a proposed order, not authorization to launch jobs or to relax environment gates.

## Live Kaggle status limitation

Read-only CLI queries were attempted during this audit. `kernels list --mine` returned `Not found`; status queries for saved SwiGLU, P2 attention and FC2 notebook IDs returned permission `kernels.get` denied. Therefore **current queued/running status cannot be verified** from this session. No empty-queue, cancellation or new-completion claim is made. The historical completion statements above come from downloaded, version-bound results and receipts, not today's live status.
