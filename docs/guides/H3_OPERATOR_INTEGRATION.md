# H3 TP2 operator integration

Use the [pinned local ComfyUI runtime](H3_LEGACY_RUNTIME.md) for deployment;
the legacy Video VAE attention API is retained.

As of 2026-10-22, the H3 loader and `RuntimeConfig` default to **FP16-fix +
`fused_stored` + scaled FP16 transport**:
`tensor_flow=fp16`, TE/DiT `residual_exponent=6` (divisor 64), and vision/Video
gauge exponents 0. Audio VAE remains FP32. Select `tensor_flow=legacy` in the
loader, or pass `tensor_flow=TensorFlowPolicy('legacy')` to `RuntimeConfig`,
to use the original FP32 tensor flow. Existing INT8 operators and the frozen
dependencies are retained. See the [Native FP32 comparison](../reports/H3_NATIVE_FP32_FLOW_20261021.md)
for component evidence and its scope.

Update 2026-10-24: supported SM75 DiT columns now default to
`cuda_int8_tilelang`. Lifecycle/host-pressure/adaptive lookahead are enabled,
with a historical long-shape reserve of9GiB and a1500s command deadline.
The loader defaults to offline LoRA; use TP2 JSON descriptors for bounded CPU
merges. Safetensors offline still creates a baked checkpoint. Short/medium online
remains explicit, with guarded FC1/QKV owned B enabled. Long-sequence online
tests are cancelled. See the [configuration review](../reports/H3_CONFIGURATION_REVIEW_20261024.md).

Update 2026-10-25: TE now defaults to `kitchen_causal`, preserving the complete
incoming mask through Kitchen 0.2.37's masked INT8 API. It performs no mask
classification or native extension compilation. Dense DiT/Video defaults
remain Kitchen while additional SM75 dense candidates are evaluated.

Update 2026-10-27: the existing Kitchen selections use the
[shared corrected output adapter](KITCHEN_ATTENTION_DEFAULT.md). FP32 inference
now stores FP32 directly for causal and non-causal calls; FP16/BF16 delegate to
the pinned implementation. FP32 workers use a source/binary-verified native
library cache. Mask classification remains unnecessary. Installed packages,
the frozen ComfyUI source, legacy attention API and tensor-flow default remain.

Implemented 2026-10-18. The ComfyUI loader now carries an explicit per-session
operator policy through checkpoint preparation, host caches, GPU residency
copies and persistent TP2 workers. This supersedes the earlier benchmark-only
status of these operator entry points. No complete H3 generation was run here.

| Component | Default | Alternatives / scope |
|---|---|---|
| TE text decoder | Kitchen causal INT8 | Complete causal/padding/additive mask; Native SDPA remains the explicit override |
| DiT main blocks | Shared Kitchen INT8 adapter | `sol` All-INT8 or `native`; optional per-NFE backend schedule |
| DiT token refiner | Native PyTorch SDPA | All DiT modes; Native projections, RMSNorm and RoPE |
| Video VAE | Kitchen INT8 attention | Forced Native SDPA; Temporal default, Query explicit |
| TP row projections | `tilelang_scaled_fp16`: approximate FP16 SUM + fused FP32 epilogue | `tilelang_packed` INT32 SUM or Native; SM75 |
| TE down / DiT FC2 | `fused_stored`: FP32 SwiGLU / regular ConvRot256 | `fused_recompute`, prior `owned_fp32` / `native`; shared global row MAX retained |
| LoRA B GEMM | Owned B always enabled for eligible online FC1; QKV enabled by default | Rank128 H3 guards; inactive offline; unsupported shapes retain full B |
| DiT columns | `cuda_int8_tilelang` | SM75 aligned QKV/FC1; `preallocated` fallback; epilogue-only control is benchmark-only |
| Audio VAE | Native single-GPU FP32 decode | Parallel audio decoding is removed, including when an old workflow selects `auto` |

The original INT8 GEMM still writes INT32 partials. The default row backend
encodes them with a power-of-two scale, performs FP16 AllReduce SUM, then
reconstructs/dequantizes in FP32 with TileLang. Online LoRA is added before
one FP16 output store. This transport and fused rotation summation order are
approximate. Select `tilelang_packed` for INT32 SUM; select `owned_fp32` or
`native` preparation to restore the prior rotation choice. The base runtime
fallback reduction is hard-coded to Native INT32 SUM, below the policy-selected
scaled codec. See the [promotion record](../reports/H3_FP16_FUSED_DEFAULT_20261022.md)
and [physical T4 results](../reports/FP16_FC2_FUSION_VIDEO_SPLIT_20261022.md).

## Configuration and workflow

The linked node is installed at:

`C:/Users/Administrator/Documents/project/2xT4 minimax h3/ComfyUI/custom_nodes/minimax_h3_tp2`

It is a junction to this repository's `custom_nodes/minimax_h3_tp2`. The new
workflow is in both `workflows/video_minimax_h3_r2v_tp2_operators.json` and the
ComfyUI `user/default/workflows/` directory. The source workflow and existing
Native fallback workflow are preserved. Restart ComfyUI to load the node and
updated dependency, then open **video_minimax_h3_r2v_tp2_operators.json**.

`MiniMaxH3TP2Loader` exposes `te_attention`, `dit_attention`,
`dit_attention_schedule`, `sol_tau`, `video_attention` and `row_linear_backend`, plus
`adaptive_lookahead=True`. Removed attention backends cannot be re-enabled by
retired loader options. Existing memory budgets, LoRA selection and row-LoRA
partial reduction remain independent of attention selection.

Python callers can pass the same policy:

```python
from patches.comfy_tp_runtime import RuntimeConfig
from patches.h3_operator_policy import H3OperatorPolicy

config = RuntimeConfig(
    te_path, dit_path,
    operators=H3OperatorPolicy(
        te_attention='kitchen_causal',
        dit_attention='kitchen',           # or sol / native
        row_linear_backend='tilelang_scaled_fp16',
        fc2_quantizer='fused_stored',
        column_linear_backend='cuda_int8_tilelang',
    ),
    dit_attention_schedule=((2, 7, 'sol', 1.0),),
    sol_tau=1.0,                           # inherited by Sol entries without an override
    video_attention='kitchen',             # or native; Sol is DiT-only
)
```

For adapter-bearing TP2 JSON descriptors, direct `RuntimeConfig` callers must
set `bundle_offline_lora=True` to use offline block merging. The loader derives
this from its default `lora_mode='offline'` and the descriptor suffix. A bare
runtime configuration does not guess whether an ordinary checkpoint was baked.

The one-value quantization/base-SUM controls, deprecated Concat A and full-row
column-read scratch path have been removed from runtime configuration.
Old Python loader calls accept only a narrow retired-argument shim; it cannot
reactivate removed algorithms. Browser workflows migrate removed widget slots
by name. Both maintained workflows use configuration version2. Restart ComfyUI
and reload its page to load the migration extension.

Changing policy recreates the loader service. Policy identity also participates
in streamed block and preprocessing cache keys, including next-request idle
prefetch. Residency copies retain the policy on each module. Worker `stats`
reports the selected policy. The Native fallback for another architecture is
logged; kernels do not pretend to run SM75 code on the local SM120 GPU.

## Attention boundaries

TE receives BHND Q/K/V after Native Q/K norms and RoPE. Since 2026-10-25 its
default is Kitchen causal INT8: all masks pass unchanged to Kitchen 0.2.37,
with compact GQA inputs and BND output. No native build or mask readback is
required. The optional experimental implicit-causal kernel is not selected
automatically: classifying inference-mode masks cost more than it saved in
the short T4 probe. The [physical T4 causal/precision audit](../reports/TE_CAUSAL_PREFIX_PRECISION_20261026.md)
confirmed that packed masking passes while future floating-input changes can
alter earlier outputs, even at fixed lengths. FP32 inputs use INT8 attention
and BF16 output storage before conversion back to FP32; the dtype does not
imply FP32 numerical precision.

The retired TileLang Group32 TE path is removed. TE attention now exposes
only `kitchen_causal` and `native`. Browser workflows that saved
Saved workflows that contain `tilelang_group32` migrate to `kitchen_causal`;
Python policy construction rejects the removed backend.

DiT attention exposes `kitchen`, `sol` and `native`. Sol derives its
protected span from the current Native `PackedLayout`; the protected prefix
ends at the start of the final target-audio segment, so generated target audio
and video remain normally routed. Protection is bidirectional at Q64/K64 routing
granularity. Optional `dit_attention_schedule` selects a backend by 1-based
denoiser NFE. A `sol@tau` schedule entry overrides the global `sol_tau` for that
NFE range; plain `sol` inherits the global value. Kitchen and Native entries do
not accept tau overrides.

Selected DiT Sol is AOT-prepared before denoising. Explicit DiT Sol
selection is fail-closed rather than silently falling back to Kitchen.

Video VAE attention supports only Kitchen INT8 or Native SDPA. The Video VAE
Sol routing, D64 adaptation and Sol AOT path have been retired. Temporal
clip parallelism remains the default; Query partitioning remains explicit.
Audio remains Native single-GPU FP32.

## Dependencies and deployment

Current core gates require Kitchen **0.2.37**, TileLang **0.1.13** and the
preserved legacy source. The 2026-10-22 dual-T4 batch used native Kaggle Torch
**2.11.0+cu128**, CUDA **12.8**, and nvcc **12.8.93**. This qualifies its recorded
operator/component cases, not full H3 generation or the local Windows ABI on
Linux. Historical `requirements-tp2-operators.txt` describes the older 0.1.15
qualification and must not be installed over the preserved environment. Keep
the CPU-prepared pinned-core overlay and native-Torch profile for Kaggle. The
runtime still requires NCCL, two matching GPUs and the CUDA 12.8 compiler.

Each worker sets `NCCL_P2P_DISABLE=1`, min/max channels4 and leaves NCCL protocol
selection at its default. These settings are local to the workers. Single-GPU
local preview remains a separate test harness; it cannot start the physical
TP2 service.

The local ComfyUI environment originally contained Kitchen0.2.33, whose private
quantization API rejected the retained TP2 arguments. After isolated0.2.37
wiring passed, that environment was upgraded to **Kitchen0.2.37** with
`--no-deps`. Its Torch2.12.1+cu132 and TileLang0.1.13 were not upgraded; they
were used only for the explicitly labeled local SM120 arithmetic previews.
The T4 deployment pin is separate. Restarting ComfyUI is required for an
already-running process to import the new package/node.

## Validation and evidence

[Kaggle operator smoke job](https://www.kaggle.com/code/chinokiki9/h3-tp2-operator-integration),
version1, ran the source-bound suite on two physical T4s. Its internal status
is `complete`, with **10 passing gate records per rank**. No H3 checkpoint was
attached, uploaded or loaded. Setup took about 122s for Torch and the gate stage
about 91s; these are setup/test durations, not operator latency measurements.

The target gates cover policy/UI/cache/residency propagation, Selective B's
two owners and ordered two-entry stack, audio channel ownership, masked/ragged
GQA including empty rows, fused TE preparation and actual dense/drop/scalar
dispatch. Scalar precomputed and inline routes are compared. Actual NCCL
DiT-style packed7168 and TE-style separate12800 local-width row operators
match the independent full-gather/Native quantization plus INT32 SUM reference
**bit-for-bit on both ranks**. These row gates use17 rows, not full generation
shapes or new performance measurements.

The local suite passes seven arithmetic/control gates, plus four tiny-block
wiring checks: actual DiT Kitchen/row calls, TE Group32/GQA/mask/down-projection
calls on two logical owners, Native refiner wrapper after residency copying,
and the current Native model's zero-block forward through the inline projection
bridge. The last check matches Native video/audio outputs bit-for-bit on both
logical owners. CPU controls also exercise both old/new packing seams and loader
reuse/recreation on backend changes.
These use seeded small tensors, not learned full-model quality comparisons.
The initial local dependency mismatch log is retained alongside the passing
isolated and final-environment runs.

Raw evidence and machine-readable results:

- `D:/attention_eval/h3_operator_integration_20261018/integration_results.json`
- `D:/attention_eval/h3_operator_integration_20261018/t4/v1/integration/`
- `D:/attention_eval/h3_operator_integration_20261018/t4/v1/provider/`
- `D:/attention_eval/h3_operator_integration_20261018/local_gates.json`
- `D:/attention_eval/h3_operator_integration_20261018/local_pipeline_wiring.json`

The provider notebook and runtime hashes are compared with the immutable local
submitted-source snapshot. GPU operator sources remain identical to that
snapshot except the separately recorded residency compatibility bridge.
Dependency/compiler preflight, that current-ComfyUI bridge and final CPU/UI/local
GPU tests were added afterward; their final checks and hashes are recorded
separately. The bridge itself was not run on physical T4 here. The prior P2 report
remains historical evidence, not a performance claim for this new dispatch mix.

Re-run the local gates without loading H3:

```powershell
python benchmarks/test_h3_operator_integration.py --cuda-preview --output D:/attention_eval/local_operator_gates.json
python benchmarks/test_h3_pipeline_wiring.py
```

Full 50-layer sampling, real generated-video/audio comparison, cancellation
through the new complete dispatch mix, and integrated two-T4 latency/VRAM
remain unmeasured. The earlier P2 speedups are not added together or advertised
as a pipeline gain. Optional sparse attention still requires generation-level
semantic/distribution review; operator agreement does not replace that review.
