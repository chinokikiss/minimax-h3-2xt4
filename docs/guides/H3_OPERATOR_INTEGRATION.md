# H3 TP2 operator integration

Use the [pinned local ComfyUI runtime](H3_LEGACY_RUNTIME.md) for deployment;
the legacy Video VAE attention API is retained.

Implemented 2026-10-18. The ComfyUI loader now carries an explicit per-session
operator policy through checkpoint preparation, host caches, GPU residency
copies and persistent TP2 workers. This supersedes the earlier benchmark-only
status of these operator entry points. No complete H3 generation was run here.

| Component | Default | Alternatives / scope |
|---|---|---|
| TE text decoder | TileLang Group32 dense QK, FP16 PV | Native attention override; compact GQA and actual boolean/additive masks |
| DiT main blocks | `comfy_kitchen.int8_attention` | `tilelang_drop`, `tilelang_scalar`, `tilelang_dense`, or `native` |
| DiT token refiner | Native PyTorch SDPA | All DiT modes; Native projections, RMSNorm and RoPE |
| Video VAE | Kitchen INT8 attention | Existing query partitioning and Native public decode planning remain |
| TP row projections | TileLang packed row-linear | Native override or approximate `tilelang_scaled_fp16`; SM75 |
| TE down / DiT FC2 | `tilelang_swiglu_pad` | Separate TE gate/up and packed DiT gate/up; disable with `swiglu_padding` |
| LoRA B GEMM | Selective B for eligible FC1 | Only TP2 H3 `[128,5376]` A / `[28672,128]` B; other shapes/targets retain full B |
| Audio VAE | `multi_gpu_audio_vae.channel_decode` | Existing FP32 stereo split inside persistent workers; VAE `native` override |

The base INT8 GEMM still uses Kitchen's native INT32 accumulator. Row projection
transport defaults to **native INT32 SUM before dequantization**. Packed row-linear
means ownership padding and external-global-MAX INT8 activation preparation;
it does not enable the lossy packed-INT8 or FP16 transport codecs. As of
2026-10-19, `tilelang_scaled_fp16` explicitly selects power-of-two FP16 SUM
with the fused TileLang reconstruction/dequantization epilogue. See
[the five enhancements and validation limits](H3_OPERATOR_ENHANCEMENTS.md).
The INT8-PV attention approximation is not added to the TileLang paths here.

## Configuration and workflow

The linked node is installed at:

`C:/Users/Administrator/Documents/project/ComfyUI-master/custom_nodes/minimax_h3_tp2`

It is a junction to this repository's `custom_nodes/minimax_h3_tp2`. The new
workflow is in both `workflows/video_minimax_h3_r2v_tp2_operators.json` and the
ComfyUI `user/default/workflows/` directory. The source workflow and existing
Native fallback workflow are preserved. Restart ComfyUI to load the node and
updated dependency, then open **video_minimax_h3_r2v_tp2_operators.json**.

`MiniMaxH3TP2Loader` exposes `te_attention`, `dit_attention`,
`row_linear_backend`, `swiglu_padding`, `selective_b` and
`scalar_score_workspace_mib`, plus `adaptive_lookahead=False`. The existing memory budgets, LoRA stack and
optional row-LoRA A partial reduction stay separate. Selective B changes
neither A factor concatenation nor adapter ordering nor post-B scale/add order.
No LoRA means the Selective B switch has no effect.

Python callers can pass the same policy:

```python
from patches.comfy_tp_runtime import RuntimeConfig
from patches.h3_operator_policy import H3OperatorPolicy

config = RuntimeConfig(te_path, dit_path, operators=H3OperatorPolicy(
    te_attention='tilelang_group32',
    dit_attention='kitchen',           # or tilelang_drop / tilelang_scalar
    row_linear_backend='tilelang_packed',
    swiglu_padding=True,
    selective_b=True,
    scalar_score_workspace_mib=96,
))
```

Changing policy recreates the loader service. Policy identity also participates
in streamed block and preprocessing cache keys, including next-request idle
prefetch. Residency copies retain the policy on each module. Worker `stats`
reports the selected policy. The Native fallback for another architecture is
logged; kernels do not pretend to run SM75 code on the local SM120 GPU.

## Attention boundaries

TE receives BHND Q/K/V after Native Q/K norms and RoPE. Group32 preparation uses
the existing fused quantizer and unweighted leaf-mean centering, retaining the
FP32-to-FP16 input boundary and FP16-PV/output rounding. The new mask-aware main
kernel broadcasts compact 2D/3D/4D masks without repeating KV heads or copying
the mask to the CPU. Masked logits enter online softmax in the exp2 domain.
Entirely boolean-masked queries produce zero; a query whose first leaf is
masked can accumulate from later leaves. Native embeddings, vision and outer
conditioning interfaces stay in place. This remains prefill-only; existing
past-KV-cache rejection is unchanged.

DiT sparse modes use Group32 QK and FP16 PV with the existing tau0/leaf64
routing. Their dense prefix comes from the current Native `PackedLayout`:
all text, references, conditioning and generated audio before the final video
span stay protected. A stale/missing layout raises an error rather than using
the old 874-token benchmark constant. Both dropping and scalar pooling remain
available. Prefix-containing leaves retain the existing whole-leaf behavior.

Scalar pooling uses omitted-only precomputed centroid scores when the explicit
score tensor fits `scalar_score_workspace_mib` per rank. The measured H3 shape
needs 82,066,432 bytes (78.3 MiB), within the default 96 MiB cap. Set the cap to
0 to keep inline summaries; larger shapes exceeding the cap also use inline
summaries. Allocation failures propagate; there is no fallback after a
partially executed collective. The tensor's retained-leaf entries are not
computed or read. This is the P2 optimization's unchanged routing method.

The current `ComfyUI-master` inlines DiT input packing and has no
`_embed_and_pack` method. The runtime therefore leases/broadcasts the video and
audio projection outputs at their module boundaries and lets Native assemble
the packed sequence. Rank0's residency use plan accounts for both leases.
The older pinned ComfyUI path keeps its single packed-tensor broadcast.
Conditioning must already be refined in both paths. Proxies are restored after
each lease and released at stage teardown; Native model source is not edited.

The refiner uses a module wrapper around its Native projections/norms/RoPE,
so streamed copies do not retain bound methods pointing at canonical CPU
weights. Video uses its existing Kitchen INT8 query-partitioned decoder.
Audio's rank entry point lives in `multi_gpu_audio_vae`; it reuses each worker's
loaded decoder, preserves left/right ordering, and leaves Native chunking and
normalization in the parent. It does not instantiate the standalone dual-device
manager inside each rank or introduce temporal tiling.

## Dependencies and deployment

The target validation used Torch **2.11.0+cu130**, Kitchen **0.2.37**, TileLang
**0.1.15**, CUDA **12.8.93** nvcc, and two physical Tesla T4s. Install
`requirements-tp2-operators.txt` in that environment. The service checks the
operator ABI before launching workers, requires NCCL and matching GPU
architectures, and selects the CUDA 12.8 compiler before importing TileLang.
Set `CUDACXX` to the validated nvcc if it is not on PATH.

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
