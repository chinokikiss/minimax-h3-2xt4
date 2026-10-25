# TP2 operator enhancements — 2026-10-19

Current 2026-10-24 defaults/cleanup: [configuration review](../reports/H3_CONFIGURATION_REVIEW_20261024.md).
Actual T4 byte/timing gates support the direct DiT column route and guarded
short/medium online QKV B. Offline LoRA, lifecycle/pressure admission and adaptive
lookahead are now loader defaults. Full-row column reads, the SwiGLU-disable
path and redundant single-value controls were removed. Saved widgets migrate
by name; the implementation history below is superseded where noted.

Earlier 2026-10-24 [performance review follow-up](../reports/H3_REVIEW_OPTIMIZATIONS_20261024.md)
adds one-destination column outputs, shared TE eager preparation, metadata-only
Video remote planning, immutable-proof reuse and unused-reply elimination.
introduced column/QKV candidates and bounded column reads. The later T4
recovery and cleanup above supersede their pending status and widget layout.

Update 2026-10-21: Audio VAE parallelism has since been removed. Audio always
uses Native single-GPU FP32 decoding; the audio selection statement below
describes the original enhancement integration date.

Current update 2026-10-22: the loader defaults to FP16-fix,
`fc2_quantizer='fused_stored'` and `row_linear_backend='tilelang_scaled_fp16'`.
Native INT32 SUM remains the fixed base/fallback contract; policy
selects the scaled codec above it. Prior packed/owned-FP32 choices remain
selectable. The [promotion follow-up](../reports/H3_FP16_FUSED_DEFAULT_20261022.md)
describes FP32 preparation, LoRA-before-store order, 0.1.13 core qualification
and remaining end-to-end limits. The five enhancements below retain their
historical integration scope where explicitly dated.

## Configuration

```python
from patches.comfy_tp_runtime import RuntimeConfig
from patches.h3_operator_policy import H3OperatorPolicy

config = RuntimeConfig(
    te_path, dit_path,
    adaptive_lookahead=True,  # default; bounded to GPU depth 1–2
    operators=H3OperatorPolicy(row_linear_backend='tilelang_scaled_fp16'),
)
```

Both options are exposed in `MiniMaxH3TP2Loader`. Legacy saved widgets migrate
by name through the browser extension. Changing either selection recreates
the persistent worker service.
The linked ComfyUI node uses this repository directly. Restart ComfyUI to reload
the Python changes. The generated `video_minimax_h3_r2v_tp2_operators.json`
workflow has been updated in the repository and ComfyUI's workflows directory;
its new options retain the defaults above. Native/source workflows are untouched.

## Implementation contracts

- **Scaled FP16:** current TP2 FP32, INT8-weight, vector-scale row sites on
  physical SM75 use the existing power-of-two encoder, FP16 AllReduce SUM,
  and TileLang fused reconstruction/dequantization. It reconstructs INT32 before
  the two left-associated FP32 scale products. This transport is approximate;
  byte equivalence is to its independent FP16 codec reference, not INT32 SUM.
  Packed/global-MAX quantization and fused SwiGLU padding remain where the
  existing packed backend uses them. Native fused ConvRot sites retain their
  full-gather/native quantizer, replacing only transport and epilogue. Unsupported
  device/dtype/world-size/scale geometry falls back to native INT32 SUM with a
  warning. Native INT32 fallback SUM is hard-coded; the scaled backend remains
  a policy selection, now the default.
- **TE lifetimes:** attention buffers are deleted after residual addition,
  before MLP allocation. The temporary SDPA capability object is also released.
  Eager SwiGLU activation is deleted after projection; normalized/gate/up buffers
  are released before residual addition, then MLP output is deleted. Arithmetic
  and residual order are unchanged. This is inference lifetime management;
  autograd may retain tensors if used outside the inference runtime.
- **Checkpoint reads:** `read_tensor` uses absolute-offset `os.pread` when
  available. Short reads loop at the next absolute byte; unsupported descriptors
  resume safely through seek/readinto. Destination offsets preserve ordered,
  noncontiguous row/column selection. Pread returns bytes: it cannot write into
  a caller buffer. Those bytes are copied into the final tensor in chunks of at
  most 8 MiB. This avoids file cursor mutation, not the bytes copy or OS page
  cache; it is not `O_DIRECT`. Windows normally uses the fallback.
- **Adaptive residency:** both fresh and idle-reused windows honor
  `RuntimeConfig.adaptive_lookahead=True`. When enabled, GPU depth is 1 or 2
  distinct upcoming weight groups, based on driver free memory plus reusable
  allocator memory, reserve and configured weight budget. Admission rechecks
  the limits before each copy. Existing resident groups need no second copy.
  The existing CPU preparation timing adaptation remains opt-in under the same
  adaptive flag. Windows are constructed in the runtime before `attach_window`
  attaches them to model proxies. No second window is constructed in that hook.
- **Token Refiner:** `NativeRefinerAttention` preserves the base module's
  projections, RMSNorm and Native fused RMSNorm/RoPE, and explicitly calls
  `torch.nn.functional.scaled_dot_product_attention`. It applies independently
  of the main DiT attention selection. `KitchenRefinerAttention` remains an
  import-compatible alias; it no longer calls INT8 attention. Worker stats expose
  `refiner_attention='native_sdpa'` and the adaptive flag.

## Validation

Fresh evidence is in `D:/attention_eval/h3_operator_enhancements_20261019/`.
The previous T4 integration evidence and source manifest were not rewritten.

| Check | Result | Scope |
|---|---:|---|
| New enhancement regression suite | 12 tests pass | CPU contracts, partial reads, fallback, concurrency, dispatch, UI, byte-exact TE arithmetic and weak-reference lifetimes |
| Existing idle window/policy/loading/lifecycle/identity suites | 14 tests pass | Includes local CUDA weight reuse/lifetime checks |
| Existing operator integration | 7 records pass | CPU controls and explicit local CUDA arithmetic preview |
| Tiny actual-ComfyUI pipeline wiring | 4 records pass | Two logical owners on one local GPU; refiner matches native SDPA with and without RoPE |
| Independent codec/fused epilogue | 10 records pass | Random, zero, bounds, cancellation and ties/outliers; FP16 and INT8-gather reference arithmetic |
| New complete row-operator gates | 8 rank records pass | Four TP2 K/N shapes, two logical owners; default INT32 exact, scaled-FP16 independent reference exact |

Row cases use M=17 for arithmetic, not a deployment latency estimate:

| TP2 site | Local K | N |
|---|---:|---:|
| DiT O | 3584 | 5376 |
| DiT FC2 | 7168 | 5376 |
| TE O | 4096 | 5120 |
| TE down | 12800 | 5120 |

FC2/down also pass with SwiGLU padding disabled. The default row outputs are
byte-exact against full-gather/native quantization plus INT32 SUM. The optional
outputs are byte-exact against independent FP16 encode/sum/reconstruct followed
by native dequantization. The refiner's deliberate switch from INT8 to SDPA
does not claim byte equivalence with its former INT8 implementation.

GPU checks here used RTX 5060 Ti SM120 with explicit test-only factory
retargeting and logical transport. Production SM75 guards remain intact.
There was no new physical T4/NCCL run, full H3 generation, profiler capture or
speedup measurement. Local transport simulation does not validate NCCL behavior.
Linux pread was exercised through deterministic positional-read fixtures;
native Windows fallback was exercised by the local integration/fixture paths.

Reproduce the new tests in the ComfyUI Python environment:

```powershell
python benchmarks/test_h3_operator_enhancements.py
python benchmarks/test_h3_operator_integration.py --cuda-preview --output D:/attention_eval/h3_operator_enhancements_20261019/local_gates.json
python benchmarks/test_h3_pipeline_wiring.py
python benchmarks/test_tp2_p2_epilogue.py --local-preview --output D:/attention_eval/h3_operator_enhancements_20261019/local_epilogue.json
python benchmarks/test_h3_scaled_fp16_row.py --local-preview --output D:/attention_eval/h3_operator_enhancements_20261019/local_scaled_fp16_row.json
```

The new row gate also supports physical SM75 execution via
`torchrun --standalone --nproc_per_node=2 benchmarks/test_h3_scaled_fp16_row.py --distributed --output /path/to/row_gates.json`.
Use the existing pinned T4 environment and NCCL transport settings. No models
or full generation are needed for that gate.
