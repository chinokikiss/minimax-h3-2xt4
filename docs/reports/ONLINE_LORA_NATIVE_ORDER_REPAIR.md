# Online LoRA accuracy repair

> Historical benchmark record: streamed mode has since been removed from inference. Current optional Row Partial is documented in [ROW_PARTIAL_MEMORY.md](ROW_PARTIAL_MEMORY.md); Concat A is deprecated.

2026-10-18. Local accuracy qualification completed **before** submitting
`chinokiki9/h3-native-order-lora-repair` to physical dual T4.

## Retained inference choices

The default remains `scalemax` with Native INT32 reduction. The only optional
streaming mode is `scalemax_streamed`, also with Native INT32 reduction.
The FP32/INT8 reduction selector was removed from the ComfyUI loader. Runtime
configuration and `optimized_tp.enable()` reject those retired settings rather
than silently re-enabling them. Earlier physical results remain preserved.

## Proven numerical causes

The original implementation changed Native floating-point evaluation order:

- It folded each adapter's multiplier into B **before** the B GEMM.
- It concatenated multiple adapters' A projections into a larger GEMM.
- For row-parallel projections it summed separately rounded FP32 A partials.

Controlled ablations used the same real DMAD factors, the same trained block
and inputs, and the independent upstream `LoRAAdapter.h` sequential reference.
They varied one arithmetic choice at a time. TF32 was disabled. No accuracy
threshold was relaxed.

At block 0, S512, strengths +0.6 and −0.2:

| Controlled variant | Block relative L2 | Changed FC2 INT8 payload elements |
|---|---:|---:|
| Native order and full geometry | 0 | 0 |
| Only fold strength into B | 2.1374e-4 | 7,717 |
| Only concatenate A GEMMs | 1.2355e-4 | 5,258 |
| Only reduce FP32 A partials | 5.2042e-6 | 1 |
| Only shard B output rows | 0 | 0 |
| Previous concatenated/sharded arithmetic | 3.4077e-4 | 15,086 |

S257 independently reproduced the same three error mechanisms. These are
small FP32 rounding changes that cross discontinuous downstream INT8
decisions; packed QKV ordering, missing alpha, or omitted FC2 LoRA do not
explain the controlled results. B row slicing alone did not fail these two
cases, but is not assumed safe across every untested GEMM geometry.

Evidence: `semantic_results/h3_lora_rounding_ablation.json` and
`benchmarks/lora_rounding_ablation.py`. This diagnostic is opt-in and is never
installed into inference.

## Repair

`NativeAlignedLoRA` applies each adapter in source order, using its original
A/B rank and full Native GEMM dimensions. Scaling happens **after** B and each
delta is added separately, matching upstream bypass evaluation order.
Arithmetic rank concatenation has been removed from inference.

Column projections select the owned packed Q/K/V or gate/up rows only after
the full Native B projection. Row projections gather input features before
the full A projection, replacing FP32 partial-sum reduction. For SwiGLU,
the pointwise activation is applied locally before gathering, halving payload
relative to gathering packed gate/up; its equality is covered by the learned
FFN byte tests.

This trades performance and memory for correctness: full LoRA factors are
resident on each rank, and row projections communicate feature activations
instead of rank-sized hidden vectors. Host/GPU byte estimates include the
full factors. The adapter cache identity includes the new implementation
version. Base checkpoint identities and original checkpoints are unchanged.

The existing explicit Native refiner MLP wrapper remains necessary: fused
`linear_input_act(fc2, ..., 'swiglu')` skips ordinary FC2 forward hooks. Both
the repaired implementation and the independent reference apply FC2 bypass
explicitly, preserving the fused base operator.

## Local gate: PASS, byte exact

On one RTX 5060 Ti using two logical CUDA ranks:

- Single and two-adapter (+0.6, −0.2) complete blocks 0/25/49 at
  S33/257/512/1024, for both retained modes: **48/48 adapted block cases exact**.
- Corresponding unadapted block baselines remain exact.
- Twenty actual projection cases across blocks 0/25/49 and both refiner blocks
  are exact. Full refiner block 0 at S37/257 is exact after module copying.
- Learned FFN at S513/8192, single and multi-LoRA, both modes:
  **8/8 exact**.
- Two independent synthetic adapters with ranks 7/13, distinct non-integer
  alpha values, positive/negative/zero strengths, packed QKV and row FC2:
  **4/4 exact** against original alpha/strength passed to upstream Native h.
- Nine existing cache/identity/lifetime regression tests pass.
- Loader/runtime checks confirm FP32/INT8 reduction is unavailable.

Receipts: `h3_lora_native_order_single.json`,
`h3_lora_native_order_multi.json`, `h3_lora_native_order_long.json` under
`semantic_results/`. The job builder checks their byte gates and source hashes
before creating the T4 submission. Local timing is not T4 evidence.

## Physical validation

Submitted only after the above local gates passed. The bounded source-only
job downloads pinned ranges for blocks 0/25/49 and their DMAD factors directly
on Kaggle (about 1.275 GB); no model weights are uploaded from this machine.
It checks complete block S33/257/512/1024, learned FFN S513/8192, both retained
modes, rank agreement, latency/VRAM and targeted profiler traces. It rejects
performance acceptance if any Native byte gate fails.

**Physical dual-T4 result: PASS, byte exact.** Torch 2.11.0+cu130,
CUDA 13.0, Kitchen 0.2.37, two Tesla T4 GPUs, one NCCL rank per GPU.
The physical test phase completed in 83.15 seconds after setup/downloads;
no OOM, deadlock or runtime failure was reported.

- 72/72 complete-block cases matched Native bytes and both ranks agreed:
  24 unadapted, 24 single-LoRA, 24 multi-LoRA across all three tested blocks
  and four sequence lengths, in both retained modes.
- 16/16 learned FFN comparisons matched Native bytes through S8192, including
  two forward/reverse ordering rounds. Maximum absolute error and relative L2
  were zero for every physical case.
- Submitted source hashes match the completed worker's runtime source hashes.
  The downloaded job contains the local qualification receipts that gated it.

### Measured cost with repaired online LoRA

S8192 **whole learned FFN**, ten synchronized samples per mode/stack, median
critical wall time across ranks. Timing excludes validation and profiler passes.
Temporary peaks are maxima across ranks/rounds above resident inputs/weights;
absolute peaks also include reference weights and are not full-pipeline VRAM.

| Stack | Mode | FFN ms | Temporary peak MiB/rank | Native bytes |
|---|---|---:|---:|---|
| Single LoRA | Existing `scalemax` | 248.042 | 1792.00 | Exact |
| Single LoRA | Optional `scalemax_streamed` | 273.675 | 1736.00 | Exact |
| Multi-LoRA | Existing `scalemax` | 303.169 | 1792.00 | Exact |
| Multi-LoRA | Optional `scalemax_streamed` | 327.896 | 1736.00 | Exact |

Streaming remains optional: it is slower and saves only 56 MiB at this whole-FFN
boundary with repaired online LoRA. This differs from the earlier *isolated FC2*
memory test. Native-order LoRA adds a full activated-input gather: one 224 MiB
input payload per rank at S8192, plus gather output and concatenation staging.
The source-level live tensor sizes at that point (448 MiB packed FC1 input,
168 MiB base FC2 output, 224 MiB local activation, 448 MiB gather parts and
448 MiB concatenated input) total 1,736 MiB, matching the streamed temporary
peak. This staging is a remaining memory opportunity; eliminating it must
preserve the full contiguous Native A input and needs its own gate.

The S8192 multi-LoRA per-rank traces record one FP32 MAX, one INT32 SUM
(168 MiB input), and one FP32 feature AllGather (224 MiB input) in either mode.
These are collective **tensor payloads**, not measured PCIe link traffic.
No claim is made that the corrected path is faster than the old incorrect one.

Evidence: `semantic_results/h3_lora_repair_t4_v1/`, including compact raw
results, local gate receipts, source manifests, logs, and four compressed
per-rank Chrome traces. Kaggle job:
[H3 Native Order LoRA Repair](https://www.kaggle.com/code/chinokiki9/h3-native-order-lora-repair).

Full-model sampling, all 50 blocks, and DMAD generation quality remain outside
this bounded qualification. Matching the intended critic-backbone quality on
the public Ref2VA checkpoint is not claimed. Further performance work should
start from this exact implementation and retain the existing gate.


The subsequent [byte-exact storage optimization](ONLINE_LORA_BYTE_EXACT_OPTIMIZATION.md)
reduces S8192 FFN medians by 3.6-4.5% and streamed temporary memory by 224 MiB/rank
on physical dual T4, while preserving all tested Native bytes.
