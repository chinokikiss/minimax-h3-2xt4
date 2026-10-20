# Online LoRA choices: Native default and optional A schemes

> Historical four-choice results. Concat A is now deprecated and rejected; only Row Partial remains optional. See [ROW_PARTIAL_MEMORY.md](ROW_PARTIAL_MEMORY.md) for the current implementation and memory audit.

Streamed SwiGLU has been removed from runtime and the ComfyUI selection.
`scalemax` is the retained TP quantization mode; quantized base projections
continue to SUM Native INT32 accumulators before dequantization.

## User controls

`MiniMaxH3TP2Loader` exposes two independent booleans, both **false** by default:

| Option | Behavior |
|---|---|
| `lora_row_partial_reduce` | Each row-parallel rank owns only its A input columns. Compute local A partials, FP32 SUM the low-rank hidden activation, then apply each full B and original post-B scale. Avoid the complete activated-input gather. |
| `lora_concat_a` | Concatenate active adapters' A matrices along the rank axis during preparation. Run one A GEMM, split by the original possibly unequal ranks, then apply each separate full B, scale and output addition in source order. |

Both off selects the byte-exact Native-ordered path. Both on combines the two
schemes and uses a single concatenated low-rank SUM for each row projection.
Concatenation alone is a no-op for a single active adapter. Zero-strength
adapters are excluded; alpha normalization and signed strength are retained.

The approximation is limited to A GEMM geometry / partial-A summation. B is not
fused or output-sharded, and strength is not folded into B. These choices can
change downstream INT8 thresholds and are explicitly labeled approximate.

Restart ComfyUI to load the new code. Select the switches on the TP2 loader and
connect `MiniMaxH3TP2LoRAStack` to `lora_stack`. Existing workflows selecting
`scalemax_streamed` must select `scalemax`; retired values are rejected instead
of silently changing behavior. Changing a switch creates a new worker session
and cache identity, preventing reuse of a differently prepared adapter.

Programmatic example:

```python
RuntimeConfig(te_path, dit_path, lora_stack=stack,
              lora_row_partial_reduce=True, lora_concat_a=True)
```

## Implementation and memory accounting

`H3LoRAStore` reads only rank-owned A columns in partial mode. Exact mode still
reads complete A. Concatenated A is stored once, without duplicate per-adapter A
buffers. B remains complete; only selected output rows are scaled/added. The
proven output-view lifetime fix is retained. Adapter identity includes both
options; CPU/GPU byte bounds and residency keys use the selected representation.
Factor counts were checked against actual registered buffers on both ranks.

The refiner can use A concatenation, while row partial reduction applies only
to the TP row projections. Module copying uses registered buffers and immutable
rank metadata, with no bound-method closures or persistent activation cache.
Detailed profiling is confined to the explicit benchmark, not normal inference.

## Local validation before physical submission

- 78 required exact-default checks passed: actual blocks 0/25/49 through S1024,
  real projections/refiner copies, learned FFNs through S8192, and mixed ranks,
  alpha values, signed and zero strengths.
- 40 option cases passed rank agreement / numerical sanity checks: all four
  choices on actual blocks and learned FFNs, single and two-adapter stacks.
  Exact controls matched Native bytes. Max local relative L2: 7.46e-5 for
  partial-A and 7.14e-4 for concatenation/both.
- A separate mixed-rank control used ranks 7/13, signed strengths, independent
  partial/concatenated formulas and the runtime module-copy mechanism.
- Runtime/UI controls were checked: switches default off, configurations and
  adapter cache identities differ, and streamed is rejected at both layers. After the physical run, the new
  switches were appended after all existing widgets to preserve saved workflow
  positions. The loader was exercised with a test service: both True values
  reach RuntimeConfig, defaults stay False and changing them replaces the service.
  This final UI metadata adjustment changes no computation.

## Physical dual-T4 results

Completed [H3 Optional A Schemes](https://www.kaggle.com/code/chinokiki9/h3-optional-a-schemes), version 1.
Compute and runtime source hashes match the working tree. The UI-only widget
ordering adjustment made afterward is separately validated (below). Torch 2.11.0+cu130,
comfy-kitchen 0.2.37. No OOM, deadlock or runtime failure.

The run contains 24 complete-block cases (blocks 0/25/49, S257/512, four choices)
and 32 learned FFN cases (single/multi, S513/8192, four choices, two reversed
ordering rounds). All ranks agreed in every case. All **14 Native controls**
matched reference bytes. Approximate outputs are intentionally not described
as byte-exact.

S8192 whole learned FFN, two copies of the real DMAD adapter with strengths
+0.6/-0.2; ten synchronized samples per configuration, median of the slower
rank per sample. Inputs and computation are matched. Warmup, validation and
profiler passes are outside timing. Results are within-run comparisons, not
cross-job ratios. Adapter memory is for the complete block's LoRA factors,
per rank; temporary memory is the FFN allocation peak above resident weights
and inputs, not full-pipeline VRAM.

| Choice | FFN ms | Latency reduction | Resident adapter MiB/rank/block | FFN temporary MiB/rank |
|---|---:|---:|---:|---:|
| native | 282.303 | 0.00% | 143.50 | 1792.00 |
| row_partial | 198.775 | 29.59% | 133.00 | 1792.00 |
| concat_a | 283.223 | -0.33% | 143.50 | 1792.00 |
| both | 196.190 | 30.50% | 133.00 | 1792.00 |

Partial-A accounts for most of the improvement. Single-adapter S8192 latency
falls from 232.912 to 154.666 ms (33.6%). A concatenation alone has **no demonstrated
independent speedup** in this run; its multi-adapter S8192 median is 0.33% slower.
The combined option is 1.3% faster than partial-A alone, a small two-round
observation rather than a statistically established additional gain. With one
adapter, concat-only is algorithmically identical to Native; timing differences
are noise/order effects. Full sample ranges and round medians are retained.

Partial-A saves **10.5 MiB/rank/block** of resident factors for this two-adapter
stack (7.3% of its adapter memory). It removes the large LoRA gather allocation,
but the full FFN temporary peak is **unchanged** because base FP32
quantization/rotation already determines that peak. A concatenation alone does
not reduce factor bytes; avoiding duplicate packed buffers prevents an increase.
There is no claim that total pipeline RAM or peak VRAM drops by the adapter
percentage.

### Numerical differences

Maxima below span the measured physical block and FFN cases, in raw model-output
units. They are not image/video quality measurements.

| Choice | Max relative L2 | Max absolute difference | Minimum cosine | Native-byte matches |
|---|---:|---:|---:|---:|
| native | 0 | 0 | 1.000000000000 | 14/14 |
| row_partial | 5.98186e-05 | 171.486 | 0.999999998211 | 0/14 |
| concat_a | 0.000522388 | 630.652 | 0.999999863559 | 4/14 |
| both | 0.000522388 | 630.652 | 0.999999863559 | 0/14 |

Partial-A max relative L2 is **0.0060%**; concatenation/both reach **0.0522%**.
The automated approximate-mode gate checks finite output, rank agreement and
relative L2 <= 1% as a gross-regression guard. That guard is not a user quality
threshold; the observed errors above are the actual evidence. Full-depth
sampling can propagate quantization differences and was not run here.

### Communication evidence

The S8192 multi-adapter FFN traces show these LoRA collectives, per rank:

- Native / concat-only: one FP32 activated-input AllGather, **224 MiB** payload.
- Partial-A: two FP32 hidden SUMs, **4 MiB each**.
- Both: one FP32 concatenated hidden SUM, **8 MiB**.

The base INT32 SUM payload remains 168 MiB and its row-MAX payload remains
32 KiB. These are tensor payloads, not measured PCIe traffic. Rank-zero trace
NCCL residence: Native gather 73.10 ms, partial-A hidden SUMs 1.06/1.03 ms,
combined hidden SUM 1.99 ms. NCCL residence includes rank-arrival waits; do not
interpret it as pure link time or sum overlapping activities into wall latency.
This supports eliminating the large gather as the principal cause of the gain.

## Recommendation and evidence

Keep Native as the exact default. Enable partial-A first when accepting this
numerical drift for latency. Keep A concatenation available independently and
in combination; this run does not justify enabling it automatically.

This is selected-block / learned-FFN qualification. It does not establish
end-to-end generation speed or full-model perceptual quality. Reference-image/
video conditioning, masks, RoPE, AdaLN and base operator dispatch are unchanged.
Offline merging remains the existing separate Native merge/requantization path.

- Local: `semantic_results/h3_lora_options_exact_{single,multi,long}.json`,
  `h3_lora_options_local/results.json`, `h3_lora_options_mixed_ranks.json`,
  and the post-run UI proof `h3_lora_options_ui.json`.
- Physical raw results, local gate receipts, hashes, bounded logs and eight
  compressed traces: `semantic_results/h3_lora_options_t4_v1/`.
- Compact summary: `semantic_results/h3_lora_options_t4_v1/summary.json`.
- Reproduce: `benchmarks/h3_lora_options_audit.py` locally or with `--physical`;
  `build_h3_reduction_job.py --a-options` rejects stale/failing local gates and
  packages source only. Selected checkpoint ranges download on Kaggle.
- Summarize: `benchmarks/summarize_h3_lora_options.py <result-directory>`.

Earlier streamed results remain historical evidence in their frozen notebooks;
the live runtime no longer implements or selects streamed mode.
