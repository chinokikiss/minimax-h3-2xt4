# Kitchen dense INT8 on SM75 — 2026-10-25

TE now defaults to **Kitchen causal INT8**, using the complete incoming mask
and the pinned `comfy_kitchen.int8_attention(..., attn_mask=mask)` API. The
loader, runtime policy, two maintained workflows and local deployed INT8
workflow select `kitchen_causal`. Explicit Group32/native selections remain.
The default performs no mask classification and builds no native extension.

The new dense kernel candidates offer modest shape-dependent gains. They are
retained as experiments; DiT/Video continue to use pinned Kitchen. All timings
below are synthetic operator measurements on two physical Tesla T4s, with no
model weights, TP communication or full H3 generation.

## Dense kernel results

Complete API: quantization, attention and BND output conversion. Both GPUs ran
seven samples in each of two reversed-order rounds after warmup. The table
uses the slower GPU of each GPU's median across rounds. The paired ratios in
the [raw data and summaries](../data/kitchen_dense_sm75_20261025/manifest.json)
compare matching rank/round measurements and are preferred for speedup claims.

DiT inputs are B1/H28/D128 FP16 with strided packed V. Video is B1/H32/D64 FP16,
Q1024/773 on ranks 0/1 and K1797.

| Complete API | DiT N512 ms | DiT N6814 ms | DiT N51220 ms | Video D64 ms |
|---|---:|---:|---:|---:|
| Pinned Kitchen 0.2.37 | 0.734 | 19.362 | 994.538 | 1.109 |
| Rebuilt original schedule | 0.691 | 19.140 | 994.215 | 1.029 |
| Query64, original PV | 0.680 | withheld: spills | withheld: spills | 0.939 |
| Cached Q, query128 | — | 19.287 | 954.062 | — |
| PV n8 + 32-bit indices, query128 | — | 23.474 | 963.696 | — |
| PV n8, query64 | — | 22.298 | 1246.143 | — |
| PV n8 + 32-bit indices, query64 | — | 21.931 | 1183.220 | — |
| 32-bit indices, original PV | 0.688 | withheld: spills | withheld: spills | 0.942 |

The original-schedule rebuild also changes timing relative to the installed
binary. Consequently its entire difference cannot be attributed to a new
algorithm or tile size.

At N51220, cached Q improved the original-schedule rebuild by a paired median
**1.046×** (range 1.031–1.055×), and the n8/32-bit candidate by **1.046×**
(1.004–1.075×). Their matched median gains over installed Kitchen were about
**1.038×**. Cached Q had regressed at this length in the earlier independent
[TileLang comparison allocation](KITCHEN_ATTENTION_T4_MATRIX_20261025.md).
This is evidence for a candidate, not a stable universal dispatch rule.

At N6814, the n8/32-bit candidate had an inconsistent result, including a
substantial slower-rank regression; its paired ratios span 0.684–1.040× versus
the rebuild. Query64/n8 variants regressed at medium and long lengths. At N512,
query64's paired median gain over the rebuild was only 1.024×, including a
0.998× comparison. Video query64 improved the rebuild in all four comparisons,
but the paired median was 1.029×. There is no broad high-ROI tile-size change.

## What was changed and preserved

The experimental CUDA C++ extension derives from the retained Kitchen
**0.2.37** archive. Its vendor files and original quantizer, packed scales,
four integer PV products, FP32 FMA accumulation order and output rounding are
preserved. No TileLang IR is introduced.

The new candidates combine the retained `compute_int8_sv_pipelined_n8` helper
with query64/query128 schedules and checked 32-bit tensor addressing. Python
and C++ reject address overflow before selecting narrow indexing. The existing
64-bit path remains the control. Unsupported n8 geometries are not selected.

For D128/key128/query128 FP16, the original schedule compiled to 255 registers
and 4 bytes each of spill stores/loads. The n8/32-bit combination removed the
spill but still used 255 registers. Query64/n8 compiled spill-free, yet its
long-case runtime regressed. **Spill elimination alone did not predict speed.**
The [build receipt](../data/kitchen_dense_sm75_20261025/dense_build_receipt.json)
preserves every specialization's resource counts.

Turing supports at most 255 registers per thread and has a 64K-register SM
file, so register lifetime remains a reasonable scheduling target. This is a
source/resource inference; no SM occupancy or stall-counter measurement was
made. See NVIDIA's [Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#occupancy).

## Preparation versus attention

Median phase measurements on the two ranks, milliseconds:

| Phase | N512 | N6814 | N51220 |
|---|---:|---:|---:|
| Quantization/preparation | 0.353–0.365 | 2.724–2.742 | 29.682–29.721 |
| Prepared original attention | 0.309–0.324 | 16.204–16.390 | 952.869–953.763 |
| BND output copy | 0.086 | 0.545–0.563 | 4.014–4.106 |

These are separately timed phases and should not be summed into a measured
complete call. CUDA event intervals can include host launch gaps. A single
Torch-profiler sample at N6814 independently exposed approximately 12.6–13.6 ms
of pinned attention kernels, 1.50–1.56 ms V quantization, 0.84–0.87 ms Q/K
quantization and 0.48 ms output copy. Profiler samples are attribution evidence,
not paired speedup measurements.

Short calls warrant reducing preparation/launch overhead. At N51220,
preparation is only a few percent of the call, so the attention schedule itself
must improve to produce a large gain.

## CUDA Graph follow-up

The fixed-shape graph experiment wraps the installed Kitchen API and BND
conversion. Every measured replay copies all three current inputs into
persistent contiguous buffers, then clones the output for independent caller
ownership. Capture and three warmups are measured separately. Gates compare
output bytes before and after changing input values and check that later
replays do not overwrite earlier outputs.

The plan requires inference on its creating stream. It has no production cache
or automatic selection. Fixed addresses and graph-private memory pools impose
lifetime and memory costs, as described by the
[PyTorch 2.11 graph documentation](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs).
The follow-up uses the same CPU-qualified frozen bootstrap, with three explicit
hash-bound supplemental source modules; the qualified capsule is not edited.

Both T4s passed all four graph fixtures, including changed inputs and independent
output ownership. The graph offered no gain on the H3-shaped dense cases:

| Case | Pinned Kitchen ms | Graph with copies ms | Paired Kitchen / graph |
|---|---:|---:|---:|
| Tiny GQA: H4/Hkv2/Q65/K131/D128 | 0.249 | 0.128 | 1.953× |
| DiT H28/N512/D128 | 0.675 | 0.708 | 0.952× |
| DiT H28/N6814/D128 | 19.788 | 20.883 | 0.941× |
| Video H32/D64/Q1024 or 773/K1797 | 0.863 | 0.939 | 0.919× |

Each shape ran 11 samples per round on both GPUs, in reversed orders. The ratio
is the median of the four matched rank/round ratios. This was a separate GPU
allocation; compare against its own Kitchen row, not the first table. N51220
graph replay was not measured. The short/mid probe rejected the launch-overhead
hypothesis as a practical H3 optimization when input copies and output ownership
are included; it does not rule out a future graph spanning adjacent operators.

The N512 plan retained about 14.0 MiB of additional allocated memory, with 24 MiB
reserved growth. N6814 retained 186.3 MiB, with 220 MiB reserved growth. Capture
plus warmup took about 4.9 ms for N512 and 115–116 ms for N6814. These costs are
outside the steady-state table. The first plan in each worker, the tiny GQA
fixture, cost about 103 ms including initial graph setup. Model-resident memory
was not tested.

The corrected TE wrapper was byte-equal to direct Kitchen on both ranks. Its
FP16 complete-call latency was 0.369 ms versus Kitchen's 0.369 ms; FP32 was
0.498 versus 0.494 ms. Paired median ratios were 1.006× and 0.994×, with no
material integration penalty. The classifier is absent from this default.

The one final download returned the complete batch, both rank JSONs, both
successful supervision receipts and the source/environment binding, then a TLS
error interrupted the last supplemental-source download. No second provider
request was made. The downloaded sources and exact submitted source payload
were checked against the submission hashes; the missing TE source was decoded
from that payload into a separate recovery directory. The original collection
failure remains visible next to the
[offline evidence verification](../data/kitchen_dense_sm75_20261025/graph_evidence_verification.json).

## TE integration correction

The first integration tried to recognize a plain causal mask and invoke the
implicit-causal native kernel. That classifier rechecked inference-mode
tensors because they expose no mutation version. For N353/H32/Hkv4/D128 FP32,
complete calls measured 0.551 ms for masked Kitchen, 0.487 ms for the direct
native kernel, but **0.823 ms for the classified integration**. Mask recognition
cost more than the kernel saved.

The shipped default therefore forwards the original mask directly to pinned
Kitchen. Boolean masks, padding restrictions, additive biases and compact KV
heads are preserved. The implicit-causal experiment remains explicit through
`use_native=True`; it is not the H3 default. Kitchen's shared quantization
scales/anchor can depend on future input tokens, so causal score masking does
not establish strict streaming prefix invariance.

## Validation and provenance

The Linux CPU preflight passed 70 preserved checks, 21 experiment checks and
22 TE/default checks, plus operator control gates, and compiled the native
library with **CUDA 12.8.93** on CPU. Each T4 passed the 31 original numerical
comparisons, 22 candidate byte-equality cases and eight TE mask/dtype cases.
Timed outputs were checked for shape, dtype and finiteness. These are operator
contracts, not H3 generation-quality evidence.

After the direct-dispatch correction, 52 focused local CPU tests and the three
operator integration control gates passed. An additional 95-test CPU sweep
passed 94; the remaining historical Turbo-header test failed because its local
Turbo/base JSON fixtures had been removed. No model download or replacement
fixture was used to mask that failure. Its log is retained with the results.

The dense allocation used native Kaggle **Torch 2.11.0+cu128 / CUDA 12.8 /
NumPy 2.1.3**, pinned **Kitchen 0.2.37 / TileLang 0.1.13**. The frozen legacy
ComfyUI commit remains `d49e888586dd8ae012c0667b33466b815fee07f7`. The local
conda runtime and `runtime/local_conda.json` were not modified.

The immutable source delta is bound to commit
[`391274244d9ca7083bf40c25586317c494e76bcf`](https://github.com/chinokikiss/minimax-h3-2xt4/commit/391274244d9ca7083bf40c25586317c494e76bcf),
capsule SHA256 `dcbc4c4da405fc8092654e62ef51bf1b7bc15098fac236391774121199d9c5f4`.
It overlays the approved current runtime on the retained source distribution;
the frozen ComfyUI source and core lock remain intact. The TE regression above
was found in that immutable submitted version; the corrected direct default
is tested separately. Full raw samples, compile receipts, environment receipts,
submission bindings and one-shot collection receipts are retained
[alongside this report](../data/kitchen_dense_sm75_20261025/manifest.json).

CPU and GPU notebooks each had fixed deadlines and no dataset/kernel-output
mounts. Each final output batch was collected once after its fixed window.
No Kaggle status/list/live-log polling or H3 pipeline run was performed.
