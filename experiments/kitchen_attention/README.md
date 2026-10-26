# Native Kitchen attention experiments on T4

This is an isolated, **experimental backend with physical T4 operator checks**,
based on the retained
comfy-kitchen 0.2.37 source archive. It uses CUDA C++, nvcc and Kitchen's existing
quantizer; it does not use TileLang as an IR. Importing it neither builds code
nor changes production dispatch. The installed package, frozen ComfyUI source,
legacy attention API, environment snapshot and historical experiments remain
unchanged.

The [2026-10-25 T4 matrix](../../docs/reports/KITCHEN_ATTENTION_T4_MATRIX_20261025.md)
completed on both GPUs, with all native gates passing. Native causal masking
and the short D128/query64 dense candidate improved measured latency. Native
drop/pooled beat their long TileLang counterparts but **did not beat Kitchen
INT8**. Cached Q did not consistently improve long dense attention. No
generation quality or production dispatch acceptance is implied; keep this
backend opt-in. The matrix receipt records native Torch 2.11.0+cu128, Kitchen
0.2.37 and TileLang 0.1.13 separately from the local conda environment.

## Implemented contracts

`te_precision.py` audits the pinned TE default's raw-input prefix invariance
separately from fixed-packed masking. It compares FP16/FP32 inputs to explicit
FP32 GQA attention, checks a CPU FP64 oracle, and records the BF16 storage
boundary. It changes no runtime dispatch. See the
[dual-T4 results](../../docs/reports/TE_CAUSAL_PREFIX_PRECISION_20261026.md).

The follow-up [SM75 dense experiment](../../docs/reports/KITCHEN_DENSE_SM75_20261025.md)
adds retained n8 PV scheduling and guarded 32-bit addressing, without changing
quantization or accumulation order. `dense_bench.py` measures matched complete
calls and separate preparation/attention/output phases. `graph.py` is an
explicit fixed-shape replay plan around the unchanged installed Kitchen API;
every call copies current inputs and returns an independent output. It has no
production cache or automatic dispatch. `graph_delivery.py` reuses the exact
CPU-qualified frozen bootstrap and supplies hash-bound supplemental sources
without editing that capsule.

| Mode | Behavior |
|---|---|
| Dense baseline | Original Kitchen QK/PV primitives, layouts, quantization and default key/query tile sizes |
| Dense `q64` | Query tile 64; original key tile and packed operands retained |
| Dense `cache_q` | D128/key128/query128 candidate retaining Q in registers on SM75 |
| Causal dense | Implicit lower triangle, key index <= local query index + explicit nonnegative `query_offset`; no full token mask |
| Sparse drop | Sorted leaf64 lists, GPU ballot compaction, no QK/PV work for unselected blocks |
| Sparse causal | Explicit leaf64 routes intersected with the token causal mask |
| Sparse pooled | Fine Kitchen attention plus tiled FP16 Tensor Core attention over omitted K/V means, count-weighted normalization, fused output merge |

The adapter supports batch, ragged Q/K lengths, D64/D128 and GQA without KV-head
replication. Inputs are inference-only. FP32 consumers retain Kitchen's BF16
output rounding followed by conversion back to FP32. D256, arbitrary additive
masks, dropout, backward and streaming KV-cache mutation are outside this
experiment. Noncausal sparse content routing currently requires self-attention;
explicit block routes support rectangular/GQA attention.

The noncausal routing function ports the existing FP16-centroid, tau=0,
local/global and dense-prefix policy. CPU tests compare its routes exactly to
the old implementation, including ragged leaves. Causal sparse execution
requires explicit routes because full-query-leaf centroid routing can observe
later queries. No automatic H3 policy changes are installed.

### Causality boundary

`is_causal=True` implements **score masking on the packed operands**. This is not
a claim of strict prefix invariance of the complete floating-point-input API.
Kitchen's original Q/K shared scales, K anchor selection and V scales can depend
on tokens beyond a query's causal boundary. Changing those future inputs can
therefore change the quantization of earlier inputs. A strictly streaming causal
quantizer requires a separate numerical contract; it is not silently substituted
here. The physical gate separately checks that changing future V **with fixed
packed scales/operands** does not affect earlier queries.

### Pooled numerical boundary

The new pooled mode is **not byte-equivalent to old Group32 Scalar**. Its means
are formed in Kitchen's already rotated, centered and dequantized operand space.
This gives fine and coarse logits the same key-anchor shift, which is essential
when merging their normalization states. Fine P/V remain Kitchen INT8; coarse
P/V are FP16 with FP32 state. Fine output rounds before the merge. These are
explicit approximation boundaries requiring operator and generation validation.

The coarse kernel batches 16 centroids per update. It dequantizes Q into shared
memory and merges each output tile directly into the fine output. It does not
materialize a full Q-half copy, an N-by-centroid score matrix, a full coarse
output or a separate merge tensor. Extra persistent pooled buffers are K/V
centroids plus one fine LSE tensor. Block masks/indices are block-sized arrays,
not token-pair arrays. Peak memory under actual model residency is unmeasured.
Fully selected rows preserve fine output bytes; empty fine rows have zero
output and negative-infinite fine LSE before the coarse merge.

## CPU checks and compilation

Use the prescribed conda Python locally:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'C:/ProgramData/miniconda3/envs/comfyui/python.exe' -m experiments.kitchen_attention.test_contracts
& 'C:/ProgramData/miniconda3/envs/comfyui/python.exe' -m experiments.kitchen_attention.build --output D:/attention_eval/kitchen_native_20261025/native_build
```

Build is explicit and CPU-only. No GPU module is loaded, package installed or
source downloaded. The vendor manifest verifies every retained header and
license file. A derived header is generated in the build directory; vendor
files are never edited. The build receipt includes source/library hashes, exact
compiler identity and per-specialization register/spill data. `--device-only`
produces a cubin for resource inspection. The Python adapter deliberately
refuses execution on local SM120.

Linux device jobs use the existing CUDA 12.8.93 compiler. The CPU notebook has
no toolkit, so `cpu_compiler.py` extracts NVIDIA's exact hash-verified 12.8.1
redistribution (nvcc 12.8.93 and matching headers/libraries) into a temporary
directory for compilation only. It installs no packages and retains the native
Torch family. CPU compilation must pass before a GPU notebook can be generated.
Windows CUDA 13.3/MSVC compilation is not Linux CUDA 12.8 ABI or T4 execution
evidence.

## Physical qualification, explicitly invoked

Run after CPU bootstrap qualification on actual T4 hardware, against the native
library compiled on that platform. For example, inside an already approved T4
operator job:

```bash
python -m experiments.kitchen_attention.build --output /tmp/h3-kitchen-native
python -m experiments.kitchen_attention.qualify --library /tmp/h3-kitchen-native/kitchen_sm75.so --output /tmp/h3-kitchen-gate-gpu0.json --device 0
```

The bounded gate checks both dense raw-input and prequantized baselines against
the pinned installed Kitchen, exact dense candidate output bytes, compact-list
integrity, ragged/GQA/offset causal masks, empty rows, fixed-packed future-V
isolation, density-one pooling identity and a separate pooled mathematical
oracle. Masked/drop comparisons allow the declared INT8 probability rounding;
these tolerances are implementation checks, not generation-quality gates.

Add `--benchmark` for paired complete-API timing after the gates pass. Default
input is synthetic H28/N6814/D128; it is not a trained H3 trajectory. Use
`--tokens 51220` only within an approved bounded long-operator experiment. For
real evidence pass an existing **private** BHND Q/K/V `--capture`, its actual
`--prefix-tokens`, and `--head-rank 0` or `1` for the TP2 head partition. The
capture stays local/private; no tensors are saved to the result receipt.

Benchmarks include raw quantization, content routing, compaction, pooling and
output conversion. Both randomized and reversed orders run, with warmups,
event/host distributions and peak allocated memory. Newly spilling dense
specializations are withheld from performance selection using the exact
library's build receipt; the original baseline remains visible. Run on both
T4s and compare against Kitchen within the same allocation. Do not infer speed
from register counts, synthetic inputs, another GPU or different environments.

`delivery.py` explicitly publishes immutable sources and submits a private,
zero-mount notebook once per invocation. It never checks Kaggle status or live
logs. CPU setup and native compilation precede bounded two-T4 operator gates
and matched short/long comparisons against TileLang Group32, drop and Scalar
Sparse. Every invoked qualification writes a final pass/failure receipt. An
external job/process deadline also bounds a GPU driver hang; the script's host
deadline alone cannot interrupt a stuck kernel. No weights or H3 pipeline run.

## Provenance

Frozen official archive SHA256:
`94d06941b4bb8e523008d29157a9c0f0522cd3e37472ad651fc13ffc6a55669c`.
Source was copied from the already retained local archive extraction, with all
file hashes checked against its prior manifest. Apache-2.0 LICENSE, upstream
NOTICE and original attribution headers are retained in `native/vendor/`.

Hardware design reference: NVIDIA's
[Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html).
SM75 uses synchronous copies in the frozen Kitchen copy helper; Ampere-only
`cp.async` machinery is not assumed. The old Ada cached-Q specialization is a
candidate to test on T4, not existing T4 performance evidence.
