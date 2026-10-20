# Current TP2 operator optimization

Updated 2026-10-16. This supersedes the native/full-shape target matrix in
`OPERATOR_OPTIMIZATION_PLAN.md`. The target is the current `comfy_tp_runtime`
service with `optimized_tp.enable('scalemax')` and `AlignedVideoDecode`.
This phase runs operators only. Scalar pooling remains an explicit DiT option.

Latest scope: sparse route/pooling behavior is frozen. Generation-distribution
and semantic validation are deferred until full-H3 GPU time is available; L1/L2
error against dense attention is not a sparse quality gate. Performance-only
offline candidates and checks are tracked in `TP2_OFFLINE_OPTIMIZATION.md`.
The new bounded T4 screen, independent-process confirmation, communication
tests and profiler evidence are complete in `TP2_OPERATOR_PASS2.md`.
The following row-linear fusion pass is also complete in
`TP2_ROW_FUSION_RESULTS.md`: external-global-max quantization, direct slice
reads and combined padding improve both current DiT row shapes, with exact
Q/scales/output gates and an independent-process confirmation.

Execution completed: the first two-T4 screen, exact active Video microchain,
and warmed Video confirmation are documented in `TP2_OPERATOR_RESULTS.md`.
The initial ROI order below is superseded by the measured priorities there and
the newer `TP2_OPERATOR_PASS2.md`.

## Verified targets

The CPU/meta audit instantiates both actual rank-local TE/DiT schemas and
imports the active Video partition function. The manifest records19 logical
GEMM entries,17 distinct MNK triples, and TP source SHA256 identities.

| Component | Per-rank attention | Contract |
|---|---|---|
| TE | Q `[1,32,353,128]`; KV `[1,4,353,128]` | Causal GQA ratio8, Group32 dense, FP32 boundary |
| DiT | Q/K/V `[1,28,6814,128]` | Group32 dense/drop/scalar, protected prefix874, FP32 boundary |
| Video rank0 | Q `[1,32,1024,64]`; KV `[1,32,1797,64]` | Group32 dense, global keys/centering, FP16 boundary |
| Video rank1 | Q `[1,32,773,64]`; KV `[1,32,1797,64]` | Queries1024:1797, same global KV, FP16 boundary |

Video partitions queries and replicates decoder weights; the legacy16-head
decoder is not the active service path. Keep the current1024/773 split as the
baseline. A balanced split is a separate topology experiment.

MNK denotes `[M,K] @ [N,K].T`:

| Component | Projection | Local MNK |
|---|---|---|
| TE | Q; K/V | `(353,4096,5120)`; `(353,512,5120)` |
| TE | O | `(353,5120,4096)` |
| TE | Gate/Up; Down | `(353,12800,5120)`; `(353,5120,12800)` |
| DiT | packed QKV; O | `(6814,10752,5376)`; `(6814,5376,3584)` |
| DiT | packed Gate/Up; FC2 | `(6814,14336,5376)`; `(6814,5376,7168)` |
| Video | QKV; O | `(M,6144,2048)`; `(M,2048,2048)` |
| Video | W1; W2 | `(M,16384,2048)`; `(M,2048,8192)` |

Video M is1024 or773. Packed DiT projections shard each component separately.
FC2's raw local input is `[gate7168|up7168]`, reduced to7168 by activation.
TE Gate/Up activation is materialized before Down.

## ROI and fusion sequence

1. **Shared attention preparation/core.** Avoid GQA KV replication and compute
   only the local Video queries. Compare eager versus fused Group32 quantizers,
   then group-at-a-time versus cached Q/K loads. Preserve leaf64, route order,
   global key centering and FP16 P rounding before trying query tiling or direct
   BND stores. Drop and scalar pooling remain separate choices.
2. **Row-linear preparation/communication.** Profile global-width zero padding,
   rotation, sentinel construction, quantization, MAX, local INT8 GEMM, INT32
   SUM, and dequantization. The first proposed fusion is quantization with an
   externally supplied global maximum, eliminating sentinel buffers/copies.
   FP32 rotation retains the global-width GEMM geometry. Local scales or
   dequantization before INT32 SUM change the operator.
3. **Shared column preparation.** TE Q/K/V share an input; Gate/Up share another.
   Reuse identical rotated/quantized activations where current dispatch repeats
   them. DiT already packs QKV and Gate/Up. Video already requests norm/input-
   activation/residual fusions: verify actual backend dispatch before claiming
   a new fusion.
4. **Shape-specific GEMM dispatch.** Measure current native cuBLAS INT32 and
   complete linear APIs before implementing another SM75 tile/TileLang GEMM.
   Current TP already removes transposed-weight copies and M padding. The old
   PyTorch integer GEMM is an equivalence oracle, not the production baseline.
5. **Epilogues/exchanges.** Fuse post-SUM dequant/cast/residual only where measured
   materializations warrant it. Profile Video's packing/KV exchange separately;
   pre-exchanged attention timings do not include that communication.

Attention producer fusion must use local DiT inner width3584 and packed token
stride10752. Q RMSNorm/RoPE can potentially emit Group32 integers/scales directly;
K still needs the global-key centering stage. Preserve the existing FP32 norm/
RoPE arithmetic followed by the FP16 attention boundary. A direct BND output
store must round through FP16 before widening TE/DiT outputs to FP32.

Quantizing Video's K before KV exchange is a later communication experiment,
not a free fusion: centering still needs all29 key-leaf centroids, and distributed
reduction can change rounding. With D64/Group32, each K token has64 INT8 bytes
and8 FP32-scale bytes instead of128 FP16 bytes; V remains128bytes. The theoretical
payload reduction is21.875%, before extra collectives, packing and padding.
Only exact operand/accuracy gates plus a complete exchange-attention microchain
can establish whether that is worthwhile on the current PCIe/NCCL topology.

Each DiT row projection reduces146,530,176bytes (`6814*5376*4`) of INT32 output
per rank, versus27,256bytes for row MAX. These are tensor sizes, not measured
wire traffic or bandwidth. A faster local GEMM can have limited impact on the
whole row operator. Video's padded KV exchange has an8MiB input and16MiB
all-gather output buffer per rank, followed by concatenation.

## Initial batch and gates

The private fixture dataset contains real TE layer0, DiT layer25/call9 and
Video layer0/call0 QKV from the fixed Tokyo workload. Dataset registration reused
completed Kaggle blobs: zero new QKV bytes uploaded. Workers verify fileSHA256.
Captures originated on5060Ti; timing is guarded to physical T4SM75. This initial
screen is not a distribution-wide quality gate.

`patches/tilelang_grouped_tp.py` adds opt-in dense/drop/scalar operators.
Production dispatch is unchanged pending GPU conformance and performance gates.
CPU tests pass GQA mapping, ragged leaves, query partitions/global centering,
causal offsets, drop and scalar pooling. All six load/mode configurations also
pass static SM75 CUDA lowering; that is compilation evidence, not GPU validation.

Kaggle job `chinokiki9/h3-tp2-operator-screen`, v1, has a20-minute provider cap
and1140-second internal deadline. Both ranks execute simultaneously. It compares
native SDPA and three preparation/load configurations per allowed attention
mode, recording warm-up, repeated GPU-event/wall samples, first-call/JIT cost,
operand identity, per-head implementation/native error, route density and
density1 equivalence. GEMM fixtures use synthetic inputs at source-verified
shapes: they measure performance and integer-core equivalence, not real linear
quantization quality. Captured BHND strides are replayed, with DiT V's local
packed-QKV stride reconstructed from source; other producer-stride fusions
need a later capture at TP hooks.

Nsight Systems is attempted for successful operator families within the same
budget. NVTX separates preparation, routing, fused attention, output layout,
rotation, quantization, GEMM, MAX/SUM and dequantization. QK/softmax/PV are inside
one kernel: Systems cannot establish their instruction-level cycles, memory
stalls or occupancy. Compute previously returned `ERR_NVGPUCTRPERM`; this batch
does not spend quota repeating that permission failure. Missing evidence must
be reported explicitly.

Distributed timers require fixed identical collective call counts. A time-based
warm-up loop can advance ranks differently; a corrected timer is prepared for
subsequent batches. Retain v1 collective timings only if warm-up counts and
operation ordering agree between ranks. Timeouts/mismatches invalidate the
affected family.

CPU reproduction: `benchmarks/audit_tp2_operator_shapes.py`,
`benchmarks/test_grouped_tp_reference.py`, and
`benchmarks/lower_grouped_tp_sm75.py` (static lowering only).
Evidence root: `D:/attention_eval/operator_optimization_tp2_20261016`.
