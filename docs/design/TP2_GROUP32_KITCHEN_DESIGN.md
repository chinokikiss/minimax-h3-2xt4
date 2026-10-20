# What Group32 can borrow from Kitchen INT8 attention on T4

2026-10-17. Source/design analysis only: no new GPU session or full H3 run.
The comparison uses our current operator and the pinned Kitchen0.2.37 source
archive from the completed experiments. It does not assume a newer package has
the same implementation.

Implementation continuation: `TP2_KITCHEN_GROUP32_RESULTS.md` now records two
completed, source-identical T4 runs, runtime gates and Nsight Systems evidence.
The remaining recommendations below are hypotheses, not measured speedups.

Yes: reuse Kitchen's implementation techniques while retaining Group32 Q/K,
FP32 scales/softmax state and FP16 P/V. This can narrow the performance gap.
Equal speed and higher actual accuracy are not established outcomes, however.
Kitchen also saves work through a different numerical contract.

## The structural difference

Our Q/K scale belongs to one token and one32-channel group. A reconstructed
logit is

\[
L_{ij}=D^{-1/2}\sum_{g=0}^{D/32-1}
 s^Q_{ig}s^K_{jg}\langle Q^8_{ig},K^8_{jg}\rangle.
\]

At D64/128 there are2/4 independently scaled integer partial dots per logit.
Because the scale product changes with group, they cannot be replaced by one
INT32 head dot and one final scale without changing the quantization contract.
This is **not** a claim of2/4 times as many INT8 MMA instructions: the total
integer reduction length remains D. The extra work is per-group accumulator
handling, integer-to-float conversion, scale application, and the scheduling/
storage needed around those partials.

Kitchen's per-thread Q/K scale layout covers the channel reduction and permits
scaling completed integer QK fragments. It additionally uses signed INT8 V and
unsigned INT8 probabilities for integer PV, with FP32 softmax/output state.
Our current P/V operands are FP16. Both implementations already use Tensor Cores,
online softmax, exp2-domain updates and a fused attention main kernel; simply
adding those features again is not an optimization opportunity.

Kitchen also has rotated Q/K quantization and data-dependent key-anchor
centering. Those can affect accuracy as well as performance. Its source is not
just a coarser version of our quantizer. Changing centering, rotation, rounding,
scales or PV is a new numerical variant, not a precision-preserving kernel port.

## Remaining priorities after the completed T4 pass

The measured starting point is TE fused preparation, DiT dense BQ64/Q reuse/
FullRow/256 threads, and Video BQ64 plus fused preparation. They improve paired
T4 API latency by roughly13–14%,6–8% and3–4%, respectively. The original TE
repeat has an unstable baseline and is not used to promise45% savings.
Video remains1.68–1.77 times the matched Kitchen latency. No remaining idea
below has a validated speedup estimate.

1. **Warp-local score/softmax layout and register lifetime: highest main-kernel
   research priority.** Kitchen's `update_mdo_*` helpers reduce maxima and
   denominator terms with lane shuffles and use score-storage reuse. Our
   generated BQ64 DiT code has block synchronization and shared workspaces for
   cross-warp max/sum reductions, plus separate `dot`, `scores`, `pv` and `out`
   arrays. Design the MMA fragment layout and row ownership together so fewer
   reductions cross warp boundaries. Investigate temporary lifetimes and
   compatible shared-buffer reuse. Actual T4 launch registers are221/thread;
   this is a reason to inspect allocation, not proof of a measured register
   bottleneck. Native's16-row D128 warp tile is inspiration, not a directly
   interchangeable TileLang CTA parameter. Simply accumulating PV into a
   rescaled running output can change FP32 rounding, and Group32 cannot alias
   all INT32 dot storage with the still-live FP32 grouped score sum.

2. **Group32-aware K/scales staging: high main-kernel research priority.**
   Borrow Kitchen's128-bit vector transfers, explicit shared-memory swizzles,
   MMA-fragment mapping and scale placement. Our code loads K group by group
   and reads KS inside each group epilogue. Test explicit K/scale staging and
   ownership before assuming the compiler removes repeated loads; inspect
   actual SM75 PTX/SASS. If a leaf/group-major physical layout is helpful,
   emit it directly from preparation rather than adding a standalone packing
   pass. Preserve logical QI/KI/scales and group accumulation order. Current
   TileLang already emits vector loads and swizzled `ldmatrix` accesses, so
   merely enabling those features again is not a new optimization. Full-K
   caching is a tradeoff, not an automatic win.

3. **Warp-level Group32 preparation and staged mean: high implementation ROI,
   limited dense-DiT ceiling.** Kitchen's common standalone Q/K paths use
   vector loads/packed INT8 stores and warp-local max reductions. Our fused
   preparer still uses a shared scale array and a generic reduction. Map
   contiguous32-channel groups onto warp/subwarp ownership while retaining
   FP16 input conversion, FP32 scales, rounded division and saturation. Compare
   fused QKV against specialized Q/K and V paths on long DiT: native also
   deliberately separates common Q/K kernels, so fewer launches alone is not
   the objective. In our T4 trace, total prep improves despite the new fused
   kernel itself being longer than the old quantizer pair. Replace remaining
   eager K-centering intermediates with staged kernels only if they preserve
   the existing unweighted mean of leaf centroids and ragged-leaf counts.
   A global K mean is a dependency before K quantization; do not assume an
   ordinary one-launch kernel provides a global synchronization point.

4. **Video epilogue and wrapper work: easiest bounded next test.** Borrow the
   principle of native output conversion in the final store. Our BND store is
   already fused, and `finite_output=True` already implements the FP16
   finite-value behavior, but the completed harness used the default false
   followed by a separate `torch.nan_to_num`. Gate and benchmark enabling that
   existing epilogue for the Video FP16 boundary, removing one extra launch
   and output pass. Test NaN and both infinities explicitly. This is an
   unmeasured caller-level opportunity, not another new main-kernel design.
   For short TE, cached dispatch, reusable stream-safe workspaces or CUDA
   Graphs can reduce host overhead, but those are wrapper/deployment ideas;
   the local graph result is not yet a T4 graph benchmark.

5. **Software staging and interior/tail specialization: second wave.** After
   controlling register/shared-memory demand, test loading the next K/scale
   group around current arithmetic. T4 uses synchronous vector loads, not
   hardware `cp.async`; buffering can lose through resource pressure. Two
   staged64-key leaves must retain two ordered softmax updates. Specialize
   full interior leaves separately from ragged or causal boundaries where
   this removes unnecessary predicates. Masks and query offsets remain exact.

6. **Transfer to sparse only after dense conformance.** The DiT winner was
   dense-only. Measure BQ32/BQ64 layouts separately for frozen drop and scalar
   modes, retaining route decisions,64-query route rows and leaf order.
   Scalar has extra raw-Q/centroid/pooled-update state and may have a different
   register optimum. Current drop already skips K/V work and MMA for omitted
   leaves; do not count that as a new optimization. Compact sorted kept-leaf
   descriptors might remove loop overhead for drop, but scalar must still
   process pooled contributions and cannot use the same shortcut blindly.

For DiT, the selected profiled kernel takes roughly60ms versus about5ms of
preparation on one T4 rank. Even an impossible zero-cost preparation only
saves about8% of that profiled GPU work. Further substantial gains therefore
need main-kernel work. This is a trace-based ceiling for that fixture, not an
end-to-end pipeline forecast or a counter-based diagnosis.

Keep the next matrix small: Video finite epilogue; one warp-quantizer variant;
one staged mean variant; at most two explicit fragment/scale-layout variants.
Use exact current TP2 shapes and original inputs, gate operands/outputs,
then repeat complete APIs and profile finalists on T4. No new GPU submission
or implementation is triggered by this design update.

## Original transferable ideas before the completed pass

| Priority | Kitchen idea | Group32 application | Qualification |
|---|---|---|---|
| First | Vectorized, warp-level preprocessing; Q/K share a launch where appropriate | Fuse existing casts, padding and Group32 quantization; replace eager centroid/mean intermediates with staged kernels; handle packed input strides directly | Preserve the current leaf mean, true ragged counts, FP16 casts, scale floor and rounding; QI/KI/scales must match |
| First | MMA-aware warp tiles and shared-memory swizzles | Lay out Group32 K and its scales for coalesced loads; keep Q/scales resident where measured useful; avoid group-by-group block barriers when warp-local ownership is sufficient | Actual SM75 codegen and barriers must be inspected; cannot blindly remove required synchronization |
| First | Control accumulator lifetime/register layout | Reuse temporary storage after each phase; avoid simultaneously live score/probability/PV intermediates where possible; specialize dense/drop/scalar kernels | Group32 still needs its persistent FP32 score sum plus an integer partial; some native integer-to-float aliasing cannot be copied directly |
| Next | Query tiling with a small warp query tile at D128 | Test query-tile size together with warp count and explicit fragment layout, rather than enlarging BQ at fixed128 threads | Kitchen uses warp query16 at D>=128; our scalar BQ64 winner already reaches255 registers/thread |
| Next | Vectorized staging and shared-buffer reuse | Stage K/V/scale loads for the next leaf; reuse shared storage at nonoverlapping lifetimes | SM75 uses ordinary vector loads; no hardware cp.async benefit |
| Later, separate numerical mode | INT8 P/V | Consider only if a speed/precision tradeoff is explicitly wanted; Group32 QK can coexist with INT8 PV | Changes precision; keeping Group32 QK does not remove P/V quantization error |

The best first target for Video/TE is preprocessing and launch overhead. The
best first target for long DiT is the main kernel's group-scale/MMA layout and
register lifetimes. Dense, drop and scalar need separate decisions; scalar
still performs pooled updates on omitted leaves.

## T4-specific details that matter

1. Kitchen's `cp_async.cuh` enables actual asynchronous copies at SM80 and newer.
   On SM75 its128-bit helpers use synchronous `uint4` loads/stores. T4 does not
   gain Ampere asynchronous-copy machinery by copying the helper names.
2. Kitchen's INT8 MMA wrappers provide an SM75 path using m8n8k16 instructions.
   TileLang must target the same architecture and sensible fragment layouts;
   emitting a newer-architecture intrinsic is not a valid T4 optimization.
3. Its special cached-Q/double-K path in the inspected attention kernel is
   enabled for SM89 only. It is not a measured native-T4 recommendation. Our
   own Group32 Q-only reuse did win on T4 for DiT dense, which is distinct
   evidence for our scale/tiling regime.
4. D128 native launchers choose a16-row warp query tile to limit per-thread
   output accumulators. This is more useful inspiration than copying BQ128
   into our fixed128-thread kernel without changing the warp mapping.
5. Native long-D128 can use a128-key CTA. Our sparse route and online-softmax
   leaves are64 keys. A staged load of two leaves could retain two sequential
   leaf64 updates. Replacing them with a single128-key softmax/PV update changes
   floating-point behavior and cannot merge two sparse decisions.

Do not use BQ128 for current sparse attention: it crosses two independent
64-query route rows. The current sparse guard remains correct. Dense Video can
explore a larger BQ, subject to actual register/shared-memory results and a
complete-API win.

## What the existing measurements establish

The matched active Video microchain confirmation measured Kitchen attention
at1.068ms versus Group32 full-Q/K-cache attention at2.329ms; their complete KV
exchange + attention + output paths were2.773ms and4.071ms. Kitchen had4
attention-API launches versus16 for that Group32 version. These are same-run
comparisons, not predictions of an attainable lower bound.

The later Group32-only screen selected BQ64/direct BND at2.110ms in its
confirmation. It did not remeasure Kitchen in the same allocation, so
2.110/1.068 must not be presented as a fresh controlled speed ratio.

For long DiT dense, the newer trace has roughly61.84ms in attention/output
versus5.83ms in preparation on the first device. Preprocessing fusion alone
cannot close that main-kernel gap. Q-only reuse improved the complete API by
8.6% in confirmation, but raised launch registers197→220 and dynamic shared
bytes24,576→28,160. Scalar BQ64 improved21.6%, reaching255 registers/thread and
46,336 shared bytes. Those results favor deliberate warp/register work over a
blind larger-tile or full-cache sweep.

The new row-linear external-max quantizer is a different operator. Its fast
division matches that native linear contract; it must not be transplanted into
the Group32 attention quantizer, which currently uses its own rounded-division
and saturation rules. Faster arithmetic is not automatically identical
quantization at rounding thresholds.

## Higher precision: a goal, not a blanket conclusion

Keeping per-token/channel32 scales and FP16 P/V gives a reasonable design for
smaller quantization error. It does not prove lower output error than Kitchen:
the basis rotation, centering, probability quantization, input/output casts and
error cancellation differ. On the earlier captured Video boundary, worst-head
SDPA-relative error was0.653% for Kitchen versus0.661% for Group32. That single
fixture supports neither universal superiority nor generation-quality claims.

For the next performance-only variant, retain Group32 scale precision, FP32
softmax/denominator/output accumulation, FP16 P/V rounding, existing centering
and64-key order. Native causal/GQA behavior and Video query offsets must also
remain intact. New arithmetic or reduction order needs implementation gates;
the previous bitwise results cannot be assumed to extend to a rewritten kernel.

INT8 PV, anchor centering, whole-head rotations or direct FP32-input quantization
can be separate dense numerical experiments later. They are not part of the
frozen sparse-policy optimization. Sparse generation distribution/semantics
remain a future full-generation check, not an L1/L2 optimization exercise.

## Small next operator study

Start with staged preparation fusion at the existing winning tile settings,
then a narrow warp-layout shortlist for Video dense and DiT dense. Reject spill-
heavy/resource-heavy variants in CPU compilation before using Kaggle time.
Check prepared QI/KI/scales, sparse routes and same-operator outputs, then measure
the complete API on both ranks with warmed repeated calls. Repeat finalists
in fresh processes and profile only those finalists. Keep current native
Kitchen in the same Video batch to establish the remaining controlled gap.

All target shapes remain those of current TP2: TE Q32/KV4, N353,D128 with causal
GQA; DiT H28,N6814,D128; Video H32,Q1024/773,K1797,D64 with global KV. No full
H3 pipeline is needed for this performance study.

## Source and evidence

Current implementation: `patches/tilelang_grouped_tp_optimized.py`, especially
`prepare`, the per-group loop in `kernel`, and the native BND store.
T4 evidence: `TP2_OPERATOR_RESULTS.md` and `TP2_OPERATOR_PASS2.md`.

Kitchen0.2.37 primary source files inspected:

- `comfy_kitchen/sage_attention.py`
- `backends/cuda/sage_attention/quant_qk_int8.cu`
- `backends/cuda/sage_attention/quant_v_int8.cu`
- `backends/cuda/sage_attention/sage_attn_launcher.cu`
- `backends/cuda/sage_attention/qk_int_sv_i8_cuda.cuh`
- `backends/cuda/sage_attention/mma.cuh` and `cp_async.cuh`

They were extracted from the already retained official v0.2.37 archive, SHA256
`94d06941b4bb8e523008d29157a9c0f0522cd3e37472ad651fc13ffc6a55669c`, under
`D:/attention_eval/operator_optimization_tp2_20261017/kitchen_group32_design`.
No network fetch, package upgrade or GPU work was required for this analysis.
