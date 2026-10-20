# INT8 sparse attention designs for NVIDIA T4 / SM75

Design date: 2026-10-12. Based on the two local source trees named below, plus NVIDIA's architecture documentation. These are proposed implementations and analytical estimates, not measured T4 results. The available local GPU is an RTX 5060 Ti (SM120); it cannot establish SM75 execution or performance.

The recommended first implementation is **A: Sol pooling with INT8 QK and FP16 PV**. Establish a dense SM75 INT8 baseline first, then add sparsity. **B** evaluates INT8 PV independently; **C** targets predictable work budgets; **D** targets video locality; **E** reduces routing and pooled-attention overhead at long sequence lengths.

## 1. What the supplied implementations actually provide

| Local source | Reusable mechanism | Required change for T4 |
|---|---|---|
| [Sol INT8 forward](C:/Users/Administrator/Documents/project/ComfyUI-SolAttn_triton-main/_int8_fwd.py), `_forward_int8_ptr`, line 209 | INT8 token QK with per-token scales; pooled correction for unrouted blocks; optional INT8 PV; forced neighboring/sink blocks | Replace BF16 dots/output conversions with FP16 computation; shrink tiles; avoid descriptor/TMA route; compile for SM75 |
| [Sol preprocessing](C:/Users/Administrator/Documents/project/ComfyUI-SolAttn_triton-main/_fused_prep.py), line 62 | Center K using pooled-key mean; fuse Q quantization with routing-threshold computation | FP32 reductions with FP16 summaries; retain count metadata; implement efficient GPU reductions |
| [Sol summaries](C:/Users/Administrator/Documents/project/ComfyUI-SolAttn_triton-main/_preprocess.py), lines 32, 74, 122 | K **mean**, V **sum**, diagonal covariance threshold | Store V mean and block length to reduce FP16 overflow risk |
| [Sol quantization](C:/Users/Administrator/Documents/project/ComfyUI-SolAttn_triton-main/_fused_quant.py) | Per-token Q/K scales, per-head/channel V scales | Preserve scale algebra; consider block/channel V scales for B |
| [Sol H3 permutation](C:/Users/Administrator/Documents/project/ComfyUI-SolAttn_triton-main/_morton_h3.py), line 110 | Permutes video tokens and associated position information, restores output ordering | Preserve multimodal boundaries; use geometric metadata rather than assuming consecutive Morton blocks are all spatial neighbors |
| [Sage Triton forward](C:/Users/Administrator/Documents/project/SageAttention-main/sageattention/triton/attn_qk_int8_per_block.py), lines 22, 130 | Dense tiled online softmax, INT8 QK, FP16 PV, grouped-query head indexing | Replace dense traversal with block lists; reduce its 128x64 tile; use FP32 PV accumulation for the initial accuracy baseline |
| [Sage Triton quantizer](C:/Users/Administrator/Documents/project/SageAttention-main/sageattention/triton/quant_per_block.py) | One scale per Q/K block; folds softmax scale and log2(e) into Q scaling | Align quantization blocks with execution tiles or explicitly map scales; guard zero scales and padding |
| [Sage dispatcher](C:/Users/Administrator/Documents/project/SageAttention-main/sageattention/core.py), line 143; [build](C:/Users/Administrator/Documents/project/SageAttention-main/setup.py), line 48 | Architecture-specific backend selection | This checkout has no SM75 dispatch branch; CUDA build targets begin at SM80 |
| [Sage CUDA MMA helpers](C:/Users/Administrator/Documents/project/SageAttention-main/csrc/mma.cuh), lines 133, 329 | Tensor Core kernel structure | Its FP16 m16n8k16 / INT8 m16n8k32 instructions and `cp_async.cuh` are unsuitable for direct SM75 reuse |

The Sol wrapper currently requires BF16 and head dimension 128 and rejects masks/cross-attention (`__init__.py`, line 170). Adding FP16 only in the inner kernel would therefore be insufficient: wrapper eligibility and allocations must change too.

Two accuracy distinctions matter:

- Sol's “exact” branch means token-level evaluation of selected blocks. With quantization, it is still approximate relative to dense FP16 attention.
- Subtracting a single shared K vector leaves softmax unchanged in real arithmetic. The source's comment about bit-identical BF16 routing is stronger than this mathematical identity: reductions, casts, quantization and threshold rounding can change routes.

The Sol smoothing vector is an **unweighted mean of block centroids**, including a potentially shorter last block. It differs from the token-weighted K mean for ragged sequences, but either is a valid shared shift. Preserve the block mean for a faithful routing port; record a token-weighted variant as a separate numerical experiment.

## 2. Shared SM75 execution foundation

Turing provides INT8 Tensor Core multiplication with INT32 accumulation, FP16 Tensor Core multiplication, 64 KiB shared memory per SM, 65,536 32-bit registers per SM and at most 32 resident warps. Static shared memory is limited to 48 KiB; larger dynamic allocations need explicit opt-in. These limits favor small working sets and careful register accounting. [NVIDIA Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)

Use legal SM75 instructions:

```text
QK:  mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32
PV:  mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

SM75 supports the corresponding saturating INT8 form too. The larger INT8 m16n8k32 form requires SM80. `cp.async` requires SM80; BF16 Tensor Core MMA also requires SM80. TMA, WGMMA, FP8 and hardware 2:4 sparse MMA are outside this design. Sparsity here comes from skipping attention tiles in software. [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), [CUTLASS SM75 MMA definitions](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm75.h)

Use synchronous global loads into registers/shared memory, explicit warp/block synchronization and shared-memory layouts compatible with the chosen MMA fragments. Register prefetch and double buffering can be evaluated later; increasing Triton `num_stages` does not create SM80 asynchronous-copy capability.

### Data types and quantization

Initial API: inference-only FP16 Q/K/V, D=64 or 128; optional conversion from BF16 outside the kernel, charged to runtime. Support strided BHND or BTHD inputs with contiguous feature dimension. Quantization can pack directly into head-major block storage to avoid a separate full-size transpose. Quantize Q/K **after** Q/K normalization and RoPE.

For query token i and key token j:

\[
K'_j=K_j-\mu_K,\quad
s^Q_i=\max(\|Q_i\|_\infty/127,\epsilon),\quad
s^K_j=\max(\|K'_j\|_\infty/127,\epsilon),
\]
\[
\widehat Q_i=\operatorname{clip}(\operatorname{round}(Q_i/s^Q_i),-127,127),\quad
\widehat K_j=\operatorname{clip}(\operatorname{round}(K'_j/s^K_j),-127,127),
\]
\[
z_{ij}=\left(\widehat Q_i\widehat K_j^T\right)_{\mathrm{INT32}}
s^Q_i s^K_j /\sqrt D.
\]

Use per-token scales initially (Sol's choice). Compare Sage-style block scales separately; their smaller metadata and scalar dequantization can help, but outliers share a larger scale. Do not use per-feature K scales unless the dot is split into separately scaled reductions: those scales cannot be pulled outside a single QK MMA.

Compute softmax maxima, exponentials, denominators, dequantization and output accumulation in FP32. With `exp2`, convert logits by log2(e) exactly once. Sage's quantizer already folds this conversion into Q scaling; combining that quantizer with Sol's forward scaling unchanged would double-scale the scores.

The worst absolute QK dot at D=128 and signed magnitude 127 is 128*127^2=2,064,512, within INT32. Padding must quantize to zero, contribute no moments/counts, and be masked out of softmax. Use 64-bit address arithmetic where tensor offsets can exceed signed 32-bit range.

### Tiles and memory targets

Keep a **routing leaf of 64 tokens** initially. Execute one leaf as two Q microtiles of 32 rows; K tiles can be 32 or 64. A shared 64-token route must be computed from the full leaf centroid, not independently from each 32-row half, when comparing against Sol semantics.

| Candidate, D=128 | Warps/CTA | Minimal mixed QK/PV staging | Minimal dual-INT8 staging | FP32 output elements/thread at 128 threads |
|---|---:|---:|---:|---:|
| M32, N32 | 4 | 18 KiB | 13 KiB | 32 |
| M32, N64 | 4 | 32 KiB | 22 KiB | 32 |
| M64, N64 | 4 or 8 | 40 KiB | 28 KiB | 64 at 4 warps |

Mixed estimate: Q8 + K8 + V16 + P16. Dual estimate: Q8 + K8 + V8 + P8. These are buffer estimates, not compiler resource reports; they exclude layout padding, route summaries, scales, and extra staging. Some buffers can share lifetimes. Use M32/N32 as the conservative starting point, M32/N64 as the main performance candidate.

Aim for roughly two resident CTAs where possible; inspect actual registers, shared memory and spills rather than enforcing occupancy alone. For M32/N64, the mixed estimate already consumes 32 KiB, so any additional shared storage can prevent two CTAs fitting. Splitting V across separate CTAs repeats QK, routing and softmax work; avoid it until a full-D tile demonstrably spills. For D=64, try M64/N64 too. Autotune on D, tile dimensions, sparsity bucket and dtype, not only sequence length.

## 3. Scheme A: Sol pooling + INT8 QK / FP16 PV

**Use:** first T4 port; long video self-attention where preserving a contribution from every key block is desirable.

For each 64-token leaf b, form centered K mean c_b, V mean v_b and valid token count n_b using FP32 reductions. Retain FP16 summaries, plus FP32 routing statistics. For query leaf a, form qbar_a from all its valid queries. Route b to token-level attention when

\[
r_{ab}=qbar_a^T c_b/\sqrt D > t_a,
\quad t_a=m_a+\tau_h\sigma_a,
\]

or b is a mandatory block. As in Sol, estimate sigma using the diagonal covariance of the K centroids. If c_b is centered by their mean, m_a is near zero; retain the measured mean or explicitly account for rounding instead of assuming it is bitwise zero.

Mandatory blocks initially reproduce Sol's leaf a and neighbors a±1 and configured conditioning KV blocks. Configured conditioning **query** blocks evaluate all allowed keys at token level. A boundary leaf containing both conditioning and video tokens should be conservatively protected or split; round-out protection costs extra work but preserves the conditioning prefix.

For selected blocks, evaluate INT8 QK, apply scale factors, softmax and FP16 PV with FP32 accumulation. For each unselected block, every query token i contributes a pooled logit

\[
\widetilde z_{ib}=Q_i^T c_b/\sqrt D + \log n_b
\]

with value v_b. Merge selected tokens and pooled entries into **one** online softmax. In base-2 form, use log2(n_b). This is equivalent in real arithmetic to the source's `exp(score)*V_sum` numerator and `exp(score)*length` denominator. Mean storage avoids multiplying a large FP16 V sum by a small coefficient; FP16 casting still requires validation.

Do not normalize token and pooled branches separately and average them. Selected blocks must be removed from the pooled branch to avoid double counting.

Implementation options:

1. A simple pipeline computes centroids/thresholds and a GPU route bitset, then evaluates token-level and pooled entries. A bitset or sorted block list is shared by both Q microtiles.
2. A later fused variant computes routes and pooled scores in small summary groups (16 or 32), then visits selected blocks. Preserve the full-leaf centroid for routing. Compare its lower metadata traffic against longer live register lifetimes.

Starting sweep: tau={0,0.5,1,1.5,2}, recording actual selected density per head/layer/step. Tau is a threshold parameter, **not** a requested density and not a normal-distribution guarantee. Do not carry the UI's empirical density claims from other GPUs into T4 planning.

**Cost:** token GEMMs roughly rho*N^2*D; pooled QK/PV roughly N^2*D/64; preprocessing linear in N*D. It remains asymptotically quadratic when all summaries are evaluated per token. It is not selected-block-only attention.

**Risk:** a K centroid hides rare high-score keys and variation across queries; mandatory blocks and dense handling for sensitive layers reduce but do not eliminate that error. FP16 pooling differs from the supplied BF16 implementation.

## 4. Scheme B: Sol pooling + INT8 QK / INT8 PV

**Use:** extension of A when PV cost or repeated V traffic limits measured T4 performance. Keep the pooled branch FP16; its small summaries rarely justify more quantization metadata.

For each selected K block, exponentiate against its own row maximum h_i:

\[
p_{ij}=\exp(z_{ij}-h_i),\quad
\widehat p_{ij}=\operatorname{round}(127p_{ij}),\quad
\beta_i=\exp(h_i-m_i^{new}).
\]

Quantize V with per-block/channel scale s^V_{b,d}; evaluate P8*V8 with the same SM75 INT8 MMA family. Restore the block numerator as

\[
U_{i,d}^{block}=\frac{\beta_i s^V_{b,d}}{127}
\sum_{j\in b}\widehat p_{ij}\widehat V_{j,d}.
\]

Update the denominator with the **FP32, unquantized** p values:

\[
l_i^{new}=\alpha_i l_i+\beta_i\sum_{j\in b}p_{ij},\quad
U_i^{new}=\alpha_i U_i+U_i^{block},\quad
\alpha_i=\exp(m_i-m_i^{new}).
\]

Do not mix a denominator based on rounded probabilities with a numerator correction based on unrounded probabilities. The two must follow a specified numerical contract.

The source uses one global V scale per head/channel. Proposed block/channel scaling improves resolution in blocks with smaller magnitude and removes a global V-maximum reduction, at the cost of more scales and repeated dequantization. A per-token V scale cannot factor out of PV's reduction axis. Compare both scale strategies under the same routes.

Optional global V centering: V'=V-mu_V, quantize V', use centered V summaries throughout and restore mu_V after dividing by l. Because the chosen attention distribution is normalized, this algebra also holds for pooled or dropped-block variants. It requires a global mean reduction and is an independent quality/performance option. Block-specific centering would need a separate weighted mean contribution for every block; it cannot be restored by one final add.

For a 64-key tile, signed-127 rounding error in p is at most 1/254 per element, so |delta U_d| <= sum_j |V_jd|/254 before normalization and V quantization. Low probabilities can round to zero even though they remain in the denominator. Block-local maxima improve dynamic range but cannot remove this error.

A CUDA variant may use U8 probabilities scaled by 255 with S8 V: SM75 CUTLASS exposes U8*S8 MMA. Treat this as a separately verified implementation; do not assume Triton will generate mixed-signedness tensor instructions. Splitting probability into a second quantized residual is another accuracy option, costing a second PV dot.

**Cost:** selected-branch operands use INT8 for both GEMMs, but softmax, conversions, staging/layout changes and preprocessing remain. INT8 PV can lose to FP16 PV at high sparsity. No throughput multiplier is assumed.

**Risk:** compounds Q/K quantization, P/V quantization and pooling error. Keep A available per head/layer where B fails quality acceptance.

## 5. Scheme C: Budgeted content routing + selected-block-only INT8 attention

**Use:** predictable block work and removal of A's per-token scan of all pooled summaries. This deliberately drops unselected keys and renormalizes over retained keys.

Compute all qbar_a^T c_b scores in a batched small router kernel. Seed the retained set with mandatory blocks. Add the highest-scoring remaining leaves until either a selected count budget is reached or a cumulative **proxy** mass target is reached:

\[
w_{ab}^{proxy}\propto n_b\exp(qbar_a^T c_b/\sqrt D).
\]

Start with budget fractions {1/16,1/8,1/4}, or proxy coverage {0.95,0.98,0.995}; these are experiment settings, not quality guarantees. Mandatory blocks consume budget. If they exceed it, expand the budget rather than evicting them. Per-head/layer budgets should come from calibration; cap violations and dense conditioning queries need separate counters.

Write unique, sorted GPU block lists with counts. Process only listed blocks through A's INT8 QK / FP16 PV kernel. Use list-count buckets to reduce CTA runtime variation. A warp ballot plus prefix compaction is preferable to repeatedly taking the minimum of a route vector as in the supplied Sol loop. Avoid CPU synchronization, per-call host sorting and score matrices at token resolution.

Routing is shared per 64-token leaf. For query leaves with heterogeneous attention, evaluate 2-4 query subcentroids and use the union of their selected lists. The union may exceed the original budget; recalibrate against its actual size. Threshold selection with count buckets can also be cheaper than full top-k; benchmark router time.

**Cost:** full centroid router O((N/64)^2*D), selected token GEMMs O(rho*N^2*D). No pooled per-token PV. The centroid matrix is much smaller than a token score matrix but can still grow quadratically; stream selection rather than retaining it at very long lengths.

**Accuracy contract:** centroid proxy coverage is not actual attention coverage. On representative query rows, measure true omitted mass with a tiled dense reference. If actual omitted mass is epsilon and each V has norm <=Vmax, dropping and renormalizing gives ||O_dense-O_kept|| <=2*epsilon*Vmax for otherwise identical logits/values. This bound does not certify centroid selection and does not include quantization error.

**Risk:** isolated high-score tokens, head diversity and query diversity can defeat centroid ranking. Use larger budgets, subcentroids or dense handling where needed. If remote pooled correction is added, identify that configuration as a hybrid with A/E and account for its cost.

## 6. Scheme D: Geometric local/global INT8 block attention

**Use:** video heads whose attention locality has been verified on this model. This is the smallest routing overhead option.

Partition video tokens into spatial/temporal bricks, pack those bricks into 64-token leaves and preserve explicit (t,y,x) metadata. A Morton order can improve packing, following the supplied H3 hooks. Build a deterministic union of:

- spatial neighboring bricks within a tunable radius;
- corresponding spatial regions in a short temporal window;
- conditioning text/reference/audio blocks designated as global by model layout;
- a small calibrated set of long-range anchors, if useful.

For example, start with 4x4x4-token bricks only where the patch grid supports them. Sweep temporal radius {1,2}, spatial brick radius {1,2}; choose actual block counts from the grid rather than declaring a universal density. An alternative frame-wise 8x8 brick avoids crossing temporal boundaries. Keep modality boundaries explicit and deduplicate all overlapping regions.

Run INT8 QK / FP16 PV on this fixed block list; use B's PV only where independently accepted. Conditioning query leaves can remain dense. Consecutive Morton leaves are **not** a complete geometric neighborhood; use coordinate tests to construct the lists. Permute position metadata/masks consistently and restore output order. Do not apply this permutation to causal text without rebuilding its mask semantics.

**Cost:** for local window W tokens and global set G tokens, attention is O(N*(W+G)*D), plus any dense conditioning queries. It is approximately linear only if W and G remain bounded as N grows. Precomputed geometry lists can persist across denoising steps; Q/K/V quantization and logits still refresh every call.

**Risk:** fast motion, scene changes, spatially distant objects and cross-modal interactions can require distant keys. Restrict this scheme to qualified heads/layers; retain A/C for others. Adding far-region pooled summaries can improve coverage but changes the cost and should use E's partition rules.

## 7. Scheme E: Hierarchical content routing with pooled remote regions

**Use:** N>=16K-32K when A's scan of every leaf summary becomes a substantial fraction of attention time. This retains a remote contribution while evaluating fewer per-token summaries.

Group g=8 adjacent, spatially coherent 64-token leaves into a 512-token parent. Store parent K/V means and count using token-count-weighted child reductions. Also store a dispersion statistic or 2-4 parent subcentroids for routing, to reduce cancellation inside the parent.

For each query leaf:

1. Score all parents with its centroid/subcentroids and always open parents containing mandatory leaves.
2. Open a calibrated number/fraction of promising parents; score their children.
3. Select children for token-level INT8 attention using A's threshold or C's budget.
4. Unselected children in opened parents contribute individual pooled entries.
5. Unopened parents contribute one pooled entry each, with their own log(count).

Each key token must belong to exactly one terminal entry: either a token-level leaf, a pooled child, or a pooled parent. Do not add a full parent summary beside selected children. The simplest implementation always replaces an opened parent with its full child partition; partial residual-parent summaries require correct residual counts/means and a separate validation path.

Merge all terminal entries with the same online softmax as A. Parent pooling is a coarser approximation; it is not algebraically equivalent to summing pooled child softmax contributions. Dispersion and content probes determine which parents need opening, but do not certify quality.

For P=N/(64*g) parents and L_open inspected leaves, routing costs about O((N/64)*(P+L_open)*D). Per-token pooled work scales with P plus unselected children of opened parents. At N=32768, g=8 gives 64 parents rather than 512 leaves; opening 8 parents adds 64 child candidates. Actual savings depend on final selected count and terminal partition size. Fixed opening counts reduce growth, but scanning all parents still has a quadratic component.

**Risk:** a salient child can be hidden by a parent centroid. Subcentroids/dispersion triggers, mandatory opening and denser treatment of sensitive heads are necessary experiments. Implement after A/C because they supply the validated token kernel and routing tools.

## 8. Shared integration and numerical rules

### Online softmax and masks

Initialize m=-infinity, l=0, U=0. For a tile with maximum h, use m_new=max(m,h), rescale old state and add current contributions. Explicitly skip tiles with no allowed keys. Define an all-masked row to return zero; prevent exp(-infinity-(-infinity)) from generating NaNs. Invalid query rows never store output.

A fully valid leaf can be pooled. A partially allowed leaf must be evaluated token by token or summarized only over its allowed subset. In causal attention, future leaves are excluded, the diagonal leaf is token-masked, and summaries never include future keys. Arbitrary per-query masks/additive biases generally prevent sharing one pooled summary; initially route those calls to a tested masked dense backend. An SM75 dense INT8 backend may support them later, but its numerical contract must be validated too.

Add an explicit `attn_mask` parameter to the integration API so wrappers checking the function signature see support/fallback correctly. Never silently ignore mask, scale, GQA mapping or custom bias. Keep short sequences and unsupported head dimensions on a proven fallback until their crossover/semantics are established.

K smoothing shifts row log-normalizers: if returning a natural-log LSE, restore Q_i dot mu_K * original softmax_scale. Convert base-2 LSE by ln(2) first. This matters when merging rank-local partitions whose K means differ.

### Suggested implementation modules

```text
sm75_sparse_attention/
  dispatch.py       # SM75, dtype/layout/mask checks; backend/fallback counters
  prepare.py        # FP32 summaries; K shift; Q/K quantize/pack; optional V8
  routes.py         # Sol thresholds, budget lists, geometry, hierarchy
  triton_forward.py# small-tile prototype, inspected generated PTX
  cuda_forward.cu  # optional explicit SM75 MMA/staging implementation
  merge.cu         # optional split-K partial-state merge
```

Start with a Triton prototype **only if** the installed compiler accepts SM75 and emits the intended integer Tensor Core instructions. Otherwise implement CUDA/CUTLASS using SM75 helpers. Successful Python dispatch is not proof the intended kernel ran. Keep compilation/imports independent of Sage's unsupported SM80+ extension dependencies.

First establish dense QK8/PV16 with the same tile sizes, Q/K scale policy and FP32 accumulation as the sparse backend. A's all-leaves-selected case must reduce to this baseline. Keep separate counters for backend entered, compiled SM target, dtype, route density, dense fallback and reason.

### Route reuse across denoising steps

Cache route lists only after measuring overlap at the same layer/head across adjacent steps. Refresh Q/K/V, scale values and softmax scores every call. Begin by refreshing routes every step; later try a refresh interval of two with sampled route probes and forced refresh at shape/layout changes or poor overlap. A cached mask does not make old pooled K/V summaries valid. Reuse is an experimental scheduling option for C/E, not a claim that attention is stationary.

### Two T4s

The workspace's TP DiT uses 56 heads, 28 heads/rank, D=128 (`patches/tp_dit.py`, line 129). In this arrangement each rank has complete sequence keys for its local heads, so routing/attention is rank-local. No attention route all-gather is needed; existing output-projection communication remains. Sparse attention does not remove the surrounding linear-layer/residency costs.

The current `SPDiTBlockINT8` gathers native activations before invoking TP attention. Do not infer distributed K/V sparsity or communication savings from that wrapper's name.

For a future genuinely sequence-sharded backend, exchange compact summaries for global routing, then request only selected K/V blocks or compute partial attention at the owning ranks and merge (m,l,U). Use a consistent K shift or correct row shifts before merging; denominators and maxima must reflect a global normalization. T4 PCIe transfer overhead can outweigh sparse compute savings. This requires a separate measured design.

Split-K is initially unnecessary: at N=8192/M32 and H=28, there are already 7168 query CTAs per batch. If a few dense/global query leaves create a runtime tail, split only those leaves and merge FP32 states:

\[
m=\max_r m_r,\quad l=\sum_r e^{m_r-m}l_r,\quad
U=\sum_r e^{m_r-m}U_r,\quad O=U/l.
\]

## 9. Memory and performance model

For B=1, H=28/rank, D=128, input/output FP16:

| Tokens | Leaves of 64 | Original Q/K/V | Added Q8/K8 | Added Q8/K8/V8 | FP32 dense score matrix alone |
|---:|---:|---:|---:|---:|---:|
| 8192 | 128 | 168 MiB | 56 MiB | 84 MiB | 7 GiB |
| 16384 | 256 | 336 MiB | 112 MiB | 168 MiB | 28 GiB |
| 32768 | 512 | 672 MiB | 224 MiB | 336 MiB | 112 GiB |

These are tensor-size calculations, not peak memory measurements. Quantized buffers normally coexist with inputs; scales, summaries, routes, output and compiler workspaces add memory. Scheme B does not automatically reduce peak memory if FP16 V is still owned by the caller. Avoid the full token score matrix in every production scheme; this matrix is also why an eager dense-reference implementation is unsuitable at these sizes.

For A with a 64-token leaf, a rough dense-equivalent GEMM work fraction is rho+1/64. At rho=1/8 this is 0.140625, corresponding to about 7.1x fewer multiply-accumulate operations. It is **not** a wall-time speedup: INT8 and FP16 dots have different costs, and routing, exponentials, scales, memory access and kernel launches remain. Scheme C removes the pooled term but adds selection/list overhead. Scheme D can remove content-routing work. Scheme E reduces the pooled term at the expense of another approximation and hierarchical routing.

Use the measured model

```text
T_call = T_preprocess + T_route + T_selected_attention
       + T_pooled_attention + T_merge + T_layout + T_launch
```

with absent terms set to zero. Determine the short-sequence/high-density fallback crossover from complete calls, not kernel-only plots. High selected density can justify dense INT8 even while tau is nonzero. Do not promise generation speedup from an attention-only gain; measure the attention share of total runtime and apply Amdahl's law.

## 10. Qualification plan and implementation order

1. **Dense SM75 baseline:** verify S8 Tensor Core PTX/SASS, SM75 target, ragged sequences, zero tensors, strides, D=64/128 and FP32 online softmax. Compare against a tiled dense FP16/FP32 reference; report fallback execution explicitly.
2. **A:** all-selected parity with the matching dense INT8 baseline; isolated pooled-mean/count correctness; verify no overlap/double counting; preserve conditioning queries/keys. Compare source-like FP16 Sol with its INT8 variant under frozen routes before attributing differences to sparsity.
3. **B:** freeze routes and compare QK8/PV16 versus QK8/PV8 to isolate P/V error. Include global V scales, block/channel scales and optional V centering.
4. **C and D:** measure dropped true attention mass on sampled queries, query-subcentroid unions, conditioning behavior and task outputs. Dense-reference mass can be obtained with streaming blocks; do not allocate N^2 scores.
5. **E and optional route reuse:** compare against A at matched selected counts and report both selected leaf density and the number of terminal pooled entries.

Physical T4 sweep: B={1,2}, D={64,128}, H including 28, N={1024,4096,8192,16384,32768} plus ragged N={65,513,8193}. Start with inference dropout=0; reject unsupported training/dropout contracts. Include outlier inputs, nearly uniform attention, peaked attention, constant/zero K/V, partially protected leaves, masks and fully masked rows. Use representative layer/step Q/K/V captured from the actual H3 workload.

For every scheme, separately record relative output L2, max absolute error, cosine similarity, row-normalizer/LSE behavior, per-head/layer error, selected density and task-level video/image behavior. Uniform/peaked attention both matter: a sparse method that passes only one is poorly characterized. Long-run generation must check semantic retention, fine detail, motion continuity and audio/video behavior using fixed seeds/inputs. Per-layer probes like the supplied Sol calibration hook can guide dense exemptions, but full generation is needed to expose accumulated error.

Benchmark after compilation and warmup using CUDA events, repeated calls and median/p95 latency. Report preprocessing, routing, forward and end-to-end latency; peak allocated memory; registers/spills; selected-count imbalance; backend/fallback counts. Use Compute Sanitizer for memory/race/synchronization checks on the SM75 implementation. Do not establish acceptance thresholds after seeing favorable results: agree task-level error and quality criteria before selecting the fastest configuration.

**Priority:** dense baseline -> A -> B -> C -> D/E as workload evidence warrants. For this project's first 2xT4 H3 experiment, use A with 64-token leaves, M32/N32 or N64, per-token Q/K scales, FP16 PV/FP32 accumulation, protected conditioning and a per-layer tau sweep. Keep B opt-in until probability/V error and complete-call latency are measured.

## 11. Design arithmetic checks performed

Checked the shared K-shift identity, V-sum versus V-mean/log-count pooling, and the split-partition softmax merge using JavaScript double arithmetic on a deterministic 11-token example partitioned into ragged blocks of lengths 4,4,3. Maximum absolute discrepancies were 5.6e-17, 6.1e-18 and 1.1e-16 respectively. Also recomputed the staging, tensor-memory and INT32 range figures above. These checks validate the algebra and accounting only; they do not test FP16 rounding, quantization quality, compiler lowering or GPU kernel behavior.
