# Group32 audit and optimization

The audited Group32 baseline is patches/tilelang_grouped_qk.py. Grouping is contiguous channels within each token, D/32groups per head; it is not32tokens/warp and is not upstream SageAttention. For D128, scales have[B,H,Npad,4] FP32 layout; QI/KI[B,H,Npad,128] INT8, V FP16.
Q and K are first FP16. K smoothing converts to FP32, computes each64-token leaf centroid using actual ragged counts, then averages leaf centroids equally. A short final leaf receives equal centroid weight. This is not the true token mean. Each token/channel group scale=max(absmax/127,1e-8); division uses Torch scalar reciprocal semantics and normalized values round to nearest-even, clamp[-127,127]. Q is not centered. Padded integer/scales are ignored outside N.
Each32-channel integer dot accumulates in INT32. Its FP32 conversion is multiplied by the query/key group scales and1/sqrt(D), then accumulated in FP32 across groups. Softmax is online FP32/exp2, P is rounded FP16 per key leaf, and PV uses FP16 inputs/FP32 accumulation. Output is FP16; LSE is FP32 with the centering shift when requested. There is no global partial-GEMM materialization in the original kernel.
Preserved baseline: fused Sage32 source unchanged. It uses32-token Q blocks and64-token K blocks, FP32 true token-mean smoothing, half-away rounding and one scaled integer dot over D. That is a different quantization contract.
Existing selected DiT head: legacy Group32 raw2.656ms/API3.320ms versus fused Sage32 raw2.677ms/API2.945ms, but old API boundaries differ. These numbers motivate fused preparation; they do not prove Group32 is slower or quality-safe.
New implementation patches/tilelang_grouped_fused.py preserves the audited Group32 centering/scale/rounding contract. Two TileLang quantizers replace PyTorch per-group chains. Q integers/scales are staged once per query CTA; K integers/scales once per key tile, then group dot-products and scaled accumulation stay inside the attention kernel. Channel groups16/32/64/128 are explicit variants;32/64 are priorities. FP32 scale storage is retained for faithful comparison.
Correctness gate: actual-T4 valid QI/KI equality and scale differences against legacy Group32, plus independent dequantized streaming attention (per-leaf FP16 P). Implementation error and native approximation differences are reported separately; neither is generation acceptance.
Benchmark compares current fused Sage32 and legacy Group32 with standardized original-capture-to-FP16-output boundaries, warm randomized measurements and three independent processes. Compilation/first execution is separated. Systems traces requested. No register pressure, occupancy or Tensor Core utilization claim without counters.
Hardware execution initially encountered Kaggle maximum2GPU batch sessions; another existing TE critical-path job was not cancelled. Version1 compiled kernels but stopped in a ragged reference-checker error: the last V leaf included padded rows while the reference K leaf stopped at valid N. The checker was fixed and CPU regression65/353tokens, causal/noncausal, passed. Version2 hardware results below supersede that failed attempt; no production readiness is claimed.

Collected group_v1\grouped\process0\results.json: running.
| Component | Candidate | Status | Raw median ms | API median ms |
|---|---|---|---:|---:|
| te | fused_group16 | failed | pending | pending |
| te | fused_group32 | failed | pending | pending |

Collected group_v2\grouped\process0\results.json: complete.
| Component | Candidate | Status | Raw median ms | API median ms |
|---|---|---|---:|---:|
| te | fused_group16 | complete | 0.0798719972372055 | 0.5470399856567383 |
| te | fused_group32 | complete | 0.07887999713420868 | 0.5485439896583557 |
| te | fused_group64 | complete | 0.08191999793052673 | 0.5875200033187866 |
| te | fused_group128 | complete | 0.07606399804353714 | 0.8689600229263306 |
| te | legacy_group32 | complete | 0.10598400235176086 | 0.853983998298645 |
| te | fused_sage32 | complete | 0.08054400235414505 | 0.41606399416923523 |
| dit | fused_group16 | complete | 3.336832046508789 | 3.8417599201202393 |
| dit | fused_group32 | complete | 2.885119915008545 | 3.3520960807800293 |
| dit | fused_group64 | complete | 2.7317121028900146 | 3.1831040382385254 |
| dit | fused_group128 | complete | 2.7013120651245117 | 3.1604480743408203 |
| dit | legacy_group32 | complete | 2.6766719818115234 | 3.3812479972839355 |
| dit | fused_sage32 | complete | 2.855776071548462 | 3.21126389503479 |
| video | fused_group64 | complete | 0.14336000382900238 | 0.5736640095710754 |
| video | fused_group16 | complete | 0.155008003115654 | 0.586143970489502 |
| video | fused_group32 | complete | 0.22118400037288666 | 0.5895360112190247 |
| video | legacy_group32 | complete | 0.1632319986820221 | 0.8676159977912903 |
| video | fused_sage32 | complete | 0.15638400614261627 | 0.3798080086708069 |

Collected group_v2\grouped\process1\results.json: complete.
| Component | Candidate | Status | Raw median ms | API median ms |
|---|---|---|---:|---:|
| te | fused_group16 | complete | 0.07974400371313095 | 0.5580480098724365 |
| te | legacy_group32 | complete | 0.10998400300741196 | 0.8847360014915466 |
| te | fused_group64 | complete | 0.07593599706888199 | 0.5806400179862976 |
| te | fused_sage32 | complete | 0.08195199817419052 | 0.3957119882106781 |
| te | fused_group32 | complete | 0.07948800176382065 | 0.5534080266952515 |
| te | fused_group128 | complete | 0.08144000172615051 | 0.5610560178756714 |
| dit | fused_group16 | complete | 3.4734079837799072 | 4.006720066070557 |
| dit | legacy_group32 | complete | 2.7811520099639893 | 3.534559965133667 |
| dit | fused_group64 | complete | 2.8590080738067627 | 3.346560001373291 |
| dit | fused_sage32 | complete | 3.014496088027954 | 3.3259520530700684 |
| dit | fused_group32 | complete | 3.1006720066070557 | 3.555327892303467 |
| dit | fused_group128 | complete | 2.857343912124634 | 3.336319923400879 |
| video | legacy_group32 | complete | 0.16592000424861908 | 0.8744959831237793 |
| video | fused_sage32 | complete | 0.16383999586105347 | 0.36662399768829346 |
| video | fused_group64 | complete | 0.15619200468063354 | 0.6056320071220398 |
| video | fused_group32 | complete | 0.15955199301242828 | 0.5818560123443604 |
| video | fused_group16 | complete | 0.16019199788570404 | 0.5631999969482422 |

Collected group_v2\grouped\process2\results.json: complete.
| Component | Candidate | Status | Raw median ms | API median ms |
|---|---|---|---:|---:|
| te | fused_group128 | complete | 0.06963200122117996 | 0.5406720042228699 |
| te | legacy_group32 | complete | 0.10406400263309479 | 0.884768009185791 |
| te | fused_group32 | complete | 0.07660800218582153 | 0.9661120176315308 |
| te | fused_sage32 | complete | 0.08396799862384796 | 0.37481600046157837 |
| te | fused_group64 | complete | 0.06963200122117996 | 0.546783983707428 |
| te | fused_group16 | complete | 0.07168000191450119 | 0.612384021282196 |
| dit | fused_group128 | complete | 2.7708799839019775 | 3.3123199939727783 |
| dit | legacy_group32 | complete | 2.8034238815307617 | 3.5431039333343506 |
| dit | fused_group32 | complete | 3.074399948120117 | 3.5610880851745605 |
| dit | fused_sage32 | complete | 2.9839038848876953 | 3.2913599014282227 |
| dit | fused_group64 | complete | 2.885983943939209 | 3.3865599632263184 |
| dit | fused_group16 | complete | 3.5903360843658447 | 4.074783802032471 |
| video | fused_group16 | complete | 0.16358399391174316 | 0.5672000050544739 |
| video | legacy_group32 | complete | 0.1674560010433197 | 0.8606719970703125 |
| video | fused_group32 | complete | 0.15961599349975586 | 0.5735359787940979 |
| video | fused_sage32 | complete | 0.16383999586105347 | 0.3749760091304779 |
| video | fused_group64 | complete | 0.1945600062608719 | 0.5743039846420288 |

## Completed version2 controlled comparison

All three independent processes completed51candidate/component cases in total on actual T4. Group16/32/64/128 passed the independent attention-reference check where D permits. Group32 valid QI/KI were bit-identical to legacy on the three native TE/DiT/VAE diagnostic heads; both scale maximum differences were zero. Maximum implementation relative difference is about0.0217%, separate from native quantization approximation error and generation quality.
| Component | Candidate | Median raw ms | Median preparation ms | Median complete API ms |
|---|---|---:|---:|---:|
| te | fused_sage32 | 0.0820 | 0.2625 | 0.3957 |
| te | legacy_group32 | 0.1060 | 0.7332 | 0.8847 |
| te | fused_group32 | 0.0789 | 0.4429 | 0.5534 |
| te | fused_group64 | 0.0759 | 0.4567 | 0.5806 |
| dit | fused_sage32 | 2.9839 | 0.2893 | 3.2914 |
| dit | legacy_group32 | 2.7812 | 0.7096 | 3.5346 |
| dit | fused_group32 | 3.0744 | 0.4506 | 3.5553 |
| dit | fused_group64 | 2.8590 | 0.4670 | 3.3466 |
| video | fused_sage32 | 0.1638 | 0.2365 | 0.3750 |
| video | legacy_group32 | 0.1659 | 0.6799 | 0.8676 |
| video | fused_group32 | 0.1596 | 0.4432 | 0.5819 |
| video | fused_group64 | 0.1562 | 0.4190 | 0.5743 |

The cached-QK Group32 kernel regressed raw DiT latency enough to offset its preparation savings. It is not accepted as a faster replacement. Group64 is close to fused Sage32 on this selected head, but neither performance nor lower diagnostic error establishes a quality benefit. The next real-H56 study adds fused Group32 preparation with the unchanged legacy attention kernel, directly testing whether the new shared-memory strategy was counterproductive.
Actual DiT trace launch attributes: cached Group32 uses210registers/thread and35,328dynamic shared bytes; Group64 uses165registers/thread and34,560bytes; Sage uses199registers/thread and54,272bytes. These resources do not measure achieved occupancy or spills. Group32/64 preparation+casts launch16kernels/API versus9for Sage. Full scale-centroid mean preparation remains a fusion opportunity; no hypothetical speedup is claimed.
Full native56-head job completed: chinokiki9/h3-attention-native-full-shapes, version1. It uses actual full captures at layers0/10/25, step9, without duplicating one head. Its results below supersede any performance-candidate choice based on H1. The private subset registers already completed upload blobs; no new QKV bytes were uploaded. Creation evidence: D:\attention_eval\quality_constrained_20261015\dit_subset_metadata\creation_status.json.
Microbench API boundary remains captured contiguous BHND original dtype to FP16 BHND, with necessary input casts/preparation. Native ComfyUI Attention uses interleaved QKV views and returns FP32 BND; absolute microbench API times are not the native integration boundary. The forthcoming generation profile records actual strides, dtype, output casting and layout overhead; current operator ratios do not substitute for that profile.

## Completed real full-head T4 study

Version1 completed on actual T4:36 Group cases and99 matched-density cases, three independent processes, native layers0/10/25 at denoising step9. Every input is actual[1,56,6814,128]; no head replication. Fifteen warmed measurements per case, randomized candidate order. Tables show medians of the three process medians for each captured layer.

| Layer | Candidate | Raw ms | Preparation ms | Complete API ms | API process-median range ms |
|---:|---|---:|---:|---:|---:|
| 0 | fused_sage32 | 163.049 | 10.295 | 175.513 | 170.023–177.648 |
| 0 | fused_group32 | 157.114 | 11.541 | 170.990 | 160.856–177.731 |
| 0 | fused_group32_legacy_mma | 148.437 | 11.515 | 161.671 | 149.885–164.981 |
| 0 | fused_group64 | 161.147 | 11.291 | 173.059 | 164.846–181.580 |
| 10 | fused_sage32 | 161.871 | 10.345 | 174.360 | 172.419–177.605 |
| 10 | fused_group32 | 158.745 | 11.563 | 171.466 | 169.661–177.049 |
| 10 | fused_group32_legacy_mma | 150.827 | 11.584 | 163.370 | 161.787–164.200 |
| 10 | fused_group64 | 157.593 | 11.274 | 170.404 | 169.378–181.811 |
| 25 | fused_sage32 | 160.881 | 10.259 | 173.821 | 172.305–174.691 |
| 25 | fused_group32 | 156.916 | 11.569 | 171.123 | 169.038–173.525 |
| 25 | fused_group32_legacy_mma | 148.617 | 11.539 | 161.796 | 160.607–163.366 |
| 25 | fused_group64 | 159.345 | 11.284 | 171.377 | 169.642–173.566 |

The full-head result differs from the diagnostic-head result: fused preparation with the unchanged legacy Group32 MMA is the fastest measured Group32 variant. Across the nine layer/process cells its median API is161.796ms, versus174.360ms for fused Sage32 (7.2% lower); cached-QK Group32 is171.123ms. This selects a generation candidate, not a quality-qualified winner. Group64 is171.377ms and is not prioritized over the faster exact-contract Group32 candidate.
Group32 valid QI/KI match legacy on all56heads of all three captures; QS/KS maximum differences are zero. Independent reference checks use heads0/28/55 per capture; maximum grouped implementation relative L2 is0.00021230. They verify implementation only, not generation quality.
Exported SM75 PTX contains mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 for INT8 QK and mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 for FP16 PV. Opcode presence proves generated instruction types, not achieved Tensor Core utilization.
Nine Group Systems traces have zero unattributed kernels. Full-H56 preparation/casts launch18kernels/API for cached Group32/64 versus11for Sage, plus one attention kernel each. Shape-dependent ATen reduction dispatch differs from the earlier H1 launch count. The faster hybrid was measured unprofiled; its generation timeline is requested separately. Do not attribute its speedup to an unmeasured occupancy/stall mechanism.
Machine summary: D:\attention_eval\quality_constrained_20261015\native_full_summary.json; raw events, commands, source hashes, PTX and eighteen Systems traces: D:\attention_eval\quality_constrained_20261015\native_full_v1\native_full.
Native integration remains unvalidated here: these captures are contiguous BHND and return FP16 BHND. The generation test includes actual QKV strides and FP32 BND output conversions.
