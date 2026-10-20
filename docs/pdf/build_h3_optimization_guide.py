"""Generate the source-audited H3 optimization guide without loading H3/CUDA.

Uses the bundled document runtime, not the frozen inference environment.
All diagrams are vectors; measured charts read the retained physical results.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import html

from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
W,H=595.276,841.890
L,R=44,44
CW=W-L-R
INK='#142B40'; TEAL='#087E8B'; BLUE='#416DA1'; AMBER='#AD691E'
MUTED='#54677A'; LIGHT='#EDF3F7'; PALE='#E7F3F1'; OPTIONAL='#FCF1DE'
RED='#A03E47'; RULE='#D9E2E9'
for name,file in [('Body','segoeui.ttf'),('Bold','segoeuib.ttf'),('Semi','seguisb.ttf'),('Mono','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
STYLES={
 'body':ParagraphStyle('body',fontName='Body',fontSize=10.2,leading=14.4,textColor=colors.HexColor(INK),spaceAfter=7),
 'small':ParagraphStyle('small',fontName='Body',fontSize=8.6,leading=11.8,textColor=colors.HexColor(MUTED)),
 'table':ParagraphStyle('table',fontName='Body',fontSize=9,leading=12.2,textColor=colors.HexColor(INK)),
 'head':ParagraphStyle('head',fontName='Bold',fontSize=9,leading=12.2,textColor=colors.white),
 'box':ParagraphStyle('box',fontName='Body',fontSize=8.6,leading=11.3,textColor=colors.HexColor(INK),alignment=1),
 'boxhead':ParagraphStyle('boxhead',fontName='Bold',fontSize=10,leading=12.4,textColor=colors.HexColor(INK),alignment=1),
 'code':ParagraphStyle('code',fontName='Mono',fontSize=8.4,leading=12,textColor=colors.HexColor(INK)),
}

SOURCES=[
 ('S01','patches/h3_operator_policy.py','Operator defaults, architecture guards and refiner SDPA'),
 ('S02','custom_nodes/minimax_h3_tp2/__init__.py','Node controls and service identity'),
 ('S03','patches/comfy_tp_runtime.py','Persistent workers, runtime budgets and phase protocol'),
 ('S04','patches/comfy_tp_models.py','Native model construction and staged startup'),
 ('S05','patches/comfy_sampler.py','Native conditioning/sampling and public decode adapters'),
 ('S06','patches/tp_qwen3vl.py','TE TP2 geometry, attention boundary and tensor lifetimes'),
 ('S07','patches/tp_dit.py','DiT TP2 geometry, AdaLN and residual order'),
 ('S08','patches/tilelang_group32_masked.py','Mask-aware Group32 TE kernel'),
 ('S09','patches/tilelang_grouped_tp_kitchen_v2.py','Dense preparation, query caching and kernel mapping'),
 ('S10','patches/tilelang_grouped_tp_optimized.py','Sparse preparation, tau0 routing and reference equations'),
 ('S11','patches/tilelang_attention_p2.py','Omitted-only centroid score workspace'),
 ('S12','patches/tilelang_row_quant.py','Global-MAX activation quantizer and packed row preparation'),
 ('S13','patches/tilelang_swiglu_pad.py','SwiGLU fusion, lifetime repair and owned FP32 rotation'),
 ('S14','patches/tilelang_transport_p2.py','Optional FP16 transport and fused dequantization'),
 ('S15','patches/tilelang_transport_tradeoffs.py','Immutable transport codec and power-of-two bound'),
 ('S16','patches/parallel_int8.py','INT8 GEMM, INT32 SUM and native dequantization reference'),
 ('S17','patches/optimized_tp.py','Native/SM75 rotation dispatch'),
 ('S18','patches/h3_lora.py','Ordered online adapters, selective B and row partials'),
 ('S19','patches/offline_lora.py','Whole-layer dual-GPU baking and atomic publication'),
 ('S20','patches/windowed_tp.py','Positional reads, streamed specs and model proxies'),
 ('S21','patches/rank_local_int8.py','Owned checkpoint ranges and optional rank-local exports'),
 ('S22','patches/host_weight_cache.py','Shared host budget, pageable cache and pin staging'),
 ('S23','patches/weight_residency.py','H2D events, leases, eviction and adaptive lookahead'),
 ('S24','patches/staged_vae.py','Encoder/decoder-only construction and nonpersistent buffers'),
 ('S25','patches/parallel_vae.py','Video query ownership, ABI gates and decoder replicas'),
 ('S26','patches/audio_channel_decode.py','Stereo FP32 channel ownership and final AllGather'),
 ('S27','patches/comfy_idle_lifecycle.py','Queue-idle admission, pressure and cancellation hooks'),
 ('S28','patches/legacy_runtime.py','Pinned source selection and source identity checks'),
 ('S29','runtime/local_conda.json','Frozen user environment and ComfyUI identity'),
 ('S30','docs/reports/FC2_QUANTIZER_MEMORY.md','FC2 memory study, qualification scope and activation defaults'),
 ('S31','docs/data/fc2_quantizer_t4_v1_summary.json','Physical T4 FC2/FFN measurements at S513/8192/16384'),
 ('S32','docs/data/fc2_quantizer_t4_v2_summary.json','Integrated physical T4 S32768 confirmation'),
 ('S33','docs/guides/H3_OPERATOR_INTEGRATION.md','Historical target gates and integration boundaries'),
 ('S34','docs/guides/H3_OPERATOR_ENHANCEMENTS.md','Transport/lifetime/read/refiner enhancement qualification'),
 ('S35','docs/guides/H3_LOADING_STAGES.md','LoRA bake, VAE component read sizes and idle encoder stages'),
 ('S36','docs/guides/PARALLEL_VAE_INTEGRATION.md','Public decode preservation and audio numerical limits'),
 ('S37','docs/data/H3_PIPELINE_REVIEW_RESULTS.json','Recorded pipeline review checks and source closure'),
 ('S38','docs/reports/TP2_P2_OPTIMIZATION_REPORT.md','Historical profiling, scalar workspace and transport caveats'),
 ('S39','docs/reports/H3_RUNTIME_PROVENANCE_AUDIT.md','Local source provenance and Linux qualification limits'),
]
SOURCE_MAP={s[0]:s for s in SOURCES}
pages=[]
def page(section,title,dek,*blocks,refs=''):
    pages.append(dict(section=section,title=title,dek=dek,blocks=blocks,refs=refs))
def P(t):return ('p',t)
def Q(t,b):return ('note',(t,b))
def H2(t):return ('h',t)
def T(headers,rows,widths=None):return ('table',(headers,rows,widths))
def D(kind,height=220):return ('diagram',(kind,height))
def C(kind,height=160):return ('chart',(kind,height))
def Code(t):return ('code',t)


page('ORIENTATION','How to read this guide','A source-first explanation of what runs, why it helps, and what is still unproven.',
 P('The target is a two-rank MiniMax-H3 inference service for two T4 GPUs. The optimization is a combination of numerical operators, model partitioning, bounded weight movement and lifecycle control. Faster kernels alone do not explain its behavior.'),
 T(['Read in this order','Questions answered'],[
  ['Architecture and policy - pp. 3-5','Who owns the work? Which choices are defaults?'],
  ['Attention - pp. 6-10','What is quantized? How do dropping and pooling differ?'],
  ['Linear operators and LoRA - pp. 11-17','Why global scales and INT32 SUM? Where are buffers removed?'],
  ['Loading and VAEs - pp. 18-25','How do disk, CPU, GPU windows and decode stages cooperate?'],
  ['Preservation and validation - pp. 26-28','Which evidence applies? How should the system be operated?'],
  ['Source catalogue - pp. 29-30','Where can each mechanism and measurement be checked?']],[.45,.55]),
 H2('Three distinct meanings of correctness'),
 P('<b>Same-method conformance</b> means an optimized implementation matches its own numerical reference. <b>Native equivalence</b> means it matches the selected original operator at a defined boundary. <b>Generation quality</b> means trajectories, content, temporal structure and audio remain acceptable. Passing one does not imply the others.'),
 Q('Reading the figures','Teal marks a current default or active work. Amber marks an optional approximation or policy. Grey marks Native behavior, preserved control flow or inactive storage. Diagrams are explanatory schematics unless a chart explicitly names a measurement.'),
 P('<b>Notation:</b> S is the complete packed token count, M the flattened row count, K the linear input width, N the linear output width, D the attention head dimension, and r a TP rank. MiB = 2^20 bytes; GiB = 2^30 bytes. Kernel time, operator time and request time have different boundaries.'),refs='S01, S03, S30, S33-S39')

page('01 / SYSTEM','One Native pipeline, two persistent ranks','Keep ComfyUI semantics in the parent; distribute the expensive model work.',
 D('architecture',235),
 P('The ComfyUI parent retains prompt/image/audio preprocessing, public model interfaces, sampler scheduling and public VAE planning. A session owns two persistent workers, each bound to one CUDA device. Weights can remain in bounded CPU caches while GPU weight windows change between TE, DiT and decode phases.'),
 T(['Boundary','Ownership and consequence'],[
 ['Parent to ranks','Bounded synchronous CPU packets preserve tensor dtypes, nested containers and PackedLayout metadata. Persistence amortizes startup; this is not an asynchronous transport-overlap claim.'],
 ['Between ranks','Actual distributed collectives complete the TP row projections and VAE exchanges. The workers share the same configuration and collective order.'],
 ['Phase transition','TE, sampling and VAE decode have explicit lifetimes. Incompatible phases release or fence their owners before another stage is admitted.']],[.24,.76]),
 P('Workers set NCCL_P2P_DISABLE=1 and both minimum/maximum channel counts to 4, leaving NCCL protocol selection at its default. Direct GPU peer copies are not assumed. Those settings are a runtime contract, not a measurement of physical PCIe bandwidth.'),
 Q('Important limit','One-GPU logical-rank previews test arithmetic and wiring. They do not validate physical NCCL behavior, two-T4 latency or concurrent-request throughput.'),refs='S03-S05, S23-S26')

page('02 / PARTITIONING','TP2 changes the shapes that matter','Partition each packed component correctly, then recombine row-projection sums.',
 D('tp',170),
 T(['Per-rank quantity','TE / Qwen3-VL','DiT main blocks'],[
 ['Hidden / output width','5120','5376'],
 ['Attention heads; D','32 Q, 4 KV; D=128','28 Q/K/V; D=128'],
 ['Local Q or attention width','4096','3584'],
 ['Local FFN activation width','12800','7168'],
 ['Gate/up representation','Separate tensors','Packed local width 14336'],
 ['Row O: local K -> N','4096 -> 5120','3584 -> 5376'],
 ['Row FFN: local K -> N','12800 -> 5120','7168 -> 5376']],[.40,.30,.30]),
 P('Column-parallel projections split output rows. DiT QKV is three components, and FC1 is two: each rank owns half of <i>each</i> component, not a single half of the concatenated tensor. Row-parallel projections split the input dimension and leave the full output width on both ranks.'),
 P('For a row projection, each rank computes a local INT32 partial P[r]. The output is reconstructed from SUM(P[r]) using the shared activation scale and output-channel weight scale. RMSNorm and DiT AdaLN are replicated. Sequence lengths vary with Native packing; a historical S=6814 fixture is not a fixed deployment contract.'),refs='S06, S07, S16, S21')

page('03 / CONTROL','A per-session operator policy','Backend selection is explicit and becomes part of cache and worker identity.',
 T(['Setting','Current default','Alternative / scope'],[
 ['TE attention','tilelang_group32','native'],
 ['DiT main attention','kitchen','tilelang_dense / tilelang_drop / tilelang_scalar / native'],
 ['Token Refiner','Native SDPA','Independent of DiT main attention mode'],
 ['Row-linear backend','tilelang_packed','native / tilelang_scaled_fp16'],
 ['FC2 quantizer','owned_fp32','native full-width rotation'],
 ['SwiGLU padding / selective B','True / True','Each has eligibility guards'],
 ['Scalar score cap','96 MiB per rank','0 selects inline scalar summaries'],
 ['LoRA loading','online','offline baking'],
 ['Adaptive lookahead / idle','False / False','Explicit runtime options'],
 ['VAE decode','auto','Per-VAE native override']],[.31,.27,.42]),
 D('policy',125),
 P('The loader constructs an immutable H3OperatorPolicy and RuntimeConfig. Configuration equality decides whether the persistent service may be reused. Operator identity also follows prepared blocks, residency copies and idle-prefetch cache keys; changing a backend must not reuse a block configured for another backend.'),
 Q('Do not confuse two settings','The default native_int32 reduction contract remains in RuntimeConfig. Selecting tilelang_scaled_fp16 is a separate explicit backend choice. The fc2_quantizer preference only affects eligible packed rows with SwiGLU fusion; it does not override the scaled-FP16 backend.'),refs='S01-S03, S13, S14, S18')

page('04 / DENSE ATTENTION','What Group32 INT8 attention computes','Fine-grained Q/K scales, an INT32 dot product, and an FP16 probability-value path.',
 D('groups',150),
 P('For each token and each 32-channel group g, a scale is computed from the absolute maximum. Q is quantized directly. K is centered first. With D=128, the dot product consists of four 32-channel INT8 products, each restored with its own pair of FP32 scales.'),
 Code('sQ[i,g] = max(abs(Q[i,g])) / 127, floored at 1e-8\nQI[i,g] = clamp(round(Q[i,g] / sQ[i,g]), -127, 127)\nlogit[i,j] = sum_g INT32(QI[i,g] dot KI[j,g])\n             * sQ[i,g] * sK[j,g] / sqrt(D)'),
 P('K centering uses 64-key leaf centroids and the unweighted mean of those centroids, including the ragged final leaf with its actual count. Subtracting one common K offset changes every logit in a query row by a common constant in exact arithmetic, so softmax is invariant. Quantization and finite-precision boundaries still introduce approximation.'),
 P('Online softmax maintains a running maximum, denominator and weighted numerator as each 64-key leaf arrives. It avoids allocating a full S-by-S score tensor. Softmax state and output accumulation use FP32; probabilities and V feed an FP16 PV GEMM. The input preparation retains the selected FP16 rounding boundary.'),
 Q('Precision meaning','Group32 offers a more local Q/K representation than coarser scaling, but this document does not claim universal accuracy superiority over Kitchen. FP16 PV is the integrated TileLang path; the experimental INT8-PV attention variants are not enabled here.'),refs='S08-S10, S33')

page('05 / TE','TE: compact GQA, real masks, short lifetimes','Preserve Native embeddings, Q/K normalization and RoPE before the attention boundary.',
 D('te',190),
 P('Each rank has 32 Q heads but only 4 K/V heads. A query head maps to its shared KV head inside the TileLang kernel, avoiding explicit replication of KV storage. This is grouped-query attention (GQA), not a change to the learned projections.'),
 T(['Contract','Implementation behavior'],[
 ['Mask shapes','2D, 3D and 4D masks must broadcast to the local BHND query/key shape. Boolean masks become additive 0/-infinity values on device.'],
 ['Online softmax','Masks enter the exp2-domain logits. A fully masked query returns zero. A masked early leaf does not prevent later valid leaves from contributing.'],
 ['Native fallback','Unsupported architecture logs a fallback. Native backend selection preserves the full-model size policy, including short-prompt masked GQA behavior.'],
 ['Scope','TE prefill only. Past-KV-cache execution remains rejected.']],[.24,.76]),
 P('After the attention residual accumulation, normed input, Q/K/V and attention output references are deleted before allocating MLP intermediates. After down projection, gate/up/normalized inputs are released before the MLP residual, followed by its output. This changes reference lifetimes, not residual arithmetic order.'),
 Q('Why explicit deletion helps','A view or closure can keep an entire storage alive. Removing only one alias is insufficient. In inference, releasing the last Python owner permits allocator reuse after stream safety is satisfied; autograd outside this runtime can retain its own references.'),refs='S06, S08, S34')

page('06 / DIT','Kitchen attention is the main DiT default','The Token Refiner deliberately retains Native SDPA.',
 D('dit',165),
 P('Main DiT blocks normalize and modulate their inputs, project local Q/K/V, apply Native Q/K norms and RoPE, then dispatch through the operator policy. The default calls comfy_kitchen.int8_attention. Its result is converted to token-major local features before the row-parallel O projection and residual update.'),
 P('Kitchen and TileLang Group32 are different numerical methods. Kitchen remains the selected DiT and Video VAE attention implementation; the optional TileLang dense path uses the Group32 QK / FP16 PV contract. Historical experiments with alternative PV precision or centering do not silently change this selection.'),
 H2('The refiner is a separate boundary'),
 P('NativeRefinerAttention wraps the existing refiner module rather than storing bound methods that could still point at canonical CPU weights after residency copying. It keeps the Native projections, RMSNorm and fused RMSNorm/RoPE behavior, and explicitly calls PyTorch scaled_dot_product_attention. Refiner tokens are never routed through sparse DiT attention.'),
 Q('An old name is not an old execution path','KitchenRefinerAttention remains only as an import-compatible alias for the Native SDPA wrapper. It no longer selects INT8 attention. This deliberate change is not byte equivalence with the former INT8 refiner.'),
 P('For the optional dense Group32 DiT kernel, the current caller uses a 64-query tile, 128 threads, a row warp policy and cached Q operands. Such launch/resource choices are architecture-specific implementation details, not independent guarantees of Tensor Core utilization or pipeline speedup.'),refs='S01, S07, S09, S33, S34')

page('07 / SPARSE ROUTING','Which DiT attention blocks may be omitted?','Route 64-token leaves; protect conditioning through the current Native layout.',
 D('routes',235),
 P('The sparse modes require DiT self-attention with matching Q/K/V shapes. Query and key tokens are grouped into leaves of 64. A query-leaf centroid is compared with each centered key-leaf centroid; tau0 retains scores above the query-dependent mean threshold. Nearby leaves and the first key leaf are mandatory.'),
 P('The runtime obtains the dense prefix from the current PackedLayout. All text, references, conditioning and generated audio before the final generated-video segment remain protected. A missing or stale layout is rejected, not replaced with the historical 874-token fixture constant. Protection rounds to whole leaves, so a leaf intersecting the prefix is retained as a whole.'),
 T(['Always retained','Policy consequence'],[
 ['Prefix query leaves','Attend all key leaves.'],
 ['Prefix key leaves','Remain visible to all query leaves.'],
 ['Local neighborhood / first key leaf','Provide the routing method\'s mandatory connectivity.'],
 ['Other video-video pairs','Centroid threshold decides whether full token work is kept.']],[.43,.57]),
 Q('Sparsity is not speedup','Routing, preparation, launch overhead and work on retained blocks remain. Scalar pooling also processes omitted-leaf summaries. A percentage of omitted blocks is not a measured percentage of request-time savings.'),refs='S01, S10, S11, S38')

page('08 / SPARSE SEMANTICS','Dropping and scalar pooling are different','Both are optional zero-shot approximations; retain both choices.',
 D('sparse_modes',155),
 T(['For an omitted leaf','Dropping','Scalar pooling'],[
 ['Logits / denominator','No contribution','One centroid logit plus log(actual token count)'],
 ['Value contribution','No contribution','Weighted mean V of the omitted leaf'],
 ['Retained leaf','Full Group32 QK and FP16 PV','Same full Group32 QK and FP16 PV'],
 ['What is lost','All omitted probability mass','Within-leaf logit/value variation and correlation']],[.30,.35,.35]),
 Code('pooled_logit[i,b] = dot(Q[i], K_centroid[b]) / sqrt(D)\n                    + log(count[b])\npooled_numerator += exp(pooled_logit - running_max)\n                    * V_centroid[b]'),
 P('The log-count term represents multiple tokens with one summary; simply adding one unweighted centroid token would change the intended mass. For a ragged leaf, count is the true number of keys. The FP16 pooled-probability boundary and streaming normalization remain part of this approximation.'),
 P('Scalar mode may precompute only omitted-leaf query-centroid scores. The workspace is B*H*Np*Nb*4 bytes per rank. A retained real H3 fixture requires 82,066,432 bytes (78.3 MiB), below the 96 MiB default cap. A zero cap or a larger required workspace selects inline summaries. This trades one preparation kernel and memory for less centroid work in the main kernel.'),
 Q('Quality boundary','Sparse implementation conformance is checked against the same routing/pooling method. Sparse-versus-dense L1/L2 is not a generation-quality gate. Semantic/distribution drift and temporal behavior require full-pipeline generation review, which is outside these integrated operator checks.'),refs='S01, S10, S11, S38')

page('09 / ROW LINEAR','Global row scales make TP INT8 coherent','Local integer GEMMs are only valid together with the right cross-rank contract.',
 D('row',200),
 Code('m[i] = MAX_r max(abs(rotated_X[r][i,:]))\nsA[i] = max(m[i] * float32(1/127), 1e-30)\nQ8[r] = clamp(nearbyint(rotated_X[r] / sA), -128, 127)\nP[r] = Q8[r] @ W8[r].T                  # INT32\nY[i,j] = float32(SUM_r P[r][i,j]) * sA[i] * sW[j]'),
 P('The maximum must cover the entire logical activation row, even though each rank owns only part of it. Independent half-row scales would make the integer partials incompatible. Per-output weight scales also retain the full-width checkpoint quantization semantics; they are not recomputed independently on input-column shards.'),
 P('The packed row path optimizes ownership padding and activation preparation. It still uses Kitchen\'s native INT32-accumulating INT8 GEMM and an INT32 AllReduce SUM before dequantization. Here, <i>packed</i> does not mean compressed INT8 network transport.'),
 P('ConvRot groups contain 256 channels and are fully owned locally. The row quantizer follows its own Kitchen-compatible floor, fast division, nearby rounding and [-128,127] saturation. This is distinct from attention\'s Group32 quantizer with floor 1e-8 and [-127,127] range; interchanging them would change the numerical method.'),refs='S12, S13, S16, S17, S21')

page('10 / FUSION','Fuse SwiGLU with ownership padding','One output write replaces eager activation and padding stages.',
 D('swiglu',210),
 P('SwiGLU computes silu(gate)*up. TE supplies separate gate and up tensors; DiT supplies strided halves of one packed FC1 result. The fused TileLang kernel reads either layout directly. It writes the owning rank\'s output columns and explicitly writes positive zero into unowned columns.'),
 P('The full-width mode feeds the same Native FP32 ConvRot GEMM geometry. It preserves FP32 exp, rounded addition/division and multiplication; it does not substitute a fast approximate sigmoid, FP16 activation or a different row quantizer. With eligible shapes, one launch replaces eager SwiGLU plus padding work and removes the separate activated local tensor.'),
 T(['Site','Input / local activation','Padded output'],[
 ['TE down','Separate gate/up, each M x 12800','M x 25600 FP32'],
 ['DiT FC2, full-width mode','Packed M x 14336; local activation 7168','M x 14336 FP32'],
 ['DiT FC2, owned mode','Same packed input; local activation 7168','M x 7168 FP32, no global padding']],[.28,.41,.31]),
 Q('Fusion boundary','This removes intermediate traffic and launches around activation preparation. It does not fuse the INT8 GEMM, MAX collective, SUM collective and residual into a single kernel. Complete-row and full-request effects must be measured at their own boundaries.'),
 P('The shared helper also releases the full padded tensor in the actual caller after its last rotation read. An inner helper\'s del cannot free a buffer still referenced by the caller. Same-stream last-use ordering is retained.'),refs='S12, S13, S30, S34')

page('11 / FC2 MEMORY','Rotate only the channels the rank owns','The current DiT FC2 default removes full-width FP32 temporaries.',
 D('owned',225),
 P('At S8192, each rank\'s packed FC1 output is 448 MiB. The original path overlaps it with 448 MiB of zero padding and 448 MiB of full-width rotation. Padding was also retained through quantization, where the owned INT8 output adds 56 MiB. Repairing lifetime alone reduces that latter overlap.'),
 P('Owned FP32 rotation materializes only the local 7168-channel SwiGLU activation (224 MiB) and its rotation (224 MiB). All local 256-channel ConvRot groups are complete. The same global FP32 MAX, INT8 rounding, learned weight scales and INT32 SUM remain. No lossy activation conversion or compressed transport is introduced.'),
 Q('The arithmetic caveat','The flattened M dimension of the FP32 rotation GEMM changes. Algebraic group locality alone cannot prove identical rounding across algorithms/backends. The trained qualification matrix observed identical payload, scale bytes and complete FC2 outputs; this is not a universal bit-exactness theorem.'),
 P('Eligibility is deliberately narrow: packed H3 FC2 width 14336, FP32, TP2, group size 256, packed row backend and enabled SwiGLU fusion. Other shapes/backends retain their existing path. TE down keeps the full-width rotation with the lifetime repair. Set fc2_quantizer=native to restore full-width geometry.'),refs='S01, S13, S30-S32')

page('12 / MEASURED EVIDENCE','FC2: memory reduction survives integration','Actual historical two-T4 operator measurements, not an end-to-end forecast.',
 C('fc2memory',145),
 C('fc2latency',145),
 P('The charts use S8192 with the retained two-entry LoRA configuration and its existing optional Row Partial mode. Fourteen synchronized wall samples span two reversed rounds; the slower rank is selected per iteration. Temporary memory is measured above the resident entry in separate peak passes. It excludes allocator reservations and NCCL internal allocations.'),
 P('The owned path reduces whole-FFN temporary peak by 33.00% and the pooled FFN median by 5.44% in that allocation. The peak moves to unchanged base FC1 dispatch, whose two output buffers plus INT8 activation/scales total 938.031 MiB. FC2 itself is no longer the largest explicit FFN boundary.'),
 Q('Scope and version','These physical results used the historical T4 software stack. They do not qualify the current frozen conda stack on Linux. S32768 integrated confirmation also passes byte gates, but its two-entry whole-FFN timing varies by round: -0.56% then +5.02% reduction. Do not advertise its pooled gain as stable.'),refs='S30-S32, S29, S39')

page('13 / OPTIONAL TRANSPORT','Scaled FP16 SUM with a fused epilogue','Trade communication precision explicitly; retain Native INT32 as the default.',
 D('transport',175),
 Code('shift = max(0, ceil(log2(2*Klocal*127*127 / 60000)))\npayload[r] = FP16(float32(P[r]) * 2**(-shift))\nreduced = AllReduce_SUM(payload)\nrestored = INT32(float32(reduced) * 2**shift)\nY = (float32(restored) * activation_scale) * weight_scale'),
 P('The power-of-two factor uses a worst-case bound and adds no extra MAX collective. FP16 halves the logical per-rank SUM payload from four to two bytes per element. That is payload accounting, not a guarantee of half the wire traffic, collective duration or total operator latency.'),
 P('The TileLang epilogue reconstructs the specified INT32 value and performs the two left-associated FP32 scale products in one kernel, avoiding a separate reconstructed INT32 allocation. Current support requires TP2 FP32 input, aligned local groups, INT8 weights and contiguous vector FP32 weight scales on physical SM75. Unsupported contracts fall back before collectives.'),
 Q('Cancellation makes this approximate','Let P0=2^shift*2049 and P1=-2^shift*2048. The exact sum is 2^shift; FP16 encoding can round both magnitudes to 2048, producing zero. Agreement with an independent FP16 codec reference proves implementation correctness, not equality to Native INT32 SUM or preservation of generation semantics.'),refs='S01, S14-S16, S34, S38')

page('14 / ONLINE LORA','Selective B removes unowned FC1 outputs','Keep adapter order, separate GEMMs, and post-B scaling.',
 D('lora',190),
 P('Online LoRA adds a floating-point bypass around the INT8 base projection. For each adapter, hidden=linear(x,A), delta=linear(hidden,B), then multiply by its effective strength and add to the target. Adapters execute in Native order. Strength is not folded into B, and A factors are not concatenated into a different GEMM.'),
 P('Selective B is enabled only for qualified main DiT FC1 geometry: A=[128,5376], B=[28672,128]. Each rank needs 7168 gate rows and 7168 up rows. Those rows are read and packed once on CPU, so B produces 14336 owned outputs rather than all 28672. Unqualified shapes, QKV, refiner sites and other targets keep full B behavior.'),
 T(['LoRA choice','Semantics / status'],[
 ['Selective FC1 B, default on','Skips unowned B rows for the guarded geometry; validated selected-output arithmetic, not a general GEMM-rounding theorem.'],
 ['Native row-input gather, default','Assemble the complete input before A, retaining Native A GEMM geometry.'],
 ['Row A partial reduction, optional','Compute A partials locally and SUM the small hidden. Changes FP32 reduction grouping, so remains an explicit approximation.'],
 ['Concat A','Retired; requesting it raises a clear error.']],[.38,.62]),
 P('Row Partial staging is bounded to 32 MiB of small hidden tensors and only used when this can replace a larger SwiGLU activation. Neither selected B nor bounded staging changes the base INT8 GEMM or enables approximate SUM for that base projection.'),refs='S01-S03, S18, S30, S33')

page('15 / OFFLINE LORA','Bake whole layers across two GPUs','Move repeated inference adapter work to a reusable checkpoint conversion.',
 D('bake',220),
 P('Offline mode first releases the previous TP service. A size-balanced assignment gives each affected DiT/refiner layer to one of up to two spawned GPU workers. Each worker processes one full-width weight matrix at a time; a matrix and its quantization scale are never split between GPUs.'),
 P('For an INT8 target, the worker performs Native FP32 dequantization, ordered adapter merging, and deterministic full-input-width ConvRot requantization. Dense targets return to their original storage dtype. Quantizing the full width before later TP splitting is essential: independent half-width weight maxima would change the model.'),
 P('The base checkpoint is copied once to a new scratch artifact. Workers write disjoint weight/scale ranges through independent descriptors and fsync their writes. The parent publishes only after all workers finish, with source/stack identities, output signature, assignments and checksums in the receipt. No merged weight tensors travel over NCCL or parent pipes.'),
 Q('Warm reuse and numerical meaning','The cache key binds the base and ordered adapter stack, including strengths. A valid baked checkpoint is reused and the online stack is cleared to prevent double application. Baking introduces a new requantization boundary; it is not byte-exact with online LoRA over the original INT8 weights.'),
 P('Cancellation is checked during copying and between matrices. Failure removes only unpublished scratch owned by this invocation. A new bake needs space for another full diffusion checkpoint. Baking/loading costs are one-time preparation costs, not steady-state inference latency.'),refs='S02, S18, S19, S35')

page('16 / CHECKPOINT I/O','Read owned ranges, not full temporary weights','CheckpointReader supports positional access without changing the model schema.',
 D('io',190),
 P('The safetensors header maps each tensor to a dtype, shape and byte interval. The TP plan reads owned output rows for column projections, owned input columns for row projections, and replicated scales/norms where required. Current files retain the qualified INT8 ConvRot metadata. Source size and modification-time signatures guard against changing a checkpoint during a planned request.'),
 T(['Path','What it actually does'],[
 ['os.pread, when supported','Read at an absolute byte offset, loop on short reads, and copy returned bytes into the final tensor in chunks up to 8 MiB. It removes shared file-cursor mutation.'],
 ['Seek + readinto fallback','Used when pread is unavailable or a descriptor rejects it, including ordinary Windows operation. Correct destination offsets and range order remain.'],
 ['Rank-local offline export','Optional contiguous per-layer/rank safetensors with Native state hashes and source identity. This changes storage layout, not tensor math.']],[.31,.69]),
 Q('Pread is not direct I/O','Python os.pread returns a bytes object; it cannot fill a caller-supplied tensor buffer. This path is neither zero-copy nor O_DIRECT and does not bypass the operating-system page cache. No local SSD result is substituted for T4 storage behavior.'),
 P('CPU decompression is currently a separate experiment. ZSTD, BitShuffle and FSE benchmarks are not installed in the active loader, so the diagrams do not place a decompressor in the production critical path.'),refs='S20-S22, S34')

page('17 / RESIDENCY','Bound the pipeline from disk to GPU','Pageable cache, pinned staging and GPU windows have separate owners and limits.',
 D('residency',225),
 P('HostWeightCache prepares immutable CPU modules under a process-shared HostBudget. Per-rank cache capacity limits retained pageable modules, while construction reservations cover preparation peaks. A cache hit can bypass disk/CPU preparation without implying that the module is already pinned or resident on a GPU.'),
 P('ResidencyWindow admits pin staging, copies modules on a CUDA copy stream, waits on copy events before use, and delays storage retirement until recorded compute reads are safe. Eviction considers future uses and explicit retained keys. CPU and GPU lookahead are scheduling horizons, not permissions to exceed their byte budgets.'),
 T(['RuntimeConfig default','Meaning'],[
 ['GPU weights: 6 GiB/rank','Live/pending managed weight bound; does not include all activations or library state.'],
 ['Pinned: 6 GiB/rank','Admission bound for staging, with allocator-aware accounting.'],
 ['Pageable cache: 3 GiB/rank','Retained prepared CPU weight cache.'],
 ['Shared host: 16 GiB','Cross-rank managed reservations, not total machine RSS.'],
 ['CPU/GPU horizon: 16 / 2','Bounded upcoming preparation and distinct GPU groups; reserve remains enforced.']],[.43,.57]),
 P('Adaptive lookahead defaults off. When enabled, the GPU depth becomes one or two upcoming distinct groups based on driver free memory plus reusable allocator memory, the weight cap and reserve. Admission rechecks headroom immediately before every copy. The opt-in also retains the existing CPU timing adaptation.'),refs='S03, S20, S22, S23, S34')

page('18 / LIFETIMES','Release stages, not just local variables','The last owner determines when a large tensor can leave the live set.',
 D('lifetimes',205),
 P('Peak storage is determined by overlap. Summing the largest buffer in each stage overstates a peak if those buffers never coexist; counting only returned tensors understates it when views, closures, caller arguments or autograd retain a backing allocation. The code therefore gives attention, MLP, preprocessing and VAE phases explicit release boundaries.'),
 T(['Release point','Why it exists'],[
 ['TE attention residual','Drop normalized input, Q/K/V and attention output before the MLP starts.'],
 ['DiT attention residual','Release attention views, packed QKV and projection temporaries before FC1/FC2 work.'],
 ['After FC2 rotation','The caller drops padding after its last read; helper-local deletion alone cannot release caller ownership.'],
 ['Encoder to sampling / decoder','Release staged encoder model, patcher and host lease before admitting the next large phase.'],
 ['Close / abort','Fence/release workers, parent leases and idle speculation; unregister closed-service atexit callbacks.']],[.36,.64]),
 Q('What memory numbers mean','Named tensor payload, PyTorch allocated peak, allocator reserved peak, managed host reservation, process RSS and device/NVML usage are different quantities. The FC2 chart reports scoped PyTorch temporary allocation; it is not a 33% claim for full-pipeline VRAM.'),
 P('The owned-FC2 ledger moves a representative block-created payload maximum to the O projection, where attention buffers remain alive during the row call. Optimizing one stage can expose another bottleneck.'),refs='S03, S06, S07, S13, S23, S24, S30')

page('19 / VAE STAGES','Load the encoder first, decoder only at output','Native public planning remains; the unused model half has no parameter storage.',
 C('vaebytes',205),
 P('MiniMaxH3ParallelVAELoader recognizes Native H3 video/audio checkpoint signatures and creates a StagedVAE. A meta-device shell retains Native metadata and public methods. At initial load, only encoder-side parameters and shared latent normalization are read. The inactive decoder is removed before real storage allocation.'),
 P('Before denoising, registered encoder stages are explicitly released. At first decode, the encoder patcher, model backing and host lease are torn down before decoder reads and construction. A later encode reloads the encoder automatically. The session lock and VAE lock serialize these switches.'),
 T(['Stage','Retained checkpoint components'],[
 ['Video encode / decode','encoder + quant_conv / decoder + post_quant_conv'],
 ['Audio encode / decode','encoder + pre_block + posterior projections / decoder + dec_in_proj'],
 ['Shared / reconstructed','Latent normalization, Native pixel statistics, deterministic nonpersistent decoder RoPE and unit qk_norm_scale buffers.']],[.31,.69]),
 Q('Chart boundary','Bars show checkpoint payload bytes read for each component, with shared latent buffers included in both stages. They are not peak RAM/VRAM. Parent GPU transfer/offload remains under Native model management; worker decoders use a separate conservative admission.'),refs='S02, S03, S24, S35')

page('20 / VIDEO VAE','Query parallelism keeps global context','Split query ownership, exchange all keys and values, join only final projections.',
 D('video',245),
 P('The video decoder uses all 36 trained transformer blocks. Both ranks hold decoder weights; token/query ownership is split contiguously at a 128-query scale-group boundary. The split includes the five appended register/padding positions. Rank 0 performs Native prefix projection once and broadcasts the other owner\'s token range. RoPE uses the original global coordinates.'),
 P('At each attention layer, both ranks project their local Q/K/V. Local Q is kept, while K and V are gathered globally. Each owner therefore attends over the entire key/value context with Kitchen INT8 attention. After the last decoder block, projected outputs are joined and Native patch reconstruction is completed on rank 0.'),
 Q('Why not independent video slabs?','Decoding cropped spatial/temporal pieces independently would remove global attention context and can alter boundaries or consistency. This implementation partitions queries while retaining full KV. It is not head sharding or independent cropped-slab generation.'),
 P('The parent replaces only the raw _decode_pixels boundary. Native normalization, temporal chunking, spatial tiles, blending, trimming, public output buffers and finalization remain in place. Decoder replicas are reused across raw calls within one public decode and released afterward.'),refs='S05, S25, S36')

page('21 / VIDEO FALLBACK','Compatibility is checked before collectives','The old attention API is preserved because it is part of the qualified contract.',
 D('fallback',250),
 P('Query partitioning requires the Native Kitchen dispatch flag, quantized QKV wrappers, qk_norm_scale and the legacy attention signature accepting pre_norm, residual and residual_scale. Package import success alone does not prove those contracts. The staged decoder rebuilds nonpersistent buffers that are absent from safetensors.'),
 P('The parent loads its required stage before VRAM admission so that allocation is visible. Both ranks agree with a MIN collective on conservative host/GPU admission before raw decode starts. Decode mode native bypasses parallel preparation. Very short sequences without two aligned ranges run Native raw decode on rank 0.'),
 Q('Two different failure boundaries','A compatibility or admission rejection can select Native public decode before raw collectives. An error after collective decode begins must propagate and abort/clean up the service; silently switching one rank to Native could deadlock or accept a partial output.'),
 P('The sibling ComfyUI-master has a different attention wrapper. The pinned checkout retains the required API. A changed wrapper can disable query parallelism even with the same Kitchen package version. The solution here is source/version preservation and a recorded fallback reason, not an automatic upstream API migration.'),refs='S24, S25, S28, S29, S35, S36, S39')

page('22 / AUDIO VAE','Two channels, two FP32 decode owners','Channel independence creates parallel work without temporal tiling.',
 D('audio',210),
 P('The active worker path is audio_channel_decode.channel_decode. It accepts FP32 stereo latents shaped [B,32,2,T]. Rank 0 decodes channel 0, rank 1 decodes channel 1, using the already-loaded decoder replica. A waveform AllGather restores left/right order; rank 0 concatenates the channels for the Native public wrapper.'),
 P('The public VAE continues to own chunking, normalization and output layout conversion. The implementation does not construct a second dual-device manager inside each persistent worker, and it does not import the historical temporal/tiled benchmark manager.'),
 T(['Numerical comparison','Recorded result / interpretation'],[
 ['Sequential mono vs channel ownership','Exact in the retained local qualification.'],
 ['Native stereo vs split mono','Relative L2 approximately 1.280e-5, 1.237e-5 and 1.205e-5 at 5/10/20 seconds. The old strict 1e-5 gate still fails.'],
 ['Perceptual/audio claim','No claim of inaudibility follows from those metrics. Native decode remains selectable.']],[.40,.60]),
 Q('Accepted approximation, not renamed exactness','Splitting the channel/batch geometry can change the FP32 convolution/backend reduction choices. The weights and mathematical channel independence remain, but floating-point evaluation is not guaranteed byte-identical to Native stereo. This difference is explicitly retained and disclosed.'),refs='S05, S25, S26, S36')

page('23 / IDLE WORK','Preload encoders, then a bounded TE prefix','Speculation is optional, cancellable and tied to the next compatible request.',
 D('idle',225),
 P('With idle_residency=True, successful queue-idle completion starts a background task for the registered VAE encoders, then asks both workers to begin partial TE preload. Encoders live in the parent on the ordinary Native VAE device, normally GPU 0. They are not replicated on the two TP ranks, and workers see their allocation through driver-memory admission.'),
 P('Encoder reads check cancellation between tensors. GPU preload needs weight storage plus the configured idle reserve. If GPU headroom is insufficient, a prepared CPU encoder may still be reused. If encoder speculation fails, the reason is recorded and TE speculation may still proceed. No reference encoding, prompt conditioning, initial noise or decoder inference is executed speculatively.'),
 T(['Event','Lifecycle action'],[
 ['New prompt','Cancel outstanding speculation; reuse only compatible completed entries.'],
 ['Active graph changed','Filter registered VAE names and release stale cached nodes.'],
 ['Memory pressure / incompatible identity','Release encoder and TE speculation; foreground work has priority.'],
 ['Denoising starts','Explicitly release staged encoders; active GPU window takes ownership.']],[.37,.63]),
 P('Default idle weight cap is 4 GiB/rank, with a 4 GiB idle reserve and 0.25-second delay; idle remains disabled by default. This is separate from the active 6 GiB managed GPU weight cap. Reuse never bypasses source/adapter/operator identity checks or stream-lifetime fences.'),refs='S02, S03, S23, S24, S27')

page('24 / PRESERVATION','The validated runtime is an input','Operator adaptation must not silently upgrade source, APIs or dependencies.',
 T(['Contract','Preserved local stack'],[
 ['ComfyUI source','d49e888586dd8ae012c0667b33466b815fee07f7'],
 ['Default interpreter','C:/ProgramData/miniconda3/envs/comfyui/python.exe'],
 ['Torch / Kitchen / TileLang','2.12.1+cu132 / 0.2.37 / 0.1.13'],
 ['Frozen environment record','runtime/local_conda.json'],
 ['Historical physical T4 operator stack','Torch 2.11.0+cu130, Kitchen 0.2.37, TileLang 0.1.15; nvcc 12.8.93'],
 ['Current Linux qualification','Not established by the historical stack. Explicit platform lock / ABI work remains.']],[.39,.61]),
 D('versions',145),
 P('The launcher selects the pinned local ComfyUI and checks normalized source identity. It does not replace it with the API-different sibling checkout. The original sibling has no Git identity in the retained provenance record; this document does not claim it was synchronized upstream or rolled back.'),
 P('Kaggle jobs must use the frozen source archive and exact package versions. Linux-only CUDA/NCCL/Triton dependencies and wheel/ABI differences require explicit resolution and validation. Current setup blocks rather than substituting versions. The historical TileLang 0.1.15 gate does not automatically qualify 0.1.13 on Linux.'),
 Q('No hidden environment switch','The historical D:/h3_tp2_legacy_env is not the default. This PDF is generated with the separate bundled document runtime; it does not install or change inference packages. No GPU time is consumed for document creation.'),refs='S28, S29, S33, S39')

page('25 / EVIDENCE','What the retained checks actually establish','Conformance, physical transport and full-model quality are separate evidence classes.',
 T(['Evidence','What it covers','What it does not cover'],[
 ['Pipeline review: 56 tests','Lifecycle, source/runtime preservation, policy/UI, loading, cache identity, debug isolation','Full H3 generation speed or physical NCCL timing'],
 ['Historical physical integration: 10 gate records/rank','Masked GQA, selected B ownership, drop/scalar dispatch, native INT32 row collectives','Current frozen Linux stack; full-depth quality'],
 ['Enhancements: local arithmetic / contracts','Scaled-FP16 codec reference, refiner SDPA, positional-read fallback, explicit lifetimes','New physical T4 performance for those final sources'],
 ['FC2 physical v1/v2','Trained owned-rotation payload/scale/output identity; scoped memory/latency; integrated S32768 dispatch','Universal GEMM rounding equivalence or pipeline VRAM'],
 ['Trained VAE local checks','36-block video query and staged encode/decode agreement in tested fixtures; audio differences recorded','Physical integrated near-capacity two-T4 request performance']],[.25,.38,.37]),
 P('The charts in this guide use retained measurements and do not launch new experiments. Historical sources/results stay version-bound. A test against an approximate reference can pass perfectly while still differing from dense attention or Native INT32 transport.'),
 P('Nsight Systems evidence in prior operator investigations explains launches, preparation, collective intervals and allocation boundaries. Nsight Compute returned ERR_NVGPUCTRPERM on the retained Kaggle probes. Registers/shared-memory sizes and kernel interval coverage are not substitutes for achieved occupancy, SM utilization, memory-stall or Tensor Core counter measurements.'),
 Q('Full-pipeline scope','The complete current default/optional dispatch combinations have not all been qualified for full-depth two-T4 generation, temporal/audio semantics, queued concurrency or end-to-end latency. Operator gains must not be added together into a pipeline speedup.'),refs='S30-S39')

page('26 / OPERATIONS','Operate, inspect and change one boundary at a time','Use the explicit configuration and preserve Native fallbacks.',
 Code('from patches.comfy_tp_runtime import RuntimeConfig\nfrom patches.h3_operator_policy import H3OperatorPolicy\nconfig = RuntimeConfig(\n    te_path, dit_path,\n    adaptive_lookahead=False,\n    operators=H3OperatorPolicy(\n        te_attention="tilelang_group32",\n        dit_attention="kitchen",\n        row_linear_backend="tilelang_packed",\n        fc2_quantizer="owned_fp32",\n        swiglu_padding=True, selective_b=True,\n        scalar_score_workspace_mib=96,\n    ),\n)'),
 P('Use the linked MiniMaxH3TP2Loader and parallel VAE nodes so that all stages share one persistent session. A changed policy or checkpoint/adapter identity recreates the service. Production TileLang dispatch checks SM75; an SM120 arithmetic preview must not be reported as a target run. Physical TP2 requires matching CUDA devices and an available NCCL backend.'),
 T(['Symptom / investigation','First boundary to inspect'],[
 ['Unexpected Native fallback','Selected source identity, GPU capability, VAE attention ABI and explicit decode_mode.'],
 ['High peak memory','Live aliases, parent VAE plus worker replicas, FC1/O stage peaks, weight/pinned/host budgets.'],
 ['No sparse speedup','Preparation/routing/summary costs and complete-API timings; never only omitted-block count.'],
 ['Quality drift','Selected approximation: sparse mode, FP16 SUM, Row Partial or offline requantization. Hold inputs/noise fixed.']],[.39,.61]),
 P('Runtime code is under patches/ and the custom node. Read-only stats snapshots and bounded counters are separate from explicit profilers/captures in benchmarks/. Native reference loaders are lazy qualification paths. Legacy PP/SP/tiled interfaces remain compatible but are not selected by this TP2 loader.'),
 Q('Not integrated by this document','TE compression codecs, experimental attention INT8 PV, packed-INT8 transport and broad auto-tuning are not enabled. No H3 inference, dependency setup job, new profiler capture or Git publication was performed to produce this guide.'),refs='S01-S03, S28, S33-S39')

# Two dedicated source pages keep references legible and make the PDF auditable.
page('REFERENCE / 1','Implementation source catalogue','Paths are relative to the root on the cover. The PDF embeds a JSON manifest with source hashes.',
 ('sources',SOURCES[:20]),refs='Current source audit: 2026-10-20')
page('REFERENCE / 2','Evidence and lifecycle source catalogue','Read code for current selection; read version-bound results for measurement scope.',
 ('sources',SOURCES[20:]),refs='Current source audit: 2026-10-20')


def para(c,text,x,y,width,style='body'):
    p=Paragraph(text,STYLES[style]);_,hh=p.wrap(width,10000);p.drawOn(c,x,y-hh)
    return hh

def box(c,x,y,w,h,title,body='',fill=PALE):
    c.setFillColor(colors.HexColor(fill));c.setStrokeColor(colors.HexColor(RULE));c.roundRect(x,y,w,h,7,stroke=1,fill=1)
    th=para(c,title,x+6,y+h-10,w-12,'boxhead')
    if body:para(c,body,x+7,y+h-15-th,w-14,'box')

def line(c,x1,y1,x2,y2,color=TEAL,arrow=True,label=''):
    c.setStrokeColor(colors.HexColor(color));c.setFillColor(colors.HexColor(color));c.setLineWidth(1.3);c.line(x1,y1,x2,y2)
    if arrow:
        import math
        a=math.atan2(y2-y1,x2-x1);p=c.beginPath();p.moveTo(x2,y2)
        p.lineTo(x2-6*math.cos(a-.45),y2-6*math.sin(a-.45));p.lineTo(x2-6*math.cos(a+.45),y2-6*math.sin(a+.45));p.close();c.drawPath(p,fill=1,stroke=0)
    if label:
        c.setFont('Body',8);c.drawCentredString((x1+x2)/2,(y1+y2)/2+5,label)

def label(c,text,x,y,size=9,color=MUTED):
    c.setFillColor(colors.HexColor(color));c.setFont('Body',size);c.drawString(x,y,text)

def diagram(c,kind,height):
    w=CW;h=height
    if kind=='architecture':
        box(c,0,h-73,w,65,'ComfyUI parent','Native reference encoding, conditioning, sampler and public VAE planner',LIGHT)
        for x,title in [(0,'Rank 0 / GPU 0'),(266,'Rank 1 / GPU 1')]:
            box(c,x,40,241,91,title,'TE / DiT rank shards<br/>Bounded weight window + CPU cache<br/>Decoder replica when admitted')
            line(c,x+120,h-73,x+120,131,label='CPU packets')
        line(c,241,84,266,84,label='NCCL')
        label(c,'One session lock and an explicit phase protocol coordinate both owners.',0,14)
    elif kind=='tp':
        box(c,0,84,145,65,'Column projection','Replicated input<br/>Own half of each output component',LIGHT)
        box(c,181,84,145,65,'Local attention / MLP','Per-rank heads or<br/>gate/up channels')
        box(c,362,84,145,65,'Row projection','Own input columns<br/>Full-width INT32 partial')
        line(c,145,117,181,117);line(c,326,117,362,117)
        box(c,181,4,326,51,'INT32 SUM, then dequantize','The residual width is replicated again.')
        line(c,433,84,433,55)
    elif kind=='policy':
        for x,title,body in [(0,'Node controls','Named policy fields'),(175,'RuntimeConfig','Immutable session identity'),(350,'Prepared block','Policy survives H2D copies')]:box(c,x,37,157,73,title,body,LIGHT if x==175 else PALE)
        line(c,157,73,175,73);line(c,332,73,350,73)
        label(c,'Different policy -> different service/cache identity; no stale backend reuse.',0,10)
    elif kind=='groups':
        for j,t in enumerate(['Channels 0-31','32-63','64-95','96-127']):box(c,j*129,73,120,54,t,'Own Q/K scale')
        for j in range(4):line(c,j*129+60,73,j*129+60,50)
        box(c,60,0,387,47,'Four INT32 group products -> FP32 logits','Restore each group scale before summing.',LIGHT)
    elif kind=='te':
        box(c,0,112,153,64,'Native Q/K norm + RoPE','32 local Q heads<br/>4 local KV heads',LIGHT)
        box(c,177,112,153,64,'Group32 masked kernel','Map 8 Q heads to 1 KV head<br/>No repeated KV tensor')
        box(c,354,112,153,64,'Row O + residual','4096 -> 5120<br/>INT32 SUM')
        line(c,153,144,177,144);line(c,330,144,354,144)
        box(c,0,20,239,61,'Release attention storage','Drop Q/K/V, normalized input and output',LIGHT)
        box(c,269,20,238,61,'Then allocate MLP stage','Gate/up -> down -> residual')
        line(c,430,112,430,95);line(c,430,95,120,95,arrow=False);line(c,120,95,120,81);line(c,239,50,269,50)
    elif kind=='dit':
        box(c,0,83,239,68,'Main DiT blocks','Kitchen INT8 attention by default<br/>Optional Group32 dense/drop/scalar')
        box(c,268,83,239,68,'Token Refiner','Native SDPA<br/>Native projections, RMSNorm and RoPE',LIGHT)
        box(c,0,4,507,51,'Native packed sequence and conditioning boundaries','The refiner selection does not follow the main DiT sparse switch.',LIGHT)
        line(c,120,83,120,55);line(c,387,83,387,55)
    elif kind=='routes':
        size=22;ox=162;oy=31;n=8
        for i in range(n):
            for j in range(n):
                mandatory=i<2 or j<2 or abs(i-j)<=1 or j==0
                retained=mandatory or (i+j)%4==0
                c.setFillColor(colors.HexColor(TEAL if mandatory else BLUE if retained else '#E6EBF0'));c.rect(ox+j*size,oy+(n-1-i)*size,size-1,size-1,fill=1,stroke=0)
        label(c,'Key leaves ->',ox,oy+n*size+12)
        label(c,'Query',106,oy+100);label(c,'leaves',106,oy+87)
        label(c,'Illustration: 2 prefix leaves + 6 generated-video leaves',28,7)
        box(c,363,126,144,71,'Protected / mandatory','Prefix rows + columns<br/>Neighbor leaves',PALE)
        box(c,363,38,144,71,'Other video pairs','Blue: threshold kept<br/>Grey: omitted',LIGHT)
        label(c,'Not a captured route map.',0,h-10,8)
    elif kind=='sparse_modes':
        box(c,0,80,145,61,'Same route map','Retained leaf = full attention',LIGHT)
        box(c,184,80,145,61,'Dropping','Omitted leaf contributes nothing',OPTIONAL)
        box(c,362,80,145,61,'Scalar pooling','Omitted leaf becomes one summary',OPTIONAL)
        line(c,145,110,184,110);line(c,145,110,162,110,arrow=False);line(c,162,110,162,52,arrow=False);line(c,162,52,435,52,arrow=False);line(c,435,52,435,80)
        label(c,'Pooling preserves approximate leaf mass, not token-level attention structure.',0,9)
    elif kind=='row':
        for x,title in [(0,'Rank 0'),(268,'Rank 1')]:
            box(c,x,123,239,63,title,'Owned rotated features -> local absmax')
            box(c,x,18,239,60,'Quantize + INT8 GEMM','Shared row scale -> local INT32 partial')
            line(c,x+120,123,x+120,78,label='global MAX')
        line(c,239,154,268,154,label='MAX')
        line(c,239,48,268,48,label='SUM')
        label(c,'SUM is over INT32 partials; dequantization follows the collective.',0,0)
    elif kind=='swiglu':
        label(c,'Before / full-width preparation',0,195,10,INK)
        for x,title,body in [(0,'Gate + up','Input tensors'),(175,'SwiGLU tensor','Local activated buffer'),(350,'Ownership padding','Full-width output')]:box(c,x,123,157,57,title,body,LIGHT)
        line(c,157,151,175,151);line(c,332,151,350,151)
        label(c,'Fused path',0,94,10,INK)
        box(c,0,17,157,60,'Gate / up views','Separate TE or packed DiT',LIGHT)
        box(c,205,17,302,60,'One TileLang launch','Compute FP32 SiLU*up and store owned columns / zero padding')
        line(c,157,47,205,47)
    elif kind=='owned':
        label(c,'Illustrated live payload at S8192, per rank (not total VRAM)',0,211,9,INK)
        rows=[('Original',[('Packed input',448),('Padding',448),('Rotation',448)],LIGHT),('Owned',[('Packed input',448),('Activation',224),('Rotation',224)],PALE)]
        for ri,(name,parts,fill) in enumerate(rows):
            yy=137-ri*86;label(c,name,0,yy+34,10,INK);x=83
            for title,amount in parts:
                ww=amount/1344*424
                box(c,x,yy,ww-3,64,title,f'{amount} MiB',fill);x+=ww
        label(c,'At quantization the original also retains 56 MiB INT8 + row vectors.',0,12,8.7)
    elif kind=='transport':
        for x,title,body,fill in [(0,'INT32 partial','Exact local accumulation',PALE),(175,'Scaled FP16 SUM','Explicit approximate transport',OPTIONAL),(350,'Fused epilogue','Restore INT32, then two FP32 scales',OPTIONAL)]:box(c,x,80,157,79,title,body,fill)
        line(c,157,119,175,119);line(c,332,119,350,119)
        box(c,0,4,507,48,'Default remains native INT32 SUM','The optional backend changes representation, not merely the epilogue.',LIGHT)
    elif kind=='lora':
        box(c,0,113,151,64,'A projection','x -> rank-128 hidden',LIGHT)
        box(c,177,113,153,64,'Selected B rows','Owned gate half<br/>+ owned up half')
        box(c,356,113,151,64,'Scale then add','Adapter 1, then adapter 2<br/>Native ordered bypass')
        line(c,151,145,177,145);line(c,330,145,356,145)
        for x,t in [(0,'Gate0'),(126,'Gate1'),(253,'Up0'),(380,'Up1')]:box(c,x,29,122,51,t,'Rank '+('0' if t.endswith('0') else '1'),PALE if t.endswith('0') else LIGHT)
        label(c,'B output row order is [gate | up], not [all rank-0 | all rank-1].',0,6)
    elif kind=='bake':
        box(c,0,151,507,56,'Original checkpoint + ordered adapter stack','Copied once to an unpublished new checkpoint',LIGHT)
        for x,t in [(0,'GPU worker 0'),(268,'GPU worker 1')]:
            box(c,x,60,239,63,t,'Whole assigned layers<br/>One full-width matrix at a time');line(c,x+120,151,x+120,123)
        box(c,0,0,507,41,'Join workers -> fsync -> receipt -> publish immutable artifact','Disjoint file writes; no merged-tensor NCCL exchange',LIGHT)
        line(c,120,60,120,41);line(c,387,60,387,41)
    elif kind=='io':
        box(c,0,109,154,67,'Safetensors header','Absolute ranges<br/>Dtype + shape + source identity',LIGHT)
        box(c,178,109,154,67,'TP ownership plan','Column: selected rows<br/>Row: selected columns')
        box(c,356,109,151,67,'Final CPU tensors','Replicated scales / norms<br/>No full-weight temporary')
        line(c,154,142,178,142);line(c,332,142,356,142)
        box(c,0,12,239,59,'Positional read if supported','pread returns bytes, then copy to destination',LIGHT)
        box(c,268,12,239,59,'Fallback','Independent seek + readinto',LIGHT)
        line(c,255,109,255,84,arrow=False);line(c,255,84,120,84,arrow=False);line(c,120,84,120,71);line(c,255,84,387,84,arrow=False);line(c,387,84,387,71)
    elif kind=='residency':
        boxes=[(0,'Disk','Owned ranges'),(130,'CPU cache','Pageable modules'),(260,'Pinned staging','Bounded leases'),(390,'GPU window','Copy/use events')]
        for x,t,b in boxes:box(c,x,136,117,65,t,b,LIGHT if x==0 else PALE)
        for x in [117,247,377]:line(c,x,168,x+13,168)
        label(c,'Planned overlap (schematic, not a profiler trace)',0,109,9,INK)
        for yy,name in [(68,'Copy'),(25,'Compute')]:label(c,name,0,yy+13)
        for x,t in [(83,'L0'),(204,'L1'),(325,'L2')]:box(c,x,65,105,29,t,'',PALE)
        for x,t in [(143,'L0'),(264,'L1'),(385,'L2')]:box(c,x,22,105,29,t,'',LIGHT)
        label(c,'A copy must complete before use; eviction waits for last-use safety.',0,2,8.5)
    elif kind=='lifetimes':
        label(c,'Logical execution ->',0,190,9,INK)
        marks=[('Attention',0,238),('Residual',242,80),('MLP',326,181)]
        for title,x,ww in marks:box(c,x,133,ww-4,41,title,'',LIGHT)
        label(c,'Attention buffers',0,102);label(c,'MLP buffers',0,53)
        c.setFillColor(colors.HexColor(TEAL));c.roundRect(112,94,202,18,4,fill=1,stroke=0)
        c.setFillColor(colors.HexColor(BLUE));c.roundRect(328,45,179,18,4,fill=1,stroke=0)
        c.setDash(3,3);c.setStrokeColor(colors.HexColor(AMBER));c.line(320,24,320,179);c.setDash()
        label(c,'Release last attention owners before creating the MLP live set.',0,8)
    elif kind=='video':
        box(c,0,177,507,55,'Native token projection + global coordinates','Rank 0 projects once; contiguous query ranges are aligned to 128.')
        for x,title in [(0,'Rank 0: local Q0'),(268,'Rank 1: local Q1')]:
            box(c,x,78,239,70,title,'Local K/V -> global KV AllGather<br/>Attend against all keys and values')
            line(c,x+120,177,x+120,148)
        line(c,239,114,268,114,label='KV')
        box(c,0,7,507,46,'Repeat 36 blocks -> project locally -> join projected outputs','Native patch reconstruction and public decode planning stay outside.',LIGHT)
        line(c,120,78,120,53);line(c,387,78,387,53)
    elif kind=='fallback':
        box(c,101,186,305,53,'Legacy ABI + quantized-wrapper checks','Required signature, flag and deterministic buffers',LIGHT)
        box(c,0,92,239,65,'Supported + admitted','Both ranks agree before raw decode')
        box(c,268,92,239,65,'Rejected before raw collectives','Record reason -> Native public decode',LIGHT)
        line(c,194,186,120,157,label='yes');line(c,319,186,387,157,label='no')
        box(c,0,7,239,52,'Query-parallel decode','Errors after start propagate / abort')
        box(c,268,7,239,52,'Native public decode','Native override also enters directly here',LIGHT)
        line(c,120,92,120,59);line(c,387,92,387,59)
    elif kind=='audio':
        box(c,0,145,507,53,'FP32 stereo latents [B,32,2,T]','Native public wrapper owns normalization and chunking.',LIGHT)
        for x,title in [(0,'Rank 0: left channel'),(268,'Rank 1: right channel')]:
            box(c,x,59,239,58,title,'Decode with the resident FP32 replica');line(c,x+120,145,x+120,117)
        box(c,0,0,507,37,'Waveform AllGather -> concatenate left/right on rank 0','',LIGHT)
        line(c,120,59,120,37);line(c,387,59,387,37)
    elif kind=='idle':
        for x,title,body in [(0,'Queue becomes idle','Optional policy enabled'),(175,'Parent encoder preload','Native VAE device / host lease'),(350,'Worker TE prefix','Partial weights on both ranks')]:box(c,x,134,157,74,title,body,PALE if x else LIGHT)
        line(c,157,171,175,171);line(c,332,171,350,171)
        box(c,0,34,239,65,'Compatible next prompt','Cancel outstanding jobs<br/>Reuse compatible completed storage')
        box(c,268,34,239,65,'Pressure / identity change','Cancel + release speculation<br/>Foreground request takes priority',OPTIONAL)
        line(c,120,134,120,99);line(c,427,134,387,99)
        label(c,'No speculative prompt inference or decoder computation is performed.',0,8)
    elif kind=='versions':
        box(c,0,51,239,77,'Frozen local snapshot','Torch 2.12.1 / TileLang 0.1.13<br/>Pinned legacy ComfyUI',LIGHT)
        box(c,268,51,239,77,'Historical T4 evidence','Torch 2.11 / TileLang 0.1.15<br/>Version-bound source snapshot',LIGHT)
        label(c,'Not interchangeable qualification',139,24,10,AMBER)
        line(c,239,89,268,89,color=AMBER,arrow=False)
    else:raise ValueError(kind)


v1=json.loads((ROOT/'docs/data/fc2_quantizer_t4_v1_summary.json').read_text())
def measured(variant,field):
    r=next(x for x in v1['comparisons'] if x['rows']==8192 and x['stack']=='multi' and x['scope']=='ffn' and x['variant']==variant)
    return r['timing']['wall_ms']['median'] if field=='latency' else max(m['temporary_mib'] for m in r['memory'])

def bars(c,title,rows,height,unit,maxval=None):
    label(c,title,0,height-13,10,INK)
    vmax=maxval or max(r[1] for r in rows)
    x=127;length=291;rowheight=(height-31)/len(rows)
    for i,(name,value,col) in enumerate(rows):
        yy=height-36-i*rowheight
        label(c,name,0,yy+5,9)
        c.setFillColor(colors.HexColor('#EDF1F5'));c.roundRect(x,yy,length,17,3,fill=1,stroke=0)
        c.setFillColor(colors.HexColor(col));c.roundRect(x,yy,max(1,length*value/vmax),17,3,fill=1,stroke=0)
        label(c,f'{value:,.3f} {unit}',x+length+7,yy+4,8.5,INK)

def chart(c,kind,h):
    if kind=='fc2memory':bars(c,'Whole-FFN temporary peak, S8192 + two-entry stack',[(n,measured(v,'memory'),col) for n,v,col in [('Original','before',MUTED),('Lifetime fix','lifetime',BLUE),('Owned FP32','owned',TEAL)]],h,'MiB')
    elif kind=='fc2latency':bars(c,'Whole-FFN median synchronized wall time, same fixture',[(n,measured(v,'latency'),col) for n,v,col in [('Original','before',MUTED),('Lifetime fix','lifetime',BLUE),('Owned FP32','owned',TEAL)]],h,'ms')
    elif kind=='vaebytes':bars(c,'Checkpoint component read payload, not live device memory',[(n,b/2**20,col) for n,b,col in [('Video encoder',360747328,TEAL),('Video decoder',2450218528,BLUE),('Audio encoder',345477248,TEAL),('Audio decoder',259672288,BLUE)]],h,'MiB')
    else:raise ValueError(kind)


layout=[]
PDF=ROOT/'docs/guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf'
cv=canvas.Canvas(str(PDF),pagesize=(W,H),pageCompression=1)
cv.setTitle('MiniMax-H3 TP2: Integrated Optimization Technical Guide')
cv.setAuthor('Technical documentation - source-audited local implementation')
cv.setCreator('H3 source-audited technical documentation generator')
cv.setSubject('Attention, INT8 linear operators, TP2 transport, LoRA, residency and staged parallel VAEs')
cv.showOutline()

def footer(num,refs=''):
    cv.setStrokeColor(colors.HexColor(RULE));cv.setLineWidth(.6);cv.line(L,43,W-R,43)
    label(cv,'H3 / TP2 IMPLEMENTATION GUIDE  |  20 OCT 2026',L,28,7.4)
    cv.setFont('Semi',8);cv.drawRightString(W-R,28,f'{num:02d} / 30')
    if refs:label(cv,'Source: '+refs,L,53,7.3)

# Cover.
cv.setFillColor(colors.HexColor(INK));cv.rect(0,0,W,H,fill=1,stroke=0)
cv.setFillColor(colors.HexColor('#53D0C1'));cv.rect(L,H-82,72,5,fill=1,stroke=0)
label(cv,'MINIMAX-H3 / TWO-GPU INFERENCE',L,H-116,11,'#B9D1DF')
for text,yy in [('Integrated',H-178),('Optimization',H-225),('Technical Guide',H-272)]:label(cv,text,L,yy,37,'#FFFFFF')
# Cover paragraph is deliberately light against the dark background.
coverstyle=ParagraphStyle('cover',parent=STYLES['body'],textColor=colors.HexColor('#D1E0E9'),fontSize=12,leading=18)
p=Paragraph('An implementation-level explanation of attention, INT8 row operators, TP2 communication, LoRA, memory residency and parallel VAEs.',coverstyle);_,hh=p.wrap(CW,100);p.drawOn(cv,L,H-310-hh)
for i,(title,body) in enumerate([('TE','Group32 dense'),('DiT','Kitchen INT8'),('Video VAE','Kitchen + Q split')]):
    x=L+i*173;box(cv,x,318,161,72,title,body,PALE)
label(cv,'CURRENT SOURCE, NOT A BLENDED BENCHMARK CLAIM',L,280,10,'#53D0C1')
pp=Paragraph('Default INT32 SUM. Owned FP32 FC2 rotation. Optional sparse attention and scaled FP16 transport. Native conditioning, sampling and public VAE planning.',ParagraphStyle('cover2',parent=STYLES['body'],textColor=colors.HexColor('#D1E0E9'),fontSize=10.4,leading=15));_,hh=pp.wrap(CW,100);pp.drawOn(cv,L,259-hh)
label(cv,'AUDITED 20 OCTOBER 2026  |  30 PAGES',L,170,10,'#FFFFFF')
label(cv,'Pinned ComfyUI: d49e888586dd8ae012c0667b33466b815fee07f7',L,145,8.4,'#B9D1DF')
label(cv,'Repository: 2xT4 minimax h3',L,124,9,'#B9D1DF')
label(cv,str(ROOT).replace('\\','/'),L,105,7.4,'#B9D1DF')
label(cv,'Historical T4 results are scoped separately from the frozen local stack.',L,78,9,'#53D0C1')
cv.bookmarkPage('cover');cv.addOutlineEntry('Cover','cover',level=0);cv.showPage()

for num,pag in enumerate(pages,2):
    key=f'page{num}';cv.bookmarkPage(key);cv.addOutlineEntry(f'{num:02d}  {pag["title"]}',key,level=0)
    label(cv,pag['section'].upper(),L,H-49,8.9,TEAL)
    y=H-70
    title=Paragraph(html.escape(pag['title']),ParagraphStyle('title',fontName='Bold',fontSize=23,leading=27,textColor=colors.HexColor(INK)))
    _,hh=title.wrap(CW,100);title.drawOn(cv,L,y-hh);y-=hh+10
    y-=para(cv,pag['dek'],L,y,CW,'small')+18
    for bi,(kind,value) in enumerate(pag['blocks']):
        top=y
        if kind=='p':y-=para(cv,value,L,y,CW)+9
        elif kind=='h':
            y-=5;cv.setFont('Bold',12);cv.setFillColor(colors.HexColor(INK));cv.drawString(L,y-12,value);y-=23
        elif kind=='note':
            title,body=value
            th=Paragraph('<b>'+title+'</b><br/>'+body,STYLES['body']);_,nh=th.wrap(CW-23,1000)
            cv.setFillColor(colors.HexColor(OPTIONAL));cv.roundRect(L,y-nh-18,CW,nh+18,6,fill=1,stroke=0)
            cv.setFillColor(colors.HexColor(AMBER));cv.rect(L,y-nh-18,3,nh+18,fill=1,stroke=0)
            th.drawOn(cv,L+12,y-9-nh);y-=nh+29
        elif kind=='table':
            heads,rows,widths=value; widths=[CW*z for z in widths] if widths else [CW/len(heads)]*len(heads)
            data=[[Paragraph(html.escape(str(z)),STYLES['head']) for z in heads]]+[[Paragraph(html.escape(str(z)),STYLES['table']) for z in row] for row in rows]
            tb=Table(data,colWidths=widths,hAlign='LEFT');tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(INK)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F2F6F8'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor(RULE)),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor(RULE))]))
            _,hh=tb.wrap(CW,2000);tb.drawOn(cv,L,y-hh);y-=hh+13
        elif kind in ('diagram','chart'):
            name,hh=value;cv.saveState();cv.translate(L,y-hh);(diagram if kind=='diagram' else chart)(cv,name,hh);cv.restoreState();y-=hh+14
        elif kind=='code':
            lines=value.splitlines();hh=len(lines)*12+18
            cv.setFillColor(colors.HexColor(LIGHT));cv.roundRect(L,y-hh,CW,hh,6,fill=1,stroke=0)
            cv.setFillColor(colors.HexColor(INK));cv.setFont('Mono',8.4)
            for i,text in enumerate(lines):
                if pdfmetrics.stringWidth(text,'Mono',8.4)>CW-20:raise ValueError('Code line too wide: '+text)
                cv.drawString(L+10,y-15-i*12,text)
            y-=hh+12
        elif kind=='sources':
            for sid,rel,desc in value:
                text=f'<b>{sid}</b>  {html.escape(rel)}<br/><font color="{MUTED}">{html.escape(desc)}</font>'
                y-=para(cv,text,L,y,CW,'small')+7
        else:raise ValueError(kind)
        layout.append(dict(page=num,title=pag['title'],block=bi,type=kind,top=top,bottom=y))
        if y<70:raise ValueError(f'Page {num} overflow after {kind}, bottom={y:.1f}: {pag["title"]}')
    footer(num,pag['refs']);cv.showPage()

assert len(pages)+1==30,len(pages)+1
cv.save()
source_manifest=[]
for sid,rel,desc in SOURCES:
    path=ROOT/rel
    if not path.is_file():raise FileNotFoundError(path)
    raw=path.read_bytes();source_manifest.append(dict(id=sid,path=str(path),description=desc,sha256=hashlib.sha256(raw).hexdigest(),lines=len(raw.splitlines())))
manifest=dict(date='2026-10-20',repository=str(ROOT),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),pdf=str(PDF),pages=30,sources=source_manifest,chart_boundary='Historical physical dual-T4, S8192 two-entry LoRA stack, FFN scopes; no full pipeline claim',charts={v:dict(temporary_mib=measured(v,'memory'),wall_ms=measured(v,'latency')) for v in ['before','lifetime','owned']})
(OUT/'h3_optimization_guide_source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
(OUT/'h3_optimization_guide_layout.json').write_text(json.dumps(layout,indent=2),encoding='utf8')
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject
writer=PdfWriter(clone_from=str(PDF))
writer.add_attachment('h3_source_manifest.json',json.dumps(manifest,indent=2).encode('utf8'))
writer.root_object[NameObject('/Lang')]=TextStringObject('en-US')
temporary=PDF.with_suffix('.tmp.pdf')
with temporary.open('wb') as stream:writer.write(stream)
temporary.replace(PDF)
print(json.dumps(dict(pdf=str(PDF),pages=30,bytes=PDF.stat().st_size,minimum_body_bottom=min(x['bottom'] for x in layout)),indent=2))
