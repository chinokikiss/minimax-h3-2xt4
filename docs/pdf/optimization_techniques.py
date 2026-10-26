"""Canonical technique catalogue. Mechanisms and contracts, no measurements."""
CHAPTERS=[]
TECHNIQUES=[]


def chapter(key,title,intro,flow):
    CHAPTERS.append(dict(id=key,title=title,intro=intro,flow=flow))


def tech(id,title,method,saves,contract,sources,legacy='',status='active'):
    TECHNIQUES.append(dict(id=id,chapter=CHAPTERS[-1]['id'],title=title,status=status,
        method=method,saves=saves,contract=contract,sources=sources.split('|'),legacy_ids=legacy.split()))


chapter('SYS','Partitioning and persistent execution',
    'Keep Native conditioning, sampling and public interfaces in the parent. Partition learned computation inside persistent workers. Component ownership, collective order and resource ownership are parts of the algorithm.',
    ['Native parent','Persistent rank 0 + rank 1','Owned projections','SUM / gather','Native output'])
tech('SYS01','Component-aware tensor parallelism',
    'Split column-parallel weight output rows and row-parallel input columns. For packed QKV and gate/up tensors, split each component independently and concatenate the owned pieces in Native component order.',
    'Each rank stores and multiplies its owned matrix pieces instead of replicating the complete TE and DiT block.',
    'Replicate the required scales, norms and metadata. A half of a packed tensor is not generally the same as half of each component.',
    'patches/rank_local_int8.py|patches/tp_qwen3vl.py|patches/tp_dit.py','TP2')
tech('SYS02','Persistent serial worker sessions',
    'Create one spawned process per CUDA device and reuse its initialized modules, readers and caches across commands. Carry sequence identifiers through a serial request/acknowledgement protocol.',
    'Amortize process, CUDA and module startup and retain useful prepared weights between actual evaluations.',
    'Preserve collective ordering and propagate failures before reusing a session. Persistence does not imply concurrent sampling requests.',
    'patches/comfy_tp_runtime.py|patches/comfy_sampler.py','SERVICE')
tech('SYS03','Bounded CPU packet handoff',
    'Serialize the required tensors and nested layout metadata through the parent/worker protocol at explicit phase boundaries. Preserve dtype and PackedLayout rather than reconstructing context from tensor size.',
    'Keep Native orchestration outside GPU workers while avoiding repeated model construction.',
    'Handoff is synchronous. Do not infer communication/compute overlap from the existence of persistent processes.',
    'patches/comfy_tp_runtime.py|patches/comfy_sampler.py')
tech('SYS04','Discard unused secondary-rank replies',
    'After both ranks finish computation and synchronization, discard rank-1 nondiagnostic output before making the CPU packet. Keep acknowledgements and explicit diagnostic responses.',
    'Remove a redundant device-to-host tensor copy, CPU allocation and pipe payload.',
    'Both ranks still participate in all collectives and error handling; output suppression is not computation suppression.',
    'patches/comfy_tp_runtime.py')
tech('SYS05','Stable collective topology',
    'Use NCCL with peer-to-peer disabled and four minimum/maximum channels; retain automatic protocol selection. Run corresponding collectives in the same order on both owners.',
    'Bound the communication configuration for the two-device deployment and avoid depending on unavailable peer-copy topology.',
    'Actual distributed collectives remain necessary. Logical-rank execution on one GPU is not a replacement for this topology.',
    'patches/comfy_tp_runtime.py|patches/quantized_collectives.py','NCCL')
tech('SYS06','Policy-keyed prepared module reuse',
    'Attach an immutable operator and numerical-flow policy to prepared modules, streamed specifications and cache identities. Recreate the service when loader choices change.',
    'Reuse prepared representations without applying a second fold or serving weights prepared under a different backend or LoRA stack.',
    'Residency copies must preserve registered buffers, rank metadata and policy; avoid methods bound to the canonical CPU module.',
    'patches/h3_operator_policy.py|patches/comfy_tp_runtime.py|patches/weight_residency.py')
tech('SYS07','Architecture and selected-backend guards',
    'Select specialized kernels only for supported inference geometry and SM75. Retain the selected public Kitchen implementation for other backends, autograd, compiled calls, hooks and unsupported shapes.',
    'Concentrate specialized code on the H3 shapes it implements without replacing the global operator registry.',
    'Validate the pinned core ABI before execution. A fallback must preserve the public shape, dtype and mask interface.',
    'patches/h3_operator_dependencies.py|patches/legacy_runtime.py|patches/h3_configuration.py|custom_nodes/minimax_h3_tp2/__init__.py')
tech('SYS08','Separate hot-path execution from diagnostics',
    'Keep detailed events, tensor scans, observers and profiler capture behind explicit diagnostic entry points. Normal worker commands retain only required protocol and lifecycle accounting.',
    'Avoid tracing allocation, host synchronization and repeated scalar extraction during ordinary inference.',
    'Required acknowledgements, failures and admission decisions remain available. Diagnostics must not become an implicit production dependency.',
    'patches/diagnostic_metrics.py|patches/runtime_diagnostics.py|patches/comfy_tp_runtime.py')

tech('SYS09','Full-model backend geometry at sharded boundaries',
    'When Native attention backend priority depends on the original query size, reconstruct that size from the head shard and world size before selecting the retained backend policy.',
    'Keep tensor parallelism from accidentally changing a shape-dependent Native kernel selection.',
    'Preserve the complete model threshold and supported public attention API even though each rank owns fewer heads.',
    'patches/tp_qwen3vl.py')

chapter('LIN','INT8 projections, fusion and communication',
    'A coherent TP row operator uses a shared activation quantizer, local INT8 GEMMs, a collective on partial results and a deliberate dequantization order. Fusion removes materialized intermediates while preserving the selected arithmetic boundaries.',
    ['Rotate / activate','Shared row MAX','Owned INT8 packing','INT32 GEMM','SUM + dequantize'])
tech('LIN01','Direct Turing INT8 matrix multiplication',
    'Call the retained Kitchen CUDA/cuBLAS INT8 matrix primitive on contiguous activation codes and owned weight rows, producing an INT32 accumulator.',
    'Avoid eager per-call transposed-weight and padded-row intermediates where the scoped direct path applies.',
    'Keep the complete intended GEMM geometry, signed integer operands and original activation/weight scales.',
    'patches/optimized_tp.py|patches/parallel_int8.py','GEMM')
tech('LIN02','Global-row activation scale across owners',
    'Compute local maxima over the owned rotated activation slice, then MAX-reduce them. Quantize both owners with the same full-row maximum before local GEMMs.',
    'Avoid gathering the entire floating activation merely to reproduce a full-row quantizer.',
    'Independent rank-local scales cannot be followed by a single common INT32 SUM/dequantization. Preserve the shared scale contract.',
    'patches/optimized_tp.py|patches/tilelang_row_quant.py','SCALEMAX')
tech('LIN03','External-maximum quantization',
    'Pass the already reduced full-row maximum directly into the owned-slice quantizer. Apply the pinned division, nearest rounding, clamp and scale floor in the packing kernel.',
    'Remove an artificial maximum sentinel, extra scan, temporary copy and restoration stage.',
    'Match the selected quantizer rounding and scale-floor order, including zero rows; algebraically similar division is not automatically bit-equivalent.',
    'patches/tilelang_row_quant.py','EXTERNALMAX')
tech('LIN04','Strided maxima and direct owned reads',
    'Read each owner from the original packed or strided activation and reduce its absolute maximum in place. Use explicit source strides and ownership offsets.',
    'Remove full-width contiguous copies and intermediate owned-slice materialization.',
    'Handle component packing and padding separately. Source views must remain live until the consuming kernel finishes.',
    'patches/tilelang_row_quant.py','STRIDEDMAX')
tech('LIN05','Positive-zero ownership padding',
    'Write the owner codes directly to aligned GEMM storage and initialize every unowned/padded position as positive zero. Define padding from ConvRot groups and packed component boundaries.',
    'Fuse ownership packing, layout conversion and padding into one destination.',
    'Signed zero and uninitialized tail bytes can change raw-byte equivalence or subsequent transforms. Read only valid source channels.',
    'patches/tilelang_row_quant.py|patches/tilelang_swiglu_pad.py','PACK')
tech('LIN06','SwiGLU plus ownership-padding fusion',
    'Compute silu(gate) * up in the selected floating island and store directly into the padded/owned destination. Support separate TE gate/up tensors and packed DiT branch inputs.',
    'Remove eager activation/product tensors and a later ownership-padding copy.',
    'Promoted FP16-flow preparation converts operands to FP32 before the nonlinear product. Converting an already overflowed product is too late.',
    'patches/tilelang_swiglu_pad.py|patches/h3_operator_policy.py','SWIGLU')
tech('LIN07','Owned FP32 ConvRot preparation',
    'Compute the supported rank-owned SwiGLU/ConvRot channels in FP32 and feed the common global-MAX quantization procedure. Retain complete transform groups across ownership boundaries.',
    'Avoid a full-width FP32 rotated activation while retaining the required local slice.',
    'This is the owned_fp32 alternative. Its grouping and input layout guards differ from the newer regular-H4 fused producer.',
    'patches/tilelang_swiglu_pad.py','OWNED',status='alternative')
tech('LIN08','Stored fused FP32 SwiGLU and regular H4 rotation',
    'Fuse branch activation and regular ConvRot256 preparation in FP32, retain the rotated FP32 intermediate, and produce owned INT8 codes using the shared row maximum.',
    'Collapse separate activation, rotation and packing launches while reusing the rotated result in subsequent preparation.',
    'Use the pinned H4/ConvRot order and explicit rounding boundaries. This is fused_stored, the default eligible FC2/down preparation.',
    'patches/tilelang_swiglu_quant.py|patches/h3_operator_policy.py')
tech('LIN09','Recomputed fused preparation',
    'Calculate the same FP32 branch/rotation once for maxima and recompute it for the final packing rather than retaining the complete rotated activation.',
    'Trade additional arithmetic and reads for a shorter/smaller FP32 intermediate lifetime.',
    'Keep the stored variant arithmetic contract. Recompute is an explicit memory alternative, not an automatic faster path.',
    'patches/tilelang_swiglu_quant.py','',status='alternative')
tech('LIN10','Exact INT32 partial SUM',
    'SUM the local INT32 GEMM accumulators before applying the common row scale and output-channel weight scale, then bias and output conversion in the required order.',
    'Preserve the integer reduction contract without gathering full activations.',
    'INT32 SUM is the explicit accuracy alternative to scaled FP16 communication. The native fallback reduction is fixed, not a second hidden policy switch.',
    'patches/parallel_int8.py|patches/optimized_tp.py|patches/tilelang_swiglu_pad.py','INT32')
tech('LIN11','Power-of-two FP16 partial transport',
    'Choose an exponent from the local-K accumulator bound and world size. Encode partial INT32 values as FP16 after power-of-two scaling, AllReduce SUM, and reconstruct/dequantize in FP32.',
    'Halve partial-sum payload size relative to INT32 and fuse reconstruction with the final epilogue.',
    'FP16 encoding and SUM are approximate, especially under cancellation. Bound both rank contributions below FP16 maximum; scaling changes range, not mantissa bits.',
    'patches/tilelang_transport_p2.py|patches/tilelang_transport_tradeoffs.py','FP16SUM')
tech('LIN12','Fused row dequantization epilogue',
    'Combine transport reconstruction, shared row scale, weight scale, bias and final dtype store in one kernel after the collective.',
    'Remove full-size reconstructed and separately dequantized floating temporaries.',
    'Keep explicit FP32 operation order and the final activation-store boundary; apply online LoRA at its defined post-base boundary.',
    'patches/tilelang_transport_p2.py|patches/parallel_int8.py','DEQUANT')
tech('LIN13','Fixed-size feature gather fallback',
    'Gather equal-size rank-owned feature pieces into fixed receive storage and restore Native order before the retained fallback operator.',
    'Provide a predictable fallback without variable-size object transport or repeated gather-list construction.',
    'The gathered floating path is distinct from shared-scale local INT32 partials. It remains a fallback, not an INT8 communication claim.',
    'patches/optimized_tp.py|patches/parallel_int8.py','GATHER',status='fallback')
tech('LIN14','Single-destination chunked column dequantization',
    'Allocate the final output once. Process bounded accumulator row chunks, form row_scale * weight_scale in FP32, multiply the converted accumulator in place, and copy to final rows.',
    'Remove a list of output chunks, their final concatenation and a redundant full-size scaled intermediate.',
    'Keep the two FP32 multiplication boundaries in this order. Chunking the epilogue does not change GEMM dimensions.',
    'patches/dit_column_linear.py')
tech('LIN15','Direct column GEMM with ordered TileLang epilogue',
    'Reuse eager activation rotation/quantization, call the direct INT8 GEMM for aligned SM75 DiT QKV/FC1 weights, then apply an epilogue with explicit rounded FP32 products and one output store.',
    'Remove eager weight transpose/padding and broadcast dequantization scratch on the supported column shapes.',
    'Require inference, the selected eager backend and the guarded DiT geometry. Preserve the public backend elsewhere; epilogue-only tilelang_exact remains a private comparator.',
    'patches/dit_column_linear.py|patches/tilelang_column_epilogue.py','COLUMN')
tech('LIN16','Shared TE input rotation and quantization',
    'Prepare a common normalized input once for independent Q/K/V column GEMMs, and similarly once for gate/up. Reuse only deterministic activation codes and row scales.',
    'Remove repeated ConvRot and row-quantization work without merging projection GEMMs.',
    'Keep hooks, subclasses, overridden forwards, autograd and compiled calls on their original module paths. Preserve each projection weight and GEMM geometry.',
    'patches/dit_column_linear.py|patches/tp_qwen3vl.py')
tech('LIN17','Bounded complete column-row scheduling',
    'Partition large inference activation rows into bounded calls around the public linear API; write each completed block to its final output position.',
    'Bound per-call accumulator/workspace residency for long sequences.',
    'A changed GEMM row geometry can change floating/backend behavior. This experimental schedule is distinct from epilogue-only chunking.',
    'benchmarks/bounded_column_int8.py','',status='experimental')

chapter('NUM','Numerical storage and lifetime management',
    'Reduce activation bytes with explicit storage boundaries. Preserve FP32 statistics and nonlinear preparation where range or rounding requires them. Residual gauges are coordinate changes on privately prepared modules, not checkpoint requantization.',
    ['FP32 stable island','Scale branch / residual','FP16 storage','INT8 projection','FP32 public boundary'])
tech('NUM01','FP16 residual storage with a power-of-two gauge',
    'Represent the residual stream as y = x / c, with c = 2^e and default e = 6 for TE/DiT. Divide branch outputs before storage and use epsilon / c^2 for gauge-aware normalization.',
    'Reduce residual/activation storage and movement while preserving range headroom.',
    'Power-of-two scaling does not add precision. Vision/Video gauges remain independently specified; legacy flow retains the original coordinate system.',
    'patches/h3_tensor_flow.py|patches/comfy_tp_runtime.py')
tech('NUM02','Private scale and bias folding',
    'Fold the scalar residual divisor into private FP32 INT8 weight scales or dense weights and biases once during module preparation. Adjust LoRA post-B coefficients consistently.',
    'Avoid separate runtime branch-rescale tensors and repeated multiplications.',
    'Never mutate checkpoint bytes or apply a fold twice. Nonuniform gains cannot be commuted through ConvRot as a scalar fold.',
    'patches/h3_tensor_flow.py')
tech('NUM03','FP32 normalization and modulation islands',
    'Compute normalization statistics, gains, epsilon, AdaLN modulation and gated additions in FP32, then cast at the declared activation storage boundary.',
    'Allow compact FP16 storage around operations whose reductions and products need wider range.',
    'Keep sampler state, time curves, statistics and final DiT heads in their FP32 islands. Restore public conditioning in its unscaled representation.',
    'patches/h3_tensor_flow.py|patches/tp_dit.py|patches/tp_qwen3vl.py')
tech('NUM04','FP32 online adapter products before scaling',
    'For FP16 tensor flow, perform the A/B products in FP32 and apply the adapter strength after B before the final activation store.',
    'Prevent an overflowing half-precision B product from destroying information before a small post-B coefficient is applied.',
    'Preserve adapter order and strength placement; a small scale cannot recover a product that already overflowed.',
    'patches/h3_lora.py|patches/h3_tensor_flow.py')
tech('NUM05','Prompt release of attention and MLP intermediates',
    'Release caller references to Q/K/V, attention buffers, branch tensors, LoRA output views and old residual owners immediately after their last consuming operation.',
    'Reduce overlap between attention, FFN, residual and next-block activation lifetimes.',
    'Deleting a local name is effective only if callers and views no longer retain the storage. Respect stream/event lifetime before reuse.',
    'patches/tp_qwen3vl.py|patches/tp_dit.py|patches/h3_lora.py','DEL')
tech('NUM06','Retained fused norms and rotary embedding',
    'Use the pinned Kitchen RMSNorm/RoPE helpers at the established Native boundary, including compact head layouts and required gain casts.',
    'Fuse normalization/rotary work and avoid otherwise separate layout and elementwise passes.',
    'Preserve Native residual, normalization, projection and rotation ordering. This does not authorize a replacement of the frozen legacy attention API.',
    'patches/tp_dit.py|patches/tp_qwen3vl.py|patches/h3_operator_policy.py','NATIVEFUSE')

tech('NUM07','BF16 reference activation storage',
    'Use the numerical-flow wrapper with BF16 storage while retaining the same explicit FP32 stability islands and private residual-gauge preparation.',
    'Provide a 16-bit storage/reference representation with a different range/precision tradeoff from FP16.',
    'This is reference-only; deployment RuntimeConfig rejects bf16_reference on the T4 path. Storage support does not imply native BF16 Tensor Core arithmetic on SM75.',
    'patches/h3_tensor_flow.py|patches/comfy_tp_runtime.py','',status='experimental')

chapter('ATT','Dense attention and preparation techniques',
    'Dense attention techniques change operand preparation, reuse, layout or scheduling. State the quantization and output-store contract before discussing fusion: INT8, Group32 and Native SDPA are distinct numerical paths.',
    ['Q/K norm + RoPE','Quantize / pack','QK + online softmax','Probability x V','BND output'])
tech('ATT01','Pinned Kitchen INT8 dense attention',
    'Rotate/quantize Q/K, use packed per-thread scales and an optional representative-key shift, and accumulate attention with the retained signed-INT8 Q/K and unsigned-probability/INT8 V primitives.',
    'Use Tensor Core integer products and online softmax without materializing a token-by-token score matrix.',
    'Preserve the installed quantizer and scale packing. FP32 input uses BF16 output storage followed by an FP32 cast, not full FP32 numerical precision.',
    'patches/h3_operator_policy.py|experiments/kitchen_attention/attention.py|experiments/kitchen_attention/native/vendor/attn_utils.cuh','KITCHEN')
tech('ATT02','Direct masked Kitchen TE with compact GQA',
    'Forward the original boolean/additive mask and compact KV heads directly to pinned Kitchen, then expose the BND output expected by TE.',
    'Avoid KV-head replication, full-mask reconstruction, host mask classification and a separate native-extension build in default dispatch.',
    'Score masking and raw-input prefix invariance are separate contracts: shared data-dependent quantization can couple earlier representations to future inputs.',
    'patches/kitchen_te_attention.py|patches/h3_operator_policy.py')
tech('ATT03','Native SDPA for the token refiner',
    'Keep the Native refiner projections, normalization, rotary embedding and SDPA while wrapping the module so policy survives residency copies.',
    'Avoid applying sparse/quantized main-DiT attention preparation to the small refiner sequence.',
    'Refiner tokens are not sparsified. Use a registered module wrapper rather than a closure bound to a CPU canonical object.',
    'patches/h3_operator_policy.py','REFINER')
tech('ATT04','Group32 QK with FP16 PV',
    'Partition each head dimension into 32-channel groups. Compute an INT32 QK product per group, apply the corresponding FP32 scales, and use FP16 probabilities/values for PV.',
    'Use narrower quantization groups and Tensor Core QK while retaining a wider PV path.',
    'This is an explicit numerical alternative to Kitchen per-thread packing. Preserve the centering and rounding selected by its preparation.',
    'patches/tilelang_grouped_qk.py|patches/tilelang_grouped_tp.py|patches/tilelang_grouped_tp_kitchen.py','DENSE',status='alternative')
tech('ATT05','Mask-aware Group32 TE kernel',
    'Broadcast compact 2D/3D/4D masks within online softmax and map query heads to shared KV heads without expanding K/V. Apply additive bias in the kernel score domain.',
    'Avoid repeated KV copies and full expanded mask storage for GQA prefill.',
    'Preserve empty-row handling and later-leaf accumulation after masked leaves. tilelang_group32 is an explicit override, not the current TE default.',
    'patches/tilelang_group32_masked.py','TEATTN',status='alternative')
tech('ATT06','Fused stride-aware cast, pad and quantize',
    'Read packed/strided Q/K/V directly and combine input conversion, ragged-tail padding, Group32 scale reduction and code production. Keep only mode-required representations.',
    'Remove separate contiguous/cast/pad passes and avoid duplicate padded floating tensors.',
    'Use the selected input FP16 rounding boundary, nearest rounding and scale floor. Arbitrary layout fusion must retain source strides and valid tails.',
    'patches/tilelang_grouped_tp_kitchen_v2.py|patches/tilelang_grouped_fused.py','PREP',status='alternative')
tech('ATT07','Prepared dense-only fast path',
    'Build an internally owned all-dense route object and prepare only dense operands. Omit pooled V, V8 and LSE work when the consumer does not read them.',
    'Remove unused routing metadata and repeated GPU-to-CPU validation for internally generated immutable routes.',
    'Only the exact internally generated route object may use the prevalidated path. Older GQA preparation may explicitly expand heads and must not be confused with compact GQA.',
    'patches/tilelang_dense_preparation.py|patches/tilelang_sage_preparation.py','',status='experimental')
tech('ATT08','Bounded immutable geometry reuse',
    'Cache counts, padding geometry and all-dense metadata by device, stream and tensor geometry; keep content-dependent Q/K/V and routes outside immutable caches.',
    'Remove repeated allocation and geometry kernel launches on recurring shapes.',
    'Bound cache size and distinguish stream ownership. Reusing immutable metadata does not make activation-dependent summaries reusable.',
    'patches/tilelang_grouped_tp_optimized.py|patches/tilelang_dense_preparation.py','GEOMETRY',status='alternative')
tech('ATT09','Query-only and full-QK reuse',
    'Retain query fragments across key tiles; the both-cache candidate additionally retains key fragments when its kernel mapping permits. The selected optional dense path uses query-only reuse.',
    'Reduce repeated shared/global operand loads at the cost of longer register lifetimes.',
    'Cache choice interacts with head dimension, query tile, scale fragments and occupancy. A both-cache flag is a separate candidate, not implied by query-only selection.',
    'patches/tilelang_grouped_tp_kitchen_v2.py|patches/tilelang_attention_tradeoffs.py','QCACHE',status='alternative')
tech('ATT10','Warp, query-tile and scale-placement specialization',
    'Specialize BQ32/BQ64 schedules, 128/256-thread blocks, row/square warp mappings, register or staged scales and row-score fragments for fixed geometry.',
    'Trade reuse, parallel rows, register pressure and staging traffic without changing the intended attention equation.',
    'Every specialization must keep valid tail/mask handling. Exposed candidate flags do not mean the production caller selects all of them.',
    'patches/tilelang_grouped_tp_kitchen_v2.py|patches/tilelang_attention_tradeoffs.py','',status='experimental')
tech('ATT11','Direct BND and dtype epilogue stores',
    'Map output fragments directly to batch-token-flattened-head storage and perform the declared dtype conversion in the kernel epilogue.',
    'Remove a BHND-to-BND transpose/contiguous copy and a later cast.',
    'Retain an intermediate FP16 rounding boundary when that is part of the original path. The finite-output epilogue is a separate guarded candidate, not permission to hide nonfinite values.',
    'patches/tilelang_group32_masked.py|patches/tilelang_attention_tradeoffs.py','BND',status='alternative')
tech('ATT12','Sage-style block scales and centering alternatives',
    'Quantize Q and K with independently selected block sizes, token-mean or leaf-mean centering, and explicit half-away or nearest-even rounding variants.',
    'Reduce preparation cost or adapt the quantizer granularity while retaining INT8 QK and floating PV.',
    'Different group sizes, centering means and rounding rules define different approximations. Ragged leaf means must use actual counts.',
    'patches/tilelang_sage_qk.py|patches/tilelang_sage_preparation.py','',status='experimental')
tech('ATT13','Prequantize then release floating operands',
    'Separate the pinned Q/K/V quantizer from its attention consumer. Retain packed integer operands, scales, mask and launch metadata while releasing floating inputs when their callers no longer need them.',
    'Shorten raw Q/K/V lifetimes and expose prepared-operand reuse without rerunning quantization.',
    'The snapshot owns a mask representation and is immutable with respect to input/mask changes. Storage/API dtype is not a promise of full FP32 precision.',
    'experiments/kitchen_attention/attention.py|experiments/kitchen_attention/te_precision.py','',status='experimental')
tech('ATT14','Implicit causal index masking',
    'Intersect key traversal with key_index <= query_index + query_offset, avoiding a materialized token-pair triangle in the explicit native kernel.',
    'Remove full causal-mask storage and mask loads in supported prefill geometry.',
    'Keep offsets, ragged bounds and GQA mapping explicit. Implicit score masking does not make the preserved global quantizer prefix invariant.',
    'experiments/kitchen_attention/native/kitchen_sm75.cu|experiments/kitchen_attention/attention.py','',status='experimental')
tech('ATT15','Kitchen query64 and cached-Q schedules',
    'Derive an SM75 extension from the retained Kitchen source with a smaller query tile or a query128 schedule retaining Q fragments in registers.',
    'Change CTA parallelism and repeated operand loading while keeping quantization and core integer arithmetic intact.',
    'Keep the original schedule as the control. Cached Q requires the supported D128/key128 geometry; register/spill checks guide specialization selection.',
    'experiments/kitchen_attention/derive.py|experiments/kitchen_attention/native/kitchen_sm75.cu','',status='experimental')
tech('ATT16','n8 probability-value scheduling',
    'Dispatch to the retained compute_int8_sv_pipelined_n8 helper for its supported D128/key128 tile. Shorten live V/partial fragments while preserving the four integer products and FP32 FMA order.',
    'Reduce live fragment pressure and change instruction scheduling without replacing the probability quantizer.',
    'Use only supported warp/tile geometry. Spill elimination alone is not a performance guarantee; unsupported geometries retain the original helper.',
    'experiments/kitchen_attention/native/h3_dense_utils.cuh|experiments/kitchen_attention/derive.py','',status='experimental')
tech('ATT17','Guarded narrow tensor addressing',
    'Use uint32 tensor offsets for supported dense specializations only after checking the batch/head/length/dimension product against the address range in Python and CUDA dispatch.',
    'Reduce wide pointer-offset arithmetic and associated register lifetimes.',
    'Reject overflow before allocation/launch and retain the uint64 implementation. Narrow addressing is not valid merely because an individual sequence length is small.',
    'experiments/kitchen_attention/attention.py|experiments/kitchen_attention/native/kitchen_sm75.cu','',status='experimental')
tech('ATT18','Explicit fixed-shape CUDA Graph replay',
    'Capture the unchanged Kitchen operator and BND conversion after side-stream warmup. Copy current inputs into persistent buffers for each replay and clone the output for independent ownership.',
    'Amortize repeated Python/driver launch setup for a fixed sequence of kernels.',
    'Enforce creating-stream, dtype and geometry identity; retain graph-private storage and count input/output copies and capture lifetime. No production graph cache is installed.',
    'experiments/kitchen_attention/graph.py','',status='experimental')

tech('ATT19','Mutation-aware causal mask recognition cache',
    'Recognize only an exact causal triangle, then cache its identity and mutation version using weak references. Recheck versionless inference tensors and reject padding/custom bias from this shortcut.',
    'Amortize repeated classification when a normal immutable mask is explicitly routed to the native causal experiment.',
    'The default TE path bypasses recognition. Classification can synchronize the host, and inference tensors cannot safely use a mutation-version cache.',
    'patches/kitchen_te_attention.py','',status='experimental')
tech('ATT20','Fingerprint-keyed native build reuse',
    'Hash the retained native/Python build inputs, serialize CPU compilation with a bounded exclusive lock, and reuse a binary only after source, compiler, architecture and artifact-hash validation.',
    'Avoid redundant compilation by multiple workers and accidental reuse of a stale experimental binary.',
    'Compilation is explicit and installs no dependencies. Imports/default TE dispatch do not compile; a foreign build lock is never removed by a waiter.',
    'patches/kitchen_te_attention.py|experiments/kitchen_attention/build.py','',status='experimental')
tech('ATT21','Group32 and Group64 scale-granularity alternatives',
    'Choose the channel-group width for independent Q/K scales and restore each group product scale before summing logits. Keep the PV path and input/output rounding boundaries explicit.',
    'Trade quantization-range locality against the number of group products, scale loads and fragments.',
    'Different group widths are separate approximations. The selected Group32 path must not be relabeled as a Group64 or whole-head result.',
    'patches/tilelang_grouped_qk.py|benchmarks/attention_grouped_study.py','',status='experimental')
tech('ATT22','Cast/pad centering preparation',
    'Fuse the FP16 input cast and ragged padding stage used to build key leaf means, then feed the selected mean/centering recipe into operand quantization.',
    'Remove a separate conversion/padding launch and duplicate preparation storage.',
    'Preserve actual leaf counts and the recipe distinction between unweighted leaf means and token-weighted means. castpad is an explicit preparation variant.',
    'patches/tilelang_grouped_tp_kitchen_v2.py','',status='experimental')

chapter('SPR','Sparse routing and pooled attention',
    'Route blocks of tokens, preserve protected conditioning, and explicitly define omitted attention. Sparse dropping and scalar pooling implement different approximation equations. Scheduling and preparation are independent optimization dimensions.',
    ['Leaf means + counts','Protected routes','Fine selected blocks','Optional omitted centroids','Online normalize + merge'])
tech('SPR01','Leaf64 centroid routing',
    'Compute query/key means over 64-token leaves, account for ragged counts, center keys and compare centroid scores to the selected threshold. Retain first/global and adjacent leaves as mandatory routes.',
    'Replace token-pair routing decisions with a block-sized route matrix and skip fine QK/PV for omitted blocks.',
    'Keep the exact centering and threshold recipe for the selected variant. A routing approximation changes which token interactions are computed.',
    'patches/tilelang_grouped_tp_optimized.py|patches/tilelang_sparse_attention.py')
tech('SPR02','PackedLayout-derived protected prefix',
    'Read the current Native packed segment layout and protect every complete leaf intersecting the prefix before the final generated-video span.',
    'Apply sparsity to the intended video region while retaining text, references, conditioning and generated audio interactions.',
    'Reject stale/missing layouts or ambiguous final video spans. Do not substitute a fixed historical prefix token count.',
    'patches/h3_operator_policy.py','PREFIX',status='alternative')
tech('SPR03','Mode-specific preparation and finite tau0 shortcut',
    'For drop mode, omit pooled/V8/LSE representations. For scalar mode, prepare its required summaries. Use the tau0 algebraic shortcut only when centered means and finite inputs satisfy its assumptions.',
    'Remove preparation and threshold work the chosen sparse consumer cannot use.',
    'The shortcut is not valid for arbitrary nonfinite tensors or another centering convention. Keep dense-prefix and mandatory-route rules.',
    'patches/tilelang_grouped_tp_optimized.py|patches/tilelang_sol_drop_fused.py','PRUNE',status='alternative')
tech('SPR04','Sparse dropping',
    'Traverse selected fine leaves and update online softmax/output only from their tokens. Renormalize over the retained interactions.',
    'Eliminate fine QK and PV products for omitted leaves.',
    'Omitted mass is discarded. This is not numerically equivalent to dense attention or count-weighted pooling.',
    'patches/tilelang_grouped_tp_optimized.py|patches/h3_operator_policy.py','DROP',status='alternative')
tech('SPR05','Scalar pooling of omitted leaves',
    'Represent each omitted leaf by a mean key/value and weight its exponential score by the valid token count. Merge fine and coarse mass in a shared normalization state.',
    'Replace many omitted-token products with one centroid contribution while retaining an approximation to their attention mass.',
    'Use actual ragged counts and omitted-only contributions. Selected leaves must not be counted again as centroids.',
    'patches/tilelang_scalar_pool.py|patches/tilelang_grouped_tp_optimized.py','SCALAR',status='alternative')
tech('SPR06','Capped omitted-centroid score workspace',
    'Precompute omitted centroid scores only when B * H * padded_queries * leaf_count * 4 fits the configured cap. Otherwise calculate coarse contributions inline.',
    'Move repeated scalar-score work out of the fine loop without permitting an unbounded query-by-centroid allocation.',
    'A zero cap selects inline processing, not disabled sparsity. Workspace geometry includes all head owners and padding.',
    'patches/tilelang_attention_p2.py|patches/h3_operator_policy.py','',status='alternative')
tech('SPR07','GPU ballot route compaction',
    'Convert block masks into sorted selected-key leaf indices with per-query counts, using GPU ballot/prefix operations. Traverse only initialized active list slots.',
    'Avoid scanning rejected leaves inside every fine attention tile.',
    'Keep ordering, empty-row behavior and list capacity explicit. Unused index slots are unspecified and must never be consumed.',
    'experiments/kitchen_attention/native/kitchen_sm75.cu|patches/tilelang_attention_tradeoffs.py','',status='experimental')
tech('SPR08','Native Kitchen routed leaf execution',
    'Use compact selected-leaf lists with the retained Kitchen quantization and INT8 QK/PV primitives. Intersect explicit routes with token causal bounds when requested.',
    'Skip omitted native QK/PV products while reusing Kitchen operand packing.',
    'Routing changes scheduling to supported query64/key64 geometry. Noncausal content routes and explicit causal routes have separate contracts.',
    'experiments/kitchen_attention/native/routed_loop.cuh|experiments/kitchen_attention/attention.py','',status='experimental')
tech('SPR09','Pooled centroids in quantized operand space',
    'Build omitted K/V centroids after the retained rotation, centering and dequantization so fine and coarse scores share the same key-anchor shift.',
    'Avoid mixing incompatible score origins when combining native INT8 fine attention and FP16 coarse attention.',
    'This is a different approximation from older Group32 scalar pooling. Coarse means and fine output rounding remain declared numerical boundaries.',
    'experiments/kitchen_attention/native/kitchen_sm75.cu|experiments/kitchen_attention/attention.py','',status='experimental')
tech('SPR10','Tiled coarse attention and fused LSE merge',
    'Batch centroid contributions, retain FP32 maximum/denominator state, and merge a fine output/LSE with coarse count-weighted attention directly into the final output tile.',
    'Avoid a full Q-half copy, full query-by-centroid score tensor, separate coarse output and standalone merge tensor.',
    'Fine and coarse normalization must use a common score shift. Density-one routing reduces to the fine path; partially causal centroids are rejected.',
    'experiments/kitchen_attention/native/kitchen_sm75.cu|experiments/kitchen_attention/attention.py','',status='experimental')
tech('SPR11','INT8 PV and compact-drop TileLang alternatives',
    'Quantize V per leaf/channel and quantize probabilities for an integer PV path; optionally compact drop routes before the existing fine kernel.',
    'Trade FP16 PV traffic/work for integer operands and shorten selected-block traversal.',
    'Probability/V quantization changes the approximation. These flags are retained experiments and are not selected by current integrated Group32 sparse callers.',
    'patches/tilelang_attention_tradeoffs.py|patches/tilelang_sparse_attention.py','',status='experimental')
tech('SPR12','Legacy Sol drop preparation fusion',
    'Fuse preparation for the retained Sol centering/nearest-even recipe, omit unused V8/coarse/LSE arrays, and bind the prepared geometry to its routes.',
    'Remove legacy drop-only preparation stages without importing scalar-pooling work.',
    'Preserve this older method own quantization and route equation; it is not interchangeable with the Group32 or native Kitchen sparse recipes.',
    'patches/tilelang_sol_drop_fused.py|benchmarks/attention_sol_drop_study.py','',status='experimental')

chapter('LOR','Online adapter ownership and arithmetic',
    'For delta = scale * B(Ax), preserve Native adapter order and the post-B scale. Reduce work by changing storage ownership or communication only where the selected numerical contract permits it.',
    ['Common activation','Full or owned A','Optional FP32 hidden SUM','Owned B output','Post-B scale + ordered add'])
tech('LOR01','Native ordered adapter application',
    'Apply each adapter own A/B products, strength and addition in source order. Activate SwiGLU before gathering the adapter input when the Native operation uses activated features.',
    'Provide a stable base for storage/ownership optimizations without rebuilding full dense delta weights online.',
    'Do not pre-scale B or concatenate adapters implicitly: those change floating GEMM/reduction boundaries.',
    'patches/h3_lora.py|patches/offline_lora_reference.py','NATIVEORDER')
tech('LOR02','Single rank-major input receive buffer',
    'Gather activated rank-owned inputs into one receive storage, then expose the Native feature order to the full-A computation.',
    'Avoid a list of per-rank receive tensors, a concatenation copy and simultaneous duplicate gathered inputs.',
    'Packed gate/up ownership must be reordered component-wise. Keep the backing storage alive only until all views are consumed.',
    'patches/h3_lora.py','LORASTORE')
tech('LOR03','Owned adapter output views',
    'Scale and add only the rank-owned B output rows, then release the view and its full backing product before the next adapter.',
    'Avoid full-width delta scaling/add buffers and unintended lifetime overlap between adapters.',
    'A narrow view can retain a full allocation. Explicitly drop both caller and internal references after the ordered add.',
    'patches/h3_lora.py')
tech('LOR04','Selected-B FC1 computation',
    'Load and multiply only the owner half of each gate/up B component for the qualified ordinary DiT FC1 rank128 adapter geometry.',
    'Remove unowned FC1 B factor storage and unowned output computation.',
    'Retain full A and post-B scale/order. Wrong geometry, other adapter ranks and refiner targets use full B.',
    'patches/h3_lora.py','SELECTB')
tech('LOR05','Component-aware selected-B QKV',
    'Pack each owner rows independently from all three QKV B components for the guarded rank128 DiT target. Carry packed representation sizes into admission accounting.',
    'Avoid materializing the other owner Q/K/V delta outputs and B rows.',
    'Keep separate adapter GEMMs and full A. Selection is an H3 shape guard, not a universal LoRA slicing rule.',
    'patches/h3_lora.py|patches/comfy_tp_runtime.py')
tech('LOR06','FP32 Row-A partial hidden reduction',
    'Read only owned A columns, compute A_r x_r locally, FP32-SUM the small low-rank hidden state, and apply full B followed by the original strength.',
    'Replace full-width floating input gathering with communication proportional to adapter rank.',
    'The changed reduction order is approximate relative to one full A GEMM. Keep the hidden SUM in FP32 independently of base partial-sum transport.',
    'patches/h3_lora.py','ROWPARTIAL')
tech('LOR07','Bounded hidden-state staging',
    'Limit live gathered/low-rank temporary rows, reuse only the selected staging representation and release per-adapter outputs before advancing.',
    'Bound Row-A partial and adapter output peaks for long sequences or multiple adapters.',
    'Account for real backing tensors, not only view sizes. Preserve adapter ordering across staged work.',
    'patches/h3_lora.py|docs/reports/ROW_PARTIAL_MEMORY.md')
tech('LOR08','Policy-aware factor residency and identity',
    'Validate canonical targets, factor shapes, alpha and finite strengths once; include adapter source identity, owner representation and selected modes in cache keys.',
    'Cache only the factor rows/columns required by the selected ownership method and reuse their prepared representation.',
    'Detect file mutation and reject unconsumed or duplicate targets. Zero-strength adapters need no factor computation.',
    'patches/h3_lora.py|patches/weight_residency.py')
tech('LOR09','Registered-buffer module copies',
    'Represent online adapters and refiner wrappers as modules with registered tensors and immutable rank metadata, copied through the residency mapper.',
    'Reuse canonical CPU preparation and controlled H2D staging without reconstructing adapter closures for each block.',
    'A bound method referencing the CPU owner can defeat GPU residency. Copy parameters, buffers and ownership consistently.',
    'patches/h3_lora.py|patches/weight_residency.py')

chapter('IO','Offline baking and checkpoint access',
    'Move repeatable work to preparation and read only admitted payloads. Preserve checkpoint representation and immutable identity while replacing full-file reads, repeated validation or repeated adapter merging with bounded owned operations.',
    ['Immutable descriptor','Owned tensor ranges','Prepare / bake once','Verified CPU backing','Bounded H2D'])
tech('IO01','Whole-layer offline adapter baking',
    'Dequantize a complete layer, apply ordered LoRA updates in the reference domain and quantize once into the normal checkpoint representation. Assign whole layers across up to two GPUs.',
    'Eliminate recurring online A/B GEMMs and adapter communication during inference.',
    'Do not split a matrix in a way that changes its quantizer domain. Publish atomically and identify the complete recipe and source stack.',
    'patches/offline_lora.py|patches/offline_lora_reference.py','BAKE')
tech('IO02','Bounded offline merging into TP2 bundles',
    'For a rank-sharded bundle, reconstruct the full matrix required by the merge, apply the ordered adapter update, requantize, then split the result into the owned component layout.',
    'Bound host/device preparation by matrix/block instead of creating an entire merged model in RAM.',
    'Internal bundle merging must be explicitly enabled for descriptor callers; a path suffix alone cannot prove a checkpoint was already baked.',
    'patches/tp2_bundle.py|patches/comfy_tp_runtime.py')
tech('IO03','Persistent disk-baked rank shards',
    'Bake the owned outputs to immutable rank-block safetensors and publish a descriptor/state manifest after all required blocks are complete. Reopen them through DiskBakedReader.',
    'Avoid repeating offline merges on each request, process start or unchanged adapter stack.',
    'Verify recipe, base identity, adapter strengths, auxiliary tensors and complete state. Partial writes must never become a complete published descriptor.',
    'patches/tp2_disk_bake.py')
tech('IO04','Cached proof of immutable baked state',
    'Validate state schema on every load, but memoize a full file/state proof for an unchanged shard/manifest identity. Recheck the admitted set at block boundaries and addressed files on reads.',
    'Avoid repeated expensive full integrity work for unchanged persisted shards.',
    'Cache proof identities and reader metadata, not merged tensors. Detect mutations and require reload instead of trusting stale evidence.',
    'patches/tp2_disk_bake.py|patches/tp2_bundle.py')
tech('IO05','Positional owned-range reads',
    'Translate safetensors metadata and row/column ownership into exact file byte ranges. Use positional pread where available, otherwise a scoped seek/readinto implementation.',
    'Read the owner tensor pieces without loading full matrices or moving a shared file cursor between concurrent readers.',
    'Validate file identity, bounds, dtype and tensor sizes before constructing views. Platform-specific readers preserve the same bytes.',
    'patches/windowed_tp.py|patches/rank_local_int8.py','READ')
tech('IO06','Contiguous verified rank-block export',
    'Export prepared rank-owned weights, replicated scales and required metadata into contiguous safetensors shards plus a verified descriptor.',
    'Replace many strided source-range reads and repeated ownership reconstruction with block-local contiguous payloads.',
    'Retain component-aware ownership and source hashes. Exports change storage layout, not tensor values or upstream checkpoints.',
    'patches/rank_local_int8.py|patches/tp2_bundle.py','SHARDS',status='alternative')
tech('IO07','Bounded full-row and contiguous column reads',
    'Allow a bounded full-row scratch read before slicing owned columns. Read full-width column projections contiguously without the row-stride path.',
    'Trade additional source bytes for fewer small read calls while capping scratch memory.',
    'Keep direct owned ranges as the normal path. Row amplification and I/O concurrency are distinct from GPU numerical kernels.',
    'patches/windowed_tp.py|patches/rank_local_int8.py','',status='experimental')
tech('IO08','Prepare-once canonical CPU modules',
    'Build canonical CPU modules with policy, LoRA representation and tensor-flow folds once. Cache these prepared owners and copy them through explicit staging leases.',
    'Avoid repeated parsing, module construction and nonlinear preparation during recurring layer evaluations.',
    'Canonical CPU tensors remain immutable under GPU execution. Cache keys include source and complete numerical policy.',
    'patches/windowed_tp.py|patches/host_weight_cache.py|patches/custom_te_loader.py')
tech('IO09','Coalesced TE block backing views',
    'Read a complete rank-local TE payload sequentially into one CPU backing allocation and expose disjoint tensor views during preparation.',
    'Amortize many tiny tensor reads and allocations into a single block read.',
    'This retained TE loading ablation is not a production path. Views retain the complete payload; count that backing allocation in peak budgets.',
    'benchmarks/historical_te_loading.py','',status='retired')
tech('IO10','Descriptor, mmap and read-advice alternatives',
    'Study shared versus independent descriptors, preadv, mmap views, sequential/random advice and bounded additional reader workers as alternative byte-delivery schedules.',
    'Trade read-call overhead, page-cache behavior and allocation copies for a fixed tensor-byte workload.',
    'These are storage-study variants. Do not change the pinned source reader or assume that mmap-backed pages are free resident memory.',
    'benchmarks/te_io_storage_study.py|benchmarks/te_io_pipeline.py','',status='experimental')
tech('IO11','Source-only immutable bootstrap overlays',
    'Deliver frozen source by immutable commit/hash and install only exact approved package differences into an isolated overlay, retaining the native matching Torch/CUDA family where authorized.',
    'Avoid large wheel/model dataset mounts and repeated transfers of unchanged runtime material.',
    'This optimizes delivery, not attention arithmetic. Keep core pins and source ABI fixed; authenticate private activations without publishing credentials.',
    'benchmarks/prepare_kaggle_direct_pip.py|experiments/kitchen_attention/dense_delivery.py')
tech('IO12','Hash-bound source deltas and lazy payload delivery',
    'Apply a verified small source delta over an immutable retained distribution, reconstruct its capsule hash, and fetch learned block payloads only for the bounded work that needs them.',
    'Reduce repeated full source/archive transfers and avoid loading unused model payloads.',
    'Record the reconstructed source identity and preserve historical archives. Source deltas cannot silently replace frozen ComfyUI or unpin core operators.',
    'experiments/kitchen_attention/delivery.py|benchmarks/build_kaggle_light_delivery.py')

chapter('RES','Host cache, GPU windows and phase scheduling',
    'Treat disk construction, pageable cache, pinned staging and GPU residency as distinct owners under explicit byte budgets. Prefetch is useful only while it respects foreground work and the next actual use.',
    ['Disk preparation','Shared host budget','Pinned staging lease','H2D ready event','GPU use lease / eviction'])
tech('RES01','Process-shared host admission',
    'Pass one synchronized host budget to both spawned ranks. Reserve bytes for pageable modules, preparation and pinned staging, with system-memory headroom.',
    'Avoid multiplying the nominal RAM allowance by rank or ignoring simultaneous construction and staging allocations.',
    'A tensor view is charged by its retained backing. Denied speculative work must not strand a reservation.',
    'patches/host_weight_cache.py','HOST')
tech('RES02','Pageable cache with bounded pin staging',
    'Keep reusable prepared modules in a pageable cache. Pin only the admitted transfer representation while its staging lease is active, then release it after H2D completion.',
    'Avoid permanently pinning the entire CPU cache and duplicate long-lived pinned owners.',
    'Pageable, pin-construction and transfer lifetimes are separately accounted. A nonblocking transfer alone does not prove its source can be freed.',
    'patches/host_weight_cache.py|patches/weight_residency.py')
tech('RES03','Foreground-priority deadline prefetch',
    'Queue CPU preparation by next-use deadline and reuse value, with one worker per cache and foreground demand admitted ahead of speculative modules at safe boundaries.',
    'Overlap useful disk/CPU preparation with GPU computation without allowing speculative work to block imminent layers.',
    'Changing priority does not cancel an already executing unsafe operation. Scope cancellation at module/preparation boundaries.',
    'patches/host_weight_cache.py','REUSE')
tech('RES04','Reuse-aware host eviction',
    'Choose cached owners for eviction using recency and planned next use, preserve useful repeated-evaluation groups, and release their budget reservations explicitly.',
    'Avoid discarding a soon-reused DiT block merely because a speculative TE block arrived.',
    'Evict only unleased owners. Identity and reuse plans must be reset when a request/policy changes.',
    'patches/host_weight_cache.py|patches/weight_residency.py')
tech('RES05','Event-owned H2D residency windows',
    'Prefetch admitted modules on the transfer stream, record completion events, and wait on those events at use. Associate every resident module with byte, preparation and use owners.',
    'Overlap transfers with computation while bounding live GPU weights and staging storage.',
    'Ready, in-flight and leased states differ. Do not evict an in-use module or reuse an unfinished transfer buffer.',
    'patches/weight_residency.py','WINDOW')
tech('RES06','Next-use GPU eviction and explicit leases',
    'Track remaining uses, evict unleased far-next-use groups when making room, and retain currently leased attention/MLP/auxiliary modules until their operation ends.',
    'Use a bounded GPU window without reloading every group immediately after use or evicting active work.',
    'Admission accounts for pending allocations and external reservations. Completion and exception paths release leases deterministically.',
    'patches/weight_residency.py')
tech('RES07','Adaptive one/two-group lookahead',
    'Choose supported lookahead depth from available VRAM, activation reserve, group sizes and current in-flight residency. Reduce speculation under pressure.',
    'Prefetch an additional useful group when safe without reserving a fixed large window on every shape.',
    'Lookahead is byte-bounded and distinct from CPU cache depth. Larger activation sequences can force a smaller safe window.',
    'patches/weight_residency.py','ADAPT')
tech('RES08','Linux cgroup and PSI-aware admission',
    'Read available cgroup-v2 charge, nonfile/file-cache usage and pressure stall information. Protect useful file cache and cap concurrent preparation/speculation before exhausting the effective host limit.',
    'Avoid reclaim/thrashing caused by treating nominal machine RAM as the only memory constraint.',
    'Metrics are read-only and optional. Missing metrics retain the explicit managed/system-RAM budget rather than inventing zero pressure.',
    'patches/host_memory_pressure.py|patches/host_weight_cache.py','PRESSURE')
tech('RES09','Separated startup groups',
    'Prepare vision and embedding/startup modules as independent CPU groups rather than an undifferentiated model payload.',
    'Allow only the next required startup owner to occupy a GPU window or pinned stage.',
    'Preserve Native embedding and vision execution order and canonical CPU backing.',
    'patches/comfy_tp_runtime.py|patches/comfy_tp_models.py','STARTUP')
tech('RES10','On-demand vision GPU lifetime',
    'Copy the vision owner to its device only for the actual frontend work, fence completion and release its GPU tensors before admitting the next phase.',
    'Avoid retaining the vision tower through text layers or diffusion sampling.',
    'A synchronization fence is part of ownership. Full frontend replication remains a separate property from where weight groups reside.',
    'patches/comfy_tp_models.py','VISION')
tech('RES11','Rank-zero auxiliary projection leases',
    'Lease refiner and input/output auxiliary modules on rank zero, compute the required Native interfaces and broadcast shared results to both ranks.',
    'Avoid replicating auxiliary learned work and weights that are not tensor-parallel block computation.',
    'Keep public conditioning, refiner normalization and layout semantics. Only the explicitly owned auxiliary work is rank-zero-only.',
    'patches/windowed_tp.py|patches/comfy_tp_runtime.py','AUX')
tech('RES12','Retain DiT groups between real evaluations',
    'Extend next-use plans over the actual sequence of denoiser evaluations, retaining useful groups rather than treating each call as the end of model lifetime.',
    'Reuse expensive prepared/resident diffusion weights across sampler steps.',
    'Use actual command/evaluation boundaries rather than a guessed timestep count, and surrender residency when another phase needs the budget.',
    'patches/comfy_tp_runtime.py','RETAIN')
tech('RES13','Cross-model CPU preparation',
    'When TE completes, prepare the known next DiT CPU groups while the parent completes conditioning/packing. Preserve separate CPU and GPU admission.',
    'Move deterministic next-phase CPU loading off the immediate DiT start path.',
    'Cross-model CPU prefetch does not imply early DiT GPU residency or permission to exceed host preparation limits.',
    'patches/comfy_tp_runtime.py','CROSSMODEL')
tech('RES14','Decode-start next-request preparation',
    'At output decode start, schedule reusable TE startup groups and a bounded lifecycle-controlled prefix for a compatible next request.',
    'Use decode time for preparation that can be reused after the current sampling phase.',
    'Speculation obeys foreground decode and memory-pressure ownership. It does not preload an arbitrary next prompt activation.',
    'patches/comfy_tp_runtime.py','DECODEPREFETCH')
tech('RES15','Idle compatible-prefix residency',
    'After prompt compatibility checks, load only required parent reference encoders and a bounded worker TE prefix during an admitted idle interval.',
    'Amortize compatible next-request startup without making idle residency mandatory.',
    'Idle is explicit and off by default. Cancellation, incompatible prompts and foreground work revoke speculation at defined boundaries.',
    'patches/comfy_idle_lifecycle.py|patches/comfy_tp_runtime.py','IDLE LIFECYCLE',status='alternative')
tech('RES16','Event-owned activation handoff',
    'Copy a boundary activation through a dedicated handoff owner with completion tracking, and close that owner when the receiving stage is finished.',
    'Avoid accidental duplicate cross-phase activation residency and synchronize only the required handoff.',
    'Keep source and destination alive until stream work completes; a pipe acknowledgement does not by itself complete a CUDA transfer.',
    'patches/weight_residency.py')

chapter('VAE','Component loading and parallel decode',
    'Keep Native temporal/spatial planning, normalization and blending in the public interface. Parallelize only the declared learned work, with coordinated admission before distributed execution and a scoped fallback owner.',
    ['Reference-dependent encoder','Native decode planner','Admit decoder owners','Aligned queries + global KV','Native normalize / blend'])
tech('VAE01','Metadata-only staged VAE shells',
    'Construct component metadata without reading learned tensors. Instantiate/load only the encoder required by reference input and load the decoder at output time.',
    'Avoid unconditional video/audio encoder and decoder payload reads and simultaneous component residency.',
    'Retain required normalization buffers and the frozen constructor/forward API. Decoder-first execution must work without an encoder preload.',
    'patches/staged_vae.py|patches/reference_loaders.py','STAGEVAE')
tech('VAE02','Reference-demand encoder planning',
    'Derive immutable hints from graph-required image/video/audio references and pass only those encoder needs into idle/startup loading.',
    'Avoid speculative encoder construction for absent reference modalities.',
    'Hints identify component demand, not activation values. Actual reference encoding still selects the Native public encoder path.',
    'patches/h3_reference_plan.py|patches/comfy_sampler.py')
tech('VAE03','Independent decoder memory admission',
    'Separate Video decoder weight and activation reserve budgets from the DiT weight window. Coordinate room/refusal on both ranks before starting decoder collectives.',
    'Prevent a narrow DiT window policy from incorrectly excluding or over-admitting a replicated decoder phase.',
    'Count learned decoder replicas and external activation headroom. Both owners must make the same distributed execution decision.',
    'patches/comfy_tp_runtime.py|patches/weight_residency.py|patches/parallel_vae.py')
tech('VAE04','Weight-free Native parent decode planner',
    'Build a metadata-only Native Video planner with tiny normalization buffers and a remote decode-pixels callback. Keep learned decoder layers only in admitted workers.',
    'Avoid a redundant learned parent decoder and its CPU/GPU weight lease during remote decoding.',
    'Preserve Native chunking, interpolation, blending and output writes. Restore the original patcher/model owner in a scoped finally path.',
    'patches/video_decode_planner.py|patches/staged_vae.py')
tech('VAE05','Global-KV query-parallel Video attention',
    'Split query ranges on the retained 128-query scale-group boundary while preserving global K/V and required full-sequence inputs. Exchange outputs in Native token order.',
    'Share query-side attention and downstream owned-token work between two decoder replicas.',
    'Global KV/quantization context must not be replaced by cropped spatial tiles. Check the trained legacy decoder shape/API and aligned range availability.',
    'patches/parallel_vae.py','VIDEO')
tech('VAE06','Owned final normalization and projection',
    'Apply the decoder final norm and proj_out to each owned query/token range, then exchange the completed output in the required layout.',
    'Avoid reconstructing and projecting the complete final activation on both replicas.',
    'Changing matrix row geometry can change backend rounding. Preserve declared numerical/output contracts and Native global planning.',
    'patches/parallel_vae.py')
tech('VAE07','Coordinated Native fallback and scoped leases',
    'Resolve unsupported geometry, unavailable ABI and admission refusal before collectives. Exit remote planner/decoder leases and invoke the preserved Native decode owner.',
    'Provide a bounded supported path without retaining both a remote and parent learned GPU decoder.',
    'One rank cannot silently fall back while the other enters collectives. Restore patcher state and release in-flight ownership on exceptions.',
    'patches/video_decode_planner.py|patches/parallel_vae.py|patches/comfy_sampler.py')
tech('VAE08','Native public tiling and chunk normalization',
    'Delegate temporal/spatial chunk coordinates, overlap fades, normalization buffers and final assembly to the frozen public VAE API around the selected learned decoder.',
    'Reuse the existing bounded decode planner instead of retaining an extra full-resolution intermediate pipeline.',
    'Preserve crop coordinates and temporal context. A raw decode fragment is not a complete substitute for the public normalized output.',
    'patches/staged_vae.py|patches/comfy_sampler.py|patches/parallel_vae.py')
tech('VAE09','Single-owner FP32 Audio decoding',
    'Load the required audio decoder on one GPU and run the complete Native FP32 stereo/chunk pipeline through the public API.',
    'Avoid an additional audio decoder replica, channel collectives and alternative temporal reconstruction.',
    'The runtime selects Native single-GPU audio even when an old workflow requests an obsolete automatic parallel mode.',
    'patches/comfy_sampler.py|patches/staged_vae.py')
tech('VAE10','Overlapping spatial slab decoding',
    'Decode two independently loaded spatial slabs with explicit latent overlap, map them to output coordinates and blend the overlap.',
    'Bound each decoder spatial workload and distribute independent slab work.',
    'Cropping changes global attention context and is an approximation. Keep latent/output scale ratios and fade coordinates explicit; this is not the current query-parallel route.',
    'patches/slab_video_vae.py|patches/multi_gpu_video_vae.py','',status='historical')
tech('VAE11','Video tensor/sequence-parallel block alternatives',
    'Partition decoder projection weights and row partials, or partition token-boundary work while gathering the global context needed by the block.',
    'Distribute learned matrix work or activations instead of replicating every decoder block.',
    'Keep quantizer grouping, global norm/attention context and legacy block arguments. These retained adapters are distinct from the active replicated-decoder query split.',
    'patches/tp_video_vae.py','',status='historical')
tech('VAE12','Stereo channel-parallel Audio decoding',
    'Assign one stereo latent channel to each persistent rank decoder, decode its mono view in FP32 and gather the resulting channels in original order.',
    'Distribute left/right learned decode work and output storage.',
    'Require stereo geometry and account for shape-dependent normalization/GEMM rounding. This retained technique is removed from normal runtime dispatch.',
    'patches/audio_channel_decode.py|patches/multi_gpu_audio_vae.py','AUDIO',status='retired')
tech('VAE13','Temporal and hybrid Audio tiling',
    'Distribute overlapping temporal chunks across decoders, cross-fade reconstructed waveforms, and optionally combine temporal ownership with stereo-channel ownership.',
    'Parallelize chunked audio work and bound each chunk activation footprint.',
    'Preserve temporal context, normalization and sample-aligned fades. These historical manager paths are not the Native audio default.',
    'patches/multi_gpu_audio_vae.py','',status='historical')

chapter('ALT','Retained alternative optimization methods',
    'Retired and experimental methods remain part of the technique inventory. Their mechanisms are described here without treating an old result or a surviving function as a current deployment selection.',
    ['Alternative representation','Explicit numerical contract','Owned implementation','Scoped activation','Retained source'])
tech('ALT01','Two-stage layer pipeline parallelism',
    'Place contiguous layer ranges on two GPUs and transfer the boundary activation once per model pass. Preserve mask, positions, KV/intermediate state and Native model ordering across the boundary.',
    'Fit separate layer weight sets without a collective at every row projection.',
    'A single stream traverses stages serially; this is a different weight/communication schedule from TP2, not simultaneous execution of both layer halves.',
    'patches/pp_dit.py|patches/pp_qwen3vl.py','',status='historical')
tech('ALT02','Sequence-sharded block boundaries',
    'Gather Native floating activations from local token ranges, execute the repaired tensor-parallel block with global RoPE/modulation/masks, then return the owner token slice.',
    'Store only local sequence slices at block boundaries while retaining the existing TP arithmetic inside.',
    'The repaired adapters gather activations, not legacy INT8 sequence payloads. Global attention context and modulation cannot be inferred from one local slice.',
    'patches/sp_int8_dit.py|patches/sp_int8_qwen3vl.py','',status='historical')
tech('ALT03','Groupwise packed INT8 partial transport',
    'Quantize groups of INT32 partial outputs to signed byte codes plus serialized FP32 group scales, gather owner payloads and reconstruct/add them before the output epilogue.',
    'Trade a smaller byte payload for group metadata and encode/decode work.',
    'Separate rank scales require reconstruction before addition; byte-code SUM would overflow or apply incompatible scales. This is approximate and not default transport.',
    'patches/tilelang_transport_tradeoffs.py|patches/quantized_collectives.py','',status='experimental')
tech('ALT04','Floating partial reduction alternative',
    'Convert integer GEMM partials to floating point before distributed SUM instead of preserving the exact INT32 collective or the bounded scaled-FP16 codec.',
    'Permit a floating collective representation and different communication kernel selection.',
    'Conversion and floating SUM change cancellation/reduction behavior. This retired control does not override the current policy-selected transport.',
    'docs/reports/TP2_P2_OPTIMIZATION_REPORT.md|patches/tilelang_transport_tradeoffs.py','',status='retired')
tech('ALT05','Concatenated LoRA A projections',
    'Stack multiple adapter A factors into one projection, split its low-rank outputs and apply the corresponding B products and post-B strengths.',
    'Amortize repeated A input reads and GEMM setup across an adapter stack.',
    'Changing the A GEMM shape changes rounding and staging. This older option is rejected by current runtime controls; separate ordered adapters remain the supported method.',
    'benchmarks/h3_lora_options_audit.py|docs/reports/ONLINE_LORA_OPTIONS.md','',status='retired')
tech('ALT06','Streamed SwiGLU row preparation',
    'Produce/rotate/quantize activation chunks in a bounded schedule instead of retaining a complete branch product and its full rotated representation.',
    'Trade recomputation and chunk orchestration for shorter temporary lifetimes.',
    'The old streamed control is retired. The supported fused_recompute implementation is a separate, explicitly defined FP32 preparation path.',
    'benchmarks/qualify_streamed_swiglu.py|docs/reports/FC2_QUANTIZER_MEMORY.md','',status='retired')
tech('ALT07','Independent lossless ZSTD tensor frames',
    'Encode bounded chunks within each tensor/rank as independent ZSTD frames and retain a side index. Decode frames through a bounded outer worker pool.',
    'Reduce stored/transferred checkpoint bytes while permitting random frame-level reconstruction.',
    'This CPU codec is not integrated into the active loader. Bound output allocation and context/thread counts; decompression alone does not include disk or tensor assembly.',
    'benchmarks/te_compression_local.py|docs/reports/TE_COMPRESSION_LOCAL_20261020.md','',status='experimental')
tech('ALT08','BitShuffle plus block compression',
    'Transpose bit planes within bounded blocks, then apply the selected Blosc/ZSTD compressor, retaining independently decodable tensor/rank frames.',
    'Expose repeated bit-plane patterns to the entropy codec and enable bounded parallel decoding.',
    'Keep dtype typesize and filter order explicit. Bit-plane structure is a property of the encoded data, not a guarantee of compression for every INT8 tensor.',
    'benchmarks/te_compression_local.py','',status='experimental')
tech('ALT09','Direct and ZigZag FSE byte coding',
    'Entropy-code tensor bytes with FSE directly, or map signed values through ZigZag before the same framed coding procedure.',
    'Compress nonuniform byte distributions with a table-based entropy representation.',
    'Retain a reversible frame/index format and actual decoder output allocation. Direct and ZigZag variants are separate codecs, not interchangeable stored payloads.',
    'benchmarks/te_compression_local.py|benchmarks/build_te_compression_codecs.ps1','',status='experimental')
tech('ALT10','Typed ZipNN codec path',
    'Use the codec typed-tensor entry point rather than treating an arbitrary byte buffer as an assumed supported INT8 numerical format.',
    'Apply specialized lossless representation coding where the library supports the tensor dtype.',
    'Keep unsupported dtype handling explicit; a proposed or blocked codec is not an active H3 decompression stage.',
    'benchmarks/te_compression_local.py|benchmarks/requirements-te-compression.txt','',status='experimental')
tech('ALT11','Codec frame-size and worker scheduling',
    'Choose independently decodable frame sizes and a bounded number of outer decode workers, with one library thread per frame job. Release materialized outputs when consumed.',
    'Balance frame/index overhead, CPU parallelism, allocation peaks and random-read granularity.',
    'Avoid nested oversubscription and count live decompressed buffers. This scheduling method does not change quantized tensor values.',
    'benchmarks/te_compression_local.py','',status='experimental')


tech('ALT12','Folded-strength concatenated low-rank updates',
    'Stack adapter A factors vertically and combine strength-scaled B factors horizontally, representing the sum of several updates as a composed larger low-rank operator.',
    'Amortize A/B projection setup and repeated activation reads across multiple adapters.',
    'Pre-folding strengths and changing GEMM/reduction shapes changes floating arithmetic. This historical fused-B variant is distinct from supported ordered post-B scaling.',
    'docs/reports/H3_INT8_SWIGLU_LORA.md|docs/reports/ONLINE_LORA_NATIVE_ORDER_REPAIR.md','',status='retired')
tech('ALT13','Quantized sequence activation exchange',
    'Exchange INT8 activation codes together with the scales and ownership metadata needed to reconstruct a global sequence representation before the consuming operation.',
    'Reduce exchanged activation payload relative to a full floating-point gather.',
    'Quantizer domains, global normalization and rotary context must remain coherent. The old sequence-payload variant is not the repaired adapter that gathers Native activations.',
    'docs/reports/TEXT_ENCODER_SP_INT8_REPORT.md|patches/sp_int8_qwen3vl.py','',status='historical')


DEFAULTS=[
    ('TE attention','kitchen_causal','Pinned masked Kitchen INT8; compact GQA'),
    ('DiT / refiner','kitchen / Native SDPA','Sparse modes remain explicit'),
    ('Columns','cuda_int8_tilelang','Guarded aligned SM75 DiT shapes; preallocated fallback'),
    ('FC2 / down preparation','fused_stored','Explicit fused_recompute and owned_fp32 alternatives'),
    ('Row communication','tilelang_scaled_fp16','Explicit INT32 SUM alternative'),
    ('Tensor storage','FP16-fix; divisor 64','FP32 numerical islands; public unscaled outputs'),
    ('LoRA','offline','Optional ordered online adapters with FP32 Row-A SUM'),
    ('Scheduling','Lifecycle / pressure / adaptive enabled','Idle residency remains explicit'),
    ('Video / Audio','Query-parallel Kitchen / Native FP32','Coordinated Native Video fallback'),
]
