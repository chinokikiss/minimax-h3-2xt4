"""Report only collected evidence; leave decisions pending without quality data."""
import hashlib
import json
from pathlib import Path
import statistics
from summarize_native_full_attention import sections
from summarize_sol_fused_attention import section as sol_section
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('D:/attention_eval/quality_constrained_20261015');OUT.mkdir(exist_ok=True)
PREV=Path('D:/attention_eval/quality_first_20261014')
GEN=PREV/'generation_v4/generation_screen'


def write(name,lines):
    text='\n'.join(lines)+'\n';(ROOT/name).write_text(text,encoding='utf8');(OUT/name).write_text(text,encoding='utf8')


def next_quality_evidence():
    paths=sorted(OUT.glob('quality_challenge_v*/generation_screen/status.json'))
    if not paths:return []
    directory=paths[-1].parent;status=json.loads(paths[-1].read_text())
    lines=['','## Collected next-generation screen','',f"Provider artifact status: {status['status']}; {len(status.get('runs',[]))} completed run records. Evidence: {directory}."]
    if status.get('reason'):lines.append('Reported blocker: '+status['reason'])
    valid=directory/'media_and_pairing_validation.json'
    if valid.exists():
        v=json.loads(valid.read_text())
        lines.append(f"Local CPU media/provenance validation completed for {len(v['validated_runs'])} new runs; full ten-run set complete: {v['complete_new_run_set']}. Exact conditioning/noise/sigma hashes are checked within each paired workload/seed. Media integrity does not establish perceptual quality.")
        lines+=['','| Run | Reference DINO | Prompt CLIP | Consecutive DINO | Post2s flow residual |','|---|---:|---:|---:|---:|']
        for run in v['validated_runs']:
            p=directory/run['name']/'embedding_metrics.json'
            if not p.exists():continue
            m=json.loads(p.read_text())
            if m.get('embedding_status')!='measured':continue
            lines.append('| '+run['name']+' | '+' | '.join(f"{m[k]['mean']:.4f}" for k in ['reference_dino','prompt_clip','consecutive_dino'])+f" | {m['flow_proxy']['post_two_seconds_residual']['mean']:.5f} |")
        lines+=['',f"New blinded packets: {directory/'blinded_review/fixed/review.html'} and {directory/'blinded_review/challenge/review.html'}. Existing user ratings are never replaced; new ratings remain blank until independent review.",
            f"Instrumented per-layer/step CUDA events, real QKV strides, compilation records and denoiser walls: {directory/'instrumented_timing_summary.json'}. Call0 and observed call9 are excluded from within-trajectory timing summaries; these are not independent steady request or TP2 throughput samples."]
    return lines


def main():
    full_group,full_sparse=sections(OUT)
    live=json.loads((OUT/'next_execution_state.json').read_text()) if (OUT/'next_execution_state.json').exists() else {}
    live_note='Latest saved job state: '+('; '.join(k+': '+str(live[k].get('status'))+', '+str(live[k].get('provider_status','not submitted')) for k in ['quality','sol']) if live else 'not yet recorded')+'.'
    quality_collected=live.get('quality',{}).get('status')=='evidence_collected'
    quality_stage=('P3 next screen chinokiki9/h3-attention-quality-challenge version1 completed: all ten new audiovisual runs passed local media and paired-input validation; CPU DINO/CLIP/flow diagnostics completed. Two fixed-workload and six challenging-workload blinded pairs are ready and remain unrated. Actual per-layer/step events, input strides and five generation Systems timelines are collected; detailed timeline attribution remains pending.'
        if quality_collected else 'P3 next screen chinokiki9/h3-attention-quality-challenge version1 submitted: two fixed-workload Group32 clips plus eight challenging motion/reference clips, paired cached inputs, two seeds, profiling step9. Completion and quality review remain pending.')
    sol_stage=('Targeted Sol rebenchmark completed36cases on physical T4: preparation is about51.0–51.1ms legacy versus12.55–12.61ms fused; API improves21–25% across the three native captures. Valid INT8 operands/scales/centroids/counts/routes are exact, and full-head output difference from legacy is zero on tested captures. Removing public validation/LSE correction alone shows no consistent API benefit; preparation changes produce the substantial measured gain. Optimized Sol has not yet been integrated into the full generation comparison.'
        if (OUT/'sol_fused_summary.json').exists() else 'Targeted faithful Sol preparation optimization is implemented; its actual-T4 conformance/performance results are pending.')
    audit=json.loads((OUT/'v4_route_audit.json').read_text())
    rows=[];metrics={}
    for record in json.loads((GEN/'status.json').read_text())['runs']:
        name=record['name'];m=json.loads((GEN/name/'embedding_metrics.json').read_text());metrics[name]=m
        rows.append('| '+name+' | '+' | '.join(f"{m[key]['mean']:.4f}" for key in ['reference_dino','prompt_clip','consecutive_dino'])+f" | {m['post_two_seconds']['prompt_clip']['mean']:.4f} | {m['flow_proxy']['post_two_seconds_residual']['mean']:.5f} |")
    (OUT/'v4_metric_summary.json').write_text(json.dumps({name:{k:v for k,v in m.items() if k not in ['flow_proxy']} for name,m in metrics.items()},indent=2))
    write('GENERATION_QUALITY_REPORT.md',['# Generation quality: existing V4 screen','',
        'Measured: eight complete90-frame640×352 clips and3.75-second stereo audio on physical T4. One Tokyo reference/prompt, two paired seeds. Native, fused Sage32 DiT, scalar tau0 DiT and dropping; TE/VAEs native. All outputs decode; exact checkpoint/source manifests are preserved.',
        'Automated CLIP/DINO/flow diagnostics completed for all eight runs. User review feedback: "Not obvious / hard to tell the difference." This is informal qualitative evidence; per-pair coverage, numeric ratings and listening conditions were not specified. No statistical quality noninferiority, identity guarantee or absence of severe failures is inferred.',
        '', '| Run | Reference DINO | Prompt CLIP | Consecutive DINO | Post2s prompt CLIP | Post2s flow residual |','|---|---:|---:|---:|---:|---:|',*rows,'',
        'These are descriptive cosine/flow proxies, not calibrated perceptual quality measures. CLIP truncates the long prompt; processor crops can omit reference details. High consecutive similarity can reflect low motion. Two seeds are two paired samples; frames do not create independent replication. No definitive scalar-pooling benefit or severe-failure rate can be established from these proxies.',
        f'Blinded review packet: {PREV/"generation_v4/blinded_review/review.html"}. All six pairs retain blank numeric ratings; the user\'s qualitative response is preserved separately, without invented scores.',
        'P3 next screen submitted as chinokiki9/h3-attention-quality-challenge version1 on actual2×T4. Full-head conformance/performance selected fused Group32 preparation with the unchanged legacy Group32 MMA. Two fixed-workload Group32 clips reuse V4 native conditioning/noise/sigmas. Eight new512×512 challenging motion/reference clips compare native, unchanged fused Sage32, selected Group32 and Sol tau0 dropping across two paired seeds. Candidate order is randomized within each challenging seed. TE/reference inputs are prepared once per seed; all four candidates share their exact hashes. These cached-input DiT comparisons are not full-pipeline TE/request latency measurements.',
        'The new screen requests per-layer/step attention events, input strides, loading/JIT/qualification timings, step9 CPU profiles and Nsight CUDA/NVTX/OSRT timelines. First-call initialization and profiled-step observer overhead must be excluded from steady-state interpretations. No new large QKV dumps are requested. Job submission alone is not evidence of successful generation; verify the collected results below when available. Independent quality review remains required.',
        'Audio export integrity is checked for V4; there is no validated audio-consistency score or completed listening review. Selective sensitivity is not defined using operator L2 and selective quantization remains deferred.',
        live_note]+next_quality_evidence())
    bottleneck=['# Attention bottleneck audit','',
        'P0 controlled T4 single-head and native56-head experiments and Systems attribution are complete. The generation timeline needed for full pipeline attribution is pending. Historical hypotheses and newer measured results are distinguished below.',
        '', '## Measured existing observations','',
        'Native DiT captures:15 tensors, five layers×three denoising steps, each Q/K/V[1,56,6814,128] FP32. Kernel tiles query32/key64,128threads,107key leaves. Packed prefix874tokens:353text+221reference+300audio. Fourteen64-token leaves are forced dense for both prefix keys and prefix queries, conservatively retaining22additional video tokens in the boundary leaf.',
        '| V4 run | Mean block density | Min layer/step density | Max layer/step density |','|---|---:|---:|---:|']
    bottleneck += [f"| {a['run']} | {a['mean']:.4f} | {a['minimum']:.4f} | {a['maximum']:.4f} |" for a in audit]
    bottleneck += ['',
        'V4 route statistics cover all50layers×20steps averaged across heads. Native-capture CPU route analysis additionally resolves all56heads at the15captured layer/step points; these are native trajectories, not the altered V4 trajectories. Block density and token-pair-weighted density are recorded separately.',
        'V4 second-seed sampling: Sage569.548s, dropping545.443s, scalar547.317s. The approximately4.2% drop-versus-Sage difference is descriptive cold fixed-order data. It cannot isolate kernel speed, compilation, loading or clocks. First native sampling also includes extra placement qualification. No controlled Amdahl fraction is measured from these runs.',
        '', '## Source-confirmed differences and testable hypotheses','',
        'The V4 sparse implementation computes unused V INT8 operands/scales and LSE shift in generic prepare, whereas fused Sage32 avoids them. Sparse routing computes centroids, small matrix multiplication and selection in PyTorch. Public dropping execute uses a Python bool on a GPU reduction to validate nonempty rows, introducing a device-to-host synchronization; the Sage internal route path avoids it. Sparse QK is per-token channelwise scale with Sol centroid centering; Sage uses Q32/K64 scales with token-mean centering. Consequently the original APIs differ in approximation and overhead.',
        'At each query CTA, dropping still scans all107key-block route entries. It skips the selected-block QK/softmax/PV branch for omitted leaves, not the route scan. Scalar pooling performs a centroid dot and value outer-product for every omitted leaf. Quantization/preparation, routing and non-attention model work are not reduced by the selected-block fraction. Their actual contributions require the new traces; these facts do not establish memory-stall or occupancy explanations.',
        'Earlier real-T4 sparse diagnostic head at density0.415: raw drop1.299ms, scalar1.545ms, original padded pool2.847ms; API drop2.932ms and scalar3.315ms. That single deliberately chosen head is not representative of the V4 head-average density0.624. These traces motivate measuring common inputs/precision instead of extrapolating a37.6% sparsity reduction to pipeline speed.',
        '', '## Controlled replay and feasibility','',
        'New matrix: native diagnostic DiT head, lengths2048/4096/native6814, requested effective densities25/33/50/62/100%, dense fused Sage32/drop/scalar. Common Sage Q32/K64 INT8 operands and FP16 PV; API boundary includes original FP32-to-FP16 casts, preparation, necessary summaries and routing. Three independent processes, randomized order, at least1s/5warmupcalls,15event/wall samples per timing. Independent streaming math distinguishes implementation from approximation.',
        'Prefix-only density floors:68.36% at2048,38.96% at4096,24.46% at6814; local neighbors can increase the achievable minimum. Report actual density rather than claiming infeasible budgets were reached.2048/4096 are contiguous native truncations with the same prefix, not independent deployment distributions.8192/16384 are skipped because no captured native sequence of those lengths exists; tokens are not tiled or synthesized to call them real distributions.',
        'Current matrix source is single-head GPU0, not full56-head API/TP2 throughput. Systems NVTX/OSRT traces and CPU cProfile are requested. Existing Compute probe gives ERR_NVGPUCTRPERM: no measured occupancy, spills, stalls, Tensor Core utilization, or instruction-level QK/softmax/PV apportionment. Fused QK/softmax/PV timing remains a single GPU interval.',
        'Nsight capture-range repeat semantics follow NVIDIA: https://docs.nvidia.com/nsight-systems/UserGuide/index.html .',
        f'Native per-head audit: {OUT/"native_capture_head_routes.json"}; V4 per-layer/step audit: {OUT/"v4_route_audit.json"}.']
    p0dirs=sorted((OUT/'p0_v1/p0').glob('process*/results.json'))
    if p0dirs:
        bottleneck += ['', '## Collected controlled results','']
        for path in p0dirs:
            data=json.loads(path.read_text());bottleneck.append(f"{path.parent.name}: {data['status']}, {len(data['cases'])} cases collected.")
        allrows=[row for path in p0dirs for row in json.loads(path.read_text())['cases']]
        bottleneck += ['', '| N | Kernel | Actual density | Median raw ms across processes | Median API ms across processes |', '|---:|---|---:|---:|---:|']
        for n in [2048,4096,6814]:
            for kind,target in [('sage32',1.),('drop',.62),('scalar_pool',.62),('drop',.33),('drop',.25),('drop',1.)]:
                rr=[r for r in allrows if r['shape'][2]==n and r['kind']==kind and r['requested_density']==target]
                if not rr:continue
                bottleneck.append(f"| {n} | {kind} requested{target} | {statistics.mean(r['effective_density'] for r in rr):.4f} | {statistics.median(r['timings']['raw']['gpu_median_ms'] for r in rr):.4f} | {statistics.median(r['timings']['API']['gpu_median_ms'] for r in rr):.4f} |")
        bottleneck += ['',
            'All three independent timing processes completed33cases each with implementation checks in process0. The notebook later reported ERROR because nsys stats refused an older SQLite export; all three raw traces were preserved. Missing SQLite exports were recovered locally with Nsight Systems2026.1.3, performing CPU-only analysis of actual-T4 traces. No benchmark was rerun on SM120.',
            'At native length/density0.6263, raw dropping saves35.0% versus dense while API saves13.1%; at2K/density0.7344 the sparse API is slower despite a faster raw kernel. At100% routes, the raw drop and dense kernels coincide closely, while routing/summary preparation remains additional API work. This isolates preparation/routing overhead from sparse compute; the V4 sampling fraction still requires full-H56 pipeline traces, so no complete Amdahl attribution is claimed.',
            'Duplicate infeasible density requests at2K and4K produced identical routes and redundant measurements. Retain the original evidence; future extensions should deduplicate route-equivalent cases rather than repeat them.']
    attribution=OUT/'p0_trace_attribution.json'
    if attribution.exists():
        bottleneck += ['', '## Actual Systems phase attribution','',
            'Three API calls per trace. Kernel durations are correlated to the innermost NVTX CPU enqueue scope using runtime/driver correlation IDs; zero kernels are unattributed. CPU scope durations measure enqueue, not GPU execution. Summed intervals can overlap host activity and must not be added as serialized pipeline costs.',
            '| NVTX phase | GPU kernels / API | GPU duration / API ms | CPU enqueue / API ms |', '|---|---:|---:|---:|']
        for trace in json.loads(attribution.read_text()):
            for phase,t in trace['phases'].items():
                bottleneck.append(f"| {phase} | {t['GPU_kernel_count']/3:g} | {t['GPU_kernel_sum_ms']/3:.4f} | {t['CPU_scope_sum_ms']/3:.4f} |")
        bottleneck += ['',
            'Matched dropping/scalar APIs launch41GPU kernels each versus10for dense Sage32, including3input casts and6quantization/mean kernels. Sparse-only centroid preparation adds11kernels/API and routing20kernels/API. Routing CPU enqueue is about0.65–0.73ms/API for only0.106–0.107ms summed GPU kernel work. This directly identifies launch/enqueue and extra preparation as optimization targets for this single-head shape.',
            'Trace kernel-interval busy fractions are75.1% dropping,80.6% scalar and93.8% dense across first GPU start to final GPU end. These are timeline duty fractions, not SM utilization or achieved occupancy. Profiled absolute kernel times differ from unprofiled event timing; do not mix these as one latency sample.',
            f'Raw attribution: {attribution}; legacy CPU cProfile summary: {OUT/"p0_cpu_profile_summary.json"}. The latter includes first-call TileLang IR specialization (about438ms over the three legacy calls), so it is a cold initialization diagnostic, not steady-state CPU overhead. The legacy public validation synchronization is source-confirmed; its exact contribution is not isolated by the current matched traces.']
    write('ATTENTION_BOTTLENECK_REPORT.md',bottleneck+full_sparse+sol_section(OUT)+['',live_note])
    group=['# Group32 audit and optimization','',
        'The audited Group32 baseline is patches/tilelang_grouped_qk.py. Grouping is contiguous channels within each token, D/32groups per head; it is not32tokens/warp and is not upstream SageAttention. For D128, scales have[B,H,Npad,4] FP32 layout; QI/KI[B,H,Npad,128] INT8, V FP16.',
        'Q and K are first FP16. K smoothing converts to FP32, computes each64-token leaf centroid using actual ragged counts, then averages leaf centroids equally. A short final leaf receives equal centroid weight. This is not the true token mean. Each token/channel group scale=max(absmax/127,1e-8); division uses Torch scalar reciprocal semantics and normalized values round to nearest-even, clamp[-127,127]. Q is not centered. Padded integer/scales are ignored outside N.',
        'Each32-channel integer dot accumulates in INT32. Its FP32 conversion is multiplied by the query/key group scales and1/sqrt(D), then accumulated in FP32 across groups. Softmax is online FP32/exp2, P is rounded FP16 per key leaf, and PV uses FP16 inputs/FP32 accumulation. Output is FP16; LSE is FP32 with the centering shift when requested. There is no global partial-GEMM materialization in the original kernel.',
        'Preserved baseline: fused Sage32 source unchanged. It uses32-token Q blocks and64-token K blocks, FP32 true token-mean smoothing, half-away rounding and one scaled integer dot over D. That is a different quantization contract.',
        'Existing selected DiT head: legacy Group32 raw2.656ms/API3.320ms versus fused Sage32 raw2.677ms/API2.945ms, but old API boundaries differ. These numbers motivate fused preparation; they do not prove Group32 is slower or quality-safe.',
        'New implementation patches/tilelang_grouped_fused.py preserves the audited Group32 centering/scale/rounding contract. Two TileLang quantizers replace PyTorch per-group chains. Q integers/scales are staged once per query CTA; K integers/scales once per key tile, then group dot-products and scaled accumulation stay inside the attention kernel. Channel groups16/32/64/128 are explicit variants;32/64 are priorities. FP32 scale storage is retained for faithful comparison.',
        'Correctness gate: actual-T4 valid QI/KI equality and scale differences against legacy Group32, plus independent dequantized streaming attention (per-leaf FP16 P). Implementation error and native approximation differences are reported separately; neither is generation acceptance.',
        'Benchmark compares current fused Sage32 and legacy Group32 with standardized original-capture-to-FP16-output boundaries, warm randomized measurements and three independent processes. Compilation/first execution is separated. Systems traces requested. No register pressure, occupancy or Tensor Core utilization claim without counters.',
        'Hardware execution initially encountered Kaggle maximum2GPU batch sessions; another existing TE critical-path job was not cancelled. Version1 compiled kernels but stopped in a ragged reference-checker error: the last V leaf included padded rows while the reference K leaf stopped at valid N. The checker was fixed and CPU regression65/353tokens, causal/noncausal, passed. Version2 hardware results below supersede that failed attempt; no production readiness is claimed.']
    for path in sorted(OUT.glob('group_v*/grouped/process*/results.json')):
        data=json.loads(path.read_text());group += ['',f"Collected {path.relative_to(OUT)}: {data['status']}.",
            '| Component | Candidate | Status | Raw median ms | API median ms |','|---|---|---|---:|---:|']
        for row in data['cases']:
            t=row.get('timings',{});group.append(f"| {row['capture']['component']} | {row['candidate']} | {row['status']} | {t.get('raw',{}).get('gpu_median_ms','pending')} | {t.get('API',{}).get('gpu_median_ms','pending')} |")
    complete=sorted((OUT/'group_v2/grouped').glob('process*/results.json'))
    if len(complete)==3 and all(json.loads(p.read_text())['status']=='complete' for p in complete):
        rr=[r for p in complete for r in json.loads(p.read_text())['cases']]
        group += ['', '## Completed version2 controlled comparison','',
            'All three independent processes completed51candidate/component cases in total on actual T4. Group16/32/64/128 passed the independent attention-reference check where D permits. Group32 valid QI/KI were bit-identical to legacy on the three native TE/DiT/VAE diagnostic heads; both scale maximum differences were zero. Maximum implementation relative difference is about0.0217%, separate from native quantization approximation error and generation quality.',
            '| Component | Candidate | Median raw ms | Median preparation ms | Median complete API ms |', '|---|---|---:|---:|---:|']
        for component in ['te','dit','video']:
            for name in ['fused_sage32','legacy_group32','fused_group32','fused_group64']:
                values=[r for r in rr if r['capture']['component']==component and r['candidate']==name]
                group.append(f"| {component} | {name} | "+' | '.join(f"{statistics.median(r['timings'][key]['gpu_median_ms'] for r in values):.4f}" for key in ['raw','preparation','API'])+' |')
        group += ['',
            'The cached-QK Group32 kernel regressed raw DiT latency enough to offset its preparation savings. It is not accepted as a faster replacement. Group64 is close to fused Sage32 on this selected head, but neither performance nor lower diagnostic error establishes a quality benefit. The next real-H56 study adds fused Group32 preparation with the unchanged legacy attention kernel, directly testing whether the new shared-memory strategy was counterproductive.',
            'Actual DiT trace launch attributes: cached Group32 uses210registers/thread and35,328dynamic shared bytes; Group64 uses165registers/thread and34,560bytes; Sage uses199registers/thread and54,272bytes. These resources do not measure achieved occupancy or spills. Group32/64 preparation+casts launch16kernels/API versus9for Sage. Full scale-centroid mean preparation remains a fusion opportunity; no hypothetical speedup is claimed.',
            f'Full native56-head job completed: chinokiki9/h3-attention-native-full-shapes, version1. It uses actual full captures at layers0/10/25, step9, without duplicating one head. Its results below supersede any performance-candidate choice based on H1. The private subset registers already completed upload blobs; no new QKV bytes were uploaded. Creation evidence: {OUT/"dit_subset_metadata/creation_status.json"}.',
            'Microbench API boundary remains captured contiguous BHND original dtype to FP16 BHND, with necessary input casts/preparation. Native ComfyUI Attention uses interleaved QKV views and returns FP32 BND; absolute microbench API times are not the native integration boundary. The forthcoming generation profile records actual strides, dtype, output casting and layout overhead; current operator ratios do not substitute for that profile.']
    write('GROUP32_OPTIMIZATION_REPORT.md',group+full_group)
    write('FINAL_PARETO_REPORT.md',['# Quality-constrained attention decisions: interim','',
        'No quality-qualified highest-throughput or production configuration is established. Existing eight V4 audiovisual runs and all proxy metrics are complete. The user reports no obvious difference; structured per-pair ratings/statistical noninferiority remain unavailable. Physical-T4 native56-head Group32 and matched sparse matrices are complete, with18Systems traces collected and attributed.',
        'Preserve fused Sage32 as the measured INT8 comparison baseline. Native dense remains the reference. Group32/64 are candidates, not accepted replacements. Use dropping as the sparse comparison default; scalar pooling has no established downstream quality benefit yet.',
        'Do not select sensitive heads/layers/steps using L2. Selective quantization remains deferred until generation evidence identifies reproducible sensitivity and dispatch/specialization overhead can be measured.',
        'Current GPU0 operator tests, two-device serial resident generation and actual TP2 are distinct. No TP/NCCL/concurrent ComfyUI throughput, P50/P95 or deployment speedup is inferred. Within-trajectory per-layer/step profiles are collected; independent warmed full-request repetitions remain pending.',
        'P0:99single-head cases plus99native H56cases completed across three independent processes each. P1:selected-head Group version2 completed51cases; full-H56 study completed36cases and exact Group32 QI/KI/scales on all valid inputs. Full-H56 selected hybrid Group32 median API161.796ms versus fused Sage32174.360ms (7.2% lower), whereas cached-QK Group32 is171.123ms. These are original captured FP32BHND→FP16BHND operator timings, not native integration/quality/TP2 speedups.',
        'Full-H56 matched dropping at density0.6263 saves about24% complete API time. The original Sol path spends about51.2ms in preparation and can lose to Sage at layer0density0.7465. Pipeline Amdahl attribution still needs the requested generation trace. Quantization/preparation and density effects are measured; Nsight Compute hardware counters remain unavailable.',
        quality_stage,
        sol_stage,
        'Generation uses serial two-device placement. Selective policies, independent warm request repetitions, true TP2/NCCL and concurrent deployment remain incomplete. Nsight Compute counters remain unavailable.',
        live_note,
        f'All collected machine-readable evidence: {OUT}. Previous validated evidence: {PREV}.'])
    sources=['patches/tilelang_sage_preparation.py','patches/tilelang_grouped_qk.py','patches/tilelang_grouped_fused.py','benchmarks/attention_p0_matrix.py','benchmarks/attention_grouped_study.py']
    (OUT/'local_source_hashes.json').write_text(json.dumps({s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},indent=2))
    print('Updated four evidence reports at',OUT)


if __name__=='__main__':main()
