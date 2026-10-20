"""Report completed evidence only; numerical differences never decide quality."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('D:/attention_eval/quality_first_20261014');OUT.mkdir(exist_ok=True)
quant=OUT/'quant_v2/quality_first';pool=OUT/'pool_v2/quality_first'
q=json.loads((quant/'unprofiled/results.json').read_text());p=json.loads((pool/'unprofiled/results.json').read_text())
assert q['status']==p['status']=='complete'
lines=['# Quality-first T4 study: completed operator stages and pending generation validation','',
    'Acceptance objective: generation quality preservation at maximum practical throughput. No optimization is accepted or rejected solely from attention L2, omitted mass or trajectory divergence.',
    '', '## Completed scope','',
    'Stages 1–2 completed in private Kaggle version-2 quantization and pooling jobs on physical Tesla T4 capability7.5, Torch2.11.0+cu128, TileLang0.1.15. Both allocation devices were T4; operator measurements use GPU0. Four deliberately selected diagnostic heads preserve native full sequence lengths. This is neither population-wide accuracy nor TP2/full-pipeline performance evidence.',
    'The original report’s numerical-only production rejection language has been removed. Existing captures and prior Systems evidence are reused; no new Q/K/V capture was required.',
    '', '## Sage-inspired QK diagnostics','',
    'The upstream local Sage dispatcher rejects SM75. These are TileLang SM75 ports of mathematical quantization configurations, not upstream Sage kernel benchmarks. Token-mean smoothing, Q32/K64 and Q128/K64 scale granularity, no-smoothing and Sol-centering controls, token scales, Group32, FP32-source versus FP16-source and INT8-PV are compared.',
    'Version2 uses FP32 mean subtraction and half-away rounding, matching the stated inspired contract. Further source audit found the CUDA path stores the mean in input dtype and uses nearest-even integer conversion with reciprocal-max multiplication; a separately labeled CUDA-contract follow-up was implemented. These differences are not silently treated as upstream equality.',
    '', '| Selected native head | Existing INT8 diagnostic L2 | Token-mean/token-scale | Sage Q32/K64 | Group32 |',
    '|---|---:|---:|---:|---:|']
for case in q['cases']:
    c=case['capture'];m=case['modes']
    lines.append('| '+c['component']+' / head'+str(c['source_head'])+' | '+' | '.join(f"{m[n]['versus_native']['relative_l2'][0]*100:.3f}%" for n in ['existing','token_center_token_scale','sage_warp32','group32'])+' |')
maximum=max(e['implementation']['relative_l2'][0] for result in [q,p] for case in result['cases'] for e in case['modes'].values() if 'implementation' in e)
lines+=['',f'Independent streaming references validate the stated approximation; maximum kernel/reference relative difference across completed dense configurations is {maximum*100:.4f}%. Sampled logits/probabilities use deterministic query rows and every key. Full operator diagnostics use all query rows. These implementation checks do not imply generation noninferiority.',
    'TE centering is a real source of quantization sensitivity: Q32/K64 with Sol-centering gives25.02% diagnostic error, token-centering13.51%, no-centering20.58%. For the selected DiT QK head, block scales produce larger differences than token scales or Group32. FP32-source quantization does not resolve that head (30.93% versus30.76% with FP16-source). These are head-specific mechanisms, not a recommendation to discard INT8.',
    '', '## Correction and density sweep','',
    '| DiT25/step9/head14 route | Actual selected density | Drop raw ms | Original pool raw ms | Scalar pool raw ms | Drop / scalar API ms |',
    '|---|---:|---:|---:|---:|---:|']
for s in p['cases'][0]['routing_sweep']:
    v=s['variants'];raw=lambda n:v[n]['timings']['raw']['gpu_median_ms'];api=lambda n:v[n]['timings']['API']['gpu_median_ms']
    lines.append(f"| {s['routing']} {s['parameter']} | {s['density']:.3f} | {raw('drop'):.3f} | {raw('original_pool'):.3f} | {raw('scalar_pool'):.3f} | {api('drop'):.3f} / {api('scalar_pool'):.3f} |")
lines+=['',
    'Each routing configuration uses identical routes for dropping and both corrections. Mandatory packed text/reference/audio key and query leaves increase effective density beyond the nominal top-k budget. Density is measured after these constraints.',
    'At Sol tau0, scalar pooling reduces the raw kernel from2.847ms to1.545ms (1.84×); complete cast/preparation/routing/kernel API4.458ms to3.315ms. Drop is faster (1.299ms raw,2.932ms API). Across density0.33–0.79, original pooling stays2.85–2.96ms, consistent with evaluating padded pooled GEMMs for every omitted leaf. Scalar correction has a nonzero cost beyond dropping and must earn that cost through downstream quality; similar head L2 cannot settle its utility.',
    'Per-route retained/omitted mass, block-normalization and value oracle counterfactuals are retained in results.json. They separate exact mass/conditional-weighted V from centroid/mean approximations. Pooled summaries approximate omitted contributions; high omitted mass is not automatically lost mass or poor generation quality.',
    '', '## Profiler evidence and timing limits','',
    'Actual Systems CUDA/NVTX/OSRT traces, SQLite exports and telemetry are retained. Scalar-pool trace has267 GPU kernels over three API calls and3 attention kernels totaling3.974ms; attention launch attributes show42,112bytes shared memory and168registers/thread. Sage32 TE trace attributes show54,272bytes and243registers/thread. These are launch resources, not measured occupancy or spills/stalls.',
    'The Sage32 TE Systems summary records123 cudaLaunchKernel calls over three API invocations (41/call), plus the three driver attention launches. Its attention kernel accounts for26.2% of summed GPU kernel duration; preparation creates many small kernels and CPU enqueue work. This identifies a fusion opportunity. CUDA/NVTX ranges mark CPU enqueue scopes; their duration is not the device duration of asynchronous fused QK/softmax/PV.',
    'Nsight Compute counters remain ERR_NVGPUCTRPERM, established by the previous actual T4 probe. Instruction-level QK/softmax/PV breakdown, achieved occupancy and memory stalls are unmeasured. Do not infer them from launch attributes.',
    'Unprofiled timings warm the actual operation for at least one second/five calls and retain15 GPU-event and wall samples; loading, compilation and first call are excluded. These are one-process exploratory distributions, not confidence intervals over deployment requests. Most dense preparation APIs start from pre-cast FP16 operands; Group32/native SDPA and sparse API entries include input casts. Do not directly rank mixed timing contracts as full native-input APIs. Finalists require uniform input boundaries, randomized order and repeated independent processes.',
    '', '## Targeted preparation follow-up','',
    'A fused TileLang Sage preparation replaces PyTorch padding/absmax/rounding chains with two quantization kernels. First attempts passed TE but caught DiT bin differences caused by floating division/scaling transformations. Those failed jobs are preserved; the optimization is not labeled universally equivalent. The revised implementation explicitly matches scalar reciprocal multiplication and IEEE tensor division and awaits a new actual-T4 validation slot. The CUDA-contract comparison is also pending completion across all three heads.',
    '', '## Real generation and deployment status','',
    'A private12MB source/reference bundle preserves the requested ComfyUI-master files by SHA256 (the source directory has no verified Git commit). The generation job verifies source identity and pinned public checkpoint SHA256 before native/candidate20-step,90-frame640×352 runs. Its small screen uses two paired seeds, native attention, Sage32 DiT, scalar tau0 DiT and same-route dropping; TE/VAE remain native to isolate DiT. No pixel or trajectory equality is required. No new full QKV or denoiser dumps are enabled.',
    'Generation version1 stopped before inference because Kaggle expanded ZIP inputs; subsequent versions discover expanded source trees and individually verify source hashes. Version2 timed out; version3 completed two audiovisual runs before a statistics-export failure. Version4 outcomes are recorded below. Current resident execution places serial blocks on two GPUs; this is not TP2 throughput.',
    'Automatic appearance/semantic/flow diagnostics and blinded paired-review packet builders are implemented. Metric models resolve and record checkpoint revisions before loading. CLIP long-prompt truncation, crop bias and uncalibrated flow/embedding proxies are explicit limitations. Human ratings are not yet available; automated assessment is never described as human review.',
    'Reference/prompt/seed expansion, separate TE/VAE interventions, validated layer/stage policies and actual2×T4 TP/NCCL/PCIe/concurrent-request benchmarks remain pending. Additional references are needed for broad identity claims. No quality-versus-throughput Pareto frontier or best validated production configuration exists yet.',
    '', '## Reproduction and evidence','',
    f'- Plan: {ROOT/"QUALITY_FIRST_T4_EXPERIMENT_PLAN.md"}',
    f'- Quantization version2: {quant/"unprofiled/results.json"}',
    f'- Pooling version2: {pool/"unprofiled/results.json"}',
    f'- Commands/source manifests, Systems exports and telemetry: {quant} and {pool}',
    '- Kaggle CLI2.2.4 ignores historical versions for some commands. The SDK downloader explicitly supplies version_label=v2 and hashes collected artifacts; do not mix current-version telemetry with older measurements.',
    '- Primary upstream references: https://github.com/Kaggle/kaggle-cli/issues/1200 ; https://github.com/Vchitect/VBench ; https://github.com/facebookresearch/dinov2',
    '', 'This is an evidence-backed interim report, not completion of all five experimental stages.']
followup=OUT/'quant_v5/quality_first/unprofiled/results.json'
if followup.exists():
    f=json.loads(followup.read_text())
    if f['status']=='complete':
        start=lines.index('## Targeted preparation follow-up')
        end=lines.index('## Real generation and deployment status')
        lines[start:end]=['## Targeted preparation follow-up','',
            'Version5 completed on actual T4. Fused Sage preparation matches all valid INT8 Q/K operands and scales bit-for-bit on all three selected heads; each scale maximum difference is zero. The failed versions exposed scalar-division lowering differences, resolved by matching Torch scalar reciprocal multiplication and using explicit IEEE tensor division.',
            '', '| Diagnostic head | Fused preparation ms | Fused API ms | CUDA-contract API ms | CUDA-contract native diagnostic L2 |',
            '|---|---:|---:|---:|---:|']+[
            f"| {c['capture']['component']} | {c['modes']['sage_warp32_fused']['timings']['preparation']['gpu_median_ms']:.3f} | {c['modes']['sage_warp32_fused']['timings']['API']['gpu_median_ms']:.3f} | {c['modes']['sage_cuda_warp32_contract']['timings']['API']['gpu_median_ms']:.3f} | {c['modes']['sage_cuda_warp32_contract']['versus_native']['relative_l2'][0]*100:.3f}% |" for c in f['cases']]+['',
            'The fused TE Systems capture has12 runtime launches plus9 driver launches over three calls:7 compute launches/call including two quantizers and attention, versus42/call for the previous preparation. Raw timelines and exact source hashes are retained in quant_v5. This validates a preparation optimization, not a generation or full-pipeline speedup. The CUDA-contract variant implements native-dtype token mean and nearest-even conversion; it is separately labeled from the FP32-mean/half-away variant.', '']
generation=OUT/'generation_v2/generation_screen/status.json'
if generation.exists():
    g=json.loads(generation.read_text())
    if g['status']=='blocked':
        lines += ['', '## Generation version2: actual failure and recovery','',
            'The provider reports the notebook COMPLETE because its wrapper caught the exception; its experiment status is blocked/TimeoutExpired. Source import and all four exact checkpoint hashes passed. Native reference/TE took120.157s. Native sampling reached denoiser call17, then the2400s request budget stopped the run. No completed video/audio pair was generated, and no generation-quality result is reported.',
            'Source inspection independently confirms ComfyUI-master/comfy/quant_ops.py disables Kitchen CUDA acceleration for Torch runtimes below13. Version2 used cu128 and its log records that warning. No timeline evidence yet apportions the timeout between eager GPU computation, weight loading and CPU overhead.',
            'Version3 is submitted with Torch2.11.0+cu130, a verified CUDA12.8 nvcc retained for SM75 TileLang, an actual-T4 compatibility gate before checkpoint download, two-device resident Native loading and a first-real-denoiser-input qualification against streamed Native. The request budget is3600s with an overall job bound. This is native block placement, not TP/NCCL throughput evidence.',
            'A multi-device route-summary export fix and the now-validated fused Sage preparation are prepared for subsequent execution. They have not been injected into the already-running version3. Final artifacts and status must determine which runs actually completed.']
generation3=OUT/'generation_v3/generation_screen'
if (generation3/'status.json').exists():
    g3=json.loads((generation3/'status.json').read_text())
    lines += ['', '## Generation version3: completed outputs and export failure','',
        'The experiment ran on two physical Tesla T4 devices with Torch2.11.0+cu130. The runtime gate passed TileLang SM75 and Kitchen CUDA execution. All four checkpoint hashes matched. Streamed and resident native outputs matched exactly on the first real denoiser input (both components relative L2=0); this qualifies placement at that input, not downstream quality.',
        'Seed20261013 native and Sage32 DiT each completed20 steps and exported90-frame640×352 video plus3.75-second stereo audio. Both remain unrated. Scalar tau0 DiT also completed20 steps and exported video, then route-summary torch.stack failed because scalars resided on both cuda:0 and cuda:1. Its audio export did not execute; it is an incomplete audiovisual run. Dropping and the second seed did not execute.',
        '', '| Run | Recorded sampling wall seconds | Completion |',
        '|---|---:|---|']
    for name in ['native_seed20261013','sage32_dit_seed20261013','scalar_tau0_dit_seed20261013']:
        state=json.loads((generation3/name/'run_state.json').read_text())
        lines.append(f"| {name} | {state['stages']['native_sampling']['wall_seconds']:.3f} | {state['status']} |")
    lines += ['',
        'These are cold single generation runs in fixed order. Native sampling includes the extra full streamed-versus-resident loading qualification; candidate runs do not. Compilation/loading, qualification, order and cache effects prevent interpreting their ratio as controlled steady-state speedup. There is no TP/NCCL or concurrent-request performance conclusion.',
        'Version4 consolidates route scalars on CPU after sampling and uses the already T4-validated fused Sage preparation. It repeats the two-seed/four-mode screen. The exact version3 outputs and source hashes are retained; version4 changes are not attributed retrospectively to version3.',
        f'Collected status and error evidence: {generation3/"status.json"} and {generation3/"scalar_tau0_dit_seed20261013.log"}.',
        f'A one-pair blinded native/Sage audiovisual review is available at {OUT/"generation_v3/blinded_review/review.html"}. The incomplete scalar run is excluded; all ratings remain pending.']
generation4=OUT/'generation_v4/generation_screen'
if (generation4/'status.json').exists():
    g4=json.loads((generation4/'status.json').read_text())
    lines += ['', '## Generation version4: completed two-seed quality screen','',
        f"Version-bound experiment status: {g4['status']}. All eight runs generated audiovisual outputs on two physical T4 SM75 devices, Torch2.11.0+cu130. Four interventions across seeds20261013/20261014: native, fused Sage32 DiT, scalar tau0 DiT and same-route dropping. TE and both VAEs remain native. The route-statistics export fix succeeded.",
        '', '| Seed / mode | Sampling wall seconds (cold, unqualified for speedup) | Post-2s flow residual proxy | Selected block density |',
        '|---|---:|---:|---:|']
    for run in g4['runs']:
        name=run['name'];directory=generation4/name
        state=json.loads((directory/'run_state.json').read_text())
        metrics=json.loads((directory/'video_metrics.json').read_text())
        density='dense'
        if (directory/'routing.json').exists():
            route=json.loads((directory/'routing.json').read_text());values=[r['selected_density'] for r in route]
            density=f'{sum(values)/len(values):.3f}'
        lines.append(f"| {name} | {state['stages']['native_sampling']['wall_seconds']:.3f} | {metrics['flow_proxy']['post_two_seconds_residual']['mean']:.5f} | {density} |")
    lines += ['',
        'Farneback flow residual at256×144 is a supporting proxy, not calibrated temporal quality. Different motion and attention trajectories can change it without perceptual degradation. No quality winner or noninferiority finding is claimed. CLIP/DINO assessment and blinded ratings are pending; two seeds of one prompt/reference/resolution cannot establish broad failure rates.',
        'Timing remains a cold fixed-order screen; native seed20261013 includes a streamed-versus-resident qualification absent in the other runs. These numbers must not be used as deployment speedup ratios. Compilation/loading separation, warm randomized repeats, Systems pipeline traces and actual TP/NCCL/concurrency measurements remain required.',
        f'Six blinded audiovisual pairs: {OUT/"generation_v4/blinded_review/review.html"}. Ratings are blank until independently supplied.',
        'Local Kaggle access initially failed with a TLS EOF. Process-scoped TLS1.2 restored API access with certificate verification enabled; no machine networking configuration or credentials were changed.',
        f'Version-bound status: {generation4/"status.json"}; artifact byte counts and SHA256: {OUT/"generation_v4/download_manifest.json"}.']
(OUT/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
audit={}
for source in [Path('C:/Users/Administrator/Documents/project/SageAttention-main/sageattention/core.py'),
    Path('C:/Users/Administrator/Documents/project/SageAttention-main/sageattention/quant.py'),
    Path('C:/Users/Administrator/Documents/project/SageAttention-main/sageattention/triton/quant_per_block.py'),
    Path('C:/Users/Administrator/Documents/project/SageAttention-main/csrc/fused/fused.cu'),
    Path('C:/Users/Administrator/Documents/project/SageAttention-main/csrc/numeric_conversion.cuh')]:
    audit[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
(OUT/'upstream_source_hashes.json').write_text(json.dumps(audit,indent=2))
print(OUT/'EXPERIMENT_REPORT.md')
