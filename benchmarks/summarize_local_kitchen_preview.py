"""Report local screening separately from any T4 deployment evidence."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('D:/attention_eval/operator_optimization_tp2_20261017/kitchen_inspired')


def main():
    source=BASE/'local_preview_vectorized/results.json';screen=json.loads(source.read_text())
    graph_source=BASE/'local_preview_graph/results.json';graph=json.loads(graph_source.read_text())
    assert screen['status']==graph['status']=='complete'
    assert len(screen['cases'])==22 and all(r['status']=='complete' for r in screen['cases'])
    group=[r for r in screen['cases'] if not r['config'].get('native')]
    assert all(all(r['operand_identity'].values()) for r in group)
    frozen=json.loads((BASE/'submitted_sources.json').read_text())
    unchanged={name:hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()==value for name,value in frozen.items()}
    assert all(unchanged.values()),unchanged
    adaptation_audit={}
    for folder in (source.parent,graph_source.parent):
        audit={}
        for adapted in (folder/'adapted/preview_patches').glob('tilelang_*.py'):
            original=ROOT/'patches'/adapted.name
            text=original.read_text(encoding='utf-8')
            expected=text.replace('from patches.','from preview_patches.').replace("'sm_75'","'sm_120'").replace('!=(7,5)','!=(12,0)')
            expected=expected.replace('Actual SM75','Local preview SM120').replace('actual SM75','local preview SM120')
            actual=adapted.read_text(encoding='utf-8');assert actual==expected,adapted
            audit[adapted.name]=dict(original_sha256=hashlib.sha256(text.encode()).hexdigest(),adapted_normalized_sha256=hashlib.sha256(actual.encode()).hexdigest(),only_documented_changes=True)
        # Explicit UTF8 audit corrects locale-dependent hash text in the
        # initial runtime manifest; the initial manifest remains preserved.
        (folder/'utf8_adaptation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        adaptation_audit[str(folder)]=audit
    (BASE/'local_preview_runner.py').write_text((ROOT/'benchmarks/local_kitchen_preview.py').read_text(encoding='utf-8'),encoding='utf-8')
    lines=['# Local Kitchen-inspired Group32 preview — 2026-10-17','',
        'This is a completed local RTX 5060 Ti (SM120) screen, not a T4 result. Real captured H3 inputs were sliced to the current TP2 contracts. Both logical ranks ran sequentially on one GPU. No H3 generation, sparse policy change, NCCL or full-pipeline test ran.','',
        'Preparation fusion is the promising candidate. The larger query tiles / 256-thread warp variants regressed locally. Do not use this architecture-specific ranking to reject or select T4 kernels.','',
        '## Reproducible scope','',
        f"Hardware: {screen['hardware']}; driver 616.56; Torch {screen['versions']['torch']}, CUDA runtime {screen['cuda_runtime']}, TileLang {screen['versions']['tilelang']}, Kitchen {screen['versions']['comfy-kitchen']}. Local nvcc is CUDA 13.3 with MSVC. These differ from queued T4 dependencies (Torch 2.11/cu130, TileLang 0.1.15, Kitchen 0.2.37, CUDA compiler 12.8).",
        '', 'TP2 shapes (BHND, batch 1): TE Q=(32,353,128), KV=(4,353,128), causal GQA8; DiT QKV=(28,6814,128) with V token stride 10752 matching packed QKV; Video Q=(32,1024 or 773,64), global KV=(32,1797,64). Captures: TE layer0, DiT layer25/call9, Video layer0/call0. Fixture hashes and actual strides are recorded in JSON. These are native-trajectory captures sliced for TP2, not newly generated TP2 trajectories.',
        '', 'The screen used five warm-up calls and ten individual CUDA-event / synchronized-wall samples per phase. API timings were repeated in reverse order without compilation. A separate rank0 confirmation used fifteen samples and CUDA Graph replay as a Python-dispatch ablation. First-call JIT timings were excluded from steady-state samples. Windows was not an exclusive benchmark environment; small timing differences and inter-round drift should not be overinterpreted.',
        '', '## Short confirmation (logical rank 0 only)','',
        'Median CUDA-event API interval in ms. The range spans first and reversed repeat-round medians; graph replay removes per-operation Python dispatch and excludes graph capture. It is diagnostic, not an enabled TP2 production feature.','',
        '| Component | Current API | Fused API | Current graph | Fused graph | Graph reduction |',
        '|---|---:|---:|---:|---:|---:|']
    confirmation={}
    for component in ('te','dit','video'):
        rows={r['name']:r for r in graph['cases'] if r['component']==component}
        current,fused=rows['current'],rows['fused_prepare']
        span=lambda r:'–'.join(f'{x:.3f}' for x in sorted([r['API']['gpu_median_ms'],r['API_repeat']['gpu_median_ms']]))
        a=current['graph_API']['gpu_median_ms'];b=fused['graph_API']['gpu_median_ms']
        confirmation[component]=dict(current_graph_ms=a,fused_graph_ms=b,reduction_percent=100*(1-b/a))
        lines.append(f'| {component} | {span(current)} | {span(fused)} | {a:.3f} | {b:.3f} | {100*(1-b/a):.1f}% |')
    lines+=['', 'The TE eager-vs-graph gap supports a substantial Python/launch contribution. DiT remains dominated by GPU work: fused API 17.53 ms and graph 17.48 ms. This ablation is not Nsight evidence of a particular memory stall or instruction bottleneck. Video Kitchen confirmation is 0.383–0.392 ms API / 0.275 ms graph; fused Group32 is 0.608 / 0.560 ms, so it still has a substantial local gap to Kitchen.',
        '', '## Full screen: both logical TP2 partitions','',
        'Phase timings are separately measured event intervals and need not add to API latency. They include applicable launch gaps; they are not pure GPU kernel durations.','',
        '| Component | Logical rank | Candidate | API first / repeat ms | Prep ms | Attention + output ms | Bitwise current |',
        '|---|---:|---|---:|---:|---:|---|']
    for r in sorted(screen['cases'],key=lambda r:(r['component'],r['rank'],r['name'])):
        fmt=lambda key:f"{r[key]['gpu_median_ms']:.3f}" if key in r else '—'
        lines.append(f"| {r['component']} | {r['rank']} | {r['name']} | {fmt('API')} / {fmt('API_repeat')} | {fmt('preparation')} | {fmt('attention_and_output')} | {r.get('bitwise_equal_to_current','different algorithm')} |")
    lines+=['', '## Numerical checks and limits','',
        f"All {len(group)} Group32 cases had exact QI/KI/QS/KS/V/count identity against the baseline at their own tile geometry and passed full-output rtol=0.001, atol=0.0002. Fused preparation alone was bitwise output-identical in all six component/partition cases. {sum(r['bitwise_equal_to_current'] for r in group)}/{len(group)} complete Group32 outputs were bitwise identical; tile/warp changes can alter rounding. Kitchen is a separate numerical method, not part of this equivalence gate.",
        '', 'The optional dense-reference check samples 32 evenly spaced real queries per head using every key with FP32 math and TF32 disabled. It is a smoke check, not full-distribution or generation-semantic validation. Fused preparation leaves these errors unchanged. In Video rank0, median-head relative RMSE is 0.0934% for Group32 vs 0.1412% for Kitchen, but worst-head error is 0.6822% vs 0.6811%; this does not establish uniformly higher precision.',
        '', '## Compatibility and representativeness','',
        'The production/submitted sources are untouched and all eight normalized submitted-source hashes still match. Isolated copies change only the package namespace, SM75 guards and target to SM120. Installed TileLang 0.1.13 generated an invalid cutlass::half_t → __half conversion in vectorized V padding. The final run repairs the generated pack_float16x4 helper to accept any 2-byte type, using the identical 16-bit shift/OR packing. Original and repaired CUDA sources are retained; vectorization and attention scheduling are preserved. The earlier scalar-vectorization-disabled diagnostic run is excluded from the final timings.',
        '', 'Representative for real tensor distribution, TP2 partition sizes/strides, operand contracts and local implementation screening. Not representative of T4 instruction mapping, occupancy, cache/memory behavior, relative kernel ranking, dual-GPU concurrency or NCCL/PCIe cost. The T4 job remains the authority for deployment decisions; its last status in this turn was QUEUED.',
        '', 'Next on T4: confirm fused preparation first; retain the small scheduling matrix to measure actual SM75 behavior. Keep existing sparse drop/scalar semantics and numerical contracts unchanged. No production switch is justified by this local preview alone.',
        '', '## Artifacts','',f'- Screen: `{source}`',f'- Confirmation/graph ablation: `{graph_source}`',f'- Adaptations/CUDA sources: `{source.parent}`',f'- Runner: `{ROOT / "benchmarks/local_kitchen_preview.py"}`']
    report=ROOT/'docs/reports/LOCAL_KITCHEN_GROUP32_PREVIEW.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    metadata=dict(screen_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),confirmation_sha256=hashlib.sha256(graph_source.read_bytes()).hexdigest(),
        submitted_sources_unchanged=unchanged,adaptation_audit=adaptation_audit,confirmation=confirmation,group32_cases=len(group),completed_cases=len(screen['cases']))
    (BASE/'local_preview_summary.json').write_text(json.dumps(metadata,indent=2))
    print(report);print(json.dumps(confirmation,indent=2))


if __name__=='__main__':main()
