"""Collected full-head T4 evidence; no synthetic data or device execution."""
import json
from pathlib import Path
import pstats
import re
import statistics


def summarize(out):
    full=out/'native_full_v1/native_full'
    if not (full/'status.json').exists():return None
    if json.loads((full/'status.json').read_text())['status']!='complete':return None
    groups=[json.loads((full/f'group_process{i}/results.json').read_text()) for i in range(3)]
    sparse=[json.loads((full/f'p0_process{i}/results.json').read_text()) for i in range(3)]
    assert all(d['status']=='complete' for d in groups+sparse)
    gr=[r for d in groups for r in d['cases']];sp=[r for d in sparse for r in d['cases']]
    assert len(gr)==36 and len(sp)==99
    assert all(r['shape']==[1,56,6814,128] for r in gr+sp)
    def timing(rows,keys):
        return {key:{'median_ms':statistics.median(r['timings'][key]['gpu_median_ms'] for r in rows),
                     'process_median_min_ms':min(r['timings'][key]['gpu_median_ms'] for r in rows),
                     'process_median_max_ms':max(r['timings'][key]['gpu_median_ms'] for r in rows)} for key in keys}
    summary={'status':'complete','scope':'Physical T4 GPU0 original contiguous captured FP32 BHND to FP16 BHND, not native QKV/BND or TP2',
             'shape':[1,56,6814,128],'group_case_count':len(gr),'matched_sparse_case_count':len(sp),
             'replication':'Three independent processes; three captured layers at step9; 15 warmed measurements/case',
             'group':[],'sparse':[],'legacy':[],'cpu_profiles':[],'PTX_mma_opcodes':{}}
    for layer in [0,10,25]:
        for candidate in ['fused_sage32','fused_group32','fused_group32_legacy_mma','fused_group64']:
            rows=[r for r in gr if r['capture']['layer']==layer and r['candidate']==candidate]
            summary['group'].append({'layer':layer,'candidate':candidate,'timings':timing(rows,['raw','preparation','API'])})
        for kind,target in [('sage32',1.),('drop',.25),('drop',.33),('drop',.5),('drop',.62),('drop',1.),('scalar_pool',.62)]:
            rows=[r for r in sp if r['capture']['layer']==layer and r['kind']==kind and r['requested_density']==target]
            summary['sparse'].append({'layer':layer,'kind':kind,'requested_density':target,
                'density':statistics.mean(r['effective_density'] for r in rows),
                'timings':timing(rows,['raw','cast_and_preparation','API']+([] if kind=='sage32' else ['routing']))})
        rows=[r for d in sparse for r in d['legacy_cases'] if r['capture']['layer']==layer]
        summary['legacy'].append({'layer':layer,'density':statistics.median(r['density'] for r in rows),
            **{k:statistics.median(r[k]['gpu_median_ms'] for r in rows) for k in ['raw','preparation','API']}})
    summary['maximum_group_implementation_relative_l2']=max(max(r.get('implementation_error',{}).get('relative_l2',[0])) for r in gr)
    summary['maximum_matched_implementation_relative_l2']=max(max(r.get('implementation_error',{}).get('relative_l2',[0])) for r in sp)
    summary['group32_all_valid_operands_equal']=all(r['legacy_operand_match']=={'qi_equal':True,'ki_equal':True,'qs_max_diff':0.,'ks_max_diff':0.}
        for r in gr if r.get('legacy_operand_match'))
    for p in sorted(full.glob('p0_process*/*.prof')):
        stat=pstats.Stats(str(p));top=[]
        for (filename,line,function),(cc,nc,tt,ct,_) in sorted(stat.stats.items(),key=lambda item:item[1][3],reverse=True)[:12]:
            top.append({'file':filename,'line':line,'function':function,'calls':nc,'self_seconds':tt,'cumulative_seconds':ct})
        summary['cpu_profiles'].append({'file':str(p),'total_seconds':stat.total_tt,'top_cumulative':top,
            'scope':'Three warmed legacy calls plus final synchronization; host blocking time is not CPU computation or isolated synchronization penalty'})
    for p in sorted(full.glob('group_process0/*.ptx')):
        summary['PTX_mma_opcodes'][p.stem]=sorted(set(re.findall(r'\bmma\.sync[^\s{;]+',p.read_text())))
    for name in ['group','p0']:
        p=out/f'native_full_{name}_trace_attribution.json'
        if p.exists():summary[name+'_traces']=json.loads(p.read_text())
    (out/'native_full_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def sections(out):
    s=summarize(out)
    if s is None:return [],[]
    group=['','## Completed real full-head T4 study','',
        'Version1 completed on actual T4:36 Group cases and99 matched-density cases, three independent processes, native layers0/10/25 at denoising step9. Every input is actual[1,56,6814,128]; no head replication. Fifteen warmed measurements per case, randomized candidate order. Tables show medians of the three process medians for each captured layer.',
        '', '| Layer | Candidate | Raw ms | Preparation ms | Complete API ms | API process-median range ms |',
        '|---:|---|---:|---:|---:|---:|']
    for r in s['group']:
        t=r['timings'];a=t['API']
        group.append(f"| {r['layer']} | {r['candidate']} | {t['raw']['median_ms']:.3f} | {t['preparation']['median_ms']:.3f} | {a['median_ms']:.3f} | {a['process_median_min_ms']:.3f}–{a['process_median_max_ms']:.3f} |")
    group+=['',
        'The full-head result differs from the diagnostic-head result: fused preparation with the unchanged legacy Group32 MMA is the fastest measured Group32 variant. Across the nine layer/process cells its median API is161.796ms, versus174.360ms for fused Sage32 (7.2% lower); cached-QK Group32 is171.123ms. This selects a generation candidate, not a quality-qualified winner. Group64 is171.377ms and is not prioritized over the faster exact-contract Group32 candidate.',
        f"Group32 valid QI/KI match legacy on all56heads of all three captures; QS/KS maximum differences are zero. Independent reference checks use heads0/28/55 per capture; maximum grouped implementation relative L2 is{s['maximum_group_implementation_relative_l2']:.8f}. They verify implementation only, not generation quality.",
        'Exported SM75 PTX contains mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 for INT8 QK and mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 for FP16 PV. Opcode presence proves generated instruction types, not achieved Tensor Core utilization.',
        'Nine Group Systems traces have zero unattributed kernels. Full-H56 preparation/casts launch18kernels/API for cached Group32/64 versus11for Sage, plus one attention kernel each. Shape-dependent ATen reduction dispatch differs from the earlier H1 launch count. The faster hybrid was measured unprofiled; its generation timeline is requested separately. Do not attribute its speedup to an unmeasured occupancy/stall mechanism.',
        f'Machine summary: {out/"native_full_summary.json"}; raw events, commands, source hashes, PTX and eighteen Systems traces: {out/"native_full_v1/native_full"}.',
        'Native integration remains unvalidated here: these captures are contiguous BHND and return FP16 BHND. The generation test includes actual QKV strides and FP32 BND output conversions.']
    sparse=['','## Completed native H56 controlled replay','',
        'These are content-based top-k budget routes with mandatory prefix/local blocks, using matched fused-Sage QK precision. They isolate density/overhead effects; they are not the Sol tau0 routes used in V4 generation. No native dense-attention timing is inferred from this INT8 comparison.',
        '', '| Layer | Mode | Requested density | Actual density | Raw ms | Cast/preparation ms | API ms |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in s['sparse']:
        t=r['timings'];sparse.append(f"| {r['layer']} | {r['kind']} | {r['requested_density']:.2f} | {r['density']:.4f} | {t['raw']['median_ms']:.3f} | {t['cast_and_preparation']['median_ms']:.3f} | {t['API']['median_ms']:.3f} |")
    sparse+=['',
        'At density0.6263, full-H56 dropping reduces raw compute about35% and complete matched API latency about24% across these layers. It is not intrinsically speed-neutral. At density1 the extra routing/preparation makes the sparse API slower than Sage despite similar raw compute. Lower density improves the measured kernel/API further; quality has not been established for these budget routes.',
        '', '| Layer | Original Sol tau0 actual density | Original raw ms | Original preparation ms | Original API ms |',
        '|---:|---:|---:|---:|---:|']
    for r in s['legacy']:
        sparse.append(f"| {r['layer']} | {r['density']:.4f} | {r['raw']:.3f} | {r['preparation']:.3f} | {r['API']:.3f} |")
    sparse+=['',
        'The actual V4-style Sol approximation has different INT8 operands and heterogeneous native densities: layer0.7465, layer10.6586, layer25.5793 here. Its preparation costs about51.2ms/API, and complete API is180.6/167.3/155.6ms. Layer0 is slower than fused Sage despite faster raw sparse compute. This directly demonstrates substantial legacy preparation cost and a density-dependent break-even point. These native inputs are not the evolving V4 candidate inputs; no pipeline gain is back-calculated as a measured result.',
        'Full-H56 matched Systems traces attribute all kernels. Representative layer0 dropping:3casts(3.548ms),8QK preparation kernels(6.399ms),12centroid/packing kernels(9.233ms),20routing kernels(2.845ms),1fused QK/softmax/PV kernel(101.330ms), per API under profiling. Dense Sage uses3casts+8preparation+1attention. Fused QK/softmax/PV remains one measured interval, not independently apportioned.',
        'Full-H56 timeline kernel-union/span duty fractions are about99.9–100%; the earlier H1 launch-gap fraction is not representative of H56. This is timeline activity, not SM utilization. Warm legacy CPU profiles now exclude initial JIT: most host time is inside public execute and final device synchronization. The bool(GPU reduction) validation blocks the host on preceding device work; cProfile does not measure pure CPU computation or an independently removable134ms penalty.',
        'The actionable source targets remain unnecessary V8/LSE work, centroid/mean preparation and host validation synchronization. Group32 fused preparation is implemented and measured; a mathematically faithful optimized Sol path still requires its own conformance and T4 benchmarks. Pipeline Amdahl attribution is pending the generation timeline, so the original small generation gain is only partly explained.',
        f"Matched reference maximum relative L2{s['maximum_matched_implementation_relative_l2']:.8f}; faithful implementation check only. Nsight Compute remains blocked by ERR_NVGPUCTRPERM; memory stalls, achieved occupancy, spills and Tensor Core utilization are unavailable.",
        f'Full-head CPU/trace attribution and repetition ranges: {out/"native_full_summary.json"}.']
    return group,sparse


if __name__=='__main__':
    result=summarize(Path('D:/attention_eval/quality_constrained_20261015'))
    print('Full-head summary:',result['group_case_count'],'group and',result['matched_sparse_case_count'],'matched cases')
