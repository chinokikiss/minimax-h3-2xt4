"""Read completed artifacts only; compare aligned TP2 trials using slower rank."""
import argparse
import json
from pathlib import Path
import statistics


def aggregate(ranks,name):
    measurements=[rank['measurements'][name] for rank in ranks]
    counts={len(m['samples']) for m in measurements}
    if len(counts)!=1 or next(iter(counts))<3:
        raise ValueError('At least three aligned repeats on both ranks are required')
    samples=[dict(cuda_ms=max(a['cuda_ms'] for a in pair),wall_ms=max(a['wall_ms'] for a in pair))
             for pair in zip(*(m['samples'] for m in measurements))]
    return dict(samples=samples,median_cuda_ms=statistics.median(s['cuda_ms'] for s in samples),
        median_wall_ms=statistics.median(s['wall_ms'] for s in samples),
        worst_rank_peak_allocated_gib=max(m['peak_allocated'] for m in measurements)/2**30,
        peaks_allocated_gib_by_rank=[m['peak_allocated']/2**30 for m in measurements],
        initial_allocated_gib_by_rank=[m['initial_allocated']/2**30 for m in measurements],
        peaks_reserved_gib_by_rank=[m['peak_reserved']/2**30 for m in measurements],
        sampled_device_used_after_gib_by_rank=[m['device_used_after']/2**30 for m in measurements])


def summarize(root):
    root=Path(root);batch=json.loads((root/'result.json').read_text())
    result=dict(batch_status=batch['status'],full_pipeline=False,source_capsule_sha256=batch.get('capsule_sha256'),
        interpretation='Seeded long-shape operator/block fixture; no generation or full-model window qualification',
        sections={},blockers=[])
    devices=batch.get('hardware',{}).get('devices',[])
    physical=(len(devices)==2 and len({v.get('uuid') for v in devices})==2
              and all(v.get('capability')==[7,5] and 'T4' in v.get('name','') for v in devices))
    result['physical_two_t4_receipt']=physical
    if not physical:result['blockers'].append('Missing/mismatched two-T4 hardware receipt')
    for section in ('norm_gate','column','row_epilogue','attention','block'):
        ranks=[]
        for rank in (0,1):
            path=root/section/f'rank{rank}/result.json'
            if path.is_file():ranks.append(json.loads(path.read_text()))
        if len(ranks)!=2 or any(r.get('status')!='complete' for r in ranks):
            result['sections'][section]=dict(status='incomplete',rank_results=[dict(rank=r.get('rank'),status=r.get('status'),error=r.get('error'),reason=r.get('reason'),completed_errors=r.get('errors')) for r in ranks])
            result['blockers'].append(section+' incomplete; no speedup claim');continue
        names=set(ranks[0].get('measurements',{}))&set(ranks[1].get('measurements',{}))
        rows={name:aggregate(ranks,name) for name in sorted(names)}
        for name,row in rows.items():
            baseline=('norm_baseline' if name.startswith('norm_') else 'gate_baseline') if section=='norm_gate' else ('production' if section=='column' else 'baseline')
            if baseline in rows:
                row.update(cuda_speedup=rows[baseline]['median_cuda_ms']/row['median_cuda_ms'],
                    worst_rank_peak_saved_gib=rows[baseline]['worst_rank_peak_allocated_gib']-row['worst_rank_peak_allocated_gib'])
        record=dict(status='complete',measurements=rows,
            numerical_by_rank=[{k:r[k] for k in ('errors','error','norm_error','gate_error','stress_norm_error','exact_qualified','signed_extreme_int32_exact','copy_stream_lifetime') if k in r} for r in ranks],
            profiler_by_rank=[r.get('profiles') for r in ranks])
        if section=='block':
            record['resident_weight_gib_by_rank']=[r['resident_weight_bytes']/2**30 for r in ranks]
            record['phase_memory_by_rank']=[r.get('phase_memory') for r in ranks]
            record['collectives_by_rank']=[r.get('collectives') for r in ranks]
            record['excluded_variants_by_rank']=[r.get('excluded_variants') for r in ranks]
            record['numerically_byte_qualified']=[name for name in sorted(names-{'baseline'})
                if all(r.get('errors',{}).get(name,{}).get('byte_exact',False) for r in ranks)]
            record['non_weight_allocation_caveat']='Subtracting weight bytes also leaves inputs, outputs and workspaces; not pure activation memory'
        result['sections'][section]=record
        if section=='column' and any(r.get('exact_qualified') is not True for r in ranks):
            result['blockers'].append('Column candidates failed byte qualification; timings do not authorize promotion')
        if any(p.get('status')!='complete' for r in ranks for p in r.get('profiles',{}).values()):
            result['blockers'].append(section+' has incomplete profiler capture')
    result['default_promoted']=False
    result['caveats']=['Component peak savings overlap and must not be added',
        'CUDA time uses the slower rank for each aligned repeat, then median',
        'Sampled full-device usage is not the full-device peak',
        'Compilation/loading and error checks are outside warmed timing',
        'Missing CUDA trace coverage blocks attribution even if latency samples exist']
    return result


def markdown(result):
    lines=['# DiT VRAM fusion operator results','',
        f"Batch status: **{result['batch_status']}**. Physical two-T4 receipt: **{result['physical_two_t4_receipt']}**.",'',
        result['interpretation']+'.','',
        '| Section / variant | CUDA ms, slower-rank median | Speedup | Worst rank allocated GiB | Allocated GiB saved |',
        '|---|---:|---:|---:|---:|']
    for section,record in result['sections'].items():
        if record['status']!='complete':
            lines.append(f'| {section}: incomplete | — | — | — | — |');continue
        for name,row in record['measurements'].items():
            lines.append(f"| {section}/{name} | {row['median_cuda_ms']:.3f} | {row.get('cuda_speedup',0):.3f}× | {row['worst_rank_peak_allocated_gib']:.3f} | {row.get('worst_rank_peak_saved_gib',0):.3f} |")
    lines+=['','Numerical results, phase peaks, collective counts and trace coverage are retained in the JSON summary. No production default is promoted.','']
    lines += ['- '+value for value in result['caveats']]
    if result['blockers']:lines+=['','Blockers: '+ '; '.join(result['blockers'])+'.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.input);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.output/'summary.md').write_text(markdown(result),encoding='utf-8')
    print(json.dumps(dict(status=result['batch_status'],blockers=result['blockers'],default_promoted=False)))


if __name__=='__main__':main()
