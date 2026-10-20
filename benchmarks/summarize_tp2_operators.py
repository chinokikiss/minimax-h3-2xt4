"""Audit collected TP2 results; never infer missing timings or counters."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    root=args.root;state=json.loads((root/'status.json').read_text());report={'provider_state':state,'attention':[],'gemm':[],'limitations':[]}
    for family in ('attention','gemm'):
        paths=[root/family/f'rank{r}.json' for r in range(2)]
        if not all(p.exists() for p in paths):
            report['limitations'].append(f'{family}: results missing on at least one rank');continue
        results=[json.loads(p.read_text()) for p in paths]
        if not all(r['status']=='complete' for r in results):report['limitations'].append(f'{family}: at least one rank incomplete')
        assert all('T4' in r['hardware'] for r in results)
        if family=='attention':
            lookup=[{r['candidate']:r for r in data['cases']} for data in results]
            for name in sorted(set(lookup[0])&set(lookup[1])):
                if name=='native':continue  # native records are keyed by component below
                pair=[d[name] for d in lookup];ms=[r['timings']['API']['gpu_median_ms'] for r in pair]
                report[family].append({'candidate':name,'rank_API_median_ms':ms,'slower_rank_API_ms':max(ms),
                    'rank_raw_median_ms':[r['timings']['raw']['gpu_median_ms'] for r in pair],
                    'rank_preparation_median_ms':[r['timings']['preparation']['gpu_median_ms'] for r in pair],
                    'max_implementation_relative_l2':max(x for r in pair for x in r['implementation_error']['relative_l2']),
                    'rank_density':[r['density'] for r in pair],
                    'operand_match':all(all(r['operand_match'].values()) for r in pair),
                    'rank_native_error_max_relative_l2':[max(r['native_error']['relative_l2']) for r in pair]})
            report['native_attention']=[{'rank':data['rank'],'component':r['component'],'API':r['timings']['API']} for data in results for r in data['cases'] if r['candidate']=='native']
        else:
            assert len(results[0]['cases'])==len(results[1]['cases'])
            for pair in zip(*(d['cases'] for d in results)):
                assert pair[0]['target']['module']==pair[1]['target']['module']
                row=pair[0]['target']['partition']=='row';counts=[r['API']['warmup_actual_calls'] for r in pair]
                match=not row or counts[0]==counts[1]
                if row:
                    match=match and all(pair[0][p]['warmup_actual_calls']==pair[1][p]['warmup_actual_calls'] for p in ('INT32_SUM','FP32_MAX'))
                if not match:report['limitations'].append('Mismatched collective warm-up counts: '+pair[0]['target']['module'])
                report[family].append({'component':pair[0]['target']['component'],'module':pair[0]['target']['module'],
                    'rank_MNK':[[r['target'][k] for k in ('M','N','K')] for r in pair],
                    'rank_API_median_ms':[r['API']['gpu_median_ms'] for r in pair],
                    'rank_raw_GEMM_median_ms':[r['raw_INT8_GEMM']['gpu_median_ms'] for r in pair],
                    'row_route':[r['row_route'] for r in pair],'collective_warmup_counts_match':match,
                    'INT32_core_exact':all(r['INT32_core_exact'] for r in pair),
                    'rank_INT32_SUM_ms':[r['INT32_SUM']['gpu_median_ms'] for r in pair] if row else None})
    report['systems_reports']=[str(p.relative_to(root)) for p in root.glob('*.nsys-rep')]
    if not report['systems_reports']:report['limitations'].append('No collected Nsight Systems reports')
    report['limitations']+=['No current Nsight Compute hardware-counter evidence; no stall/occupancy claims.',
        'Real attention fixture subset only; synthetic linear inputs, not a distribution-wide quality or full-pipeline study.',
        'Per-phase event medians are descriptive and non-additive; use NVTX-correlated Systems intervals for attribution.']
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k in ('limitations','attention','gemm','systems_reports')},indent=2))


if __name__=='__main__':main()
