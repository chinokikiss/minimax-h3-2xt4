"""Validate frozen physical pilot artifacts; small-sample descriptive results."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'benchmarks'))
from idle_residency_matrix import checked_result
from tools.experiments.analyze_causal_audit import read_json


def main(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    matrix=read_json(folder/'idle_ablation.json')
    if manifest['status']!='complete' or matrix['status']!='complete':raise ValueError('Pilot incomplete')
    binding=json.loads((ROOT/'kaggle_idle_residency/source_binding.json').read_text())
    rows=[];groups={};lifecycle=[]
    for case in matrix['plan']:
        key=f"{case['policy']}_idle{int(case['idle'])}_gap{case['gap']}_r{case['repeat']}"
        result=checked_result(folder/key/'result.json',case['requests'],case['idle'],case['gap'])
        for name,sha in result['source_sha256'].items():
            if name.startswith('patches/') and name in binding and sha!=binding[name]:raise ValueError('Physical source mismatch: '+name)
        if case['idle'] and case['gap']:
            at_gap=result['requests'][0]['idle_after']['ranks']
            start=result['requests'][1]['start_ns']
            lifecycle.append(dict(case=key,
                copy_submissions_at_gap=[r['loads'] for r in at_gap],
                total_copy_submissions=[r['idle']['loads'] for r in result['runtime']],
                last_idle_submission_after_request_start_s=[max(0,(r['idle']['last_load_ns']-start)/1e9) for r in result['runtime']],
                gpu_hits=[sum(e.get('window',{}).get('idle_gpu_hits',0) for e in r['events']) for r in result['runtime']]))
        for request in result['requests']:
            row=dict(case=key,request=request['request'],warm=request['request']>0,
                idle=case['idle'],gap=case['gap'],te_s=request['stages']['reference_and_TE']['host_wall_ms']/1000,
                e2e_s=request['latency_ms']/1000,handoff_gap_ms=request['handoff_gap_ms'])
            rows.append(row)
            if row['warm']:
                group=groups.setdefault(f"idle{int(case['idle'])}_gap{case['gap']}",dict(te_s=[],e2e_s=[]))
                for metric in ('te_s','e2e_s'):group[metric].append(row[metric])
    output=dict(passed=True,physical=True,requests=len(rows),samples=rows,
        passed_scope='Native outputs, full workload, source binding, no observed OOM/idle errors and clean managed ledgers. Not ComfyUI lifecycle equivalence.',
        lifecycle_audit=lifecycle,performance_promotion_eligible=False,
        warm_summary={g:{m:dict(n=len(v),median=statistics.median(v),minimum=min(v),maximum=max(v),p95=None)
            for m,v in metrics.items()} for g,metrics in groups.items()},
        limitations=['Small paired pilot; no stable p95 or statistically credible general speedup.',
            'Standalone runner does not pause idle at request start; more copies are submitted during reference encoding. ComfyUI PromptBridge.start pauses idle, so measured latency is not verified for that production lifecycle.',
            'Idle load counts are submitted copies, not a direct count of ready entries at the boundary.',
            'Output-ready latency excludes think time. Background I/O and foreground I/O must be interpreted together.',
            'No automatic strategy promotion; inspect GPU hits, contention and larger-workload confirmation.'])
    (folder/'idle_validated_summary.json').write_text(json.dumps(output,indent=2))
    print('VERIFIED physical idle pilot requests',len(rows),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);main(parser.parse_args().folder)
