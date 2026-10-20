"""Reuse physical kernels to distinguish NCCL occupancy from useful progress."""
import argparse
import gc
import json
from pathlib import Path
from analyze_causal_audit import load_trace,union,duration,clipped
from analyze_cpu_path import complement,intersection


def analyze(folder):
    result=json.loads((folder/'result.json').read_text())
    cpu=json.loads((folder/'cpu_path_analysis.json').read_text())
    all_busy=[];model_busy=[];nccl=[]
    for rank in ('parent',0,1):
        _,kernels,_=load_trace(folder,rank)
        all_busy+=union([(e['start_ns'],e['end_ns']) for e in kernels])
        model_busy+=union([(e['start_ns'],e['end_ns']) for e in kernels
            if e['category'] not in ('nccl','memcpy','memset')])
        nccl+=union([(e['start_ns'],e['end_ns']) for e in kernels if e['category']=='nccl'])
        print('PROGRESS kernels loaded rank',rank,len(kernels),flush=True)
        del kernels;gc.collect()
    report=dict(physical=result['physical'],case=result['case'],requests=[],
        semantics='All rank/thread waits and device intervals are unions. NCCL-only occupancy during a foreground Future wait is evidence of no observed model-kernel progress, not a complete causal DAG or removable time.',
        model_activity='GEMM, attention, quantization, layout and other non-NCCL kernels; excludes memcpy/memset. Kernel categories are not exact SM utilization.')
    for request in result['requests']:
        row=dict(request=request['request'],stages={})
        for stage in ('reference_and_TE','sampler'):
            a,b=(request['stages'][stage][k] for k in ('start_ns','end_ns'))
            waits=union([(w['start_ns'],w['end_ns']) for w in cpu['proven_foreground_waits']
                if w['request']==request['request'] and a<=w['start_ns']<b])
            no_model=complement(model_busy,a,b);no_any=complement(all_busy,a,b)
            masked=intersection(intersection(waits,no_model),nccl)
            row['stages'][stage]=dict(stage_ms=(b-a)/1e6,foreground_wait_union_ms=duration(waits),
                foreground_wait_without_any_GPU_activity_ms=duration(intersection(waits,no_any)),
                foreground_wait_without_model_kernel_progress_ms=duration(intersection(waits,no_model)),
                NCCL_occupancy_without_model_progress_during_foreground_wait_ms=duration(masked),
                largest_masked_intervals_s=sorted([[(x-a)/1e9,(y-a)/1e9] for x,y in masked],
                    key=lambda v:v[1]-v[0],reverse=True)[:8])
        report['requests'].append(row)
    (folder/'useful_progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['requests'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    analyze(parser.parse_args().folder)
