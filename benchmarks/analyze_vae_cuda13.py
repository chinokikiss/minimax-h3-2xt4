"""Read saved CUDA13 Chrome traces without double-counting GPU annotations.

This is offline analysis only. Submitted sources and raw traces are preserved.
Kernel phases are assigned by CUDA launch correlation and the enclosing CPU
record_function call site, including CUDA extensions without External id.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def union_us(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<=start:continue
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return sum(end-start for start,end in merged)


def family(name):
    if 'cutlass1x' in name and 'Igemm' in name:return 'legacy_INT8_GEMM'
    if 'cutlass_75_tensorop_i8816' in name or ('OpClassTensorOp' in name and 'signed char' in name):return 'SM75_INT8_GEMM'
    if 'turing_fp16' in name:return 'FP16_GEMM_or_rotation'
    if 'qk_int_sv' in name or 'fmha_' in name:return 'attention_core'
    if 'convrot' in name:return 'ConvRot_quantization'
    if 'quantize' in name or 'quant_per' in name:return 'other_quantization'
    if 'elementwise' in name or 'copy_kernel' in name:return 'elementwise_or_layout'
    return 'other'


def attribute(events):
    scopes=defaultdict(list)
    for event in events:
        if event.get('cat')=='user_annotation' and event.get('dur',0)>0:
            scopes[event.get('pid'),event.get('tid')].append(event)
    launches={}
    for event in events:
        if event.get('cat') not in ('cuda_runtime','cuda_driver'):continue
        correlation=event.get('args',{}).get('correlation')
        if correlation is None:continue
        active=[s for s in scopes[event.get('pid'),event.get('tid')]
                if s['ts']<=event['ts']<s['ts']+s['dur']]
        sites=[s for s in active if s['name'].startswith('VAE_LINEAR_SITE/')]
        attention=[s for s in active if s['name'].startswith('VAE_ATTENTION/')]
        chosen=min(sites or attention or active,key=lambda s:s['dur']) if active else None
        label=chosen['name'] if chosen and (sites or attention) else 'norm_RoPE_layout_or_outer'
        launches[correlation]=label
    groups=defaultdict(lambda:dict(kernels=0,work_us=0.0))
    families=defaultdict(lambda:dict(kernels=0,work_us=0.0))
    kernels=[e for e in events if e.get('cat')=='kernel']
    copies=[e for e in events if e.get('cat') in ('gpu_memcpy','gpu_memset')]
    unmatched=[]
    for event in kernels:
        correlation=event.get('args',{}).get('correlation')
        phase=launches.get(correlation,'unmatched_launch')
        if phase=='unmatched_launch':unmatched.append(event['name'])
        for row in (groups[phase],families[family(event['name'])]):
            row['kernels']+=1;row['work_us']+=event.get('dur',0)
    annotations=[e for e in events if e.get('cat')=='gpu_user_annotation']
    return dict(complete=bool(kernels) and not unmatched,kernel_count=len(kernels),
        kernel_work_ms=sum(e.get('dur',0) for e in kernels)/1000,
        kernel_union_ms=union_us((e['ts'],e['ts']+e['dur']) for e in kernels)/1000,
        copy_or_memset_count=len(copies),copy_or_memset_work_ms=sum(e.get('dur',0) for e in copies)/1000,
        excluded_gpu_annotations=len(annotations),unmatched_kernel_count=len(unmatched),
        phases={name:dict(kernel_count=r['kernels'],kernel_work_ms=r['work_us']/1000) for name,r in groups.items()},
        kernel_families={name:dict(kernel_count=r['kernels'],kernel_work_ms=r['work_us']/1000) for name,r in families.items()})


def analyze(root):
    result=json.loads((root/'result.json').read_text())
    if result['status']!='complete' or len(result['cases'])!=30:raise ValueError('Incomplete factorial')
    rows=[]
    for case in result['cases']:
        label=f"gpu{case['device']}_b{case['batch']}_layer{case['layer']}_{case['linear']}_{case['attention']}.json"
        if case['linear']=='fp16_native':label=label.replace('_fp16_native_','_fp16_')
        trace=json.loads((root/label).read_text())
        metrics=attribute(trace['traceEvents'])
        if not metrics['complete']:raise ValueError('Missing correlated GPU kernels: '+label)
        row={k:case[k] for k in ('device','batch','layer','linear','attention','gpu_median_ms','wall_median_ms','peak_allocated')}
        row.update(profile=metrics,trace=label)
        rows.append(row)
    aggregates=[]
    for batch in (1,4):
        medians={}
        for linear,attention in (('eager','sdpa'),('cuda','sdpa'),('eager','kitchen'),('cuda','kitchen'),('fp16_native','sdpa')):
            cases=[c for c in rows if c['batch']==batch and c['linear']==linear and c['attention']==attention]
            medians[linear+'/'+attention]=statistics.median(c['gpu_median_ms'] for c in cases)
        aggregates.append(dict(batch=batch,median_of_three_layer_medians_ms=medians,
            cuda_vs_eager_speedup_fixed_kitchen=medians['eager/kitchen']/medians['cuda/kitchen'],
            cuda_int8_latency_reduction_vs_fp16=1-medians['cuda/kitchen']/medians['fp16_native/sdpa']))
    return dict(status='complete',scope='30 real-input VAE block replays; no full decoder timing or quality claim',
        correction='Only Chrome category kernel counts as a kernel. gpu_user_annotation ranges are excluded; do not use the submitted CUDA_EVENTS count as a launch count.',
        aggregates=aggregates,cases=rows,counterfactuals=result['counterfactuals'],
        limitations=['Profile work times are single diagnostic captures, separate from seven warmed measurements.',
            'SM75 GEMM kernel names establish selected kernels, not measured occupancy or stall counters.',
            'FP16 uses a different checkpoint. No accuracy equivalence or full-video speedup was tested.',
            'Other registry operations use the same CUDA13 automatic dispatch within the factorial.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=analyze(args.input);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(status=report['status'],aggregates=report['aggregates'])))


if __name__=='__main__':main()
