"""Validate scoped BF16 evidence and export a small, source-bound summary."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=('dit_7895.json','dit_layer24_stress.json','dit_layer49_stress.json','te_layer0.json',
       'te_layer49_stress.json','dit_lora_513_v2.json','frontend.json','runtime.json','vision.json','video_reference_direct.json')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    records=[];environment=None
    for name in FILES:
        path=root/'final'/name;data=json.loads(path.read_text(encoding='utf-8'))
        if data['status']!='complete':raise ValueError(f'Incomplete: {path}')
        observed={k:data[k] for k in ('hardware','capability','torch','runtime_lock_sha256')}
        if environment is None:environment=observed
        elif observed!=environment:raise ValueError('Mixed environments in the selected component records')
        for source,expected in data['source_sha256'].items():
            candidates=[ROOT/source,*root.glob('sources/*/'+source)]
            if not any(p.is_file() and digest(p)==expected for p in candidates):
                raise ValueError(f'Missing measured source snapshot: {source} {expected}')
        cases=[]
        for case in data['cases']:
            row={k:v for k,v in case.items() if k!='operators'}
            events=[v for k,v in case.get('operators',{}).items() if k.startswith('__activation_storage_cast_')]
            row['storage_cast_events']=len(events)
            row['max_single_cast_nonzero_to_zero']=max((v['cast_nonzero_to_zero'] for v in events),default=0)
            row['max_single_cast_finite_to_nonfinite']=max((v['cast_finite_to_nonfinite'] for v in events),default=0)
            row['observed_nonfinite']=any(v.get('nonfinite',0) for v in case.get('operators',{}).values())
            cases.append(row)
        records.append(dict(file=name,sha256=digest(path),configuration=data['configuration'],
            distribution=data.get('distribution'),cases=cases,
            bf16_reference_operators={k:v['storage_dtype'] for k,v in data.get('bf16_reference',{}).get('operators',{}).items()},
            matched_kitchen_bf16_vs_native=data.get('matched_kitchen_bf16_reference',{}).get('vs_native_bf16')))
    selected=[c for r in records for c in r['cases'] if c.get('flow')=='fp16' and c['exponent']==6]
    if len(selected)!=6 or any(c['output']['nonfinite'] or c['observed_nonfinite'] or not c['rank_equal'] or not c['repeat_equal'] for c in selected):
        raise ValueError('A selected c64 block probe is missing or nonfinite')
    runtime=next(r for r in records if r['file']=='runtime.json')['cases'][0]
    if not (runtime['refined_rank_exact'] and runtime['packed_rank_exact'] and runtime['collective_steps_released']):
        raise ValueError('Preprocessing ownership/broadcast gate failed')
    video=next(r for r in records if r['file']=='video_reference_direct.json')['cases']
    if not video[0]['native_fp16_exact'] or not video[1]['query_parallel_exact']:
        raise ValueError('Video Native/default or optional query-parallel gate failed')
    return dict(reference_correction='Historical BF16 pairwise distances only, not ground-truth error. Use h3_native_fp32_flow_20261021.json for the requested reference.',
        scope='Local SM120 BF16 numerical component evidence. Logical TP2 only. No full H3, T4 performance, NCCL or generation-quality qualification.',
        environment=environment,runtime='Frozen conda comfyui / pinned ComfyUI d49e888',
        decisions=dict(legacy_default_preserved=True,fp16_opt_in=True,residual_divisor=64,vision_divisor=1,video_divisor=1,
                       audio_vae='unchanged FP32',whole_trajectory_qualified=False,t4_qualified=False),
        status='bounded_component_gates_passed; production_accuracy_unqualified',
        case_count=sum(len(r['cases']) for r in records),records=records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('D:/attention_eval/h3_tensor_flow_20261021'))
    parser.add_argument('--output',type=Path,default=ROOT/'docs/data/h3_tensor_flow_bf16_20261021.json')
    args=parser.parse_args();result=collect(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for record in result['records']:
        for case in record['cases']:
            if case.get('exponent')==6 and case.get('flow')=='fp16':
                print(record['file'],case['vs_native_bf16'],case['input_cast'],flush=True)
    print(result['status'],result['case_count'],flush=True)


if __name__=='__main__':main()
