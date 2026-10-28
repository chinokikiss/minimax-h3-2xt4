"""Validate saved cu128 artifacts and summarize actual kernel events offline."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=args.input;result_path=root/'cu128_dispatch/result.json'
    result=json.loads(result_path.read_text(encoding='utf-8'))
    setup=json.loads((root/'light_setup/setup_receipt.json').read_text(encoding='utf-8'))
    binding=json.loads((root/'light_setup/delivery_binding.json').read_text(encoding='utf-8'))
    assert result['capsule_sha256']==setup['capsule_sha256']==binding['capsule_sha256']
    assert setup['api']['kitchen']=='0.2.37' and setup['api']['tilelang']=='0.1.13'
    assert result['hardware']['torch']=='2.11.0+cu128' and result['hardware']['cuda']=='12.8'
    assert len(result['hardware']['devices'])==2
    assert all(d['name']=='Tesla T4' and d['capability']==[7,5] for d in result['hardware']['devices'])
    rows=[]
    for item in sorted(result['cases'],key=lambda r:r['label']):
        label=item['label'];value=item['result'];folder=root/'cu128_dispatch'/label
        assert json.loads((folder/'result.json').read_text(encoding='utf-8'))==value
        row=dict(label=label,device=item['device'],exit_code=item['exit_code'],status=value['status'])
        for key in ('case','error','reason','error_vs_eager','inplace_error_vs_functional_eager',
                    'after_global_enable_implementation','global_cuda_unchanged','extension_calls',
                    'loaded_cuda_libraries','baseline','candidate','speedup'):
            if key in value:row[key]=value[key]
        if value['status']=='executed_not_quality_qualified':
            path=folder/'candidate_trace.json'
            events=json.loads(path.read_text(encoding='utf-8'))['traceEvents']
            kernels=[e for e in events if e.get('cat')=='kernel']
            work_ms=sum(e['dur'] for e in kernels)/1000
            assert len(kernels)==value['trace']['kernel_count']>0
            assert abs(work_ms-value['trace']['kernel_work_ms'])<1e-8
            assert sorted({e['name'] for e in kernels})==value['trace']['names']
            assert value['global_cuda_unchanged'] and item['exit_code']==0
            errors=value['error_vs_eager'];errors=errors if isinstance(errors,list) else [errors]
            assert all(e['finite'] for e in errors)
            if label.startswith('te_dequant_'):assert all(e['byte_exact'] for e in errors)
            row['trace']=dict(relative_path=str(path.relative_to(root)).replace('\\','/'),
                sha256=sha(path),kernel_count=len(kernels),kernel_work_ms=work_ms,
                gpu_annotation_count=sum(e.get('cat')=='gpu_user_annotation' for e in events))
        rows.append(row)
    counts=dict(Counter(r['status'] for r in rows))
    assert len(rows)==29 and counts==dict(executed_not_quality_qualified=27,
        unsupported_dtype_confirmed=1,failed=1)
    summary=dict(status=result['status'],hardware=result['hardware'],core_pins=dict(kitchen='0.2.37',tilelang='0.1.13'),
        capsule_sha256=result['capsule_sha256'],distribution=binding,
        result_sha256=sha(result_path),counts=counts,verified_actual_kernel_traces=27,
        measurement='Two warmups, five CUDA-event samples, fixed candidate-then-eager order; one separate candidate trace per case',
        limits='Synthetic operator inputs; no full pipeline, trained-weight error, generation-quality qualification, Nsight Systems/Compute or observer-off comparison',
        cases=rows)
    (args.output/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (args.output/'timings.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(('case','status','eager_median_ms','candidate_median_ms','speedup',
            'eager_peak_bytes','candidate_peak_bytes','candidate_kernels','candidate_kernel_work_ms'))
        for row in rows:
            b=row.get('baseline',{});c=row.get('candidate',{});t=row.get('trace',{})
            writer.writerow((row['label'],row['status'],b.get('median_ms'),c.get('median_ms'),row.get('speedup'),
                b.get('peak_allocated'),c.get('peak_allocated'),t.get('kernel_count'),t.get('kernel_work_ms')))
    print(json.dumps(dict(counts=counts,verified_actual_kernel_traces=27)))


if __name__=='__main__':main()
