"""Compact actual gate evidence; never elevate diagnostic head timings to deployment."""
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('D:/attention_eval/t4_redo_20261013/replay_gate_tiny_v1/h3_gate_results')
accuracy=json.loads((root/'accuracy/accuracy.json').read_text())
result={'scope':'real diagnostic heads, full sequences; correctness/profiling gate only',
        'hardware':accuracy['hardware'],'accuracy':[],'benchmark':[]}
for record in accuracy['records']:
    for chunk in record['head_chunks']:
        diagnostic=chunk['root_causes']
        result['accuracy'].append({'capture':record['capture'],'full_sequence_errors':chunk['outputs'],
            'sampled_ablations':diagnostic['ablations'],'input_outliers':diagnostic['input_outliers']})
for path in (root/'benchmarks').rglob('benchmark.json'):
    benchmark=json.loads(path.read_text())
    for case in benchmark['cases']:
        for mode,value in case['modes'].items():
            result['benchmark'].append(dict(mode=mode,gpu_median_ms=value['raw_kernel_or_native_SDPA']['gpu_median_ms'],
                api_gpu_median_ms=value['full_attention_API']['gpu_median_ms'],
                phases={name:data.get('gpu_median_ms') if isinstance(data,dict) else data for name,data in value['phases'].items()},
                compile_seconds=value['compile_or_cache_lookup_seconds'],error=value['vs_saved_native_output']))
(root/'gate_summary.json').write_text(json.dumps(result,indent=2))
for row in result['accuracy']:
    print('ACCURACY',row['capture']['component'],row['capture'].get('source_head'),
          {name:round(value['relative_l2'][0]*100,4) for name,value in row['full_sequence_errors'].items()})
    print('ABLATIONS',{name:round(value['output']['relative_l2'][0]*100,4) for name,value in row['sampled_ablations'].items()})
for row in result['benchmark']:
    print('TIMING',row['mode'],row['gpu_median_ms'],row['api_gpu_median_ms'],row['phases'])
