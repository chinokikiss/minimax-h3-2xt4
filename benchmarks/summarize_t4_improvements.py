"""Summarize physical candidate proof without extending diagnostic scope."""
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
data=json.loads((root/'benchmarks/targeted_improvements.json').read_text())
for case in data['cases']:
    print(case['capture']['component'],case['capture'].get('source_head'),
          'old_error',case['baseline_error']['relative_l2'],
          'new_group32_error',case['group32_error']['relative_l2'],
          'math_error',case['group32_implementation_error']['relative_l2'],
          'fast_bit_equal',case['fast_bit_equal'])
    for name,values in case['timings'].items():
        print(name,values['gpu_median_ms'],min(values['gpu_ms']),max(values['gpu_ms']))
for name in ['baseline','fast_dense','group32']:
    path=root/f'stats_{name}.log'
    if path.exists():
        lines=path.read_text().splitlines()
        print(name,[line for line in lines if line.endswith(',cudaLaunchKernel') or line.endswith(',cuLaunchKernel')])
