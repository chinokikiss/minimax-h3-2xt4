"""Offline instruction inventory; not dynamic hardware counters."""
import json
from pathlib import Path
import re
import subprocess

root=Path('D:/attention_eval/tp2_p2_20261018/offline');rows=[]
for cubin in sorted(root.glob('*.cubin')):
    p=subprocess.run(['cuobjdump','--dump-sass',str(cubin)],capture_output=True,text=True,timeout=30,check=True)
    cubin.with_suffix('.sass').write_text(p.stdout)
    counts={}
    for op in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s',p.stdout):
        counts[op]=counts.get(op,0)+1
    rows.append(dict(source=cubin.stem,static_instruction_counts=counts,
        INT8_MMA_static=sum(v for k,v in counts.items() if k.startswith('IMMA')),FP16_MMA_static=sum(v for k,v in counts.items() if k.startswith('HMMA')),
        barriers_static=sum(v for k,v in counts.items() if k.startswith('BAR')),conversions_static=sum(v for k,v in counts.items() if k.startswith(('I2F','F2I','F2F'))),
        scope='Offline Windows CUDA13.3 + TileLang0.1.13 SM75 codegen, not Kaggle Linux CUDA12.8 + TileLang0.1.15. Static instruction occurrences, not dynamic counts/stalls/achieved occupancy.'))
(root/'instruction_inventory.json').write_text(json.dumps(rows,indent=2))
print('Offline SASS inventories',len(rows))
