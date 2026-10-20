"""Read exported Nsight Systems kernel resources and API/kernel timelines.

Resources are launch attributes, not Nsight Compute occupancy or stall counters.
"""
import json
from pathlib import Path
import sqlite3
import sys

root=Path(sys.argv[1]);results=[]
for path in root.glob('*.sqlite'):
    db=sqlite3.connect(path)
    db.row_factory=sqlite3.Row
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    item={'file':path.name,'resource_semantics':'CUPTI launch attributes; achieved occupancy/memory stalls not measured'}
    if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
        cols={r[1] for r in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
        names={r[0]:r[1] for r in db.execute('SELECT id,value FROM StringIds')}
        rows=list(db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL'))
        kernels=[]
        for row in rows:
            name=names.get(row['shortName'],'') if 'shortName' in cols else names.get(row['demangledName'],'')
            if 'main' in name:
                kernels.append({'name':name,'duration_us':(row['end']-row['start'])/1000,
                    **{k:row[k] for k in ['registersPerThread','staticSharedMemory','dynamicSharedMemory',
                         'localMemoryPerThread','localMemoryTotal','gridX','gridY','gridZ','blockX','blockY','blockZ'] if k in cols}})
        item['attention_kernels']=kernels
        item['gpu_kernel_count']=len(rows)
        item['gpu_kernel_sum_ms']=sum(row['end']-row['start'] for row in rows)/1e6
        item['attention_kernel_sum_ms']=sum(k['duration_us'] for k in kernels)/1000
    results.append(item);db.close()
(root/'nsys_launch_evidence.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
