"""P2 Systems attribution + per-API CPU enqueue ranges, preserving process IDs."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
files=list(args.root.rglob('*.sqlite'));assert files,'Export nsys-rep files first'
output=args.root/'trace_attribution.json'
subprocess.run([sys.executable,str(Path(__file__).with_name('attribute_tp2_tradeoffs.py')),*map(str,files),'--output',str(output)],check=True)
results=json.loads(output.read_text())
for result in results:
    db=sqlite3.connect(result['file']);db.row_factory=sqlite3.Row;names=dict(db.execute('select id,value from StringIds'))
    pids={r['globalPid']:r['deviceId'] for r in db.execute('select globalPid,deviceId from CUPTI_ACTIVITY_KIND_KERNEL')}
    ranges=[]
    for row in db.execute('select * from NVTX_EVENTS where end>start'):
        r=dict(row);label=r['text'] or names.get(r['textId'],'')
        if label.endswith('/API'):
            ranges.append(dict(device=pids.get(r['globalTid']&~((1<<24)-1)),name=label,start=r['start'],end=r['end'],CPU_enqueue_range_ms=(r['end']-r['start'])/1e6))
    result['API_CPU_enqueue_ranges']=ranges
    result['interpretation']='CPU API range is asynchronous enqueue time, not synchronized full API latency. CUDA runtime API durations include their own synchronization when present. Compare independent warmed event and synchronized wall samples for latency.'
    db.close()
output.write_text(json.dumps(results,indent=2));print('P2 trace attribution',len(results))
