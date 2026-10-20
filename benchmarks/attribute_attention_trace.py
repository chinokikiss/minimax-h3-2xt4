"""Correlate actual T4 kernel launches with innermost CPU NVTX API scopes."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3


def main():
    p=argparse.ArgumentParser();p.add_argument('files',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=[]
    for file in args.files:
        db=sqlite3.connect(file);db.row_factory=sqlite3.Row
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        names=dict(db.execute('select id,value from StringIds'))
        ranges=[dict(r) for r in db.execute('select * from NVTX_EVENTS where end>start')]
        for r in ranges:r['label']=r['text'] or names.get(r['textId'],'')
        calls={};api_counts=Counter();api_ns=Counter()
        for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
            if table not in tables:continue
            for r in db.execute('select * from '+table):
                calls.setdefault(r['correlationId'],dict(r));api_counts[names.get(r['nameId'],'unknown')]+=1
                api_ns[names.get(r['nameId'],'unknown')]+=r['end']-r['start']
        phase=defaultdict(lambda:{'GPU_kernel_count':0,'GPU_kernel_sum_ms':0.,'CPU_scope_sum_ms':0.,'resources':[]})
        for r in ranges:
            if '/' in r['label'] and not r['label'].startswith('controlled/'):
                phase[r['label']]['CPU_scope_sum_ms']+=(r['end']-r['start'])/1e6
        kernels=list(db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'));unattributed=0;intervals=[]
        for r in kernels:
            intervals.append((r['start'],r['end']));launch=calls.get(r['correlationId']);label='unattributed'
            if launch:
                parent=[nv for nv in ranges if nv['globalTid']==launch['globalTid'] and nv['start']<=launch['start']<nv['end']]
                if parent:label=min(parent,key=lambda x:x['end']-x['start'])['label']
            if label=='unattributed':unattributed+=1
            phase[label]['GPU_kernel_count']+=1;phase[label]['GPU_kernel_sum_ms']+=(r['end']-r['start'])/1e6
            name=names.get(r['shortName'],'')
            if 'main' in name:
                phase[label]['resources'].append({'name':name,'duration_ms':(r['end']-r['start'])/1e6,
                    'registers_per_thread':r['registersPerThread'],'static_shared_bytes':r['staticSharedMemory'],'dynamic_shared_bytes':r['dynamicSharedMemory'],
                    'grid':[r['gridX'],r['gridY'],r['gridZ']],'threads':r['blockX']*r['blockY']*r['blockZ']})
        active=0;end=0
        for a,b in sorted(intervals):active+=max(0,b-max(a,end));end=max(end,b)
        span=max(b for a,b in intervals)-min(a for a,b in intervals)
        result.append({'file':str(file),'phases':dict(phase),'kernel_count':len(kernels),'unattributed_kernels':unattributed,
            'GPU_active_interval_union_ms':active/1e6,'GPU_span_ms':span/1e6,'timeline_busy_fraction':active/span,
            'busy_fraction_scope':'Union of GPU kernel intervals / first-start to last-end. Not SM occupancy or Tensor Core utilization.',
            'API_counts':dict(api_counts),'API_sum_ms':{k:v/1e6 for k,v in api_ns.items()},
            'resources_scope':'Launch attributes only; achieved occupancy, spills and stalls unmeasured'})
        db.close()
    args.output.write_text(json.dumps(result,indent=2));print('Attributed',len(result),'traces')


if __name__=='__main__':main()
