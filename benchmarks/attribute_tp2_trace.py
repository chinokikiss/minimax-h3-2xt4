"""Process-safe Systems attribution for simultaneous TP ranks; CPU analysis."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def union_ms(intervals):
    total=0;end=0
    for a,b in sorted(intervals):total+=max(0,b-max(a,end));end=max(b,end)
    return total/1e6


def main():
    ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+',type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    results=[]
    for path in args.files:
        db=sqlite3.connect(path);db.row_factory=sqlite3.Row
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")};names=dict(db.execute('select id,value from StringIds'))
        ranges=defaultdict(list)
        for row in db.execute('select * from NVTX_EVENTS where end>start'):
            r=dict(row);r['label']=r['text'] or names.get(r['textId'],'');ranges[r['globalTid']].append(r)
        calls={};api=defaultdict(lambda:{'count':0,'CPU_sum_ms':0.})
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
            if table not in tables:continue
            for row in db.execute('select * from '+table):
                r=dict(row);pid=r['globalTid']&~((1<<24)-1)
                key=(pid,r['correlationId']);calls.setdefault(key,r)
                ak=(pid,names.get(r['nameId'],'unknown'));api[ak]['count']+=1;api[ak]['CPU_sum_ms']+=(r['end']-r['start'])/1e6
        phases=defaultdict(lambda:{'kernel_count':0,'kernel_sum_ms':0.,'kernels':defaultdict(lambda:{'count':0,'sum_ms':0.}),'resources':[]})
        intervals=defaultdict(list);matched=0;unmatched=0;pid_device={}
        for row in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
            r=dict(row);device=r['deviceId'];pid_device[r['globalPid']]=device;intervals[device].append((r['start'],r['end']))
            launch=calls.get((r['globalPid'],r['correlationId']));parent=[]
            if launch:parent=[nv for nv in ranges[launch['globalTid']] if nv['start']<=launch['start']<nv['end']]
            semantic=[nv for nv in parent if nv['label'].startswith(('NCCL/','linear/','row_fusion/')) or '/QK_softmax_PV' in nv['label']]
            inner=min(semantic or parent,key=lambda x:x['end']-x['start'])['label'] if parent else 'unattributed'
            outer=next((nv['label'] for nv in sorted(parent,key=lambda x:x['end']-x['start'],reverse=True) if nv['label'].endswith('/API')),'no_API_range')
            key=(device,outer,inner);p=phases[key];p['kernel_count']+=1;ms=(r['end']-r['start'])/1e6;p['kernel_sum_ms']+=ms
            name=names.get(r['shortName'],'unknown');p['kernels'][name]['count']+=1;p['kernels'][name]['sum_ms']+=ms
            if parent:matched+=1
            else:unmatched+=1
            if name in ('main','main_kernel') or 'QK_softmax_PV' in inner:
                p['resources'].append({'registers_per_thread':r['registersPerThread'],'static_shared_bytes':r['staticSharedMemory'],
                    'dynamic_shared_bytes':r['dynamicSharedMemory'],'local_bytes_per_thread':r['localMemoryPerThread'],
                    'threads':r['blockX']*r['blockY']*r['blockZ'],'duration_ms':ms})
        timeline=[]
        for device,iv in intervals.items():
            span=(max(b for a,b in iv)-min(a for a,b in iv))/1e6
            timeline.append({'device':device,'span_ms':span,'kernel_active_union_ms':union_ms(iv),'timeline_busy_fraction':union_ms(iv)/span})
        copies=[]
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            copies=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')]
        results.append({'file':str(path),'matched':matched,'unattributed':unmatched,'timeline':timeline,
            'phases':[{'device':key[0],'outer_API':key[1],'phase':key[2],**v} for key,v in sorted(phases.items())],
            'CPU_APIs':[{'device':pid_device.get(key[0]),'name':key[1],**v} for key,v in api.items()],
            'memcpy_count':len(copies),'memcpy_bytes':sum(r.get('bytes',0) for r in copies),
            'limitations':'Kernel interval coverage is not achieved occupancy/SM utilization. Launch resources are not measured stalls or spills. Process-qualified correlation IDs prevent cross-rank collisions.'})
        db.close()
    args.output.write_text(json.dumps(results,indent=2));print('Attributed',len(results),'TP2 traces')


if __name__=='__main__':main()
