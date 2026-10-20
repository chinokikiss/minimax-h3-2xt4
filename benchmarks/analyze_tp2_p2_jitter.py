"""Per-call observational FC2 attribution; no hardware stall inference."""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import numpy as np

ROOT = Path('D:/attention_eval/tp2_p2_20261018/transport-confirm/v1/p2')


def stats(v):
    return dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                p90=float(np.percentile(v, 90)), minimum=float(min(v)), maximum=float(max(v)))


def main():
    db = sqlite3.connect(ROOT/'profile.sqlite'); db.row_factory = sqlite3.Row
    names = dict(db.execute('select id,value from StringIds'))
    ranges = defaultdict(list); apis = defaultdict(list); gc = []
    for row in db.execute('select * from NVTX_EVENTS where end>start'):
        r = dict(row); r['label'] = r['text'] or names.get(r['textId'], '')
        ranges[r['globalTid']].append(r)
        if r['label'].endswith('/API'): apis[r['globalTid'] & ~((1 << 24)-1)].append(r)
        if r['label'].startswith('CPU_GC'): gc.append(r)
    calls = {}; alloc = []
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables: continue
        for row in db.execute('select * from '+table):
            r = dict(row); pid = r['globalTid'] & ~((1 << 24)-1)
            calls.setdefault((pid, r['correlationId']), r)
            if any(s in names.get(r['nameId'], '') for s in ('Malloc', 'Free', 'MemAlloc', 'MemFree')):
                alloc.append(dict(name=names[r['nameId']], CPU_ms=(r['end']-r['start'])/1e6))
    per_call = {}; pid_device = {}
    for row in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
        r = dict(row); pid_device[r['globalPid']] = r['deviceId']
        launch = calls.get((r['globalPid'], r['correlationId']))
        if not launch: continue
        parent = [n for n in ranges[launch['globalTid']] if n['start'] <= launch['start'] < n['end']]
        outer = next((n for n in parent if n['label'].endswith('/API')), None)
        if outer is None: continue
        key = (r['deviceId'], outer['start'])
        rec = per_call.setdefault(key, dict(device=r['deviceId'], name=outer['label'].split('/')[-2],
            CPU_enqueue_ms=(outer['end']-outer['start'])/1e6, phases=defaultdict(float), intervals=[]))
        semantic = [n for n in parent if n['label'].startswith(('NCCL/', 'linear/', 'row_fusion/'))]
        phase = min(semantic or parent, key=lambda n: n['end']-n['start'])['label']
        rec['phases'][phase] += (r['end']-r['start'])/1e6
        rec['intervals'].append((r['start'], r['end']))
    output = dict(scope='Profiled synchronized calls only. Phase covariation is observational; physical causes need counters or a controlled follow-up.',
                  CPU_GC_ranges=len(gc), CUDA_allocation_free_calls=alloc, variants=[])
    for device in (0, 1):
        raw = json.loads((ROOT/'profiled'/f'rank{device}.json').read_text())
        for case in raw['cases']:
            rows = [r for _, r in sorted(per_call.items()) if r['device']==device and r['name']==case['name']]
            assert len(rows)==len(case['API']['gpu_ms'])==30
            phases = sorted({p for r in rows for p in r['phases']})
            result = dict(device=device, name=case['name'], event_ms=stats(case['API']['gpu_ms']),
                          wall_ms=stats(case['API']['wall_ms']), CPU_enqueue_ms=stats([r['CPU_enqueue_ms'] for r in rows]), phases={})
            for phase in phases:
                v = [r['phases'].get(phase, 0.) for r in rows]
                result['phases'][phase] = dict(**stats(v),
                    correlation_with_event=float(np.corrcoef(v,case['API']['gpu_ms'])[0,1]) if np.std(v)>0 else None)
            gaps=[]
            for row in rows:
                iv = sorted(row.pop('intervals')); total=0; end=0
                for a,b in iv: total+=max(0,b-max(a,end)); end=max(b,end)
                gaps.append((max(b for a,b in iv)-iv[0][0]-total)/1e6)
            result['within_call_kernel_interval_gaps_ms']=stats(gaps)
            output['variants'].append(result)
    (ROOT/'jitter_attribution.json').write_text(json.dumps(output, indent=2))
    print('FC2_JITTER_ATTRIBUTION',len(output['variants']),'rank/variant records')


if __name__=='__main__':main()
