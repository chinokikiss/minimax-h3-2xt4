"""CUPTI correlation and per-worker coverage gates; no timing attribution."""
from collections import Counter,defaultdict
import sqlite3


def check(database, prefix='H3_CAPTURE/', minimum=4, require_blocks=False,minimum_per_label=0):
    db=sqlite3.connect(database);db.row_factory=sqlite3.Row
    try:
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        apis=[name for name in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER') if name in tables]
        if not {'StringIds','NVTX_EVENTS','CUPTI_ACTIVITY_KIND_KERNEL'}<=tables or not apis:
            return dict(passed=False,blocker='Required worker CUDA/NVTX tables missing')
        names=dict(db.execute('select id,value from StringIds'))
        ranges=defaultdict(list)
        blocks=defaultdict(list)
        for row in db.execute('select * from NVTX_EVENTS where end>start'):
            text=row['text'] or names.get(row['textId'],'')
            if text.startswith(prefix):ranges[row['globalTid']].append((row['start'],row['end'],text))
            if text.startswith('H3_DIT/') and "/block=('dit', " in text and text.endswith('/block_compute'):
                blocks[row['globalTid']].append((row['start'],row['end'],text))
        launches={}
        for api in apis:
            for row in db.execute('select * from '+api):
                tid=row['globalTid'];pid=tid&~((1<<24)-1)
                selected=[r for r in ranges.get(tid,()) if r[0]<=row['start']<r[1]]
                if selected:launches[(pid,row['correlationId'])]=(tid,row['start'],[r[2] for r in selected])
        counts=Counter();seen_blocks=defaultdict(set);devices=defaultdict(set)
        label_counts=defaultdict(Counter)
        for row in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
            launch=launches.get((row['globalPid'],row['correlationId']))
            if launch:
                tid,start,labels=launch;counts[tid]+=1;devices[tid].add(row['deviceId'])
                label_counts[tid].update(labels)
                for a,b,label in blocks.get(tid,()):
                    if a<=start<b:seen_blocks[tid].add(label)
        workers=[dict(global_tid=tid,pid=(tid>>24)&((1<<24)-1),labels=[r[2] for r in rs],
            correlated_kernels=counts[tid],devices=sorted(devices[tid]),
            label_kernel_counts={r[2]:label_counts[tid][r[2]] for r in rs},
            trained_blocks_with_kernels=len(seen_blocks[tid])) for tid,rs in ranges.items()]
        passed=(len(workers)==2 and len({w['pid'] for w in workers})==2 and
            all(w['correlated_kernels']>=minimum and len(w['devices'])==1 for w in workers) and
            len({w['devices'][0] for w in workers if w['devices']})==2 and
            all(all(n>=minimum_per_label for n in w['label_kernel_counts'].values()) for w in workers) and
            (not require_blocks or all(w['trained_blocks_with_kernels']==50 for w in workers)))
        return dict(passed=passed,workers=workers,api_tables=apis,scope='Coverage qualification only; no performance attribution',
            blocker=None if passed else 'Incomplete rank CUDA correlation / layer coverage')
    finally:db.close()
