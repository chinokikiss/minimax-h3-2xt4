"""CPU-only mounted-dataset throughput sweep with bounded reader memory."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import threading
import time

MIB=2**20


def memory():
    status=Path('/proc/self/status').read_text()
    rss=next(int(s.split()[1])*1024 for s in status.splitlines() if s.startswith('VmRSS:'))
    base=Path('/sys/fs/cgroup');values={}
    for name in ('memory.current','memory.max','memory.stat','memory.pressure'):
        try:values[name]=(base/name).read_text()
        except OSError:values[name]=None
    return dict(rss=rss,cgroup=values)


def inventory(root):
    rows=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and not path.is_symlink() and path.stat().st_size>=MIB:
            with path.open('rb',buffering=0) as f:sample=f.read(4096)
            rows.append(dict(path=str(path),bytes=path.stat().st_size,
                head4096_sha256=hashlib.sha256(sample).hexdigest(),device=path.stat().st_dev))
    return rows


def trial(groups,workers,seconds,cap,drop_cache):
    # At most one 1 MiB readinto buffer per worker, no payload-bearing futures.
    extents=[]
    for files in groups:
        queue=deque((r['path'],offset,min(64*MIB,r['bytes']-offset))
            for r in files for offset in range(0,r['bytes'],64*MIB))
        random.Random(20261030).shuffle(queue);extents.append(queue)
    lock=threading.Lock();barrier=threading.Barrier(workers+1);stop=threading.Event()
    before=memory();samples=[];errors=[];reserved=0;start=0
    def reader(lane):
        nonlocal reserved
        buf=bytearray(MIB);total=0;hints=0;per_dataset=[0]*len(groups);group=lane%len(groups)
        barrier.wait()
        try:
            while not stop.is_set() and time.monotonic()-start<seconds:
                with lock:
                    if not any(extents) or reserved>=cap:break
                    for _ in groups:
                        if extents[group]:break
                        group=(group+1)%len(groups)
                    selected=group;group=(group+1)%len(groups)
                    path,offset,length=extents[selected].popleft();length=min(length,cap-reserved);reserved+=length
                with open(path,'rb',buffering=0) as stream:
                    stream.seek(offset);done=0
                    while done<length and not stop.is_set() and time.monotonic()-start<seconds:
                        n=stream.readinto(memoryview(buf)[:min(len(buf),length-done)])
                        if not n:break
                        done+=n;total+=n;per_dataset[selected]+=n
                    if drop_cache and hasattr(os,'posix_fadvise'):
                        try:os.posix_fadvise(stream.fileno(),offset,done,os.POSIX_FADV_DONTNEED);hints+=1
                        except OSError:pass
        except Exception as error:
            with lock:errors.append(type(error).__name__)
            stop.set()
        return total,hints,per_dataset
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(reader,i) for i in range(workers)]
        start=time.monotonic();barrier.wait()
        while not all(f.done() for f in futures):
            row=memory();samples.append(row)
            if row['rss']>before['rss']+256*MIB:
                errors.append('RSS_growth_guard');stop.set()
            time.sleep(.2)
        totals=[f.result() for f in futures]
    elapsed=time.monotonic()-start;count=sum(r[0] for r in totals)
    return dict(workers=workers,seconds=elapsed,bytes=count,mib_per_second=count/MIB/elapsed,
        buffer_cap_bytes=workers*MIB,baseline_rss=before['rss'],
        peak_rss=max([before['rss']]+[r['rss'] for r in samples]),
        drop_cache_hint=drop_cache,successful_dontneed_hints=sum(r[1] for r in totals),errors=errors,
        dataset_count=len(groups),per_dataset_bytes=[sum(r[2][i] for r in totals) for i in range(len(groups))],
        samples=samples,unique_extent_assignment_within_trial=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--roots',nargs='+',required=True);a=p.parse_args()
    assert not list(Path('/dev').glob('nvidia[0-9]*')),'CPU instance required'
    a.output.mkdir(parents=True,exist_ok=True)
    datasets=[inventory(Path(r)) for r in a.roots]
    state=dict(status='running',python=platform.python_version(),cpu_count=os.cpu_count(),
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text(),dataset_roots=a.roots,datasets=datasets,trials=[],
        mountinfo=Path('/proc/self/mountinfo').read_text(),
        caveats=['Mounted binary-read proxy, not a T4 measurement or network downloader benchmark.',
            'DONTNEED is advisory; neither first pass nor hints prove physically cold reads.',
            'Kernel/file/FUSE/backend caches and shared host activity remain uncontrolled.',
            'RSS buffer bounds do not eliminate kernel page-cache usage; cgroup file/anon counters are retained.'])
    def save():(a.output/'result.json').write_text(json.dumps(state,indent=2))
    save()
    if len(datasets)!=6 or not all(datasets):
        state.update(status='blocked',blocker='Six nonempty mounted datasets with binary files >=1MiB are required');save();return
    # Same per-trial cap and deadline across single-dataset and mixed cohorts.
    began=time.monotonic();cohorts=[(str(n)+'_datasets',datasets[:n]) for n in (1,2,4,6)]
    cases=[(label,groups,w,repeat,True) for repeat in range(2) for label,groups in cohorts for w in (1,2,4,8,16,32,64)]
    cases += [('single_'+str(i),[groups],1,repeat,True) for i,groups in enumerate(datasets[1:],1) for repeat in range(2)]
    random.Random(20261030).shuffle(cases)
    cases += [(label,groups,w,0,False) for label,groups in cohorts for w in (1,64)]
    for label,groups,w,repeat,hint in cases:
        if time.monotonic()-began>1020:state['deadline_reached']=True;break
        row=trial(groups,w,12,4*2**30,hint)|dict(cohort=label,repeat=repeat)
        state['trials'].append(row);save()
        print('READ_TRIAL',label,w,repeat,round(row['mib_per_second'],2),flush=True)
        if row['errors']:state['stopped_on_error']=True;break
    state.update(status='complete',seconds=time.monotonic()-began)
    state['median_curves']={label:[dict(workers=w,mib_per_second_median=statistics.median(values),measurements=len(values))
        for w in (1,2,4,8,16,32,64)
        if (values:=[r['mib_per_second'] for r in state['trials']
            if r['cohort']==label and r['workers']==w and r['drop_cache_hint'] and not r['errors']])]
        for label,_ in cohorts}
    save()


if __name__=='__main__':main()
