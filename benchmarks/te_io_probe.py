"""Controlled rank-local TE I/O probes on the same physical pipeline host.

These measure storage/preparation, not TP inference or GPU speedups. Each
tensor is checked against the immutable Native-qualified shard manifest.
File-specific cache advice never deletes a file or changes global VM policy.
"""
import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import te_io_telemetry as telemetry


def hashes(state,torch):
    return {k:dict(dtype=str(t.dtype),shape=list(t.shape),sha256=hashlib.sha256(
        memoryview(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy()).cast('B')).hexdigest())
        for k,t in state.items()}


def worker(rank,directory,output,commands,results,start_event):
    import torch
    from patches.windowed_tp import CheckpointReader
    torch.set_num_threads(1)
    observer=telemetry.ReadObserver()
    class Recorder:
        io_observer=observer
        rows=[]
        def save_read(self,row,path):row['source']=str(path);self.rows.append(row)
    recorder=Recorder();original=Path.open;enabled=False;active_advice=None
    def observed(path,*a,**k):
        file=original(path,*a,**k)
        mode=a[0] if a else k.get('mode','r')
        if enabled and mode=='rb' and path.name.startswith('te.block') and path.suffix=='.safetensors':
            # SEQUENTIAL belongs to an open file description. Apply it to
            # the descriptor that actually reads, not a closed setup handle.
            if active_advice=='sequential':
                os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_SEQUENTIAL)
            return telemetry.ObservedFile(file,path,recorder)
        return file
    Path.open=observed
    destination_buffers={}
    try:
        results.put(dict(ready=rank,pid=os.getpid()))
        while True:
            case=commands.get()
            if case is None:break
            start_event.wait();data=[];active_advice=case.get('advice')
            for block in case['blocks']:
                path=directory/f'te.block{block:02d}.rank{rank}.safetensors'
                manifest=json.loads(path.with_suffix('.json').read_text())
                with original(path,'rb',buffering=0) as file:
                    advice={}
                    if case['cache']=='evict':
                        # Ensure our completed downloaded file is clean before
                        # best-effort eviction; measure eviction outside reads.
                        os.fdatasync(file.fileno())
                        os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                    if case.get('advice')=='sequential':
                        os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_SEQUENTIAL)
                    if case.get('advice')=='willneed':
                        before=time.monotonic_ns()
                        os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_WILLNEED)
                        time.sleep(case['lead_s'])
                        advice=dict(lead_ns=time.monotonic_ns()-before,requested='WILLNEED',
                            interpretation='Best-effort; residency observed, not assumed')
                    initial=telemetry.residency(file.fileno())
                reader=CheckpointReader(path);recorder.rows=[]
                state={};start=time.monotonic_ns();cpu=time.thread_time_ns();enabled=True
                try:
                    for name in manifest['state_sha256']:
                        if case['reader']=='native':state[name]=reader.read_tensor(name)
                        else:
                            # Same immutable bytes into reused, previously
                            # touched final CPU buffers: isolate destination
                            # allocation/minor-fault costs from source readiness.
                            info=reader.header[name];a,b=info['data_offsets']
                            key=(block,name);tensor=destination_buffers.get(key)
                            if tensor is None:
                                tensor=torch.empty(info['shape'],dtype=reader.DTYPES[info['dtype']])
                                tensor.zero_();destination_buffers[key]=tensor
                            buffer=memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast('B')
                            with path.open('rb',buffering=0) as file:
                                file.seek(reader.offset+a);remaining=buffer
                                while remaining:
                                    count=file.readinto(remaining)
                                    if not count:raise IOError('Incomplete shard')
                                    remaining=remaining[count:]
                            if len(buffer)!=b-a:raise ValueError('Tensor extent changed')
                            state[name]=tensor
                finally:
                    end=time.monotonic_ns();cpu_ns=time.thread_time_ns()-cpu;enabled=False
                # Hashes never go in the read interval. Every probe variant
                # must produce exactly the saved Native-qualified tensor state.
                if hashes(state,torch)!=manifest['state_sha256']:raise ValueError('Read probe changed Native shard bytes')
                record=dict(rank=rank,block=block,start_ns=start,end_ns=end,wall_ns=end-start,
                    cpu_ns=cpu_ns,read_wall_ns=sum(r['wall_ns'] for r in recorder.rows),
                    read_cpu_ns=sum(r['cpu_ns'] for r in recorder.rows),initial_residency=initial,
                    advice=advice,correct=True,reads=recorder.rows)
                data.append(record);del state,reader;gc.collect()
            results.put(dict(case=case['name'],rank=rank,passed=True,blocks=data))
    except BaseException as error:
        results.put(dict(rank=rank,passed=False,error=type(error).__name__+': '+str(error),traceback=traceback.format_exc()))
    finally:observer.close();Path.open=original


def pressure_lease(target_gib):
    """Bounded, touched anonymous memory; retain conservative RAM headroom."""
    buffers=[];step=128*2**20;target=int(target_gib*2**30);reason='target reached'
    before=telemetry.snapshot();mem_events=before['memory_events']
    while len(buffers)*step<target:
        memory=telemetry.counters('/proc/meminfo');cg=telemetry.CGROUP
        stats=telemetry.counters(cg/'memory.stat') if cg else {}
        limit=telemetry.text(cg/'memory.max') if cg else None
        available=memory.get('MemAvailable',0)*1024
        if available<8*2**30+step:reason='8 GiB system available headroom';break
        if limit and limit!='max':
            # memory.current includes reclaimable page cache. Bound persistent
            # anonymous/shmem/kernel charges independently and leave 8 GiB.
            nonreclaimable=stats.get('anon',0)+stats.get('shmem',0)+stats.get('kernel',0)
            if int(limit)-nonreclaimable<8*2**30+step:
                reason='8 GiB cgroup non-file headroom';break
        item=bytearray(step)
        for i in range(0,step,4096):item[i]=1
        buffers.append(item)
        current=telemetry.counters(cg/'memory.events') if cg else {}
        if any(current.get(k,0)>mem_events.get(k,0) for k in ('oom','oom_kill')):
            buffers.clear();raise RuntimeError('Unexpected OOM counter increment; pressure experiment aborted')
    return buffers,dict(bytes=len(buffers)*step,requested_gib=target_gib,reason=reason,
        before=before,after=telemetry.snapshot(),semantics='Synthetic anonymous retention; not an inference optimization')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--blocks',default='10,25')
    parser.add_argument('--repeats',type=int,default=3);args=parser.parse_args()
    if os.name!='posix':raise RuntimeError('Physical Linux host required')
    args.output.mkdir(parents=True,exist_ok=True);blocks=[int(v) for v in args.blocks.split(',')]
    for rank in (0,1):
        for block in blocks:
            if not (args.directory/f'te.block{block:02d}.rank{rank}.safetensors').is_file():raise FileNotFoundError('Probe shard missing')
    initial=telemetry.snapshot();capabilities=telemetry.capabilities()
    capabilities['mountinfo']=telemetry.text('/proc/self/mountinfo')
    capabilities['diskstats']=telemetry.text('/proc/diskstats')
    capabilities['file_devices']={str(p):os.stat(p).st_dev for p in args.directory.glob('te.block10.rank*.safetensors')}
    (args.output/'capabilities.json').write_text(json.dumps(capabilities,indent=2))
    context=mp.get_context('spawn');results=context.Queue();start_event=context.Event()
    commands=[context.Queue() for _ in (0,1)]
    processes=[context.Process(target=worker,args=(r,args.directory,args.output,commands[r],results,start_event)) for r in (0,1)]
    evidence=dict(status='running',scope='Controlled reads of actual Native-qualified rank-local TE shards; no GPU speedup claims',
        blocks=blocks,repeats=args.repeats,cases=[],pressure=[],initial=initial)
    save=lambda:(args.output/'io_probe.json').write_text(json.dumps(evidence,separators=(',',':')))
    save()
    try:
        for process in processes:process.start()
        for _ in processes:
            row=results.get(timeout=120)
            if 'ready' not in row:raise RuntimeError('Probe worker failed initialization')
        def run(name,ranks,cache='evict',reader='native',advice=None,lead_s=0,repeat=0):
            start_event.clear();before=telemetry.snapshot()
            case=dict(name=name,blocks=blocks,cache=cache,reader=reader,advice=advice,lead_s=lead_s,repeat=repeat)
            for rank in ranks:commands[rank].put(case)
            start_event.set();rows=[]
            for rank in ranks:
                row=results.get(timeout=240)
                if not row.get('passed'):raise RuntimeError('Probe worker failure: '+str(row))
                rows.append(row)
            evidence['cases'].append(dict(case=case,ranks=ranks,before=before,after=telemetry.snapshot(),results=rows));save()
            reads=[read for row in rows for block in row['blocks'] for read in block['reads']]
            pages=sum(r['resident_before'].get('pages',0) for r in reads)
            resident=sum(r['resident_before'].get('resident_pages',0) for r in reads)
            print('AUDIT IO CASE',name,json.dumps(dict(
                read_wall_s=round(sum(r['wall_ns'] for r in reads)/1e9,4),
                read_cpu_s=round(sum(r['cpu_ns'] for r in reads)/1e9,4),
                read_bytes=sum(r['counters']['io'].get('read_bytes',0) for r in reads),
                logical_bytes=sum(r.get('returned_bytes',0) for r in reads),
                resident_fraction_before=round(resident/pages,4) if pages else None,
                major_faults=sum(r['counters']['usage'].get('ru_majflt',0) for r in reads),
                D_samples=sum(r['samples']['states'].get('D',0) for r in reads)),separators=(',',':')),flush=True)
        for repeat in range(args.repeats):
            # Counterbalance serial/dual order between repetitions.
            for ranks,label in (([0],'single'),([0,1],'dual')) if repeat%2==0 else (([0,1],'dual'),([0],'single')):
                run(f'{label}_cold_{repeat}',ranks,repeat=repeat)
                run(f'{label}_immediate_reread_{repeat}',ranks,cache='keep',repeat=repeat)
            run(f'dual_destination_reuse_cold_{repeat}',[0,1],reader='reuse',repeat=repeat)
            run(f'dual_destination_reuse_hot_{repeat}',[0,1],cache='keep',reader='reuse',repeat=repeat)
            run(f'dual_sequential_cold_{repeat}',[0,1],advice='sequential',repeat=repeat)
            run(f'dual_willneed_2s_{repeat}',[0,1],advice='willneed',lead_s=2,repeat=repeat)
        # Warm sources first, then observe whether added anonymous memory
        # evicts them. Record residency after the pressure ramp, not just totals.
        run('pressure_before_hot',[0,1],cache='keep')
        buffers,lease=pressure_lease(12);evidence['pressure'].append(lease);save()
        try:
            for repeat in range(args.repeats):
                run(f'pressure_existing_pages_{repeat}',[0,1],cache='keep',repeat=repeat)
                run(f'pressure_cold_{repeat}',[0,1],repeat=repeat)
                run(f'pressure_immediate_reread_{repeat}',[0,1],cache='keep',repeat=repeat)
        finally:buffers.clear();gc.collect()
        run('pressure_released_hot',[0,1],cache='keep')
        evidence.update(status='complete',final=telemetry.snapshot());save()
    finally:
        for command in commands:command.put(None)
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():process.terminate();process.join(timeout=10)
    print('AUDIT IO COMPLETE',len(evidence['cases']),'Native shard byte checks passed',flush=True)


if __name__=='__main__':main()
