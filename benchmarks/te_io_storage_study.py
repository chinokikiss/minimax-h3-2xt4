"""Large-working-set physical I/O controls, separate from inference timings."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import te_io_telemetry as telemetry
from te_io_probe import pressure_lease
from te_shard_io import read,layout,MODES


def state_hashes(state,torch):
    return {k:dict(dtype=str(t.dtype),shape=list(t.shape),sha256=hashlib.sha256(
        memoryview(t.reshape(-1).view(torch.uint8).numpy()).cast('B')).hexdigest()) for k,t in state.items()}


def download(directory,out):
    import requests
    from huggingface_hub import get_hf_file_metadata,hf_hub_url
    folder=ROOT/'semantic_results/kaggle_hf_publication_v6'
    publication=json.loads((folder/'publication.json').read_text())
    receipts=json.loads((folder/'upload_receipts.json').read_text())['objects']
    native=json.loads((ROOT/'semantic_results/offline_conversion_plan.json').read_text())['blocks']
    def one(name):
        expected=receipts[name];stem=name.removesuffix('.safetensors')
        manifest=json.loads((folder/(stem+'.json')).read_text())
        if not manifest['native_tensor_bit_exact'] or manifest['state_sha256']!=native[stem]['state_sha256']:
            raise ValueError('Native provenance mismatch')
        try:
            meta=get_hf_file_metadata(hf_hub_url(publication['repo'],name,revision=publication['commit']),token=False)
            if meta.size!=expected['bytes'] or meta.etag!=expected['sha256']:raise ValueError('Immutable object mismatch')
            path=directory/name;sha=hashlib.sha256();size=0
            with requests.get(meta.location,stream=True,timeout=(15,90)) as response:
                if response.status_code!=200:raise RuntimeError('Download HTTP status '+str(response.status_code))
                with path.open('wb') as stream:
                    for chunk in response.iter_content(2**20):stream.write(chunk);sha.update(chunk);size+=len(chunk)
                    stream.flush();os.fdatasync(stream.fileno())
            if size!=expected['bytes'] or sha.hexdigest()!=expected['sha256']:raise ValueError('Download checksum mismatch')
            path.with_suffix('.json').write_text(json.dumps(manifest))
            return dict(name=name,bytes=size,sha256=sha.hexdigest())
        except Exception as error:raise RuntimeError('TE shard download failed: '+name+' '+type(error).__name__) from None
    names=[f'te.block{b:02d}.rank{r}.safetensors' for b in range(50) for r in (0,1)]
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(one,names):
            rows.append(result)
            if len(rows)%20==0:print('VERIFIED TE storage download',len(rows),'/100',flush=True)
    (out/'source_identity.json').write_text(json.dumps(dict(publication=publication,objects=rows),indent=2))


def worker(rank,directory,commands,results,go):
    import torch
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    observer=telemetry.ReadObserver(period=.1)
    results.put(dict(ready=rank,pid=os.getpid()))
    try:
        while True:
            case=commands.get()
            if case is None:break
            go.wait();rows=[]
            for block in case['blocks']:
                path=directory/f'te.block{block:02d}.rank{rank}.safetensors'
                expected=json.loads(path.with_suffix('.json').read_text())['state_sha256']
                with path.open('rb',buffering=0) as stream:
                    if case.get('inode_drop'):
                        os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                    resident=telemetry.residency(stream.fileno())
                before=telemetry.snapshot();tid=threading.get_native_id()
                samples=dict(states=Counter(),wchans=Counter(),stacks=[],stack_unavailable_samples=0)
                with observer.lock:observer.active[tid]=samples
                begin=time.monotonic_ns();cpu=time.thread_time_ns()
                try:state=read(path,list(expected),case['mode'])
                finally:
                    end=time.monotonic_ns();cpu_ns=time.thread_time_ns()-cpu
                    with observer.lock:observer.active.pop(tid,None)
                after=telemetry.snapshot();counters=telemetry.delta(before,after)
                if state_hashes(state,torch)!=expected:raise ValueError('Candidate differs from Native tensor proof')
                rows.append(dict(rank=rank,block=block,start_ns=begin,end_ns=end,wall_ns=end-begin,cpu_ns=cpu_ns,
                    logical_bytes=sum(t.numel()*t.element_size() for t in state.values()),
                    source_residency_before=resident,thread_io=counters['io'],usage=counters['usage'],
                    schedstat=counters.get('schedstat'),
                    sampled_waits={k:dict(v) if isinstance(v,Counter) else v for k,v in samples.items()},
                    native_tensor_bit_exact=True))
                del state;gc.collect()
            results.put(dict(rank=rank,passed=True,blocks=rows))
    except BaseException as error:
        import traceback
        results.put(dict(rank=rank,passed=False,error=type(error).__name__+': '+str(error),traceback=traceback.format_exc()))
    finally:observer.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--directory',type=Path,default=Path('/tmp/te_storage_shards'))
    parser.add_argument('--download',action='store_true');args=parser.parse_args()
    import torch
    if os.name!='posix' or torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)):
        raise RuntimeError('Actual Linux dual-T4 host required; this is CPU/storage evidence, not GPU inference timing')
    args.output.mkdir(parents=True,exist_ok=True);args.directory.mkdir(parents=True,exist_ok=True)
    if args.download:download(args.directory,args.output)
    capabilities=telemetry.capabilities();capabilities['mountinfo']=telemetry.text('/proc/self/mountinfo')
    capabilities['block_topology']={}
    for path in Path('/sys/block').glob('*'):
        capabilities['block_topology'][path.name]=dict(dev=telemetry.text(path/'dev'),
            slaves=[p.name for p in (path/'slaves').glob('*')],
            rotational=telemetry.text(path/'queue/rotational'),
            read_ahead_kb=telemetry.text(path/'queue/read_ahead_kb'),
            nr_requests=telemetry.text(path/'queue/nr_requests'))
    capabilities['file_device']=os.stat(args.directory/'te.block10.rank0.safetensors').st_dev
    (args.output/'capabilities.json').write_text(json.dumps(capabilities,indent=2))
    layouts={}
    for rank in (0,1):
        p=args.directory/f'te.block10.rank{rank}.safetensors'
        layouts[str(rank)]=layout(p,list(json.loads(p.with_suffix('.json').read_text())['state_sha256']))
    (args.output/'layout.json').write_text(json.dumps(layouts,indent=2))
    context=mp.get_context('spawn');go=context.Event();results=context.Queue();commands=[context.Queue() for _ in (0,1)]
    processes=[context.Process(target=worker,args=(r,args.directory,commands[r],results,go)) for r in (0,1)]
    evidence=dict(status='running',physical=True,gpus=[torch.cuda.get_device_name(i) for i in (0,1)],cases=[],
        scope='Byte-verified actual 24.4 GB TE shard working set; I/O controls only, not inference speedups',
        interpretation='No global cache drop. Inode DONTNEED is not physical-cold. Shared device/cgroup counters are sampled once per common epoch and layered devices must not be added.',
        instrumentation='One worker/rank maximum; whole-block mincore/OS snapshots outside read latency; tensor hashes after read timing. Phase wall includes verification and is not pure storage service time.')
    save=lambda:(args.output/'storage_controls.json').write_text(json.dumps(evidence,separators=(',',':')))
    save();buffers=[]
    try:
        for p in processes:p.start()
        for _ in processes:
            if 'ready' not in results.get(timeout=120):raise RuntimeError('Worker startup failure')
        def run(name,mode='baseline',blocks=None,serial=False,inode_drop=False):
            before=telemetry.snapshot();start=time.monotonic_ns();rows=[]
            case=dict(name=name,mode=mode,blocks=list(range(50)) if blocks is None else blocks,inode_drop=inode_drop)
            groups=((0,),(1,)) if serial else ((0,1),)
            for ranks in groups:
                go.clear()
                for rank in ranks:commands[rank].put(case)
                go.set()
                for _ in ranks:
                    row=results.get(timeout=900)
                    if not row['passed']:raise RuntimeError('Read control failed: '+str(row))
                    rows.append(row)
            end=time.monotonic_ns();after=telemetry.snapshot()
            record=dict(case=case,serial=serial,start_ns=start,end_ns=end,phase_wall_s=(end-start)/1e9,
                before=before,after=after,ranks=rows)
            evidence['cases'].append(record);save()
            payload=sum(b['logical_bytes'] for r in rows for b in r['blocks'])
            print('AUDIT STORAGE CASE',name,'phase_seconds',round(record['phase_wall_s'],3),
                'logical_GB',round(payload/1e9,3),'rank_read_wall_s',
                [round(sum(b['wall_ns'] for b in r['blocks'])/1e9,3) for r in rows],flush=True)
        run('small_inode_drop',blocks=[10,25],inode_drop=True)
        run('small_immediate_warm',blocks=[10,25])
        run('full_natural_low_anon')
        run('full_repeat_low_anon')
        buffers,lease=pressure_lease(16);evidence['anonymous_control']=lease;save()
        run('full_natural_pressure')
        run('full_repeat_pressure')
        # Counterbalanced identical full scans: no extra I/O worker concurrency.
        for repetition,modes in enumerate((MODES,tuple(reversed(MODES)))):
            for mode in modes:run(f'pressure_{mode}_r{repetition}',mode)
        run('pressure_serial_both_ranks',serial=True)
        buffers.clear();gc.collect();run('pressure_released_full')
        evidence.update(status='complete',final=telemetry.snapshot());save()
    except BaseException as error:
        evidence.update(status='failed',error=type(error).__name__+': '+str(error));save();raise
    finally:
        buffers.clear()
        for command in commands:command.put(None)
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():p.terminate();p.join(timeout=10)
    print('AUDIT STORAGE COMPLETE',len(evidence['cases']),flush=True)


if __name__=='__main__':main()
