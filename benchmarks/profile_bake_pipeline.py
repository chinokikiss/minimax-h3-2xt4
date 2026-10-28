"""Six-layer AB/BA bake comparison: real shards, Native byte proofs, bounded IO."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import functools
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import threading
import time


class Recorder:
    def __init__(self):self.local=threading.local();self.events=[];self.restore=[];self.lock=threading.Lock()

    @contextmanager
    def span(self,name):
        stack=getattr(self.local,'stack',None)
        if stack is None:stack=self.local.stack=[]
        row=dict(phase=name,thread=threading.get_ident(),start_ns=time.monotonic_ns(),children=0)
        start_cpu=time.thread_time_ns();stack.append(row)
        try:yield
        finally:
            row['end_ns']=time.monotonic_ns();row['thread_cpu_ns']=time.thread_time_ns()-start_cpu
            row['wall_ns']=row['end_ns']-row['start_ns'];row['exclusive_wall_ns']=row['wall_ns']-row.pop('children')
            stack.pop()
            if stack:stack[-1]['children']+=row['wall_ns']
            with self.lock:self.events.append(row)

    def wrap(self,module,name,label):
        fn=getattr(module,name)
        @functools.wraps(fn)
        def call(*args,**kwargs):
            with self.span(label):return fn(*args,**kwargs)
        self.restore.append((module,name,fn));setattr(module,name,call)

    def close(self):
        for module,name,fn in reversed(self.restore):setattr(module,name,fn)


def worker(base,lora,output,layers,device,pipeline,profile):
    import torch,resource
    from patches import tp2_disk_bake as baker,tp2_bundle,offline_lora,offline_lora_reference
    from patches.windowed_tp import CheckpointReader
    torch.cuda.set_device(device);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    recorder=Recorder();profile=Path(profile);profile.mkdir(parents=True,exist_ok=True)
    for mod,name,label in ((baker,'_persist_layer','persist_layer'),(baker,'state_proof','state_hash'),
        (baker,'sha','output_file_hash'),(tp2_bundle,'sha','base_file_hash'),
        (baker.os,'fsync','fsync'),(baker,'save_file','serialize_write'),
        (CheckpointReader,'read_tensor','checkpoint_read'),(offline_lora,'merged_weight','merge'),
        (offline_lora_reference,'verify_weight','independent_verification')):recorder.wrap(mod,name,label)
    original=offline_lora.merged_weight;captured=[False]
    def first_merge(*args,**kwargs):
        if captured[0]:return original(*args,**kwargs)
        captured[0]=True
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as cap:
            value=original(*args,**kwargs);torch.cuda.synchronize()
        path=profile/'first_merge.json';cap.export_chrome_trace(str(path))
        trace=json.loads(path.read_text());kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel']
        assert kernels,'Incomplete worker CUDA capture'
        path.with_suffix('.json.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0));path.unlink()
        (profile/'trace_check.json').write_text(json.dumps(dict(kernels=len(kernels),kernel_work_ms=sum(e['dur'] for e in kernels)/1000,flush_complete=True)))
        return value
    offline_lora.merged_weight=first_merge
    start=time.monotonic()
    try:
        with recorder.span('worker_total'):
            result=baker._bake_layers(base,((lora,1.),),output,layers,device,'operator-batch-layer-proof-v1',pipeline=pipeline)
        result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gpu_peak_allocated=torch.cuda.max_memory_allocated(),events=recorder.events,
            cpu_count=len(__import__('os').sched_getaffinity(0)),
            ownership='At most one layer queued for writer plus current layer; CPU-only writer')
        (profile/'result.json').write_text(json.dumps(result,indent=2));return result
    finally:offline_lora.merged_weight=original;recorder.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--lora',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected',type=Path,required=True)
    args=p.parse_args();args.scratch.mkdir(exist_ok=False);args.output.mkdir(exist_ok=True)
    expected=json.loads(args.expected.read_text());layers=((0,24,48),(1,25,49));cases=[]
    # Reversed second order reduces a simple serial-first page-cache/order bias.
    for trial,order in enumerate(((False,True),(True,False))):
      for pipeline in order:
        label=f'trial{trial}_'+('pipeline' if pipeline else 'serial')
        destination=args.scratch/label;destination.mkdir();profile=args.output/label
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
            jobs=[pool.submit(worker,str(args.base),str(args.lora),str(destination),owned,f'cuda:{rank}',pipeline,
                str(profile/f'rank{rank}')) for rank,owned in enumerate(layers)]
            results=[j.result() for j in jobs]
        seconds=time.monotonic()-start;records={k:v for row in results for k,v in row['shards'].items()}
        assert len(records)==12 and sum(len(row['verified_targets']) for row in results)==24
        assert all(value['state_sha256']==expected['shards'][name] for name,value in records.items())
        cases.append(dict(label=label,pipeline=pipeline,wall_including_process_startup_seconds=seconds,
            worker_seconds=[r['seconds'] for r in results],all_12_shards_match_prior_t4=True,
            independent_native_targets=24,shards=records))
        (args.output/'result.json').write_text(json.dumps(dict(status='running',cases=cases),indent=2))
        print('BAKE_LAYER_CASE',label,seconds,flush=True)
        # Generated benchmark payloads only; proofs/traces remain in final outputs.
        if destination.resolve().parent!=args.scratch.resolve():raise RuntimeError('Cleanup escaped owned scratch')
        __import__('shutil').rmtree(destination)
    for row in cases[1:]:
        assert all(value['state_sha256']==cases[0]['shards'][name]['state_sha256'] for name,value in row['shards'].items())
    (args.output/'result.json').write_text(json.dumps(dict(status='complete',cases=cases,
        scope='Six real trained layers, AB/BA two-GPU bake; not a full 50-layer bake or parent-tail benchmark',
        bytes_preserved=True,profiling_overhead_included=True),indent=2))


if __name__=='__main__':main()
