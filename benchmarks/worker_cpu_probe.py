"""Opt-in worker CPU call stacks and selected-call deterministic profiles."""
from collections import Counter
import profile as py_profile
import json
import math
import os
from pathlib import Path
import pstats
import sys
import threading
import time


def validate_profile(stats):
    """Reject impossible timings rather than attribute corrupted clock data."""
    errors=[]
    for key,(_,_,own,cumulative,_) in stats.stats.items():
        if (not math.isfinite(own) or not math.isfinite(cumulative) or own < -1e-6
                or cumulative < own-1e-6 or own > stats.total_tt+1e-6):
            errors.append(dict(function=key[2],self_seconds=own,cumulative_seconds=cumulative))
    passed=math.isfinite(stats.total_tt) and stats.total_tt>=0 and not errors
    return dict(passed=passed,total_wall_seconds=stats.total_tt,invalid_entries=len(errors),examples=errors[:5])


class WorkerCPUProbe:
    def __init__(self,rank,output,interval=.01,*,deterministic=False,checkpoint_interval=15.):
        self.rank=rank;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.interval=interval;self.enabled=False;self.phase=None
        self.counts=Counter();self.stacks=Counter();self.rows=[];self.segments=None
        self.stop=threading.Event();self.ready=threading.Event();self.gaps=[];self.thread=None
        self.deterministic=deterministic
        self.checkpoint_interval=checkpoint_interval;self.updated=0.;self.progress=None
        self.lock=threading.RLock();self.write_lock=threading.Lock()

    def start(self):
        self.enabled=True
        self.thread=threading.Thread(target=self.sample,name='h3-cpu-stack-sampler',daemon=True);self.thread.start()
        if not self.ready.wait(1):raise RuntimeError('CPU sampler did not start')

    def sample(self):
        previous=time.monotonic()
        while not self.stop.is_set():
            now=time.monotonic()
            with self.lock:self.gaps.append(now-previous)
            previous=now
            thread_names={t.ident:t.name for t in threading.enumerate()}
            for tid,frame in sys._current_frames().items():
                if tid==threading.get_ident():continue
                stack=[]
                for _ in range(24):
                    if frame is None:break
                    stack.append((Path(frame.f_code.co_filename).name,frame.f_code.co_name,frame.f_lineno));frame=frame.f_back
                with self.lock:
                    self.stacks[(thread_names.get(tid,str(tid)),self.phase or 'outside_norm_residual',tuple(stack))]+=1
            self.ready.set()
            if now-self.updated>=self.checkpoint_interval:self.checkpoint()
            self.stop.wait(self.interval)

    def invoke(self,label,fn,args,kwargs):
        if not self.enabled:return fn(*args,**kwargs)
        with self.lock:
            self.counts[label]+=1;number=self.counts[label];before=self.phase;self.phase=label
            if self.segments is None:
                segments=args[-1]
                self.segments=[dict(start=a,end=b,row=row if isinstance(row,int) else dict(
                    shape=list(row.shape),dtype=str(row.dtype),device=str(row.device))) for a,b,row in segments]
        selected=self.deterministic and number in (1,2,50,99)
        # Native 3.13 cProfile produced invalid concurrent statistics in the
        # CPU gate, even with its default clock. The stdlib Python profiler
        # installs a hook on this thread only. Its overhead is measured by the
        # resident replay; live DiT never enables deterministic profiling.
        profiler=py_profile.Profile(timer=time.perf_counter) if selected else None
        wall=time.monotonic_ns();cpu=time.thread_time_ns()
        try:
            if profiler:return profiler.runcall(fn,*args,**kwargs)
            return fn(*args,**kwargs)
        finally:
            with self.lock:self.phase=before
            row=dict(label=label,call=number,wall_seconds=(time.monotonic_ns()-wall)/1e9,
                thread_cpu_seconds=(time.thread_time_ns()-cpu)/1e9,cprofile_enabled=False,
                deterministic_profile_enabled=selected)
            if profiler:
                name=f'rank{self.rank}_{label.replace("/","_")}_{number}.pstats'
                profiler.dump_stats(str(self.output/name))
                stats=pstats.Stats(profiler)
                row['profile_validation']=validate_profile(stats)
                row['profile_clock']='monotonic_wall'
                row['profile_engine']='stdlib_profile_thread_local'
                row['functions']=[dict(file=key[0],line=key[1],function=key[2],primitive_calls=v[0],
                    calls=v[1],self_wall_seconds=v[2],cumulative_wall_seconds=v[3])
                    for key,v in sorted(stats.stats.items(),key=lambda item:-item[1][2])[:50]]
            with self.lock:self.rows.append(row)
            if profiler:
                self.checkpoint()
                if not row['profile_validation']['passed']:raise RuntimeError('Invalid resident profile timing; attribution rejected')

    def mark_progress(self,stage,block):
        with self.lock:self.progress=dict(stage=stage,block=str(block),monotonic_ns=time.monotonic_ns())

    def checkpoint(self,*,final=False):
        with self.write_lock:
            with self.lock:
                value=dict(rank=self.rank,pid=os.getpid(),final=final,interval_seconds=self.interval,
                    monotonic_ns=time.monotonic_ns(),segments=self.segments,progress=self.progress,
                    calls=list(self.rows),stack_samples=[dict(thread=key[0],phase=key[1],stack=list(key[2]),samples=count)
                        for key,count in self.stacks.most_common(500)],
                    sampling_gaps=dict(count=len(self.gaps),maximum_seconds=max(self.gaps,default=0),
                        over_50ms=sum(g>.05 for g in self.gaps)),
                    deterministic_profiles=self.deterministic,
                    limitations='Python samples require the GIL. Phase is owner-thread context. Per-call thread CPU is distinct from deterministic wall time; live NFE does not enable a deterministic profiler.')
            target=self.output/f'rank{self.rank}_cpu.json';temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(target)
            self.updated=time.monotonic()
            return value

    def finish(self):
        self.enabled=False;self.stop.set()
        if self.thread:self.thread.join(3)
        return self.checkpoint(final=True)
