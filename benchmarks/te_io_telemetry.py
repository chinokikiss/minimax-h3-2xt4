"""Opt-in, Linux file-read causality measurements; no inference policy changes.

Counters bracket one actual buffered read. Kernel wait states are sampled,
not assigned exact durations. Global/cgroup deltas include other activity.
"""
import ctypes
import math
import mmap
import os
from pathlib import Path
try:import resource
except ImportError:resource=None
import threading
import time
from collections import Counter


def text(path):
    try:
        with open(path) as stream:return stream.read().strip()
    except OSError:return None


def counters(path):
    result={}
    for line in (text(path) or '').splitlines():
        parts=line.replace(':','').split()
        if len(parts)>=2:
            try:result[parts[0]]=int(parts[1])
            except ValueError:pass
    return result


def pressure(path):
    result={}
    for line in (text(path) or '').splitlines():
        parts=line.split()
        if parts:
            for item in parts[1:]:
                if item.startswith('total='):result[parts[0]]=int(item.split('=')[1])
    return result


def diskstats():
    result={}
    for line in (text('/proc/diskstats') or '').splitlines():
        p=line.split()
        if len(p)>=14:
            result[p[2]]={k:int(p[i]) for k,i in (('reads',3),('read_sectors',5),
                ('read_ms',6),('writes',7),('write_sectors',9),('write_ms',10),
                ('inflight',11),('busy_ms',12),('weighted_busy_ms',13))}
    return result


def cgroup_root():
    # cgroup v2 is namespaced on Kaggle; try the visible mount root first.
    root=Path('/sys/fs/cgroup')
    if (root/'memory.stat').exists():return root
    for line in (text('/proc/self/cgroup') or '').splitlines():
        if line.startswith('0::'):
            candidate=root/line[3:].lstrip('/')
            if (candidate/'memory.stat').exists():return candidate
    return None


CGROUP=cgroup_root() if os.name=='posix' else None
VM_KEYS=('pgmajfault','pgfault','pgscan_direct','pgsteal_direct','pgscan_kswapd',
         'pgsteal_kswapd','allocstall','pswpin','pswpout','workingset_refault_file',
         'workingset_activate_file','workingset_restore_file','pgscan_direct_throttle')


def snapshot():
    tid=threading.get_native_id();root=f'/proc/self/task/{tid}'
    stat=text(root+'/stat');fields=stat[stat.rfind(')')+2:].split() if stat else []
    usage=resource.getrusage(resource.RUSAGE_THREAD) if resource and hasattr(resource,'RUSAGE_THREAD') else None
    vm=counters('/proc/vmstat')
    cg=str(CGROUP) if CGROUP else ''
    return dict(tid=tid,thread_cpu_ns=time.thread_time_ns(),
        usage={k:getattr(usage,k) for k in ('ru_minflt','ru_majflt','ru_nvcsw','ru_nivcsw')} if usage else {},
        io=counters(root+'/io'),
        schedstat=[int(v) for v in (text(root+'/schedstat') or '').split()],
        blkio_ticks=int(fields[39]) if len(fields)>39 else None,
        vm={k:v for k,v in vm.items() if any(k==p or k.startswith(p+'_') for p in VM_KEYS)},
        memory_stat=counters(cg+'/memory.stat') if cg else {},
        memory_events=counters(cg+'/memory.events') if cg else {},
        io_stat=text(cg+'/io.stat') if cg else None,diskstats=diskstats(),
        memory_current=text(cg+'/memory.current') if cg else None,
        meminfo={k:v for k,v in counters('/proc/meminfo').items() if k in
            ('MemAvailable','Cached','SwapFree','AnonPages','Active(file)','Inactive(file)','Dirty','Writeback')},
        psi_memory=pressure(cg+'/memory.pressure') if cg else {},
        psi_io=pressure(cg+'/io.pressure') if cg else {},
        cpu_stat=counters(cg+'/cpu.stat') if cg else {})


def delta(before,after):
    result={}
    for key in ('usage','io','vm','memory_events','psi_memory','psi_io','cpu_stat'):
        result[key]={k:after[key][k]-v for k,v in before[key].items() if k in after[key]}
    # Gauges and monotone counters are kept separate: memory_stat includes both.
    result['memory_stat']={k:after['memory_stat'][k]-v for k,v in before['memory_stat'].items()
        if k in after['memory_stat'] and (k.startswith(('pg','workingset_')))}
    if len(before['schedstat'])==len(after['schedstat'])==3:
        result['schedstat']=[b-a for a,b in zip(before['schedstat'],after['schedstat'])]
    if before['blkio_ticks'] is not None and after['blkio_ticks'] is not None:
        result['blkio_ticks']=after['blkio_ticks']-before['blkio_ticks']
    result['diskstats']={device:{k:after['diskstats'][device][k]-v for k,v in values.items()
        if k!='inflight'} for device,values in before['diskstats'].items() if device in after['diskstats']}
    return result


def residency(fd,offset=0,length=None):
    """mincore observes file-cache residency without touching mapped contents."""
    if os.name!='posix':return dict(unavailable='Linux mincore only')
    size=os.fstat(fd).st_size
    stop=min(size,offset+(length if length is not None else size))
    if stop<=offset:return dict(pages=0,resident_pages=0,resident_fraction=1.0)
    page=mmap.PAGESIZE;start=offset//page*page;extent=stop-start
    # ACCESS_COPY makes the mapping's address available to ctypes. It does
    # not write or touch file bytes; munmap occurs before the measured read.
    try:
        with mmap.mmap(fd,extent,access=mmap.ACCESS_COPY,offset=start) as mapping:
            pages=math.ceil(extent/page);vector=(ctypes.c_ubyte*pages)()
            address=ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            libc=ctypes.CDLL(None,use_errno=True)
            libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_ubyte)]
            status=libc.mincore(address,extent,vector)
            if status:return dict(error_errno=ctypes.get_errno())
            count=sum(v&1 for v in vector)
            return dict(pages=pages,resident_pages=count,resident_fraction=count/pages)
    except (OSError,ValueError) as error:return dict(error_type=type(error).__name__)


class ReadObserver:
    def __init__(self,period=.05):
        self.period=period;self.active={};self.lock=threading.Lock();self.closed=False
        self._local=threading.local()
        self.thread=threading.Thread(target=self.sample,name='te-io-observer',daemon=True)
        self.thread.start()

    def sample(self):
        while not self.closed:
            with self.lock:active=list(self.active.items())
            for tid,row in active:
                root=f'/proc/self/task/{tid}';stat=text(root+'/stat')
                state=stat[stat.rfind(')')+2:].split()[0] if stat else '?'
                wchan=text(root+'/wchan') or '?'
                # Unprivileged containers may deny stacks. Record that explicitly.
                stack=text(root+'/stack') if state=='D' else None
                with self.lock:
                    if self.active.get(tid) is row:
                        row['states'][state]+=1;row['wchans'][wchan]+=1
                        if stack and len(row['stacks'])<4 and stack not in row['stacks']:row['stacks'].append(stack)
                        if state=='D' and stack is None:row['stack_unavailable_samples']+=1
            time.sleep(self.period)

    def measure(self,file,size,operation):
        tid=threading.get_native_id();offset=file.tell();fd=file.fileno()
        observe_start=time.monotonic_ns();resident_before=residency(fd,offset,size)
        before=snapshot();samples=dict(states=Counter(),wchans=Counter(),stacks=[],stack_unavailable_samples=0)
        with self.lock:self.active[tid]=samples
        start=time.monotonic_ns();cpu=time.thread_time_ns();error=None
        try:return_value=operation()
        except BaseException as failure:error=type(failure).__name__;raise
        finally:
            end=time.monotonic_ns();cpu_ns=time.thread_time_ns()-cpu
            with self.lock:self.active.pop(tid,None)
            after=snapshot();resident_after=residency(fd,offset,size)
            row=dict(start_ns=start,end_ns=end,tid=tid,offset=offset,requested_bytes=size,
                cpu_ns=cpu_ns,wall_ns=end-start,error=error,
                resident_before=resident_before,resident_after=resident_after,
                counters=delta(before,after),before=before,after=after,
                samples={k:dict(v) if isinstance(v,Counter) else v for k,v in samples.items()})
            row['observer_outside_read_ns']=time.monotonic_ns()-observe_start-(end-start)
            self.last_local.row=row
        row['returned_bytes']=len(return_value) if isinstance(return_value,(bytes,bytearray)) else return_value
        return return_value,row

    @property
    def last_local(self):
        if not hasattr(self,'_local'):self._local=threading.local()
        return self._local

    def close(self):
        self.closed=True;self.thread.join(timeout=2)
        if self.thread.is_alive():raise RuntimeError('Read observer did not stop')


def capabilities():
    return dict(platform=os.uname().release if os.name=='posix' else os.name,
        cgroup=str(CGROUP) if CGROUP else None,
        sched_schedstats=text('/proc/sys/kernel/sched_schedstats'),
        delayacct_enabled=text('/proc/sys/kernel/task_delayacct'),
        memory_max=text(CGROUP/'memory.max') if CGROUP else None,
        memory_high=text(CGROUP/'memory.high') if CGROUP else None,
        cpu_max=text(CGROUP/'cpu.max') if CGROUP else None,
        io_max=text(CGROUP/'io.max') if CGROUP else None,
        io_weight=text(CGROUP/'io.weight') if CGROUP else None,
        limits=['mincore reports residency, not page readiness/lock ownership',
                'Buffered file-cache misses need not increment caller major faults',
                'I/O read_bytes is kernel-accounted storage traffic, not device service-time attribution',
                'cgroup/global/diskstats counters include other tasks; interval correlation is not sole-cause proof',
                'state/wchan/stack samples do not reconstruct exact blocked durations',
                'schedstat runnable delay and blkio delay require kernel accounting enabled'])


class ObservedFile:
    def __init__(self,file,path,recorder):self.file=file;self.path=path;self.recorder=recorder
    def __getattr__(self,name):return getattr(self.file,name)
    def __enter__(self):self.file.__enter__();return self
    def __exit__(self,*a):return self.file.__exit__(*a)
    def read(self,*a,**k):
        size=a[0] if a else k.get('size',-1)
        if size<0:size=os.fstat(self.file.fileno()).st_size-self.file.tell()
        return self._read(size,lambda:self.file.read(*a,**k))
    def readinto(self,buffer):return self._read(len(buffer),lambda:self.file.readinto(buffer))
    def _read(self,size,operation):
        result,row=self.recorder.io_observer.measure(self.file,size,operation)
        self.recorder.save_read(row,self.path)
        return result
