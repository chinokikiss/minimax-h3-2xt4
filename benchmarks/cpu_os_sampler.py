"""Read-only external Linux thread telemetry; unavailable counters stay unknown."""
import argparse
import json
import os
from pathlib import Path
import time


def read(path):
    try:return Path(path).read_text().strip()
    except OSError:return None


def task_sample(pid,tid):
    root=Path(f'/proc/{pid}/task/{tid}');text=read(root/'stat')
    if text is None:return None
    # comm can contain whitespace/parentheses; fields follow its last ')'.
    fields=text[text.rfind(')')+2:].split();sched=read(root/'schedstat')
    return dict(pid=pid,tid=int(tid),state=fields[0],utime_ticks=int(fields[11]),
        stime_ticks=int(fields[12]),starttime_ticks=int(fields[19]),processor=int(fields[36]),
        schedstat=[int(v) for v in sched.split()] if sched else None,
        name=read(root/'comm'),wchan=read(root/'wchan'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--until-pid',type=int,required=True);parser.add_argument('--period',type=float,default=.1)
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    cgroups={}
    for name in ('cpu.max','cpu.stat','cpuset.cpus.effective','cpu/cpu.cfs_quota_us','cpu/cpu.cfs_period_us'):
        cgroups[name]=read(Path('/sys/fs/cgroup')/name)
    config=dict(period_s=args.period,clock='host_monotonic_ns',clock_ticks=os.sysconf('SC_CLK_TCK'),
        sched_schedstats=read('/proc/sys/kernel/sched_schedstats'),cgroup_initial=cgroups,
        limitations=['schedstat run-delay is usable only when scheduler accounting is enabled',
            'state/wchan are samples, not exact blocked intervals','cgroup counters include all members, not solely inference',
            'observation is external and still consumes CPU; compare perturbation before accepting timings'])
    (args.root/'cpu_os_capabilities.json').write_text(json.dumps(config,indent=2))
    with (args.root/'cpu_thread_samples.jsonl').open('w',buffering=1024*1024) as stream:
        while Path(f'/proc/{args.until_pid}').exists():
            begin=time.monotonic();members=[]
            for file in args.root.rglob('cpu_process_r*.json'):
                try:
                    row=json.loads(file.read_text());pid=row['pid']
                    if (file.parent/f"cpu_done_r{row['rank']}.json").exists():continue
                    for task in Path(f'/proc/{pid}/task').iterdir():
                        sample=task_sample(pid,task.name)
                        if sample:members.append(dict(sample,rank=row['rank'],capture=str(file.parent)))
                except (OSError,ValueError,KeyError):continue
            if members:stream.write(json.dumps(dict(start_ns=time.monotonic_ns(),threads=members,
                cgroup_cpu_stat=read('/sys/fs/cgroup/cpu.stat')))+'\n');stream.flush()
            time.sleep(max(.001,args.period-(time.monotonic()-begin)))


if __name__=='__main__':main()
