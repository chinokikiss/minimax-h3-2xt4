"""External scheduler/GIL sampling around the unchanged real request runner."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('Real request command required')
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=dict(python_stack_tool=shutil.which('py-spy'),perf=shutil.which('perf'),
        gil_samples='owns_gil and Python frames sampled once per second; not exact GIL waiting time',
        profiling_perturbation='External sampling and observer hooks may change scheduling; compare with matched baseline')
    perf=metadata['perf'];perf_available=False
    if perf:
        probe=subprocess.run([perf,'record','--clockid','mono','-e','sched:sched_switch',
            '-o',str(args.output/'perf_probe.data'),'--',sys.executable,'-c','import time;time.sleep(.05)'],
            capture_output=True,text=True,timeout=15)
        (args.output/'perf_probe.log').write_text(probe.stdout+probe.stderr)
        perf_available=probe.returncode==0
    metadata['scheduler_trace_available']=perf_available
    if perf_available:
        command=[perf,'record','--clockid','mono','-e','sched:sched_switch',
            '-e','sched:sched_wakeup','-e','sched:sched_wakeup_new','--inherit',
            '-o',str(args.output/'scheduler.data'),'--',*command]
        metadata['scheduler_scope']='Target process/descendants; switch-in and external wakeup coverage may be incomplete; verify event coverage before exact attribution'
    else:metadata['scheduler_limitation']='perf unavailable or denied; /proc counters/state snapshots remain available'
    environment=dict(os.environ,CPU_OBSERVER_PARENT=str(os.getpid()))
    start=time.monotonic_ns();process=subprocess.Popen(command,env=environment)
    heartbeat=time.monotonic();previous_cpu=0;sample_count=0
    sampler=subprocess.Popen([sys.executable,str(Path(__file__).with_name('cpu_os_sampler.py')),
        '--root',str(args.output),'--until-pid',str(process.pid),'--period','.1'])
    # py-spy 0.4.1 rejects native stack unwinding in nonblocking mode.
    # Nonblocking Python stacks preserve GIL ownership sampling without
    # periodically stopping inference. Native GIL wait frames are unavailable.
    failed=0;spy=metadata['python_stack_tool'];spy_flags=['--json','--subprocesses','--nonblocking']
    metadata['native_stack_limitation']='Native unwinding is incompatible with nonblocking sampling; no native GIL wait duration inferred'
    with (args.output/'gil_stack_samples.jsonl').open('w') as stream:
        while process.poll() is None:
            parent=args.output/'cpu_process_rparent.json'
            if spy and parent.exists():
                target=json.loads(parent.read_text())['pid'];begin=time.monotonic_ns()
                try:
                    response=subprocess.run([spy,'dump','--pid',str(target),*spy_flags],
                        capture_output=True,text=True,timeout=2)
                    end=time.monotonic_ns()
                    if response.returncode:
                        failed+=1
                        if failed==1:(args.output/'py_spy_probe.log').write_text(response.stderr)
                        if failed>=3:
                            spy=None;metadata['gil_sampling_limitation']='py-spy repeatedly denied/unsupported; inspect probe log; no GIL wait duration claimed'
                    else:
                        rows=json.loads(response.stdout)
                        if isinstance(rows,dict):rows=rows.get('traces',[])
                        # Never request locals, tensor values, tokens or env.
                        stream.write(json.dumps(dict(start_ns=begin,end_ns=end,traces=rows))+'\n');stream.flush()
                        sample_count+=1
                        failed=0
                except (subprocess.TimeoutExpired,ValueError):
                    failed+=1
                    if failed>=3:
                        spy=None;metadata['gil_sampling_limitation']='Sampling repeatedly timed out or returned invalid JSON'
            if time.monotonic()-heartbeat>=60:
                cpu_ticks=0;count=0
                for file in args.output.glob('cpu_process_r*.json'):
                    try:
                        pid=json.loads(file.read_text())['pid'];text=Path(f'/proc/{pid}/stat').read_text()
                        fields=text[text.rfind(')')+2:].split()
                        cpu_ticks+=int(fields[11])+int(fields[12]);count+=1
                    except (OSError,ValueError,KeyError):pass
                now_cpu=cpu_ticks/os.sysconf('SC_CLK_TCK')
                print('AUDIT CPU ACTIVITY elapsed_seconds',round((time.monotonic_ns()-start)/1e9),
                    'observed_processes',count,'process_cpu_seconds_delta',round(max(0,now_cpu-previous_cpu),2),
                    'stack_snapshots',sample_count,flush=True)
                previous_cpu=now_cpu;heartbeat=time.monotonic()
            time.sleep(1)
    sampler.wait(timeout=20)
    metadata.update(start_ns=start,end_ns=time.monotonic_ns(),returncode=process.returncode)
    (args.output/'cpu_tool_capabilities.json').write_text(json.dumps(metadata,indent=2))
    if perf_available:
        with (args.output/'scheduler.txt').open('w') as stream:
            converted=subprocess.run([perf,'script','--ns','-i',str(args.output/'scheduler.data'),
                '-F','comm,pid,tid,cpu,time,event,trace'],stdout=stream,stderr=subprocess.PIPE,text=True,timeout=120)
        metadata['perf_script_returncode']=converted.returncode
        (args.output/'cpu_tool_capabilities.json').write_text(json.dumps(metadata,indent=2))
    raise SystemExit(process.returncode)


if __name__=='__main__':main()
