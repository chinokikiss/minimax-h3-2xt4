"""Generate a supervised notebook body; no imports of Torch or Kaggle here.

The deadline starts when the notebook cell executes, not during provider boot
or input mounting. The worker's active stage has its own process group, which
must also be terminated when the outer deadline expires.
"""
import ast
import base64
from pathlib import Path


def wrap_notebook(body, budget=600, output='/kaggle/working/t4_validation',omitted_stages=()):
    ast.parse(body)
    replacements={
        "        try:entry['exit_code']=process.wait(timeout=remaining)":
        "        (OUT/'active_process_group.json').write_text(json.dumps(dict(pid=process.pid)))\n"
        "        try:entry['exit_code']=process.wait(timeout=remaining)",
        "    remaining=min(cap,deadline-time.monotonic());entry=dict(name=name,argv=command)":
        "    mark('stage_start',name=name)\n"
        "    remaining=min(cap,deadline-time.monotonic());entry=dict(name=name,argv=command)",
    }
    for old,new in replacements.items():
        if body.count(old)!=1:raise ValueError('Unrecognized stage-runner source')
        body=body.replace(old,new)
    markers=[
        ("    receipts=list(", 'find_cpu_receipt'),
        ("    asset_files=list(", 'find_asset_receipt'),
        ("    frozen.prepare_source(", 'verify_extract_frozen_source'),
        ("    base=prep.inventory();", 'inventory_native_packages'),
        ("    with archive.open('rb')", 'verify_wheel_archive'),
        ("    with tarfile.open(archive,", 'extract_pinned_wheels'),
        ("    prep.install_overlay(", 'install_pinned_overlay'),
        ("    assert prep.inventory()==base", 'verify_native_packages_unchanged'),
        ("    nvcc=shutil.which(", 'locate_compiler'),
    ]
    lines=body.splitlines(keepends=True);seen=set();instrumented=[]
    for line in lines:
        for prefix,name in markers:
            if line.startswith(prefix):
                instrumented.append(f"    mark({name!r})\n");seen.add(name)
        instrumented.append(line)
    expected={name for _,name in markers}
    if not set(omitted_stages)<=expected:raise ValueError('Unknown omitted setup stage')
    if seen!=expected-set(omitted_stages):raise ValueError('Missing setup progress boundary')
    prelude="""import json,pathlib,time
OUT=pathlib.Path(OUTPUT);OUT.mkdir(parents=True,exist_ok=True)
def mark(stage,**extra):
    event=dict(stage=stage,elapsed_seconds=time.monotonic()-WORKER_STARTED,**extra)
    with (OUT/'progress.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\\n');stream.flush()
    print('T4_PROGRESS '+json.dumps(event),flush=True)
WORKER_STARTED=time.monotonic()
mark('worker_started')
""".replace('OUTPUT',repr(output))
    return supervise_worker(prelude+''.join(instrumented),budget,output)


def supervise_worker(body,budget=600,output='/kaggle/working/t4_validation'):
    """Supervise an arbitrary worker, including preparation or an E2E batch."""
    ast.parse(body)
    payload=base64.b64encode(body.encode()).decode()
    outer='''print('T4_SUPERVISOR notebook_cell_started',flush=True)
import base64,json,os,pathlib,signal,subprocess,sys,threading,time
OUT=pathlib.Path(OUTPUT);OUT.mkdir(parents=True,exist_ok=True)
worker=OUT/'bounded_worker.py';worker.write_bytes(base64.b64decode(PAYLOAD))
started=time.monotonic()
receipt=dict(status='starting',deadline_seconds=BUDGET,scope='Notebook cell execution only; provider startup/mounting excluded')
def save():(OUT/'supervision.json').write_text(json.dumps(receipt,indent=2))
save()
process=subprocess.Popen([sys.executable,'-u',str(worker)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
    start_new_session=(os.name=='posix'),text=True,bufsize=1)
receipt.update(status='running',worker_pid=process.pid);save()
def relay():
    with (OUT/'notebook_worker.log').open('w') as stream:
        for line in process.stdout:
            stream.write(line);stream.flush();print(line,end='',flush=True)
reader=threading.Thread(target=relay,daemon=True);reader.start()
def stop_worker():
    # Freeze the worker before reading its stage PID so it cannot launch a
    # new independent stage group while the supervisor is cleaning up.
    if os.name=='posix':
        try:os.kill(process.pid,signal.SIGSTOP)
        except ProcessLookupError:pass
    groups={process.pid}
    # Discover independent descendant groups too, including the small interval
    # between Popen returning and the worker persisting its active-stage PID.
    pending=[process.pid]
    if os.name=='posix' and pathlib.Path('/proc').exists():
        while pending:
            pid=pending.pop()
            try:
                children=[int(v) for v in pathlib.Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
                pending.extend(children)
                for child in children:
                    try:groups.add(os.getpgid(child))
                    except ProcessLookupError:pass
            except (OSError,ValueError):pass
    active=OUT/'active_process_group.json'
    if active.exists() and not pathlib.Path('/proc').exists():
        try:groups.add(json.loads(active.read_text())['pid'])
        except (OSError,ValueError,KeyError):pass
    for group in groups:
        try:
            if os.name=='posix':os.killpg(group,signal.SIGKILL)
            elif group==process.pid:process.kill()
        except ProcessLookupError:pass
    process.wait(timeout=5)
try:
    receipt['exit_code']=process.wait(timeout=max(.01,BUDGET-(time.monotonic()-started)))
    receipt['status']='worker_exited'
except subprocess.TimeoutExpired:
    stop_worker();receipt.update(status='deadline_exceeded',exit_code=process.returncode)
except BaseException:
    stop_worker();receipt.update(status='interrupted',exit_code=process.returncode);raise
finally:
    reader.join(timeout=2)
    receipt['seconds']=time.monotonic()-started;save()
    print('T4_SUPERVISOR_FINAL '+json.dumps(receipt),flush=True)
'''
    outer=outer.replace('OUTPUT',repr(output)).replace('BUDGET',repr(budget)).replace('PAYLOAD',repr(payload))
    ast.parse(outer)
    return outer
