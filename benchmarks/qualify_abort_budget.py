"""Exercise real worker termination with a reserved and owner-held budget."""
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from patches.host_weight_cache import HostBudget
from patches.comfy_tp_runtime import PhysicalTPService


def owner(budget,ready,go,poisoned,hold_lock):
    if not budget.acquire(16*2**20,preparation=True):raise RuntimeError('Small test reservation rejected')
    ready.set();go.wait()
    if hold_lock:
        budget.lock.acquire();poisoned.set()
    while True:time.sleep(.1)


def main():
    ctx=mp.get_context('spawn');budget=HostBudget(128*2**20,reserve_bytes=0,context=ctx,pressure_aware=True)
    ready=[ctx.Event(),ctx.Event()];go=ctx.Event();poisoned=ctx.Event()
    processes=[ctx.Process(target=owner,args=(budget,ready[i],go,poisoned,i==0)) for i in range(2)]
    for process in processes:process.start()
    service=PhysicalTPService.__new__(PhysicalTPService)
    service.closed=False;service.processes=processes;service.connections=[];service.store=None;service.budget=budget
    try:
        if not all(event.wait(20) for event in ready):raise RuntimeError('Test workers did not become ready')
        go.set()
        if not poisoned.wait(10):raise RuntimeError('Budget lock was not held at termination')
        service.abort()
        snapshot=budget.snapshot()
        result=dict(passed=all(snapshot[k]==0 for k in ('reserved_bytes','preparing','pending_preparation_bytes'))
                    and all(not p.is_alive() for p in processes),
                    killed_lock_owner=True,budget=snapshot,workers_reaped=all(not p.is_alive() for p in processes))
        (ROOT/'semantic_results/abort_budget_ownership.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        if not result['passed']:raise AssertionError('Closed service retained reservations')
    finally:
        if any(p.is_alive() for p in processes):service.abort()


if __name__=='__main__':main()
