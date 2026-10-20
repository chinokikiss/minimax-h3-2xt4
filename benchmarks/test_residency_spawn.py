"""Physical runner config and process-shared pressure budget survive spawn."""
import os
os.environ['MINIMAX_H3_RESIDENCY_POLICY']='p0_p1'
from residency_pipeline_ablation import ResidencyConfig
from patches.host_weight_cache import HostBudget
import multiprocessing as mp


def worker(config,budget,queue):
    assert config.prefetch_lifecycle and config.memory_pressure_aware
    assert budget.acquire(128,preparation=True)
    queue.put(budget.snapshot()['reserved_bytes'])
    budget.finish_preparation(128);budget.release(128)


if __name__=='__main__':
    context=mp.get_context('spawn');queue=context.Queue()
    budget=HostBudget(1024*1024,reserve_bytes=0,context=context,pressure_aware=True)
    process=context.Process(target=worker,args=(ResidencyConfig('unused','unused'),budget,queue))
    process.start();assert queue.get(timeout=60)==128;process.join(60)
    assert process.exitcode==0 and budget.snapshot()['reserved_bytes']==0
    queue.close();print('VERIFIED residency spawn config and shared leases')
