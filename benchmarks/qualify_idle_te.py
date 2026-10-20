"""Trained rank-local INT8 TE reuse on logical ranks; semantics, never T4 timing."""
import hashlib
import json
from pathlib import Path
import time
import torch
import simulate_weight_residency as helpers
import te_dit_study as study
import retest
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from patches.host_weight_cache import HostBudget,HostWeightCache
from patches.weight_residency import ResidencyWindow,module_tensors
from patches.windowed_tp import CheckpointReader,tp_specs


def main():
    import comfy_kitchen as ck
    import patches.tp_qwen3vl as te
    from patches.optimized_tp import enable
    root=Path(__file__).resolve().parents[1]
    files=['patches/weight_residency.py','patches/host_weight_cache.py','patches/windowed_tp.py',
        'patches/rank_local_int8.py','patches/tp_qwen3vl.py','patches/optimized_tp.py',
        'patches/parallel_int8.py','benchmarks/qualify_idle_te.py']
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda _:49152
    isolate_cublas_workspaces();enable('scalemax')
    comm=LogicalDualGPU(timeout=180);te.dist=comm
    x,args=study.inputs('te',helpers.PATHS['te'],512,1,torch.device('cuda:0'))
    reference=helpers.native_stack('te',helpers.PATHS['te'],(x,args),layers=2)
    reader=CheckpointReader(helpers.PATHS['te']);reader.rank_checkpoint_dir='D:/minimaxH3/rank_local_tp2'
    budget=HostBudget(4*2**30,reserve_bytes=512*2**20)
    caches=[HostWeightCache(budget,2**30,r,reuse_aware=True) for r in range(2)]
    windows=[];rows=[]
    try:
        for rank in range(2):
            specs=tp_specs(reader,'te',rank,layers=2)
            windows.append(ResidencyWindow(specs,(),'cuda:0',max_weight_bytes=2**30,
                pinned_bytes=2**30,reserve_bytes=256*2**20,host_cache=caches[rank]))
        for iteration in range(2):
            for window,cache in zip(windows,caches):
                cache.begin_request(list(window.specs.values()))
                window.begin_idle(window.specs,max_bytes=2**30,reserve_bytes=256*2**20)
            deadline=time.monotonic()+90
            while any(len(w.entries)<2 for w in windows):
                if time.monotonic()>deadline:raise TimeoutError('Trained TE idle prefill did not complete')
                for window in windows:window.idle_tick()
                time.sleep(.005)
            pointers=[{k:tuple(t.data_ptr() for t in module_tensors(e.module)) for k,e in w.entries.items()}
                for w in windows]
            for window in windows:window.resume();window.extend_uses(window.specs)
            def run(rank):
                out=x.clone();window=windows[rank]
                for key in window.uses:
                    with window.use(key) as block:
                        assert tuple(t.data_ptr() for t in module_tensors(block))==pointers[rank][key]
                        out=block(out,args[0],args[1])
                return out.cpu()
            outputs=comm.run([lambda r=r:run(r) for r in range(2)])
            for rank,output in enumerate(outputs):
                score=retest.metrics(output,reference);score['passed']=retest.passes(score,helpers.GATE)
                if not score['passed']:raise RuntimeError('Trained TE Native mismatch')
                rows.append(dict(iteration=iteration,rank=rank,correctness=score,
                    same_gpu_storage=True,idle_gpu_hits=windows[rank].stats['idle_gpu_hits'],
                    avoided_h2d_bytes=windows[rank].stats['idle_avoided_h2d_bytes']))
            print('VERIFIED trained idle TE reuse',iteration,flush=True)
    finally:
        for window in windows:window.close()
        for cache in caches:cache.close()
    state=budget.snapshot()
    assert all(state[k]==0 for k in ('reserved_bytes','preparing','pending_preparation_bytes'))
    assert hashes=={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
    result=dict(passed=True,physical_t4=False,scope='Two trained TE blocks, S512, two idle/forward cycles, logical ranks on one 5060 Ti; no performance evidence',
        cases=rows,host_budget_after_close=state,source_sha256=hashes)
    target=root/'semantic_results/idle_trained_te_reuse.json';target.write_text(json.dumps(result,indent=2))
    print('VERIFIED trained idle TE Native equivalence and zero leases',flush=True)


if __name__=='__main__':main()
