"""Complete trained-model validation of immutable auxiliary weight staging."""
import simulate_ref2va as base
import hashlib
import torch
import retest


def main():
    base.enable('scalemax')
    import comfy_kitchen as ck
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda x:49152
    torch.backends.cuda.matmul.allow_tf32=False
    base.isolate_cublas_workspaces()
    comm=base.LogicalDualGPU(timeout=180);base.te.dist=comm;base.dit.dist=comm
    rows=[]
    def record(kind,scope,length,index,out,ref):
        score=retest.metrics(out,ref)
        score.update(kind=kind,scope=scope,length=length,index=index,passed=retest.passes(score,base.study.blocks.GATE))
        rows.append(score);print('SEMANTIC',scope,index,score['is_bit_exact'],flush=True)
        if not score['passed']:raise RuntimeError('Native semantic mismatch')
    with torch.inference_mode():base.full_check(comm,record)
    paths=list((base.ROOT/'patches').glob('*.py'))+[base.ROOT/'benchmarks'/n for n in ['ref2va_workloads.py','ref2va_study.py','simulate_ref2va.py','simulate_ref2va_aux.py','simulate_full_study.py','full_study.py','full_models.py','te_dit_study.py','logical_dual_gpu.py','fixture_inputs.py','retest.py']]
    retest.write_json(base.ROOT/'semantic_results/ref2va_part_aux.json',{'passed':True,'scope':'local full50 Native vs TP2 shared preprocessing with immutable CPU auxiliary weights; correctness only',
        'cases':rows,'source_sha256':{p.relative_to(base.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})


if __name__=='__main__':main()
