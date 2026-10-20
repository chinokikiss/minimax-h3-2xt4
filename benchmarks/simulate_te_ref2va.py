"""Long TE pipeline semantic checks, no performance measurements."""
import simulate_ref2va as base
import hashlib
import torch
import retest
from full_models import load_full,forward,PipelineRunner
from ref2va_workloads import request


def main():
    base.isolate_cublas_workspaces()
    rows=[]
    with torch.inference_mode():
        for length in [512,2048,4096]:
            data=[request('te',length,1700+i,'cuda:0') for i in range(2)]
            native=load_full('te',base.PATHS['te'],'streamed',devices=('cuda:0','cuda:0'),layers=2)
            ref=torch.stack([forward('te',native,d) for d in data]).cpu()
            del native;retest.cleanup()
            runner=PipelineRunner('te',split=1,devices=('cuda:0','cuda:0'))
            runner.model=load_full('te',base.PATHS['te'],'placed',split=1,runner=runner,devices=('cuda:0','cuda:0'),layers=2)
            for repeat in range(3):
                out=runner(data)
                score=retest.metrics(out.cpu(),ref)
                score.update(kind='te',scope='PP native2 long causal mask/RoPE',length=length,repeat=repeat,passed=retest.passes(score,base.study.blocks.GATE))
                print('SEMANTIC te PP',length,repeat,score['is_bit_exact'],flush=True)
                rows.append(score)
                if not score['passed']:raise RuntimeError('Native semantic mismatch')
            del runner,data,out,ref;retest.cleanup()
    paths=list((base.ROOT/'patches').glob('*.py'))+[base.ROOT/'benchmarks'/n for n in ['ref2va_workloads.py','ref2va_study.py','simulate_ref2va.py','simulate_te_ref2va.py','simulate_full_study.py','full_study.py','full_models.py','te_dit_study.py','logical_dual_gpu.py','fixture_inputs.py','retest.py']]
    retest.write_json(base.ROOT/'semantic_results/ref2va_part_te_pp.json',{'passed':True,'scope':'Local 5060 Ti, two trained TE layers and two logical pipeline streams; correctness only',
        'cases':rows,'source_sha256':{p.relative_to(base.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})


if __name__=='__main__':main()
