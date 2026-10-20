"""Validate planned TP workloads with real CUDA operators, two logical ranks."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT,ROOT/'ComfyUI',ROOT/'.retest-deps',ROOT/'benchmarks']:sys.path.insert(0,str(p))
import hashlib
import json
import torch
from te_dit_study import CASES, GATE, inputs, invoke
from retest import load_native_block, metrics, passes, write_json
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
import patches.tp_qwen3vl as te
import patches.tp_dit as dit


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['legacy','fixed','scalemax'],default='legacy');parser.add_argument('--t4-rotation-policy',action='store_true');options=parser.parse_args()
    if options.mode!='legacy':
        from patches.optimized_tp import enable
        enable(options.mode)
    if options.t4_rotation_policy:
        import comfy_kitchen as ck
        # T4 traces show the shared-memory-limited matmul rotation route.
        # Both Native and TP use that route on the local GPU for this
        # additional correctness comparison, without a performance claim.
        ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda x:49152
    torch.backends.cuda.matmul.allow_tf32=False
    isolate_cublas_workspaces();comm=LogicalDualGPU(timeout=90);te.dist=comm;dit.dist=comm
    rows=[]
    with torch.inference_mode():
        for kind in ['te','dit']:
            path=ROOT/'repaired_fixtures'/(kind+'_block0.safetensors')
            native=load_native_block(kind,path)
            cls=te.TPTransformerBlockINT8 if kind=='te' else dit.TPDiTBlockINT8
            ranks=[cls(rank=r,world_size=2,device='cuda:0').cuda().eval().requires_grad_(False) for r in range(2)]
            for rank in ranks:rank.load_from_full_block(native)
            for length,batch in CASES[kind]:
                x,args=inputs(kind,path,length,batch,torch.device('cuda:0'))
                ref=invoke(kind,native,x,args,True)
                outputs=comm.run([lambda r=r:invoke(kind,ranks[r],x,args,False) for r in range(2)])
                for r,out in enumerate(outputs):
                    score=metrics(out,ref);score.update(kind=kind,length=length,batch=batch,rank=r,passed=passes(score,GATE));rows.append(score)
                    print('SIMULATED',kind,length,batch,r,score['passed'],score['is_bit_exact'],flush=True)
                    if not score['passed']:raise RuntimeError('Semantic mismatch')
                del x,args,ref,outputs
            del native,ranks;torch.cuda.empty_cache()
    files=list((ROOT/'patches').glob('*.py'))+[Path(__file__),ROOT/'benchmarks/te_dit_study.py']
    write_json(ROOT/'semantic_results'/('te_dit_study_'+options.mode+('_t4route' if options.t4_rotation_policy else '')+'.json'),{'passed':True,'scope':'two logical ranks on one 5060 Ti, correctness only','gate':GATE,'cases':rows,'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})


if __name__=='__main__':main()
