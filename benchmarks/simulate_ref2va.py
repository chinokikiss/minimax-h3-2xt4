"""Execute fixed Ref2VA fixtures on two logical ranks of the local 5060 Ti."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT,ROOT/'ComfyUI',ROOT/'.retest-deps',ROOT/'benchmarks']:sys.path.insert(0,str(p))
import hashlib
import json
import torch
from torch import nn
import retest
import ref2va_study as study
from ref2va_workloads import request,capture_inputs,DIT_LENGTHS
from full_models import load_full,skeleton,forward,PipelineRunner,stage_dit_aux
from simulate_full_study import StreamedTP
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
import patches.tp_qwen3vl as te
import patches.tp_dit as dit
from patches.optimized_tp import enable

PATHS={'te':Path('D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'),
       'dit':Path('D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')}


def main():
    import os
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--pp-length',type=int,choices=[68,8192,16384,32768])
    parser.add_argument('--blocks-only',action='store_true')
    parser.add_argument('--microrequests',type=int,default=1)
    options=parser.parse_args()
    folder=ROOT/'semantic_results/ref2va_inputs';folder.mkdir(exist_ok=True)
    target=ROOT/('semantic_results/ref2va_part_blocks.json' if options.blocks_only else ('semantic_results/ref2va_part_pp'+str(options.pp_length)+'.json' if options.pp_length else 'semantic_results/ref2va_fixed_matrix.json'));rows=[]
    enable('scalemax')
    import comfy_kitchen as ck
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda x:49152
    torch.backends.cuda.matmul.allow_tf32=False
    isolate_cublas_workspaces();comm=LogicalDualGPU(timeout=180);te.dist=comm;dit.dist=comm
    def record(kind,scope,length,index,out,ref):
        score=retest.metrics(out,ref);score.update(kind=kind,scope=scope,length=length,index=index,passed=retest.passes(score,study.blocks.GATE));rows.append(score)
        retest.write_json(target,{'passed':False,'cases':rows})
        print('SEMANTIC',kind,scope,length,index,score['passed'],score['is_bit_exact'],flush=True)
        if not score['passed']:raise RuntimeError('Native semantic mismatch')
    for length in DIT_LENGTHS:
        for i in range(2 if length==8192 else 1):
            path=folder/f's{length}_{i}.pt'
            if not path.exists():
                print('CAPTURE native preprocessing',length,i,flush=True)
                torch.save(capture_inputs(PATHS['dit'],length,900+i),path)
    os.environ['REF2VA_INPUTS']=str(folder);study.configure()
    with torch.inference_mode():
        for kind in ([] if options.pp_length else ['te','dit']):
            native=retest.load_native_block(kind,PATHS[kind])
            cls=te.TPTransformerBlockINT8 if kind=='te' else dit.TPDiTBlockINT8
            ranks=[cls(rank=r,world_size=2,device='cuda:0').cuda().eval().requires_grad_(False) for r in range(2)]
            for rank in ranks:rank.load_from_full_block(native)
            for length,batch in study.blocks.CASES[kind]:
                x,args=study.inputs(kind,PATHS[kind],length,batch,torch.device('cuda:0'))
                ref=study.blocks.invoke(kind,native,x,args,True).cpu()
                outputs=comm.run([lambda r=r:study.blocks.invoke(kind,ranks[r],x,args,False) for r in range(2)])
                for r,out in enumerate(outputs):record(kind,'trained block TP',length,r,out.cpu(),ref)
                del outputs,x,args,ref
            del native,ranks;retest.cleanup()
        # Complete 50-layer native entry point checks the new shared/staged
        # preprocessing and the reference modality/time tags, without timing.
        if not options.pp_length and not options.blocks_only:
            full_check(comm,record)
        pp_cases=[] if options.blocks_only else ([(options.pp_length,options.microrequests)] if options.pp_length else [(68,4),(8192,2),(16384,1),(32768,1)])
        for length,count in pp_cases:
            data=[request('dit',length,1700+i,'cuda:0') for i in range(count)]
            native=load_full('dit',PATHS['dit'],'streamed',devices=('cuda:0','cuda:0'),layers=2)
            refs=[forward('dit',native,d).cpu() for d in data]
            del native;retest.cleanup()
            runner=PipelineRunner('dit',split=1,devices=('cuda:0','cuda:0'))
            runner.model=load_full('dit',PATHS['dit'],'placed',split=1,runner=runner,devices=('cuda:0','cuda:0'),layers=2,offload_dit_aux=True)
            for repeat in range(3):
                outputs=runner(data)
                for i,out in enumerate(outputs):record('dit',f'PP native2 staged repeat{repeat}',length,i,out.cpu(),refs[i])
            del runner,outputs,refs,data;retest.cleanup()
    paths=list((ROOT/'patches').glob('*.py'))+[ROOT/'benchmarks'/n for n in ['ref2va_workloads.py','ref2va_study.py','simulate_ref2va.py','simulate_full_study.py','full_study.py','full_models.py','te_dit_study.py','logical_dual_gpu.py','fixture_inputs.py','retest.py']]
    retest.write_json(target,{'passed':True,'scope':'local 5060 Ti logical two ranks, correctness only; T4 fallback rotation policy',
                           'cases':rows,'gate':study.blocks.GATE,'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})


def full_check(comm,record):
        data=request('dit',68,1700,'cuda:0')
        native=load_full('dit',PATHS['dit'],'streamed',devices=('cuda:0','cuda:0'))
        ref=forward('dit',native,data).cpu();del native;retest.cleanup()
        ranks=[]
        for r in range(2):
            model=skeleton('dit',PATHS['dit'],'cuda:0')
            stage_dit_aux(model,'cuda:0',shared=True,rank=r,comm=comm)
            model.blocks=nn.ModuleList([StreamedTP('dit',PATHS['dit'],i,r) for i in range(50)])
            ranks.append(model)
        outputs=comm.run([lambda r=r:forward('dit',ranks[r],data) for r in range(2)])
        for r,out in enumerate(outputs):record('dit','full50 staged/shared TP',68,r,out.cpu(),ref)
        del model,ranks,outputs,ref,data;retest.cleanup()
        # Different trained layers, native preprocessing/heads and stage handoff.
        # Long cases use one microrequest because two physical devices' maximum
        # allocations may not fit concurrently on this single 16 GB GPU.


if __name__=='__main__':main()
