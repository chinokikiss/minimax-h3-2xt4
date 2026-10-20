"""Full native entry-point semantics and scheduled PP on the local 5060 Ti."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT,ROOT/'ComfyUI',ROOT/'.retest-deps',ROOT/'benchmarks']:sys.path.insert(0,str(p))
import json
import torch
from torch import nn
import retest
from full_models import load_full,skeleton,request,forward,TPAdapter,PipelineRunner
from te_dit_study import GATE
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
import patches.tp_qwen3vl as te
import patches.tp_dit as dit

PATHS={'te':Path('D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'),
       'dit':Path('D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')}


class StreamedTP(nn.Module):
    def __init__(self,kind,path,index,rank):super().__init__();self.kind=kind;self.path=path;self.index=index;self.rank=rank
    def forward(self,*args,**kwargs):
        native=retest.load_native_block(self.kind,self.path,self.index,'cpu')
        cls=te.TPTransformerBlockINT8 if self.kind=='te' else dit.TPDiTBlockINT8
        block=cls(rank=self.rank,world_size=2,device='cuda:0').cuda().eval().requires_grad_(False)
        block.load_from_full_block(native);del native
        out=TPAdapter(block,self.kind)(*args,**kwargs)
        torch.cuda.synchronize();del block
        return out


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--kinds',nargs='+',default=['te','dit']);parser.add_argument('--schedule-only',action='store_true');parser.add_argument('--length',type=int);parser.add_argument('--microbatches',type=int,default=4);parser.add_argument('--mode',choices=['legacy','scalemax'],default='legacy');parser.add_argument('--t4-rotation-policy',action='store_true');options=parser.parse_args()
    if options.mode!='legacy':
        from patches.optimized_tp import enable
        enable()
    if options.t4_rotation_policy:
        import comfy_kitchen as ck
        ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda x:49152
    target=ROOT/'semantic_results'/('full_study_'+'_'.join(options.kinds)+('_schedule' if options.schedule_only else '')+('_s'+str(options.length) if options.length else '')+'_m'+str(options.microbatches)+'_'+options.mode+('_t4route' if options.t4_rotation_policy else '')+'.json')
    isolate_cublas_workspaces();comm=LogicalDualGPU(timeout=120);te.dist=comm;dit.dist=comm
    rows=[]
    def check(kind,mode,index,out,ref):
        value=retest.metrics(out,ref);value.update(kind=kind,mode=mode,request=index,passed=retest.passes(value,GATE));rows.append(value)
        retest.write_json(target,{'passed':False,'cases':rows})
        print(kind,mode,index,value['passed'],value['is_bit_exact'],flush=True)
        if not value['passed']:raise RuntimeError('Full entry-point semantic mismatch')
    with torch.inference_mode():
        for kind in options.kinds:
            path=PATHS[kind];length=options.length or (37 if kind=='te' else 68)
            data=[request(kind,length,1700+i,'cuda:0') for i in range(options.microbatches)]
            # Test the exact original complete native forward, one trained block
            # resident at a time; this timing is not performance evidence.
            if not options.schedule_only:
                native=load_full(kind,path,'streamed',devices=('cuda:0','cuda:0'))
                refs=[forward(kind,native,data[0]).cpu()]
                del native;retest.cleanup()
                ranks=[skeleton(kind,path,'cuda:0') for _ in range(2)]
                for r,model in enumerate(ranks):setattr(model,'layers' if kind=='te' else 'blocks',nn.ModuleList([StreamedTP(kind,path,i,r) for i in range(50)]))
                outputs=comm.run([lambda r=r:forward(kind,ranks[r],data[0]) for r in range(2)])
                for r,out in enumerate(outputs):check(kind,'full50_TP',r,out.cpu(),refs[0])
                del ranks,outputs;retest.cleanup()
            # Scheduling changes ownership and overlap, not the operator graph.
            # Two distinct trained layers exercise the native full entry/head.
            native=load_full(kind,path,'streamed',devices=('cuda:0','cuda:0'),layers=2)
            refs=[forward(kind,native,d).cpu() for d in data]
            del native;retest.cleanup()
            runner=PipelineRunner(kind,split=1,devices=('cuda:0','cuda:0'))
            runner.model=load_full(kind,path,'placed',split=1,runner=runner,devices=('cuda:0','cuda:0'),layers=2)
            for repeat in range(3):
                outputs=runner(data)
                for i,out in enumerate(outputs):check(kind,f'PP_{options.microbatches}_microrequests_repeat{repeat}',i,out.cpu(),refs[i])
            del runner,outputs,data;retest.cleanup()
    import hashlib
    paths=[ROOT/p for p in ['patches/parallel_int8.py','patches/tp_qwen3vl.py','patches/tp_dit.py','patches/optimized_tp.py','benchmarks/full_models.py','benchmarks/simulate_full_study.py','benchmarks/fixture_inputs.py']]
    retest.write_json(target,{'passed':True,'scope':'full50 native/TP semantics unless schedule-only; two distinct layers and independent native-entry PP microrequests; one physical GPU; no performance evidence','gate':GATE,'cases':rows,'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})


if __name__=='__main__':main()
