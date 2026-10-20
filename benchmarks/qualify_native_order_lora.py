"""Long learned FFN + mixed-rank/alpha local gates before physical submission."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import simulate_weight_residency as setup
import torch
from safetensors.torch import save_file
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from qualify_h3_lora import NativeSequential
import retest
from patches.h3_lora import H3LoRAStore,NativeLoraMLP
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.optimized_tp import enable,disable


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--receipt",default="semantic_results/h3_lora_native_order_long.json");args=parser.parse_args()
    import comfy_kitchen as ck
    import patches.tp_dit as dit
    from comfy.weight_adapter.lora import LoRAAdapter
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda _:49152
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;isolate_cublas_workspaces()
    root=Path(__file__).resolve().parents[1];rows=[];base=CheckpointReader(setup.PATHS['dit'])
    path='D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors'
    saved=dit.dist
    def record(label,actual,ref,**extra):
        m=retest.metrics(actual.cpu(),ref.cpu());exact=torch.equal(actual.contiguous().view(torch.uint8).cpu(),ref.contiguous().view(torch.uint8).cpu());m['is_bit_exact']=exact
        rows.append(dict(label=label,passed=exact,**m,**extra));print('NATIVE_ORDER',label,'exact',exact,'rel',m.get('rel_l2'),flush=True)
        if not exact:raise RuntimeError('Native byte equivalence failed before physical submission')
    try:
        with tempfile.TemporaryDirectory(prefix='h3_lora_factors_') as directory:
            # Two independent adapters, unequal ranks and nontrivial alpha;
            # zero-strength entries exercise source ordering without adding work.
            paths=[]
            for index,(rank,alpha) in enumerate(((7,5.5),(13,9.25))):
                torch.manual_seed(771+index);data={}
                for target in ('blocks.0.attn.qkv_proj','blocks.0.mlp.fc2'):
                    shape=base.header[target+'.weight']['shape'];raw='diffusion_model.'+target
                    data[raw+'.lora_A.weight']=(torch.randn(rank,shape[1])*.01).half()
                    data[raw+'.lora_B.weight']=(torch.randn(shape[0],rank)*.01).half()
                    data[raw+'.alpha']=torch.tensor(alpha)
                p=Path(directory)/f'rank{rank}.safetensors';save_file(data,str(p));paths.append(str(p))
            stack=((paths[0],.7),(paths[1],-.15),(paths[0],0.))
            store=H3LoRAStore(stack,base)
            for target in ('blocks.0.attn.qkv_proj','blocks.0.mlp.fc2'):
                for count in (33,257):
                    width=base.header[target+'.weight']['shape'][1];out_width=base.header[target+'.weight']['shape'][0]
                    torch.manual_seed(123);x=torch.randn(count,width,device='cuda');ref=torch.zeros(count,out_width,device='cuda')
                    # Reference reads original alpha and strength independently.
                    for reader,strength,mapping in store.sources:
                        if not strength:continue
                        a,b,_=mapping[target];alpha=reader.read_tensor('diffusion_model.'+target+'.alpha').item()
                        adapter=LoRAAdapter(set(),(reader.read_tensor(b).cuda().float(),reader.read_tensor(a).cuda().float(),alpha,None,None,None))
                        adapter.multiplier=strength;ref=ref+adapter.h(x,ref)
                    adapters=[store.adapter(target,r).cuda() for r in (0,1)];comm=LogicalDualGPU(180)
                    if target.endswith('qkv_proj'):
                        outs=comm.run([lambda r=r:adapters[r].add_to(x,torch.zeros(count,out_width//2,device='cuda')) for r in (0,1)])
                        actual=torch.cat([torch.cat([outs[r].chunk(3,-1)[c] for r in (0,1)],-1) for c in range(3)],-1)
                    else:
                        parts=x.chunk(2,-1)
                        outs=comm.run([lambda r=r:adapters[r].add_to(parts[r],torch.zeros(count,out_width,device='cuda'),comm) for r in (0,1)])
                        actual=outs[0];assert torch.equal(outs[0],outs[1])
                    record(f'mixed_rank_alpha_{target}_S{count}',actual,ref)
                    del adapters,outs,actual,x,ref;torch.cuda.empty_cache()
        for strengths in ((1.,),(.6,-.2)):
            store=H3LoRAStore(tuple((path,s) for s in strengths),base)
            native=retest.load_native_block('dit',base.path);mlp=native.mlp;del native
            native=NativeLoraMLP(mlp,NativeSequential(store.terms('blocks.0.mlp.fc1')),NativeSequential(store.terms('blocks.0.mlp.fc2')))
            base.lora_store=store;blocks=[prepare_tp_block(base,'dit',0,r).cuda() for r in (0,1)]
            for count in (513,8192):
                torch.manual_seed(900);x=torch.randn(count,5376,device='cuda')
                reference=native(x).cpu()
                for mode in ('scalemax',):
                    enable(mode);comm=LogicalDualGPU(180);dit.dist=comm
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=torch.cuda.memory_allocated()
                    outs=comm.run([lambda r=r:blocks[r].fc2(blocks[r].fc1(x),input_act='swiglu') for r in (0,1)])
                    peak=torch.cuda.max_memory_allocated()-start
                    assert torch.equal(outs[0],outs[1]);record(f'learned_FFN_S{count}_{strengths}_{mode}',outs[0],reference,temporary_peak_bytes=peak)
                    del outs
                del x,reference;torch.cuda.empty_cache()
            del native,mlp,blocks,store;torch.cuda.empty_cache()
    finally:dit.dist=saved;disable()
    result=dict(passed=all(r['passed'] for r in rows),physical_t4=False,cases=rows,
        scope='Local numerical and memory gates only; two logical CUDA ranks on one RTX 5060 Ti',
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ('patches/h3_lora.py','patches/tp_dit.py','patches/optimized_tp.py','benchmarks/qualify_native_order_lora.py')})
    (root/args.receipt).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
