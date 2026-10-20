"""Real DMAD factors and Native blocks on local logical CUDA ranks.

No physical dual-T4 performance claims. Only retained Native INT32 modes run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import simulate_weight_residency as setup
import torch
from torch import nn
import torch.nn.functional as F
import retest
import te_dit_study as study
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from patches.h3_lora import H3LoRAStore,NativeLoraLinear,NativeLoraMLP
from patches.windowed_tp import CheckpointReader,prepare_tp_block,tp_specs
from patches.weight_residency import copy_module
from patches.optimized_tp import enable,disable


class NativeSequential(nn.Module):
    """Independent upstream bypass reference, including per-adapter scaling."""
    def __init__(self,terms):
        super().__init__()
        from comfy.weight_adapter.lora import LoRAAdapter
        self.adapters=[]
        for a,b,scale in terms:
            adapter=LoRAAdapter(set(),(b.cuda().float(),a.cuda().float(),None,None,None,None))
            adapter.multiplier=scale
            self.adapters.append(adapter)
    def add_to(self,x,out,comm=None,input_act=None):
        if input_act=='swiglu':
            gate,up=x.chunk(2,-1);x=F.silu(gate)*up
        for adapter in self.adapters:out=out+adapter.h(x,out)
        return out


def native_adapt(block,store,index,prefix=None):
    prefix=prefix or f'blocks.{index}.'
    for name in ('qkv_proj','out_proj'):
        terms=store.terms(prefix+'attn.'+name)
        if terms:setattr(block.attn,name,NativeLoraLinear(getattr(block.attn,name),NativeSequential(terms)))
    a=store.terms(prefix+'mlp.fc1');b=store.terms(prefix+'mlp.fc2')
    block.mlp=NativeLoraMLP(block.mlp,NativeSequential(a) if a else None,NativeSequential(b) if b else None)
    return block


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--blocks-only',action='store_true')
    parser.add_argument('--base',default=str(setup.PATHS['dit']))
    parser.add_argument('--lora',default='D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors')
    parser.add_argument('--strengths',nargs='+',type=float,default=[.6,-.2])
    parser.add_argument('--receipt',default='semantic_results/h3_lora.json')
    parser.add_argument('--indices',nargs='+',type=int,default=[0])
    parser.add_argument('--rows',nargs='+',type=int,default=[33,257,512])
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];cases=[]
    import comfy_kitchen as ck
    import patches.tp_dit as dit
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda _:49152
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;isolate_cublas_workspaces()
    base=CheckpointReader(args.base)
    # Different signs exercise concatenation and alpha preservation, not just a single adapter.
    store=H3LoRAStore(tuple((args.lora,s) for s in args.strengths),base)
    print('LORA targets',len(store.targets),'sources',len(store.sources),flush=True)
    def record(label,actual,expected,required=True,**extra):
        score=retest.metrics(actual.cpu(),expected.cpu())
        # Compare representation too, including signed zero.
        exact=torch.equal(actual.detach().contiguous().view(torch.uint8).cpu(),
                          expected.detach().contiguous().view(torch.uint8).cpu())
        score['is_bit_exact']=exact
        passed=exact
        cases.append(dict(label=label,required=required,passed=passed,**score,**extra))
        print(label,'pass',passed,'rel_l2',score.get('rel_l2'),'max_abs',score.get('max_abs'),flush=True)
    if not args.blocks_only:
        for prefix in ('blocks.0.','blocks.25.','blocks.49.','token_refiner.blocks.0.','token_refiner.blocks.1.'):
            for name in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2'):
                target=prefix+name;terms=store.terms(target)
                width=terms[0][0].shape[1];output=terms[0][1].shape[0]
                torch.manual_seed(123);x=torch.randn(37,width,device='cuda')
                ref=NativeSequential(terms).add_to(x,torch.zeros(37,output,device='cuda'))
                comm=LogicalDualGPU(120);adapters=[store.adapter(target,r).cuda() for r in (0,1)]
                if name.endswith(('qkv_proj','fc1')):
                    result=comm.run([lambda r=r:adapters[r].add_to(x,torch.zeros(37,output//2,device='cuda')) for r in (0,1)])
                    components=3 if name.endswith('qkv_proj') else 2
                    actual=torch.cat([torch.cat([result[r].chunk(components,-1)[c] for r in (0,1)],-1) for c in range(components)],-1)
                else:
                    parts=x.chunk(2,-1)
                    result=comm.run([lambda r=r:adapters[r].add_to(parts[r],torch.zeros(37,output,device='cuda'),comm) for r in (0,1)])
                    actual=result[0];assert torch.equal(result[0],result[1])
                record(target,actual,ref)
                del adapters,terms,x,ref,result,actual;torch.cuda.empty_cache()
        # Real dense BF16 refiner backing is expanded to Native FP32; test
        # wrappers after the same CPU→GPU shell copy used by the pipeline.
        import comfy.ops
        from comfy.ldm.minimax.model import RefinerBlock
        ops=comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.float32)
        raw=RefinerBlock(5376,56,128,14336,1e-5,1e-5,device='cpu',dtype=torch.float32,operations=ops)
        raw.load_state_dict(base.read('token_refiner.blocks.0.'),strict=True)
        shell=nn.Module();shell.token_refiner=nn.Module();shell.token_refiner.blocks=nn.ModuleList([raw])
        candidate=copy_module(shell,lambda t:t.cuda())
        store.attach_refiner(candidate)
        # Exercise one more shell copy with adapter buffers present.
        candidate=copy_module(candidate,lambda t:t.to('cuda',copy=True)).token_refiner.blocks[0]
        expected=native_adapt(raw.cuda(),store,0,'token_refiner.blocks.0.')
        for count in (37,257):
            torch.manual_seed(123);x=torch.randn(count,5376,device='cuda')
            record(f'Native_refiner_block0_S{count}',candidate(x.clone()),expected(x.clone()))
        del candidate,expected,raw,shell,x;torch.cuda.empty_cache()
    original_comm=dit.dist
    try:
        for index in args.indices:
            for count in args.rows:
                xlist,alist=study.inputs('dit',args.base,count,1,'cuda:0');x=xlist[0];a=alist[0]
                # Require the existing base implementation to remain bit-exact.
                native=retest.load_native_block('dit',args.base,index=index)
                no_lora_reference=native(x.clone(),*a)
                enable('scalemax');comm=LogicalDualGPU(180);dit.dist=comm
                plain=[prepare_tp_block(CheckpointReader(args.base),'dit',index,r).cuda() for r in (0,1)]
                outputs=comm.run([lambda r=r:plain[r](x.clone(),*a) for r in (0,1)])
                record(f'base_B{index}_S{count}',outputs[0],no_lora_reference)
                assert torch.equal(outputs[0],no_lora_reference)
                del plain,outputs,no_lora_reference
                native=native_adapt(native,store,index)
                captures={}
                def capture(name):
                    def hook(module,inputs,output):captures[name]=output.detach().clone()
                    return hook
                native.mlp.fc1.register_forward_hook(capture('native_base_fc1'))
                # NativeLoraMLP applies the bypass outside fc1.forward.
                original_add=native.mlp.fc1_adapter.add_to
                def capture_add(*pos,**kw):
                    out=original_add(*pos,**kw);captures['native_fc1']=out.detach().clone();return out
                native.mlp.fc1_adapter.add_to=capture_add
                reference=native(x.clone(),*a);del native;torch.cuda.empty_cache()
                base.lora_store=store
                assert tp_specs(base,'dit',0,1)[0].key!=tp_specs(CheckpointReader(args.base),'dit',0,1)[0].key
                blocks=[prepare_tp_block(base,'dit',index,r).cuda() for r in (0,1)]
                for r in (0,1):blocks[r].fc1.register_forward_hook(capture(f'tp_fc1_{r}'))
                for mode,reduction in [('scalemax','native_int32')]:
                    enable(mode,swiglu_reduction=reduction);comm=LogicalDualGPU(180);dit.dist=comm
                    outputs=comm.run([lambda r=r:blocks[r](x.clone(),*a) for r in (0,1)])
                    assert torch.equal(outputs[0],outputs[1])
                    record(f'block{index}_S{count}_{mode}_{reduction}',outputs[0],reference,required=True)
                    if not torch.equal(outputs[0],reference):raise RuntimeError('Native-order LoRA must be byte exact before T4 submission')
                    if mode=='scalemax' and reduction=='native_int32':
                        from patches.parallel_int8 import quantize
                        packed=torch.cat([torch.cat([captures[f'tp_fc1_{r}'].chunk(2,-1)[c] for r in (0,1)],-1) for c in (0,1)],-1)
                        nq,ns=quantize(captures['native_fc1'],input_act='swiglu');tq,ts=quantize(packed,input_act='swiglu')
                        cases.append(dict(label=f'fc1_quantization_threshold_S{count}',required=False,
                            pre_quantization=retest.metrics(packed,captures['native_fc1']),
                            mismatched_int8=int((nq!=tq).sum()),total_int8=nq.numel(),
                            changed_row_scales=int((ns!=ts).sum())))
                        del packed,nq,ns,tq,ts
                    del outputs
                captures.clear();del blocks,xlist,alist,x,a,reference;torch.cuda.empty_cache()
    finally:dit.dist=original_comm;disable()
    result=dict(passed=all(c['passed'] for c in cases if c['required']),physical_t4=False,
        all_required_byte_exact=all(c.get('is_bit_exact') for c in cases if c['required']),
        indices=args.indices,rows=args.rows,
        scope='Real DMAD weights; independent upstream sequential bypass reference; logical ranks on RTX 5060 Ti',
        stack=[[str(Path(args.lora).resolve()),s] for s in args.strengths],cases=cases,
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ('patches/h3_lora.py','patches/tp_dit.py','patches/optimized_tp.py','benchmarks/qualify_h3_lora.py')})
    (root/args.receipt).write_text(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Native LoRA qualification failed')


if __name__=='__main__':main()
