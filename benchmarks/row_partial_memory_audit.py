"""Row Partial memory A/B: preserve prior outputs and measure actual peaks."""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI')),str(ROOT/'benchmarks')]
if '--physical' not in sys.argv:import simulate_weight_residency as setup
import torch
import torch.distributed as dist
import retest
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from physical_h3_lora_repair import native_adapt
from physical_h3_reduction import SequentialNative,timed,profile,ProfileComm
from fixture_inputs import block_input
from patches.h3_lora import H3LoRAStore,NativeLoraMLP
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.optimized_tp import enable,disable
import patches.optimized_tp as optimized
from row_partial_memory_baseline import BaselineRowPartial,global_row_quantize as OLD_QUANT
NEW_QUANT=optimized.global_row_quantize

CONFIGS=('before','native','quant_only','row_partial')
SOURCES=('patches/h3_lora.py','patches/optimized_tp.py','patches/tp_dit.py','patches/comfy_tp_runtime.py',
         'custom_nodes/minimax_h3_tp2/__init__.py','benchmarks/row_partial_memory_audit.py','benchmarks/row_partial_memory_baseline.py')


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--checkpoint',default='D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')
    p.add_argument('--lora',default='D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import patches.tp_dit as dit
    import comfy_kitchen as ck
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(2)
    rank=int(os.environ['LOCAL_RANK']) if args.physical else 0
    if args.physical:
        torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=timedelta(seconds=180))
        assert dist.get_world_size()==2 and all('T4' in torch.cuda.get_device_name(i) for i in (0,1))
    else:
        ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda _:49152;isolate_cublas_workspaces()
    enable('scalemax');saved=dit.dist;args.output.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',physical_dual_t4=args.physical,
        scope='Real DMAD Row Partial before/after allocation lifetime changes; selected blocks, learned FFN and isolated row LoRA, not full generation',
        quality_gate='Native byte exact; all Row Partial memory variants must exactly match previous Row Partial bytes',
        torch=torch.__version__,block_cases=[],projection_cases=[],profiles=[],source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES})
    base=CheckpointReader(args.checkpoint);curve=base.read_tensor('adaln_t_table')
    def save():
        if rank==0:retest.write_json(args.output/'results.json',state)
    def store_for(strengths,partial=False):
        return H3LoRAStore(tuple((args.lora,s) for s in strengths),base,row_partial=partial)
    def select(name):optimized.global_row_quantize=OLD_QUANT if name=='before' else NEW_QUANT
    def blocks_for(store,index,name):
        reader=CheckpointReader(args.checkpoint);reader.lora_store=store
        blocks=[prepare_tp_block(reader,'dit',index,r).cuda() for r in ((rank,) if args.physical else (0,1))]
        if name in ('before','quant_only'):
            for block in blocks:
                for target in ('out_proj','fc2'):
                    old=getattr(block,target).lora_adapter
                    setattr(getattr(block,target),'lora_adapter',BaselineRowPartial(
                        [(t.down,t.up,t.scale) for t in old.terms],columns=old.columns,row_partial=True))
        return blocks
    def execute(blocks,fn):
        if args.physical:dit.dist=dist;return fn(blocks[0])
        transport=LogicalDualGPU(180);dit.dist=transport
        results=transport.run([lambda r=r:fn(blocks[r]) for r in (0,1)])
        assert torch.equal(results[0].contiguous().view(torch.uint8),results[1].contiguous().view(torch.uint8))
        return results[0]
    def checked(actual,reference,label,previous=None):
        cpu=actual.cpu();score=retest.metrics(cpu,reference)
        exact=torch.equal(cpu.contiguous().view(torch.uint8),reference.contiguous().view(torch.uint8))
        same=True
        if args.physical:
            hashes=[None,None];dist.all_gather_object(hashes,hashlib.sha256(memoryview(cpu.numpy()).cast('B')).hexdigest());same=hashes[0]==hashes[1]
        prior_exact=(torch.equal(cpu.contiguous().view(torch.uint8),previous.contiguous().view(torch.uint8)) if previous is not None else None)
        passed=same and (exact if label=='native' else prior_exact)
        score.update(native_bytes_exact=exact,previous_row_bytes_exact=prior_exact,replicas_equal=same,passed=passed)
        if not passed:raise RuntimeError('Output gate failed: '+label)
        return score
    def release():
        gc.collect();torch.cuda.empty_cache()
        if args.physical:dist.barrier()
    save()
    try:
        strengths=(.6,-.2);reference_store=store_for(strengths)
        for index in (0,25,49):
            native=native_adapt(retest.load_native_block('dit',args.checkpoint,index=index),reference_store,index)
            for count in (257,512):
                data=block_input('dit',count,900,curve);x=data['input'].cuda()
                call_args=(data['t_emb'].cuda(),data['segments'],data['rope'].cuda())
                reference=native(x.clone(),*call_args).cpu()
                previous=None
                for name in CONFIGS:
                    select(name);store=store_for(strengths,name!='native');blocks=blocks_for(store,index,name)
                    # Validate rank-local factor accounting, including no duplicate A buffers.
                    for offset,block in enumerate(blocks):
                        owned_rank=rank if args.physical else offset
                        actual_bytes=sum(t.numel()*t.element_size() for target in ('qkv_proj','out_proj','fc1','fc2')
                            for t in getattr(block,target).lora_adapter.buffers())
                        assert actual_bytes==store.block_bytes('dit',index,owned_rank)
                    actual=execute(blocks,lambda b:b(x.clone(),*call_args))
                    if name=='before':previous=actual.cpu()
                    row=dict(config=name,block=index,rows=count,adapter_bytes=store.block_bytes('dit',index,rank),**checked(actual,reference,name,previous))
                    state['block_cases'].append(row);save();print('OPTION_BLOCK',name,index,count,row['rel_l2'],flush=True)
                    del actual,blocks,store;release()
                del x,data,call_args,reference,previous
            del native;release()
        for strengths in ((1.,),(.6,-.2)):
            ref_store=store_for(strengths);raw=retest.load_native_block('dit',args.checkpoint).mlp
            native=NativeLoraMLP(raw,SequentialNative(ref_store.terms('blocks.0.mlp.fc1')),SequentialNative(ref_store.terms('blocks.0.mlp.fc2')))
            for count in (513,8192):
                torch.manual_seed(900);x=torch.randn(count,5376,device='cuda');reference=native(x).cpu()
                def ffn(b):return b.fc2(b.fc1(x),input_act='swiglu')
                select('before');store=store_for(strengths,True);blocks=blocks_for(store,0,'before')
                previous=execute(blocks,ffn).cpu();del blocks,store;release()
                for turn in ((0,1) if args.physical else (0,)):
                    for name in (CONFIGS if turn==0 else CONFIGS[::-1]):
                        select(name);store=store_for(strengths,name!='native');blocks=blocks_for(store,0,name)
                        actual=execute(blocks,ffn)
                        row=dict(scope='learned_FFN',config=name,adapter_mode='single' if len(strengths)==1 else 'multi',rows=count,round=turn,
                            adapter_bytes=store.block_bytes('dit',0,rank),**checked(actual,reference,name,previous))
                        del actual
                        if args.physical:
                            row['timing']=timed(lambda:ffn(blocks[0]),warmups=2,repeats=5)
                            if args.profile and count==8192 and len(strengths)==2 and turn==0:
                                class Comm(ProfileComm):
                                    def all_gather_into_tensor(self,output,value):
                                        self.calls.append(dict(op='ALL_GATHER',dtype=str(value.dtype),payload_bytes=value.numel()*value.element_size()))
                                        return dist.all_gather_into_tensor(output,value)
                                transport=Comm();dit.dist=transport
                                trace=profile(lambda:ffn(blocks[0]),args.output,'RowMemory_'+name)
                                state['profiles'].append(dict(config=name,rank=rank,trace=trace,collective_calls=transport.calls));dit.dist=dist
                        state['projection_cases'].append(row);save();print('OPTION_FFN',name,strengths,count,row['rel_l2'],flush=True)
                        del blocks,store;release()
                del x,reference,previous
            # Expose the adapter-only memory change separately from base quantization.
            torch.manual_seed(901);packed=torch.randn(8192,14336,device='cuda')
            previous=None
            for name in ('before','row_partial'):
                select(name);store=store_for(strengths,True);blocks=blocks_for(store,0,name)
                def adapter(b):
                    return b.fc2.lora_adapter.add_to(packed,torch.zeros(8192,5376,device='cuda'),dit.dist,input_act='swiglu')
                actual=execute(blocks,adapter)
                if name=='before':previous=actual.cpu()
                row=dict(scope='row_lora_only',config=name,adapter_mode='single' if len(strengths)==1 else 'multi',
                    rows=8192,round=0,**checked(actual,previous,name,previous));del actual
                if args.physical:row['timing']=timed(lambda:adapter(blocks[0]),warmups=2,repeats=5)
                state['projection_cases'].append(row);save()
                del blocks,store;release()
            del packed,previous,native,raw,ref_store;release()
        all_cases=state['block_cases']+state['projection_cases']
        state.update(status='complete',native_exact_passed=all(r['native_bytes_exact'] for r in all_cases if r['config']=='native'),
            previous_row_exact_passed=all(r['previous_row_bytes_exact'] for r in all_cases if r['config']!='native'),passed=True)
        save()
    except BaseException as error:state.update(status='failed',error_type=type(error).__name__,passed=False);save();raise
    finally:
        dit.dist=saved;optimized.global_row_quantize=NEW_QUANT;disable()
        if args.physical:torch.cuda.synchronize();dist.destroy_process_group()


if __name__=='__main__':main()
