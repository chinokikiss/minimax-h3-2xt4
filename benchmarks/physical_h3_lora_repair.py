"""Native-order LoRA repair: local-gated physical dual-T4 byte qualification."""
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
import torch
import torch.distributed as dist
import retest
from fixture_inputs import block_input
from physical_h3_reduction import timed,profile,ProfileComm,SequentialNative,score,agree
from patches.h3_lora import H3LoRAStore,NativeLoraLinear,NativeLoraMLP
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.optimized_tp import enable

MODES=('scalemax','scalemax_streamed')


def native_adapt(block,store,index):
    prefix=f'blocks.{index}.'
    for name in ('qkv_proj','out_proj'):
        setattr(block.attn,name,NativeLoraLinear(getattr(block.attn,name),SequentialNative(store.terms(prefix+'attn.'+name))))
    block.mlp=NativeLoraMLP(block.mlp,SequentialNative(store.terms(prefix+'mlp.fc1')),SequentialNative(store.terms(prefix+'mlp.fc2')))
    return block


def checked(actual,reference):
    cpu=actual.cpu();m=score(cpu,reference)
    exact=agree(torch.equal(cpu,reference))
    replicas=[None,None];dist.all_gather_object(replicas,hashlib.sha256(memoryview(cpu.numpy()).cast('B')).hexdigest())
    m.update(native_bytes_exact=exact,replicas_equal=replicas[0]==replicas[1],passed=exact and replicas[0]==replicas[1])
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--lora',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',action='store_true');args=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    assert dist.get_world_size()==2 and torch.cuda.device_count()==2
    assert all('T4' in torch.cuda.get_device_name(i) for i in (0,1))
    args.output.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',physical_dual_t4=True,implementation='native-order-full-geometry-v2',
        scope='Real DMAD single/multi-LoRA, complete H3 blocks0/25/49 and learned FFN; not full-pipeline generation',
        torch=torch.__version__,cuda=torch.version.cuda,devices=[torch.cuda.get_device_name(i) for i in (0,1)],
        memory_scope='Absolute peaks include resident reference weights and TP blocks. Temporary deltas are measured above resident inputs/weights.',
        block_cases=[],projection_cases=[],profiles=[],
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('patches/h3_lora.py','patches/tp_dit.py','patches/optimized_tp.py','benchmarks/physical_h3_lora_repair.py')})
    def save():
        if rank==0:retest.write_json(args.output/'results.json',state)
    save();original=None
    try:
        base=CheckpointReader(args.checkpoint);curve=base.read_tensor('adaln_t_table')
        for strengths in ((),(1.,),(.6,-.2)):
            adapter_mode='base' if not strengths else 'online_single' if len(strengths)==1 else 'online_multi'
            store=H3LoRAStore(tuple((args.lora,s) for s in strengths),base) if strengths else None
            reader=CheckpointReader(args.checkpoint)
            if store is not None:reader.lora_store=store
            for index in (0,25,49):
                native=retest.load_native_block('dit',args.checkpoint,index=index)
                if store is not None:native=native_adapt(native,store,index)
                block=prepare_tp_block(reader,'dit',index,rank).cuda()
                for count in (33,257,512,1024):
                    data=block_input('dit',count,900,curve);x=data['input'].cuda()
                    call_args=(data['t_emb'].cuda(),data['segments'],data['rope'].cuda())
                    reference=native(x.clone(),*call_args).cpu()
                    for mode in MODES:
                        enable(mode)
                        def call():return block(x.clone(),*call_args)
                        actual=call();m=checked(actual,reference);del actual
                        row=dict(adapter_mode=adapter_mode,block=index,rows=count,mode=mode,**m)
                        state['block_cases'].append(row);save()
                        if not m['passed']:raise RuntimeError('Native byte gate failed; performance rejected')
                        row['timing']=timed(call,warmups=1,repeats=3);save()
                        if rank==0:print('NATIVE_ORDER_BLOCK',adapter_mode,index,count,mode,'exact',m['native_bytes_exact'],flush=True)
                    del data,x,call_args,reference;torch.cuda.empty_cache()
                del native,block;gc.collect();torch.cuda.empty_cache();dist.barrier()
        import patches.tp_dit as dit
        original=dit.dist
        for strengths in ((1.,),(.6,-.2)):
            adapter_mode='online_single' if len(strengths)==1 else 'online_multi'
            store=H3LoRAStore(tuple((args.lora,s) for s in strengths),base)
            reader=CheckpointReader(args.checkpoint);reader.lora_store=store
            native=retest.load_native_block('dit',args.checkpoint)
            raw=native.mlp;del native
            native=NativeLoraMLP(raw,SequentialNative(store.terms('blocks.0.mlp.fc1')),SequentialNative(store.terms('blocks.0.mlp.fc2')))
            block=prepare_tp_block(reader,'dit',0,rank).cuda()
            for count in (513,8192):
                # Exactly the locally qualified learned FFN shapes and input distribution.
                torch.manual_seed(900);x=torch.randn(count,5376,device='cuda')
                reference=native(x).cpu()
                for round_index in (0,1):
                    for mode in (MODES if round_index==0 else MODES[::-1]):
                        enable(mode)
                        def call():return block.fc2(block.fc1(x),input_act='swiglu')
                        actual=call();m=checked(actual,reference);del actual
                        row=dict(scope='learned_FFN',adapter_mode=adapter_mode,rows=count,mode=mode,round=round_index,**m)
                        state['projection_cases'].append(row);save()
                        if not m['passed']:raise RuntimeError('Long FFN Native byte gate failed; performance rejected')
                        row['timing']=timed(call,warmups=2,repeats=5)
                        if count==8192 and adapter_mode=='online_multi' and round_index==0 and args.profile:
                            class Comm(ProfileComm):
                                def all_gather(self,parts,value):
                                    self.calls.append(dict(op='ALL_GATHER',dtype=str(value.dtype),elements=value.numel(),payload_bytes=value.numel()*value.element_size()))
                                    with torch.profiler.record_function('collective/ALL_GATHER/LoRA_activated_input'):
                                        return dist.all_gather(parts,value)
                            transport=Comm();dit.dist=transport
                            name=f'NativeOrder_S8192_multi_{mode}'
                            trace=profile(call,args.output,name);dit.dist=original
                            state['profiles'].append(dict(trace=trace,rank=rank,mode=mode,collective_calls=transport.calls))
                        save()
                        if rank==0:print('NATIVE_ORDER_FFN',adapter_mode,count,round_index,mode,'exact',m['native_bytes_exact'],
                                         'critical_ms',row['timing']['critical_wall_median_ms'],flush=True)
                del x,reference;torch.cuda.empty_cache()
            del block,native,raw,store,reader;gc.collect();torch.cuda.empty_cache();dist.barrier()
        state['native_exact_passed']=all(r['passed'] for r in state['block_cases']+state['projection_cases'])
        state['status']='complete';save()
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__);save();raise
    finally:
        if original is not None:dit.dist=original
        torch.cuda.synchronize();dist.destroy_process_group()


if __name__=='__main__':main()
