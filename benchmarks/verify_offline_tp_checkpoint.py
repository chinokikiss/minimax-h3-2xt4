"""Small physical TP2/Native gate for a fully offline-merged checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
import retest
from fixture_inputs import block_input
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.optimized_tp import enable
from types import SimpleNamespace
from gpu_memory_overhead import MemoryAudit


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    assert torch.cuda.device_count()==2 and torch.cuda.get_device_capability(rank)==(7,5)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(2)
    dist.init_process_group('nccl');enable('scalemax')
    reader=CheckpointReader(args.checkpoint);curve=reader.read_tensor('adaln_t_table');cases=[]
    os.environ['MINIMAX_H3_MEMORY_DIR']='/tmp/h3_offline_gate_memory'
    observed_runtime=SimpleNamespace(rank=rank,device=torch.device(f'cuda:{rank}'),
        context={'nfe':1},window=None,model=None,vae_model=None)
    audit=MemoryAudit(observed_runtime)
    try:
        for index in (0,49):
            module=prepare_tp_block(reader,'dit',index,rank).to(f'cuda:{rank}')
            observed_runtime.model=module
            assert not any(getattr(m,'lora_adapter',None) is not None for m in module.modules())
            native=retest.load_native_block('dit',args.checkpoint,index=index) if rank==0 else None
            for count in (33,513):
                data=block_input('dit',count,900,curve)
                x=data['input'].to(f'cuda:{rank}');t=data['t_emb'].to(f'cuda:{rank}');rope=data['rope'].to(f'cuda:{rank}')
                expected=native(x.clone(),t,data['segments'],rope) if rank==0 else torch.empty_like(x)
                dist.broadcast(expected,src=0)
                actual=module(x.clone(),t,data['segments'],rope)
                audit.begin('gate');audit.block=index
                profiled=module(x.clone(),t,data['segments'],rope)
                torch.cuda.synchronize();audit.flush()
                error=retest.metrics(actual,expected)
                observer_exact=torch.equal(profiled,actual)
                passed=torch.equal(actual,expected) and observer_exact
                agreed=torch.tensor(int(passed),device=f'cuda:{rank}');dist.all_reduce(agreed,op=dist.ReduceOp.MIN)
                cases.append(dict(block=index,sequence=count,**error,observer_byte_exact=observer_exact,passed=bool(agreed.item())))
                if not agreed.item():raise AssertionError('Merged-checkpoint Native/TP2 mismatch')
            del native,module;torch.cuda.empty_cache()
        if rank==0:
            args.output.write_text(json.dumps(dict(passed=True,cases=cases,
                scope='Physical merged-checkpoint Native versus TP2, Native attention/row arithmetic; not sparse-attention quality or online-LoRA equivalence'),indent=2))
            print('OFFLINE_PHYSICAL_TP_GATE_PASS',len(cases),'cases',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
