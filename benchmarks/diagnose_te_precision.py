import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'ComfyUI',ROOT/'.retest-deps',ROOT/'benchmarks'):sys.path.insert(0,str(p))
import torch
import retest
from te_dit_study import inputs,invoke
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from patches.optimized_tp import enable
from patches.tp_qwen3vl import TPTransformerBlockINT8
import patches.tp_qwen3vl as te
import comfy_kitchen as ck
ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda x:49152
isolate_cublas_workspaces();enable('scalemax');torch.backends.cuda.matmul.allow_tf32=False
path='D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'
with torch.inference_mode():
    native=retest.load_native_block('te',path,0)
    x,args=inputs('te',path,24,1,torch.device('cuda:0'));x=x.float();args=(args[0].float(),args[1],args[2])
    ref=invoke('te',native,x.clone(),args,True)
    blocks=[]
    for rank in range(2):
        block=TPTransformerBlockINT8(rank=rank,device='cuda:0').cuda()
        block.load_from_full_block(native);blocks.append(block)
    norm=native.input_layernorm(x)
    from comfy.ldm.common_dit import rms_norm
    norm2=rms_norm(x,blocks[0].input_layernorm.weight,1e-6)
    print('NORM',retest.metrics(norm2,norm),flush=True)
    q=native.self_attn.q_proj(norm)
    qp=torch.cat([b.q_proj(norm) for b in blocks],dim=-1)
    print('Q',retest.metrics(qp,q),flush=True)
    comm=LogicalDualGPU();te.dist=comm
    out=comm.run([lambda b=b:b(x.clone(),args[0],args[1]) for b in blocks])
    print('BLOCK FP32',retest.metrics(out[0],ref),flush=True)
