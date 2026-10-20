"""Local INT8/BF16 bake arithmetic with two spawned workers on one GPU.

This validates workers/serialization, not physical dual-GPU performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import os
import tempfile
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),os.environ.get('COMFY_ROOT',str(Path(__file__).resolve().parents[1]/'ComfyUI'))]
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,
        default=Path('D:/attention_eval/h3_loading_stages_20261019/local_layer_bake.json'))
    args=parser.parse_args();sys.argv=sys.argv[:1]
    from safetensors.torch import save_file
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as Layout
    from patches.offline_lora import convert
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(774)
    with tempfile.TemporaryDirectory(prefix='h3_layer_bake_') as folder:
        root=Path(folder);weights={};targets=[]
        for layer in range(2):
            for projection in ('attn.out_proj','mlp.fc2'):
                target=f'blocks.{layer}.{projection}';targets.append(target)
                q,params=Layout.quantize(torch.randn(256,256,device='cuda')*.03,
                    per_channel=True,convrot=True,convrot_groupsize=256,stochastic_rounding=0)
                weights[target+'.weight']=q.cpu();weights[target+'.weight_scale']=params.scale.cpu()
                weights[target+'.comfy_quant']=torch.tensor(list(json.dumps(dict(format='int8_tensorwise',convrot=True,convrot_groupsize=256)).encode()),dtype=torch.uint8)
        target='token_refiner.blocks.0.attn.out_proj';targets.append(target)
        weights[target+'.weight']=torch.randn(128,256).bfloat16()
        base=root/'base.safetensors';save_file(weights,str(base));before=base.read_bytes();stack=[]
        for i,strength in ((0,.6),(1,-.2)):
            factors={};rank=4+4*i
            for target in targets:
                rows=weights[target+'.weight'].shape[0]
                factors['diffusion_model.'+target+'.lora_A.weight']=torch.randn(rank,256)*.01
                factors['diffusion_model.'+target+'.lora_B.weight']=torch.randn(rows,rank)*.01
                factors['diffusion_model.'+target+'.alpha']=torch.tensor(float(rank)*.7)
            path=root/f'lora{i}.safetensors';save_file(factors,str(path));stack.append((str(path),strength))
        serial=convert(base,stack,root/'serial.safetensors',verify_native=True,devices=('cuda:0',))
        dual=convert(base,stack,root/'workers.safetensors',verify_native=True,devices=('cuda:0','cuda:0'))
        assert Path(serial).read_bytes()==Path(dual).read_bytes()
        assert base.read_bytes()==before
        receipt=json.loads(Path(dual+'.json').read_text())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(scope='Two spawned CUDA workers sharing one local GPU. Arithmetic/serialization only, no T4 or physical dual-GPU speedup claim.',
            hardware=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
            serial_parallel_checkpoint='bitwise_pass',independent_native_verified_targets=receipt['verified_native_targets'],
            base_unchanged=True,layer_partitions=receipt['layer_partitions'],
            checkpoint_sha256=hashlib.sha256(Path(dual).read_bytes()).hexdigest()),indent=2))
    print('LOCAL_LAYER_BAKE_PASS',len(targets),'targets',flush=True)


if __name__=='__main__':main()
