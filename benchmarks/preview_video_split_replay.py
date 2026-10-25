"""One local trained block checks diagnostic API calls; no T4 conclusion."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch


def main():
    sys.argv=sys.argv[:1]
    import comfy.ops
    from comfy.ldm.minimax.vae import TransformerBlock,RotaryEmbeddingND,create_token_ids
    from patches.windowed_tp import CheckpointReader
    from run_t4_video_split_localize import replay
    assert torch.cuda.get_device_capability()==(12,0)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    ops=comfy.ops.mixed_precision_ops({'mixed_ops':True},torch.float16)
    block=TransformerBlock(32,64,operations=ops)
    reader=CheckpointReader(Path('D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors'))
    block.load_state_dict(reader.read('decoder.transformer_blocks.0.'),strict=True)
    block=block.cuda().eval().requires_grad_(False)
    x=torch.randn(1,261,2048,device='cuda',dtype=torch.float16)
    coords=create_token_ids((1,16,16),'cuda',torch.float16)
    coords=torch.cat((coords,torch.zeros(1,5,3,device='cuda',dtype=torch.float16)),1)
    rope=RotaryEmbeddingND(48,n_dim=3)(coords)
    with torch.inference_mode():record=replay(block,x,rope,256,261)
    assert all(row['nonfinite']==0 for row in record.values())
    print(json.dumps(dict(scope='SM120 diagnostic API/arithmetic preview only; not physical T4 evidence',
        replay_operations=list(record),all_finite=True)))


if __name__=='__main__':main()
