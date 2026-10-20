"""Full-tensor Native reference loader, used only for explicit qualification."""
import torch


def prepare_tp_block_legacy(reader, kind, index, rank):
    """Use Native checkpoint loading and the existing validated shard loader."""
    import comfy.ops
    if kind == 'te':
        from comfy.text_encoders.llama import Qwen3VL_32BConfig, TransformerBlock
        from patches.tp_qwen3vl import TPTransformerBlockINT8
        operations = comfy.ops.mixed_precision_ops({'mixed_ops': True}, torch.float16)
        native = TransformerBlock(Qwen3VL_32BConfig(), index=index,
                                  device='cpu', dtype=torch.float16, ops=operations)
        native.load_state_dict(reader.read(f'model.layers.{index}.'), strict=True)
        block = TPTransformerBlockINT8(rank=rank, world_size=2, device='cpu')
    elif kind == 'dit':
        from comfy.ldm.minimax.model import DiTBlock
        from patches.tp_dit import TPDiTBlockINT8
        operations = comfy.ops.mixed_precision_ops({'mixed_ops': True}, torch.float32)
        native = DiTBlock(5376, 56, 128, 14336, 8, 1e-5, 1e-5,
                          apply_silu=False, adaln_dtype=torch.float32,
                          device='cpu', dtype=torch.float32, operations=operations)
        native.load_state_dict(reader.read(f'blocks.{index}.'), strict=True)
        block = TPDiTBlockINT8(rank=rank, world_size=2, device='cpu')
    else:
        raise ValueError('Only validated TE and DiT TP blocks are supported')
    block.load_from_full_block(native)
    return block.eval().requires_grad_(False)
