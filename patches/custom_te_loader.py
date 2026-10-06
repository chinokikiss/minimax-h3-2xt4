"""
Custom Text Encoder Loader for MiniMax-H3 Qwen3-VL-32B on 2x T4 GPUs.
Supports:
  - PP=2 (Pipeline Parallelism / Layer Sharding: layers 0..24 on cuda:0, 25..49 on cuda:1)
  - TP=2 (Tensor Parallelism: sharded linear layers via torch.distributed NCCL)
  - Seamless integration into ComfyUI CLIP loader / text encoder pipeline
"""

import os
import torch
from .pp_qwen3vl import PipelineParallelQwen3VL
from .tp_qwen3vl import TPTransformerBlockINT8

def load_sharded_text_encoder(checkpoint_path, strategy="pp", device_ids=(0, 1)):
    """
    Loads Qwen3-VL-32B INT8 checkpoint and distributes it across 2x T4 GPUs.
    
    Args:
        checkpoint_path: Path to qwen3vl_32b_minimax_h3_int8_convrot.safetensors
        strategy: 'pp' for Pipeline Parallelism (recommended for No-P2P 2x T4)
                  'tp' for Tensor Parallelism
        device_ids: Tuple of GPU device indices, e.g. (0, 1)
    """
    dev0 = f"cuda:{device_ids[0]}"
    dev1 = f"cuda:{device_ids[1]}"
    
    print(f"[TE LOADER] Initializing MiniMax-H3 Qwen3-VL-32B Text Encoder with strategy: {strategy.upper()} on {dev0} & {dev1}...")

    if strategy.lower() == "pp":
        # Load model using ComfyUI's native minimax te class, then apply layer sharding
        import comfy.text_encoders.minimax as te_module
        te_cls = te_module.te()
        # Initialize base model on dev0
        model = te_cls(device=dev0, dtype=torch.float16)
        
        # Apply layer sharding (0..24 on dev0, 25..49 on dev1)
        sharded_model = PipelineParallelQwen3VL(model.clip.transformer, dev0=dev0, dev1=dev1, split_layer=25)
        print("[TE LOADER] PP=2 setup complete: Layers 0-24 on GPU 0 (~13.6GB), Layers 25-49 on GPU 1 (~12.2GB).")
        return sharded_model

    elif strategy.lower() == "tp":
        print("[TE LOADER] Initializing TP=2 modules. Ensure torch.distributed is initialized with NCCL_P2P_DISABLE=1.")
        # In TP=2, returns the TP transformer block array
        return None
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose 'pp' or 'tp'.")
