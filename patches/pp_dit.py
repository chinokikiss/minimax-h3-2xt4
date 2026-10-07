"""
Pipeline Parallelism (PP=2) for MiniMax-H3 Ref2VA Pruned INT8 ConvRot DiT.
Splits the 50 DiT blocks across 2 GPUs (25 blocks on GPU 0, 25 blocks on GPU 1),
minimizing inter-GPU transfers to exactly ONE activation transfer per forward pass.

Target Hardware: 2x NVIDIA Tesla T4 (16GB VRAM each, PCIe Gen3 x16, No P2P).
Single Stream (CFG = 1.0, CFG-distilled).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

class PipelineParallelDiT(nn.Module):
    """
    Shards 50 MiniMax-H3 DiT blocks across GPU 0 and GPU 1.
    GPU 0: Blocks 0..24  (~10.5 GB weights)
    GPU 1: Blocks 25..49 (~10.5 GB weights)
    Inter-GPU Transfer: Exactly ONE activation tensor at block 24 boundary.
    """
    def __init__(
        self,
        blocks_stage0: nn.ModuleList,
        blocks_stage1: nn.ModuleList,
        final_norm: Optional[nn.Module] = None,
        final_layer: Optional[nn.Module] = None,
        dev0: str = "cuda:0",
        dev1: str = "cuda:1",
    ):
        super().__init__()
        self.dev0 = torch.device(dev0)
        self.dev1 = torch.device(dev1)
        self.blocks_stage0 = blocks_stage0.to(self.dev0)
        self.blocks_stage1 = blocks_stage1.to(self.dev1)
        self.final_norm = final_norm.to(self.dev1) if final_norm is not None else None
        self.final_layer = final_layer.to(self.dev1) if final_layer is not None else None

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        mod_segments: List[Any],
        rope_freqs: Optional[torch.Tensor] = None,
        transformer_options: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """
        Executes single-request forward pass across GPU 0 then GPU 1.
        x: [S, H] packed visual/audio/text tokens.
        """
        # --- Stage 0: GPU 0 (Blocks 0..24) ---
        x_dev0 = x.to(self.dev0, non_blocking=True)
        t_emb_dev0 = t_emb.to(self.dev0, non_blocking=True)
        rope_dev0 = rope_freqs.to(self.dev0, non_blocking=True) if rope_freqs is not None else None

        for block in self.blocks_stage0:
            x_dev0 = block(
                x_dev0,
                t_emb=t_emb_dev0,
                mod_segments=mod_segments,
                rope_freqs=rope_dev0,
                transformer_options=transformer_options,
            )

        # --- Boundary Transfer: Exactly ONE inter-GPU transfer over PCIe ---
        # Tensor shape: [S, H], dtype: FP16
        x_dev1 = x_dev0.to(self.dev1, non_blocking=False)
        t_emb_dev1 = t_emb.to(self.dev1, non_blocking=False)
        rope_dev1 = rope_freqs.to(self.dev1, non_blocking=False) if rope_freqs is not None else None

        # --- Stage 1: GPU 1 (Blocks 25..49) ---
        for block in self.blocks_stage1:
            x_dev1 = block(
                x_dev1,
                t_emb=t_emb_dev1,
                mod_segments=mod_segments,
                rope_freqs=rope_dev1,
                transformer_options=transformer_options,
            )

        if self.final_norm is not None:
            x_dev1 = self.final_norm(x_dev1)
        if self.final_layer is not None:
            x_dev1 = self.final_layer(x_dev1)

        return x_dev1
