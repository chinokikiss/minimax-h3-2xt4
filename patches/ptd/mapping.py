# SPDX-License-Identifier: Apache-2.0
# Adapted from vita-epfl/PTD-inference (ptd_comfy/checkpoint.py), Apache-2.0,
# Copyright (c) 2026 EPFL VITA Lab. See NOTICE.
"""Map this repository's Diffusers adapter keys onto ComfyUI's native MiniMax-H3 modules.

The released checkpoints name modules as Diffusers does, minus the ``transformer.`` prefix:
``transformer_blocks.{i}.attn.to_q.lora_down``. ComfyUI's native H3 fuses Q/K/V into one
``qkv_proj`` and uses its own block names, so each adapter needs a target module, an optional
fused-QKV slice, and — for the SwiGLU gate projection — a row permutation.

Two things differ from the upstream PTD interchange format:

* Factors are stored stacked over hops (``lora_down [H, r, in]``), not as one tensor per hop.
* The output heads are a PDD per-interval head bank, not adapters, so they are not mapped here;
  ``heads.py`` installs them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

# The six backbone projections PTD adapts (no AdaLN / time-embed LoRA, per the paper).
LAYER_PARTS = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "ff.net.0.proj", "ff.net.2",
)
HEAD_PREFIXES = ("proj_out", "audio_proj_out")
# ComfyUI's native H3 names for the two output heads this repo replaces wholesale.
NATIVE_HEADS = {"proj_out": "final_layer.video_out", "audio_proj_out": "final_layer.audio_out"}

_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<factor>down|up)$")
_BLOCK = re.compile(r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(0|[1-9][0-9]*)\.(.+)$")
_QKV = {"attn.to_q": 0, "attn.to_k": 1, "attn.to_v": 2}
_DENSE = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}


@dataclass(frozen=True)
class AdapterTarget:
    """Native module name; ``qkv_index`` selects one third of a fused projection."""

    module: str
    qkv_index: int | None = None

    @property
    def swap_halves(self) -> bool:
        """Native SwiGLU expects [gate, up]; Diffusers stores [up, gate]."""
        return self.module.endswith(".mlp.fc1")


def map_adapter_key(key: str) -> tuple[AdapterTarget, str]:
    """Map one stacked adapter key to ``(target, "down" | "up")``.

    Unknown prefixes, projections and factor names fail explicitly rather than being skipped,
    so a checkpoint this mapping does not fully understand cannot load half-applied.
    """
    match = _KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"Not a supported PTD adapter key: {key}")
    module = match["module"]
    block = _BLOCK.fullmatch(module)
    if block is None:
        raise ValueError(f"Unknown PTD adapter module: {module}")
    prefix = "blocks" if block[1] == "transformer_blocks" else "token_refiner.blocks"
    base, part = f"{prefix}.{block[2]}", block[3]
    if part in _QKV:
        return AdapterTarget(f"{base}.attn.qkv_proj", _QKV[part]), match["factor"]
    if part not in _DENSE:
        raise ValueError(f"Unknown PTD adapter projection: {module}")
    return AdapterTarget(f"{base}.{_DENSE[part]}"), match["factor"]


def expected_adapter_keys(blocks: int, refiner_blocks: int) -> set[str]:
    """Every adapter key a complete checkpoint of this topology must carry."""
    modules = [f"transformer_blocks.{i}.{part}" for i in range(blocks) for part in LAYER_PARTS]
    modules += [
        f"token_refiner.refiner_blocks.{i}.{part}"
        for i in range(refiner_blocks)
        for part in LAYER_PARTS
    ]
    return {f"{module}.lora_{factor}" for module in modules for factor in ("down", "up")}


def expected_head_keys() -> set[str]:
    return {f"{head}.{part}" for head in HEAD_PREFIXES for part in ("weight", "bias")}
