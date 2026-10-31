"""Minimal SM75 Sol All-INT8 adapter for isolated T4 qualification."""
from dataclasses import dataclass
import torch
from . import sol_sm75_backend as native

DEFAULT_TAU = 1.0

@dataclass(frozen=True)
class Result:
    output: torch.Tensor
    density: float
    minimum: int
    maximum: int
    query_blocks: int
    key_blocks: int


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, tau: float = DEFAULT_TAU) -> Result:
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("Sol All-INT8 requires equal [B,H,L,128] Q/K/V")
    if q.dtype != torch.float16 or any(t.dtype != torch.float16 for t in (k, v)):
        raise TypeError("Sol All-INT8 requires FP16 Q/K/V")
    if not all(t.is_cuda and t.device == q.device and t.is_contiguous() for t in (q, k, v)):
        raise ValueError("Sol All-INT8 requires contiguous same-device CUDA Q/K/V")
    if q.shape[-1] != 128:
        raise ValueError("Sol All-INT8 requires head_dim=128")
    if torch.cuda.get_device_capability(q.device) != (7, 5):
        raise RuntimeError("This experiment is qualified only on SM75")
    routing = native.prepare(q, k, v, tau=float(tau), quantize_qk=True)
    q8, qs = routing["q_int8"], routing["q_scale"]
    k8, ks = routing["k_int8"], routing["k_scale"]
    query_blocks = (q.shape[-2] + native.SOL_BLOCK_Q - 1) // native.SOL_BLOCK_Q
    required_q_scales = query_blocks * 4
    if qs.shape[-1] < required_q_scales:
        qs = torch.nn.functional.pad(
            qs, (0, required_q_scales - qs.shape[-1]), value=1.0,
        )
    kc8, kcs = native.quantize(routing["k_centroid"], 64)
    approximation = (
        kc8, routing["v_centroid"], kcs, routing["exact_mask"],
        int(routing["centroid_count"]), int(routing["centroid_padded"]),
    )
    out = native.run(
        q8, k8, v, qs, ks, routing["row_count"], routing["lut"],
        all_int8=True, approximation=approximation,
        validated_routing=bool(routing.get("validated_routing", False)),
    )
    key_blocks = (q.shape[-2] + native.SOL_BLOCK_K - 1) // native.SOL_BLOCK_K
    return Result(out, float(routing["density"]), int(routing["minimum"]),
                  int(routing["maximum"]), query_blocks, key_blocks)
