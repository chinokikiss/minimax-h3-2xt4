"""Opt-in approximate INT8 SUM with shared scales and overflow headroom."""
import torch


def int8_sum(value, comm, world_size):
    """Return FP32 reduced values; input is not mutated.

    All ranks use a shared per-row scale and clamp to floor(127 / ranks).
    Thus the *INT8* SUM cannot overflow, even with equal-sign maxima.
    On TP2 this gives 63 levels per sign per rank (about six bits), not
    independent full-range INT8 scales. This is intentionally approximate.
    Caller must bound the number of rows to bound FP32 staging allocations.
    """
    if world_size == 1:
        return value.float()
    if not 2 <= world_size <= 127:
        raise ValueError('INT8 reduction supports 2–127 ranks')
    if value.ndim != 2 or value.dtype != torch.float32:
        raise ValueError('INT8 SUM expects a finite FP32 row matrix')
    levels = 127 // world_size
    maximum = value.abs().amax(-1, keepdim=True)
    comm.all_reduce(maximum, op=comm.ReduceOp.MAX)
    # Zero rows have an arbitrary nonzero scale and an exactly zero payload.
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / levels)
    payload = (value / scale).round().clamp_(-levels, levels).to(torch.int8)
    comm.all_reduce(payload, op=comm.ReduceOp.SUM)
    return payload.float().mul_(scale)
