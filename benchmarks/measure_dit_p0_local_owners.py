"""Read-only local CUDA Storage ownership ledger at production tensor geometries.

Measures bytes released after the producer has finished. This does not execute
SM75 kernels and is not an E2E PyTorch Allocated Peak measurement.
"""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
torch.cuda.set_device(0)
torch.set_grad_enabled(False)

def gib(value):
    return value / 2**30

def measure(rows):
    d = torch.device("cuda")
    result = {"rows": rows}
    torch.cuda.empty_cache()

    # P0-3: FC1 output Tensor must own its 1.740-GiB Storage.
    fc1 = torch.empty((rows, 14336), device=d, dtype=torch.float16)
    view = fc1.view(-1, 14336)
    before = torch.cuda.memory_allocated(d)
    view.set_()
    result["p0_3_view_only_gib"] = gib(before - torch.cuda.memory_allocated(d))
    fc1.set_()
    result["p0_3_owner_and_view_gib"] = gib(before - torch.cuda.memory_allocated(d))
    del fc1, view

    # P0-2: attention FP16 is consumable after value.float(); the FP32
    # staging can be consumed once the packing kernel has read its input.
    attn = torch.empty((rows, 3584), device=d, dtype=torch.float16)
    staged = attn.float()
    before = torch.cuda.memory_allocated(d)
    attn.set_()
    result["p0_2_attention_fp16_gib"] = gib(before - torch.cuda.memory_allocated(d))
    padded = torch.empty((rows, 7168), device=d, dtype=torch.float32)
    flat = staged.reshape(-1, 3584)
    before = torch.cuda.memory_allocated(d)
    flat.set_()
    result["p0_2_fp32_view_only_gib"] = gib(before - torch.cuda.memory_allocated(d))
    staged.set_()
    result["p0_2_staged_fp32_gib"] = gib(before - torch.cuda.memory_allocated(d))
    del padded, flat, attn, staged

    # P0-1: h2 and ColumnParallel's x_2d alias must both be detached.
    h2 = torch.empty((rows, 5376), device=d, dtype=torch.float16)
    quant_view = h2.view(-1, 5376)
    before = torch.cuda.memory_allocated(d)
    quant_view.set_()
    result["p0_1_view_only_gib"] = gib(before - torch.cuda.memory_allocated(d))
    h2.set_()
    result["p0_1_owner_and_view_gib"] = gib(before - torch.cuda.memory_allocated(d))
    del quant_view, h2
    torch.cuda.synchronize()
    return result

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Requires local CUDA")
    final = {"device": torch.cuda.get_device_name(0),
             "torch": str(torch.__version__),
             "warning": "Storage geometry ledger only; no SM75 real-kernel overlap or global peak proof",
             "geometries": [measure(rows) for rows in (59134, 65166)]}
    output = ROOT / "docs/data/h3_dit_p0_lifetime_20261102_v2/local_gpu_owner_ledger.json"
    output.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
