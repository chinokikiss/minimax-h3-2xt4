"""Archived P0-1 spot-check; requires frozen pre-removal sources, not live runtime."""
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ComfyUI")]
import comfy.cli_args
from patches.dit_column_linear import _prepare_activation
from inspect import signature
if 'inplace_quant' not in signature(_prepare_activation).parameters:
    raise RuntimeError('Archived P0-1 quantization study requires frozen source; live P0-1 was removed')

DEVICE = torch.device("cuda", 0)
torch.cuda.set_device(DEVICE)
torch.set_grad_enabled(False)
torch.manual_seed(20261102)


def one(rows):
    torch.cuda.empty_cache()
    x = torch.empty((rows, 5376), device=DEVICE, dtype=torch.float16).normal_(0., 1 / 512)
    def compute(inplace):
        return _prepare_activation(x, True, 256, inplace_quant=inplace)
    baseline_q, baseline_scale = compute(False)
    candidate_q, candidate_scale = compute(True)
    torch.cuda.synchronize()
    exact_q = bool(torch.equal(baseline_q, candidate_q))
    exact_scale = bool(torch.equal(baseline_scale, candidate_scale))
    del baseline_q, baseline_scale, candidate_q, candidate_scale
    if not exact_q or not exact_scale:
        raise AssertionError("P0-1 in-place quantization changed CUDA quantized codes/scales")
    results = {}
    for name, mode in (("baseline", False), ("inplace", True)):
        for _ in range(2):
            q, scale = compute(mode)
            del q, scale
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        initial = torch.cuda.memory_allocated(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        q, scale = compute(mode)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(DEVICE)
        results[name] = {
            "peak_allocated_gib": peak / 2**30,
            "incremental_peak_gib": (peak - initial) / 2**30,
            "input_allocated_gib": initial / 2**30,
        }
        del q, scale
    # Timed trials after warm-up, rotating order to reduce systematic bias.
    samples = {"baseline": [], "inplace": []}
    for repetition in range(8):
        order = [("baseline", False), ("inplace", True)]
        if repetition % 2:
            order.reverse()
        for name, mode in order:
            torch.cuda.synchronize()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            q, scale = compute(mode)
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end))
            del q, scale
    for name in samples:
        results[name]["median_cuda_ms"] = statistics.median(samples[name])
        results[name]["cuda_ms_samples"] = samples[name]
    del x
    torch.cuda.empty_cache()
    return {
        "rows": rows,
        "exact_codes": exact_q,
        "exact_scales": exact_scale,
        "variants": results,
        "inplace_minus_baseline_peak_gib":
            results["inplace"]["peak_allocated_gib"] - results["baseline"]["peak_allocated_gib"],
        "inplace_vs_baseline_latency_percent":
            100 * (results["inplace"]["median_cuda_ms"] / results["baseline"]["median_cuda_ms"] - 1),
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("local conda CUDA device unavailable")
    final = {
        "environment": sys.executable,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "workload": "Actual _prepare_activation ConvRot+int8, no trained model, no full E2E",
        "geometries": [],
    }
    for shape in (6814, 59134, 65166):
        row = one(shape)
        final["geometries"].append(row)
        print("LOCAL_P01", shape, "exact", row["exact_codes"], row["exact_scales"],
              "saved_gib", -row["inplace_minus_baseline_peak_gib"], flush=True)
    dest = ROOT / "docs/data/h3_dit_p0_lifetime_20261102_v2/local_gpu_quant.json"
    dest.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("LOCAL_P01_FINAL", dest, flush=True)
