"""Isolated, non-production local checks for proposed DiT P0-A / P0-C.

This script never changes runtime dispatch. SM120 is not a Turing T4: it can
validate Kitchen output-allocation source and numerical preparation, but cannot
certify Turing CUTLASS GEMM or the SM75 TileLang quantizer or E2E latency.
The P0-C padded reference uses a PyTorch fill, not the T4 packing kernel.
"""
from __future__ import annotations

import argparse
import inspect
from importlib.metadata import version
from pathlib import Path
import json
import statistics
import time

import torch
import comfy_kitchen.backends.cuda as kitchen


G = 256
COLUMN_K = 5376
ROW_K = 3584
WORLD = 2


def mib(n):
    return round(n / (1024**2), 3)


def compare(name, expected, actual, *, strict=True):
    raw_dtype = {
        torch.float16: torch.int16, torch.float32: torch.int32,
        torch.float64: torch.int64,
    }.get(expected.dtype, expected.dtype)
    raw_expected = expected.contiguous().view(raw_dtype)
    raw_actual = actual.contiguous().view(raw_dtype)
    same = torch.equal(raw_expected, raw_actual)
    result = {"exact": bool(same), "shape": list(expected.shape)}
    if not same:
        result["mismatched_elements"] = int(torch.count_nonzero(raw_expected != raw_actual).item())
        result["max_absolute_difference"] = float(
            torch.max(torch.abs(expected.float() - actual.float())).item()
        )
    if not same and strict:
        raise AssertionError(f"{name}: failed exact equality: {result}")
    return result


def kitchen_source_audit():
    body = inspect.getsource(kitchen.int8_linear)
    inner = inspect.getsource(kitchen._int8_linear_turing_quantized)
    alloc = "out = torch.empty((m, n), dtype=out_dtype, device=x.device)"
    turing = "turing_out = _int8_linear_turing_quantized("
    internal = "output = torch.empty((m, n), dtype=out_dtype, device=x_qdata.device)"
    assert body.count(alloc) >= 1  # The m==2 branch also has an allocation.
    assert body.rfind(alloc, 0, body.index(turing)) >= 0
    assert internal in inner
    assert "return _finish(turing_out)" in body
    eligible = {
        "qkv": bool(kitchen._prefer_turing_fused_int8(59134, 10752, COLUMN_K)),
        "fc1": bool(kitchen._prefer_turing_fused_int8(59134, 14336, COLUMN_K)),
    }
    assert all(eligible.values())
    capability = torch.cuda.get_device_capability()
    turing_on_this_device = bool(
        kitchen._cuda_device_should_use_turing_kernels(torch.cuda.current_device())
    )
    return {
        "two_separate_fp16_output_allocations_in_turing_branch": True,
        "preconditions_for_t4_geometry": eligible,
        "device_capability": list(capability),
        "actual_local_turing_dispatch": turing_on_this_device,
        "local_t4_gemm_certified": False,
    }


@torch.no_grad()
def p0a_quant(rows, device):
    generator = torch.Generator(device=device).manual_seed(20261102 + rows)
    x_data = torch.randn((rows, COLUMN_K), dtype=torch.float16, device=device, generator=generator)
    h = kitchen._build_hadamard(G, device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    x_ref = x_data.clone()
    q_ref, scale_ref = kitchen.quantize_and_rotate_rowwise(x_ref, h, G)
    x_trial = x_data.clone()
    q_trial, scale_trial = kitchen.quantize_and_rotate_rowwise(x_trial, h, G)
    # The candidate would detach after the rotation/quantization's last reader,
    # never in advance; this is not a live TP2 call-stack ownership check.
    x_trial.set_()
    result = {
        "rows": rows,
        "codes": compare("p0a_codes", q_ref, q_trial),
        "scale": compare("p0a_scale", scale_ref, scale_trial),
        "detached_trial_storage_elements": int(x_trial.numel()),
        "note": "Preparation-only equality; Turing GEMM and caller aliasing untested",
    }
    del q_ref, scale_ref, q_trial, scale_trial, x_ref, x_trial, x_data
    return result


def packed_rotation(x, h, rank):
    rows = x.shape[0]
    full = torch.zeros((rows, ROW_K * WORLD), dtype=x.dtype, device=x.device)
    full[:, rank * ROW_K : (rank + 1) * ROW_K].copy_(x)
    rotated = kitchen._rotate_activation(full, h, G)
    return rotated[:, rank * ROW_K : (rank + 1) * ROW_K].contiguous()


def local_rotation(x, h):
    return kitchen._rotate_activation(x, h, G)


@torch.no_grad()
def p0c_exact(rows, device):
    h = kitchen._build_hadamard(G, device=device, dtype=torch.float32)
    reference_rot = []
    local_rot = []
    local_max_ref = []
    local_max_trial = []
    data = {}
    for rank in range(WORLD):
        generator = torch.Generator(device=device).manual_seed(10000 + rows + rank)
        x = torch.randn((rows, ROW_K), dtype=torch.float32, device=device, generator=generator)
        ref = packed_rotation(x, h, rank)
        trial = local_rotation(x, h)
        data[f"rank{rank}_rotation"] = compare("p0c_rotated", ref, trial, strict=False)
        rmax = ref.abs().amax(-1, keepdim=True)
        tmax = trial.abs().amax(-1, keepdim=True)
        data[f"rank{rank}_local_max"] = compare("p0c_local_max", rmax, tmax, strict=False)
        reference_rot.append(ref)
        local_rot.append(trial)
        local_max_ref.append(rmax)
        local_max_trial.append(tmax)
        del x
    global_ref = torch.maximum(*local_max_ref)
    global_trial = torch.maximum(*local_max_trial)
    data["global_max"] = compare("p0c_global_max", global_ref, global_trial, strict=False)
    scale_ref = torch.clamp(global_ref * (1. / 127.), min=1e-30)
    scale_trial = torch.clamp(global_trial * (1. / 127.), min=1e-30)
    data["scale"] = compare("p0c_scale", scale_ref, scale_trial, strict=False)
    for rank in range(WORLD):
        # Proxy for T4 quantizer; exact FP32 input+maximum parity is the
        # prerequisite, but this is NOT the TileLang __fdividef kernel.
        qr = torch.round(reference_rot[rank] / scale_ref).clamp(-128, 127).to(torch.int8)
        qt = torch.round(local_rot[rank] / scale_trial).clamp(-128, 127).to(torch.int8)
        data[f"rank{rank}_proxy_codes"] = compare("p0c_proxy_codes", qr, qt, strict=False)
        del qr, qt
    del reference_rot, local_rot, local_max_ref, local_max_trial
    return {"rows": rows, "checks": data, "t4_tilelang_quantizer_executed": False}


def rotation_work(x, h, rank, local):
    if local:
        output = kitchen._rotate_activation(x, h, G)
        x.set_()
        return output
    rows = x.shape[0]
    full = torch.empty((rows, ROW_K * WORLD), dtype=torch.float32, device=x.device)
    full.zero_()
    full[:, rank * ROW_K : (rank + 1) * ROW_K].copy_(x)
    x.set_()
    output = kitchen._rotate_activation(full, h, G)
    del full
    return output


@torch.no_grad()
def p0c_rotation_proxy(rows, device, repeats=3):
    h = kitchen._build_hadamard(G, device=device, dtype=torch.float32)
    data = torch.randn((rows, ROW_K), device=device, dtype=torch.float32)
    result = {}
    for local in (False, True):
        mode = "local" if local else "padded"
        samples_ms = []
        peaks = []
        for trial in range(repeats + 1):
            x = data.clone()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            output = rotation_work(x, h, rank=0, local=local)
            stop.record()
            torch.cuda.synchronize()
            if trial:
                samples_ms.append(float(start.elapsed_time(stop)))
                peaks.append(torch.cuda.max_memory_allocated() - baseline)
            del output, x
        result[mode] = {
            "median_ms_sm120_proxy": round(statistics.median(samples_ms), 4),
            "incremental_allocated_peak_mib": mib(statistics.median(peaks)),
        }
    result["measured_proxy_peak_reduction_mib"] = round(
        result["padded"]["incremental_allocated_peak_mib"]
        - result["local"]["incremental_allocated_peak_mib"], 3
    )
    result["caveat"] = "SM120 PyTorch zero/fill and cuBLAS only; not SM75 packing/TileLang or E2E"
    return result


@torch.no_grad()
def p0a_allocator_proxy(rows, device, repeats=3):
    """Pure allocation ledger, not a Kitchen Turing kernel execution."""
    h = kitchen._build_hadamard(G, device=device, dtype=torch.float16)
    data = torch.randn((rows, COLUMN_K), device=device, dtype=torch.float16)
    outcomes = {}
    for label, consume, allocate_dead in (
        ("baseline", False, True),
        ("skip_dead_output_only", False, False),
        ("skip_dead_output_and_detach_input", True, False),
    ):
        peaks = []
        for trial in range(repeats + 1):
            x = data.clone()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            codes, scale = kitchen.quantize_and_rotate_rowwise(x, h, G)
            if consume:
                x.set_()
            dead = (
                torch.empty((rows, 14336), device=device, dtype=torch.float16)
                if allocate_dead else None
            )
            output = torch.empty((rows, 14336), device=device, dtype=torch.float16)
            torch.cuda.synchronize()
            if trial:
                peaks.append(torch.cuda.max_memory_allocated() - before)
            del dead, output, codes, scale, x
        outcomes[label] = {"incremental_allocated_peak_mib": mib(statistics.median(peaks))}
    baseline = outcomes["baseline"]["incremental_allocated_peak_mib"]
    outcomes["skip_dead_reduction_mib"] = round(
        baseline - outcomes["skip_dead_output_only"]["incremental_allocated_peak_mib"], 3
    )
    outcomes["detach_additional_reduction_mib"] = round(
        outcomes["skip_dead_output_only"]["incremental_allocated_peak_mib"]
        - outcomes["skip_dead_output_and_detach_input"]["incremental_allocated_peak_mib"], 3
    )
    outcomes["caveat"] = "SM120 Kitchen activation quant + torch.empty ledger; no T4 CUTLASS or TP caller lifetime"
    return outcomes


def p0a_move_dead_alloc_candidate():
    """Clone only the pinned Kitchen function into an experiment namespace."""
    source = inspect.getsource(kitchen.int8_linear)
    old = (
        "    out = torch.empty((m, n), dtype=out_dtype, device=x.device)\n"
        "    weight_scale = _int8_weight_scale_arg(weight_scale, x.device)\n"
    )
    new = "    weight_scale = _int8_weight_scale_arg(weight_scale, x.device)\n"
    anchor = "    used_cutlass = False\n"
    assert source.count(old) == 1
    assert source.count(anchor) == 1
    source = source.replace(old, new)
    source = source.replace(
        anchor,
        "    out = torch.empty((m, n), dtype=out_dtype, device=x.device)\n" + anchor,
    )
    namespace = vars(kitchen).copy()
    exec(compile(source, "pinned_kitchen_p0a_dead_alloc_only", "exec"), namespace)
    return namespace["int8_linear"]


@torch.no_grad()
def p0a_forward_candidate_local(device, rows=128):
    candidate = p0a_move_dead_alloc_candidate()
    x = torch.randn((rows, COLUMN_K), device=device, dtype=torch.float16) * (1. / 512.)
    checks = {}
    for name, output_width in (("qkv", 10752), ("fc1", 14336)):
        weight = torch.randint(-100, 100, (output_width, COLUMN_K),
                               device=device, dtype=torch.int8)
        weight_scale = torch.full((output_width,), 1. / 256., device=device)
        kwargs = {
            "out_dtype": torch.float16,
            "convrot": True,
            "convrot_groupsize": G,
        }
        baseline = kitchen.int8_linear(x, weight, weight_scale, **kwargs)
        trial = candidate(x, weight, weight_scale, **kwargs)
        checks[name] = compare("p0a_full_column_forward", baseline, trial)
        del baseline, trial, weight, weight_scale
    return {
        "rows": rows, "local_backend": "SM120 Kitchen non-Turing fallback",
        "checks": checks,
        "t4_cutlass_executed": False,
        "note": "One-line allocation movement has unchanged fallback output; "
                "T4 selected kernel still needs physical verification",
    }


@torch.no_grad()
def p0a_alias_ownership_gate(device):
    """Reproduce ColumnParallel.forward's x.view() storage ownership pitfall."""
    owner = torch.empty((64, COLUMN_K), dtype=torch.float16, device=device)
    view = owner.view(-1, COLUMN_K)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    view.set_()
    after_view_detach = torch.cuda.memory_allocated()
    assert before == after_view_detach, (
        "Unexpected allocation change: view.set_ released owner storage"
    )
    owner.set_()
    after_owner_detach = torch.cuda.memory_allocated()
    assert after_owner_detach < before
    return {
        "view_set_releases_owner_storage": False,
        "owner_set_releases_owner_storage": True,
        "bytes_held_after_view_set": after_view_detach - after_owner_detach,
        "caveat": "Actual outer block and Kitchen stack still require owner-safety validation",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="128,6814")
    parser.add_argument("--include-proxy", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA required; no CPU CUDA result substituted")
    device = torch.device("cuda:0")
    results = {
        "environment": {
            "torch": torch.__version__,
            "kitchen": version("comfy-kitchen"),
            "gpu": torch.cuda.get_device_name(device),
            "note": "Local conda comfyui; no physical dual T4 here",
        },
        "p0a_static": kitchen_source_audit(),
        "p0a_alias_ownership": p0a_alias_ownership_gate(device),
        "p0a_forward_candidate": p0a_forward_candidate_local(device),
        "p0a": [],
        "p0a_allocator_proxy": [],
        "p0c_exact": [],
        "p0c_rotation_proxy": [],
    }
    for rows in [int(x) for x in args.rows.split(",")]:
        began = time.monotonic()
        results["p0a"].append(p0a_quant(rows, device))
        print("P0-A preparation exact", rows, flush=True)
        results["p0c_exact"].append(p0c_exact(rows, device))
        print("P0-C rotation / maxima / proxy quant checked", rows, flush=True)
        if args.include_proxy:
            results["p0a_allocator_proxy"].append({
                "rows": rows, "metrics": p0a_allocator_proxy(rows, device)
            })
            results["p0c_rotation_proxy"].append({
                "rows": rows, "metrics": p0c_rotation_proxy(rows, device)
            })
        print("rows elapsed_s", rows, round(time.monotonic() - began, 2), flush=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("P0A_P0C_LOCAL_JSON_BEGIN", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("P0A_P0C_LOCAL_JSON_END", flush=True)


if __name__ == "__main__":
    main()
