"""Experiment-only P0-A1/P0-C hooks for physical SM75 TP2 A/B.

Never imported by the production runtime. All dispatch and source gates fail
closed; historical P0-1 is not resurrected. Do not monkeypatch wheels on disk.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import inspect
from importlib.metadata import version
from unittest.mock import patch

import torch
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as kitchen


COLUMN_K = 5376
ROW_WIDTH = 3584
GROUP = 256


def build_p0a1(record):
    """Move exactly one unused pre-Turing allocation behind the early return."""
    if version("comfy-kitchen") != "0.2.37":
        raise RuntimeError("P0-A1 requires the pinned Kitchen 0.2.37 source")
    source = inspect.getsource(kitchen.int8_linear)
    old = (
        "    out = torch.empty((m, n), dtype=out_dtype, device=x.device)\n"
        "    weight_scale = _int8_weight_scale_arg(weight_scale, x.device)\n"
    )
    anchor = "    used_cutlass = False\n"
    if source.count(old) != 1 or source.count(anchor) != 1:
        raise RuntimeError("Kitchen source changed; reject P0-A1 experiment")
    source = source.replace(
        old,
        "    weight_scale = _int8_weight_scale_arg(weight_scale, x.device)\n",
    ).replace(
        anchor,
        "    out = torch.empty((m, n), dtype=out_dtype, device=x.device)\n" + anchor,
    )
    namespace = vars(kitchen).copy()
    original = kitchen._int8_linear_turing_quantized

    def counted_turing(*args, **kwargs):
        record["turing_called"] = record.get("turing_called", 0) + 1
        out = original(*args, **kwargs)
        if out is not None:
            record["turing_success"] = record.get("turing_success", 0) + 1
        return out

    namespace["_int8_linear_turing_quantized"] = counted_turing
    exec(compile(source, "p0a1_pinned_kitchen_0_2_37", "exec"), namespace)
    return namespace["int8_linear"]


def _t4_column_dispatch(x, weight, scale, kwargs):
    if (not x.is_cuda or torch.cuda.get_device_capability(x.device) != (7, 5)
            or weight.shape not in ((10752, COLUMN_K), (14336, COLUMN_K))):
        raise RuntimeError("P0-A1 column target must be a physical aligned SM75 projection")
    call = dict(
        x=x, weight=weight, weight_scale=scale, bias=None,
        out_dtype=kwargs.get("out_dtype", torch.float16),
        convrot=kwargs.get("convrot", True),
        convrot_groupsize=kwargs.get("convrot_groupsize", GROUP),
        input_act=None, input_act_weight=None, input_act_eps=0.0,
        residual=None, residual_scale=None,
    )
    implementation = ck.registry.get_implementation("int8_linear", kwargs=call)
    if implementation is not kitchen.int8_linear:
        raise RuntimeError("Kitchen CUDA column registry was not selected")
    if not kitchen._cuda_device_should_use_turing_kernels(x.get_device()):
        raise RuntimeError("Turing hardware selection gate not taken")
    if not kitchen._prefer_turing_fused_int8(x.numel() // COLUMN_K, weight.shape[0], COLUMN_K):
        raise RuntimeError("Kitchen Turing geometry gate not taken")


def local_o_proj_quantize(
    x, comm, world_size=2, groupsize=256, input_act=None, variant="packed",
    *, consume_input=False, counter=None,
):
    """P0-C experiment: local ConvRot and global maxima, then same SM75 quantizer.

    Mirrors the qualified P0-2 consumption contract after local rotation's
    last read. Fallbacks and other quantizers must not be intercepted.
    """
    from patches import tilelang_row_quant as row
    if (not x.is_cuda or torch.cuda.get_device_capability(x.device) != (7, 5)
            or x.dtype != torch.float32 or x.ndim != 2
            or x.shape[1] != ROW_WIDTH or world_size != 2 or groupsize != GROUP
            or input_act is not None or variant != "packed" or not consume_input
            or type(x) is not torch.Tensor or x._base is not None or not x.is_contiguous()):
        raise RuntimeError("P0-C experimental hook reached unsupported O-Proj geometry/owner")
    rows = x.shape[0]
    if rows not in (128, 6814, 59134, 65166):
        raise RuntimeError("P0-C must target a frozen study row count")
    flat = kitchen._apply_input_act(x.reshape(-1, ROW_WIDTH).contiguous(), None, None, 0.)
    h = kitchen._build_hadamard(GROUP, device=x.device, dtype=x.dtype)
    rotated = kitchen._rotate_activation(flat, h, GROUP)
    # Kitchen rotation returned; the original FP32 Tensor is its own owner.
    x.set_()
    del flat
    maximum = row.maximum_kernel(rows, ROW_WIDTH, ROW_WIDTH, 0)(rotated)
    comm.all_reduce(maximum, op=comm.ReduceOp.MAX)
    q, scale = row.quantize(rotated, maximum, ROW_WIDTH, 0)
    del rotated, maximum
    if counter is not None:
        counter["o_proj_quant_called"] = counter.get("o_proj_quant_called", 0) + 1
    return q, scale


def bytes_compare(a, b):
    """Raw bytes, including FP signed zero, and finite-check diagnostics."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"bit_exact": False, "shape_mismatch": True}
    if a.dtype == torch.float16:
        raw = torch.int16
    elif a.dtype == torch.float32:
        raw = torch.int32
    elif a.dtype == torch.float64:
        raw = torch.int64
    else:
        raw = a.dtype
    mismatched = int((a.contiguous().view(raw) != b.contiguous().view(raw)).sum().item())
    max_abs = float((a.float() - b.float()).abs().max().item())
    return {
        "bit_exact": mismatched == 0,
        "mismatched": mismatched,
        "max_abs": max_abs,
        "finite": bool(torch.isfinite(a).all().item()) and bool(torch.isfinite(b).all().item()),
        "shape": list(a.shape),
    }


@torch.no_grad()
def probe_p0c(rows, device, comm):
    """Actual SM75 TileLang packed/local preparation, one synthetic rank per GPU."""
    from patches import tilelang_row_quant as row
    rank = comm.get_rank()
    generator = torch.Generator(device=device).manual_seed(10000 + rows + rank)
    x = torch.randn((rows, ROW_WIDTH), device=device, dtype=torch.float32, generator=generator)
    h = kitchen._build_hadamard(GROUP, device=device, dtype=torch.float32)
    padded = row.packing_kernel(rows, ROW_WIDTH, 2, rank)(x)
    padded_rotated = kitchen._rotate_activation(padded, h, GROUP)
    sliced = padded_rotated[:, rank * ROW_WIDTH:(rank + 1) * ROW_WIDTH].contiguous()
    local_rotated = kitchen._rotate_activation(x, h, GROUP)
    rotation = bytes_compare(sliced, local_rotated)
    del padded, padded_rotated, sliced, local_rotated
    base_q, base_scale = row.global_row_quantize(
        x, comm, 2, GROUP, None, "packed", consume_input=False,
    )
    # This probe needs its input for both arms; the trial production hook
    # consumes its separate copied owner.
    candidate = x.clone()
    from unittest.mock import patch
    trial_q, trial_scale = local_o_proj_quantize(
        candidate, comm, 2, GROUP, None, "packed", consume_input=True,
    )
    q_cmp = bytes_compare(base_q, trial_q)
    scale_cmp = bytes_compare(base_scale, trial_scale)
    return {"rows": rows, "rank": rank, "rotation": rotation, "codes": q_cmp,
            "scale": scale_cmp, "all_bit_exact": all(v["bit_exact"] for v in (rotation, q_cmp, scale_cmp))}


@contextmanager
def active_variant(name, record, candidate_column=None):
    """Intercept only target column/O-Proj sites for a single frozen block call."""
    from patches import tilelang_row_quant as row
    if name not in ("baseline", "p0_a1", "p0_c", "p0_a1_p0_c"):
        raise ValueError(name)
    record.setdefault("variant", name)
    original_public = ck.int8_linear
    original_row = row.global_row_quantize
    if "p0_a1" in name and candidate_column is None:
        raise RuntimeError("P0-A1 candidate must be constructed before timing")
    with ExitStack() as stack:
        def column(*args, **kwargs):
            x, weight, scale = args[:3]
            if weight.ndim == 2 and weight.shape in (
                (10752, COLUMN_K), (14336, COLUMN_K)
            ):
                _t4_column_dispatch(x, weight, scale, kwargs)
                tag = "fc1" if weight.shape[0] == 14336 else "qkv"
                record[tag + "_called"] = record.get(tag + "_called", 0) + 1
                if candidate_column is not None:
                    return candidate_column(*args, **kwargs)
            return original_public(*args, **kwargs)

        def row_quant(*args, **kwargs):
            x = args[0]
            if (x.ndim == 2 and x.shape[-1] == ROW_WIDTH and
                    kwargs.get("consume_input", False)):
                if "p0_c" in name:
                    return local_o_proj_quantize(
                        *args, **kwargs, counter=record,
                    )
                record["baseline_o_proj_quant_called"] = (
                    record.get("baseline_o_proj_quant_called", 0) + 1
                )
            return original_row(*args, **kwargs)

        stack.enter_context(patch.object(ck, "int8_linear", column))
        stack.enter_context(patch.object(row, "global_row_quantize", row_quant))
        yield


def assert_hits(record, name):
    for key in ("qkv_called", "fc1_called"):
        if not record.get(key):
            raise RuntimeError(f"Missing expected {key} for {name}")
    if name in ("p0_a1", "p0_a1_p0_c"):
        if record.get("turing_success", 0) < 2:
            raise RuntimeError("P0-A1 did not execute the real Turing CUTLASS branch")
    if name in ("p0_c", "p0_a1_p0_c"):
        if not record.get("o_proj_quant_called"):
            raise RuntimeError("P0-C candidate did not hit production out_proj")
    elif not record.get("baseline_o_proj_quant_called"):
        raise RuntimeError("Baseline P0-2 packed path not executed")
