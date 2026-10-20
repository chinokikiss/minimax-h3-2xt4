"""Auditable reruns. Timings are measurements; failed correctness never earns a speedup.

Run each suite in a fresh process. This intentionally tests the existing TP/SP
proposals against checkpoint-loaded native blocks before accepting any ranking.
The TE/DiT tests are block tests, not estimates of full generation time.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("COMFY_ROOT", str(ROOT / "ComfyUI")))
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open

# ComfyUI d49e888 adds fused activation/residual keywords that kitchen 0.2.33
# does not implement. Preserve the mathematics with explicit eager operations;
# report this as an unfused compatibility baseline in environment.json.
def install_kitchen_compatibility():
    import inspect
    import comfy_kitchen as ck
    if "input_act_weight" in inspect.signature(ck.int8_linear).parameters:
        return False
    original = ck.int8_linear
    def compatible(x, weight, weight_scale, bias=None, out_dtype=None,
                   convrot=False, convrot_groupsize=256, input_act=None,
                   input_act_weight=None, input_act_eps=0.0,
                   residual=None, residual_scale=None):
        from comfy.ops import _eager_input_act
        x = _eager_input_act(x, input_act, input_act_weight, input_act_eps)
        out = original(x, weight, weight_scale, bias, out_dtype,
                       convrot=convrot, convrot_groupsize=convrot_groupsize)
        return out if residual is None else torch.addcmul(residual, out, residual_scale)
    ck.int8_linear = compatible
    return True

KITCHEN_EAGER_COMPAT = install_kitchen_compatibility()

MODEL_REVISION = "e5eb578a89295337b8ff433a035929ce0279e0b6"
CHECKPOINTS = {
    "te": "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "dit": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "video": "vae/minimax_h3_video_vae_int8_convrot.safetensors",
    "audio": "vae/minimax_h3_audio_vae_fp32.safetensors",
}
VIDEO_WORKLOADS = {
    "512x512_1F": (1, 24, 1, 32, 32),
    "512x512_2F": (1, 24, 2, 32, 32),
    "768x768_1F": (1, 24, 1, 48, 48),
    "768x1344_1F": (1, 24, 1, 48, 84),
    "768x1344_2F": (1, 24, 2, 48, 84),
}
# Predeclared engineering gates; bit equality is separately reported.
GATES = {
    "block": {"rel_l2_max": 0.01, "cosine_min": 0.999},
    "video": {"rel_l2_max": 0.01, "cosine_min": 0.999, "max_abs_max": 0.05},
    "audio_channel": {"rel_l2_max": 1e-5, "cosine_min": 0.999999},
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def digest_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def metrics(test, ref):
    if test.shape != ref.shape:
        return {"valid": False, "reason": "shape mismatch", "test_shape": list(test.shape), "ref_shape": list(ref.shape)}
    # Chunked FP64 accumulation avoids several simultaneous full RGB copies.
    a, b = test.detach().cpu().reshape(-1), ref.detach().cpu().reshape(-1)
    aa = bb = ab = dd = max_abs = abs_sum = 0.0
    for start in range(0, a.numel(), 1_000_000):
        x, y = a[start:start + 1_000_000].double(), b[start:start + 1_000_000].double()
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            return {"valid": False, "reason": "nonfinite output"}
        delta = x - y
        aa += x.square().sum().item()
        bb += y.square().sum().item()
        ab += (x * y).sum().item()
        dd += delta.square().sum().item()
        abs_sum += delta.abs().sum().item()
        max_abs = max(max_abs, delta.abs().max().item())
    if aa == 0 or bb == 0:
        return {"valid": False, "reason": "degenerate zero output", "is_bit_exact": torch.equal(a, b)}
    return {"valid": True, "rel_l2": math.sqrt(dd / bb),
            "cosine": max(-1.0, min(1.0, ab / math.sqrt(aa * bb))),
            "max_abs": max_abs, "mean_abs": abs_sum / a.numel(),
            "is_bit_exact": torch.equal(a, b)}


def passes(value, gate):
    return bool(value.get("valid") and value["rel_l2"] <= gate["rel_l2_max"]
                and value["cosine"] >= gate["cosine_min"]
                and value.get("max_abs", 0) <= gate.get("max_abs_max", float("inf")))


def summary(samples):
    samples = np.array(samples, dtype=np.float64)
    rng = np.random.default_rng(20261007)
    boot = np.median(rng.choice(samples, (2000, len(samples)), replace=True), axis=1)
    return {"n": len(samples), "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)), "min_ms": float(samples.min()),
            "max_ms": float(samples.max()), "median_bootstrap_95ci_ms": np.percentile(boot, [2.5, 97.5]).tolist()}


def cleanup():
    gc.collect()
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    torch.cuda.empty_cache()


def measure(call, devices, warmups, iterations, distributed=False, reference=None, gate=None):
    if reference is not None and gate is None:
        raise ValueError("An equivalence reference requires an explicit numerical gate")
    with torch.inference_mode():
        # Validate one forward BEFORE any warmup or timing sweep. All ranks
        # agree on the decision before raising, so a mismatch cannot strand a
        # peer in the next collective. Baselines still get a validity check.
        probe = call()
        preflight = metrics(probe, reference if reference is not None else probe)
        valid = passes(preflight, gate) if reference is not None else preflight["valid"]
        if distributed:
            agreed = torch.tensor(int(valid), device=devices[0], dtype=torch.int32)
            dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
            valid = bool(agreed.item())
        del probe
        if not valid:
            raise RuntimeError("Pre-timing semantic validation failed: " + json.dumps(preflight))
        for _ in range(warmups):
            out = call()
            del out
        for dev in devices:
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        samples = []
        allocated_observations, reserved_observations = [], []
        for _ in range(iterations):
            if distributed:
                dist.barrier()
            for dev in devices:
                torch.cuda.synchronize(dev)
            start = time.perf_counter()
            out = call()
            for dev in devices:
                torch.cuda.synchronize(dev)
            elapsed = (time.perf_counter() - start) * 1000
            if distributed:
                elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=devices[0])
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                elapsed = elapsed_tensor.item()
            samples.append(elapsed)
            allocated_observations.append([torch.cuda.max_memory_allocated(d) / 1024**3 for d in devices])
            reserved_observations.append([torch.cuda.max_memory_reserved(d) / 1024**3 for d in devices])
            del out
        out = call().detach().cpu()
        final_validation = metrics(out, reference if reference is not None else out)
        valid = passes(final_validation, gate) if reference is not None else final_validation["valid"]
        if distributed:
            agreed = torch.tensor(int(valid), device=devices[0], dtype=torch.int32)
            dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
            valid = bool(agreed.item())
        if not valid:
            raise RuntimeError("Post-timing semantic validation failed; discard timings: " + json.dumps(final_validation))
        allocated_observations.append([torch.cuda.max_memory_allocated(d) / 1024**3 for d in devices])
        reserved_observations.append([torch.cuda.max_memory_reserved(d) / 1024**3 for d in devices])
        peaks = np.max(allocated_observations,axis=0).tolist()
        reserved = np.max(reserved_observations,axis=0).tolist()
    return {"pre_timing_validation": {"kind": "reference equivalence" if reference is not None else "self validity only",
                                      "accuracy": preflight}, "post_timing_accuracy": final_validation,
            "samples_ms": samples, **summary(samples), "peak_allocated_observations_gib":allocated_observations,
            "peak_reserved_observations_gib":reserved_observations,"peak_allocated_gib": peaks,
            "peak_reserved_gib": reserved}, out


def checkpoint(kind):
    from huggingface_hub import hf_hub_download
    return hf_hub_download("Comfy-Org/MiniMax-H3", CHECKPOINTS[kind], revision=MODEL_REVISION)


def state(path, prefix=""):
    if os.name == 'nt' and Path(path).stat().st_size > 8 * 1024 ** 3:
        # Read only the selected block. Mapping a 20–27 GB checkpoint can
        # exhaust Windows commit even when the experiment needs <500 MB.
        import struct
        dtypes = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
                  'I8': torch.int8, 'U8': torch.uint8, 'I64': torch.int64,
                  'I32': torch.int32, 'BOOL': torch.bool}
        selected = {}
        with open(path, 'rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(length))
            for key, info in header.items():
                if key == '__metadata__' or not key.startswith(prefix):
                    continue
                start, end = info['data_offsets']
                stream.seek(8 + length + start)
                data = bytearray(stream.read(end - start))
                selected[key[len(prefix):]] = torch.frombuffer(data, dtype=dtypes[info['dtype']]).reshape(info['shape'])
        return selected
    with safe_open(path, framework="pt", device="cpu") as f:
        return {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}


def load_native_block(kind, path, index=0, device="cuda:0"):
    import comfy.ops
    dtype = torch.float32 if kind == "dit" else torch.float16
    ops = comfy.ops.mixed_precision_ops({"mixed_ops": True}, dtype)
    if kind == "te":
        from comfy.text_encoders.llama import Qwen3VL_32BConfig, TransformerBlock
        block = TransformerBlock(Qwen3VL_32BConfig(), index=index, device="cpu", dtype=torch.float16, ops=ops)
        sd = state(path, f"model.layers.{index}.")
    else:
        from comfy.ldm.minimax.model import DiTBlock
        from comfy.supported_models import MiniMaxH3
        if dtype not in MiniMaxH3.supported_inference_dtypes:
            raise RuntimeError("The DiT baseline must use an upstream-supported inference dtype")
        block = DiTBlock(5376, 56, 128, 14336, 8, 1e-5, 1e-5,
                         apply_silu=False, adaln_dtype=torch.float32, device="cpu", dtype=dtype, operations=ops)
        sd = state(path, f"blocks.{index}.")
    block.load_state_dict(sd, strict=True)
    return block.to(device).eval().requires_grad_(False)


def load_video_native(path, device="cpu"):
    import comfy.ops
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    sd = state(path)
    depth = sum(k.startswith("decoder.transformer_blocks.") and k.endswith(".scale1") for k in sd)
    if depth != 36:
        raise RuntimeError(f"Expected the full 36-block checkpoint, got {depth}")
    ops = comfy.ops.mixed_precision_ops({"mixed_ops": True}, torch.float16)
    model = MiniMaxH3VideoVAE(operations=ops, num_layers=depth)
    model.load_state_dict(sd, strict=True)
    model.tiling = False
    return model.to(device=device, dtype=torch.float16).eval().requires_grad_(False), sd


def native_call(kind, block, x, t_emb=None, segments=None, rope=None):
    # Native CUDA extensions may require the current device to match their
    # tensors and stream; an activation transfer does not switch that context.
    with torch.cuda.device(x.device):
        return _native_call(kind, block, x, t_emb, segments, rope)


def _native_call(kind, block, x, t_emb=None, segments=None, rope=None):
    if kind == "dit":
        return block(x.clone(), t_emb, segments, rope)
    from comfy.text_encoders.llama import Qwen3VL_32BConfig, precompute_freqs_cis
    from comfy.ldm.modules.attention import optimized_attention_for_device
    config = Qwen3VL_32BConfig()
    positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    freq = precompute_freqs_cis(128, positions, config.rope_theta,
                              rope_dims=config.rope_dims, device=x.device, interleaved_mrope=True)
    mask = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min / 4,
                      dtype=x.dtype, device=x.device).triu(1)
    return block(x.clone(), attention_mask=mask, freqs_cis=freq,
                 optimized_attention=optimized_attention_for_device(x.device, mask=True, small_input=True))[0]


def load_proposal(kind, strategy, path, rank):
    if kind == "te":
        from patches.tp_qwen3vl import TPTransformerBlockINT8
        from patches.sp_int8_qwen3vl import SPTransformerBlockINT8
        cls = TPTransformerBlockINT8 if strategy == "tp" else SPTransformerBlockINT8
        block = cls(rank=rank, world_size=2, device=torch.device(f"cuda:{rank}"))
        sd = state(path, "model.layers.0.")
        norms = {"input_layernorm": "input_layernorm", "post_attention_layernorm": "post_attention_layernorm",
                 "q_norm": "self_attn.q_norm", "k_norm": "self_attn.k_norm"}
        columns = {n: f"self_attn.{n}" for n in ["q_proj", "k_proj", "v_proj"]}
        columns.update({n: f"mlp.{n}" for n in ["gate_proj", "up_proj"]})
        rows = {"o_proj": "self_attn.o_proj", "down_proj": "mlp.down_proj"}
    else:
        from patches.tp_dit import TPDiTBlockINT8
        from patches.sp_int8_dit import SPDiTBlockINT8
        cls = TPDiTBlockINT8 if strategy == "tp" else SPDiTBlockINT8
        block = cls(rank=rank, world_size=2)
        sd = state(path, "blocks.0.")
        norms = {"norm1": "norm1", "norm2": "norm2", "q_norm": "attn.q_norm", "k_norm": "attn.k_norm"}
        columns = {"qkv_proj": "attn.qkv_proj", "fc1": "mlp.fc1"}
        rows = {"out_proj": "attn.out_proj", "fc2": "mlp.fc2"}
        block.adaln_proj.load_state_dict({"weight": sd["adaln_proj.linear.weight"], "bias": sd["adaln_proj.linear.bias"]})
    for name, key in norms.items():
        getattr(block, name).load_state_dict({"weight": sd[key + ".weight"]}, strict=True)
    for name, key in {**columns, **rows}.items():
        getattr(block, name).load_shard_from_full(sd[key + ".weight"], sd[key + ".weight_scale"])
    return block.to(f"cuda:{rank}").eval().requires_grad_(False)


def _worker(rank, mode, kind, path, inputs_dir, output_dir, port, warmups, iterations, raw_video=False, gate_override=None):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    device = torch.device(f"cuda:{rank}")
    result = {}
    try:
        if kind == "video":
            from patches.tp_video_vae import patch_video_vae_tp, patch_video_vae_sp
            model, sd = load_video_native(path, device=device)
            patcher = patch_video_vae_tp if mode == "tp" else patch_video_vae_sp
            model = patcher(model, sd, rank, 2, device)
            del sd
        else:
            model = load_proposal(kind, mode, path, rank)
        for input_file in sorted(Path(inputs_dir).glob("*.pt")):
            data = torch.load(input_file, weights_only=True)
            name = input_file.stem
            x = data["input"].to(device=device, dtype=torch.float32 if kind == "dit" else data["input"].dtype)
            if kind == "video":
                call = (lambda: model._decode_pixels(x)) if raw_video else (lambda: model.decode(x))
            else:
                te = data.get("t_emb")
                te = te.to(device=device, dtype=model.adaln_proj.weight.dtype) if te is not None else None
                segments = data.get("segments")
                rope = data.get("rope")
                rope = rope.to(device) if rope is not None else None
                if kind == "te":
                    from comfy.text_encoders.llama import Qwen3VL_32BConfig, precompute_freqs_cis
                    config = Qwen3VL_32BConfig()
                    freq = precompute_freqs_cis(config.head_dim, torch.arange(x.shape[1], device=device)[None],
                        config.rope_theta, rope_dims=config.rope_dims, device=device, interleaved_mrope=True)
                    mask = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min / 4,
                        dtype=x.dtype, device=device).triu(1)
                if mode == "sp":
                    sequence_dim = 1 if kind == "te" else 0
                    x_local = torch.tensor_split(x, 2, dim=sequence_dim)[rank].contiguous()
                    def call():
                        from patches.parallel_int8 import gather
                        y = model(x_local.clone(), mask, freq) if kind == "te" else model(x_local.clone(), te, segments, rope)
                        return gather(y, dist, sequence_dim)[0]
                else:
                    call = (lambda: model(x.clone(), mask, freq)) if kind == "te" else (lambda: model(x.clone(), te, segments, rope))
            reference_file = Path(output_dir) / f"native_{name}_output.pt"
            if not reference_file.exists():
                raise RuntimeError("Proposal timing requires the native reference first")
            reference = torch.load(reference_file, map_location="cpu", weights_only=True)
            measured, out = measure(call, [device], warmups, iterations, distributed=True,
                                    reference=reference, gate=gate_override or GATES["video" if kind == "video" else "block"])
            peaks = torch.tensor([measured["peak_allocated_gib"][0], measured["peak_reserved_gib"][0]], device=device)
            all_peaks = [torch.empty_like(peaks) for _ in range(2)]
            dist.all_gather(all_peaks, peaks)
            if rank == 0:
                measured["peak_allocated_gib"] = [p[0].item() for p in all_peaks]
                measured["peak_reserved_gib"] = [p[1].item() for p in all_peaks]
                measured["input_sha256"] = digest_tensor(data["input"])
                torch.save(out, Path(output_dir) / f"{mode}_{name}_output.pt")
                result[name] = measured
                write_json(Path(output_dir) / f"{mode}_raw.json", result)
                print(f"[{kind}/{mode}] {name}: {measured['median_ms']:.2f} ms", flush=True)
            del out, x, data
        del model
    finally:
        dist.destroy_process_group()


def distributed_run(mode, kind, path, inputs, output, args):
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(mode, kind, path, str(inputs), str(output), port, args.warmups, args.iterations,
                           getattr(args, "raw_video", False), getattr(args, "equivalence_gate", None)), nprocs=2, join=True)
    return json.loads((output / f"{mode}_raw.json").read_text())


def make_inputs(kind, output, path=None):
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    lengths = [64, 128, 256, 512, 1024, 2048] if kind == "te" else [1024, 2048, 4096, 8192]
    specs = VIDEO_WORKLOADS if kind == "video" else {f"S{s}": (1, s, 5120) if kind == "te" else (s, 5376) for s in lengths}
    curve = state(path, "adaln_t_table")[""] if kind == "dit" and path else None
    for i, (name, shape) in enumerate(specs.items()):
        generator = torch.Generator(device="cpu").manual_seed(42 + i)
        data = {"input": torch.randn(shape, generator=generator, dtype=torch.float32).half(), "seed": 42 + i}
        if kind == "dit":
            if curve is None:
                raise ValueError("DiT retests require the checkpoint's real timestep curve")
            curve_index = [128, 384, 640, 896][i]
            data["t_emb"] = curve[curve_index:curve_index + 1].float().clone()
            data["conditioning_timestep"] = curve_index / (curve.shape[0] - 1)
            s = shape[0]
            data["segments"] = [(0, s // 4, 0), (s // 4, s // 2, 1), (s // 2, s, 2)]
            # Nontrivial 3D rotary tables exercise the real positional path.
            angle = torch.linspace(0, 8, s).view(1, s, 1, 1) * torch.linspace(.01, 1, 48).view(1, 1, 1, 48)
            c, si = angle.cos(), angle.sin()
            data["rope"] = torch.stack([c, -si, si, c], -1).reshape(1, s, 1, 48, 2, 2).half()
        torch.save(data, inputs / f"{name}.pt")
    write_json(output / "input_manifest.json", {name: {"shape": list(shape), "seed": 42 + i,
                "sha256": digest_tensor(torch.load(inputs / f"{name}.pt", weights_only=True)["input"])}
                for i, (name, shape) in enumerate(specs.items())})
    return inputs


def run_blocks(kind, path, output, args):
    inputs = make_inputs(kind, output, path)
    references, baseline = {}, {}
    block = load_native_block(kind, path)
    for input_file in sorted(inputs.glob("*.pt")):
        data = torch.load(input_file, weights_only=True)
        x = data["input"].cuda().to(dtype=torch.float32 if kind == "dit" else torch.float16)
        te = data.get("t_emb")
        te = te.cuda() if te is not None else None
        rope = data.get("rope")
        rope = rope.to(device=x.device,dtype=x.dtype) if rope is not None else None
        measured, ref = measure(lambda: native_call(kind, block, x, te, data.get("segments"), rope),
                                [0], args.warmups, args.iterations)
        baseline[input_file.stem] = measured
        references[input_file.stem] = ref
        torch.save(ref, output / f"native_{input_file.stem}_output.pt")
        write_json(output / "native_raw.json", baseline)
        print(f"[{kind}/native block] {input_file.stem}: {measured['median_ms']:.2f} ms", flush=True)
        del x, data, ref, te, rope
    del block
    cleanup()
    results = {"scope": "one native checkpoint block versus the existing TP/SP proposals; no full-model extrapolation",
               "gates": GATES["block"], "native": baseline, "proposals": {}}
    for mode in random.Random(args.round).sample(["tp", "sp"], 2):
        try:
            values = distributed_run(mode, kind, path, inputs, output, args)
            for name, value in values.items():
                pred = torch.load(output / f"{mode}_{name}_output.pt", weights_only=True)
                value["accuracy"] = metrics(pred, references[name])
                value["correctness_passed"] = passes(value["accuracy"], GATES["block"])
                value["accepted_speedup"] = baseline[name]["median_ms"] / value["median_ms"] if value["correctness_passed"] else None
                print("BLOCK_ACCURACY",kind,mode,name,json.dumps(value["accuracy"]),"passed",value["correctness_passed"],flush=True)
            results["proposals"][mode] = values
        except Exception:
            results["proposals"][mode] = {"status": "failed", "traceback": traceback.format_exc()}
        write_json(output / "results.json", results)
    return results


def run_video(path, output, args):
    inputs = make_inputs("video", output)
    results = {"scope": "real checkpoint full 36-block decode, same saved latents for every strategy",
               "gates": GATES["video"], "strategies": {}}
    references = {}
    order = ["native"] + random.Random(100 + args.round).sample(["tp", "sp"], 2)
    results["strategy_order"] = order
    for mode in order:
        try:
            if mode == "native":
                model, sd = load_video_native(path, device=torch.device("cuda:0"))
                del sd
                values = {}
                for input_file in sorted(inputs.glob("*.pt")):
                    data = torch.load(input_file, weights_only=True)
                    latent = data["input"].cuda()
                    measured, ref = measure(lambda: model.decode(latent), [0], args.warmups, args.iterations)
                    measured["input_sha256"] = digest_tensor(data["input"])
                    values[input_file.stem] = measured
                    references[input_file.stem] = ref
                    torch.save(ref, output / f"native_{input_file.stem}_output.pt")
                    write_json(output / "native_raw.json", values)
                    print(f"[video/native] {input_file.stem}: {measured['median_ms']:.2f} ms", flush=True)
                    del latent, data, ref
                del model
                cleanup()
            else:
                values = distributed_run(mode, "video", path, inputs, output, args)
            results["strategies"][mode] = values
        except Exception:
            results["strategies"][mode] = {"status": "failed", "traceback": traceback.format_exc()}
        write_json(output / "results.json", results)
    if "status" not in results["strategies"].get("native", {"status": "missing"}):
        for mode in ["tp", "sp"]:
            for name, value in results["strategies"].get(mode, {}).items():
                if not isinstance(value, dict) or "median_ms" not in value:
                    continue
                pred = torch.load(output / f"{mode}_{name}_output.pt", weights_only=True)
                value["accuracy"] = metrics(pred, references[name])
                value["correctness_passed"] = passes(value["accuracy"], GATES["video"])
                value["accepted_speedup"] = results["strategies"]["native"][name]["median_ms"] / value["median_ms"] if value["correctness_passed"] else None
    write_json(output / "results.json", results)
    return results


def run_tiling(kind, path, output, args):
    if kind == "audio":
        from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
        from patches.multi_gpu_audio_vae import MultiGPUAudioVAEDecoder
        model = MiniMaxH3AudioVAE()
        model.load_state_dict(state(path), strict=True)
        decoder = MultiGPUAudioVAEDecoder(base_vae=model, dev1="cuda:0", dtype=torch.float32)
        replica = MiniMaxH3AudioVAE()
        replica.load_state_dict(state(path), strict=True)
        decoder.vae_dev1 = replica.to(device="cuda:1", dtype=torch.float32).eval()
        workloads = {f"{seconds}s": (1, 32, 2, seconds * 40) for seconds in [5, 10, 20]}
        dtype = torch.float32
    else:
        from patches.multi_gpu_video_vae import MultiGPUVideoVAEDecoder
        model, sd = load_video_native(path)
        del sd
        decoder = MultiGPUVideoVAEDecoder(base_vae=model, dev1="cuda:0", dtype=torch.float16)
        replica, replica_sd = load_video_native(path, device="cuda:1")
        del replica_sd
        decoder.vae_dev1 = replica
        workloads = {k: v for k, v in VIDEO_WORKLOADS.items() if k.endswith("2F") or k == "768x768_1F"}
        dtype = torch.float16
    # QuantizedTensor does not support deepcopy; initialize both replicas from
    # the same checkpoint instead of the wrapper's deepcopy construction path.
    decoder.dev1 = torch.device("cuda:1")
    decoder.is_multi_gpu = True
    decoder.stream0 = torch.cuda.Stream(device=decoder.dev0)
    decoder.stream1 = torch.cuda.Stream(device=decoder.dev1)
    results = {"scope": "checkpoint-loaded decoder; identical latent input; tiling is approximate",
               "replication": "both models loaded separately from identical checkpoint; no QuantizedTensor deepcopy",
               "workloads": {}}
    for index, (name, shape) in enumerate(workloads.items()):
        x = torch.randn(shape, generator=torch.Generator().manual_seed(300 + index), dtype=torch.float32).to(dtype)
        torch.save(x, output / f"input_{name}.pt")
        with torch.inference_mode():
            ref, _ = decoder.decode_normal(x)
        ref = ref.cpu()
        torch.save(ref, output / f"native_{name}_output.pt")
        if kind == "audio":
            modes = {"native": lambda: decoder.decode_normal(x),
                     "tiled_1gpu": lambda: decoder.decode_tiled_single(x),
                     "channel_2gpu": lambda: decoder.decode_channel_parallel(x),
                     "tiled_2gpu": lambda: decoder.decode_tiled_multi(x)}
        else:
            modes = {"native": lambda: decoder.decode_normal(x)}
            for tile in [256, 384]:
                modes[f"tiled_1gpu_{tile}"] = lambda tile=tile: decoder.decode_tiled_single(x, tile_size=tile, tile_overlap_min=64)
                modes[f"tiled_2gpu_{tile}"] = lambda tile=tile: decoder.decode_tiled_multi(x, tile_size=tile, tile_overlap_min=64)
        workload_result = {"input_sha256": digest_tensor(x), "shape": list(shape), "strategies": {}}
        for mode in random.Random(args.round + index).sample(list(modes), len(modes)):
            try:
                matching_reference, gate = None, None
                if mode == "channel_2gpu":
                    matching_reference, gate = ref, GATES["audio_channel"]
                elif "tiled_2gpu" in mode:
                    with torch.inference_mode():
                        matching_reference = modes[mode.replace("2gpu", "1gpu")]()[0].detach().cpu()
                    gate = GATES["audio_channel"]
                measured, out = measure(lambda: modes[mode]()[0], [0, 1], args.warmups, args.iterations,
                                        reference=matching_reference, gate=gate)
                measured["accuracy_vs_normal"] = metrics(out, ref)
                if mode == "channel_2gpu":
                    measured["correctness_passed"] = passes(measured["accuracy_vs_normal"], GATES["audio_channel"])
                torch.save(out, output / f"{mode}_{name}_output.pt")
                workload_result["strategies"][mode] = measured
                print(f"[{kind}/{mode}] {name}: {measured['median_ms']:.2f} ms", flush=True)
                del out
            except Exception:
                workload_result["strategies"][mode] = {"status": "failed", "traceback": traceback.format_exc()}
            results["workloads"][name] = workload_result
            write_json(output / "results.json", results)
        # Matching tiles must agree between the sequential and concurrent paths.
        for mode in list(modes):
            if "tiled_2gpu" in mode:
                other = mode.replace("2gpu", "1gpu")
                a, b = output / f"{mode}_{name}_output.pt", output / f"{other}_{name}_output.pt"
                if a.exists() and b.exists():
                    value = metrics(torch.load(a, weights_only=True), torch.load(b, weights_only=True))
                    workload_result["strategies"][mode]["accuracy_vs_matching_tiles"] = value
                    workload_result["strategies"][mode]["matching_tiles_passed"] = passes(value, GATES["audio_channel"])
        write_json(output / "results.json", results)
    return results


def run_native_stack(kind, path, output, args):
    """Measure all 50 different checkpoint blocks, with no layer-time multiplication.

    Scope excludes embedding/vision, DiT packing/refiner/output heads and sampling.
    The reference streams one checkpoint block at a time to a single GPU; its
    latency is not called a ComfyUI CPU-offload baseline.
    """
    inputs = make_inputs(kind, output, path)
    references = {}
    for input_file in sorted(inputs.glob("*.pt")):
        data = torch.load(input_file, weights_only=True)
        x = data["input"].cuda().to(dtype=torch.float32 if kind == "dit" else torch.float16)
        te = data.get("t_emb")
        te = te.cuda() if te is not None else None
        rope = data.get("rope")
        rope = rope.to(device=x.device,dtype=x.dtype) if rope is not None else None
        with torch.inference_mode():
            for index in range(50):
                block = load_native_block(kind, path, index=index)
                x = native_call(kind, block, x, te, data.get("segments"), rope)
                del block
                cleanup()
        references[input_file.stem] = x.cpu()
        torch.save(references[input_file.stem], output / f"reference_{input_file.stem}_output.pt")
        del x, te, rope, data
        cleanup()
        print(f"[{kind}/50-block reference] {input_file.stem} ready", flush=True)
    blocks = [load_native_block(kind, path, index=i, device=f"cuda:{int(i >= 25)}") for i in range(50)]
    results = {"scope": "measured 50-block checkpoint stack, PP=2; excludes embedding, packing, refiner, heads, sampler, VAEs",
               "reference": "single-GPU block-by-block checkpoint streaming; correctness reference only",
               "gates": GATES["block"], "workloads": {}}
    for input_file in sorted(inputs.glob("*.pt")):
        data = torch.load(input_file, weights_only=True)
        x = data["input"].cuda().to(dtype=torch.float32 if kind == "dit" else torch.float16)
        t_by_dev = [data["t_emb"].to(f"cuda:{i}") for i in range(2)] if "t_emb" in data else [None, None]
        rope_by_dev = [data["rope"].to(device=f"cuda:{i}",dtype=x.dtype) for i in range(2)] if "rope" in data else [None, None]
        def call():
            h = x.clone()
            for index, block in enumerate(blocks):
                dev = int(index >= 25)
                if index == 25:
                    h = h.to("cuda:1", non_blocking=False)
                h = native_call(kind, block, h, t_by_dev[dev], data.get("segments"), rope_by_dev[dev])
            return h
        try:
            if not metrics(references[input_file.stem],references[input_file.stem])["valid"]:
                raise RuntimeError("Native full-stack reference is nonfinite or degenerate; timing is not accepted")
            measured, out = measure(call, [0, 1], args.warmups, args.iterations,
                                    reference=references[input_file.stem], gate=GATES["block"])
            measured["accuracy"] = metrics(out, references[input_file.stem])
            measured["correctness_passed"] = passes(measured["accuracy"], GATES["block"])
            measured["input_sha256"] = digest_tensor(data["input"])
            results["workloads"][input_file.stem] = measured
            torch.save(out,output/f"pp_{input_file.stem}_output.pt")
            print(f"[{kind}/PP full 50 blocks] {input_file.stem}: {measured['median_ms']:.2f} ms", flush=True)
            del out
        except Exception:
            results["workloads"][input_file.stem] = {"status":"failed", "traceback":traceback.format_exc()}
            print("STACK_FAILED",kind,input_file.stem,traceback.format_exc(),flush=True)
            cleanup()
        write_json(output / "results.json", results)
        del x, data
    return results


def environment(output, args, path):
    import comfy_kitchen as ck
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Two physical GPUs are required. No simulated results are permitted.")
    if not args.allow_non_t4 and any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("This rerun requires two Tesla T4 GPUs.")
    packages = {}
    for name in ["torch", "comfy-kitchen", "comfy-aimdo", "safetensors", "numpy", "huggingface-hub"]:
        packages[name] = importlib.metadata.version(name)
    info = {"suite": args.suite, "round": args.round, "warmups": args.warmups, "iterations": args.iterations,
            "packages": packages, "cuda": torch.version.cuda, "nccl": list(torch.cuda.nccl.version()) if isinstance(torch.cuda.nccl.version(), tuple) else torch.cuda.nccl.version(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
            "p2p": torch.cuda.can_device_access_peer(0, 1), "checkpoint_revision": MODEL_REVISION,
            "checkpoint_file": path, "checkpoint_size": Path(path).stat().st_size,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "kitchen_eager_activation_residual_compatibility": KITCHEN_EAGER_COMPAT,
            "kitchen_backends":ck.list_backends(),
            "native_compute_dtype":"float32" if args.suite.startswith("dit") else "float32" if args.suite == "audio" else "float16",
            "proposal_compute_dtype":"float32" if args.suite.startswith("dit") else None,
            "timing": "wall time with explicit CUDA synchronization; max rank time for distributed tests",
            "confidence_interval": "bootstrap of timing samples within one process; not independent-session uncertainty"}
    write_json(output / "environment.json", info)
    proc = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True)
    (output / "nvidia-smi.txt").write_text(proc.stdout, encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (output / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")


def validate_runtime():
    import comfy.ops
    import comfy_kitchen as ck
    cuda_backend = ck.list_backends()["cuda"]
    if tuple(int(v) for v in torch.version.cuda.split(".")) < (13,):
        raise RuntimeError("CUDA 13+ PyTorch is required; ComfyUI disables the native CUDA backend on older builds")
    if not cuda_backend["available"] or cuda_backend["disabled"]:
        raise RuntimeError(f"Native CUDA kernels are unavailable: {cuda_backend}")
    print("RUNTIME_VALIDATED", torch.__version__, torch.version.cuda, json.dumps(cuda_backend), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["te", "dit", "te_stack", "dit_stack", "video", "video_tiling", "audio"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--allow-non-t4", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    kind = "video" if args.suite == "video_tiling" else args.suite.replace("_stack", "")
    validate_runtime()
    path = checkpoint(kind)
    environment(output, args, path)
    if args.suite in ["te", "dit"]:
        run_blocks(kind, path, output, args)
    elif args.suite.endswith("_stack"):
        run_native_stack(kind, path, output, args)
    elif args.suite == "video":
        run_video(path, output, args)
    else:
        run_tiling(kind, path, output, args)
    print(f"[COMPLETE] {args.suite} round {args.round}", flush=True)


if __name__ == "__main__":
    main()
