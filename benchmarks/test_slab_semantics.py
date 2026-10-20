"""Bounded local two-rank simulation; never a simulated two-T4 speedup.

Uses deterministic coordinate oracles and the actual ComfyUI raw decoder graph
with initialized, reduced INT8 weights. Optional --checkpoint loads all 36
trained blocks instead. No downloads or remote experiments are started.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ComfyUI"))
sys.path.insert(0, str(ROOT / ".retest-deps"))

import torch
from patches.slab_video_vae import SlabPlan, TwoSlabDecoder, stitch_slabs


def accuracy(a, b):
    a, b = a.detach().cpu().double().reshape(-1), b.detach().cpu().double().reshape(-1)
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    delta = a - b
    return {"is_bit_exact": torch.equal(a, b), "mae": delta.abs().mean().item(),
            "max_abs": delta.abs().max().item(),
            "rel_l2": (delta.norm() / b.norm()).item(),
            "cosine": (torch.dot(a, b) / (a.norm() * b.norm())).item()}


def coordinate_checks():
    checks = 0
    # Exhaustive odd/even widths, asymmetric halos, empty/full overlap, and
    # one-pixel/full/center fades. Independent oracle writes global columns.
    for width in range(2, 14):
        for overlap in range(width + 1):
            for fade in sorted(set([0, min(1, overlap * 2), overlap, overlap * 2])):
                plan = SlabPlan.create(width, overlap * 2, ratio=2, fade_pixels=fade)
                global_x = torch.arange(width * 2, dtype=torch.float32)
                left = (global_x[:plan.left_end * 2] + 10).view(1, 1, 1, 1, -1).expand(2, 3, 4, 6, -1)
                right = (global_x[plan.right_start * 2:] - 20).view(1, 1, 1, 1, -1).expand(2, 3, 4, 6, -1)
                expected = torch.empty(width * 2)
                for col in range(width * 2):
                    if col < plan.fade_start:
                        expected[col] = col + 10
                    elif col >= plan.fade_start + fade:
                        expected[col] = col - 20
                    else:
                        alpha = .5 if fade == 1 else (col - plan.fade_start) / (fade - 1)
                        expected[col] = col + 10 - 30 * alpha
                out = stitch_slabs(left, right, plan)
                assert torch.allclose(out, expected, atol=3e-6, rtol=0)
                # If both slabs agree, stitching must add no numerical seam.
                a = global_x[:plan.left_end * 2].view(1, 1, 1, 1, -1)
                b = global_x[plan.right_start * 2:].view(1, 1, 1, 1, -1)
                assert torch.equal(stitch_slabs(a, b, plan).flatten(), global_x)
                checks += 1
    for kwargs in [dict(width=1), dict(width=8, overlap_pixels=65),
                   dict(width=8, overlap_pixels=144), dict(width=8, fade_pixels=65),
                   dict(width=8, fade_pixels=-1), dict(width=8, overlap_pixels=-16)]:
        try:
            SlabPlan.create(**kwargs)
        except ValueError:
            checks += 1
        else:
            raise AssertionError(f"Invalid plan accepted: {kwargs}")
    plan = SlabPlan.create(8)
    a, b = torch.ones(1, 3, 4, 16, 96), torch.ones(1, 3, 4, 16, 95)
    try:
        stitch_slabs(a, b, plan)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("Inconsistent output accepted")
    return {"passed": True, "cases": checks, "oracle": "independent per-column global-coordinate oracle"}


def make_fixture(heads=4):
    import comfy.ops
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE, ViT3DDecoder

    class Fixture(torch.nn.Module):
        _decode_pixels = MiniMaxH3VideoVAE._decode_pixels

        def __init__(self):
            super().__init__()
            self.post_quant_conv = torch.nn.Conv3d(24, 24, 1)
            operations = comfy.ops.mixed_precision_ops({"mixed_ops": True}, torch.float16)
            self.decoder = ViT3DDecoder(num_layers=2, heads=heads, dim_head=64, operations=operations)

    torch.manual_seed(901)
    model = Fixture().half()
    # ComfyUI's no-init modules and suffix buffers must all be initialized.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.fill_(1)
            elif "scale" in name:
                p.fill_(.5)
            elif p.ndim > 1:
                p.normal_(0, .02)
            else:
                p.zero_()
        model.decoder.mask_token.zero_()
    quant_keys = []
    for layer in range(2):
        for suffix in ["attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2"]:
            key = f"decoder.transformer_blocks.{layer}.{suffix}"
            quant_keys.append(key)
    # Mixed precision Linear has no registered weight until checkpoint loading.
    # Build its synthetic checkpoint explicitly, rather than reading empty ops.
    state = {name: value.detach().clone() for name, value in model.named_parameters()}
    state["decoder.mask_token"] = model.decoder.mask_token.clone()
    for key in quant_keys:
        linear = model.get_submodule(key)
        shape = (linear.out_features, linear.in_features)
        state[key + ".weight"] = torch.randint(-8, 9, shape, dtype=torch.int8)
        state[key + ".weight_scale"] = torch.full((shape[0], 1), .002)
        state[key + ".bias"] = torch.zeros(shape[0], dtype=torch.float16)
        state[key + ".comfy_quant"] = torch.tensor(list(json.dumps({
            "format": "int8_tensorwise", "convrot": True,
            "convrot_groupsize": 256}).encode()), dtype=torch.uint8)
    model.load_state_dict(state, strict=True)
    replica = Fixture().half()
    replica.load_state_dict(state, strict=True)
    del state
    for item in [model, replica]:
        item.cuda().eval().requires_grad_(False)
        for key in quant_keys:
            assert isinstance(item.get_submodule(key).weight, comfy.ops.QuantizedTensor)
    return model, replica


def sequential_oracle(models, z, overlap, fade):
    # Independently derive bounds; never call production plan/stitch helpers.
    width, units = z.shape[-1], overlap // 16
    right_start = width // 2 - units // 2
    left_end = right_start + units
    with torch.cuda.device(0), torch.inference_mode():
        left = models[0]._decode_pixels(z[..., :left_end].contiguous())
        right = models[1]._decode_pixels(z[..., right_start:].contiguous())
        start = (right_start * 16 + left_end * 16 - fade) // 2
        end = start + fade
        out = torch.empty((*left.shape[:-1], width * 16), device=z.device, dtype=left.dtype)
        out[..., :start] = left[..., :start]
        out[..., end:] = right[..., end - right_start * 16:]
        if fade:
            weights = torch.arange(fade, device=z.device, dtype=torch.float32) / (fade - 1) if fade > 1 else torch.tensor([.5], device=z.device)
            l = left[..., start:end].float()
            r = right[..., start - right_start * 16:end - right_start * 16].float()
            out[..., start:end] = l + (r - l) * weights
        return out, (left, right)


def classify_kernels(names):
    # Profiler versions return either mangled or demangled CUTLASS names.
    # Turing uses i8816 GEMMs; later GPUs use the visitor implementation.
    int8_gemm = [n for n in names if (
        ("DefaultGemmWithVisitor" in n and "OpClassTensorOp" in n
         and ("VisitorIa" in n or "<signed char" in n))
        or ("tensorop_i" in n and "gemm_s8" in n))]
    int8_attention = [n for n in names if "attn" in n and ("int8" in n or "i8" in n)]
    fp32_fallback = [n for n in names if "sgemm" in n.lower()]
    return int8_gemm, int8_attention, fp32_fallback


def kernel_probe(model, device="cuda:0"):
    """One tiny trace to detect the FP32 fallback seen in the earlier proposal."""
    z = torch.ones(1, 24, 1, 3, 8, device=device, dtype=torch.float16)
    with torch.cuda.device(device), torch.inference_mode():
        model._decode_pixels(z)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            model._decode_pixels(z)
            torch.cuda.synchronize(device)
    names = sorted({e.name for e in prof.events() if str(e.device_type).endswith("CUDA")})
    int8_gemm, int8_attention, fp32_fallback = classify_kernels(names)
    if not int8_gemm or not int8_attention or fp32_fallback:
        raise AssertionError("Native INT8 kernel identity check failed; inspect kernel evidence")
    return {"device": device, "int8_tensorcore_gemm_confirmed": True,
            "int8_attention_confirmed": True, "fp32_sgemm_observed": False,
            "kernel_names": names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "slab_results/local_semantics.json")
    args = parser.parse_args()
    started = time.perf_counter()
    result = {"scope": "one physical GPU, two logical ranks and two CUDA streams; no two-T4 timing claim",
              "cpu_geometry": coordinate_checks(), "torch": torch.__version__,
              "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / "patches/slab_video_vae.py"]},
              "gpu": torch.cuda.get_device_name(0), "physical_gpu_count": 1,
              "cases": [], "trained_checkpoint": bool(args.checkpoint)}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    save()
    if args.checkpoint:
        with args.checkpoint.open("rb") as checkpoint_file:
            digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        result["checkpoint"] = {"path": str(args.checkpoint.resolve()),
                                "bytes": args.checkpoint.stat().st_size, "sha256": digest}
        from retest import load_video_native
        a, sd = load_video_native(args.checkpoint, "cuda:0")
        del sd
        b, sd = load_video_native(args.checkpoint, "cuda:0")
        del sd
        models = (a, b)
        result["model_scope"] = "full trained 36-block checkpoint, replicas loaded separately"
    else:
        models = make_fixture()
        result["model_scope"] = "native _decode_pixels + ViT3DDecoder, 2 blocks, 4 heads x 64, initialized synthetic INT8 convrot weights; 4 register + 1 zero suffix tokens"
    result["native_kernel_probe"] = kernel_probe(models[0])
    save()
    observed = [None, None]
    def callback(rank):
        def decode(z):
            out = models[rank]._decode_pixels(z)
            observed[rank] = out
            return out
        return decode
    decoder = TwoSlabDecoder([callback(i) for i in range(2)], devices=("cuda:0", "cuda:0"))
    try:
        workloads = [("odd_batch_temporal", (2, 24, 3, 3, 9), 303),
                     ("768x768_1F", (1, 24, 1, 48, 48), 301),
                     ("768x1344_2F", (1, 24, 2, 48, 84), 302)]
        baseline = json.loads((ROOT / "kaggle_retest_output/cu130/retest_results/round_0/video_tiling/results.json").read_text())
        for name, shape, seed in workloads:
            cpu_z = torch.randn(shape, generator=torch.Generator().manual_seed(seed)).half()
            source = "locally generated synthetic fixture"
            if name in baseline["workloads"]:
                cpu_z = torch.load(ROOT / f"slab_results/saved_references/input_{name}.pt", weights_only=True)
                source = "saved tensor from completed T4 allocation; seed alone is not portable"
            digest = hashlib.sha256(cpu_z.numpy().tobytes()).hexdigest()
            if name in baseline["workloads"]:
                assert digest == baseline["workloads"][name]["input_sha256"], "Historical input mismatch"
            z = cpu_z.cuda()
            with torch.cuda.device(0), torch.inference_mode():
                full = models[0]._decode_pixels(z)
            assert full.shape == (shape[0], 3, shape[2] * 4, shape[3] * 16, shape[4] * 16)
            variants = [(64, 64), (128, 128), (128, 32)]
            if name == "odd_batch_temporal":
                variants += [(0, 0), (64, 0), (64, 1), (shape[-1] * 16, shape[-1] * 16)]
            for overlap, fade in variants:
                sequential, raw_slabs = sequential_oracle(models, z, overlap, fade)
                out, stages = decoder.decode(z, overlap, fade)
                assert all(torch.equal(observed[i], raw_slabs[i]) for i in range(2)), "Logical stream scheduling changed native slab outputs"
                matching = accuracy(out, sequential)
                # Different linspace/arange weight construction can round an
                # FP16 blend by one ULP. Decode/schedule is checked separately
                # by full-width identity and pointwise geometry controls.
                assert matching["rel_l2"] <= 1e-5 and matching["max_abs"] <= 5e-4, matching
                vs_full = accuracy(out, full)
                if overlap == shape[-1] * 16:
                    assert vs_full["is_bit_exact"], "Full-width control must restore full context"
                seam = stages["fade_range"]
                seam_start, seam_end = max(0, seam[0] - 16), min(full.shape[-1], seam[1] + 16)
                seam_metrics = accuracy(out[..., seam_start:seam_end], full[..., seam_start:seam_end])
                row = {"name": name, "seed": seed, "input_sha256": digest,
                       "input_source": source, "rank_decodes_bit_exact_vs_sequential": True,
                       "overlap_pixels": overlap, "fade_pixels": fade,
                       "matching_sequential_slabs": matching,
                       "vs_full_raw_decode": vs_full, "seam_band_vs_full": seam_metrics,
                       "stages": stages}
                result["cases"].append(row)
                save()
                print(f"PASS {name} overlap={overlap} fade={fade}: matching L2={matching['rel_l2']:.3g}, full L2={vs_full['rel_l2']:.3g}", flush=True)
                observed[:] = [None, None]
                del sequential, out, raw_slabs
            del full, z
        from comfy.ldm.minimax.vae import create_token_ids
        full_ids = create_token_ids((2, 48, 84), "cpu", torch.float32).reshape(2, 48, 84, 3)
        slab_ids = create_token_ids((2, 48, 44), "cpu", torch.float32).reshape(2, 48, 44, 3)
        result["position_control"] = {"time_and_height_unchanged": torch.equal(full_ids[:, :, :44, :2], slab_ids[..., :2]),
                                      "width_renormalized": not torch.equal(full_ids[:, :, :44, 2], slab_ids[..., 2]),
                                      "full_width_x_first_last": full_ids[0, 0, [0, -1], 2].tolist(),
                                      "left_slab_x_first_last": slab_ids[0, 0, [0, -1], 2].tolist()}
        assert all(result["position_control"][k] for k in ["time_and_height_unchanged", "width_renormalized"])
        result["passed"] = True
    finally:
        decoder.close()
    result["wall_seconds"] = time.perf_counter() - started
    save()
    print(f"Saved {args.output}; {result['wall_seconds']:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
