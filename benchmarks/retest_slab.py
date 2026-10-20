"""New slab-only experiment. Requires cached inputs/references and two real T4s.

Never downloads models, launches instances, or reruns previous benchmark modes.
Historical timing ratios are exploratory comparisons across allocations.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

from retest import (ROOT, GATES, environment, load_video_native, measure, metrics,
                    passes, validate_runtime, write_json)
import torch
from patches.slab_video_vae import TwoSlabDecoder
from test_slab_semantics import kernel_probe, sequential_oracle


class UtilizationSampler:
    """200 ms nvidia-smi observations, including warmups and measured calls."""
    def __init__(self):
        self.rows = []
        self.error = None
        self.process = None

    def start(self):
        try:
            self.process = subprocess.Popen([
                "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits", "--loop-ms=200"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            def read():
                for line in self.process.stdout:
                    try:
                        gpu, util, memory = [int(v.strip()) for v in line.split(",")]
                        self.rows.append({"gpu": gpu, "util_percent": util,
                                          "memory_used_mib": memory,
                                          "monotonic_seconds": time.monotonic()})
                    except ValueError:
                        self.error = "nvidia-smi returned unavailable utilization"
            self.thread = threading.Thread(target=read, daemon=True)
            self.thread.start()
        except OSError:
            self.error = "nvidia-smi sampler unavailable"

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.thread.join(timeout=5)
        return {"interval_ms": 200, "window": "warmups + timings + final output call",
                "samples": self.rows, "error": self.error,
                "gpu_mean_percent": [statistics.mean(r["util_percent"] for r in self.rows if r["gpu"] == i)
                                     if any(r["gpu"] == i for r in self.rows) else None for i in range(2)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--historical", type=Path, default=ROOT / "kaggle_retest_output/final/aggregate_results.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--budget-seconds", type=int, default=600)
    parser.add_argument("--variant", choices=["64", "128", "64_128", "128_center32", "all"], default="64")
    args = parser.parse_args()
    if args.warmups < 0 or args.iterations < 1 or args.budget_seconds < 1:
        parser.error("Invalid iteration count or budget")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"status": "preflight", "scope": "full trained raw _decode_pixels; spatial approximation",
              "historical_ratios": "cross-allocation exploratory comparisons; not paired speedup estimates",
              "new_modes_only": True, "cases": []}
    target = args.output / "results.json"
    write_json(target, result)
    # Hard cap covers model loading as well as GPU work. Partial results are
    # persisted after each mode. This process alone is terminated on expiry.
    def expired():
        result["status"] = "budget_exceeded"
        write_json(args.output / "budget_exceeded.json", {"budget_seconds": args.budget_seconds})
        os._exit(124)
    timer = threading.Timer(args.budget_seconds, expired)
    timer.daemon = True
    timer.start()
    decoder = None
    try:
        validate_runtime()
        if torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError("Two physical Tesla T4 GPUs are required; simulation cannot produce these timings")
        expected_packages = {"torch": "2.11.0+cu130", "comfy-kitchen": "0.2.37", "comfy-aimdo": "0.5.5"}
        for package, expected in expected_packages.items():
            if importlib.metadata.version(package) != expected:
                raise RuntimeError(f"Saved-reference experiment requires {package}=={expected}")
        comfy = Path(os.environ.get("COMFY_ROOT", str(ROOT / "ComfyUI")))
        commit = subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True).strip()
        if commit != "d49e888586dd8ae012c0667b33466b815fee07f7":
            raise RuntimeError("Saved-reference experiment requires the pinned ComfyUI commit")
        if args.checkpoint.stat().st_size != 2811065184:
            raise RuntimeError("Expected the pinned full Video VAE checkpoint size")
        with args.checkpoint.open("rb") as checkpoint_file:
            checkpoint_hash = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        if checkpoint_hash != "52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e":
            raise RuntimeError("Checkpoint does not match the locally verified trained model")
        result["checkpoint_sha256"] = checkpoint_hash
        args.suite, args.round, args.allow_non_t4 = "video_slab", 0, False
        environment(args.output, args, str(args.checkpoint))
        source_files = [Path(__file__), ROOT / "patches/slab_video_vae.py",
                        ROOT / "benchmarks/test_slab_semantics.py", ROOT / "benchmarks/retest.py"]
        write_json(args.output / "source_hashes.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
        history = json.loads(args.historical.read_text())
        manifest = {row["file"]: row for row in json.loads((args.references / "manifest.json").read_text())}
        prepared = []
        for name in ["768x768_1F", "768x1344_2F"]:
            tensors = []
            for filename in [f"input_{name}.pt", f"native_{name}_output.pt"]:
                tensor = torch.load(args.references / filename, map_location="cpu", weights_only=True)
                digest = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
                if digest != manifest[filename]["tensor_sha256"]:
                    raise RuntimeError("Saved tensor no longer matches its download manifest")
                if not torch.isfinite(tensor).all():
                    raise RuntimeError("Nonfinite saved tensor")
                tensors.append(tensor)
            native = next(r for r in history if r["suite"] == "video_tiling" and r["workload"] == name and r["strategy"] == "native")
            # The aggregate retains per-round accuracy/metadata; source JSON
            # records the authoritative input bytes, independently of manifest.
            old = json.loads((args.historical.parent / Path(native["source_results"][0].replace("\\", "/"))).read_text())
            digest = hashlib.sha256(tensors[0].numpy().tobytes()).hexdigest()
            if digest != old["workloads"][name]["input_sha256"]:
                raise RuntimeError("Historical input hash mismatch")
            shape = old["workloads"][name]["shape"]
            if list(tensors[0].shape) != shape or list(tensors[1].shape) != [shape[0], 3, shape[2] * 4, shape[3] * 16, shape[4] * 16]:
                raise RuntimeError("Saved input/reference geometry mismatch")
            prepared.append((name, tensors[0], tensors[1], digest))
        models = []
        for device in ["cuda:0", "cuda:1"]:
            model, sd = load_video_native(args.checkpoint, device)
            del sd
            models.append(model)
        decoder = TwoSlabDecoder([m._decode_pixels for m in models])
        result["native_kernel_probes"] = [kernel_probe(m, f"cuda:{i}") for i, m in enumerate(models)]
        # One NEW tiny full-width control catches physical-device placement and
        # replica/transport errors before the costly experiment. No old sweep.
        tiny = torch.randn((1, 24, 1, 2, 8), generator=torch.Generator().manual_seed(901)).half().cuda()
        with torch.cuda.device(0), torch.inference_mode():
            tiny_ref = models[0]._decode_pixels(tiny).cpu()
        tiny_out, _ = decoder.decode(tiny, overlap_pixels=128)
        control = metrics(tiny_out, tiny_ref)
        result["physical_full_width_control"] = control
        if not passes(control, GATES["audio_channel"]):
            raise RuntimeError("Full-width physical scheduling/replication control failed")
        del tiny, tiny_ref, tiny_out
        result["status"] = "running"
        variants = {"64": (64, 64), "128": (128, 128), "128_center32": (128, 32)}
        selected = (list(variants) if args.variant == "all" else
                    ["64", "128"] if args.variant == "64_128" else [args.variant])
        for name, cpu_z, ref, digest in prepared:
            z = cpu_z.cuda()
            for label in selected:
                overlap, fade = variants[label]
                # Sequential-slab oracle is a new validation of this method,
                # never a rerun of the historical normal/tile timing modes.
                sequential, raw = sequential_oracle((models[0], models[0]), z, overlap, fade)
                out, metadata = decoder.decode(z, overlap, fade)
                matching = metrics(out, sequential)
                if not passes(matching, GATES["audio_channel"]) or matching["max_abs"] > .002:
                    raise RuntimeError("Scheduling/stitching differs from sequential slabs")
                versus = metrics(out, ref)
                row = {"name": name, "variant": label, "input_sha256": digest,
                       "timing_input_scope": "CPU latent input; both input copies included; output resident on GPU 0",
                       "geometry": metadata, "accuracy_vs_matching_slabs": matching,
                       "accuracy_vs_full_saved_raw": versus,
                       "strict_full_decode_gate_passed": passes(versus, GATES["video"])}
                cpu_out = out.detach().cpu()
                start, end = metadata["fade_range"]
                lo, hi = max(0, start - 16), min(cpu_out.shape[-1], end + 16)
                row["seam_band_vs_full_saved_raw"] = metrics(cpu_out[..., lo:hi], ref[..., lo:hi])
                row["boundary_gradient_error_mae"] = {
                    str(col): ((cpu_out[..., col].float() - cpu_out[..., col - 1].float())
                               - (ref[..., col].float() - ref[..., col - 1].float())).abs().mean().item()
                    for col in sorted(set([start, end])) if 0 < col < cpu_out.shape[-1]}
                row["temporal_difference_error_mae"] = (torch.diff(cpu_out.float(), dim=2)
                                                         - torch.diff(ref.float(), dim=2)).abs().mean().item()
                del cpu_out
                out_reference = sequential.detach().cpu()
                del sequential, raw, out
                stages = []
                def call():
                    output, metadata = decoder.decode(cpu_z, overlap, fade)
                    stages.append(metadata)
                    return output
                sampler = UtilizationSampler()
                sampler.start()
                try:
                    timing, output = measure(call, [0, 1], args.warmups, args.iterations,
                                             reference=out_reference, gate=GATES["audio_channel"])
                finally:
                    row["utilization"] = sampler.stop()
                timed_stages = stages[args.warmups + 1:args.warmups + 1 + args.iterations]
                row["timing"] = timing
                row["stage_samples"] = timed_stages
                for key in ["output_transfer_ms", "stitch_ms"]:
                    row["timing"][key + "_median"] = statistics.median(s[key] for s in timed_stages)
                row["accuracy_of_timing_output"] = metrics(output, ref)
                comparisons = [r for r in history if r["suite"] == "video_tiling" and r["workload"] == name]
                native = next(r for r in comparisons if r["strategy"] == "native")
                best_tiles = min((r for r in comparisons if r["strategy"].startswith("tiled_2gpu")), key=lambda r: r["median_ms"])
                row["historical_native_time_ratio"] = native["median_ms"] / timing["median_ms"]
                row["historical_best_2gpu_tile_time_ratio"] = best_tiles["median_ms"] / timing["median_ms"]
                row["historical_best_2gpu_tile_strategy"] = best_tiles["strategy"]
                result["cases"].append(row)
                write_json(target, result)
                print(f"NEW slab {name}/{label}: {timing['median_ms']:.2f}ms; full L2={versus['rel_l2']:.5f}", flush=True)
                del output, out_reference
            del z
        result["status"] = "complete"
        write_json(target, result)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        write_json(target, result)
        raise
    finally:
        if decoder is not None:
            decoder.close()
        timer.cancel()


if __name__ == "__main__":
    main()
