"""
Multi-GPU Spatial Tile Decoder for MiniMax-H3 Video VAE on 2x NVIDIA Tesla T4.
Supports:
  1. Single-GPU Normal Decode (full non-tiled forward pass)
  2. Single-GPU Tiled Decode (sequential tiles on 1 GPU)
  3. 2-GPU Parallel Tiled Decode (tiles distributed across GPU 0 and GPU 1, decoded concurrently, then stitched/blended)

Preserves exact spatial overlap/halo and blending logic from reference ComfyUI ViT3D decoder.
"""

import os
import sys
import time
import math
import queue
import threading
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any

# Ensure ComfyUI is in path if available
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ComfyUI"))
if os.path.exists("/tmp/ComfyUI") and "/tmp/ComfyUI" not in sys.path:
    sys.path.insert(0, "/tmp/ComfyUI")

# Fallback mock for comfy_aimdo if missing
try:
    import comfy_aimdo
except ImportError:
    import types
    m = types.ModuleType("comfy_aimdo")
    m.host_buffer = types.ModuleType("host_buffer")
    sys.modules["comfy_aimdo"] = m
    sys.modules["comfy_aimdo.host_buffer"] = m.host_buffer

from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

class MultiGPUVideoVAEDecoder:
    """
    Manages dual-GPU spatial tile decoding for MiniMax-H3 Video VAE.
    """
    def __init__(
        self,
        base_vae: Optional[MiniMaxH3VideoVAE] = None,
        dev0: str = "cuda:0",
        dev1: str = "cuda:1",
        num_layers: int = 36,
        tile_size: int = 256,
        tile_overlap_min: int = 64,
        dtype: torch.dtype = torch.float16,
    ):
        self.dev0 = torch.device(dev0 if torch.cuda.is_available() else "cpu")
        self.dev1 = torch.device(dev1 if (torch.cuda.is_available() and torch.cuda.device_count() >= 2) else self.dev0)
        self.is_multi_gpu = (self.dev0 != self.dev1)
        self.dtype = dtype
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min

        print(f"[VIDEO VAE] Initializing MultiGPUVideoVAEDecoder on {self.dev0} and {self.dev1} (Multi-GPU: {self.is_multi_gpu})...")

        if base_vae is not None:
            self.vae_dev0 = base_vae.to(device=self.dev0, dtype=self.dtype)
            if self.is_multi_gpu:
                # Replicate decoder on GPU 1
                import copy
                self.vae_dev1 = copy.deepcopy(base_vae).to(device=self.dev1, dtype=self.dtype)
            else:
                self.vae_dev1 = self.vae_dev0
        else:
            # Instantiate models with exact architecture
            self.vae_dev0 = MiniMaxH3VideoVAE(
                num_layers=num_layers,
                tile_size=tile_size,
                tile_overlap_min=tile_overlap_min,
            ).to(device=self.dev0, dtype=self.dtype)
            self._init_weights(self.vae_dev0)

            if self.is_multi_gpu:
                self.vae_dev1 = MiniMaxH3VideoVAE(
                    num_layers=num_layers,
                    tile_size=tile_size,
                    tile_overlap_min=tile_overlap_min,
                ).to(device=self.dev1, dtype=self.dtype)
                self.vae_dev1.load_state_dict(self.vae_dev0.state_dict())
            else:
                self.vae_dev1 = self.vae_dev0

        self.vae_ratio = self.vae_dev0.vae_ratio # 16

        # Dedicated CUDA streams for true parallel execution
        if torch.cuda.is_available() and self.is_multi_gpu:
            self.stream0 = torch.cuda.Stream(device=self.dev0)
            self.stream1 = torch.cuda.Stream(device=self.dev1)
        else:
            self.stream0 = None
            self.stream1 = None

    def _init_weights(self, model: nn.Module):
        """Initializes uninitialized parameter buffers with valid FP16 weights."""
        for name, p in model.named_parameters():
            if "scale" in name or "norm" in name:
                p.data.fill_(1.0)
            elif p.dim() > 1:
                torch.nn.init.normal_(p, std=0.02)
            else:
                p.data.zero_()

    def split_tiles(self, input_len: int, tile_size: Optional[int] = None, overlap_min: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
        """Splits dimension into overlapping tiles according to ComfyUI's VAE formula."""
        ts = tile_size or self.tile_size
        ov_min = overlap_min or self.tile_overlap_min

        if ts >= input_len:
            return [0], [input_len], []

        N = math.ceil(input_len / ts)
        while True:
            overlaps = [ov_min] * (N - 1)
            remaining = ts * N - sum(overlaps) - input_len
            if remaining < 0:
                N += 1
            else:
                break

        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (N - 1)] += self.vae_ratio

        tile_start_idx = [0]
        for i in range(N - 1):
            tile_start_idx.append(tile_start_idx[-1] + ts - overlaps[i])

        return tile_start_idx, [ts] * N, overlaps

    def blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        """Linear cross-fade blending along specified dimension."""
        return self.vae_dev0.blend(a, b, blend_extent, dim)

    # -------------------------------------------------------------------------
    # Mode 1: Single-GPU Normal Decode (Non-tiled)
    # -------------------------------------------------------------------------
    def decode_normal(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes single-shot non-tiled decode on GPU 0.
        """
        z_dev = z.to(device=self.dev0, dtype=self.dtype)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.dev0)
            torch.cuda.synchronize(self.dev0)

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                out = self.vae_dev0._decode_pixels(z_dev)
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.dev0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            peak_vram_gb = torch.cuda.max_memory_allocated(self.dev0) / (1024**3) if torch.cuda.is_available() else 0.0
            oom = False
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                out = None
                elapsed_ms = float("inf")
                peak_vram_gb = 16.0
                oom = True
                torch.cuda.empty_cache()
            else:
                raise e

        metrics = {
            "mode": "1-GPU Normal",
            "latency_ms": elapsed_ms,
            "peak_vram_dev0_gb": peak_vram_gb,
            "peak_vram_dev1_gb": 0.0,
            "oom": oom,
            "num_tiles": 1,
        }
        return out, metrics

    # -------------------------------------------------------------------------
    # Mode 2: Single-GPU Tiled Decode (Sequential)
    # -------------------------------------------------------------------------
    def decode_tiled_single(
        self,
        z: torch.Tensor,
        tile_size: Optional[int] = None,
        tile_overlap_min: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes reference spatial tiled decode sequentially on GPU 0.
        """
        ts = tile_size or self.tile_size
        ov = tile_overlap_min or self.tile_overlap_min

        z_dev = z.to(device=self.dev0, dtype=self.dtype)
        height, width = z_dev.shape[-2] * self.vae_ratio, z_dev.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(height, ts, ov)
        x_idx, x_len, x_overlap = self.split_tiles(width, ts, ov)
        total_tiles = len(y_idx) * len(x_idx)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.dev0)
            torch.cuda.synchronize(self.dev0)

        t0 = time.perf_counter()
        tile_latencies = []

        with torch.no_grad():
            canvas = None
            strip = None
            out_y = 0

            for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
                zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
                new_strip = None
                left_tail = None
                out_x = 0

                for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                    zj, zl_x = j_pos // self.vae_ratio, j_len // self.vae_ratio
                    tile_z = z_dev[..., zi:zi + zl, zj:zj + zl_x]

                    t_tile_start = time.perf_counter()
                    tile = self.vae_dev0._decode_pixels(tile_z)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.dev0)
                    tile_latencies.append((time.perf_counter() - t_tile_start) * 1000.0)

                    # Blending
                    if i > 0:
                        tile = self.blend(strip[..., :, x_idx[j]:x_idx[j] + x_len[j]], tile, y_overlap[i - 1], dim=-2)
                    if j > 0:
                        tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)

                    left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
                    if j < len(x_idx) - 1:
                        tile = tile[..., :, :-x_overlap[j]]

                    if canvas is None:
                        canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)

                    if i < len(y_idx) - 1:
                        if new_strip is None:
                            new_strip = torch.empty(*tile.shape[:-2], y_overlap[i], width, dtype=tile.dtype, device=tile.device)
                        new_strip[..., :, out_x:out_x + tile.shape[-1]] = tile[..., -y_overlap[i]:, :]
                        tile = tile[..., :-y_overlap[i], :]

                    canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
                    tile_height = tile.shape[-2]
                    out_x += tile.shape[-1]
                    del tile

                strip = new_strip
                out_y += tile_height

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.dev0)
        total_ms = (time.perf_counter() - t0) * 1000.0
        peak_vram_gb = torch.cuda.max_memory_allocated(self.dev0) / (1024**3) if torch.cuda.is_available() else 0.0

        metrics = {
            "mode": "1-GPU Tiled",
            "latency_ms": total_ms,
            "avg_tile_lat_ms": sum(tile_latencies) / len(tile_latencies),
            "peak_vram_dev0_gb": peak_vram_gb,
            "peak_vram_dev1_gb": 0.0,
            "num_tiles": total_tiles,
            "tile_grid": f"{len(y_idx)}x{len(x_idx)}",
            "tile_size": ts,
            "tile_overlap": ov,
        }
        return canvas, metrics

    # -------------------------------------------------------------------------
    # Mode 3: 2-GPU Parallel Tiled Decode
    # -------------------------------------------------------------------------
    def decode_tiled_multi(
        self,
        z: torch.Tensor,
        tile_size: Optional[int] = None,
        tile_overlap_min: Optional[int] = None,
        dynamic_schedule: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes parallel spatial tiled decode distributed across GPU 0 and GPU 1.
        Tiles are computed concurrently on independent CUDA streams, then stitched on GPU 0.
        """
        if not self.is_multi_gpu:
            return self.decode_tiled_single(z, tile_size, tile_overlap_min)

        ts = tile_size or self.tile_size
        ov = tile_overlap_min or self.tile_overlap_min

        z_dev0 = z.to(device=self.dev0, dtype=self.dtype)
        height, width = z_dev0.shape[-2] * self.vae_ratio, z_dev0.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(height, ts, ov)
        x_idx, x_len, x_overlap = self.split_tiles(width, ts, ov)
        total_tiles = len(y_idx) * len(x_idx)

        # Build tile metadata task list
        tile_tasks = []
        tile_id = 0
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                zj, zl_x = j_pos // self.vae_ratio, j_len // self.vae_ratio
                tile_tasks.append({
                    "tile_id": tile_id,
                    "i": i, "j": j,
                    "zi": zi, "zl": zl,
                    "zj": zj, "zl_x": zl_x,
                    "i_pos": i_pos, "i_len": i_len,
                    "j_pos": j_pos, "j_len": j_len,
                })
                tile_id += 1

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.dev0)
            torch.cuda.reset_peak_memory_stats(self.dev1)
            torch.cuda.synchronize(self.dev0)
            torch.cuda.synchronize(self.dev1)

        t_overall_start = time.perf_counter()

        # Decoded tile outputs storage: index -> tensor on dev0
        decoded_tiles = [None] * total_tiles
        tile_latencies_gpu0 = []
        tile_latencies_gpu1 = []
        transfer_times = []

        # Worker functions for concurrent execution
        task_queue = queue.Queue()
        for task in tile_tasks:
            task_queue.put(task)

        def worker_dev0():
            torch.cuda.set_device(self.dev0)
            stream = self.stream0 or torch.cuda.current_stream(self.dev0)
            with torch.cuda.stream(stream):
                while not task_queue.empty():
                    try:
                        task = task_queue.get_nowait()
                    except queue.Empty:
                        break

                    t_start = time.perf_counter()
                    tile_z = z_dev0[..., task["zi"]:task["zi"] + task["zl"], task["zj"]:task["zj"] + task["zl_x"]]
                    with torch.no_grad():
                        out_tile = self.vae_dev0._decode_pixels(tile_z)
                    torch.cuda.synchronize(self.dev0)
                    tile_lat = (time.perf_counter() - t_start) * 1000.0
                    tile_latencies_gpu0.append(tile_lat)

                    decoded_tiles[task["tile_id"]] = out_tile
                    task_queue.task_done()

        def worker_dev1():
            torch.cuda.set_device(self.dev1)
            stream = self.stream1 or torch.cuda.current_stream(self.dev1)
            with torch.cuda.stream(stream):
                while not task_queue.empty():
                    try:
                        task = task_queue.get_nowait()
                    except queue.Empty:
                        break

                    t_start = time.perf_counter()
                    # Latent slice for GPU 1
                    tile_z = z_dev0[..., task["zi"]:task["zi"] + task["zl"], task["zj"]:task["zj"] + task["zl_x"]].to(self.dev1, non_blocking=True)
                    with torch.no_grad():
                        out_tile = self.vae_dev1._decode_pixels(tile_z)
                    torch.cuda.synchronize(self.dev1)
                    tile_lat = (time.perf_counter() - t_start) * 1000.0
                    tile_latencies_gpu1.append(tile_lat)

                    # Inter-GPU transfer: transfer decoded tile from GPU 1 -> GPU 0
                    t_xfer = time.perf_counter()
                    out_tile_dev0 = out_tile.to(self.dev0, non_blocking=False)
                    torch.cuda.synchronize(self.dev0)
                    transfer_times.append((time.perf_counter() - t_xfer) * 1000.0)

                    decoded_tiles[task["tile_id"]] = out_tile_dev0
                    task_queue.task_done()

        # Launch concurrent threads
        t_compute_start = time.perf_counter()
        thread0 = threading.Thread(target=worker_dev0)
        thread1 = threading.Thread(target=worker_dev1)
        thread0.start()
        thread1.start()
        thread0.join()
        thread1.join()

        torch.cuda.synchronize(self.dev0)
        torch.cuda.synchronize(self.dev1)
        total_compute_ms = (time.perf_counter() - t_compute_start) * 1000.0

        # Stitching & Blending on GPU 0
        t_stitch_start = time.perf_counter()
        canvas = None
        strip = None
        out_y = 0
        tile_ptr = 0

        with torch.no_grad():
            for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
                new_strip = None
                left_tail = None
                out_x = 0

                for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                    tile = decoded_tiles[tile_ptr]
                    tile_ptr += 1

                    if i > 0:
                        tile = self.blend(strip[..., :, x_idx[j]:x_idx[j] + x_len[j]], tile, y_overlap[i - 1], dim=-2)
                    if j > 0:
                        tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)

                    left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
                    if j < len(x_idx) - 1:
                        tile = tile[..., :, :-x_overlap[j]]

                    if canvas is None:
                        canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)

                    if i < len(y_idx) - 1:
                        if new_strip is None:
                            new_strip = torch.empty(*tile.shape[:-2], y_overlap[i], width, dtype=tile.dtype, device=tile.device)
                        new_strip[..., :, out_x:out_x + tile.shape[-1]] = tile[..., -y_overlap[i]:, :]
                        tile = tile[..., :-y_overlap[i], :]

                    canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
                    tile_height = tile.shape[-2]
                    out_x += tile.shape[-1]
                    del tile

                strip = new_strip
                out_y += tile_height

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.dev0)
        stitch_overhead_ms = (time.perf_counter() - t_stitch_start) * 1000.0
        total_ms = (time.perf_counter() - t_overall_start) * 1000.0

        peak_vram_dev0 = torch.cuda.max_memory_allocated(self.dev0) / (1024**3) if torch.cuda.is_available() else 0.0
        peak_vram_dev1 = torch.cuda.max_memory_allocated(self.dev1) / (1024**3) if torch.cuda.is_available() else 0.0

        all_lats = tile_latencies_gpu0 + tile_latencies_gpu1
        gpu0_busy = sum(tile_latencies_gpu0)
        gpu1_busy = sum(tile_latencies_gpu1)
        wall_time = max(total_compute_ms, 1e-4)
        gpu0_util = min(100.0, (gpu0_busy / wall_time) * 100.0)
        gpu1_util = min(100.0, (gpu1_busy / wall_time) * 100.0)

        metrics = {
            "mode": "2-GPU Tiled",
            "latency_ms": total_ms,
            "compute_time_ms": total_compute_ms,
            "stitch_overhead_ms": stitch_overhead_ms,
            "pcie_transfer_ms": sum(transfer_times),
            "avg_tile_lat_ms": sum(all_lats) / len(all_lats) if all_lats else 0.0,
            "tiles_gpu0": len(tile_latencies_gpu0),
            "tiles_gpu1": len(tile_latencies_gpu1),
            "gpu0_util_pct": gpu0_util,
            "gpu1_util_pct": gpu1_util,
            "gpu0_idle_pct": 100.0 - gpu0_util,
            "gpu1_idle_pct": 100.0 - gpu1_util,
            "peak_vram_dev0_gb": peak_vram_dev0,
            "peak_vram_dev1_gb": peak_vram_dev1,
            "num_tiles": total_tiles,
            "tile_grid": f"{len(y_idx)}x{len(x_idx)}",
            "tile_size": ts,
            "tile_overlap": ov,
        }
        return canvas, metrics
