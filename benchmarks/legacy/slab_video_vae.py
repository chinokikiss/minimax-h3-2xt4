"""Two independent raw VAE decodes, with explicit spatial overlap coordinates.

This is an approximation for a global-attention decoder. Passing two cuda:0
devices exercises the same scheduling/stitching code on one physical GPU.
Callers supply separately checkpoint-loaded replicas; no QuantizedTensor copy.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import time

import torch


@dataclass(frozen=True)
class SlabPlan:
    width: int
    ratio: int
    left_end: int
    right_start: int
    fade_pixels: int

    @classmethod
    def create(cls, width, overlap_pixels=64, ratio=16, fade_pixels=None):
        if any(type(v) is not int for v in (width, overlap_pixels, ratio)):
            raise ValueError("Width, overlap and ratio must be integers")
        if width < 2 or ratio < 1 or overlap_pixels < 0:
            raise ValueError("Need width >= 2, positive ratio and nonnegative overlap")
        if overlap_pixels % ratio or overlap_pixels > width * ratio:
            raise ValueError("Overlap must be latent-aligned and fit the full width")
        overlap = overlap_pixels // ratio
        mid = width // 2
        right_start = mid - overlap // 2
        left_end = mid + overlap - overlap // 2
        fade = overlap_pixels if fade_pixels is None else fade_pixels
        if type(fade) is not int or not 0 <= fade <= overlap_pixels:
            raise ValueError("Fade must be an integer between zero and total overlap")
        return cls(width, ratio, left_end, right_start, fade)

    @property
    def output_width(self):
        return self.width * self.ratio

    @property
    def fade_start(self):
        return ((self.right_start + self.left_end) * self.ratio - self.fade_pixels) // 2

    def slices(self, z):
        if z.ndim != 5 or z.shape[-1] != self.width:
            raise ValueError("Expected [B,C,T,H,W] matching the slab plan")
        return z[..., :self.left_end], z[..., self.right_start:]

    def metadata(self, shape):
        value = asdict(self)
        value.update(overlap_pixels=(self.left_end - self.right_start) * self.ratio,
                     overlap_definition="total shared width, not halo per side",
                     latent_ranges=[[0, self.left_end], [self.right_start, self.width]],
                     output_ranges=[[0, self.left_end * self.ratio],
                                    [self.right_start * self.ratio, self.output_width]],
                     fade_range=[self.fade_start, self.fade_start + self.fade_pixels],
                     slab_shapes=[list(shape[:-1]) + [self.left_end],
                                  list(shape[:-1]) + [self.width - self.right_start]])
        return value


def stitch_slabs(left, right, plan):
    """Stitch using global output columns; discard halo outside the fade band."""
    if (left.ndim != 5 or right.ndim != 5 or left.shape[:-1] != right.shape[:-1]
            or left.shape[-1] != plan.left_end * plan.ratio
            or right.shape[-1] != (plan.width - plan.right_start) * plan.ratio
            or left.device != right.device or left.dtype != right.dtype):
        raise ValueError("Decoded slabs have inconsistent geometry, device or dtype")
    start, end = plan.fade_start, plan.fade_start + plan.fade_pixels
    offset = plan.right_start * plan.ratio
    pieces = [left[..., :start]]
    if plan.fade_pixels:
        a, b = left[..., start:end], right[..., start - offset:end - offset]
        alpha = (torch.linspace(0, 1, plan.fade_pixels, device=left.device)
                 if plan.fade_pixels > 1 else torch.full((1,), .5, device=left.device))
        # a + (b-a)*alpha preserves identical values exactly and accumulates FP32.
        pieces.append((a.float() + (b.float() - a.float()) * alpha).to(left.dtype))
    pieces.append(right[..., end - offset:])
    result = torch.cat(pieces, dim=-1)
    if result.shape[-1] != plan.output_width:
        raise RuntimeError("Stitch did not reconstruct the requested width")
    return result


class TwoSlabDecoder:
    def __init__(self, decoders, devices=("cuda:0", "cuda:1"), ratio=16, temporal_ratio=4):
        if len(decoders) != 2 or len(devices) != 2:
            raise ValueError("Two decoder callbacks and two device placements are required")
        self.decoders = tuple(decoders)
        self.devices = tuple(torch.device(d) for d in devices)
        if any(d.type != "cuda" or d.index is None for d in self.devices):
            raise ValueError("Explicit CUDA device indices are required")
        self.ratio, self.temporal_ratio = ratio, temporal_ratio
        self.streams = tuple(torch.cuda.Stream(device=d) for d in self.devices)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vae-slab")

    def close(self):
        self.pool.shutdown(wait=True)

    def _rank_decode(self, rank, slab):
        device, stream = self.devices[rank], self.streams[rank]
        # Both contexts must be entered on the worker thread for native kernels.
        with torch.cuda.device(device), torch.cuda.stream(stream), torch.inference_mode():
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
            x = slab.to(device=device).contiguous()
            events[1].record()
            y = self.decoders[rank](x)
            events[2].record()
            events[2].synchronize()
            if (y.ndim != 5 or y.shape[0] != slab.shape[0]
                    or y.shape[2] != slab.shape[2] * self.temporal_ratio
                    or y.shape[-2] != slab.shape[-2] * self.ratio
                    or y.shape[-1] != slab.shape[-1] * self.ratio
                    or y.device != device):
                raise ValueError("Decoder violated raw spatial/temporal geometry or placement")
            return y, {"input_copy_and_pack_ms": events[0].elapsed_time(events[1]),
                       "decode_ms": events[1].elapsed_time(events[2])}

    def decode(self, z, overlap_pixels=64, fade_pixels=None):
        plan = SlabPlan.create(z.shape[-1], overlap_pixels, self.ratio, fade_pixels)
        slabs = plan.slices(z)
        # Make caller-produced activations visible to both worker streams.
        # CUDA input must be on rank zero; reject hidden source-device timing.
        if z.is_cuda and z.device != self.devices[0]:
            raise ValueError("CUDA input must start on rank zero")
        for device in set(self.devices):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        futures = [self.pool.submit(self._rank_decode, rank, slab)
                   for rank, slab in enumerate(slabs)]
        # Drain both futures even if one decoder fails, before any next call.
        values, errors = [], []
        for future in futures:
            try:
                values.append(future.result())
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
        left, right = values[0][0], values[1][0]
        with torch.cuda.device(self.devices[0]), torch.cuda.stream(self.streams[0]), torch.inference_mode():
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
            moved = right.to(self.devices[0])
            events[1].record()
            out = stitch_slabs(left, moved, plan)
            events[2].record()
            events[2].synchronize()
        metadata = plan.metadata(z.shape)
        metadata.update(end_to_end_ms=(time.perf_counter() - start) * 1000,
                        rank_stages=[v[1] for v in values],
                        output_transfer_ms=events[0].elapsed_time(events[1]),
                        output_transfer_bytes=right.numel() * right.element_size()
                        if self.devices[0] != self.devices[1] else 0,
                        transfer_kind="PyTorch cross-device copy; physical transport unverified"
                        if self.devices[0] != self.devices[1] else "same-device alias; no PCIe transfer",
                        stitch_ms=events[1].elapsed_time(events[2]),
                        output_shape=list(out.shape),
                        logical_devices=[str(d) for d in self.devices],
                        physical_gpu_count=len(set(self.devices)))
        return out, metadata
