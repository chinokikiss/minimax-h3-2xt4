# SPDX-License-Identifier: Apache-2.0
"""PTD/PDD sampling schedule, shared verbatim with ``minimax_h3_ptd.py``.

A PTD checkpoint is trained against one MiniMax-H3 schedule discretised into ``pdd_num_steps``
intervals (32 as released). An ``nfe``-hop student collapses that grid into ``nfe`` Euler steps,
each consuming ``block_size = pdd_num_steps // nfe`` consecutive intervals. The per-interval
output heads are fused with the normalised step sizes of the block the hop covers.

The video and audio streams run the same grid under different shifts (12.0 / 3.0 as released),
so each has its own fusion plan.
"""

from __future__ import annotations

import torch

# The released checkpoints keep the PDD schedule; all three are recorded in safetensors metadata.
DEFAULT_PDD_NUM_STEPS = 32
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0
# Upper endpoint of MiniMax-H3's shared clock, matching ptd_comfy.sampling.T_MAX.
T_MAX = 0.999


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    return shift * sigma / (1 + (shift - 1) * sigma)


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    """Ascending grid ``0 = t_0 < ... < t_N = 1`` of one MiniMax-H3 schedule."""
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(shift, sigma)


def pdd_sampling_plan(step_sizes: torch.Tensor, start: int, block_size: int) -> torch.Tensor:
    """Mean velocity of one block, which an Euler step over the block boundaries consumes."""
    plan = torch.zeros(1, step_sizes.shape[0], dtype=step_sizes.dtype, device=step_sizes.device)
    span = step_sizes[start : start + block_size].sum()
    plan[0, start : start + block_size] = step_sizes[start : start + block_size] / span
    return plan


def block_size(num_steps: int, num_hops: int) -> int:
    """Intervals consumed per hop; the metadata field may be stale, so derive it."""
    if num_steps % num_hops:
        raise ValueError(f"pdd_num_steps={num_steps} not divisible by ptd_num_hops={num_hops}.")
    return num_steps // num_hops


def hop_plans(
    num_hops: int,
    num_steps: int = DEFAULT_PDD_NUM_STEPS,
    video_shift: float = DEFAULT_VIDEO_SHIFT,
    audio_shift: float = DEFAULT_AUDIO_SHIFT,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """``(video_plan, audio_plan)`` per hop, in hop order."""
    size = block_size(num_steps, num_hops)
    video_steps = pdd_time_grid(float(video_shift), num_steps).diff()
    audio_steps = pdd_time_grid(float(audio_shift), num_steps).diff()
    return tuple(
        (
            pdd_sampling_plan(video_steps, hop * size, size).float(),
            pdd_sampling_plan(audio_steps, hop * size, size).float(),
        )
        for hop in range(num_hops)
    )


def audio_head_corrections(sigmas: torch.Tensor, audio_sigma, audio_sigma_prime) -> tuple[float, ...]:
    """Per-hop factor turning upstream's audio clock *tangent* into this checkpoint's *chord*.

    ComfyUI-PTD integrates both streams on the shared video clock and converts the audio head to
    that clock by multiplying it by ``audio_sigma_prime(sigma)``, the tangent at the hop start. These
    checkpoints' audio head is PDD's block-mean velocity on the audio grid, which Diffusers (and
    training) apply over the hop's full audio interval: ``x_a += (sigma_a - sigma_a_next) * u_a``.
    Matching that needs the chord ratio ``(sigma_a - sigma_a_next) / (sigma - sigma_next)``, so the
    head is pre-scaled by ``chord / tangent`` and upstream's hook then supplies the tangent.

    ``audio_sigma`` / ``audio_sigma_prime`` are upstream's clock functions, passed in so this module
    stays free of ComfyUI-PTD imports. The final hop (to 0) gets ``audio_sigma(s) / s``, which makes
    the x0 extrapolation land exactly on ``x_a + sigma_a * u_a`` as in Diffusers.
    """
    grid = [float(s) for s in sigmas]
    corrections = []
    for start, stop in zip(grid[:-1], grid[1:]):
        chord = (audio_sigma(start) - audio_sigma(stop)) / (start - stop)
        corrections.append(chord / audio_sigma_prime(start))
    return tuple(corrections)


def hop_sigmas(
    num_hops: int,
    num_steps: int = DEFAULT_PDD_NUM_STEPS,
    video_shift: float = DEFAULT_VIDEO_SHIFT,
    device=None,
) -> torch.Tensor:
    """Descending noise grid of ``num_hops + 1`` entries, one knot per hop plus the endpoint.

    The knots are the video-stream block boundaries of the trained 32-interval grid, expressed
    on MiniMax-H3's shared clock whose upper endpoint is ``T_MAX`` rather than 1. That rescaling
    is what makes PTD's analytic audio clock agree with this repository's two shifts: substituting
    ``t = shifted_sigma(12, s)`` into ``audio_sigma`` gives ``t*T/(4T - 3t)``, which at ``T = 1``
    is exactly ``shifted_sigma(3, s)``. The 3/12 shift ratio is upstream's ``AUDIO_RELATIVE_SHIFT``.

    Diffusers reaches the same hop count as ``num_inference_steps = nfe + 1`` because its
    scheduler counts the terminal sigma; that terminal value is the last entry here.

    UNVERIFIED: the reference pipeline never builds these knots itself — the MiniMax-H3 scheduler
    does — so the ``T_MAX`` endpoint is derived, not observed. Confirm against a Diffusers run
    before trusting output quality (see this pack's README).
    """
    size = block_size(num_steps, num_hops)
    noise = shifted_sigma(
        float(video_shift), torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    )
    sigmas = torch.tensor(
        [T_MAX * float(noise[hop * size]) for hop in range(num_hops)] + [0.0], dtype=torch.float64
    )
    return sigmas if device is None else sigmas.to(device)
