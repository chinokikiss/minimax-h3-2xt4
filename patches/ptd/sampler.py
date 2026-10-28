# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab.
# SPDX-License-Identifier: Apache-2.0
"""PTD sampler adapter. Native model/sampler implementations are not replaced."""
from __future__ import annotations
import math
from typing import Callable
import torch
from .schedule import T_MAX

def _time(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= T_MAX:
        raise ValueError(f"PTD shared time must lie in [0, {T_MAX}]")
    return value


def _streams(value):
    if getattr(value, "is_nested", False):
        streams = tuple(value.unbind())
        if len(streams) != 2:
            raise ValueError("PTD expects exactly video and audio latent streams")
        return streams
    if not isinstance(value, torch.Tensor):
        raise TypeError("PTD latents must be a Tensor or ComfyUI NestedTensor")
    return (value,)


def _map(value, operation: Callable):
    streams = _streams(value)
    result = [operation(stream) for stream in streams]
    return type(value)(result) if getattr(value, "is_nested", False) else result[0]


def _pair(left, right, operation: Callable):
    a, b = _streams(left), _streams(right)
    if len(a) != len(b) or bool(getattr(left, "is_nested", False)) != bool(getattr(right, "is_nested", False)):
        raise ValueError("model output must retain the latent stream structure")
    if any(x.shape != y.shape for x, y in zip(a, b)):
        raise ValueError("model output shape differs from the latent state")
    result = [operation(x, y) for x, y in zip(a, b)]
    return type(left)(result) if getattr(left, "is_nested", False) else result[0]


def initial_state(noise):
    """Pure-noise PTD start: both clocks equal .999, with fp32 state carry.

    Multiplication occurs in float64, matching Clock.latents. Do not call this
    on the already scaled state passed to a bare sampling function.
    """
    return _map(noise, lambda stream: (stream.to(torch.float64) * T_MAX).to(torch.float32))


def denoised_to_velocity(state, denoised, time: float):
    """Recover shared-clock mean velocity from ComfyUI's x0 convention."""
    time = _time(time)
    if time == 0.0:
        raise ValueError("cannot recover a mean velocity at the clean endpoint")
    return _pair(state, denoised, lambda x, x0: (x.float() - x0.float()) / time)


def _validate_initial_inputs(model, extra_args) -> None:
    if extra_args.get("denoise_mask") is not None:
        raise ValueError("PTD sampling does not support denoise masks or partial denoising")
    latent_image = getattr(model, "latent_image", None)
    if latent_image is not None and any(bool(torch.count_nonzero(x)) for x in _streams(latent_image)):
        raise ValueError("PTD requires an empty generated latent; supply image conditions through H3 conditioning")


@torch.no_grad()
def sample_ptd(model, x, sigmas, extra_args=None, callback=None, disable=None, *, controller):
    """The supplied PDD-head reference recurrence, exactly eight expert calls."""
    del disable
    expected = controller.sigmas
    if (not isinstance(sigmas,torch.Tensor) or sigmas.ndim != 1
            or sigmas.dtype not in (torch.float32,torch.float64)
            or sigmas.shape != expected.shape
            or not torch.equal(sigmas.detach().cpu(),expected.cpu().to(sigmas.dtype))):
        raise ValueError('PTD needs the complete unmodified eight-hop trained grid')
    extra_args = dict(extra_args or {})
    _validate_initial_inputs(model,extra_args)
    raw = getattr(model,'noise',None)
    state = initial_state(raw) if raw is not None else _map(x,lambda value:value.float())
    _pair(state,x,lambda a,b:a)
    first = _streams(state)[0]
    if first.ndim == 0 or first.shape[0] < 1 or any(v.shape[0] != first.shape[0] or v.device != first.device for v in _streams(state)):
        raise ValueError('PTD latent streams need a common nonempty batch and device')
    grid = expected.to(device=first.device)
    for hop in range(controller.hops):
        time = float(grid[hop])
        with controller.activate(hop,time):
            denoised = model(state,grid[hop].expand(first.shape[0]),**extra_args)
        velocity = denoised_to_velocity(state,denoised,time)
        if callback is not None:
            callback(dict(x=state,i=hop,sigma=grid[hop],sigma_hat=grid[hop],denoised=denoised))
        delta = (grid[hop+1]-grid[hop]).float()
        state = _pair(state,velocity,lambda value,flow:value.float()+delta*flow.float())
    return state


def make_sampler(controller):
    from comfy.samplers import KSAMPLER
    return KSAMPLER(sample_ptd,extra_options={'controller':controller})
