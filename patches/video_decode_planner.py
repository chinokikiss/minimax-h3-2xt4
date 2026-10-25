"""Weight-free parent for the frozen Native Video VAE public decode planner.

Only the two TP workers execute learned decoder layers. The parent retains
Native temporal/spatial planning, normalization, blending and output writes.
The trained parent model stays on CPU, available after the scoped lease ends.
"""
from contextlib import contextmanager

import torch
from torch import nn


def make_planner(model, decode_pixels):
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

    class Planner(MiniMaxH3VideoVAE):
        def __init__(self):
            # Native construction would allocate the complete encoder/decoder.
            nn.Module.__init__(self)
            for name in ('vae_ratio', 'vae_ratio_t', 'clip_length', 'token_drop',
                         'frame_pre_padding', 'tokens_chunk_size', 'token_overlap',
                         'frame_overlap', 'tiling', 'tile_size', 'tile_overlap_min'):
                setattr(self, name, getattr(model, name))
            self.decoder = nn.Module()
            self.decoder.out_channels = model.decoder.out_channels
            for name in ('latents_mean', 'latents_std', 'pixel_mean', 'pixel_std'):
                value = getattr(model, name)
                self.register_buffer(name, value.detach().to('cpu', copy=True),
                    persistent=name not in model._non_persistent_buffers_set)
            self.decode_pixels = decode_pixels

        def _decode_pixels(self, z):
            return self.decode_pixels(z)

    return Planner().eval().requires_grad_(False)


@contextmanager
def remote_video_planner(vae, service, decode_pixels, *, template=None):
    """Swap only this VAE's patcher/model; never patch global model management.

    A staged VAE can supply a metadata-only template, avoiding a learned parent
    component and its host lease. Existing trained VAEs retain their CPU model.
    A refusal exits this context before invoking Native fallback. A patcher
    that cannot offload yields None, before any worker collective is started.
    """
    import comfy.model_management as management
    import comfy.model_patcher
    from patches.comfy_idle_lifecycle import preserve_te
    from patches.staged_vae import StagedVAE

    parent = vae.vae if isinstance(vae, StagedVAE) else vae
    model, patcher, size = parent.first_stage_model, parent.patcher, parent.size
    if patcher is not None:
        with management.cuda_device_context(parent.device), preserve_te(service):
            management.unload_model_and_clones(patcher)
    # Offload must precede admission. An unmanaged GPU model must not silently
    # defeat the two-replica contract or be moved outside its patcher's ownership.
    if template is None and any(t.device.type != 'cpu' for t in (*model.parameters(), *model.buffers())):
        yield None
        return
    planner = make_planner(template if template is not None else model, decode_pixels)
    planner_patcher = comfy.model_patcher.CoreModelPatcher(planner,
        load_device=parent.device, offload_device=torch.device('cpu'))
    management.archive_model_dtypes(planner)
    parent.first_stage_model, parent.patcher, parent.size = planner, planner_patcher, None
    staged=isinstance(vae,StagedVAE)
    if staged:vae.remote_decode_active=True
    try:
        yield planner
    finally:
        try:
            with management.cuda_device_context(parent.device), preserve_te(service):
                management.unload_model_and_clones(planner_patcher)
        finally:
            parent.first_stage_model, parent.patcher, parent.size = model, patcher, size
            if staged:vae.remote_decode_active=False
