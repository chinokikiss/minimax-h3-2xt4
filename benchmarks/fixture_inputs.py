"""Portable deterministic synthetic activations; no runtime-dependent RNG."""
import numpy as np
import torch


def block_input(kind, length, seed, curve=None):
    shape = (1, length, 5120) if kind == 'te' else (length, 5376)
    values = np.arange(int(np.prod(shape)), dtype=np.uint32)
    values += np.uint32(seed)
    values *= np.uint32(1664525)
    values += np.uint32(1013904223)
    values ^= values >> np.uint32(13)
    values *= np.uint32(2246822519)
    values ^= values >> np.uint32(16)
    # Exactly representable 24-bit fractions; uniform synthetic activations.
    x = torch.from_numpy(((values >> np.uint32(8)).astype(np.float32) * np.float32(2 ** -23)
                         - np.float32(1)).reshape(shape)).half()
    x = x.float() if kind == 'dit' else x
    data = {'input': x, 'seed': seed, 'distribution': 'counter-generated uniform [-1,1), half-rounded'}
    if kind == 'dit':
        data['t_emb'] = curve[384:385].float().clone()
        data['segments'] = [(0, length // 4, 0), (length // 4, length // 2, 1), (length // 2, length, 2)]
        angle = torch.linspace(0, 8, length).view(1, length, 1, 1) * torch.linspace(.01, 1, 48).view(1, 1, 1, 48)
        c, s = angle.cos(), angle.sin()
        data['rope'] = torch.stack([c, -s, s, c], -1).reshape(1, length, 1, 48, 2, 2)
    return data
