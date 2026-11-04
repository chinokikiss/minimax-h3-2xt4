"""15-second regular-Turbo E2E contract; no PTD or attention-backend change.

Reuses the qualified 1MP/10s anime prompt, references, seed, 8-step Turbo
LoRA and Euler/Simple settings. Only canvas and temporal/export geometry change.
"""
import copy
from benchmarks import turbo_challenge_contract as ten_second

NAME = 'turbo_long_15s'
WIDTH, HEIGHT = 1056, 576
FRAMES, FPS = 363, 24
EXPORT_FRAMES = 360
EXPORT_AUDIO_SAMPLES = 480000


def validate(plan):
    fixed = dict(workload_contract=NAME, width=WIDTH, height=HEIGHT,
        frames=FRAMES, fps=FPS, duration_seconds=FRAMES / FPS,
        export_frames=EXPORT_FRAMES, export_audio_samples=EXPORT_AUDIO_SAMPLES,
        export_duration_seconds=15.0, steps=8, sampler='euler',
        scheduler='simple', cfg=1, video_shift=6, audio_shift=3,
        seed=20261020, offline_lora=True, disk_baked_lora=True,
        debug=False, residual_exponent=9)
    for key, expected in fixed.items():
        if plan.get(key) != expected:
            raise ValueError(f'15s ordinary Turbo contract mismatch: {key}')
    # Reuse all immutable regular-Turbo source/prompt/ref/LoRA checks without
    # expanding the historical 10s challenge's permitted canvas list.
    previous = copy.deepcopy(plan)
    previous.update(workload_contract=ten_second.NAME,
        width=1216, height=704, frames=243, fps=24,
        duration_seconds=10.125, export_frames=240,
        export_audio_samples=320000, export_duration_seconds=10.0)
    ten_second.validate(previous)
    if plan.get('ptd_release') or plan.get('lora_stack'):
        raise ValueError('No PTD bank or online adapters in Native Turbo E2E')
    return dict(width=WIDTH, height=HEIGHT, native_frames=FRAMES,
        native_duration_seconds=FRAMES / FPS, export_frames=EXPORT_FRAMES,
        export_duration_seconds=15.0, video_spatial_tokens_per_latent=WIDTH//32*(HEIGHT//32),
        note='The source prompt remains byte-identical to the historical 10s reference')


def export_views(pixels, waveform, plan):
    validate(plan)
    if pixels.ndim not in (4, 5) or (pixels.ndim == 5 and pixels.shape[0] != 1):
        raise ValueError('Exactly one decoded video required')
    if list(pixels.shape[-4:])[:3] != [FRAMES, HEIGHT, WIDTH]:
        raise ValueError('Native decoded 15s video geometry changed')
    if waveform.ndim not in (2, 3):
        raise ValueError('Native decoded audio required')
    axis = -1 if waveform.shape[-1] >= EXPORT_AUDIO_SAMPLES else -2
    if waveform.shape[axis] < EXPORT_AUDIO_SAMPLES:
        raise ValueError('Decoded audio is shorter than 15 seconds')
    return (pixels.narrow(pixels.ndim-4, 0, EXPORT_FRAMES),
            waveform.narrow(axis, 0, EXPORT_AUDIO_SAMPLES))
