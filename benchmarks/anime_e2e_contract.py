"""Exact input contract for the user-requested eight-step anime REF2VA run."""
import hashlib

EXPECTED = dict(width=1152, height=672, frames=243, fps=24, steps=8,
                sampler='euler', scheduler='simple', cfg=1.0, video_shift=6.0,
                audio_shift=3.0, seed=20261020, offline_lora=True, disk_baked_lora=True)
LORA = 'minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'


def validate(plan, references):
    for name, value in EXPECTED.items():
        if plan.get(name) != value:
            raise ValueError('Requested generation setting changed: ' + name)
    if plan.get('negative_prompt', '') != '' or len(references) != 2:
        raise ValueError('Exactly two ordered images and an empty negative prompt required')
    if [r['sha256'] for r in references] != [
            'e5b1710572f2587f7927e65b9e5f8f742bc435ca388ae23a16880fa6bf542383',
            'cd4cbfd7137b80e85e64ada86ffd93e512b874de84cb4291fc0278b747461e03']:
        raise ValueError('Reference identity/order changed')
    if plan['frames'] % 17 != 5 or plan['frames'] / plan['fps'] != 10.125:
        raise ValueError('Native frame grid/duration mismatch')
    turbo = plan['turbo_lora']
    if (turbo['repository'] != 'lightx2v/Minimax-h3-Turbo' or turbo['file'] != LORA
            or turbo['revision'] != '30f485ed874aa44c8183348ad361c5eabad318da'
            or turbo['sha256'] != '6a56f41ab4229c9dd845b9501bbd475ee57e112d846cf2e819d534a1ae928c5a'
            or turbo.get('strength') != 1.0 or turbo.get('targets') != 208):
        raise ValueError('Requested eight-step offline LoRA identity changed')
    for tag in ('<Picture 1>', '<Picture 2>', '<Subject 1>', '<Subject 2>', '<Subject 3>'):
        if tag not in plan['prompt']:
            raise ValueError('Prompt lost reference/subject tag: ' + tag)
    if hashlib.sha256(plan['prompt'].encode()).hexdigest() != plan['prompt_sha256']:
        raise ValueError('Prompt checksum mismatch')
    return dict(width=1152, height=672, frames=243, video_latent_t=72,
                audio_latent_t=405, duration_seconds=10.125, output_megapixels=.774144)
