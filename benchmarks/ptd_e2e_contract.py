"""The authorized long request, with only the trained PTD clock/adapter changed."""
import hashlib
import json
from pathlib import Path

PROMPT_SHA = 'e2ebe6a75563a0184ef5fa99299acef24ee3170855774761c44d6714fb2f7754'
REFERENCE_SHAS = ('e5b1710572f2587f7927e65b9e5f8f742bc435ca388ae23a16880fa6bf542383',
                  'cd4cbfd7137b80e85e64ada86ffd93e512b874de84cb4291fc0278b747461e03')
RELEASE = dict(repo='taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs',
    revision='d1f88c89c352e7751b56b5f32c7b657fb3e0f2c5',
    file='MiniMax-H3-Ref2VA-PTD-RefAsymOPD-8Step.safetensors', bytes=11112893024,
    sha256='6796802b9150110a3cd7f66891c9e6190502a432c30b0fa1aa299724cfc15cc3')


def from_previous(previous):
    plan = dict(previous)
    plan.pop('turbo_lora')
    plan.update(name='ptd_anime_ref2va_1152x672_243f_8hop',
        workload_contract='ptd_anime_long_8hop', sampler='ptd', scheduler='trained_ptd',
        video_shift=12., audio_shift=3., ptd_release=RELEASE,
        allow_pruned_adaln=True, auxiliary_layout='monolithic',
        hop_handoff='queue', future_cache=True, prefetch_refiner=True,
        expected_packed_tokens=59134,
        source_note='Authorized PTD E2E; original anime prompt/images/geometry/noise seed; '
                    'trained PTD clock and adapter replace Turbo/Euler. Pruned-base approximation.')
    validate(plan)
    return plan


def validate(plan):
    fixed = dict(width=1152, height=672, frames=243, fps=24, steps=8, cfg=1.,
        seed=20261020, negative_prompt='', sampler='ptd', scheduler='trained_ptd',
        video_shift=12., audio_shift=3., offline_lora=True, disk_baked_lora=True,
        workload_contract='ptd_anime_long_8hop', expected_packed_tokens=59134,
        ref_image_size='match', hop_handoff='queue', auxiliary_layout='monolithic')
    for key, expected in fixed.items():
        if plan.get(key) != expected:
            raise ValueError('PTD request contract mismatch: ' + key)
    for key, expected in dict(allow_pruned_adaln=True, future_cache=True).items():
        if plan.get(key) is not expected:
            raise ValueError('PTD requires an explicit switch: ' + key)
    if type(plan.get('prefetch_refiner',True)) is not bool:
        raise ValueError('P1 refiner prefetch requires an explicit boolean')
    if ('turbo_lora' in plan or plan.get('lora_stack') or plan.get('ptd_release') != RELEASE
            or hashlib.sha256(plan['prompt'].encode()).hexdigest() != PROMPT_SHA
            or plan.get('prompt_sha256') != PROMPT_SHA
            or tuple(row['sha256'] for row in plan['reference_images']) != REFERENCE_SHAS):
        raise ValueError('PTD input identity or offline adapter mismatch')
    return dict(prompt_sha256=PROMPT_SHA, reference_sha256=list(REFERENCE_SHAS),
        geometry=[plan[k] for k in ('width','height','frames')], layers=50, hops=8,
        pruned_adaln_approximation=True, auxiliary_layout='monolithic', full_pipeline=True)


def resource_plan(header, assets, vaes, *, free_bytes, host_limit):
    """Conservative disk estimate before downloading real model/adapter payloads."""
    import math
    blocks = sum(info['data_offsets'][1]-info['data_offsets'][0]
                 for name, info in header.items() if name.startswith('blocks.'))
    auxiliary = sum(math.prod(info['shape'])*4 for name, info in header.items()
        if name.endswith(('.weight','.bias')) and name.startswith(
            ('token_refiner.blocks.','final_layer.video_out.','final_layer.audio_out.')))
    bank_bound = 8*(math.ceil(blocks*1.02)+auxiliary)
    base_bytes = sum(row['bytes'] for row in assets['objects'].values())
    base_bytes += sum(row['bytes'] for row in assets['auxiliary'].values())
    needed = bank_bound+base_bytes+RELEASE['bytes']+sum(vaes[k]['bytes'] for k in ('video','audio'))+12*2**30
    result = dict(required_free_bytes=needed, free_bytes=free_bytes,
        estimated_bank_bytes=bank_bound, host_limit_bytes=host_limit,
        cache_bytes=4*2**30, host_bytes=20*2**30, reserve_bytes=9*2**30,
        weight_bytes=6*2**30, pinned_bytes=6*2**30, bounded_gpu_experts=True)
    if free_bytes < needed:
        raise OSError('Insufficient disk for full eight-hop E2E: ' + json.dumps(result))
    if host_limit < 27*2**30:
        raise MemoryError('At least 27 GiB host memory required: ' + json.dumps(result))
    return result
