"""Exact 1MP/10s reference identity replayed at 1280x704 with trained PTD v2 + KC."""
import hashlib

WIDTH, HEIGHT, FRAMES, FPS = 1280, 704, 243, 24
PROMPT_SHA = 'e2ebe6a75563a0184ef5fa99299acef24ee3170855774761c44d6714fb2f7754'
REFERENCE_SHAS = (
    'e5b1710572f2587f7927e65b9e5f8f742bc435ca388ae23a16880fa6bf542383',
    'cd4cbfd7137b80e85e64ada86ffd93e512b874de84cb4291fc0278b747461e03',
)


def from_previous(previous):
    from copy import deepcopy
    plan = deepcopy(previous)
    plan.pop('turbo_lora', None)
    plan.pop('geometry_retry_provenance', None)
    plan.update(
        name='H3_PTD_V2_E2E_09MP_10S_8HOP_KC_20261103',
        workload_contract='ptd_v2_kc_long_e2e', width=WIDTH, height=HEIGHT,
        sampler='ptd', scheduler='trained_ptd', video_shift=12.0, audio_shift=3.0,
        base_weights='prebaked_eight_hop_ptd_v2_bank_not_turbo_lora',
        allow_pruned_adaln=True, auxiliary_layout='split_v2',
        hop_handoff='queue', future_cache=True, prefetch_refiner=True,
        operators=dict(
            te_attention='kitchen_causal', dit_attention='kitchen',
            dit_attention_schedule='', video_attention='kitchen',
        ),
        source_note='Exact prior 1MP/10s anime prompt, both references, seed, frames, fps, '
                    'TE and VAE; resized to 1280x704. Eight trained PTD v2 experts '
                    'replace ordinary Turbo LoRA and Euler/Simple; trained 12/3 shifts. '
                    'All DiT 8 NFEs KC INT8, TE Kitchen causal, Video VAE Kitchen, '
                    'Audio VAE Native. Pruned AdaLN explicitly approximate.',
    )
    validate(plan)
    return plan


def validate(plan):
    from benchmarks.ptd_e2e_contract import RELEASE
    fixed = dict(width=WIDTH, height=HEIGHT, frames=FRAMES, fps=FPS, steps=8, cfg=1,
        seed=20261020, negative_prompt='', sampler='ptd', scheduler='trained_ptd',
        video_shift=12., audio_shift=3., offline_lora=True, disk_baked_lora=True,
        workload_contract='ptd_v2_kc_long_e2e', ref_image_size='match',
        hop_handoff='queue', auxiliary_layout='split_v2',
        export_frames=240, export_audio_samples=320000, export_duration_seconds=10)
    for name, expected in fixed.items():
        if plan.get(name) != expected:
            raise ValueError('Long PTD v2 KC input contract differs: '+name)
    if (plan.get('allow_pruned_adaln') is not True
            or plan.get('future_cache') is not True
            or plan.get('prefetch_refiner') is not True):
        raise ValueError('PTD v2 queue and approximation flags must remain qualified')
    if (plan.get('lora_stack') or 'turbo_lora' in plan
            or plan.get('ptd_release') not in (None, RELEASE)
            or plan.get('prompt_sha256') != PROMPT_SHA
            or hashlib.sha256(plan['prompt'].encode('utf-8')).hexdigest() != PROMPT_SHA
            or tuple(x.get('sha256') for x in plan.get('reference_images', ())) != REFERENCE_SHAS):
        raise ValueError('Previous 1MP reference, prompt or PTD bank identity changed')
    if plan.get('operators') != dict(te_attention='kitchen_causal',
                                       dit_attention='kitchen',
                                       dit_attention_schedule='',
                                       video_attention='kitchen'):
        raise ValueError('All 8 PTD DiT NFEs must use KC; reject Sol route')
    if WIDTH%32 or HEIGHT%32 or WIDTH*HEIGHT != 901120:
        raise ValueError('Expected exact 0.901MP SM75-compatible canvas')
    return dict(output_megapixels=WIDTH*HEIGHT/1e6, frames=FRAMES,
        exported_frames=240, exported_duration_seconds=10.0,
        trained_hops=8, dit_layers=50, video_latent_shape=[1,24,72,44,80],
        generated_video_tokens=72*22*40, attention='all_kc',
        prompt_sha256=PROMPT_SHA, references=list(REFERENCE_SHAS),
        pruned_adaln_approximation=True)
