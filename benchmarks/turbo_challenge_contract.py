"""Authorized regular Turbo canvases, retaining the previous anime inputs."""
import copy
import hashlib

NAME='turbo_1mp_10s'
PROMPT_SHA='e2ebe6a75563a0184ef5fa99299acef24ee3170855774761c44d6714fb2f7754'
CANVASES=frozenset(((1344,768),(1280,704),(1216,704)))


def from_previous(previous):
    from benchmarks.anime_e2e_contract import validate as previous_contract
    previous_contract(previous,previous['reference_images'])
    plan=copy.deepcopy(previous)
    plan.update(name=NAME,workload_contract=NAME,width=1344,height=768,
        export_frames=240,export_audio_samples=320000,export_duration_seconds=10.0,
        source_note='One authorized regular Turbo 1MP challenge; same anime prompt/references/seed, '
                    'offline disk bake, current production defaults, no PTD or online LoRA')
    validate(plan)
    return plan


def validate(plan):
    from benchmarks.anime_e2e_contract import validate as previous_contract
    width,height=plan.get('width'),plan.get('height')
    if (width,height) not in CANVASES:raise ValueError('Unauthorized Turbo challenge canvas')
    fixed=dict(workload_contract=NAME,frames=243,fps=24,
        export_frames=240,export_audio_samples=320000,export_duration_seconds=10.0)
    for key,value in fixed.items():
        if plan.get(key)!=value:raise ValueError('Turbo challenge mismatch: '+key)
    previous_contract(plan|dict(width=1152,height=672),plan['reference_images'])
    if ('ptd_release' in plan or plan.get('lora_stack') or
            hashlib.sha256(plan['prompt'].encode()).hexdigest()!=PROMPT_SHA):
        raise ValueError('Keep the previous regular Turbo prompt and no online/PTD adapter')
    return dict(width=width,height=height,output_megapixels=width*height/1e6,
        native_frames=243,native_duration_seconds=10.125,export_frames=240,
        export_duration_seconds=10.,video_latent_shape=[1,24,72,height//16,width//16],
        generated_video_tokens=72*(height//32)*(width//32),audio_latent_t=405,
        note='Actual TE/reference/packed sequence lengths are captured during execution')


def export_views(pixels,waveform,plan):
    """Crop the final three native frames; preserve full replay latents."""
    validate(plan)
    if pixels.ndim not in (4,5) or (pixels.ndim==5 and pixels.shape[0]!=1):
        raise ValueError('Exactly one decoded video required')
    if list(pixels.shape[-4:])[:3]!=[243,plan['height'],plan['width']]:
        raise ValueError('Native decoded video geometry changed')
    if waveform.ndim not in (2,3):raise ValueError('Native decoded audio required')
    axis=-1 if waveform.shape[-1]>=320000 else -2
    if waveform.shape[axis]<320000:raise ValueError('Audio is shorter than ten seconds')
    return pixels.narrow(pixels.ndim-4,0,240),waveform.narrow(axis,0,320000)
