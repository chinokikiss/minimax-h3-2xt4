"""Fixed performance fixture matching the completed 59,134-token TP2 request."""
import hashlib
import torch

TOKENS=59134
SEGMENTS=[(0,2392,'text'),(2392,3151,'ref_img'),(3151,3892,'ref_img'),
          (3892,4702,'audio'),(4702,59134,'video')]
NOISE_SHA=('8278b86d61591cabffe5d88094441d84ace65b3b14751b83e6d46c55a3123f5e',
           'ce577d202b69753b5528d9a9559f88227ec0a38557130e61851f38b03071065e')


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def fixture():
    from comfy.ldm.minimax.model import PackedLayout
    generator=torch.Generator().manual_seed(20261020)
    video=torch.randn((1,24,72,42,72),generator=generator)
    audio=torch.randn((1,32,2,405),generator=generator)
    generator.manual_seed(20261028)
    context=torch.randn((1,2392,5120),generator=generator)*.25
    refs=[dict(kind='image',latent_t=1,latent_h=h,latent_w=w,ref_audio_t=0)
          for h,w in ((46,66),(38,78))]
    layout=PackedLayout(2392,72,42,72,405,refs=refs)
    assert layout.seq_len==TOKENS and layout.segments==SEGMENTS
    payload=dict(layout=layout,refs=refs,seed=20261020,cond_video_latents=[
        torch.randn((1,24,1,r['latent_h'],r['latent_w']),generator=generator)*.1 for r in refs])
    # Frozen tokenizer: each '<Picture N>: ' has 6 tokens; vision blocks
    # expand to 759/741 rows plus their start/end tokens; prompt has 876 rows.
    from comfy.text_encoders.minimax import token_tags_from_embeds_info
    payload['text_token_tags']=token_tags_from_embeds_info(2392,[
        dict(type='image',index=7,size=759),dict(type='image',index=774,size=741)])
    return dict(x=[video,audio],timestep=torch.tensor([1000.]),context=context,
                transformer_options=dict(minimax_h3_sigma_shift_video=6.,minimax_h3_sigma_shift_audio=3.),
                minimax_payload=payload)


def identity(value):
    noise=[digest(t) for t in value['x']]
    return dict(tokens=TOKENS,segments=SEGMENTS,noise_sha256=noise,prior_noise_sha256=list(NOISE_SHA),
        noise_matches_prior_generation=noise==list(NOISE_SHA),
        conditioning_sha256=digest(value['context']),reference_sha256=[digest(t) for t in value['minimax_payload']['cond_video_latents']],
        distribution='Seeded native CPU noise; seeded synthetic text/reference tensors. Original shape/seed; actual RNG hashes are recorded, not assumed portable across Torch builds. Performance-only fixture, not a trajectory or accuracy replay.',
        text_tags='Native reconstructed presentation spans: text[0:6], vision[6:767], text[767:773], vision[773:1516], text[1516:2392]',
        text_tags_sha256=digest(value['minimax_payload']['text_token_tags']))
