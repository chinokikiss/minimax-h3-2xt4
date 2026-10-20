"""Fixed token matrix, with native Ref2VA image/video packing.

Values are reproducible fixtures, not the user's photographs or generated
conditioning. Shapes deliberately realize the requested packed token counts;
they are not estimates of the saved workflow's sequence lengths.
"""
import math
import torch
from fixture_inputs import block_input
from full_models import request as text_request

TE_LENGTHS=(512,2048,4096)
DIT_LENGTHS=(8192,16384,32768)
# (latent T/H/W, context rows, reference video latent T)
SHAPES={68:((2,4,4),32,2),
        8192:((37,18,32),1682,2),
        16384:((37,28,48),2770,2),
        32768:((37,52,52),2734,17)}


def values(shape,seed,device,dtype=torch.float32):
    n=math.prod(shape)
    x=block_input('te',math.ceil(n/5120),seed)['input'].flatten()[:n]
    return x.reshape(shape).to(device=device,dtype=dtype).clone()


def request(kind,length,seed,device):
    if kind=='te':return text_request(kind,length,seed,device)
    from comfy.ldm.minimax.model import PackedLayout
    dims,context_rows,ref_t=SHAPES[length]
    t,h,w=dims
    rh=rw=4 if length==68 else 32
    audio_t=8 if length==68 else 207
    refs=[{'kind':'image','latent_h':rh,'latent_w':rw,
           'latent':values((1,24,1,rh,rw),seed+1,device)},
          {'kind':'video','latent_t':ref_t,'latent_h':rh,'latent_w':rw,
           'ref_audio_t':0,'latent':values((1,24,ref_t,rh,rw),seed+2,device),
           'audio_latent':None}]
    layout=PackedLayout(context_rows,t,h,w,audio_t,refs=refs)
    if layout.seq_len!=length:raise ValueError('Fixture does not realize the prescribed packed sequence')
    tags=torch.ones(context_rows,dtype=torch.int64)
    # Include vision and text AdaLN tags in the precomputed conditioning span.
    chunk=min(128,context_rows//4)
    tags[chunk:2*chunk]=0;tags[3*chunk:4*chunk]=0
    payload={'refs':refs,'cond_video_latents':[r['latent'] for r in refs],
             'text_token_tags':tags,'layout':layout,'seed':seed,
             'visual_cond_noise_aug':.999,'audio_cond_noise_aug':1.0,'audio_scale':1.0}
    video=values((1,24,t,h,w),seed+3,device)
    audio=values((1,32,2,audio_t),seed+4,device)
    context=values((1,context_rows,5120),seed+5,device)
    return ([video,audio],torch.tensor([625.0],device=device),context),{
        'minimax_payload':payload,'transformer_options':{}}


def description(length):
    dims,context_rows,ref_t=SHAPES[length]
    t,h,w=dims
    return {'packed_sequence_length':length,'context_rows':context_rows,
            'target_video_latent_shape':[1,24,t,h,w],
            'target_audio_latent_shape':[1,32,2,8 if length==68 else 207],
            'reference_image':True,'reference_video_latent_frames':ref_t,
            'fixture_values':'counter-generated, half-rounded float32; precomputed conditioning',
            'scope':'one full native Ref2VA denoiser step; vision/VAE encoders and sampling loop excluded',
            'note':'Fixed-token fixture geometry, not the saved workflow token count or a pixel-resolution speed estimate'}


def capture_inputs(path,length,seed=900,device='cuda:0'):
    """Run native trained preprocessing, stop immediately before block zero."""
    from torch import nn
    from full_models import skeleton,forward
    import retest
    class Captured(Exception):pass
    captured={}
    class Capture(nn.Module):
        def forward(self,x,t_emb,mod_segments,rope_freqs,**kwargs):
            captured.update(input=x.detach().cpu(),args=(t_emb.detach().cpu(),mod_segments,rope_freqs.detach().cpu()))
            raise Captured()
    model=skeleton('dit',path,device);model.blocks=nn.ModuleList([Capture()])
    data=request('dit',length,seed,device)
    with torch.inference_mode():
        try:forward('dit',model,data)
        except Captured:pass
    del model,data;retest.cleanup()
    if not captured:raise RuntimeError('Native preprocessing did not reach block zero')
    captured['description']=description(length)
    return captured
