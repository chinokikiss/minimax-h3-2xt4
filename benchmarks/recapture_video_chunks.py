"""Capture every public decoder chunk on the saved native latent; verify pixels."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'bench_assets/tokyo_ref2va.json').read_text())
os.environ['COMFY_ROOT']=config['comfy_root']
sys.path[:0]=[config['comfy_root'],str(ROOT),str(ROOT/'benchmarks')]
import comfy_kitchen as ck
original_linear=ck.int8_linear
import retest
import comfy.ops
if not hasattr(comfy.ops,'_eager_input_act'):ck.int8_linear=original_linear
import comfy.sd
from patches.windowed_tp import CheckpointReader
from capture_real_attention import AttentionCapture

if __name__=='__main__':
    directory=Path(config['output_directory']);output=directory/'video_chunks'
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config['capture_video_calls']=list(range(1000))
    probe=AttentionCapture(config,output);probe.install()
    vae=comfy.sd.VAE(sd=CheckpointReader(Path(config['model_directory'])/'minimax_h3_video_vae_int8_convrot.safetensors').read(''),
                     device=torch.device('cuda:0'),dtype=torch.float16)
    for i,block in enumerate(vae.first_stage_model.decoder.transformer_blocks):
        original=block.forward
        def scoped(*args,_original=original,_i=i,**kwargs):
            if _i==0:probe.video_call+=1
            with probe.scope('video',_i):return _original(*args,**kwargs)
        block.forward=scoped
    latent=torch.load(directory/'native_video_latent.pt',map_location='cpu',weights_only=True)
    expected=torch.load(directory/'native_pixels.pt',map_location='cpu',weights_only=True)
    with torch.inference_mode():pixels=vae.decode(latent).cpu()
    assert torch.equal(pixels,expected),'Native decoder recapture changed output'
    probe.finish()
    (output/'validation.json').write_text(json.dumps({'bit_equal':True,'decoder_calls':probe.video_call+1,'captures':len(probe.records)},indent=2))
    manifest=json.loads((directory/'capture_manifest.json').read_text())
    manifest['records']=[r for r in manifest['records'] if r['component']!='video']
    manifest['records'] += [{**r,'file':str(Path('video_chunks')/r['file'])} for r in probe.records]
    (directory/'capture_manifest_first_video_chunk.json').write_bytes((directory/'capture_manifest.json').read_bytes())
    (directory/'capture_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('VIDEO_CHUNKS',len(probe.records),probe.video_call+1,flush=True)
