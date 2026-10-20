"""Validate real-tensor kernel math; profile kernels and decode generated audio."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import wave
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from patches.tilelang_sparse_attention import prepare,make_routes,execute,reference,get_kernel


def audio_outputs(config,directory):
    os.environ['COMFY_ROOT']=config['comfy_root']
    sys.path[:0]=[config['comfy_root'],str(ROOT/'benchmarks')]
    import comfy_kitchen as ck
    original=ck.int8_linear
    import retest
    import comfy.ops
    if not hasattr(comfy.ops,'_eager_input_act'):ck.int8_linear=original
    import comfy.sd
    from patches.windowed_tp import CheckpointReader
    path=Path(config['model_directory'])/'minimax_h3_audio_vae_fp32_clean.safetensors'
    vae=comfy.sd.VAE(sd=CheckpointReader(path).read(''),device=torch.device('cuda:0'),dtype=torch.float32)
    results=[];baseline=None
    for mode,folder in [('native',directory),('dense',directory/'runs/dense'),('sparse',directory/'runs/sparse')]:
        latent=torch.load(folder/'native_audio_latent.pt',map_location='cpu',weights_only=True)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():value=vae.decode(latent).cpu()
        seconds=time.perf_counter()-start
        torch.save(value,folder/'native_waveform.pt')
        waveform=value[0]
        if waveform.shape[0]<8:waveform=waveform.T
        array=waveform.clamp(-1,1).mul(32767).round().short().numpy()
        with wave.open(str(folder/'generated_audio.wav'),'wb') as out:
            out.setnchannels(array.shape[-1]);out.setsampwidth(2);out.setframerate(getattr(vae,'audio_sample_rate',32000));out.writeframes(array.tobytes())
        if baseline is None:baseline=value
        diff=value.float()-baseline.float()
        results.append({'mode':mode,'decode_seconds':seconds,'shape':list(value.shape),
                        'relative_l2':(diff.norm()/baseline.float().norm().clamp_min(1e-12)).item()})
    (directory/'audio_comparison.json').write_text(json.dumps(results,indent=2))
    del vae;retest.cleanup()


if __name__=='__main__':
    config=json.loads((ROOT/'bench_assets/tokyo_ref2va.json').read_text());directory=Path(config['output_directory'])
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    errors=json.loads((directory/'attention_errors.json').read_text())
    proof={'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'cases':[]}
    profile_dir=directory/'kernel_profiles';profile_dir.mkdir(exist_ok=True)
    for component in ('te','dit','video'):
        candidates=[]
        for row in errors['records']:
            if row['capture']['component']!=component:continue
            for chunk in row['modes']['dense_int8_fp16_pv']:
                for head,err in zip(chunk['head_ids'],chunk['error']['relative_l2']):candidates.append((err,row['capture'],head))
        _,capture,head=max(candidates,key=lambda x:x[0])
        data=torch.load(directory/capture['file'],map_location='cpu',weights_only=True)
        group=data['q'].shape[1]//data['k'].shape[1]
        q=data['q'][:,head:head+1].cuda();k=data['k'][:,head//group:head//group+1].cuda();v=data['v'][:,head//group:head//group+1].cuda()
        p=prepare(q.half(),k.half(),v.half());routes=make_routes(p,'dense');causal=component=='te'
        b,h,n,d=q.shape;sm=10*torch.cuda.get_device_capability()[0]+torch.cuda.get_device_capability()[1]
        for pv8 in [False,True]:
            actual,lse=execute(p,routes,False,pv8,causal=causal)
            expected,expected_lse=reference(p,routes,False,pv8,causal=causal)
            rel=((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)).item()
            le=(lse-expected_lse).abs().max().item()
            assert rel<=.002 and le<=.003,(component,pv8,rel,le)
            proof['cases'].append({'component':component,'capture':capture['file'],'head':head,'pv8':pv8,'implementation_relative_l2':rel,'lse_max_abs':le})
        kernel=get_kernel(b,h,n,d,False,False,sm=sm,causal=causal)
        kernel.export_ptx(str(profile_dir/f'{component}.ptx'))
        raw=(p.q,p.qi,p.ki,p.v,p.qs,p.ks,p.kc,p.vc,p.counts,routes,p.vi,p.vs)
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                with torch.profiler.record_function(component+'/native_fp32_sdpa'):F.scaled_dot_product_attention(q,k,v,is_causal=causal)
                with torch.profiler.record_function(component+'/tilelang_int8_kernel'):kernel(*raw)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(profile_dir/f'{component}.json'))
            (profile_dir/f'{component}_summary.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
        except Exception:
            (profile_dir/f'{component}_profiling_error.txt').write_text(traceback.format_exc())
        del data,q,k,v,p,routes,kernel;torch.cuda.empty_cache()
    (directory/'real_kernel_validation.json').write_text(json.dumps(proof,indent=2))
    audio_outputs(config,directory)
