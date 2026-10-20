"""Separate scheduling from decoder batch-shape roundoff on one physical GPU."""
import hashlib
import json
from pathlib import Path
import torch
from diagnose_audio_channels import ROOT,MiniMaxH3AudioVAE,load_file,metric
from patches.multi_gpu_audio_vae import MultiGPUAudioVAEDecoder


def main():
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    path=Path('D:/minimaxH3/minimax_h3_audio_vae_fp32_clean.safetensors')
    state=load_file(str(path));models=[]
    for rank in (0,1):
        model=MiniMaxH3AudioVAE();model.load_state_dict(state,strict=True)
        models.append(model.to('cuda:0',dtype=torch.float32).eval())
    del state
    wrapper=MultiGPUAudioVAEDecoder(models[0],dev0='cuda:0',dev1='cuda:0',dtype=torch.float32)
    wrapper.vae_dev1=models[1];wrapper.is_multi_gpu=True
    wrapper.stream0=torch.cuda.Stream();wrapper.stream1=torch.cuda.Stream()
    result=dict(scope='two trained replicas, original channel wrapper, two logical CUDA ranks; no T4 latency claim',
                physical_gpu_count=1,logical_ranks=2,cases=[],
                source_sha256={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),ROOT/'patches/multi_gpu_audio_vae.py')})
    for seconds in (5,10,20):
        cpu=torch.load(ROOT/f'semantic_results/audio_saved_references/input_{seconds}s.pt',weights_only=True)
        z=cpu.cuda()
        with torch.inference_mode():
            native=models[0].decode(z).cpu()
            seq=torch.cat([models[ch].decode(z[:,:,ch:ch+1]).cpu() for ch in (0,1)],1)
            parallel,_=wrapper.decode_channel_parallel(z);parallel=parallel.cpu()
            repeat,_=wrapper.decode_channel_parallel(z);repeat=repeat.cpu()
        row=dict(seconds=seconds,parallel_vs_native=metric(parallel,native),parallel_vs_sequential_split=metric(parallel,seq),repeat_vs_parallel=metric(repeat,parallel))
        result['cases'].append(row);print('LOGICAL AUDIO',row,flush=True)
    result['completed']=True
    result['passed']=all(r['parallel_vs_sequential_split']['bit_exact'] and r['repeat_vs_parallel']['bit_exact'] for r in result['cases'])
    (ROOT/'semantic_results/audio_logical_ranks.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
