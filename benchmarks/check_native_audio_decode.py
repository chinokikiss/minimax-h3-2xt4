"""Tiny trained Audio VAE Native-only check; no TE/DiT, sampler or Kaggle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI')]
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('D:/minimaxH3/minimax_h3_audio_vae_fp32_clean.safetensors'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=sys.argv[:1]
    if args.output.exists():raise FileExistsError(args.output)
    from patches.legacy_runtime import validate_source,validate_packages
    validate_source(ROOT/'ComfyUI');validate_packages()
    import comfy.sd
    import comfy.utils
    from patches.comfy_sampler import ParallelDecodeHandoff
    commands=[]
    def command(name,*values,**kwargs):
        if name not in ('decode_start','decode_complete'):
            raise AssertionError('Audio must not prepare workers or send latents: '+name)
        commands.append(name)
    service=SimpleNamespace(lock=threading.RLock(),closed=False,call=command)
    report=dict(status='running',scope='Local trained Audio VAE component only; no full H3 or performance qualification',
        hardware=torch.cuda.get_device_name(),torch=torch.__version__,checkpoint=str(args.checkpoint),
        input_shape=[1,32,2,16],seed=20261021,cases=[])
    try:
        with torch.inference_mode():
            vae=comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(args.checkpoint)),
                device=torch.device('cuda:0'),dtype=torch.float32)
            samples=torch.randn(report['input_shape'],generator=torch.Generator().manual_seed(report['seed']))
            vae.decode(samples)  # Warm the original Native path.
            reference=vae.decode(samples).cpu()
            for mode in ('auto','native'):
                wrapper=ParallelDecodeHandoff(vae,service,args.checkpoint,mode)
                actual=wrapper.decode(samples_in=samples).cpu()
                row=dict(mode=mode,strategy=wrapper.last_strategy,output_shape=list(actual.shape),
                    dtype=str(actual.dtype),finite=bool(torch.isfinite(actual).all()),
                    byte_exact=torch.equal(actual,reference),max_abs=float((actual-reference).abs().max()))
                if not row['byte_exact'] or not row['finite'] or row['strategy']!='native':raise AssertionError(row)
                if 'decode' in vae.first_stage_model.__dict__:raise AssertionError('Native decode was overridden')
                report['cases'].append(row)
        if any(name in sys.modules for name in ('patches.audio_channel_decode','patches.multi_gpu_audio_vae')):
            raise AssertionError('Historical audio parallel module was imported')
        report['worker_commands']=commands
        report['audio_worker_dispatch']=False
        report['status']='passed'
    except Exception as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:
        sources=['patches/comfy_sampler.py','patches/comfy_tp_runtime.py','patches/parallel_vae.py',
            'custom_nodes/minimax_h3_tp2/__init__.py','benchmarks/check_native_audio_decode.py']
        report['source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
