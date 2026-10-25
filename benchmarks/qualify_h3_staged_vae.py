"""Tiny trained VAE-only Native comparisons. No TE/DiT or full H3 generation."""
import argparse
import gc
import json
from pathlib import Path
import sys
import os
import threading
from types import SimpleNamespace
import weakref
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI'))]
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,
        default=Path('D:/attention_eval/h3_loading_stages_20261019/trained_vae.json'))
    parser.add_argument('--kind',choices=['video','audio','both'],default='both')
    parser.add_argument('--decoder-first',action='store_true',help='Also verify trained decode without any encoder payload read')
    args=parser.parse_args();sys.argv=sys.argv[:1]
    import comfy.sd,comfy.utils,comfy.model_management as management
    from patches.staged_vae import StagedVAE,component_names,component_payload
    from patches.host_weight_cache import HostBudget
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    service=SimpleNamespace(lock=threading.RLock(),budget=HostBudget(16*2**30,reserve_bytes=0),
        register_staged_vae=lambda vae:None,config=SimpleNamespace(idle_reserve_bytes=2**30),closed=False,
        call=lambda *a,**kw:None)
    rows=[]
    for kind,name in [('video','minimax_h3_video_vae_int8_convrot.safetensors'),('audio','minimax_h3_audio_vae_fp32_clean.safetensors')]:
        if args.kind not in (kind,'both'):continue
        path=Path('D:/minimaxH3')/name;dtype=torch.float16 if kind=='video' else torch.float32
        baseline=comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(path)),device=torch.device('cuda:0'),dtype=dtype)
        staged=StagedVAE(path,service)
        assert staged.stage is None and staged.first_stage_model is None
        assert staged.reader.logical_read_bytes==0
        decoder_first=None;extra_payload=0
        if args.decoder_first:
            from patches.comfy_sampler import ParallelDecodeHandoff
            shape=(1,24,1,2,2) if kind=='video' else (1,32,2,2)
            latent=torch.randn(shape,generator=torch.Generator().manual_seed(732)).to(dtype)
            expected=baseline.decode(latent.clone())
            actual=ParallelDecodeHandoff(staged,service,path,'native').decode(latent.clone())
            assert torch.equal(actual,expected) and torch.isfinite(actual).all(),(kind,'decoder-first differs')
            extra_payload=component_payload(staged.reader,component_names(staged.reader,kind,'decode'))
            assert staged.reader.logical_read_bytes==extra_payload and not hasattr(staged.first_stage_model,'encoder')
            decoder_first=dict(native_exact=True,encoder_payload_bytes=0,decoder_payload_bytes=extra_payload)
            staged.release();del actual,expected,latent
        assert staged.preload_idle(threading.Event())
        assert staged.stage=='encode' and not hasattr(staged.first_stage_model,'decoder')
        assert staged.reader.logical_read_bytes==extra_payload+component_payload(staged.reader,component_names(staged.reader,kind,'encode'))
        pixels=(torch.rand(1,32,32,3,generator=torch.Generator().manual_seed(419)) if kind=='video' else
                torch.rand(1,1600,2,generator=torch.Generator().manual_seed(419))*2-1)
        reference=baseline.encode(pixels.clone());actual=staged.encode(pixels.clone())
        assert torch.equal(actual,reference),(kind,'encode differs',float((actual-reference).abs().max()))
        encoder=weakref.ref(staged.first_stage_model)
        decode_reference=baseline.decode(reference.clone())
        from patches.comfy_sampler import ParallelDecodeHandoff
        # Component lifetime qualification uses a Native decode. Actual query
        # partitioning is checked separately with two logical rank decoders.
        wrapper=ParallelDecodeHandoff(staged,service,path,'native')
        decode_actual=wrapper.decode(actual.clone())
        gc.collect()
        assert encoder() is None and staged.stage=='decode' and not hasattr(staged.first_stage_model,'encoder')
        assert torch.equal(decode_actual,decode_reference),(kind,'decode differs',float((decode_actual-decode_reference).abs().max()))
        assert torch.isfinite(actual).all() and torch.isfinite(decode_actual).all()
        assert staged.reader.logical_read_bytes==extra_payload+sum(component_payload(staged.reader,component_names(staged.reader,kind,s)) for s in ('encode','decode'))
        rows.append(dict(kind=kind,encode='bitwise_pass',decode='bitwise_pass',
            latent_shape=list(actual.shape),output_shape=list(decode_actual.shape),stage_events=staged.events,
            inactive_component_absent=True,encoder_released_before_decoder=True,idle_encoder_gpu_preload=True,
            constructor_payload_bytes=0,decoder_first=decoder_first,
            decode_strategy=wrapper.last_strategy,fallback=wrapper.last_fallback))
        staged.release();assert service.budget.snapshot()['reserved_bytes']==0
        management.unload_model_and_clones(baseline.patcher)
        del baseline,staged,wrapper,actual,reference,decode_actual,decode_reference,pixels
        gc.collect();management.soft_empty_cache()
        print('TRAINED_STAGED_VAE_PASS',kind,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(scope='Tiny trained VAE encode/decode on one local GPU. No full H3 pipeline or T4 performance claim.',
        hardware=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),records=rows),indent=2))


if __name__=='__main__':main()
