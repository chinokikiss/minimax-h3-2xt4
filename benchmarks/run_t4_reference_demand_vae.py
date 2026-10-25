"""Real T4 component loading and Native outputs; no sampler/TP worker claim."""
import argparse
import gc
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import weakref
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
from patches.comfy_tp_runtime import PhysicalTPService
from patches.host_weight_cache import HostBudget
from patches.h3_tensor_flow import TensorFlowPolicy
from patches.staged_vae import StagedVAE,component_names,component_payload


class Parent:
    register_staged_vae=PhysicalTPService.register_staged_vae
    configure_idle_vaes=PhysicalTPService.configure_idle_vaes
    _idle_vaes=PhysicalTPService._idle_vaes
    release_staged_encoders=PhysicalTPService.release_staged_encoders
    def __init__(self):
        self.lock=threading.RLock();self.staged_vaes=weakref.WeakSet();self.idle_vae_names=None
        self.config=SimpleNamespace(tensor_flow=TensorFlowPolicy('fp16'),idle_reserve_bytes=2**30)
        self.budget=HostBudget(16*2**30,reserve_bytes=0)


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize()
    return out,1000*(time.perf_counter()-start)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=sys.argv[:1]
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in (0,1))
    torch.cuda.set_device(0);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(ROOT/'ComfyUI')
    import comfy.model_management as management
    import comfy.sd,comfy.utils
    def native_vae(path,dtype):
        return comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(path)),device=torch.device('cuda:0'),dtype=dtype)
    service=Parent();report=dict(status='running',scope='Actual T4 GPU0 trained Native VAE components with real host leases; lightweight parent, no persistent TP workers or full H3',records=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2))
    vaes={};baselines={};encodings={};pixels={}
    try:
        for kind in ('video','audio'):
            vae,ms=timed(lambda:StagedVAE(args.assets/(kind+'_vae.safetensors'),service));vaes[kind]=vae
            assert vae.stage is None and vae.reader.logical_read_bytes==0 and vae.reservation==0
            report['records'].append(dict(case='constructor',kind=kind,component_payload_bytes=0,host_lease_bytes=0,metadata_ms=ms));save()
        assert service.budget.snapshot()['reserved_bytes']==0
        def graph(image=False,audio=False):
            ref={'vae':['v',0],'audio_vae':['a',0]}
            if image:ref['ref_image_1']=['image',0]
            if audio:ref['ref_audio_1']=['audio',0]
            return {'v':dict(class_type='MiniMaxH3ParallelVAELoader',inputs={'vae_name':'video_vae.safetensors'}),
                'a':dict(class_type='MiniMaxH3ParallelVAELoader',inputs={'vae_name':'audio_vae.safetensors'}),
                'ref':dict(class_type='MiniMaxH3ReferenceToVideo',inputs=ref)}
        pixels['video']=torch.rand(1,32,32,3,generator=torch.Generator().manual_seed(419))
        pixels['audio']=torch.rand(1,1600,2,generator=torch.Generator().manual_seed(419))*2-1
        for kind in ('video','audio'):
            baselines[kind]=native_vae(args.assets/(kind+'_vae.safetensors'),torch.float16 if kind=='video' else torch.float32)
            encodings[kind]=baselines[kind].encode(pixels[kind].clone()).cpu()
        for label,kinds in (('image_only',('video',)),('audio_only',('audio',)),('mixed',('video','audio'))):
            service.configure_idle_vaes(graph(image='video' in kinds,audio='audio' in kinds))
            assert {v.kind for v in service._idle_vaes()}==set(kinds)
            entries=[]
            for kind in kinds:
                vae=vaes[kind];before=vae.reader.logical_read_bytes
                actual,cold_ms=timed(lambda:vae.encode(pixels[kind].clone()))
                assert torch.equal(actual.cpu(),encodings[kind]) and torch.isfinite(actual).all()
                loaded=vae.reader.logical_read_bytes-before
                assert loaded==component_payload(vae.reader,component_names(vae.reader,kind,'encode'))
                samples=[]
                for _ in range(2):warm=vae.encode(pixels[kind].clone());del warm
                for _ in range(3):warm,ms=timed(lambda:vae.encode(pixels[kind].clone()));samples.append(ms);del warm
                entries.append(dict(kind=kind,native_exact=True,read_bytes=loaded,first_load_and_encode_ms=cold_ms,warm_encode_samples_ms=samples))
            for kind in set(vaes)-set(kinds):assert vaes[kind].stage is None
            report['records'].append(dict(case=label,encoders=entries,selected_prefetch=sorted(service.idle_vae_names)))
            service.release_staged_encoders();assert service.budget.snapshot()['reserved_bytes']==0;save()
        service.configure_idle_vaes(graph());assert service._idle_vaes()==[]
        for kind in ('video','audio'):
            vae=vaes[kind];dtype=torch.float16 if kind=='video' else torch.float32
            shape=(1,24,1,2,2) if kind=='video' else (1,32,2,2)
            z=torch.randn(shape,generator=torch.Generator().manual_seed(732)).to(dtype)
            expected=baselines[kind].decode(z.clone()).cpu();before=vae.reader.logical_read_bytes
            actual,cold_ms=timed(lambda:vae.decode(z.clone()))
            assert torch.equal(actual.cpu(),expected) and torch.isfinite(actual).all()
            read=vae.reader.logical_read_bytes-before
            assert read==component_payload(vae.reader,component_names(vae.reader,kind,'decode')) and not hasattr(vae.first_stage_model,'encoder')
            for _ in range(2):warm=vae.decode(z.clone());del warm
            samples=[]
            for _ in range(3):warm,ms=timed(lambda:vae.decode(z.clone()));samples.append(ms);del warm
            report['records'].append(dict(case='decoder_first_after_release',kind=kind,native_exact=True,
                decoder_payload_bytes=read,encoder_payload_bytes=0,first_load_and_decode_ms=cold_ms,warm_decode_samples_ms=samples,
                public_output_shape=list(actual.shape),stage_events=vae.events))
            vae.release();management.unload_model_and_clones(baselines[kind].patcher);save()
            del actual,expected,z;gc.collect();management.soft_empty_cache()
        assert service.budget.snapshot()['reserved_bytes']==0
        report.update(status='passed',host_lease_bytes_after_release=0,limits='Tiny VAE shapes, warm disk state uncontrolled; not deployment decode latency or full TP2 lifecycle qualification')
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        for vae in vaes.values():vae.release()
        save()


if __name__=='__main__':main()
