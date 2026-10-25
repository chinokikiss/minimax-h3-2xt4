"""Reference-sensitive prefetch and zero-payload lazy VAE ownership."""
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI')]
import torch
from patches import staged_vae as staged
from patches.h3_reference_plan import encoder_names
from patches.comfy_tp_runtime import PhysicalTPService
from patches.host_weight_cache import HostBudget


def graph(**refs):
    return {'v':dict(class_type='MiniMaxH3ParallelVAELoader',inputs={'vae_name':'video.safetensors'}),
        'a':dict(class_type='MiniMaxH3ParallelVAELoader',inputs={'vae_name':'audio.safetensors'}),
        'ref':dict(class_type='MiniMaxH3ReferenceToVideo',inputs={'vae':['v',0],'audio_vae':['a',0],**refs})}


class ReferencePlan(unittest.TestCase):
    def test_no_reference_or_decode_only_does_not_select_encoders(self):
        self.assertEqual(encoder_names(graph(ref_image_size='match')),set())
        value=graph();value.pop('ref')
        value['decode']=dict(class_type='VAEDecode',inputs={'vae':['v',0],'samples':['latent',0]})
        self.assertEqual(encoder_names(value),set())

    def test_image_audio_and_mixed_references(self):
        self.assertEqual(encoder_names(graph(ref_image_1=['image',0])),{'video.safetensors'})
        self.assertEqual(encoder_names(graph(ref_audios={'ref_audio_1':['audio',0]})),{'audio.safetensors'})
        self.assertEqual(encoder_names(graph(**{'ref_images.ref_image_1':['image',0],
            'ref_audios.ref_audio_1':['audio',0]})),{'video.safetensors','audio.safetensors'})

    def test_video_soundtrack_requires_matching_video_and_video_vae(self):
        video={'ref_videos':{'ref_video_1':['frames',0]}}
        self.assertEqual(encoder_names(graph(**video)),{'video.safetensors'})
        self.assertEqual(encoder_names(graph(ref_video_audio_1=['audio',0])),set())
        self.assertEqual(encoder_names(graph(**video,ref_video_audio_2=['audio',0])),{'video.safetensors'})
        self.assertEqual(encoder_names(graph(**video,ref_video_audio_1=['audio',0])),{'video.safetensors','audio.safetensors'})
        value=graph(**video,ref_video_audio_1=['audio',0]);value['ref']['inputs'].pop('vae')
        self.assertEqual(encoder_names(value),set())

    def test_te_only_references_and_guides(self):
        value=graph(ref_image_1=['image',0],ref_audio_1=['audio',0])
        value['ref']['inputs'].pop('vae');value['ref']['inputs'].pop('audio_vae')
        self.assertEqual(encoder_names(value),set())
        value['ref']=dict(class_type='MiniMaxH3AddGuide',inputs={'audio_vae':['a',0],'audio':['wave',0]})
        self.assertEqual(encoder_names(value),{'audio.safetensors'})
        value['ref']=dict(class_type='MiniMaxH3ImageToVideo',inputs={'vae':['v',0]})
        self.assertEqual(encoder_names(value),set())
        value['ref']['inputs']['last_frame']=['image',0]
        self.assertEqual(encoder_names(value),{'video.safetensors'})

    def test_idle_selection_tracks_reference_type_changes(self):
        service=PhysicalTPService.__new__(PhysicalTPService);service.staged_vaes=weakref.WeakSet()
        service.idle_vae_names=None;service.lock=threading.RLock()
        class VAE:
            def __init__(self,name):self.reader=SimpleNamespace(path=Path(name));self.encoder_used=False;self.releases=0
            def release(self):self.releases+=1
        video=VAE('video.safetensors');audio=VAE('audio.safetensors')
        service.register_staged_vae(video);service.register_staged_vae(audio)
        self.assertEqual(service._idle_vaes(),[])
        video.encoder_used=True;self.assertEqual(service._idle_vaes(),[video])
        service.configure_idle_vaes(graph(ref_image_1=['image',0]));self.assertEqual(service._idle_vaes(),[video])
        service.configure_idle_vaes(graph(ref_audio_1=['wave',0]));self.assertEqual(service._idle_vaes(),[audio])
        self.assertGreater(video.releases,0)
        service.configure_idle_vaes(graph());self.assertEqual(service._idle_vaes(),[])


class LazyComponents(unittest.TestCase):
    def test_registration_loads_no_component_and_decode_can_be_first_use(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vae.safetensors';path.write_bytes(b'header-fixture')
            stat=path.stat();service=SimpleNamespace(config=SimpleNamespace(),lock=threading.RLock(),
                budget=HostBudget(100*2**20,reserve_bytes=0),register_staged_vae=lambda v:None)
            class Native:
                device='cpu';first_stage_model=None;patcher=None;size=None
                def encode(self,x):return x.clone()
                def decode(self,x):return x.clone()
                def model_size(self):return 4
            def model(reader,kind,stage,cancel=None):
                self.assertIsNone(vae.vae.first_stage_model)
                return torch.nn.Linear(1,1).eval().requires_grad_(False)
            for kind,sentinels in [('video',{'decoder.transformer_blocks.0.scale1':{},'encoder.down.5.block.0.conv1.weight':{}}),
                                   ('audio',{'pre_block.attn.zero_k_bias':{}})]:
                header={name:dict(data_offsets=[0,4]) for name in sentinels}
                header.update({'encoder.w':dict(data_offsets=[0,4]),'decoder.w':dict(data_offsets=[4,8])})
                reader=SimpleNamespace(path=path,signature=(stat.st_size,stat.st_mtime_ns),header=header)
                with patch.object(staged,'CheckpointReader',return_value=reader),patch.object(staged,'native_shell',return_value=Native()), \
                        patch.object(staged,'load_component',side_effect=model) as load, \
                        patch('comfy.model_patcher.CoreModelPatcher',new=lambda m,**kw:SimpleNamespace(model=m)), \
                        patch('comfy.model_management.archive_model_dtypes'),patch('comfy.model_management.unload_model_and_clones'):
                    vae=staged.StagedVAE(path,service)
                    self.assertIsNone(vae.stage);self.assertEqual(vae.model_size(),0);load.assert_not_called()
                    self.assertEqual(service.budget.snapshot()['reserved_bytes'],0)
                    vae.throw_exception_if_invalid()
                    vae.decode(torch.ones(1));self.assertEqual(vae.stage,'decode');self.assertFalse(vae.encoder_used)
                    self.assertEqual(load.call_args.args[2],'decode')
                    vae.decode(torch.ones(1));self.assertEqual(load.call_count,1)
                    vae.encode(torch.ones(1));self.assertTrue(vae.encoder_used);self.assertEqual(vae.stage,'encode')
                    self.assertEqual([e['stage'] for e in vae.events],['decode','encode'])
                    vae.release();self.assertEqual(service.budget.snapshot()['reserved_bytes'],0)


if __name__=='__main__':unittest.main()
