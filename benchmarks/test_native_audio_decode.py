"""Audio always delegates the complete input to Native; no rank replicas/GPU work."""
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI')]
import torch
from torch import nn
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
from patches.comfy_sampler import ParallelDecodeHandoff
from patches.comfy_tp_runtime import RankRuntime


class NativeAudioDecode(unittest.TestCase):
    def test_auto_and_native_forward_full_stereo_and_tiling_without_rank_prepare(self):
        for mode in ('auto','native'):
            for tiled in (False,True):
                with self.subTest(mode=mode,tiled=tiled):
                    audio=MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
                    nn.Module.__init__(audio)
                    native_decode=audio.decode.__func__
                    calls=[]
                    def command(name,*args,**kwargs):
                        self.assertIn(name,('decode_start','decode_complete'))
                        calls.append(name)
                    service=SimpleNamespace(lock=threading.RLock(),closed=False,call=command)
                    samples=torch.arange(128,dtype=torch.float32).view(1,32,2,2)
                    expected=torch.randn(1,16,2,generator=torch.Generator().manual_seed(7))
                    decode=Mock(return_value=expected)
                    vae=SimpleNamespace(first_stage_model=audio,device='cuda:0',vae_dtype=torch.float32,
                        prepare_decode=Mock(),decode=decode,decode_tiled=decode)
                    wrapper=ParallelDecodeHandoff(vae,service,'unused.safetensors',mode)
                    with patch('torch.cuda.synchronize') as sync, \
                         patch('comfy.model_management.load_models_gpu') as load:
                        if tiled:actual=wrapper.decode_tiled(samples_in=samples,tile_x=4)
                        else:actual=wrapper.decode(samples_in=samples)
                    self.assertIs(actual,expected)
                    self.assertIs(decode.call_args.kwargs['samples_in'],samples)
                    self.assertEqual(decode.call_args.kwargs.get('tile_x'),4 if tiled else None)
                    self.assertEqual(wrapper.last_strategy,'native')
                    self.assertIsNone(wrapper.last_query_partitioning)
                    self.assertEqual(calls,['decode_start','decode_complete'])
                    self.assertIs(audio.decode.__func__,native_decode)
                    self.assertNotIn('decode',audio.__dict__)
                    # Native owns its shape-dependent prepare_decode internally.
                    vae.prepare_decode.assert_not_called();sync.assert_called_once();load.assert_not_called()

    def test_staged_loader_prepares_decoder_before_native_public_call(self):
        from patches.staged_vae import StagedVAE
        audio=MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE);nn.Module.__init__(audio)
        vae=StagedVAE.__new__(StagedVAE)
        vae.vae=SimpleNamespace(first_stage_model=audio,device='cpu',decode=Mock(return_value='native'))
        vae.prepare_decode=Mock()
        service=SimpleNamespace(lock=threading.RLock(),closed=False,call=Mock())
        wrapper=ParallelDecodeHandoff(vae,service,'unused','auto')
        self.assertEqual(wrapper._decode(vae.vae.decode,torch.zeros(1,32,2,2)),'native')
        vae.prepare_decode.assert_called_once();vae.vae.decode.assert_called_once()

    def test_obsolete_worker_requests_decline_without_allocating_or_collectives(self):
        runtime=RankRuntime.__new__(RankRuntime)
        # No runtime/device/cache state: rejection must precede every GPU access.
        result=runtime._prepare_vae('unused','audio',0)
        self.assertFalse(result['accepted']);self.assertIn('single-GPU',result['reason'])
        with self.assertRaisesRegex(RuntimeError,'removed'):
            runtime.dispatch('audio_channels')

    def test_parallel_weight_loaders_reject_audio_before_opening_checkpoint(self):
        from patches.parallel_vae import checkpoint_plan,load_decode_weights
        with self.assertRaisesRegex(ValueError,'video-only'):
            checkpoint_plan('does-not-exist','audio')
        with self.assertRaisesRegex(ValueError,'video-only'):
            load_decode_weights(None,[],'audio','cuda:0')

    def test_old_loader_auto_selection_is_normalized_to_native_for_audio(self):
        spec=importlib.util.spec_from_file_location('native_audio_loader',ROOT/'custom_nodes/minimax_h3_tp2/__init__.py')
        node=importlib.util.module_from_spec(spec);spec.loader.exec_module(node)
        vae=object();session=object()
        paths=SimpleNamespace(get_full_path_or_raise=lambda *args:'audio.safetensors')
        with patch.dict(sys.modules,folder_paths=paths), \
             patch('patches.windowed_tp.CheckpointReader'), \
             patch('patches.staged_vae.checkpoint_kind',return_value='audio'), \
             patch('patches.staged_vae.StagedVAE',return_value=vae):
            for kwargs in ({}, {'decode_mode':'auto'}, {'decode_mode':'temporal'}):
                result,=node.MiniMaxH3ParallelVAELoader().load('audio.safetensors',session,**kwargs)
                self.assertEqual(result.mode,'native');self.assertIs(result.vae,vae)


if __name__=='__main__':unittest.main()
