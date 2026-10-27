"""Video-only admission and parent ownership; no CUDA kernels or checkpoints."""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'ComfyUI')]
import torch
from torch import nn
from patches.comfy_tp_runtime import RuntimeConfig, RankRuntime, GiB
from patches.comfy_sampler import ParallelDecodeHandoff
from patches.video_decode_planner import make_planner, remote_video_planner


def native_fixture():
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    model = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(model)
    for name, value in dict(vae_ratio=16, vae_ratio_t=4, clip_length=17,
            token_drop=3, frame_pre_padding=3, tokens_chunk_size=5, token_overlap=2,
            frame_overlap=5, tiling=True, tile_size=32, tile_overlap_min=16).items():
        setattr(model, name, value)
    model.decoder = nn.Linear(2, 2)
    model.decoder.out_channels = 3
    model.post_quant_conv = nn.Linear(2, 2)
    for name, value in dict(latents_mean=torch.linspace(-.1,.1,24),
            latents_std=torch.linspace(.8,1.2,24),
            pixel_mean=torch.tensor([.485,.456,.406]).view(1,3,1,1,1),
            pixel_std=torch.tensor([.229,.224,.225]).view(1,3,1,1,1)).items():
        model.register_buffer(name, value, persistent=not name.startswith('pixel_'))
    def raw(z):
        return z[:,:3].repeat_interleave(4,2).repeat_interleave(16,3).repeat_interleave(16,4)
    model._decode_pixels = raw
    return model, raw


class DecodePolicy(unittest.TestCase):
    def runtime(self, *, free=12*GiB, host=True, peer=True):
        r = RankRuntime.__new__(RankRuntime)
        r.config = RuntimeConfig('te', 'dit', reserve_bytes=9*GiB)
        r.rank, r.device = 0, torch.device('cpu')
        r.sample_active = False
        r.events = []
        r._drop_vae = Mock()
        r.cache = SimpleNamespace(reserve_staging=Mock(return_value=host),
            budget=SimpleNamespace(finish_staging=Mock(), release=Mock(),snapshot=lambda:{'limit_bytes':16*GiB}))
        r.startup_cache = SimpleNamespace(trim=Mock(), relieve_pressure=Mock())
        def collect(status, op):
            if not peer:status.zero_()
        r.comm = SimpleNamespace(ReduceOp=SimpleNamespace(MIN='min'), all_reduce=Mock(side_effect=collect))
        self.enterContext(patch('torch.cuda.mem_get_info', return_value=(free,15*GiB)))
        self.enterContext(patch('patches.parallel_vae.checkpoint_plan',
            return_value=(SimpleNamespace(identity='fixture'), ['decoder.w'], 4*GiB+256*2**20)))
        self.load = self.enterContext(patch('patches.parallel_vae.load_decode_weights', return_value=nn.Linear(1,1)))
        return r

    def test_decode_reserve_is_separate_and_configurable(self):
        r = self.runtime(free=6*GiB)
        result = r._prepare_vae('fixture','video',GiB//2)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['admission']['required_free_bytes'],5*GiB)
        self.assertEqual(r.config.reserve_bytes,9*GiB)
        r.config = replace(r.config,video_decode_reserve_bytes=5*GiB)
        self.assertFalse(r._prepare_vae('fixture','video',GiB//2)['accepted'])

    def test_local_gpu_refusal_does_not_load_or_reserve_host(self):
        r = self.runtime(free=4*GiB)
        result = r._prepare_vae('fixture','video',GiB//2)
        self.assertIn('VRAM',result['reason'])
        self.assertFalse(result['admission']['gpu_accepted'])
        r.cache.reserve_staging.assert_not_called();self.load.assert_not_called()

    def test_host_refusal_is_distinct_and_keeps_collective_order(self):
        r = self.runtime(host=False)
        result = r._prepare_vae('fixture','video',GiB//2)
        self.assertIn('Host RAM',result['reason'])
        self.assertTrue(result['admission']['gpu_accepted'])
        self.assertEqual(result['admission']['host_budget']['limit_bytes'],16*GiB)
        self.assertEqual(r.comm.all_reduce.call_count,1)
        self.load.assert_not_called()

    def test_peer_refusal_rolls_back_local_staging(self):
        r = self.runtime(peer=False)
        result = r._prepare_vae('fixture','video',GiB//2)
        self.assertIn('another rank',result['reason'])
        r.cache.budget.finish_staging.assert_called_once()
        r.cache.budget.release.assert_called_once();self.load.assert_not_called()

    def test_replica_load_failure_releases_lease_and_records_reason(self):
        r = self.runtime();self.load.side_effect=ValueError('bad replica contract')
        result = r._prepare_vae('fixture','video',GiB//2)
        self.assertIn('bad replica',result['reason'])
        self.assertEqual(r.comm.all_reduce.call_count,2)
        r.cache.budget.release.assert_called_once()
        self.assertFalse(r.events[-1]['accepted'])

    def test_loader_appends_widget_and_restarts_on_policy_change(self):
        import custom_nodes.minimax_h3_tp2 as nodes
        with self.assertRaises(ValueError):RuntimeConfig('te','dit',video_decode_reserve_bytes=-1)
        options=nodes.MiniMaxH3TP2Loader.INPUT_TYPES()['optional']
        # Future controls append after the established widgets.
        self.assertLess(list(options).index('video_decode_reserve_gib'),list(options).index('prefetch_lifecycle'))
        self.assertEqual(options['video_decode_reserve_gib'][1]['default'],2.)
        def service(config):return SimpleNamespace(config=config,closed=False,close=Mock())
        with patch.object(nodes,'_service',None),patch.object(nodes,'PhysicalTPService',side_effect=service), \
                patch.object(nodes,'create_model'),patch.object(nodes,'create_clip'):
            loader=nodes.MiniMaxH3TP2Loader();one=loader.load('te','dit')[2]
            two=loader.load('te','dit',video_decode_reserve_gib=3.)[2]
            one.close.assert_called_once();self.assertEqual(two.config.video_decode_reserve_bytes,3*GiB)
            self.assertIs(two,loader.load('te','dit',video_decode_reserve_gib=3.)[2])


class ParentOwnership(unittest.TestCase):
    def setUp(self):
        self.source,self.raw=native_fixture()
        self.service=SimpleNamespace(lock=threading.RLock(),closed=False,call=Mock())
        self.patcher=SimpleNamespace(model=self.source,patches={},forced_hooks=None)
        self.vae=SimpleNamespace(first_stage_model=self.source,patcher=self.patcher,size=123,
            device=torch.device('cuda:0'),vae_dtype=torch.float16,
            memory_used_decode=lambda *args:1024)
        self.enterContext(patch('comfy.model_management.cuda_device_context',side_effect=lambda *args:nullcontext()))
        self.unload=self.enterContext(patch('comfy.model_management.unload_model_and_clones'))
        self.enterContext(patch('comfy.model_management.archive_model_dtypes'))
        self.enterContext(patch('comfy.model_patcher.CoreModelPatcher',
            side_effect=lambda model,**kwargs:SimpleNamespace(model=model)))
        self.enterContext(patch('torch.cuda.synchronize'))
        self.enterContext(patch('comfy.model_management.intermediate_device',return_value=torch.device('cpu')))

    def test_native_tile_temporal_batch_and_pixel_processing_are_exact(self):
        planner=make_planner(self.source,self.raw)
        self.assertEqual(sum(p.numel() for p in planner.parameters()),0)
        self.assertFalse(hasattr(planner,'post_quant_conv'))
        self.assertFalse(hasattr(planner.decoder,'weight'))
        for shape in ((1,24,1,2,2),(2,24,7,2,2),(1,24,12,4,6)):
            z=torch.randn(shape,generator=torch.Generator().manual_seed(71))
            with self.subTest(shape=shape):
                self.assertTrue(torch.equal(self.source.decode(z),planner.decode(z)))
                self.assertTrue(torch.equal(self.source.decode_tiled(z),planner.decode_tiled(z)))
        for name in ('latents_mean','latents_std','pixel_mean','pixel_std'):
            self.assertNotEqual(getattr(planner,name).data_ptr(),getattr(self.source,name).data_ptr())

    def test_exception_restores_original_model_patcher_and_size(self):
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with remote_video_planner(self.vae,self.service,self.raw) as planner:
                self.assertIs(self.vae.first_stage_model,planner)
                self.assertIs(self.vae.patcher.model,planner)
                self.assertIsNone(self.vae.size)
                raise RuntimeError('injected')
        self.assertIs(self.vae.first_stage_model,self.source)
        self.assertIs(self.vae.patcher,self.patcher);self.assertEqual(self.vae.size,123)
        self.assertEqual(self.unload.call_count,2)

    def handoff(self, *, accepted=True, fail=False):
        self.enterContext(patch('patches.parallel_vae.query_partitioning_status',return_value={'available':True}))
        def command(name,*args):
            if name=='vae_prepare':
                self.assertEqual(sum(p.numel() for p in self.vae.patcher.model.parameters()),0)
                return dict(accepted=accepted,reason='fixture admission',admission={'reserve_bytes':2*GiB})
            if name=='video_pixels':
                if fail:raise RuntimeError('injected remote decode')
                return self.raw(args[0])
        self.service.call.side_effect=command
        def decode(samples_in=None,**kwargs):return self.vae.first_stage_model.decode(samples_in)
        self.vae.decode=decode;self.vae.decode_tiled=lambda samples=None,**kwargs:decode(samples)
        return ParallelDecodeHandoff(self.vae,self.service,'fixture','query')

    def test_parallel_handoff_uses_weightless_parent_and_restores_native(self):
        wrapper=self.handoff();z=torch.randn(1,24,7,2,2)
        expected=self.source.decode(z)
        self.assertTrue(torch.equal(wrapper.decode(samples_in=z),expected))
        self.assertEqual(wrapper.last_strategy,'native_aligned_query')
        self.assertEqual(wrapper.last_admission['reserve_bytes'],2*GiB)
        self.assertIs(self.vae.first_stage_model,self.source)
        self.assertIn('video_pixels',[c.args[0] for c in self.service.call.call_args_list])

    def test_admission_refusal_restores_trained_parent_before_native_fallback(self):
        wrapper=self.handoff(accepted=False);z=torch.randn(1,24,1,2,2)
        self.assertTrue(torch.equal(wrapper.decode(z),self.source.decode(z)))
        self.assertEqual(wrapper.last_strategy,'native')
        self.assertIs(self.vae.first_stage_model,self.source)
        self.assertNotIn('video_pixels',[c.args[0] for c in self.service.call.call_args_list])

    def test_collective_decode_error_propagates_and_releases_replicas(self):
        wrapper=self.handoff(fail=True)
        with self.assertRaisesRegex(RuntimeError,'injected remote'):
            wrapper.decode(torch.randn(1,24,1,2,2))
        self.assertIs(self.vae.first_stage_model,self.source)
        self.assertEqual([c.args[0] for c in self.service.call.call_args_list][-1],'vae_release')

    def test_unoffloadable_parent_falls_back_before_worker_collectives(self):
        wrapper=self.handoff()
        self.source.post_quant_conv.weight=nn.Parameter(torch.empty(2,2,device='meta'))
        z=torch.randn(1,24,1,2,2)
        self.assertTrue(torch.equal(wrapper.decode(z),self.source.decode(z)))
        self.assertEqual(wrapper.last_strategy,'native')
        self.assertIn('offload',wrapper.last_fallback)
        self.assertNotIn('vae_prepare',[c.args[0] for c in self.service.call.call_args_list])


if __name__=='__main__':unittest.main(verbosity=2)
