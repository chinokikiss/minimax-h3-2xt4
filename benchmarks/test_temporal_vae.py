"""Native temporal seams, bounded pairing, dispatch ownership and cleanup."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import nullcontext
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ComfyUI')]
import torch
from benchmarks import test_video_decode_memory as memory_tests
native_fixture = memory_tests.native_fixture
from patches.comfy_sampler import ParallelDecodeHandoff
from patches.comfy_tp_runtime import PhysicalTPService, RankRuntime, worker_reply
from patches.video_decode_planner import make_planner


class TemporalPlanning(unittest.TestCase):
    def test_native_pixel_results_include_tail_padding_blending_batch_and_tiles(self):
        source, raw = native_fixture()
        with patch('comfy.model_management.intermediate_device', return_value=torch.device('cpu')):
            for shape in [(1,24,1,2,2), (1,24,2,2,2), (1,24,7,2,2),
                          (2,24,12,2,2), (1,24,17,3,4), (1,24,107,2,2)]:
                pairs = []
                def pair(clips):
                    pairs.append([None if c is None else c.clone() for c in clips])
                    return tuple(source._adaptive_decode(c) if c is not None else None for c in clips)
                planner = make_planner(source, raw, decode_pair=pair)
                z = torch.randn(shape, generator=torch.Generator().manual_seed(17))
                with self.subTest(shape=shape):
                    reference = source.decode(z)
                    actual = planner.decode(z)
                    self.assertTrue(torch.equal(actual, reference))
                    count = 1 if shape[2] == 1 else source._decode_temporal_chunks(shape[2])[1]
                    self.assertEqual(len(pairs), (count + 1) // 2)
                    self.assertEqual(pairs[-1][1] is None, count % 2 == 1)
                    self.assertIsNone(planner.temporal_pairs)
                    buffer = torch.empty_like(reference)
                    self.assertIs(planner.decode(z, output_buffer=buffer), buffer)
                    self.assertTrue(torch.equal(buffer, reference))

    def test_failure_clears_pair_cache_and_allows_a_new_request(self):
        source, raw = native_fixture()
        fail = True
        def pair(clips):
            if fail:
                raise RuntimeError('injected pair failure')
            return tuple(source._adaptive_decode(c) if c is not None else None for c in clips)
        planner = make_planner(source, raw, decode_pair=pair)
        z = torch.randn(1,24,12,2,2)
        with patch('comfy.model_management.intermediate_device', return_value=torch.device('cpu')):
            with self.assertRaisesRegex(RuntimeError, 'injected pair'):
                planner.decode(z)
            self.assertIsNone(planner.temporal_pairs)
            fail = False
            self.assertTrue(torch.equal(planner.decode(z), source.decode(z)))
        self.assertNotIn('_adaptive_decode', source.__dict__)

    def test_rank_one_temporal_payload_is_not_silently_discarded(self):
        value = torch.tensor([7.])
        self.assertTrue(torch.equal(worker_reply(value, 1, 'video_temporal_pair'), value))
        self.assertIsNone(worker_reply(value, 1, 'video_pixels'))

    def test_parent_service_returns_both_independent_owners(self):
        service = PhysicalTPService.__new__(PhysicalTPService)
        service.lock, service.closed, service.sequence = threading.RLock(), False, 0
        service.connections = [Mock(), Mock()]
        first, second = torch.tensor([1.]), torch.tensor([2.])
        service._receive = Mock(side_effect=[(1, first), (1, second)])
        results = service.call('video_temporal_pair', (first, second))
        self.assertEqual(len(results), 2)
        self.assertTrue(torch.equal(results[0], first) and torch.equal(results[1], second))

    def test_worker_owns_only_its_clip_and_empty_tail_never_decodes(self):
        z = torch.zeros(1,24,7,2,2, dtype=torch.float16)
        runtime = RankRuntime.__new__(RankRuntime)
        runtime.rank, runtime.device = 1, torch.device('cpu')
        runtime.sample_active = False
        runtime.readers = {'dit': SimpleNamespace()}
        runtime.config = SimpleNamespace(idle_residency=False)
        runtime.vae_model, runtime.vae_kind, runtime.vae_parallelism = object(), 'video', 'temporal'
        runtime.vae_temporal_planner = SimpleNamespace(_adaptive_decode=Mock(return_value=z))
        # Dispatch's device/inference scope uses CPU normally.
        with patch('torch.cuda.device', side_effect=lambda device: nullcontext()):
            self.assertIsNone(runtime.dispatch('video_temporal_pair', ((z, None),), {}))
            runtime.vae_temporal_planner._adaptive_decode.assert_not_called()
            out = runtime.dispatch('video_temporal_pair', ((z + 1, z),), {})
        self.assertTrue(torch.equal(out, z))
        runtime.vae_temporal_planner._adaptive_decode.assert_called_once()


class TemporalHandoff(memory_tests.ParentOwnership):
    def test_temporal_handoff_uses_both_replies_and_restores_native_parent(self):
        seen = []
        def command(name, *args):
            seen.append(name)
            if name == 'vae_prepare':
                self.assertEqual(args[-1], 'temporal')
                return dict(accepted=True, admission={})
            if name == 'video_temporal_pair':
                return tuple(self.source._adaptive_decode(c) if c is not None else None for c in args[0])
        self.service.call.side_effect = command
        self.vae.decode = lambda samples_in: self.vae.first_stage_model.decode(samples_in)
        wrapper = ParallelDecodeHandoff(self.vae, self.service, 'fixture', 'temporal')
        z = torch.randn(1,24,12,2,2)
        with patch('patches.parallel_vae.query_partitioning_status', side_effect=AssertionError('Query gate must not run')):
            self.assertTrue(torch.equal(wrapper.decode(z), self.source.decode(z)))
        self.assertEqual(wrapper.last_strategy, 'native_temporal_pairs')
        self.assertIs(self.vae.first_stage_model, self.source)
        self.assertIn('vae_release', seen)
        self.assertNotIn('video_pixels', seen)


if __name__ == '__main__':
    unittest.main()
