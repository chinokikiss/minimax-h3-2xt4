"""Opt-in PTD v2 loader/adapter and composite merge contract tests."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from patches.ptd.composite_lora import CompositeLoRAStore
from patches.ptd.nodes import MiniMaxH3TP2PTDBankLoader
from patches.ptd.runtime import PTDOptions, WorkerAdapter
from patches.comfy_tp_runtime import RuntimeConfig
import benchmarks.test_tp2_bundle as bundle_fixture
import benchmarks.test_ptd_bank_v2 as v2_fixture


class CompositeTerms(unittest.TestCase):
    def test_zero_strength_ordinary_targets_are_not_advertised(self):
        fixture = bundle_fixture.Bundle('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from patches.tp2_bundle import BundleReader
        from patches.h3_lora import H3LoRAStore
        base = BundleReader(fixture.descriptor)
        inactive = H3LoRAStore(((str(fixture.lora), 0.),), base)
        self.assertEqual(inactive.targets, set())
        self.assertTrue(inactive.sources)  # Preserve schema and identity checks.
        self.assertEqual(inactive.terms('blocks.0.adaln_proj.linear'), [])
        inactive.check()
        active = H3LoRAStore(((str(fixture.lora), 0.),
                              (str(fixture.lora), .65)), base)
        self.assertIn('blocks.0.adaln_proj.linear', active.targets)
        self.assertEqual(len(active.terms('blocks.0.adaln_proj.linear')), 1)
        with patch('patches.ptd.composite_lora.HopStore') as hop:
            hop.return_value.targets = {'blocks.0.attn.qkv_proj'}
            hop.return_value.terms.return_value = []
            mixed = CompositeLoRAStore(object(), 0, base, ((str(fixture.lora), 0.),))
            self.assertEqual(mixed.targets, {'blocks.0.attn.qkv_proj'})
            self.assertNotIn('blocks.0.adaln_proj.linear', mixed.targets)
            mixed.check()

    def test_one_pass_ptd_native_bytes_and_optional_independent_audit(self):
        import os
        from patches.tp2_bundle import BundleReader
        from patches.offline_lora import merged_weight
        from patches.offline_lora_reference import verify_cached_weight, verify_weight
        from patches.ptd.bake_v2 import _merge_verified_once
        fixture = bundle_fixture.Bundle('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        base = BundleReader(fixture.descriptor)
        target = 'blocks.0.attn.qkv_proj'
        ptd_term = (torch.ones(2, 512), torch.ones(12, 2), .5)
        with patch('patches.ptd.composite_lora.HopStore') as hop:
            hop.return_value.targets = {target}
            hop.return_value.terms.return_value = [ptd_term]
            store = CompositeLoRAStore(object(), 2, base, ((str(fixture.lora), .75),))
            with patch.dict(os.environ, {'PTD_V2_INDEPENDENT_VERIFY': '0'}), \
                    patch.object(store, 'terms', wraps=store.terms) as terms:
                q, scale = _merge_verified_once(base, store, target, 'cpu')
                self.assertEqual(terms.call_count, 1)
            legacy_q, legacy_scale = merged_weight(base, store, target, 'cpu')
            self.assertTrue(torch.equal(q, legacy_q))
            self.assertTrue(torch.equal(scale, legacy_scale))
            self.assertTrue(verify_weight(base, store, target, q, scale, 'cpu'))
            with patch.dict(os.environ, {'PTD_V2_INDEPENDENT_VERIFY': '1'}), \
                    patch('patches.offline_lora_reference.verify_weight', wraps=verify_weight) as strict:
                strict_q, strict_scale = _merge_verified_once(base, store, target, 'cpu')
                strict.assert_called_once()
            self.assertTrue(torch.equal(q, strict_q))
            self.assertTrue(torch.equal(scale, strict_scale))
            _, _, ref = merged_weight(base, store, target, 'cpu', return_reference=True)
            corrupted = q.clone()
            corrupted.view(-1)[0] = -128 if int(corrupted.view(-1)[0]) != -128 else 127
            with self.assertRaisesRegex(AssertionError, 'cached reference'):
                verify_cached_weight(target, corrupted, scale, ref)

    def test_one_pass_dense_refiner_matches_independent_reference(self):
        from patches.tp2_bundle import BundleReader
        from patches.h3_lora import H3LoRAStore
        from patches.offline_lora import merged_weight
        from patches.offline_lora_reference import verify_weight
        from patches.ptd.bake_v2 import _merge_verified_once
        fixture = bundle_fixture.Bundle('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        base = BundleReader(fixture.descriptor)
        target = 'token_refiner.blocks.0.attn.out_proj'
        ordinary = H3LoRAStore(((str(fixture.lora), .65),), base)
        result, scale = _merge_verified_once(base, ordinary, target, 'cpu')
        native, native_scale = merged_weight(base, ordinary, target, 'cpu')
        self.assertIsNone(scale)
        self.assertIsNone(native_scale)
        self.assertTrue(torch.equal(result, native))
        self.assertTrue(verify_weight(base, ordinary, target, result, scale, 'cpu'))

    def test_ptd_then_ordinary_lora_one_quantization_contract(self):
        fixture = bundle_fixture.Bundle('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from patches.tp2_bundle import BundleReader
        base = BundleReader(fixture.descriptor)
        ptd_term = (torch.ones(2, 512), torch.ones(12, 2), 0.5)
        with patch('patches.ptd.composite_lora.HopStore') as hop:
            hop.return_value.targets = {'blocks.0.attn.qkv_proj'}
            hop.return_value.terms.return_value = [ptd_term]
            store = CompositeLoRAStore(object(), 2, base, ((str(fixture.lora), .75),))
            terms = store.terms('blocks.0.attn.qkv_proj')
            self.assertEqual(len(terms), 2)
            self.assertIs(terms[0], ptd_term)
            self.assertEqual(terms[1][0].shape[1], 512)
            self.assertEqual(terms[1][2], .75 * (3. / 2.))
            self.assertIn('blocks.0.adaln_proj.linear', store.targets)
            from patches.offline_lora import merged_weight
            from patches.offline_lora_reference import verify_weight
            q, scale = merged_weight(base, store, 'blocks.0.attn.qkv_proj', 'cpu')
            self.assertTrue(verify_weight(base, store, 'blocks.0.attn.qkv_proj', q, scale, 'cpu'))
            self.assertEqual(q.dtype, torch.int8)
            self.assertEqual(q.shape, (12, 512))
            store.check()
            hop.return_value.check.assert_called()

    def test_nonexistent_ordinary_target_does_not_introduce_fake_terms(self):
        with patch('patches.ptd.composite_lora.HopStore') as hop:
            hop.return_value.targets = {'blocks.0.mlp.fc1'}
            hop.return_value.terms.return_value = []
            store = CompositeLoRAStore(object(), 0, object())
            self.assertIsNone(store.ordinary)
            self.assertEqual(store.terms('condition_proj'), [])


class V2Loader(unittest.TestCase):
    def test_sidecar_node_exposes_no_runtime_lora_and_calls_original_loader(self):
        cls = MiniMaxH3TP2PTDBankLoader
        spec = cls.INPUT_TYPES()
        self.assertIn('bank_manifest', spec['required'])
        self.assertNotIn('diffusion_model', spec['required'])
        self.assertNotIn('lora_stack', spec['optional'])
        self.assertNotIn('lora_mode', spec['optional'])
        self.assertEqual(spec['optional']['ptd_io_workers'][1]['default'], 1)
        self.assertFalse(spec['optional']['ptd_retire_page_cache'][1]['default'])
        with patch('patches.ptd.bank_v2.PTDBankV2'), \
                patch('custom_nodes.minimax_h3_tp2.MiniMaxH3TP2Loader.load',
                      return_value=('model', 'clip', 'session')) as legacy:
            result = cls().load('te.safetensors', 'bank_v2.json',
                                part_roots_json='{"hop_00":"C:/mnt/hop_00"}',
                                ptd_io_workers=4, ptd_retire_page_cache=True)
            self.assertEqual(result[:3], ('model', 'clip', 'session'))
            self.assertEqual(result[3], str(Path('bank_v2.json').resolve()))
            self.assertEqual(legacy.call_args.kwargs['lora_mode'], 'online')
            self.assertEqual(legacy.call_args.kwargs['ptd_part_roots'],
                             (('hop_00', 'C:/mnt/hop_00'),))
            self.assertEqual(legacy.call_args.kwargs['ptd_io_workers'], 4)
            self.assertTrue(legacy.call_args.kwargs['ptd_retire_page_cache'])
            with self.assertRaisesRegex(ValueError, 'includes all'):
                cls().load('te', 'bank_v2.json', lora_stack=(('lora', 1.),))
            with self.assertRaisesRegex(ValueError, 'JSON object'):
                cls().load('te', 'bank_v2.json', part_roots_json='[]')

    def test_v2_worker_adapter_does_not_require_original_base(self):
        fixture = v2_fixture.SelfContainedV2('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        path = fixture.v2_manifest()
        from patches.ptd.bank_v2 import PTDBankV2
        reader = PTDBankV2(path, allow_fixture=True).reader(0)
        fixture.source.descriptor.rename(fixture.source.descriptor.with_suffix('.offline'))
        config = RuntimeConfig('te.safetensors', str(path), layers=1,
                               ptd_io_workers=4, ptd_retire_page_cache=True)
        runtime = SimpleNamespace(config=config, readers={'dit': reader},
                                  device=torch.device('cpu'), rank=0)
        options = PTDOptions(enabled=True, allow_fixture=True)
        adapter = WorkerAdapter(runtime, str(path), options)
        self.assertEqual(adapter.hop, 0)
        self.assertEqual(adapter.runtime.readers['dit'].hop, 0)
        self.assertNotIn('base', vars(adapter.bank))
        self.assertEqual(adapter.bank.io_workers, 4)
        self.assertTrue(adapter.bank.retire_page_cache)


if __name__ == '__main__':
    unittest.main(verbosity=2)
