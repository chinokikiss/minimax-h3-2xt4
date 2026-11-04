"""CPU-only bounded PTD v2 bake, part-assembly and atomic admission tests.

The synthetic PTD fixture has zero LoRA targets; independent target math is
covered by CompositeTerms. This test qualifies storage and lifecycle only.
"""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import benchmarks.test_ptd_bank_v2 as fixture_module
from patches.ptd.bake_v2 import bake_hop_part, bake_shared_part, assemble_bank_v2
from patches.ptd.bank_v2 import PTDBankV2


class EmptySyntheticPTD:
    def __init__(self, path, base):
        self.path = Path(path)
        self.hops, self.steps, self.blocks, self.refiners = 8, 32, 1, 1
        self.rank, self.alpha = 2, 2.
        self.targets = {}

    def check(self):
        return None


class BoundedParts(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.SelfContainedV2('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.source.root
        self.heads = self.root / 'tiny_heads.safetensors'
        save_file(dict(
            **{'proj_out.weight': torch.ones(32, 3, 8),
               'proj_out.bias': torch.zeros(32, 3),
               'audio_proj_out.weight': torch.ones(32, 2, 8),
               'audio_proj_out.bias': torch.zeros(32, 2)}), str(self.heads))

    def _bake(self):
        base = self.fixture.source.descriptor
        root = self.root / 'bounded'
        root.mkdir()
        with patch('patches.ptd.bake_v2.PTDCheckpoint', EmptySyntheticPTD):
            bake_shared_part(base, self.heads, root / 'shared', layers=1, allow_fixture=True)
            for hop in range(8):
                bake_hop_part(base, self.heads, hop, root / f'hop_{hop:02d}',
                              devices=('cpu',), layers=1, allow_fixture=True)
        return root

    def test_bounded_eight_hop_assembly_without_base(self):
        root = self._bake()
        manifest = root / 'bank_v2.json'
        path = assemble_bank_v2(manifest, root / 'shared',
                                [root / f'hop_{h:02d}' for h in range(8)], allow_fixture=True)
        self.fixture.source.descriptor.rename(self.fixture.source.descriptor.with_suffix('.offline'))
        bank = PTDBankV2(path, allow_fixture=True)
        self.assertEqual(bank.layers, 1)
        self.assertEqual(bank.reader(5).read_tensor('final_layer.video_out.weight').shape, (3, 8))
        self.assertEqual(bank.reader(7).read_tensor('untouched').tolist(), [19, 5])
        with self.assertRaises(FileExistsError):
            assemble_bank_v2(manifest, root / 'shared',
                             [root / f'hop_{h:02d}' for h in range(8)], allow_fixture=True)

    def test_one_hop_joint_ordinary_lora_bakes_full_width_once(self):
        from patches.tp2_bundle import BundleReader, join_matrix
        from patches.h3_lora import H3LoRAStore
        from patches.offline_lora import merged_weight
        from patches.windowed_tp import CheckpointReader
        base = self.fixture.source.descriptor
        ordinary = ((str(self.fixture.source.lora), .65),)
        root = self.root / 'joint_one_hop'
        root.mkdir()
        with patch('patches.ptd.bake_v2.PTDCheckpoint', EmptySyntheticPTD):
            bake_shared_part(base, self.heads, root / 'shared',
                             ordinary_loras=ordinary, layers=1, allow_fixture=True)
            receipt = bake_hop_part(base, self.heads, 3, root / 'hop_03',
                                    ordinary_loras=ordinary, devices=('cpu',),
                                    layers=1, allow_fixture=True)
        import json
        info = json.loads(receipt.read_text())
        self.assertIn('blocks.0.attn.qkv_proj', info['verified_targets'])
        self.assertIn('token_refiner.blocks.0.attn.out_proj', info['verified_targets'])
        first = CheckpointReader(root / 'hop_03' / 'dit.block00.rank0.safetensors')
        second = CheckpointReader(root / 'hop_03' / 'dit.block00.rank1.safetensors')
        actual = join_matrix(first.read_tensor('qkv_proj.weight'),
                             second.read_tensor('qkv_proj.weight'), 'qkv_proj')
        reader = BundleReader(base)
        store = H3LoRAStore(ordinary, reader)
        expected, _ = merged_weight(reader, store, 'blocks.0.attn.qkv_proj', 'cpu')
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_strength_ordinary_lora_does_not_requantize_untouched_targets(self):
        from patches.tp2_bundle import BundleReader, join_matrix
        from patches.windowed_tp import CheckpointReader
        base = self.fixture.source.descriptor
        ordinary = ((str(self.fixture.source.lora), 0.0),)
        root = self.root / 'zero_strength_hop'
        root.mkdir()
        with patch('patches.ptd.bake_v2.PTDCheckpoint', EmptySyntheticPTD), \
                patch('patches.ptd.bake_v2._merge_verified_once',
                      side_effect=AssertionError('Inactive LoRA target must not be baked')):
            bake_shared_part(base, self.heads, root / 'shared',
                             ordinary_loras=ordinary, layers=1, allow_fixture=True)
            receipt_path = bake_hop_part(base, self.heads, 0, root / 'hop_00',
                                         ordinary_loras=ordinary, devices=('cpu',),
                                         layers=1, allow_fixture=True)
        import json
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        self.assertEqual(receipt['verified_targets'], [])
        self.assertEqual(receipt['auxiliary']['verified_targets'], [])
        original = BundleReader(base)
        first = CheckpointReader(receipt_path.parent / 'dit.block00.rank0.safetensors')
        second = CheckpointReader(receipt_path.parent / 'dit.block00.rank1.safetensors')
        name = 'qkv_proj.weight'
        unmerged = join_matrix(first.read_tensor(name), second.read_tensor(name), 'qkv_proj')
        self.assertTrue(torch.equal(unmerged, original.read_tensor('blocks.0.attn.qkv_proj.weight')))

    def test_two_layer_bake_does_not_export_unbaked_native_header_blocks(self):
        import hashlib
        import json
        descriptor = json.loads(self.fixture.source.descriptor.read_text(encoding='utf-8'))
        header = descriptor['native_header']
        current = next(name for name in header if name.startswith('blocks.0.'))
        extra = current.replace('blocks.0.', 'blocks.49.', 1)
        header[extra] = dict(header[current])
        descriptor['native_header_sha256'] = hashlib.sha256(
            json.dumps(header, sort_keys=True).encode()).hexdigest()
        expanded = self.fixture.source.descriptor.with_name('fixture_expanded_header.json')
        expanded.write_text(json.dumps(descriptor), encoding='utf-8')
        with patch('patches.ptd.bake_v2.PTDCheckpoint', EmptySyntheticPTD):
            receipt = bake_shared_part(expanded, self.heads, self.root / 'shared_trimmed',
                                       layers=1, allow_fixture=True)
        exported = json.loads((receipt.parent / 'native_header_v2.json').read_text())
        self.assertNotIn(extra, exported)
        self.assertIn(current, exported)

    def test_incomplete_part_never_publishes_final_manifest(self):
        root = self._bake()
        missing = root / 'hop_07' / 'dit.block00.rank1.safetensors'
        missing.unlink()  # Only the test-owned disposable scratch fixture.
        manifest = root / 'bank_v2.json'
        with self.assertRaises(FileNotFoundError):
            assemble_bank_v2(manifest, root / 'shared',
                             [root / f'hop_{h:02d}' for h in range(8)], allow_fixture=True)
        self.assertFalse(manifest.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
