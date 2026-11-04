"""Synthetic, no-GPU Bank v2 regression and base-free admission contracts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import benchmarks.test_tp2_bundle as bundle_fixture
from patches.ptd.bank_v2 import FORMAT_V2, PTDBankV2
from patches.ptd.schedule import hop_sigmas
from patches.tp2_bundle import BundleReader, sha, open_checkpoint
from patches.tp2_disk_bake import _persist_layer, atomic_json, state_proof
from patches.windowed_tp import CheckpointReader


class SelfContainedV2(unittest.TestCase):
    def setUp(self):
        self.source = bundle_fixture.Bundle('runTest')
        self.source.setUp()
        self.addCleanup(self.source.doCleanups)
        fixture = self.source
        fixture.native.update({
            'final_layer.video_out.weight': torch.ones(3, 8),
            'final_layer.video_out.bias': torch.zeros(3),
            'final_layer.audio_out.weight': torch.ones(2, 8),
            'final_layer.audio_out.bias': torch.zeros(2),
        })
        save_file(fixture.native, str(fixture.source))
        save_file({k: v for k, v in fixture.native.items() if not k.startswith('blocks.')}, str(fixture.aux))
        bundle_index = json.loads((fixture.bundle / 'bundle.json').read_text())
        updated_source = dict(bytes=fixture.source.stat().st_size, sha256=sha(fixture.source))
        for kind in ('dit', 'te'):
            bundle_index['plan']['sources'][kind] = updated_source
        (fixture.bundle / 'bundle.json').write_text(json.dumps(bundle_index))
        descriptor = json.loads(fixture.descriptor.read_text())
        descriptor.update(bundle_sha256=sha(fixture.bundle / 'bundle.json'),
                          source=updated_source, native_header=CheckpointReader(fixture.source).header,
                          auxiliary_sha256=sha(fixture.aux))
        descriptor['native_header_sha256'] = hashlib.sha256(
            json.dumps(descriptor['native_header'], sort_keys=True).encode()).hexdigest()
        fixture.descriptor.write_text(json.dumps(descriptor))

        base = BundleReader(fixture.descriptor)
        self.bank_root = fixture.root / 'ptd_v2'
        self.bank_root.mkdir()
        source_recipe = dict(base=str(fixture.descriptor), base_sha256=sha(fixture.descriptor),
                             base_bundle_sha256=sha(base.bundle_file), checkpoint_sha256='synthetic-ptd',
                             layers=1, hops=8, steps=32, video_shift=12., audio_shift=3.)
        source_recipe_sha = hashlib.sha256(json.dumps(source_recipe, sort_keys=True).encode()).hexdigest()
        self.source_recipe_sha = source_recipe_sha
        self.overlay_name = 'token_refiner.blocks.0.attn.out_proj.weight'
        self.overlays = {}
        receipts = []
        parts = {'shared': {'root': '.', 'identity': source_recipe['base_bundle_sha256']}}
        for hop in range(8):
            part_id = f'hop_{hop:02d}'
            directory = self.bank_root / part_id
            directory.mkdir()
            shards = _persist_layer(fixture.local, 0, [], directory, base.plan['blocks'], source_recipe_sha)
            values = {self.overlay_name: (fixture.native[self.overlay_name].float() + hop).bfloat16(),
                      'final_layer.video_out.weight': torch.full((3, 8), hop + 1.),
                      'final_layer.video_out.bias': torch.zeros(3),
                      'final_layer.audio_out.weight': torch.full((2, 8), hop + 1.),
                      'final_layer.audio_out.bias': torch.zeros(2)}
            self.overlays[hop] = values
            overlay = directory / 'aux_overlay.safetensors'
            save_file(values, str(overlay))
            auxiliary = dict(file=overlay.name, sha256=sha(overlay), bytes=overlay.stat().st_size,
                             state_sha256=state_proof(values),
                             verified_targets=[self.overlay_name.removesuffix('.weight')],
                             part=part_id)
            parts[part_id] = dict(root=part_id, identity=source_recipe_sha + ':' + str(hop))
            receipts.append(dict(hop=hop, status='complete', source_recipe_sha256=source_recipe_sha,
                auxiliary=auxiliary,
                shards={stem: dict(row, part=part_id, file=stem + '.safetensors',
                                   manifest=stem + '.json')
                        for stem, row in shards.items()}))

        overlay_names = set(self.overlays[0])
        static = set(base.aux.header) - {'__metadata__'} - overlay_names
        shared = self.bank_root / 'shared_base_aux.safetensors'
        save_file({name: base.aux.read_tensor(name).contiguous() for name in sorted(static)}, str(shared))
        atomic_json(self.bank_root / 'native_header_v2.json', base.header)
        atomic_json(self.bank_root / 'block_plan_v2.json', {'blocks': {
            stem: base.plan['blocks'][stem] for stem in ('dit.block00.rank0','dit.block00.rank1')}})
        recipe = dict(format=FORMAT_V2, layers=1, hops=8, steps=32,
                      video_shift=12., audio_shift=3.,
                      base_bundle_sha256=source_recipe['base_bundle_sha256'],
                      base_descriptor_sha256=source_recipe['base_sha256'],
                      ptd_checkpoint_sha256=source_recipe['checkpoint_sha256'],
                      ordinary_loras=[], pruned_adaln_approximation=False,
                      source_bank_recipe_sha256=source_recipe_sha,
                      quantization='native-convrot256-int8-tp2')
        recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        self.v2 = self.bank_root / 'bank_v2.json'
        atomic_json(self.v2, dict(format=FORMAT_V2, status='complete', online_lora=False,
            full_model=False, recipe=recipe, recipe_sha256=recipe_sha, parts=parts,
            shared=dict(part='shared', aux=shared.name, aux_bytes=shared.stat().st_size,
                        aux_sha256=sha(shared), native_header='native_header_v2.json',
                        block_plan='block_plan_v2.json'),
            hops=receipts, sigmas=hop_sigmas(8).tolist()))

    def v2_manifest(self):
        """Return the directly assembled synthetic Bank v2 manifest."""
        return self.v2

    def test_self_contained_v2_without_original_base(self):
        v2 = self.v2_manifest()
        self.assertEqual(json.loads(v2.read_text())['format'], FORMAT_V2)
        fixture = self.source
        fixture.descriptor.rename(fixture.descriptor.with_suffix('.offline'))
        with patch('patches.ptd.bank_v2.sha', side_effect=AssertionError('Production must not hash')):
            bank = PTDBankV2(v2, allow_fixture=True, verify_payloads=False)
            reader = bank.reader(6)
            self.assertEqual(bank.layers, 1)
            self.assertTrue(torch.equal(reader.read_tensor(self.overlay_name), self.overlays[6][self.overlay_name]))
            self.assertTrue(torch.equal(reader.read_tensor('untouched'), fixture.native['untouched']))
            for rank in (0, 1):
                with patch('patches.ptd.bank_v2.shell',
                           side_effect=lambda _kind, r: bundle_fixture.block({
                               k: torch.zeros_like(v) for k, v in fixture.local[r].items()})):
                    state = reader.load_rank(0, rank).state_dict()
                for name, value in fixture.local[rank].items():
                    self.assertTrue(torch.equal(state[name], value))
            with self.assertRaisesRegex(ValueError, 'requires baked'):
                bank.reader(2, refiner=False)
        self.assertFalse(hasattr(bank, 'base'))

    def test_manifest_rejects_incomplete_shards(self):
        v2 = self.v2_manifest()
        data = json.loads(v2.read_text())
        data['hops'][7]['shards'].pop('dit.block00.rank1')
        atomic_json(v2, data)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            PTDBankV2(v2, allow_fixture=True)

    def test_mixed_source_recipes_and_duplicate_rank_paths_are_rejected(self):
        v2 = self.v2_manifest()
        data = json.loads(v2.read_text())
        data['hops'][4]['source_recipe_sha256'] = 'another-bake-recipe'
        atomic_json(v2, data)
        with self.assertRaisesRegex(ValueError, 'coverage/identity'):
            PTDBankV2(v2, allow_fixture=True)
        data['hops'][4]['source_recipe_sha256'] = self.fixture_recipe_sha()
        first = data['hops'][4]['shards']['dit.block00.rank0']
        second = data['hops'][4]['shards']['dit.block00.rank1']
        second['file'] = first['file']
        atomic_json(v2, data)
        with self.assertRaisesRegex(ValueError, 'distinct immutable'):
            PTDBankV2(v2, allow_fixture=True)

    def fixture_recipe_sha(self):
        return self.source_recipe_sha

    def test_mounted_part_mapping_and_absent_part_fail_closed(self):
        v2 = self.v2_manifest()
        moved = self.source.root / 'external_hop_03'
        (self.bank_root / 'hop_03').rename(moved)
        with self.assertRaises(FileNotFoundError):
            PTDBankV2(v2, allow_fixture=True)
        mounts = (('hop_03', str(moved)),)
        bank = PTDBankV2(v2, allow_fixture=True, part_roots=mounts)
        self.assertEqual(bank.reader(3).read_tensor(self.overlay_name).shape, (8, 8))

    def test_production_open_checkpoint_rejects_partial_fixture(self):
        v2 = self.v2_manifest()
        self.source.descriptor.rename(self.source.descriptor.with_suffix('.offline'))
        with self.assertRaisesRegex(ValueError, 'Unqualified Bank v2 layer count'):
            open_checkpoint(v2, verify_weights=False)
        reader = PTDBankV2(v2, allow_fixture=True).reader(0)
        self.assertEqual(reader.hop, 0)
        self.assertEqual(reader.kind, 'dit')
        self.assertFalse(reader.verify_payloads)
        staged = open_checkpoint(v2, verify_weights=False, ptd_allow_fixture=True)
        self.assertEqual(staged.hop, 0)
        self.assertEqual(staged.kind, 'dit')


if __name__ == '__main__':
    unittest.main(verbosity=2)
